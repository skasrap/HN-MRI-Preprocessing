"""Synthetic head-and-neck phantom shared by the tests (no patient data in this repository).

Anatomy is defined in millimetres (LPS scanner coordinates, as ITK uses), so the same
phantom can be sampled on any grid: 0.5 x 0.5 x 5 mm "T1c", 1 x 1 x 4 mm "EPI", etc.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

# name: (centre LPS mm, radii mm)
ANATOMY = {
    "head": ((0.0, 0.0, 0.0), (75.0, 85.0, 70.0)),
    "parotid_r": ((-50.0, 5.0, 10.0), (14.0, 18.0, 25.0)),  # LPS x < 0 is patient right
    "parotid_l": ((50.0, 5.0, 10.0), (14.0, 18.0, 25.0)),
    "tumour": ((-50.0, 5.0, 10.0), (7.0, 7.0, 9.0)),
    "spine": ((0.0, 40.0, 0.0), (12.0, 12.0, 200.0)),
    "airway": ((0.0, -15.0, 0.0), (8.0, 10.0, 200.0)),
}

CONTRASTS = {  # intensities: background, head, parotid, tumour, spine, airway
    "T1c": (0, 400, 650, 900, 250, 20),
    "T1": (0, 420, 600, 380, 300, 20),
    "T2": (0, 300, 350, 800, 150, 10),
    "B0": (0, 200, 260, 500, 120, 5),
    "DWI": (0, 60, 70, 140, 30, 2),
    "ADC": (0, 1100, 1000, 1600, 700, 0),
}

TUMOUR_LPS = np.array(ANATOMY["tumour"][0])


def _inside(pts: np.ndarray, centre, radii) -> np.ndarray:
    return (((pts - np.asarray(centre)) / np.asarray(radii)) ** 2).sum(-1) <= 1.0


def phantom(
    contrast: str = "T1c",
    spacing=(1.0, 1.0, 5.0),
    fov_mm=(220.0, 220.0, 120.0),
    translation_mm=(0.0, 0.0, 0.0),
    noise: float = 0.02,
    seed: int = 0,
) -> sitk.Image:
    """Phantom image; ``translation_mm`` moves the anatomy (LPS)."""
    spacing = np.asarray(spacing, float)
    size = np.round(np.asarray(fov_mm) / spacing).astype(int)
    origin = -(size - 1) * spacing / 2.0
    zi, yi, xi = np.meshgrid(*(np.arange(n) for n in size[::-1]), indexing="ij")
    pts = np.stack([origin[0] + xi * spacing[0], origin[1] + yi * spacing[1], origin[2] + zi * spacing[2]], -1)
    pts = pts - np.asarray(translation_mm)

    bg, head, par, tum, spine, air = CONTRASTS[contrast]
    arr = np.zeros(pts.shape[:-1], np.float32) + bg
    head_m = _inside(pts, *ANATOMY["head"])
    # Smooth texture so mutual information has structure to lock on to.
    arr[head_m] = head * (1.0 + 0.15 * np.sin(pts[head_m, 0] / 9.0) * np.cos(pts[head_m, 1] / 11.0))
    for name, val in (("parotid_r", par), ("parotid_l", par), ("tumour", tum), ("spine", spine), ("airway", air)):
        arr[_inside(pts, *ANATOMY[name]) & head_m] = val
    rng = np.random.default_rng(seed)
    arr[head_m] += rng.normal(0, noise * max(CONTRASTS[contrast]), head_m.sum())

    img = sitk.GetImageFromArray(np.clip(arr, 0, None))
    img.SetSpacing(spacing.tolist())
    img.SetOrigin(origin.tolist())
    return img


@pytest.fixture(scope="session")
def subject_dir(tmp_path_factory) -> Path:
    """Converted-subject folder (stage 2 output layout) with realistic anisotropic grids."""
    d = tmp_path_factory.mktemp("nifti") / "SUBJ001"
    d.mkdir()
    struct = dict(spacing=(0.8, 0.8, 5.0), fov_mm=(220.0, 220.0, 120.0))
    epi = dict(spacing=(1.6, 1.6, 4.0), fov_mm=(230.0, 230.0, 96.0))
    sitk.WriteImage(phantom("T1c", **struct), str(d / "T1c.nii.gz"))
    sitk.WriteImage(phantom("T1", translation_mm=(1.5, -1.0, 0.0), seed=1, **struct), str(d / "T1.nii.gz"))
    # Shorter T2 stack than T1c (common in practice): exercises coverage-aware masking.
    sitk.WriteImage(phantom("T2", spacing=(0.8, 0.8, 5.0), fov_mm=(220.0, 220.0, 100.0),
                            translation_mm=(-2.0, 1.5, 0.0), seed=2), str(d / "T2.nii.gz"))
    for i, seq in enumerate(("B0", "DWI", "ADC")):
        sitk.WriteImage(phantom(seq, translation_mm=(3.0, -2.0, 2.0), seed=3 + i, **epi), str(d / f"{seq}.nii.gz"))
    return d


def write_slicer_fiducial(path: Path, lps_mm, coordinate_system: str = "LPS") -> None:
    pos = list(map(float, lps_mm))
    if coordinate_system == "RAS":
        pos = [-pos[0], -pos[1], pos[2]]
    path.write_text(json.dumps({
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [{"type": "Fiducial", "coordinateSystem": coordinate_system,
                     "controlPoints": [{"id": "1", "position": pos, "positionStatus": "defined"}]}],
    }))
