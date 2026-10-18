"""Stage 4: fiducial-centred, fixed-size tensors for the classifier.

For every subject and every sequence this writes

* ``<box>/<subject>_<seq>.npy``: a ``box_depth`` x 224 x 224 patch of a fixed physical
  size (48 / 64 / 80 / 90 mm) centred on the tumour fiducial;
* ``Whole_3Slice/<subject>_<seq>.npy``: ``whole_depth`` full slices, padded to square.

Arrays are float32 in radiological orientation (row 0 = anterior, column 0 = patient right).

Changes relative to ``Fids_v4.ipynb``:

* the slice window always has exactly ``box_depth`` slices (before, a short EPI stack gave
  fewer slices, so tensor shapes differed between subjects);
* ``z_window_mode="shared"`` uses the same slices for structural and diffusion channels;
  before, only the diffusion window was shifted to avoid incomplete EPI slices, so channel
  k of T1c and channel k of ADC could be different anatomical levels;
* a manifest records, per subject, whether the tumour centre lies inside the window.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from hn_mri_prep.config import PipelineConfig, ROIConfig
from hn_mri_prep.roi.fiducials import find_fiducial, load_fiducial_ras, ras_to_voxel

log = logging.getLogger(__name__)

DIFFUSION_SEQS = frozenset({"B0", "DWI", "ADC", "DiffMask"})


def voxel_bounds(center: int, size_mm: float, spacing: float) -> tuple[int, int]:
    """Half-open index range of ``size_mm`` around ``center`` (may exceed the volume)."""
    r = round(size_mm / 2.0 / spacing)
    return center - r, center + r


def valid_slice_range(b0: np.ndarray, mask: np.ndarray, tolerance: float = 0.05, pad: float = 0.0) -> tuple[int, int] | None:
    """First/last slice where at most ``tolerance`` of the masked tissue is missing in B0.

    Affine EPI-to-T2 registration leaves partially empty slices at the slab ends; those are
    excluded so the classifier never sees half-empty diffusion slices.
    """
    valid = []
    for z in range(b0.shape[2]):
        tissue = mask[:, :, z] > 0
        n = tissue.sum()
        if n and (tissue & (np.abs(b0[:, :, z] - pad) <= 1e-5)).sum() / n <= tolerance:
            valid.append(z)
    return (valid[0], valid[-1]) if valid else None


def slice_window(center: int, depth: int, lo: int, hi: int) -> tuple[int, int, bool]:
    """``depth`` consecutive slices around ``center``, shifted to stay inside ``[lo, hi]``.

    Always returns exactly ``depth`` slices. If ``[lo, hi]`` is shorter than ``depth``, the
    window is centred on it and the third value is False; out-of-volume slices are later
    zero-padded.
    """
    available = hi - lo + 1
    if available < depth:
        start = lo - (depth - available) // 2
        return start, start + depth, False
    start = min(max(center - depth // 2, lo), hi + 1 - depth)
    return start, start + depth, True


def extract_padded(vol: np.ndarray, bounds: tuple[tuple[int, int], ...], pad_value: float) -> np.ndarray:
    """Crop ``vol`` to half-open ``bounds`` per axis, padding wherever they leave the volume."""
    src, pads = [], []
    for (s, e), n in zip(bounds, vol.shape, strict=False):
        src.append(slice(min(max(s, 0), n), min(max(e, 0), n)))
        pads.append((max(0, -s), max(0, e - n)))
    return np.pad(vol[tuple(src)], pads, mode="constant", constant_values=pad_value)


def to_radiological(vol: np.ndarray) -> np.ndarray:
    """RAS [i, j, k] -> [k, row, col] with anterior up and patient right on the left."""
    return np.transpose(np.flip(np.rot90(vol, k=1, axes=(0, 1)), axis=1), (2, 0, 1))


def crop_resize(vol, bi, bj, bk, target_xy: int, pad_value: float, is_mask: bool) -> np.ndarray:
    patch = extract_padded(vol, (bi, bj, bk), pad_value)
    factors = (target_xy / patch.shape[0], target_xy / patch.shape[1], 1.0)
    resized = zoom(patch, factors, order=0 if is_mask else 1, mode="nearest")
    return to_radiological(resized).astype(np.float32)


def _load(path: Path) -> nib.Nifti1Image:
    return nib.as_closest_canonical(nib.load(str(path)))


def extract_subject(subject_dir: Path, out_root: Path, cfg: ROIConfig) -> dict[str, Any]:
    sid = subject_dir.name
    files = {s: subject_dir / f"{sid}_{s}.nii.gz" for s in cfg.sequences}
    missing = [s for s, p in files.items() if not p.exists()]
    fid_path = find_fiducial(subject_dir, cfg.fiducial_filenames)
    if missing or fid_path is None:
        raise FileNotFoundError(f"missing {missing or ''}{' fiducial' if fid_path is None else ''}".strip())

    ras = load_fiducial_ras(fid_path)
    t1c, b0, dmask = _load(files["T1c"]), _load(files["B0"]), _load(files["DiffMask"])
    ref = {"struct": t1c, "diff": b0}
    center = {k: ras_to_voxel(ras, im.affine) for k, im in ref.items()}
    spacing = {k: im.header.get_zooms()[:3] for k, im in ref.items()}

    valid = valid_slice_range(b0.get_fdata(), dmask.get_fdata(), cfg.valid_slice_tolerance, cfg.pad_value)
    v_lo, v_hi = valid if valid else (0, b0.shape[2] - 1)

    windows: dict[str, dict[str, tuple[int, int]]] = {}
    rec: dict[str, Any] = {"subject": sid, "fiducial_ras": np.round(ras, 2).tolist(),
                           "center_struct": center["struct"], "center_diff": center["diff"],
                           "diff_valid_slices": v_hi - v_lo + 1 if valid else 0}
    for name, depth in (("box", cfg.box_depth), ("whole", cfg.whole_depth)):
        ds, de, fits = slice_window(center["diff"][2], depth, v_lo, v_hi)
        if cfg.z_window_mode == "shared":
            if not np.isclose(spacing["struct"][2], spacing["diff"][2]):
                raise ValueError("shared z-window requires equal slice spacing")
            offset = center["struct"][2] - center["diff"][2]
            ss, se = ds + offset, de + offset
        else:
            ss, se, _ = slice_window(center["struct"][2], depth, 0, t1c.shape[2] - 1)
        windows[name] = {"struct": (ss, se), "diff": (ds, de)}
        rec[f"{name}_window_diff"] = (ds, de)
        rec[f"{name}_window_struct"] = (ss, se)
        rec[f"{name}_fits_valid_range"] = fits
        rec[f"{name}_tumor_in_diff_window"] = ds <= center["diff"][2] < de

    if not rec["box_tumor_in_diff_window"]:
        log.warning("%s: fiducial slice outside the usable diffusion slab", sid)

    for box in (*cfg.box_sizes_mm, "Whole_3Slice"):
        (out_root / box).mkdir(parents=True, exist_ok=True)

    for seq, path in files.items():
        kind = "diff" if seq in DIFFUSION_SEQS else "struct"
        vol = _load(path).get_fdata(dtype=np.float32)
        is_mask = "Mask" in seq
        ci, cj, _ = center[kind]
        sx, sy, _ = spacing[kind]
        for box, size_mm in cfg.box_sizes_mm.items():
            arr = crop_resize(vol, voxel_bounds(ci, size_mm, sx), voxel_bounds(cj, size_mm, sy),
                              windows["box"][kind], cfg.target_xy, cfg.pad_value, is_mask)
            np.save(out_root / box / f"{sid}_{seq}.npy", arr)

        side = max(vol.shape[:2])
        bi = (-((side - vol.shape[0]) // 2), -((side - vol.shape[0]) // 2) + side)
        bj = (-((side - vol.shape[1]) // 2), -((side - vol.shape[1]) // 2) + side)
        arr = crop_resize(vol, bi, bj, windows["whole"][kind], cfg.target_xy, cfg.pad_value, is_mask)
        np.save(out_root / "Whole_3Slice" / f"{sid}_{seq}.npy", arr)

    if cfg.save_qc:
        from hn_mri_prep.roi.qc import save_roi_qc

        qc_dir = out_root / "QC"
        qc_dir.mkdir(exist_ok=True)
        save_roi_qc(sid, t1c.get_fdata(), _load(files["DWI"]).get_fdata(), b0.get_fdata(), center, spacing,
                    windows["box"], qc_dir / f"{sid}_ROI_QC.jpg")

    rec["status"] = "ok"
    return rec


def extract_dataset(preprocessed_dir: Path, out_dir: Path, cfg: PipelineConfig) -> pd.DataFrame:
    preprocessed_dir, out_dir = Path(preprocessed_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(out_dir / "config_used.yaml")
    subjects = sorted(p for p in preprocessed_dir.iterdir() if p.is_dir() and (p / f"{p.name}_T1c.nii.gz").exists())
    rows = []
    for sdir in subjects:
        try:
            rows.append(extract_subject(sdir, out_dir, cfg.roi))
        except Exception as e:
            log.error("%s: %s", sdir.name, e)
            rows.append({"subject": sdir.name, "status": "error", "error": f"{type(e).__name__}: {e}"})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "roi_manifest.csv", index=False)
    return df
