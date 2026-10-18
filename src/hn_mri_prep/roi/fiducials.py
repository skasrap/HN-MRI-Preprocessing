"""3D Slicer markup (``.mrk.json``) fiducials -> RAS millimetres -> voxel indices."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def load_fiducial_ras(json_path: str | Path) -> np.ndarray:
    """First defined control point of the first markup, in RAS mm.

    Slicer >= 4.11 writes LPS by default; older files and some exports use RAS. The
    ``coordinateSystem`` field is honoured and LPS is assumed only when it is absent.
    """
    data = json.loads(Path(json_path).read_text())
    markups = data.get("markups") or []
    if not markups:
        raise ValueError(f"{json_path}: no markups")
    markup = markups[0]
    points = [p["position"] for p in markup.get("controlPoints", []) if p.get("positionStatus", "defined") == "defined"]
    if not points:
        raise ValueError(f"{json_path}: no defined control points")
    if len(points) > 1:
        log.warning("%s: %d control points, using the first", json_path, len(points))

    xyz = np.asarray(points[0], dtype=float)
    system = str(markup.get("coordinateSystem", "LPS")).upper()
    if system == "LPS":
        xyz[:2] *= -1.0
    elif system != "RAS":
        raise ValueError(f"{json_path}: unsupported coordinateSystem {system!r}")
    return xyz


def ras_to_voxel(ras_mm: np.ndarray, affine: np.ndarray) -> tuple[int, int, int]:
    ijk = np.linalg.inv(affine) @ np.append(ras_mm, 1.0)
    return tuple(int(v) for v in np.round(ijk[:3]))


def find_fiducial(subject_dir: Path, filenames: tuple[str, ...]) -> Path | None:
    return next((subject_dir / n for n in filenames if (subject_dir / n).exists()), None)
