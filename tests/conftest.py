import json
from pathlib import Path


def write_slicer_fiducial(path: Path, lps_mm, coordinate_system: str = "LPS") -> None:
    """Write a minimal 3D Slicer markups file with one control point."""
    pos = list(map(float, lps_mm))
    if coordinate_system == "RAS":
        pos = [-pos[0], -pos[1], pos[2]]
    path.write_text(json.dumps({
        "markups": [{"type": "Fiducial", "coordinateSystem": coordinate_system,
                     "controlPoints": [{"id": "1", "position": pos, "positionStatus": "defined"}]}],
    }))
