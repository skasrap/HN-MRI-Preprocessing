"""Tissue bounding box and fixed in-plane size cropping/padding."""

from __future__ import annotations

import SimpleITK as sitk

BBox = tuple[int, int, int, int, int, int]  # (x, y, z, size_x, size_y, size_z), voxel units


def tissue_bbox(mask: sitk.Image) -> BBox | None:
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.Cast(mask, sitk.sitkUInt8))
    return tuple(stats.GetBoundingBox(1)) if stats.HasLabel(1) else None


def crop_to_bbox_xy(image: sitk.Image, bbox: BBox, target_xy: int, pad_value: float = 0.0) -> sitk.Image:
    """Crop to ``bbox``, then pad or centre-crop in-plane to exactly ``target_xy`` squared.

    The slice axis keeps the bounding-box extent. If tissue is wider than ``target_xy`` mm
    (shoulders on caudal slices), the lateral margins are cut symmetrically. The parotid
    glands lie well within +-80 mm of the midline, so they are never affected at 320 mm.
    """
    x, y, z, sx, sy, sz = bbox
    out = sitk.RegionOfInterest(image, [sx, sy, sz], [x, y, z])

    lower = [max(0, (target_xy - sx) // 2), max(0, (target_xy - sy) // 2), 0]
    upper = [max(0, target_xy - sx - lower[0]), max(0, target_xy - sy - lower[1]), 0]
    if any(lower) or any(upper):
        out = sitk.ConstantPad(out, lower, upper, pad_value)

    cx, cy, cz = out.GetSize()
    if cx > target_xy or cy > target_xy:
        out = sitk.RegionOfInterest(
            out, [target_xy, target_xy, cz], [max(0, (cx - target_xy) // 2), max(0, (cy - target_xy) // 2), 0]
        )
    return out
