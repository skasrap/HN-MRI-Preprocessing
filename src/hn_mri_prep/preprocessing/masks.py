"""Foreground (head-and-neck tissue) masks for structural and EPI images.

The morphological closing radius is given in millimetres and converted per axis, with a
single-voxel radius through-plane: a 20 mm in-plane radius on a 5 mm slice axis would
otherwise merge slices 20 mm apart (4 slices) into the mask.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def _largest_component_closed(mask: sitk.Image, closing_radius_mm: float) -> sitk.Image:
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    components = sitk.RelabelComponent(sitk.ConnectedComponent(mask), sortByObjectSize=True)
    mask = sitk.Cast(components == 1, sitk.sitkUInt8)

    sx, sy = mask.GetSpacing()[:2]
    radius = [max(1, round(closing_radius_mm / sx)), max(1, round(closing_radius_mm / sy)), 1]
    mask = sitk.BinaryMorphologicalClosing(mask, radius, sitk.sitkBall, 1)
    return sitk.BinaryFillhole(mask, fullyConnected=False, foregroundValue=1)


def structural_mask(image: sitk.Image, closing_radius_mm: float = 20.0, sigma: float = 1.0) -> sitk.Image:
    """Li threshold on a smoothed image (adapts to the bimodal T1/T2 histogram)."""
    smoothed = sitk.SmoothingRecursiveGaussian(sitk.Cast(image, sitk.sitkFloat32), sigma)
    mask = sitk.LiThreshold(smoothed, 0, 1)
    return _largest_component_closed(mask, closing_radius_mm)


def diffusion_mask(
    image: sitk.Image, closing_radius_mm: float = 20.0, sigma: float = 2.0, clip_percentile: float = 99.5
) -> sitk.Image:
    """Otsu on a clipped, log-compressed EPI image (handles its heavy-tailed histogram)."""
    image = sitk.Cast(image, sitk.sitkFloat32)
    arr = sitk.GetArrayViewFromImage(image)
    valid = arr[arr > 1e-5]
    if valid.size:
        hi = float(np.percentile(valid, clip_percentile))
        image = sitk.IntensityWindowing(image, 0.0, hi, 0.0, hi)
    log_img = sitk.Log10(image + 1.0)
    smoothed = sitk.SmoothingRecursiveGaussian(log_img, sigma)
    mask = sitk.OtsuThreshold(smoothed, 0, 1)
    return _largest_component_closed(mask, closing_radius_mm)
