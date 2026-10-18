"""Bias-field correction and intensity normalisation (all per image)."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def n4_bias_correction(image: sitk.Image, mask: sitk.Image, shrink_factor: int = 3) -> sitk.Image:
    """N4 on a down-sampled copy; the bias field is evaluated at full resolution."""
    image_f = sitk.Cast(image, sitk.sitkFloat32)
    shrink = [shrink_factor] * image.GetDimension()
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.Execute(sitk.Shrink(image_f, shrink), sitk.Shrink(sitk.Cast(mask, sitk.sitkUInt8), shrink))
    log_bias = n4.GetLogBiasFieldAsImage(image_f)
    return sitk.Divide(image_f, sitk.Exp(log_bias))


def zscore(image: sitk.Image, mask: sitk.Image, clip_percentiles=(0.5, 99.5)) -> sitk.Image:
    """Z-score with mean/std from percentile-clipped tissue voxels; background set to 0.

    Clipping is applied only to estimate the statistics; output values are not clipped.
    """
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    m = sitk.GetArrayViewFromImage(mask) > 0
    tissue = arr[m]
    if tissue.size == 0:
        raise ValueError("Empty mask: cannot z-score")
    lo, hi = np.percentile(tissue, clip_percentiles)
    clipped = np.clip(tissue, lo, hi)
    out = (arr - clipped.mean()) / (clipped.std() + 1e-8)
    out[~m] = 0.0
    img = sitk.GetImageFromArray(out)
    img.CopyInformation(image)
    return img


def apply_mask(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    arr[sitk.GetArrayViewFromImage(mask) == 0] = 0.0
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(image)
    return img


# Typical in-vivo tissue ADC is 0.5-3.0 x 1e-3 mm^2/s. Vendors export it as 1e-6 mm^2/s
# integers (most common), 1e-3 mm^2/s or plain mm^2/s floats.
_ADC_SCALES = ((0.01, 1e6, "mm2/s"), (10.0, 1e3, "1e-3 mm2/s"))


def harmonize_adc_units(image: sitk.Image, mask: sitk.Image | None = None) -> tuple[sitk.Image, str]:
    """Rescale ADC to 1e-6 mm^2/s using the order of magnitude of the median tissue value.

    The rule is fixed in advance and uses only the image itself, so it is leakage-free. It
    returns the detected source unit, which belongs in the provenance record.
    """
    arr = sitk.GetArrayViewFromImage(image)
    sel = arr[(sitk.GetArrayViewFromImage(mask) > 0) & (arr > 0)] if mask is not None else arr[arr > 0]
    if sel.size == 0:
        return sitk.Cast(image, sitk.sitkFloat32), "unknown (empty)"
    median = float(np.median(sel))
    for upper, factor, unit in _ADC_SCALES:
        if median < upper:
            return sitk.Cast(image, sitk.sitkFloat32) * factor, unit
    return sitk.Cast(image, sitk.sitkFloat32), "1e-6 mm2/s"
