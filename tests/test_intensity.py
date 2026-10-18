import numpy as np
import pytest
import SimpleITK as sitk

from hn_mri_prep.preprocessing import intensity


def _img(arr):
    return sitk.GetImageFromArray(np.asarray(arr, dtype=np.float32))


def test_zscore_inside_mask_and_zero_background():
    rng = np.random.default_rng(0)
    arr = rng.normal(500, 40, (6, 32, 32))
    mask = np.zeros_like(arr, dtype=np.uint8)
    mask[:, 8:24, 8:24] = 1
    out = sitk.GetArrayFromImage(intensity.zscore(_img(arr), sitk.GetImageFromArray(mask)))
    assert abs(out[mask > 0].mean()) < 0.05 and abs(out[mask > 0].std() - 1) < 0.05
    assert (out[mask == 0] == 0).all()


def test_uncovered_voxels_corrupt_zscore_statistics():
    """Why coverage-aware masks exist: zeros from a shorter stack shift the statistics."""
    arr = np.full((10, 16, 16), 100.0)
    arr[7:] = 0.0  # moving image did not cover the last slices
    full = np.ones_like(arr, dtype=np.uint8)
    covered = full.copy()
    covered[7:] = 0
    naive = sitk.GetArrayFromImage(intensity.zscore(_img(arr), sitk.GetImageFromArray(full)))
    aware = sitk.GetArrayFromImage(intensity.zscore(_img(arr + np.random.default_rng(0).normal(0, 1, arr.shape) * (arr > 0)),
                                                    sitk.GetImageFromArray(covered)))
    assert naive[:7].mean() > 0.5  # tissue no longer centred on 0
    assert abs(aware[:7].mean()) < 0.05


@pytest.mark.parametrize("scale,unit", [(1.0, "1e-6 mm2/s"), (1e-3, "1e-3 mm2/s"), (1e-6, "mm2/s")])
def test_adc_unit_harmonisation(scale, unit):
    arr = np.full((4, 8, 8), 1200.0) * scale
    out, detected = intensity.harmonize_adc_units(_img(arr))
    assert detected == unit
    assert np.allclose(sitk.GetArrayFromImage(out), 1200.0, rtol=1e-4)
