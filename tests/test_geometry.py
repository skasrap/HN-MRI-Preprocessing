import numpy as np
import SimpleITK as sitk

from hn_mri_prep.preprocessing import geometry


def _img(size, spacing, origin=(0.0, 0.0, 0.0)):
    im = sitk.Image(list(size), sitk.sitkFloat32)
    im.SetSpacing(spacing)
    im.SetOrigin(origin)
    return im


def _extent(im):
    lo = np.array(im.GetOrigin()) - np.array(im.GetSpacing()) / 2
    hi = lo + np.array(im.GetSize()) * np.array(im.GetSpacing())
    return lo, hi


def test_to_ras_direction():
    d = geometry.to_ras(_img((4, 4, 4), (1, 1, 1))).GetDirection()
    assert np.allclose(d, (-1, 0, 0, 0, -1, 0, 0, 0, 1))


def test_reference_grid_preserves_extent():
    anchor = _img((500, 500, 24), (0.44, 0.44, 5.0), (-110.0, -110.0, -60.0))
    grid = geometry.make_reference_grid(anchor, (1.0, 1.0, 5.0), preserve_extent=True)
    assert grid.GetSize() == (220, 220, 24)
    assert np.allclose(_extent(grid)[0], _extent(anchor)[0], atol=1e-6)


def test_reference_grid_legacy_keeps_origin():
    anchor = _img((500, 500, 24), (0.44, 0.44, 5.0), (-110.0, -110.0, -60.0))
    grid = geometry.make_reference_grid(anchor, (1.0, 1.0, 5.0), preserve_extent=False)
    assert grid.GetOrigin() == anchor.GetOrigin()


def test_antialias_never_blurs_slice_axis():
    sig = geometry.antialias_sigmas((0.37, 0.37, 5.0), (1.0, 1.0, 5.0))
    assert sig[0] > 0 and sig[1] > 0 and sig[2] == 0.0
    assert geometry.antialias_sigmas((1, 1, 5), (0.5, 0.5, 5)) == [0.0, 0.0, 0.0]  # upsampling


def test_antialias_suppresses_aliasing():
    """A 0.8 mm-period in-plane grating is unresolvable at 1 mm: it must vanish, not alias."""
    x = np.arange(400) * 0.25
    arr = np.tile(np.sin(2 * np.pi * x / 0.8)[None, None, :], (4, 400, 1)).astype(np.float32)
    src = sitk.GetImageFromArray(arr)
    src.SetSpacing((0.25, 0.25, 5.0))
    ref = geometry.make_reference_grid(src, (1.0, 1.0, 5.0))
    naive = sitk.GetArrayFromImage(geometry.resample(src, ref, antialias=False))
    filtered = sitk.GetArrayFromImage(geometry.resample(src, ref, antialias=True))
    assert naive[:, 10:-10, 10:-10].std() > 0.3
    assert filtered[:, 10:-10, 10:-10].std() < 0.05


def test_coverage_mask_marks_missing_slices():
    ref = _img((10, 10, 20), (1, 1, 5))
    moving = _img((10, 10, 10), (1, 1, 5))  # covers only the first 10 slices
    cov = sitk.GetArrayFromImage(geometry.coverage_mask(moving, ref, None))
    assert cov[:10].all() and not cov[10:].any()
