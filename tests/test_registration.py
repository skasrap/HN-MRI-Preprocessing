import numpy as np
import pytest

from hn_mri_prep.config import RegistrationConfig
from hn_mri_prep.preprocessing.registration import RegistrationError, register
from tests.conftest import phantom

SHIFT = np.array([3.0, -2.0, 2.0])  # mm, LPS


@pytest.fixture(scope="module")
def pair():
    fixed = phantom("T2", spacing=(1.0, 1.0, 5.0))
    moving = phantom("B0", spacing=(1.6, 1.6, 4.0), translation_mm=SHIFT, seed=5)
    return fixed, moving


def _translation(tx, fixed):
    """Displacement the transform applies at the image centre."""
    c = np.array(fixed.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2 for s in fixed.GetSize()]))
    return np.array(tx.TransformPoint(c.tolist())) - c


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["rigid", "affine"])
def test_recovers_known_shift_across_anisotropic_grids(pair, kind):
    fixed, moving = pair
    res = register(fixed, moving, kind, RegistrationConfig())
    err = _translation(res.transform, fixed) - SHIFT
    assert np.abs(err[:2]).max() < 0.5, err  # in-plane, mm
    assert abs(err[2]) < 1.5, err  # through-plane, 4-5 mm slices


@pytest.mark.slow
def test_seeded_registration_is_reproducible(pair):
    """Same seed -> same transform up to float summation order across threads."""
    fixed, moving = pair
    cfg = RegistrationConfig(seed=7, iterations=30)
    a = register(fixed, moving, "rigid", cfg).transform.GetParameters()
    b = register(fixed, moving, "rigid", cfg).transform.GetParameters()
    assert np.allclose(a, b, rtol=0, atol=1e-9)


@pytest.mark.slow
def test_single_thread_is_bit_identical(pair):
    fixed, moving = pair
    cfg = RegistrationConfig(seed=7, iterations=30, num_threads=1)
    a = register(fixed, moving, "rigid", cfg).transform.GetParameters()
    b = register(fixed, moving, "rigid", cfg).transform.GetParameters()
    assert a == b


def test_failure_threshold_raises(pair):
    fixed, _ = pair
    rng = np.random.default_rng(0)
    import SimpleITK as sitk

    noise = sitk.GetImageFromArray(rng.random(sitk.GetArrayFromImage(fixed).shape).astype(np.float32))
    noise.CopyInformation(fixed)
    with pytest.raises(RegistrationError):
        register(fixed, noise, "rigid", RegistrationConfig(iterations=5, fail_threshold=-0.05))
