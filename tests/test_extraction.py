import dataclasses

import nibabel as nib
import numpy as np
import pytest

from hn_mri_prep.config import ROIConfig
from hn_mri_prep.roi.extraction import (
    extract_padded,
    extract_subject,
    slice_window,
    to_radiological,
)
from tests.conftest import write_slicer_fiducial


@pytest.mark.parametrize(
    "center,lo,hi,expected",
    [
        (10, 0, 30, (5, 16, True)),  # centred
        (1, 0, 30, (0, 11, True)),  # shifted up at the bottom edge
        (29, 0, 30, (20, 31, True)),  # shifted down at the top edge
        (6, 3, 8, (1, 12, False)),  # only 6 valid slices: still 11, centred on the valid range
    ],
)
def test_slice_window_always_has_requested_depth(center, lo, hi, expected):
    s, e, fits = slice_window(center, 11, lo, hi)
    assert (s, e, fits) == expected and e - s == 11


def test_extract_padded_outside_volume():
    vol = np.arange(27, dtype=float).reshape(3, 3, 3)
    out = extract_padded(vol, ((-1, 2), (0, 3), (2, 5)), pad_value=-1)
    assert out.shape == (3, 3, 3)
    assert (out[0] == -1).all() and (out[:, :, 1:] == -1).all()
    assert out[1, 0, 0] == vol[0, 0, 2]


def test_radiological_orientation():
    """RAS volume: marker at patient right-anterior must land top-left."""
    vol = np.zeros((10, 10, 1))
    vol[9, 9, 0] = 1  # max i = Right, max j = Anterior
    img = to_radiological(vol)[0]
    assert img[0, 0] == 1


# --- extraction on a synthetic preprocessed subject ---------------------------------------

def _write_subject(root, valid_diff_slices, n_slices=20, tumour_k=10):
    """Preprocessed-layout subject on a shared 1 x 1 x 5 mm grid; slice index == z / 5 mm."""
    sid = "S1"
    d = root / sid
    d.mkdir(parents=True)
    affine = np.diag([1.0, 1.0, 5.0, 1.0])
    shape = (120, 120, n_slices)
    tumour = np.zeros(shape, np.float32)
    tumour[55:65, 55:65, tumour_k - 1:tumour_k + 2] = 1.0
    base = np.zeros(shape, np.float32)
    base[10:110, 10:110, :] = 1.0
    diff_valid = np.zeros(shape, np.float32)
    diff_valid[..., valid_diff_slices[0]:valid_diff_slices[1] + 1] = 1.0
    vols = {
        "T1c": base + 3 * tumour, "T1": base, "T2": base + 2 * tumour, "StructMask": base,
        "B0": (base + tumour) * diff_valid, "DWI": (base + tumour) * diff_valid,
        "ADC": (1000 * base + 600 * tumour) * diff_valid, "DiffMask": base,
    }
    for name, v in vols.items():
        nib.save(nib.Nifti1Image(v, affine), d / f"{sid}_{name}.nii.gz")
    # Fiducial at voxel (60, 60, tumour_k) -> RAS (60, 60, 5 * k) -> LPS (-60, -60, 5k)
    write_slicer_fiducial(d / "F.mrk.json", (-60.0, -60.0, 5.0 * tumour_k))
    return d


def _cfg(mode):
    return dataclasses.replace(ROIConfig(), z_window_mode=mode, box_sizes_mm={"64mm": 64.0}, save_qc=False)


@pytest.mark.parametrize("mode", ["shared", "independent"])
def test_tensors_have_fixed_shape(tmp_path, mode):
    d = _write_subject(tmp_path / "in", valid_diff_slices=(7, 12))  # only 6 usable EPI slices
    rec = extract_subject(d, tmp_path / "out", _cfg(mode))
    for seq in ROIConfig().sequences:
        assert np.load(tmp_path / "out" / "64mm" / f"S1_{seq}.npy").shape == (11, 224, 224)
        assert np.load(tmp_path / "out" / "Whole_3Slice" / f"S1_{seq}.npy").shape == (3, 224, 224)
    assert rec["box_fits_valid_range"] is False


def test_shared_mode_keeps_channels_on_the_same_slices(tmp_path):
    # Usable EPI slab ends at slice 13, tumour at 10 -> diffusion window must shift down.
    d = _write_subject(tmp_path / "in", valid_diff_slices=(0, 13), tumour_k=10)
    shared = extract_subject(d, tmp_path / "shared", _cfg("shared"))
    assert shared["box_window_struct"] == shared["box_window_diff"] == (3, 14)
    t1c = np.load(tmp_path / "shared" / "64mm" / "S1_T1c.npy")
    adc = np.load(tmp_path / "shared" / "64mm" / "S1_ADC.npy")
    assert np.array_equal(t1c.max(axis=(1, 2)) > 3, adc.max(axis=(1, 2)) > 1500)  # tumour on same slices

    indep = extract_subject(d, tmp_path / "indep", _cfg("independent"))
    assert indep["box_window_struct"] == (5, 16) and indep["box_window_diff"] == (3, 14)
    t1c = np.load(tmp_path / "indep" / "64mm" / "S1_T1c.npy")
    adc = np.load(tmp_path / "indep" / "64mm" / "S1_ADC.npy")
    assert not np.array_equal(t1c.max(axis=(1, 2)) > 3, adc.max(axis=(1, 2)) > 1500)


def test_tumour_is_centred_in_the_box(tmp_path):
    d = _write_subject(tmp_path / "in", valid_diff_slices=(0, 19))
    extract_subject(d, tmp_path / "out", _cfg("shared"))
    t1c = np.load(tmp_path / "out" / "64mm" / "S1_T1c.npy")
    rows, cols = np.nonzero(t1c[5] > 3)
    assert abs(rows.mean() - 111.5) < 6 and abs(cols.mean() - 111.5) < 6
