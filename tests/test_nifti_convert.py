"""Stage 2 selection logic on dcm2niix-style outputs (no dcm2niix needed)."""

import json

import nibabel as nib
import numpy as np

from hn_mri_prep.config import ConversionConfig
from hn_mri_prep.io.nifti_convert import collect_candidates, order_candidates, write_candidates

AXIAL = np.diag([0.5, 0.5, 5.0, 1.0])


def _series(d, stem, desc, number, shape=(8, 8, 4), affine=AXIAL, bvals=None, image_type=("ORIGINAL",)):
    data = np.random.default_rng(number).random(shape).astype(np.float32)
    nib.save(nib.Nifti1Image(data, affine), d / f"{stem}.nii.gz")
    (d / f"{stem}.json").write_text(json.dumps(
        {"SeriesDescription": desc, "SeriesNumber": number, "ImageType": list(image_type)}))
    if bvals is not None:
        (d / f"{stem}.bval").write_text(" ".join(map(str, bvals)))


def _run(d, cfg=None):
    cfg = cfg or ConversionConfig()
    cands, rows = collect_candidates(d, cfg)
    return order_candidates(cands, cfg), rows


def test_primary_series_is_chosen_by_series_number(tmp_path):
    _series(tmp_path, "z_late", "t1_tse_tra_fs_ce", 12)
    _series(tmp_path, "a_early", "t1_tse_tra_fs_ce", 9)
    by_slot, _ = _run(tmp_path)
    assert [c.series_number for c in by_slot["T1c"]] == [9, 12]
    by_slot, _ = _run(tmp_path, ConversionConfig(primary_series="last"))
    assert by_slot["T1c"][0].series_number == 12


def test_non_axial_structural_is_dropped(tmp_path):
    sag = np.array([[0, 0, 5.0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0, 1]])
    _series(tmp_path, "t2sag", "t2_tse_sag", 2, affine=sag)
    _series(tmp_path, "t2cor", "t2_tse_cor", 3)
    by_slot, rows = _run(tmp_path)
    assert "T2" not in by_slot
    assert all(r["axial"] is False for r in rows)


def test_4d_diffusion_split_and_adc_grid_preference(tmp_path):
    _series(tmp_path, "trace", "ep2d_diff_tra_TRACEW", 5, shape=(8, 8, 4, 3), affine=AXIAL, bvals=[50, 400, 800])
    _series(tmp_path, "adc_other", "ep2d_diff_ADC", 2, shape=(6, 6, 4))  # different grid, lower number
    _series(tmp_path, "adc", "ep2d_diff_tra_ADC", 6, shape=(8, 8, 4))
    by_slot, _ = _run(tmp_path)
    assert by_slot["B0"][0].b_value == 50 and by_slot["DWI"][0].b_value == 800
    assert by_slot["ADC"][0].stem == "adc" and by_slot["ADC"][0].extra["matches_b0_grid"]

    out = tmp_path / "out"
    out.mkdir()
    write_candidates(by_slot, out)
    assert {p.name for p in out.iterdir()} == {"B0.nii.gz", "DWI.nii.gz", "ADC.nii.gz", "ADC_run-2.nii.gz"}
    assert nib.load(out / "DWI.nii.gz").shape == (8, 8, 4)


def test_split_3d_diffusion_by_bvalue_and_text_fallback(tmp_path):
    _series(tmp_path, "b0", "dwi_b0", 7, bvals=[0])
    _series(tmp_path, "b1000", "dwi_b1000", 7, bvals=[1000])
    _series(tmp_path, "b500", "dwi_b500", 7, bvals=[500])
    by_slot, rows = _run(tmp_path)
    assert by_slot["B0"][0].stem == "b0" and by_slot["DWI"][0].stem == "b1000"
    assert any("intermediate" in str(r.get("note")) for r in rows)
