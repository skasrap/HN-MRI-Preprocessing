"""End-to-end: converted NIfTI subject -> preprocessing -> ROI tensors, on the phantom."""

import dataclasses
import json

import numpy as np
import pytest
import SimpleITK as sitk

from hn_mri_prep.config import PipelineConfig, load_config
from hn_mri_prep.preprocessing.pipeline import MissingSequenceError, preprocess_subject
from hn_mri_prep.roi.extraction import extract_subject
from tests.conftest import TUMOUR_LPS, write_slicer_fiducial

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module", params=["recommended", "thesis_v5"])
def run(request, subject_dir, tmp_path_factory):
    cfg = load_config(f"configs/{request.param}.yaml")
    out = tmp_path_factory.mktemp(request.param)
    record = preprocess_subject(subject_dir, out, cfg.preprocess, cfg.fingerprint())
    return request.param, cfg, out / subject_dir.name, record


def test_outputs_and_provenance(run):
    _name, cfg, out, record = run
    assert record["status"] == "ok"
    for seq in ("T1c", "T1", "T2", "B0", "DWI", "ADC", "StructMask", "DiffMask"):
        img = sitk.ReadImage(str(out / f"SUBJ001_{seq}.nii.gz"))
        assert img.GetSize()[:2] == (320, 320)
        assert np.allclose(img.GetSpacing(), (1.0, 1.0, 5.0))
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["config_fingerprint"] == cfg.fingerprint()
    assert prov["environment"]["SimpleITK"]
    assert (out / "SUBJ001_B0_to_T2.tfm").exists()


def test_registration_quality(run):
    _, _, _, record = run
    assert record["metric_T2_T1c"] < -0.3 and record["metric_B0_T2"] < -0.2


def test_structural_zscore(run):
    _, _, out, _ = run
    t1c = sitk.GetArrayFromImage(sitk.ReadImage(str(out / "SUBJ001_T1c.nii.gz")))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(out / "SUBJ001_StructMask.nii.gz"))) > 0
    assert abs(t1c[mask].mean()) < 0.2 and 0.7 < t1c[mask].std() < 1.3
    assert (t1c[~mask] == 0).all()


def test_recommended_specifics(run):
    name, _, out, record = run
    if name != "recommended":
        pytest.skip("recommended-config behaviour")
    assert record["adc_source_unit"] == "1e-6 mm2/s"
    assert 0.5 < record["coverage_T2"] < 1.0  # the phantom T2 stack is shorter than T1c
    s = sitk.ReadImage(str(out / "SUBJ001_T1c.nii.gz"))
    d = sitk.ReadImage(str(out / "SUBJ001_ADC.nii.gz"))
    assert s.GetOrigin() == d.GetOrigin() and s.GetSize() == d.GetSize()  # shared bbox


def test_tumour_lands_where_the_fiducial_says(run, tmp_path):
    """After registration + cropping, the fiducial still points at the bright tumour in T1c and ADC."""
    _, cfg, out, _ = run
    write_slicer_fiducial(out / "F.mrk.json", TUMOUR_LPS)
    roi_cfg = dataclasses.replace(cfg.roi, box_sizes_mm={"48mm": 48.0}, save_qc=False)
    rec = extract_subject(out, tmp_path, roi_cfg)
    assert rec["box_tumor_in_diff_window"]
    for seq, thr in (("T1c", 1.5), ("ADC", 1400)):
        arr = np.load(tmp_path / "48mm" / f"SUBJ001_{seq}.npy")
        assert arr.shape == (11, 224, 224)
        centre = arr[5, 100:124, 100:124]
        assert np.median(centre) > thr, (seq, np.median(centre))


def test_missing_required_sequence(tmp_path, subject_dir):
    d = tmp_path / "S"
    d.mkdir()
    (d / "T1c.nii.gz").write_bytes((subject_dir / "T1c.nii.gz").read_bytes())
    with pytest.raises(MissingSequenceError):
        preprocess_subject(d, tmp_path / "o", PipelineConfig().preprocess)
