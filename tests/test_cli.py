"""The documented CLI workflow, on the phantom dataset."""

import pandas as pd
import pytest

from hn_mri_prep.cli import main
from tests.conftest import TUMOUR_LPS, write_slicer_fiducial


def test_help(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "preprocess" in capsys.readouterr().out


def test_audit_spacing(subject_dir, tmp_path, capsys):
    main(["audit-spacing", str(subject_dir.parent), str(tmp_path)])
    summary = pd.read_csv(tmp_path / "spacing_summary.csv").set_index("sequence")
    assert summary.loc["T1c", "z_mm_median"] == 5.0
    assert summary.loc["ADC", "x_mm_median"] == pytest.approx(1.6)
    assert summary.loc["T2", "anisotropy_median"] == pytest.approx(6.25)


@pytest.mark.slow
def test_preprocess_then_extract(subject_dir, tmp_path):
    pre, roi = tmp_path / "pre", tmp_path / "roi"
    main(["preprocess", str(subject_dir.parent), str(pre), "--config", "configs/recommended.yaml", "--strict"])
    audit = pd.read_csv(pre / "preprocess_audit.csv")
    assert (audit["status"] == "ok").all()
    assert (pre / "config_used.yaml").exists() and (pre / "QC" / "SUBJ001_QC_compact.png").exists()

    write_slicer_fiducial(pre / "SUBJ001" / "F.mrk.json", TUMOUR_LPS)
    (pre / "QC" / "not_a_subject").mkdir()  # output folders must not be mistaken for subjects
    main(["extract-rois", str(pre), str(roi), "--config", "configs/recommended.yaml"])
    manifest = pd.read_csv(roi / "roi_manifest.csv")
    assert list(manifest["subject"]) == ["SUBJ001"] and manifest["status"].eq("ok").all()
    assert (roi / "QC" / "SUBJ001_ROI_QC.jpg").exists()
    assert len(list((roi / "64mm").glob("*.npy"))) == 8
