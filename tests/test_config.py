import pytest

from hn_mri_prep.config import PipelineConfig, load_config


def test_round_trip(tmp_path):
    cfg = PipelineConfig()
    cfg.dump(tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == cfg


def test_unknown_key_is_rejected(tmp_path):
    (tmp_path / "c.yaml").write_text("preprocess:\n  registration:\n    sead: 1\n")
    with pytest.raises(ValueError, match="sead"):
        load_config(tmp_path / "c.yaml")


def test_shipped_configs_load():
    assert load_config("configs/recommended.yaml") == PipelineConfig()
    thesis = load_config("configs/thesis_v5.yaml")
    assert thesis.roi.z_window_mode == "independent" and not thesis.preprocess.crop.shared_bbox
