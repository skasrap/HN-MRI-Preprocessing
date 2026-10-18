import numpy as np
import pytest

from hn_mri_prep.normalization.adc import (
    DataLeakageError,
    FoldwiseADCScaler,
    assert_group_disjoint,
    fit_fold_scalers,
)


def _cohort(n=20, seed=0, outlier_subject=None):
    rng = np.random.default_rng(seed)
    vols = {}
    for i in range(n):
        v = np.zeros((4, 32, 32), np.float32)  # background = 0
        v[:, 8:24, 8:24] = rng.normal(1100, 150, (4, 16, 16)).clip(200, None)
        if f"S{i:02d}" == outlier_subject:
            v[0, 10, 10] = 4095.0  # one saturated artefact voxel
        vols[f"S{i:02d}"] = v
    return vols


def test_fitted_only_on_training_subjects():
    vols = _cohort()
    train = [k for k in vols if k not in ("S00", "S01")]
    s = FoldwiseADCScaler().fit({k: vols[k] for k in train})
    assert s.fitted_ids_ == frozenset(train)
    s.transform(vols["S00"], subject_id="S00")  # held-out: fine
    with pytest.raises(DataLeakageError):
        s.transform(vols["S02"], subject_id="S02")  # training subject passed as evaluation
    s.transform(vols["S02"], subject_id="S02", evaluation=False)  # explicit training use: fine


def test_validation_data_cannot_change_the_scale():
    vols = _cohort()
    train = {k: vols[k] for k in list(vols)[2:]}
    a = FoldwiseADCScaler().fit(train)
    vols["S00"][:] = 99999.0  # arbitrary change to a held-out subject
    b = FoldwiseADCScaler().fit(train)
    assert (a.low_, a.high_) == (b.low_, b.high_)


def test_background_stays_zero_and_output_in_unit_range():
    vols = _cohort()
    s = FoldwiseADCScaler().fit(vols)
    out = s.transform(vols["S00"], evaluation=False)
    assert (out[vols["S00"] == 0] == 0).all()
    assert out.min() >= 0 and out.max() <= 1


def test_minmax_is_dominated_by_a_single_voxel_percentile_is_not():
    clean, dirty = _cohort(), _cohort(outlier_subject="S05")
    mm = [FoldwiseADCScaler(method="minmax").fit(c).high_ for c in (clean, dirty)]
    pc = [FoldwiseADCScaler(method="percentile").fit(c).high_ for c in (clean, dirty)]
    assert mm[1] - mm[0] > 2000  # 1 voxel out of ~20k moves the scale massively
    assert abs(pc[1] - pc[0]) < 5


def test_percentiles_cut_the_requested_fraction_of_pooled_tissue():
    vols = _cohort()
    s = FoldwiseADCScaler(method="percentile", percentiles=(1, 99)).fit(vols)
    pooled = np.concatenate([v[v != 0] for v in vols.values()])
    assert abs((pooled < s.low_).mean() - 0.01) < 5e-4
    assert abs((pooled > s.high_).mean() - 0.01) < 5e-4


def test_fixed_range_estimates_nothing():
    s = FoldwiseADCScaler(method="fixed", fixed_range=(0, 3000)).fit(_cohort())
    assert (s.low_, s.high_) == (0.0, 3000.0)


def test_save_load_round_trip(tmp_path):
    s = FoldwiseADCScaler().fit(_cohort())
    s.save(tmp_path / "fold0.json")
    r = FoldwiseADCScaler.load(tmp_path / "fold0.json")
    v = _cohort()["S00"]
    assert np.array_equal(s.transform(v, evaluation=False), r.transform(v, evaluation=False))
    assert r.fitted_ids_ == s.fitted_ids_


def test_fit_fold_scalers_rejects_overlapping_folds():
    vols = _cohort(6)
    with pytest.raises(DataLeakageError):
        fit_fold_scalers(vols, [(["S00", "S01", "S02"], ["S02", "S03"])])


def test_duplicate_patient_across_folds_is_caught():
    """Two archives of one patient (subjects '123' and '123_2') must not straddle a split."""
    groups = {"123": "P1", "123_2": "P1", "456": "P2"}
    with pytest.raises(DataLeakageError, match="patients"):
        assert_group_disjoint(["123", "456"], ["123_2"], groups)
