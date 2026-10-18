import pytest

from hn_mri_prep.io.series_classifier import SeriesType, classify_series

T = SeriesType


@pytest.mark.parametrize(
    "desc,expected",
    [
        ("t1_tse_tra", T.T1),
        ("t1_tse_tra_fs_ce", T.T1C),
        ("T1W_TSE_GD", T.T1C),
        ("Ax T1 FS +C", T.T1C),
        ("t1_tse_fs_tra_post", T.T1C),
        ("T1 C+ AX", T.T1C),
        ("t2_tse_tra", T.T2),
        ("t2_tse_fs_tra", T.T2),
        ("ep2d_diff_b50_800_tra_ADC", T.ADC),
        ("ep2d_diff_b50_800_tra_TRACEW", T.DIFFUSION),
        ("DWI_b0_b1000", T.DIFFUSION),
        ("eADC", T.SKIP),
        ("t1_eT1", T.SKIP),
        ("localizer", T.UNKNOWN),
    ],
)
def test_common_descriptions(desc, expected):
    assert classify_series(desc).series_type is expected


@pytest.mark.parametrize("desc", ["t1_space_tra", "t1_tse_tra_reference", "t1_tse_tra_sequence2", "T1 slice"])
def test_ce_substring_bug_is_fixed(desc):
    """'ce' inside space/reference/sequence/slice must not mean contrast-enhanced."""
    assert classify_series(desc).series_type is T.T1
    assert classify_series(desc, legacy=True).series_type is T.T1C


def test_adc_from_image_type():
    assert classify_series("ep2d_diff_tra", ["DERIVED", "PRIMARY", "ADC"]).series_type is T.ADC


def test_bval_makes_diffusion():
    assert classify_series("some_epi_tra", has_bval=True).series_type is T.DIFFUSION


def test_every_result_explains_itself():
    assert classify_series("t1_tse_tra_fs_ce").rule
