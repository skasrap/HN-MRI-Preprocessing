import numpy as np
import pytest

from hn_mri_prep.roi.fiducials import load_fiducial_ras, ras_to_voxel
from tests.conftest import write_slicer_fiducial


@pytest.mark.parametrize("system", ["LPS", "RAS"])
def test_lps_and_ras_files_give_same_point(tmp_path, system):
    p = tmp_path / "F.mrk.json"
    write_slicer_fiducial(p, (-50.0, 5.0, 10.0), coordinate_system=system)
    assert np.allclose(load_fiducial_ras(p), (50.0, -5.0, 10.0))


def test_ras_to_voxel_inverts_affine():
    affine = np.diag([1.0, 1.0, 5.0, 1.0])
    affine[:3, 3] = (-100.0, -120.0, -60.0)
    assert ras_to_voxel(np.array([0.0, 0.0, 0.0]), affine) == (100, 120, 12)


def test_no_points_is_an_error(tmp_path):
    p = tmp_path / "msk.json"
    p.write_text('{"markups": [{"coordinateSystem": "LPS", "controlPoints": []}]}')
    with pytest.raises(ValueError):
        load_fiducial_ras(p)
