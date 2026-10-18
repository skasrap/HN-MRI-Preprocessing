import SimpleITK as sitk

from hn_mri_prep.preprocessing.cropping import crop_to_bbox_xy, tissue_bbox


def _mask(size, box):
    m = sitk.Image(list(size), sitk.sitkUInt8)
    x0, y0, z0, x1, y1, z1 = box
    for z in range(z0, z1):
        for y in range(y0, y1):
            for x in range(x0, x1):
                m[x, y, z] = 1
    return m


def test_small_tissue_is_padded_to_target():
    m = _mask((60, 60, 6), (10, 20, 1, 40, 45, 5))
    out = crop_to_bbox_xy(m, tissue_bbox(m), 64)
    assert out.GetSize() == (64, 64, 4)
    assert int(sitk.GetArrayFromImage(out).sum(dtype=int)) == 30 * 25 * 4


def test_wide_tissue_is_centre_cropped_to_target():
    m = _mask((120, 50, 3), (0, 0, 0, 120, 50, 3))
    out = crop_to_bbox_xy(m, tissue_bbox(m), 64)
    assert out.GetSize() == (64, 64, 3)


def test_crop_keeps_physical_position():
    m = _mask((60, 60, 4), (10, 10, 0, 30, 30, 4))
    out = crop_to_bbox_xy(m, tissue_bbox(m), 40)
    # Tissue voxel (10, 10, 0) keeps its physical location after crop + pad.
    p = m.TransformIndexToPhysicalPoint((10, 10, 0))
    assert out[out.TransformPhysicalPointToIndex(p)] == 1
