import csv
import zipfile

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from hn_mri_prep.io.dicom_sort import clean_name, sort_archives


def _dicom(path, patient_id, series_uid, desc, series_number):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.PatientID, ds.StudyInstanceUID, ds.SeriesInstanceUID = patient_id, "1.2.3", series_uid
    ds.SeriesDescription, ds.SeriesNumber = desc, series_number
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows = ds.Columns = 2
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.PixelData = np.zeros((2, 2), np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)


def _archive(tmp_path, name, patient_id):
    src = tmp_path / f"src_{name}"
    s1, s2 = generate_uid(), generate_uid()
    for sub, uid, desc, num in (("a", s1, "t1_tse_tra", 3), ("b", s2, "t2:tse/tra", 4)):
        (src / sub).mkdir(parents=True)
        for i in range(2):
            _dicom(src / sub / f"IM{i}", patient_id, uid, desc, num)  # same names in both folders
    zdir = tmp_path / "zips"
    zdir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zdir / f"{name}.zip", "w") as z:
        for f in src.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(src))
    return zdir


def test_sort_groups_series_and_avoids_name_collisions(tmp_path):
    zdir = _archive(tmp_path, "exam1", "123456")
    rows = sort_archives(zdir, tmp_path / "out")
    subj = tmp_path / "out" / "123456"
    assert {p.name for p in subj.iterdir()} == {"t1_tse_tra", "t2_tse_tra"}
    assert len(list((subj / "t1_tse_tra").iterdir())) == 2  # IM0/IM1 from both folders kept apart
    assert rows[0]["n_series"] == 2 and not rows[0]["possible_duplicate_patient"]


def test_duplicate_patient_is_flagged_and_pseudonymised(tmp_path):
    _archive(tmp_path, "exam1", "123456")
    zdir = _archive(tmp_path, "exam2", "123456")
    rows = sort_archives(zdir, tmp_path / "out", pseudonym_prefix="PG", id_map_path=tmp_path / "key.csv")
    assert [r["subject"] for r in rows] == ["PG0001", "PG0001_2"]
    assert rows[1]["possible_duplicate_patient"]
    assert not (tmp_path / "out" / "123456").exists()
    with open(tmp_path / "key.csv") as f:
        assert {r["patient_id"] for r in csv.DictReader(f)} == {"123456"}


def test_clean_name():
    assert clean_name('a<b>:c"/d') == "a_b__c__d"
    assert clean_name("  ") == "Unknown" and clean_name(None) == "Unknown"
