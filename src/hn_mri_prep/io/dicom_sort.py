"""Unpack exported ZIP archives into ``<subject>/<series>/`` DICOM folders.

Files are renamed to ``<SOPInstanceUID>.dcm``. A PatientID seen in more than one archive is
flagged in the manifest, since both subjects have to stay in the same CV fold. Subjects can
optionally be pseudonymised (``<prefix>0001``, ...), with the key written to a separate file.
"""

from __future__ import annotations

import csv
import logging
import shutil
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pydicom
from tqdm import tqdm

log = logging.getLogger(__name__)

_INVALID_CHARS = '<>:"/\\|?*'


def clean_name(name: object) -> str:
    """Make a string safe to use as a folder name on Windows and POSIX."""
    if name is None:
        return "Unknown"
    text = str(name)
    for ch in _INVALID_CHARS:
        text = text.replace(ch, "_")
    text = text.strip().rstrip(".")
    return text or "Unknown"


@dataclass(frozen=True)
class DicomHeader:
    path: Path
    patient_id: str | None
    study_uid: str | None
    series_uid: str
    series_description: str
    series_number: int | None
    sop_uid: str | None


def read_header(path: Path) -> DicomHeader | None:
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:  # not a DICOM file (DICOMDIR, readme, thumbnails...)
        return None
    series_uid = getattr(ds, "SeriesInstanceUID", None)
    if not series_uid:
        return None
    series_number = getattr(ds, "SeriesNumber", None)
    return DicomHeader(
        path=path,
        patient_id=str(getattr(ds, "PatientID", "") or "") or None,
        study_uid=getattr(ds, "StudyInstanceUID", None),
        series_uid=str(series_uid),
        series_description=str(getattr(ds, "SeriesDescription", "") or "UnknownSeries"),
        series_number=int(series_number) if series_number not in (None, "") else None,
        sop_uid=getattr(ds, "SOPInstanceUID", None),
    )


class SubjectNamer:
    """Allocates output folder names, optionally pseudonymised."""

    def __init__(self, pseudonym_prefix: str | None = None):
        self.prefix = pseudonym_prefix
        self._counts: dict[str, int] = defaultdict(int)
        self._pseudonyms: dict[str, str] = {}
        self.rows: list[dict[str, str]] = []

    def allocate(self, patient_id: str, archive: str) -> tuple[str, bool]:
        self._counts[patient_id] += 1
        n = self._counts[patient_id]
        if self.prefix is not None:
            if patient_id not in self._pseudonyms:
                self._pseudonyms[patient_id] = f"{self.prefix}{len(self._pseudonyms) + 1:04d}"
            base = self._pseudonyms[patient_id]
        else:
            base = clean_name(patient_id)
        folder = base if n == 1 else f"{base}_{n}"
        self.rows.append({"archive": archive, "patient_id": patient_id, "subject": folder})
        return folder, n > 1


def sort_archives(
    zip_dir: Path,
    out_dir: Path,
    pseudonym_prefix: str | None = None,
    id_map_path: Path | None = None,
) -> list[dict[str, object]]:
    """Unpack every ``*.zip`` in ``zip_dir``; returns one manifest row per archive."""
    zip_dir, out_dir = Path(zip_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if pseudonym_prefix is not None and id_map_path is None:
        raise ValueError("Pseudonymisation needs id_map_path to store the re-identification key.")

    namer = SubjectNamer(pseudonym_prefix)
    manifest: list[dict[str, object]] = []

    for zip_path in tqdm(sorted(zip_dir.glob("*.zip")), desc="Archives"):
        row: dict[str, object] = {"archive": zip_path.name}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            try:
                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(tmp_path)
            except (zipfile.BadZipFile, OSError) as e:
                log.error("Could not extract %s: %s", zip_path.name, e)
                manifest.append({**row, "status": f"extract_error: {e}"})
                continue

            headers = [h for f in tmp_path.rglob("*") if f.is_file() and (h := read_header(f))]
            if not headers:
                log.warning("No DICOM files in %s", zip_path.name)
                manifest.append({**row, "status": "no_dicom"})
                continue

            patient_id = next((h.patient_id for h in headers if h.patient_id), zip_path.stem)
            subject, duplicate = namer.allocate(patient_id, zip_path.name)
            if duplicate:
                log.warning(
                    "%s: PatientID already seen -> '%s'. Possible duplicate patient; keep "
                    "these subjects in the same CV fold.", zip_path.name, subject
                )
            studies = {h.study_uid for h in headers}

            series: dict[str, list[DicomHeader]] = defaultdict(list)
            for h in headers:
                series[h.series_uid].append(h)

            folder_counts: dict[str, int] = defaultdict(int)
            for files in sorted(series.values(), key=lambda fs: fs[0].series_number or 0):
                base = clean_name(files[0].series_description)
                folder_counts[base] += 1
                name = base if folder_counts[base] == 1 else f"{base}_{folder_counts[base]}"
                dest = out_dir / subject / name
                dest.mkdir(parents=True, exist_ok=True)
                for i, h in enumerate(files):
                    fname = f"{clean_name(h.sop_uid)}.dcm" if h.sop_uid else f"{i:05d}.dcm"
                    shutil.copy2(h.path, dest / fname)

            manifest.append({
                **row,
                "subject": subject,
                "status": "ok",
                "n_series": len(series),
                "n_files": len(headers),
                "n_studies": len(studies),
                "possible_duplicate_patient": duplicate,
            })

    if id_map_path is not None and pseudonym_prefix is not None:
        id_map_path = Path(id_map_path)
        id_map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(id_map_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["archive", "patient_id", "subject"])
            writer.writeheader()
            writer.writerows(namer.rows)
        log.info("Re-identification key written to %s. Keep it out of version control.", id_map_path)

    _write_csv(out_dir / "sort_manifest.csv", manifest)
    return manifest


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
