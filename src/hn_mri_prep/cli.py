"""Command-line interface: ``hn-mri-prep <stage> ...``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hn_mri_prep import __version__
from hn_mri_prep.config import load_config
from hn_mri_prep.logging_utils import setup_logging

log = logging.getLogger("hn_mri_prep")


def _cmd_sort(a: argparse.Namespace) -> None:
    from hn_mri_prep.io.dicom_sort import sort_archives

    sort_archives(a.zip_dir, a.out_dir, a.pseudonymize, a.id_map)


def _cmd_classify(a: argparse.Namespace) -> None:
    from hn_mri_prep.io.nifti_convert import audit_classification

    df = audit_classification(a.dicom_dir)
    df.to_csv(a.out, index=False)
    changed = df[df["changed"]] if not df.empty else df
    log.info("%d series classified; %d change with the token-based rules -> %s", len(df), len(changed), a.out)
    if len(changed):
        print(changed[["subject", "description", "legacy_type", "type"]].to_string(index=False))


def _cmd_convert(a: argparse.Namespace) -> None:
    from hn_mri_prep.io.nifti_convert import convert_dataset

    convert_dataset(a.dicom_dir, a.temp_dir, a.out_dir, load_config(a.config).conversion)


def _cmd_audit_spacing(a: argparse.Namespace) -> None:
    from hn_mri_prep.audit.spacing import audit_spacing

    print(audit_spacing(a.nifti_dir, a.out_dir, excel=a.excel).to_string(index=False))


def _cmd_preprocess(a: argparse.Namespace) -> None:
    from hn_mri_prep.preprocessing.pipeline import preprocess_dataset

    df = preprocess_dataset(a.nifti_dir, a.out_dir, load_config(a.config), workers=a.workers)
    if (df["status"] != "ok").any():
        sys.exit(1 if a.strict else 0)


def _cmd_extract(a: argparse.Namespace) -> None:
    from hn_mri_prep.roi.extraction import extract_dataset

    extract_dataset(a.preprocessed_dir, a.out_dir, load_config(a.config))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hn-mri-prep", description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sort-dicom", help="Stage 1: unpack ZIP exports into subject/series folders")
    s.add_argument("zip_dir", type=Path)
    s.add_argument("out_dir", type=Path)
    s.add_argument("--pseudonymize", metavar="PREFIX", help="name subjects PREFIX0001... instead of PatientID")
    s.add_argument("--id-map", type=Path, help="where to write the PatientID -> pseudonym key (keep private)")
    s.set_defaults(func=_cmd_sort)

    s = sub.add_parser("classify", help="Dry run: sequence classification of every series (current vs legacy)")
    s.add_argument("dicom_dir", type=Path)
    s.add_argument("--out", type=Path, default=Path("series_classification.csv"))
    s.set_defaults(func=_cmd_classify)

    s = sub.add_parser("convert", help="Stage 2: DICOM -> canonical NIfTI set per subject (needs dcm2niix)")
    s.add_argument("dicom_dir", type=Path)
    s.add_argument("out_dir", type=Path)
    s.add_argument("--temp-dir", type=Path, required=True, help="scratch folder for raw dcm2niix output")
    s.add_argument("--config", type=Path)
    s.set_defaults(func=_cmd_convert)

    s = sub.add_parser("audit-spacing", help="Voxel spacing statistics of a converted dataset")
    s.add_argument("nifti_dir", type=Path)
    s.add_argument("out_dir", type=Path)
    s.add_argument("--excel", action="store_true", help="also write an .xlsx (needs openpyxl)")
    s.set_defaults(func=_cmd_audit_spacing)

    s = sub.add_parser("preprocess", help="Stage 3: register, resample, normalise, crop")
    s.add_argument("nifti_dir", type=Path)
    s.add_argument("out_dir", type=Path)
    s.add_argument("--config", type=Path)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--strict", action="store_true", help="exit 1 if any subject failed")
    s.set_defaults(func=_cmd_preprocess)

    s = sub.add_parser("extract-rois", help="Stage 4: fiducial-centred tensors (.npy)")
    s.add_argument("preprocessed_dir", type=Path)
    s.add_argument("out_dir", type=Path)
    s.add_argument("--config", type=Path)
    s.set_defaults(func=_cmd_extract)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
