"""Convert sorted DICOM series to one NIfTI set per subject.

Writes ``T1, T1c, T2, B0, DWI, ADC`` (``.nii.gz``), extra candidates as ``*_run-N``, and a
``conversion_manifest.csv`` with every series and how it was classified. If several series
qualify for one slot, the lowest SeriesNumber wins; B0/DWI come from the same acquisition and
the ADC on the B0 grid is preferred.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from hn_mri_prep.config import ConversionConfig
from hn_mri_prep.io.series_classifier import SeriesType, classify_series

log = logging.getLogger(__name__)

STRUCTURAL = {SeriesType.T1: "T1", SeriesType.T1C: "T1c", SeriesType.T2: "T2"}


@dataclass
class Candidate:
    """One volume that may become ``<slot>.nii.gz`` for a subject."""

    slot: str
    series_number: int
    stem: str
    description: str
    nifti: Path | None = None  # copy this file ...
    image: nib.Nifti1Image | None = None  # ... or save this in-memory volume
    b_value: float | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def load(self) -> nib.Nifti1Image:
        return self.image if self.image is not None else nib.load(self.nifti)


def run_dcm2niix(input_dir: Path, output_dir: Path) -> None:
    if shutil.which("dcm2niix") is None:
        raise RuntimeError("dcm2niix not found on PATH (https://github.com/rordenlab/dcm2niix)")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["dcm2niix", "-z", "y", "-f", "%p_%s", "-o", str(output_dir), str(input_dir)]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0:
        log.warning("dcm2niix (%s): %s", input_dir.name, res.stderr.decode(errors="replace"))


def is_axial(img: nib.Nifti1Image, description: str) -> bool:
    """Third voxel axis closest to S/I, and not a sagittal/coronal/3D series by name."""
    desc = description.lower()
    if any(k in desc for k in ("sag", "cor", "3d")):
        return False
    return nib.aff2axcodes(img.affine)[2] in ("S", "I")


def _read_bvals(bval_path: Path, meta: dict) -> np.ndarray | None:
    if bval_path.exists():
        return np.atleast_1d(np.loadtxt(bval_path))
    b = meta.get("DiffusionBValue")
    return None if b is None else np.array([float(b)])


def _split_volume(img: nib.Nifti1Image, index: int) -> nib.Nifti1Image:
    data = np.asarray(img.dataobj[..., index], dtype=np.float32)
    header = img.header.copy()
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(data, img.affine, header)


def _diffusion_candidates(
    nifti: Path, bval_path: Path, meta: dict, base: dict, cfg: ConversionConfig, row: dict
) -> list[Candidate]:
    img = nib.load(nifti)
    bvals = _read_bvals(bval_path, meta)
    is_4d = img.ndim == 4 and img.shape[3] > 1

    if is_4d:
        if bvals is None or len(bvals) != img.shape[3]:
            row["note"] = f"4D with {img.shape[3]} volumes but bvals={bvals}; skipped"
            return []
        lo, hi = int(np.argmin(bvals)), int(np.argmax(bvals))
        out = [Candidate("B0", image=_split_volume(img, lo), b_value=float(bvals[lo]), **base)]
        if hi != lo:
            out.append(Candidate("DWI", image=_split_volume(img, hi), b_value=float(bvals[hi]), **base))
        row["b_values"] = ",".join(f"{b:g}" for b in bvals)
        return out

    if bvals is not None and len(bvals) > 0:
        b = float(bvals[0])
        row["b_values"] = f"{b:g}"
        if b <= cfg.low_b_max:
            return [Candidate("B0", nifti=nifti, b_value=b, **base)]
        if b >= cfg.high_b_min:
            return [Candidate("DWI", nifti=nifti, b_value=b, **base)]
        row["note"] = f"intermediate b-value {b:g}; skipped"
        return []

    # No physics available: fall back to the description and flag it for manual review.
    row["text_fallback"] = True
    slot = "B0" if "b0" in base["description"].lower() else "DWI"
    log.warning("%s: no b-value metadata, '%s' -> %s by description", nifti.name, base["description"], slot)
    return [Candidate(slot, nifti=nifti, **base)]


def collect_candidates(nifti_dir: Path, cfg: ConversionConfig) -> tuple[list[Candidate], list[dict]]:
    candidates: list[Candidate] = []
    rows: list[dict] = []
    for json_path in sorted(nifti_dir.glob("*.json")):
        stem = json_path.name[: -len(".json")]
        nifti = json_path.with_name(stem + ".nii.gz")
        if not nifti.exists():
            continue
        meta = json.loads(json_path.read_text())
        desc = meta.get("SeriesDescription") or meta.get("ProtocolName") or ""
        image_type = meta.get("ImageType", [])
        bval_path = json_path.with_name(stem + ".bval")
        cls = classify_series(desc, image_type, bval_path.exists(), legacy=cfg.legacy_classifier)
        row: dict[str, object] = {
            "file": nifti.name,
            "series_number": meta.get("SeriesNumber"),
            "description": desc,
            "image_type": "\\".join(map(str, image_type)),
            "type": cls.series_type.value,
            "rule": cls.rule,
        }
        rows.append(row)
        base = {
            "series_number": int(meta.get("SeriesNumber") or 0),
            "stem": stem,
            "description": desc,
        }

        if cls.series_type in STRUCTURAL:
            axial = is_axial(nib.load(nifti), desc)
            row["axial"] = axial
            if axial:
                candidates.append(Candidate(STRUCTURAL[cls.series_type], nifti=nifti, **base))
        elif cls.series_type is SeriesType.ADC:
            img = nib.load(nifti)
            if img.ndim == 3 or (img.ndim == 4 and img.shape[3] == 1):
                candidates.append(Candidate("ADC", nifti=nifti, **base))
            else:
                row["note"] = f"ADC with shape {img.shape}; skipped"
        elif cls.series_type is SeriesType.DIFFUSION:
            candidates.extend(_diffusion_candidates(nifti, bval_path, meta, base, cfg, row))

    return candidates, rows


def _same_grid(a: nib.Nifti1Image, b: nib.Nifti1Image) -> bool:
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=1e-3)


def order_candidates(candidates: list[Candidate], cfg: ConversionConfig) -> dict[str, list[Candidate]]:
    """Deterministically order the candidates of every slot; index 0 is the primary file."""
    reverse = cfg.primary_series == "last"
    by_slot: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_slot[c.slot].append(c)
    for slot in by_slot:
        by_slot[slot].sort(key=lambda c: (c.series_number, c.stem), reverse=reverse)

    # B0/DWI must come from the same acquisition: put a series providing both first.
    b0_series = {c.series_number for c in by_slot.get("B0", [])}
    dwi_series = {c.series_number for c in by_slot.get("DWI", [])}
    paired = [s for s in (c.series_number for c in by_slot.get("B0", [])) if s in dwi_series]
    if paired:
        first = paired[0]
        for slot in ("B0", "DWI"):
            by_slot[slot].sort(key=lambda c, f=first: c.series_number != f)
    elif b0_series and dwi_series:
        log.warning("B0 and DWI come from different series (%s vs %s)", b0_series, dwi_series)

    # Prefer the ADC that lives on the primary B0's voxel grid.
    if by_slot.get("ADC") and by_slot.get("B0"):
        b0 = by_slot["B0"][0].load()
        for c in by_slot["ADC"]:
            c.extra["matches_b0_grid"] = _same_grid(c.load(), b0)
        by_slot["ADC"].sort(key=lambda c: not c.extra["matches_b0_grid"])
    return by_slot


def write_candidates(by_slot: dict[str, list[Candidate]], out_dir: Path) -> list[dict]:
    rows = []
    for slot, cands in by_slot.items():
        for k, c in enumerate(cands, start=1):
            name = f"{slot}.nii.gz" if k == 1 else f"{slot}_run-{k}.nii.gz"
            if c.image is not None:
                nib.save(c.image, out_dir / name)
            else:
                shutil.copy2(c.nifti, out_dir / name)
            rows.append({
                "output": name,
                "source": c.stem,
                "series_number": c.series_number,
                "b_value": c.b_value,
                "primary": k == 1,
                **c.extra,
            })
    return rows


def convert_subject(
    subject_dir: Path, temp_root: Path, clean_root: Path, cfg: ConversionConfig
) -> pd.DataFrame:
    subject = subject_dir.name
    tmp = temp_root / subject
    out = clean_root / subject
    for d in (tmp, out):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for series_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
        if any(tok in series_dir.name.lower() for tok in cfg.skip_description_tokens):
            continue
        run_dcm2niix(series_dir, tmp)

    candidates, series_rows = collect_candidates(tmp, cfg)
    output_rows = write_candidates(order_candidates(candidates, cfg), out)

    outputs_by_stem: dict[str, list[str]] = defaultdict(list)
    for r in output_rows:
        outputs_by_stem[r["source"]].append(r["output"] + (" (primary)" if r["primary"] else ""))
        if "matches_b0_grid" in r:
            outputs_by_stem[r["source"]].append(f"matches_b0_grid={r['matches_b0_grid']}")
    for row in series_rows:
        row["outputs"] = "; ".join(outputs_by_stem.get(row["file"][: -len(".nii.gz")], []))
    series_df = pd.DataFrame(series_rows)
    series_df.insert(0, "subject", subject)
    series_df.to_csv(out / "conversion_manifest.csv", index=False)

    missing = {"T1", "T1c", "T2", "B0", "DWI", "ADC"} - {r["output"].split(".")[0] for r in output_rows}
    if missing:
        log.warning("%s: missing %s", subject, sorted(missing))
    return series_df


def convert_dataset(dicom_root: Path, temp_root: Path, clean_root: Path, cfg: ConversionConfig) -> pd.DataFrame:
    frames = []
    subjects = sorted(p for p in Path(dicom_root).iterdir() if p.is_dir())
    for i, subject_dir in enumerate(subjects, 1):
        log.info("[%d/%d] %s", i, len(subjects), subject_dir.name)
        frames.append(convert_subject(subject_dir, Path(temp_root), Path(clean_root), cfg))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_csv(Path(clean_root) / "conversion_manifest_all.csv", index=False)
    return df


def audit_classification(dicom_root: Path) -> pd.DataFrame:
    """Classify every sorted DICOM series with the current and the legacy rules (no conversion).

    Use it to find subjects whose sequence assignment changes with the classifier fix.
    """
    import pydicom

    rows = []
    for subject_dir in sorted(p for p in Path(dicom_root).iterdir() if p.is_dir()):
        for series_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            first = next((f for f in sorted(series_dir.iterdir()) if f.is_file()), None)
            if first is None:
                continue
            ds = pydicom.dcmread(str(first), stop_before_pixels=True, force=True)
            desc = str(getattr(ds, "SeriesDescription", "") or "")
            image_type = list(getattr(ds, "ImageType", []) or [])
            has_b = "DiffusionBValue" in ds
            new = classify_series(desc, image_type, has_b)
            old = classify_series(desc, image_type, has_b, legacy=True)
            rows.append({
                "subject": subject_dir.name,
                "series_folder": series_dir.name,
                "series_number": getattr(ds, "SeriesNumber", None),
                "description": desc,
                "type": new.series_type.value,
                "rule": new.rule,
                "legacy_type": old.series_type.value,
                "legacy_rule": old.rule,
                "changed": new.series_type != old.series_type,
            })
    return pd.DataFrame(rows)
