"""Voxel-spacing statistics of a converted dataset (descriptive only)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SEQUENCES = ("T1", "T2", "T1c", "B0", "DWI", "ADC")
_RUN = re.compile(r"^(?P<seq>[A-Za-z0-9]+?)(_run-(?P<run>\d+))?\.nii(\.gz)?$")


def collect_spacings(dataset_dir: Path) -> pd.DataFrame:
    rows = []
    for subject_dir in sorted(p for p in Path(dataset_dir).iterdir() if p.is_dir()):
        for f in sorted(subject_dir.glob("*.nii*")):
            m = _RUN.match(f.name)
            if not m or m["seq"] not in SEQUENCES:
                continue
            try:
                img = nib.load(f)
            except Exception as e:
                log.warning("Cannot read %s: %s", f, e)
                continue
            zooms = img.header.get_zooms()
            if len(zooms) < 3:
                continue
            rows.append({
                "subject": subject_dir.name, "file": f.name, "sequence": m["seq"],
                "primary": m["run"] is None, "x_mm": float(zooms[0]), "y_mm": float(zooms[1]),
                "z_mm": float(zooms[2]), "nx": img.shape[0], "ny": img.shape[1], "nz": img.shape[2],
            })
    return pd.DataFrame(rows)


def flag_outliers(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Tukey fences per sequence and axis."""
    df = df.copy()
    df["outlier"] = False
    for ax in ("x_mm", "y_mm", "z_mm"):
        q1 = df.groupby("sequence")[ax].transform(lambda s: s.quantile(0.25))
        q3 = df.groupby("sequence")[ax].transform(lambda s: s.quantile(0.75))
        iqr = q3 - q1
        col = f"{ax[0]}_outlier"
        df[col] = (df[ax] < q1 - k * iqr) | (df[ax] > q3 + k * iqr)
        df["outlier"] |= df[col]
    return df


def summarize(df: pd.DataFrame, primary_only: bool = True) -> pd.DataFrame:
    """Median [IQR] and range per sequence. By default only the files the pipeline uses."""
    d = df[df["primary"]] if primary_only else df
    out = []
    for seq, g in d.groupby("sequence", sort=False):
        row = {"sequence": seq, "n_images": len(g), "n_subjects": g["subject"].nunique()}
        for ax in ("x_mm", "y_mm", "z_mm"):
            row[f"{ax}_median"] = g[ax].median()
            row[f"{ax}_q1"], row[f"{ax}_q3"] = g[ax].quantile([0.25, 0.75])
            row[f"{ax}_min"], row[f"{ax}_max"] = g[ax].min(), g[ax].max()
        row["anisotropy_median"] = float(np.median(g["z_mm"] / g[["x_mm", "y_mm"]].min(axis=1)))
        out.append(row)
    return pd.DataFrame(out)


def spacing_combinations(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(**{c: df[c].round(4) for c in ("x_mm", "y_mm", "z_mm")})
    c = d.groupby(["sequence", "x_mm", "y_mm", "z_mm"]).size().rename("n_images").reset_index()
    c["percent"] = 100 * c["n_images"] / c.groupby("sequence")["n_images"].transform("sum")
    return c.sort_values(["sequence", "n_images"], ascending=[True, False])


def audit_spacing(dataset_dir: Path, out_dir: Path, excel: bool = False) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = flag_outliers(collect_spacings(dataset_dir))
    if images.empty:
        raise FileNotFoundError(f"no NIfTI files found under {dataset_dir}")
    tables = {
        "summary": summarize(images),
        "all_images": images,
        "spacing_combinations": spacing_combinations(images),
        "outliers": images[images["outlier"]],
    }
    for name, t in tables.items():
        t.to_csv(out_dir / f"spacing_{name}.csv", index=False)
    if excel:
        with pd.ExcelWriter(out_dir / "spacing_audit.xlsx", engine="openpyxl") as w:
            for name, t in tables.items():
                t.to_excel(w, sheet_name=name[:31], index=False)
    return tables["summary"]
