"""Stage 3: register, resample, bias-correct, normalise and crop one subject.

Flow (unchanged from the thesis ``All_Preprocessing_Script_v5.py``):

1. Reorient all six series to RAS.
2. Rigid T2 -> T1c and T1 -> T1c; affine B0 -> T2 (EPI distortions are partly absorbed
   by the affine). DWI and ADC reuse the B0 transform (same acquisition), composed with
   T2 -> T1c, so every image is interpolated exactly once.
3. Resample onto a T1c-anchored 1 x 1 x 5 mm grid.
4. Masks: Li threshold on T1c (structural), Otsu on log-B0 intersected with the
   structural mask (diffusion).
5. N4 bias correction on T1c/T1/T2; per-image z-score (T1c, T1, T2, B0, DWI); ADC kept in
   physical units (only masked), so it can be scaled inside each CV fold later.
6. Crop to the tissue bounding box and pad / centre-crop to 320 x 320 in-plane.
"""

from __future__ import annotations

import logging
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk

from hn_mri_prep.config import PipelineConfig, PreprocessConfig
from hn_mri_prep.preprocessing import cropping, geometry, intensity, masks, qc
from hn_mri_prep.preprocessing.registration import register
from hn_mri_prep.provenance import write_provenance

log = logging.getLogger(__name__)

SEQUENCES = ("T1c", "T1", "T2", "B0", "DWI", "ADC")
STRUCTURAL = ("T1c", "T1", "T2")
DIFFUSION = ("B0", "DWI", "ADC")


class MissingSequenceError(FileNotFoundError):
    pass


def load_subject(subject_dir: Path) -> dict[str, sitk.Image | None]:
    images: dict[str, sitk.Image | None] = {}
    for seq in SEQUENCES:
        path = subject_dir / f"{seq}.nii.gz"
        images[seq] = geometry.to_ras(sitk.ReadImage(str(path))) if path.exists() else None
    return images


def _fraction(mask: sitk.Image, of: sitk.Image) -> float:
    m, ref = sitk.GetArrayViewFromImage(mask) > 0, sitk.GetArrayViewFromImage(of) > 0
    return float((m & ref).sum() / max(ref.sum(), 1))


def preprocess_subject(subject_dir: Path, out_root: Path, cfg: PreprocessConfig, fingerprint: str = "") -> dict[str, Any]:
    subject = subject_dir.name
    img = load_subject(subject_dir)
    missing = [s for s in cfg.required_sequences if img[s] is None]
    if missing:
        raise MissingSequenceError(f"missing {missing}")
    if cfg.registration.num_threads is not None:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(cfg.registration.num_threads)

    record: dict[str, Any] = {"subject": subject}
    for seq in ("DWI", "ADC"):
        record[f"{seq}_on_B0_grid"] = geometry.same_grid(img[seq], img["B0"])
        if not record[f"{seq}_on_B0_grid"]:
            log.warning("%s: %s is not on the B0 voxel grid; is it from the same acquisition?", subject, seq)

    # --- 2. Registration -----------------------------------------------------------------
    rc = cfg.registration
    t2_reg = register(img["T1c"], img["T2"], "rigid", rc, "T2->T1c")
    t1_reg = register(img["T1c"], img["T1"], "rigid", rc, "T1->T1c") if img["T1"] is not None else None
    b0_reg = register(img["T2"], img["B0"], "affine", rc, "B0->T2")
    # CompositeTransform applies the last transform first: T1c -> T2 -> B0.
    diff_xfm = sitk.CompositeTransform([b0_reg.transform, t2_reg.transform])
    xfm = {"T1c": sitk.Transform(), "T2": t2_reg.transform, "B0": diff_xfm, "DWI": diff_xfm, "ADC": diff_xfm}
    if t1_reg is not None:
        xfm["T1"] = t1_reg.transform
    record.update({
        "metric_T2_T1c": round(t2_reg.metric, 4),
        "metric_T1_T1c": round(t1_reg.metric, 4) if t1_reg else None,
        "metric_B0_T2": round(b0_reg.metric, 4),
    })

    # --- 3. Resampling ---------------------------------------------------------------------
    rs = cfg.resampling
    struct_grid = geometry.make_reference_grid(img["T1c"], rs.struct_spacing, rs.preserve_extent)
    diff_grid = geometry.make_reference_grid(img["T1c"], rs.diff_spacing, rs.preserve_extent)
    grid = {s: struct_grid for s in STRUCTURAL} | {s: diff_grid for s in DIFFUSION}
    res = {
        s: geometry.resample(img[s], grid[s], xfm[s], sitk.sitkLinear, antialias=rs.antialias)
        for s in SEQUENCES if img[s] is not None
    }

    # --- 4. Masks --------------------------------------------------------------------------
    mc = cfg.masks
    struct_mask = masks.structural_mask(res["T1c"], mc.closing_radius_mm, mc.struct_smoothing_sigma)
    struct_on_diff = geometry.resample(struct_mask, diff_grid, None, sitk.sitkNearestNeighbor)
    b0_mask = masks.diffusion_mask(res["B0"], mc.closing_radius_mm, mc.diff_smoothing_sigma, mc.diff_clip_percentile)
    diff_mask = sitk.And(sitk.Cast(struct_on_diff, sitk.sitkUInt8), sitk.Cast(b0_mask, sitk.sitkUInt8))

    norm_mask = {s: struct_mask for s in STRUCTURAL} | {s: diff_mask for s in DIFFUSION}
    if cfg.intensity.coverage_aware_masks:
        for s in ("T1", "T2", "B0"):
            if s in res:
                cov = geometry.coverage_mask(img[s], grid[s], xfm[s])
                record[f"coverage_{s}"] = round(_fraction(cov, norm_mask[s]), 4)
                norm_mask[s] = sitk.And(norm_mask[s], cov)
        for s in ("DWI", "ADC"):
            norm_mask[s] = norm_mask["B0"]
        diff_mask = norm_mask["B0"]

    # --- 5. Intensity ----------------------------------------------------------------------
    ic = cfg.intensity
    out: dict[str, sitk.Image] = {}
    for s in STRUCTURAL:
        if s in res:
            corrected = intensity.n4_bias_correction(res[s], norm_mask[s], ic.n4_shrink_factor)
            out[s] = intensity.zscore(corrected, norm_mask[s], ic.zscore_clip_percentiles)
    for s in ("B0", "DWI"):
        out[s] = intensity.zscore(res[s], norm_mask[s], ic.zscore_clip_percentiles)
    adc = res["ADC"]
    if ic.harmonize_adc_units:
        adc, record["adc_source_unit"] = intensity.harmonize_adc_units(adc, norm_mask["ADC"])
    out["ADC"] = intensity.apply_mask(adc, norm_mask["ADC"])

    # --- 6. Cropping -----------------------------------------------------------------------
    cc = cfg.crop
    s_bbox = cropping.tissue_bbox(struct_mask)
    if s_bbox is None:
        raise ValueError("structural mask is empty")
    if cc.shared_bbox:
        if not geometry.same_grid(struct_grid, diff_grid) or cc.struct_xy != cc.diff_xy:
            raise ValueError("shared_bbox requires identical structural/diffusion spacing and crop size")
        d_bbox = s_bbox
    else:
        d_bbox = cropping.tissue_bbox(diff_mask)
        if d_bbox is None:
            raise ValueError("diffusion mask is empty")

    final: dict[str, sitk.Image] = {}
    for s, im in out.items():
        bbox, size = (s_bbox, cc.struct_xy) if s in STRUCTURAL else (d_bbox, cc.diff_xy)
        final[s] = cropping.crop_to_bbox_xy(im, bbox, size)
    final["StructMask"] = cropping.crop_to_bbox_xy(struct_mask, s_bbox, cc.struct_xy)
    final["DiffMask"] = cropping.crop_to_bbox_xy(diff_mask, d_bbox, cc.diff_xy)

    # --- Outputs ---------------------------------------------------------------------------
    subj_out = out_root / subject
    subj_out.mkdir(parents=True, exist_ok=True)
    for name, im in final.items():
        sitk.WriteImage(im, str(subj_out / f"{subject}_{name}.nii.gz"), useCompression=True)
    if cfg.save_transforms:
        for name, r in (("T2_to_T1c", t2_reg), ("T1_to_T1c", t1_reg), ("B0_to_T2", b0_reg)):
            if r is not None:
                sitk.WriteTransform(r.transform, str(subj_out / f"{subject}_{name}.tfm"))
    if cfg.save_qc:
        qc_dir = out_root / "QC"
        qc_dir.mkdir(exist_ok=True)
        qc.save_compact_qc({k: final.get(k) for k in SEQUENCES}, final["StructMask"], final["DiffMask"],
                           subject, qc_dir / f"{subject}_QC_compact.png")
        qc.save_multislice_qc(final, final["StructMask"], final["DiffMask"], subject,
                              qc_dir / f"{subject}_QC_multislice.png")

    record.update({
        "status": "ok",
        "struct_slices": final["T1c"].GetSize()[2],
        "diff_slices": final["ADC"].GetSize()[2],
        "diff_mask_slices": int(np.any(sitk.GetArrayViewFromImage(final["DiffMask"]) > 0, axis=(1, 2)).sum()),
    })
    write_provenance(subj_out / "provenance.json", fingerprint, **record,
                     registration={k: r.stop_condition for k, r in
                                   (("T2_T1c", t2_reg), ("T1_T1c", t1_reg), ("B0_T2", b0_reg)) if r})
    return record


def _run_one(args: tuple[Path, Path, PreprocessConfig, str]) -> dict[str, Any]:
    subject_dir, out_root, cfg, fingerprint = args
    try:
        return preprocess_subject(subject_dir, out_root, cfg, fingerprint)
    except Exception as e:
        log.error("%s failed: %s", subject_dir.name, e)
        log.debug(traceback.format_exc())
        return {"subject": subject_dir.name, "status": "error", "error": f"{type(e).__name__}: {e}"}


def preprocess_dataset(input_dir: Path, output_dir: Path, cfg: PipelineConfig, workers: int = 1) -> pd.DataFrame:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(output_dir / "config_used.yaml")
    subjects = sorted(p for p in input_dir.iterdir() if p.is_dir() and (p / "T1c.nii.gz").exists())
    log.info("Preprocessing %d subjects (config %s)", len(subjects), cfg.fingerprint())

    jobs = [(s, output_dir, cfg.preprocess, cfg.fingerprint()) for s in subjects]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(_run_one, jobs))
    else:
        rows = [_run_one(j) for j in jobs]

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "preprocess_audit.csv", index=False)
    n_ok = int((df["status"] == "ok").sum()) if not df.empty else 0
    log.info("Done: %d/%d succeeded. Audit: %s", n_ok, len(df), output_dir / "preprocess_audit.csv")
    return df
