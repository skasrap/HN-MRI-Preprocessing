# Code review of the thesis preprocessing pipeline

Scope: `Unzip.ipynb` (stage 1 + 2), `Preprocess_EPI_v1.ipynb` (spacing audit),
`All_Preprocessing_Script_v5.py` (stage 3), `Fids_v4.ipynb` (stage 4).

Each finding lists what was wrong, how it was checked, whether it could have affected thesis
results, and how this repository handles it. **Config** means the fix is a switch:
`configs/thesis_v5.yaml` turns it off, `configs/recommended.yaml` turns it on.

## What is already right (and verified)

These are the choices a reviewer is most likely to probe. They hold up:

| Choice | Verification |
|---|---|
| `CompositeTransform([B0→T2, T2→T1c])` order | SimpleITK applies the last transform first, so a T1c-space point goes to T2 and then to B0. Correct. Checked numerically. |
| One interpolation per image (composite instead of chained resampling) | Correct; avoids cumulative blurring. |
| Anisotropic closing kernel `[r/sx, r/sy, 1]` | Correct. An isotropic 20 mm kernel would merge four 5 mm slices. |
| Nearest-neighbour for masks, linear for images (stage 4 `order=0/1`) | Correct. |
| LPS→RAS conversion of Slicer fiducials honouring `coordinateSystem` | Correct (tested with LPS and RAS files). |
| Radiological display math (`rot90` + `flip`, `draw_radiological_rect`) | Correct: patient right appears on image left, anterior at the top (tested with a marker voxel and on the phantom QC). |
| Per-image z-scoring of T1c/T1/T2/B0/DWI | Leakage-free: statistics come from the image itself. |
| ADC kept in physical units until training, then scaled per CV fold | Correct principle. See finding M6 for the estimator. |
| Exclusion of vendor-derived maps (eADC, exp, sDWI) | Correct; these are not reproducible across vendors. |
| Text fallback for diffusion without b-values is logged to an audit file | Good practice; kept (now a manifest column). |

## High: could change which data the model saw

**H1. Registration is not reproducible.** `SetMetricSamplingStrategy(RANDOM)` without a seed
draws new sample points on every run. Two identical runs on the same pair differed by
~0.2 mm in translation (`tests/test_registration.py` shows this). The thesis tensors therefore
cannot be regenerated bit-for-bit. *Thesis impact:* small (sub-voxel at 1 mm). It is still a
reproducibility gap a reviewer can detect. *Fix:* seed passed to the sampler (`registration.seed`,
default 42). With `num_threads: 1`, runs are bit-identical. That requires ITK's *global* thread
setting; the per-filter setting is not enough. This was verified and is handled in `register()`.

**H2. Contrast-enhanced T1 detected by the substring `"ce"`.** The rule
`any(sub in desc for sub in ["gd", "+c", "post", "gad", "ce"])` fires on `t1_space_tra`,
`t1_tse_tra_reference`, `...sequence...`, `T1 slice`. A non-contrast T1 can become `T1c.nii.gz`,
and then it is the registration anchor too. *Thesis impact:* unknown; depends on your protocol
names. **Action:** run `hn-mri-prep classify <sorted_dicom_dir>`. It lists every series whose
class changes between the legacy and the fixed rules. *Fix:* short markers match whole tokens
only (`legacy_classifier`).

**H3. The primary series depended on file-system order.** When several series qualify for one
slot, the first one `glob` returns becomes `T1c.nii.gz`, `ADC.nii.gz` and so on, and the rest
become `_run-2`. The spacing audit shows **397 ADC files for 280 subjects**, so at least ~117
subjects had more than one ADC candidate. The ADC that was used may come from a *different DWI
acquisition* than B0. It is still warped with B0's transform, which is only correct if the
patient did not move between the two. *Fix:* candidates are ordered by SeriesNumber; B0/DWI are
taken from the same series; the primary ADC is the one on the B0 voxel grid.
`conversion_manifest.csv` records the choice, and `preprocess_audit.csv` has
`ADC_on_B0_grid` / `DWI_on_B0_grid` per subject. **Action:** check those columns for the thesis cohort.

**H4. Structural and diffusion slice windows could be offset (stage 4).** The diffusion
window is shifted to avoid incomplete EPI slices, but the structural window is not. When the
tumour is near the end of the EPI slab, slice *k* of the T1c tensor and slice *k* of the ADC
tensor are different anatomical levels (2 slices = 10 mm in the regression test). *Thesis
impact:* only matters if channels were combined slice-by-slice (early fusion / 2.5D). **Action:**
the ROI manifest written with `thesis_v5.yaml` shows how many subjects had
`box_window_struct - center_struct != box_window_diff - center_diff`. *Fix:*
`roi.z_window_mode: shared`.

**H5. Variable tensor depth (stage 4).** `calculate_padded_z_window` clamps at the end, so a
usable EPI slab shorter than 11 slices gives a tensor with fewer than 11 slices. *Thesis
impact:* if it happened, the dataloader must have padded it or failed. *Fix:* the window always
has `box_depth` slices; slices outside the volume are zero-padded, and `box_fits_valid_range` is
recorded per subject. Not configurable (it is a shape bug).

## Medium: affects image quality or methodological defensibility

**M1. Uncovered voxels enter the z-score.** The normalisation mask comes from T1c. If T1 or
T2 covers fewer slices, or shifts after registration, voxels with no data (value 0) are inside
the mask. In the test case they moved the mean from 105 to 72 and inflated the std ~17×. N4
tolerates this (ITK skips non-positive voxels); the z-score does not. *Fix:*
`intensity.coverage_aware_masks` intersects each mask with the moving image's resampled field
of view. `coverage_T1`/`coverage_T2` are logged, which is a useful QC metric in itself.

**M2. Down-sampling without anti-aliasing.** T2 goes from 0.37 mm to 1.0 mm in-plane (2.7×)
with plain linear interpolation, which aliases fine texture. A grating test shows the aliased
residue (std 0.3 vs <0.05 filtered). *Fix:* `resampling.antialias` applies a per-axis Gaussian
(σ from the resolution change) only to down-sampled axes. The 5 mm slice axis is never smoothed.

**M3. Rigid transforms rotate about the scanner origin.** `Euler3DTransform()` without
initialisation has its centre at (0,0,0) mm, which can be 100+ mm from the anatomy, so any
rotation implies a large translation. *Fix:* `registration.centered_init` (geometry centre).

**M4. ADC units are vendor-dependent.** Siemens/GE typically export 10⁻⁶ mm²/s integers; others
use 10⁻³ mm²/s or mm²/s floats. Fold-wise min/max does not fix a factor-1000 scale difference
between subjects. *Fix:* `intensity.harmonize_adc_units` applies a fixed physical rule based on the
order of magnitude of median tissue ADC. It is per image and decided in advance, so it is leakage-free.
The detected unit is logged. If the cohort is single-vendor this is a no-op, and you can state that.

**M5. Repeated PatientIDs became separate subjects.** A second archive of the same patient
became `<ID>_2`. If both reached the model, they could land in different folds: patient-level
leakage. *Fix:* flagged in `sort_manifest.csv` (`possible_duplicate_patient`). Pass a
subject→patient map to `assert_group_disjoint` / `fit_fold_scalers(groups=...)`, or use
`GroupKFold`/`StratifiedGroupKFold`. **Action:** check the thesis cohort for `_2` folders.

**M6. Fold-wise ADC min/max is outlier-driven.** Leakage-free, but the maximum over millions of
voxels is set by one extreme voxel (CSF, cyst, artefact, 4095 saturation). If background zeros
are included, the minimum is always 0. In the test, one artefact voxel in one subject moved the
fold scale by >2000 units; the 99.5th percentile moved by <5. *Fix:* `FoldwiseADCScaler(method=
"percentile")`. `"minmax"` reproduces the thesis; `"fixed"` (0–3000) is the zero-estimation
sensitivity analysis. Reporting all three in a supplementary table pre-empts a reviewer question.

**M7. Re-runs were not idempotent.** `safe_copy` appends `_run-N` when a file exists, and the
temporary dcm2niix folder accumulates files across runs. A second run therefore creates
duplicates, and the primary is whatever survived from the first run. *Fix:* each subject's
temp/output folder is rebuilt on every run.

## Low: correctness hygiene and maintainability

- **L1.** The reference grid kept the anchor origin when the spacing changed, a shift of
  (new−old)/2 per axis (≤0.3 mm in-plane). `resampling.preserve_extent`.
- **L2.** `ResampleImageFilter` keeps the *input* pixel type, so int16 images were
  re-quantised to integers after interpolation. Negligible at MR intensity ranges; outputs are
  now float32.
- **L3.** Stage 1 copied files as `f.name`, so equal names from different sub-folders of an archive
  overwrote each other silently. Files are now named by SOPInstanceUID.
- **L4.** Stage 1 read every DICOM header three times (patient ID, series grouping, sort by
  InstanceNumber). The sort had no effect, since dcm2niix orders slices itself. Now one read.
- **L5.** Stage 4 iterated over every folder in the input, including its own outputs (`48mm`,
  `Expert_QC_Sheets`...), visible in the notebook output. Subjects are now selected by content.
- **L6.** The error path `print(f"... bvals={len(bvals)}")` raises `TypeError` when `bvals` is
  `None`, hiding the real message.
- **L7.** ITK exceptions (e.g. "All samples map outside moving image buffer") were caught by
  the generic batch handler. They now raise `RegistrationError` and are logged in the audit CSV.
- **L8.** Duplicate `preview_patient_extraction` definitions; the QC variable `adc_arr` holds B0.
- **L9.** Hard-coded Windows paths, `print` logging, no record of parameters or library versions.
  Now: YAML config, `logging`, `config_used.yaml` + `provenance.json` per output.

## Privacy: fix before anything is pushed

- The notebook **outputs contain patient identifiers** (`Processing: 6453537`) and calls such as
  `preview_patient_extraction("4494100", ...)`, plus local paths containing your user name.
  Do not commit the original notebooks. If you want them in the repository for provenance, strip
  outputs (`nbstripout`) and replace IDs first. Note that `git rm` does not remove a file from history.
- Hospital PatientIDs were used as folder and file names, so every tensor, QC image and CSV
  carries them. Use `sort-dicom --pseudonymize PG --id-map <private>/key.csv` for new runs.
  The `.gitignore` and the pre-commit hook block imaging, tensor, manifest and key files.
- dcm2niix strips most identifiers from the JSON sidecars by default (`-ba y`), but QC images can
  still show burnt-in annotations. Review them before sharing.

## A note on the novelty claim

Generic tools do handle anisotropy: nnU-Net's resampling treats the low-resolution axis
separately, and MONAI and TorchIO provide the building blocks. Public head-and-neck MRI
challenges also exist, e.g. the MICCAI 2024 HNTS-MRG challenge on T2-weighted radiotherapy MRI.
What I am not aware of is an open, end-to-end, *auditable* pipeline for **multiparametric**
(T1/T1c/T2/DWI/ADC) parotid MRI from heterogeneous vendor DICOM exports, including EPI-to-structural
alignment and fold-safe ADC handling. Frame the contribution that way, and do a systematic
search before writing "no existing tool". Reviewers will check.
