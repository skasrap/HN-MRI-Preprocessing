# Changes since the thesis version

The thesis results were produced with the original scripts (`Unzip.ipynb`,
`All_Preprocessing_Script_v5.py`, `Fids_v4.ipynb`). `configs/thesis_v5.yaml` switches the
changes below off to reproduce that pipeline. The only exception is the registration seed:
the thesis runs were unseeded, so their transforms can't be reproduced exactly.

## Behaviour changes (config switches)

| Switch | Thesis | Now | Why |
|---|---|---|---|
| `conversion.legacy_classifier` | substring match for contrast markers | whole tokens for `ce`/`c` | `"ce"` also matched `t1_space`, `..._reference`, `sequence`, `slice` |
| `registration.seed` | none | 42 | unseeded MI sampling gave ~0.2 mm differences between identical runs |
| `registration.centered_init` | off | on | rigid transforms otherwise rotate about the scanner origin |
| `resampling.antialias` | off | on | T2 goes from ~0.37 mm to 1 mm in-plane; the slice axis is never smoothed |
| `resampling.preserve_extent` | off | on | the grid origin was kept when spacing changed (≤0.3 mm shift) |
| `intensity.coverage_aware_masks` | off | on | zeros outside a shorter T1/T2 stack were counted in the z-score statistics |
| `intensity.harmonize_adc_units` | off | on | ADC export units differ by vendor (1e-6, 1e-3 or 1 mm²/s) |
| `crop.shared_bbox` | separate structural / diffusion boxes | one box | keeps all channels voxel-aligned |
| `roi.z_window_mode` | `independent` | `shared` | the diffusion window could shift while the structural one didn't, so slice *k* of T1c and ADC could differ by up to 2 slices |

## Fixes (not configurable)

- The primary series (e.g. `ADC.nii.gz` when a subject has several) depended on file-system
  order. It's now chosen by SeriesNumber; B0/DWI come from the same series, and an ADC on the
  B0 grid is preferred. The thesis cohort had 397 ADC files for 280 subjects, so check
  `ADC_on_B0_grid` in `preprocess_audit.csv`.
- ROI tensors could have fewer than `box_depth` slices when the usable EPI slab was short.
  Depth is now fixed, with zero padding; `box_fits_valid_range` marks these subjects.
- Re-running the conversion added `_run-2` duplicates. Subject folders are now rebuilt.
- DICOM files with the same name in different folders of one archive could overwrite each
  other. They're now named by SOPInstanceUID.
- The ROI step also iterated over its own output folders (`48mm`, `QC`, ...).
- ITK registration errors are reported as `RegistrationError` with the message in the audit CSV.

## Checking the thesis cohort

- `hn-mri-prep classify <dicom_dir>`: series whose class differs between the old and new rules.
- `preprocess_audit.csv`: `ADC_on_B0_grid`, `DWI_on_B0_grid`, `coverage_T1`, `coverage_T2`.
- `roi_manifest.csv` (run with `thesis_v5.yaml`): subjects where `box_window_struct` and
  `box_window_diff` are offset differently from their centres.
- `sort_manifest.csv`: `possible_duplicate_patient`.
