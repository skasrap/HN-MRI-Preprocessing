# Changelog

## 0.1.0 (unreleased)

First packaged release of the thesis preprocessing code (`Unzip.ipynb`,
`All_Preprocessing_Script_v5.py`, `Preprocess_EPI_v1.ipynb`, `Fids_v4.ipynb`).
`configs/thesis_v5.yaml` reproduces the thesis pipeline. See `docs/code_review.md` for the
reasoning behind every change.

### Fixed
- Contrast detection matched the substring `ce` (space, sequence, reference, slice ...). It now matches whole tokens.
- Primary series selection depended on file-system order. It now uses DICOM SeriesNumber.
- Re-running conversion accumulated `_run-2` duplicates. Re-runs are now idempotent.
- DICOM files with identical names in one archive could overwrite each other. They are now named by SOPInstanceUID.
- Registration was non-deterministic (unseeded random sampling). It is now seeded, and `num_threads: 1` makes it bit-identical.
- ITK registration exceptions now surface as `RegistrationError` with the reason in the audit CSV.
- ROI tensors could have fewer than 11 slices when the EPI slab was short. Depth is now fixed.
- The diffusion slice window could be offset from the structural window (channel misalignment). Added `z_window_mode: shared`.
- The subject loop in ROI extraction iterated over output folders (`48mm`, `QC`, ...). It now selects subject folders only.

### Added
- Optional pseudonymisation and duplicate-patient detection in DICOM sorting.
- Optional anti-aliased down-sampling, extent-preserving reference grid and centred registration initialisation.
- Coverage-aware normalisation masks and ADC unit harmonisation.
- Shared bounding box so all six channels are voxel-aligned.
- `FoldwiseADCScaler` (min/max, percentile or fixed range) with leakage checks.
- Per-subject `provenance.json`, saved transforms, conversion/preprocessing/ROI manifests.
- CLI, typed YAML config, test suite on a synthetic phantom, CI.
