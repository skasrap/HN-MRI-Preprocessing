# HN-MRI-Preprocessing

[![CI](https://github.com/skasrap/HN-MRI-Preprocessing/actions/workflows/ci.yml/badge.svg)](https://github.com/skasrap/HN-MRI-Preprocessing/actions)

Preprocessing for multiparametric head-and-neck MRI (T1, T1c, T2, DWI, ADC), from DICOM
exports to fixed-size tensors. I wrote it for my MD thesis on deep-learning classification of
parotid gland tumours; the manuscript is in preparation.

Most open preprocessing pipelines are built for brain MRI. Head-and-neck data is messier:
sequence names vary between vendors and sites, the structural series are thick 2D stacks
(~0.4 × 0.4 × 5 mm), and diffusion EPI comes on its own grid (~1 × 1 × 3.85 mm) with its own
distortion. The pipeline deals with that in four steps:

```
ZIP exports -> sort-dicom -> convert -> preprocess -> extract-rois -> .npy tensors
```

- **sort-dicom**: one folder per subject and series. Optional pseudonymisation.
- **convert**: dcm2niix plus rules that pick T1, T1c, T2, B0, DWI and ADC for each subject.
  Every decision is logged.
- **preprocess**: registration (T1, T2 → T1c rigid; B0 → T2 affine), resampling to
  1 × 1 × 5 mm, N4, z-scoring, tissue masks and cropping. ADC is kept in physical units.
- **extract-rois**: 48–90 mm boxes around a tumour fiducial, 11 × 224 × 224.

ADC is scaled to [0, 1] per cross-validation fold at training time, using only that fold's
training subjects ([docs/data_leakage.md](docs/data_leakage.md)).

The first three steps aren't parotid-specific. Box sizes, depth and sequences in the last step
are set in the config.

## Install

```bash
git clone https://github.com/skasrap/HN-MRI-Preprocessing && cd HN-MRI-Preprocessing
pip install -e ".[dev]"
```

The `convert` step needs [dcm2niix](https://github.com/rordenlab/dcm2niix) on `PATH`.

## Usage

```bash
hn-mri-prep sort-dicom    exports/ data/dicom --pseudonymize PG --id-map ~/private/key.csv
hn-mri-prep convert       data/dicom data/nifti --temp-dir data/tmp
hn-mri-prep preprocess    data/nifti data/preprocessed --workers 4
hn-mri-prep extract-rois  data/preprocessed data/tensors   # needs <subject>/F.mrk.json from 3D Slicer

hn-mri-prep audit-spacing data/nifti reports/spacing      # voxel spacing per sequence
hn-mri-prep classify      data/dicom                      # dry run of the series rules
```

All commands take `--config`. `configs/recommended.yaml` holds the defaults;
`configs/thesis_v5.yaml` reproduces the settings used in the thesis (see
[docs/changes.md](docs/changes.md)).

Output tensors are `data/tensors/{48mm,64mm,80mm,90mm,Whole_3Slice}/<subject>_<seq>.npy`:
float32, `[slice, row, col]`, radiological orientation. Each step also writes a CSV with
per-subject status, and QC images.

## Layout

```
src/hn_mri_prep/
  io/              DICOM sorting, series classification, NIfTI conversion
  preprocessing/   geometry, registration, masks, intensity, cropping, QC, per-subject pipeline
  roi/             fiducials, patch extraction, QC sheet
  normalization/   per-fold ADC scaling
  audit/           spacing statistics
configs/  docs/  tests/
```

More detail: [docs/pipeline.md](docs/pipeline.md).

## Data

No images or patient data are in this repository. `.gitignore` excludes imaging files,
tensors, manifests and ID keys, and `pre-commit install` adds a hook that blocks them and
strips notebook outputs.

## License

MIT. Research code; not for clinical use.
