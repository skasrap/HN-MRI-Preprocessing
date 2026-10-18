# Pipeline

## Stages

1. **sort-dicom**: ZIP exports to `<subject>/<series>/` folders, one per SeriesInstanceUID.
2. **convert**: dcm2niix, then rule-based assignment of each series to T1, T1c, T2, B0
   (b ≤ 100), DWI (b ≥ 700) or ADC. Only axial series are used for T1/T1c/T2. 4D diffusion
   series are split into the lowest- and highest-b volumes.
3. **preprocess**: see below.
4. **extract-rois**: patches around a tumour fiducial placed in 3D Slicer (`F.mrk.json` or
   `msk.json` in the subject folder).

## Preprocessing (stage 3)

Everything is reoriented to RAS. T1 and T2 are registered rigidly to T1c, and B0 affinely to
T2. DWI and ADC get the B0 transform composed with T2 → T1c, so each image is interpolated
once. All images go to a 1 × 1 × 5 mm grid anchored on T1c. Structural slices are already
5 mm, so for them only the in-plane axes are down-sampled; DWI is also down-sampled from
~3.85 mm through-plane. Down-sampled axes are anti-aliased first.

Masks: Li threshold on T1c, and Otsu on log B0 intersected with the T1c mask. Then the
largest component, closing (20 mm in-plane, 1 slice through-plane) and hole filling.
Structural images get N4. T1c, T1, T2, B0 and DWI are z-scored inside their mask; ADC is
only masked and stays in physical units. Volumes are cropped to the tissue bounding box and
padded/cropped to 320 × 320.

## ROI extraction (stage 4)

Boxes of 48, 64, 80 and 90 mm around the fiducial, resized to 224 × 224, 11 slices. The
slice window avoids diffusion slices with more than 5% of the tissue missing (the ends of the
EPI slab after registration). With `z_window_mode: shared` the same slices are used for every
sequence. `Whole_3Slice` holds 3 full slices padded to square.

## Parameters

| Stage | Parameter | Value |
|---|---|---|
| Conversion | b-value thresholds | ≤ 100 (B0), ≥ 700 (DWI) s/mm² |
| Registration | T1, T2 → T1c | rigid, centred initialisation |
| | B0 → T2 | affine |
| | metric | Mattes MI, 50 bins; 15 % (rigid) / 20 % (affine) random samples, seed 42 |
| | optimiser | gradient descent with line search, LR 0.1, ≤ 200 iterations |
| | pyramid | shrink 4/2/1, σ 2/1/0 mm (rigid); shrink 2/1, σ 2/1 mm (affine) |
| | rejection | final metric > −0.05 |
| Resampling | grid | 1.0 × 1.0 × 5.0 mm, linear |
| | anti-aliasing | σ = √(t² − s²) / 2.355 per down-sampled axis |
| Masks | thresholds | Li (σ 1 mm) on T1c; Otsu on log₁₀ B0 (σ 2 mm) |
| | closing | 20 mm in-plane, 1 slice |
| Intensity | N4 | shrink 3, structural only |
| | z-score | mean/std of 0.5–99.5th percentile-clipped mask voxels |
| | ADC | 1e-6 mm²/s, masked, scaled per CV fold at training time |
| Cropping | in-plane | 320 × 320 |
| ROI | boxes | 48, 64, 80, 90 mm → 224 × 224 |
| | depth | 11 slices (boxes), 3 (whole slice) |
| | EPI slice validity | ≤ 5 % missing tissue |

## Outputs worth keeping

- `config_used.yaml` in each output folder; `provenance.json` per subject (config hash, git
  commit, library versions), plus the saved `.tfm` transforms.
- `sort_manifest.csv`, `conversion_manifest_all.csv`, `preprocess_audit.csv`,
  `roi_manifest.csv`: per-subject status and failure reasons (useful for the flow diagram).
- `hn-mri-prep audit-spacing`: acquisition spacing per sequence.
