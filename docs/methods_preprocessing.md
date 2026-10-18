# Image preprocessing: methods text for the manuscript

A draft paragraph that matches `configs/recommended.yaml`. Bracketed items must be filled in
from your data; adjust any sentence whose setting you change.

> **Image preprocessing.** DICOM series were converted to NIfTI with dcm2niix (v[x]) and assigned
> to sequence types (T1-weighted, contrast-enhanced T1-weighted, T2-weighted, diffusion-weighted,
> ADC) with predefined rules on series descriptions and DICOM image-type and b-value metadata.
> Only axial acquisitions were used. When several series qualified, the one with the lowest series
> number was selected, and the low- and high-b images (b ≤ 100 and b ≥ 700 s/mm²) and the ADC map
> were taken from the same diffusion acquisition. Images were reoriented to RAS. T1-weighted and
> T2-weighted images were rigidly registered to the contrast-enhanced T1-weighted image, and the
> low-b diffusion image was affinely registered to the T2-weighted image (Mattes mutual information,
> multi-resolution gradient descent, SimpleITK v[x], fixed random seed). Diffusion-weighted and
> ADC images were transformed with the low-b transform composed with the T2-to-T1c transform, so
> each image was interpolated once. Images were resampled to a common 1.0 × 1.0 × 5.0 mm grid
> anchored on the contrast-enhanced T1-weighted image, with Gaussian anti-aliasing along the
> in-plane axes that were down-sampled. Through-plane resolution was preserved because acquisitions
> were anisotropic (median voxel [0.44 × 0.44 × 5.0] mm structural, [1.0 × 1.0 × 3.85] mm diffusion).
> Head-and-neck tissue masks were obtained by Li thresholding (structural) and Otsu thresholding
> of the log-transformed low-b image (diffusion), each followed by largest-component selection,
> morphological closing (20 mm in-plane radius) and hole filling. Structural images underwent N4
> bias-field correction. T1-weighted, T2-weighted and diffusion-weighted images were
> z-score-normalised within their masks, restricted to voxels covered by each image's field of view.
> ADC maps were converted to 10⁻⁶ mm²/s and otherwise kept in physical units. Volumes were cropped
> to the tissue bounding box and padded or centre-cropped to 320 × 320 mm in-plane. Patches of
> 48, 64, 80 and 90 mm were centred on a tumour fiducial placed by [reader, experience, blinding],
> resampled to 224 × 224 pixels and limited to 11 contiguous 5-mm slices shared by all sequences,
> excluding diffusion slices with more than 5% missing tissue. ADC values were scaled to [0, 1]
> using the [0.5th and 99.5th percentiles / minimum and maximum] of the training subjects within
> each cross-validation fold, which were then applied unchanged to that fold's validation subjects.
> All preprocessing code and parameters are available at [repository URL, release/DOI].

## Parameter table (supplementary)

| Stage | Parameter | Value |
|---|---|---|
| Conversion | Low / high b-value thresholds | ≤ 100 / ≥ 700 s/mm² |
| Registration | T1, T2 → T1c | Rigid (Euler 3D), centred initialisation |
| | B0 → T2 | Affine |
| | Metric | Mattes MI, 50 bins, 15% (rigid) / 20% (affine) random samples, seed 42 |
| | Optimiser | Gradient descent with line search, LR 0.1, ≤ 200 iterations |
| | Pyramid | Shrink 4/2/1, σ 2/1/0 mm (rigid); shrink 2/1, σ 2/1 mm (affine) |
| | Rejection | Final metric > −0.05 |
| Resampling | Grid | 1.0 × 1.0 × 5.0 mm, anchored on T1c, linear interpolation |
| | Anti-aliasing | σ = √(t² − s²) / 2.355 per down-sampled axis |
| Masks | Structural / diffusion threshold | Li (σ 1 mm smoothing) / Otsu on log₁₀ (σ 2 mm) |
| | Closing | 20 mm in-plane, 1 slice through-plane |
| Intensity | N4 | Shrink factor 3, structural images only |
| | Z-score | Within coverage-restricted mask; statistics on 0.5–99.5th percentile clipped voxels |
| | ADC | 10⁻⁶ mm²/s, masked; fold-wise scaling at training time |
| Cropping | In-plane size | 320 × 320 voxels (mm) |
| ROI | Boxes | 48, 64, 80, 90 mm, resampled to 224 × 224 |
| | Depth | 11 slices (boxes), 3 slices (whole slice); shared across sequences |
| | EPI slice validity | ≤ 5% missing tissue |

## Reporting checklist hooks

| Item | Source in this repository |
|---|---|
| CLAIM: data preprocessing steps | this document, `config_used.yaml` |
| CLAIM / STARD: exclusions and failures | `preprocess_audit.csv`, `roi_manifest.csv` (`status`, `error`) |
| CLAIM: software versions | `provenance.json` per subject |
| STARD 6–9: acquisition parameters | `hn-mri-prep audit-spacing` (`spacing_summary.csv`) |
| Code availability | tagged release + Zenodo DOI (recommended) |
