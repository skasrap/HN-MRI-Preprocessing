"""Pipeline configuration.

All parameters live here and are loaded from YAML (see ``configs/``). Defaults match
``configs/recommended.yaml``; ``configs/thesis_v5.yaml`` reproduces the thesis runs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class ConversionConfig:
    """Stage 2: DICOM series -> canonical NIfTI files per subject."""

    # substring matching for "ce" (thesis); also matches "space", "reference", ...
    legacy_classifier: bool = False
    # which candidate (by SeriesNumber) becomes T1c.nii.gz etc.; the rest are *_run-N
    primary_series: Literal["first", "last"] = "first"
    skip_description_tokens: tuple[str, ...] = ("dyn",)
    low_b_max: float = 100.0
    high_b_min: float = 700.0


@dataclass(frozen=True)
class ResamplingConfig:
    struct_spacing: tuple[float, float, float] = (1.0, 1.0, 5.0)
    diff_spacing: tuple[float, float, float] = (1.0, 1.0, 5.0)
    # Gaussian pre-filter on down-sampled axes
    antialias: bool = True
    # keep the anchor's physical extent when the spacing changes (sub-voxel shift)
    preserve_extent: bool = True


@dataclass(frozen=True)
class RegistrationConfig:
    # MI sampling seed; None = unseeded (thesis), not reproducible
    seed: int | None = 42
    # rotate about the image centre instead of the scanner origin
    centered_init: bool = True
    histogram_bins: int = 50
    struct_sampling: float = 0.15
    diff_sampling: float = 0.20
    learning_rate: float = 0.1
    iterations: int = 200
    # reject the subject if the final metric is above this
    fail_threshold: float = -0.05
    # None = all cores; 1 gives bit-identical results
    num_threads: int | None = None


@dataclass(frozen=True)
class MaskConfig:
    closing_radius_mm: float = 20.0
    struct_smoothing_sigma: float = 1.0
    diff_smoothing_sigma: float = 2.0
    diff_clip_percentile: float = 99.5


@dataclass(frozen=True)
class IntensityConfig:
    n4_shrink_factor: int = 3
    zscore_clip_percentiles: tuple[float, float] = (0.5, 99.5)
    # exclude voxels outside a moving image's FOV from its N4/z-score mask
    coverage_aware_masks: bool = True
    # rescale ADC to 1e-6 mm^2/s whatever the vendor export unit
    harmonize_adc_units: bool = True


@dataclass(frozen=True)
class CropConfig:
    struct_xy: int = 320
    diff_xy: int = 320
    # one bounding box for all channels (thesis: separate structural/diffusion boxes)
    shared_bbox: bool = True


@dataclass(frozen=True)
class PreprocessConfig:
    required_sequences: tuple[str, ...] = ("T1c", "T1", "T2", "B0", "DWI", "ADC")
    save_transforms: bool = True
    save_qc: bool = True
    resampling: ResamplingConfig = field(default_factory=ResamplingConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    masks: MaskConfig = field(default_factory=MaskConfig)
    intensity: IntensityConfig = field(default_factory=IntensityConfig)
    crop: CropConfig = field(default_factory=CropConfig)


@dataclass(frozen=True)
class ROIConfig:
    """Stage 4: fiducial-centred patch extraction."""

    target_xy: int = 224
    box_sizes_mm: dict[str, float] = field(
        default_factory=lambda: {"48mm": 48.0, "64mm": 64.0, "80mm": 80.0, "90mm": 90.0}
    )
    box_depth: int = 11
    whole_depth: int = 3
    pad_value: float = 0.0
    valid_slice_tolerance: float = 0.05
    # "shared": same slices for all sequences. "independent" (thesis): only the diffusion
    # window is shifted away from incomplete EPI slices.
    z_window_mode: Literal["shared", "independent"] = "shared"
    fiducial_filenames: tuple[str, ...] = ("msk.json", "F.mrk.json")
    sequences: tuple[str, ...] = ("T1", "T2", "T1c", "B0", "DWI", "ADC", "StructMask", "DiffMask")
    save_qc: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    conversion: ConversionConfig = field(default_factory=ConversionConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        """Short, stable hash of the configuration, stored in every provenance record."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=list)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def dump(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(_listify(self.to_dict()), f, sort_keys=False)


def _listify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _listify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_listify(v) for v in obj]
    return obj


def _build(cls: type, data: dict[str, Any] | None) -> Any:
    data = data or {}
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        default = getattr(cls(), name)
        if dataclasses.is_dataclass(default):
            kwargs[name] = _build(type(default), value)
        elif isinstance(default, tuple) and value is not None:
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load a YAML config. Missing keys fall back to defaults; unknown keys are an error."""
    if path is None:
        return PipelineConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return _build(PipelineConfig, data)
