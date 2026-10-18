"""Typed configuration for every pipeline stage.

Every numeric choice that used to be hard-coded in the thesis scripts lives here, so a
run is fully described by one YAML file (which is also copied into each output folder).

Defaults correspond to ``configs/recommended.yaml``. ``configs/thesis_v5.yaml`` switches
off the post-thesis fixes so the original pipeline can be reproduced.
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

    # Use whole-token matching for contrast markers ("ce", "c"). The thesis version used
    # substring matching, which labels e.g. "t1_space" or "t1_reference" as contrast-enhanced.
    legacy_classifier: bool = False
    # When several axial series map to the same sequence type, which one becomes the
    # primary file (``T1c.nii.gz``); the others are kept as ``T1c_run-2.nii.gz`` etc.
    # Ordered by DICOM SeriesNumber, which makes the choice deterministic.
    primary_series: Literal["first", "last"] = "first"
    skip_description_tokens: tuple[str, ...] = ("dyn",)
    low_b_max: float = 100.0
    high_b_min: float = 700.0


@dataclass(frozen=True)
class ResamplingConfig:
    struct_spacing: tuple[float, float, float] = (1.0, 1.0, 5.0)
    diff_spacing: tuple[float, float, float] = (1.0, 1.0, 5.0)
    # Gaussian pre-filter before down-sampling (e.g. 0.37 mm T2 -> 1 mm) to avoid aliasing.
    antialias: bool = True
    # Keep the physical field of view of the anchor when the spacing changes (shifts the
    # first voxel centre by half the spacing difference). Sub-voxel effect.
    preserve_extent: bool = True


@dataclass(frozen=True)
class RegistrationConfig:
    # Seed for Mattes-MI random sampling. ``None`` reproduces the thesis behaviour, which
    # is NOT deterministic (results differ between runs).
    seed: int | None = 42
    # Initialise rigid transforms at the geometric centre of the fixed image. Without it the
    # rotation centre is the scanner origin, which couples rotation and translation.
    centered_init: bool = True
    histogram_bins: int = 50
    struct_sampling: float = 0.15
    diff_sampling: float = 0.20
    learning_rate: float = 0.1
    iterations: int = 200
    # Final negative-MI must be below this value, otherwise the subject is rejected.
    fail_threshold: float = -0.05
    # ``None`` uses all cores. Set to 1 for strict bit-for-bit reproducibility across machines.
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
    # Restrict each moving image's mask to the voxels it actually covers after registration.
    # Without it, slices outside a shorter T1/T2 stack enter the z-score statistics as zeros.
    coverage_aware_masks: bool = True
    # Convert ADC to 1e-6 mm^2/s regardless of vendor export scale (a fixed physical rule,
    # not a data-derived statistic, so it is leakage-free).
    harmonize_adc_units: bool = True


@dataclass(frozen=True)
class CropConfig:
    struct_xy: int = 320
    diff_xy: int = 320
    # Crop structural and diffusion volumes with the same bounding box so all channels stay
    # voxel-aligned. The thesis used separate boxes (aligned only in physical space).
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
    # "shared": structural and diffusion windows cover the same slices (required for
    # channel-wise fusion). "independent": thesis behaviour; the diffusion window may be
    # shifted to avoid incomplete EPI slices while the structural one is not.
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
