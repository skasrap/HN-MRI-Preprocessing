"""Intra-subject registration (Mattes MI, multi-resolution gradient descent).

The random metric sampler gets a fixed seed; unseeded runs give slightly different transforms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import SimpleITK as sitk

from hn_mri_prep.config import RegistrationConfig

log = logging.getLogger(__name__)


class RegistrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RegistrationResult:
    transform: sitk.Transform
    metric: float
    stop_condition: str
    iterations: int


def robust_rescale(image: sitk.Image, low: float = 1.0, high: float = 99.5) -> sitk.Image:
    """Map the [low, high] percentile range of foreground intensities to [0, 1]."""
    arr = sitk.GetArrayViewFromImage(image)
    valid = arr[arr > 1e-5]
    if valid.size == 0:
        return sitk.Cast(image, sitk.sitkFloat32)
    p_lo, p_hi = np.percentile(valid, [low, high])
    if p_hi - p_lo < 1e-5:
        return sitk.Cast(image, sitk.sitkFloat32)
    return sitk.Cast(sitk.IntensityWindowing(image, float(p_lo), float(p_hi), 0.0, 1.0), sitk.sitkFloat32)


def register(
    fixed: sitk.Image,
    moving: sitk.Image,
    kind: Literal["rigid", "affine"],
    cfg: RegistrationConfig,
    label: str = "",
) -> RegistrationResult:
    """Return the transform mapping ``fixed`` physical points to ``moving`` points."""
    fixed_f, moving_f = robust_rescale(fixed), robust_rescale(moving)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=cfg.histogram_bins)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    sampling = cfg.diff_sampling if kind == "affine" else cfg.struct_sampling
    if cfg.seed is None:
        reg.SetMetricSamplingPercentage(sampling)
    else:
        reg.SetMetricSamplingPercentage(sampling, cfg.seed)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if kind == "affine":
        reg.SetShrinkFactorsPerLevel([2, 1])
        reg.SetSmoothingSigmasPerLevel([2.0, 1.0])
        initial: sitk.Transform = sitk.AffineTransform(3)
    else:
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
        initial = sitk.Euler3DTransform()

    if cfg.centered_init:
        initial = sitk.CenteredTransformInitializer(
            fixed_f, moving_f, initial, sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

    reg.SetOptimizerAsGradientDescentLineSearch(
        learningRate=cfg.learning_rate,
        numberOfIterations=cfg.iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial, inPlace=False)

    # The v4 metric uses ITK's global threader; only the global setting makes runs bit-identical.
    previous_threads = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    if cfg.num_threads is not None:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(cfg.num_threads)
    try:
        transform = reg.Execute(fixed_f, moving_f)
    except RuntimeError as e:  # e.g. "All samples map outside moving image buffer"
        raise RegistrationError(f"[{label}] ITK: {str(e).strip().splitlines()[-1]}") from e
    finally:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(previous_threads)
    metric = reg.GetMetricValue()
    if metric > cfg.fail_threshold:
        raise RegistrationError(
            f"[{label}] final metric {metric:.4f} > {cfg.fail_threshold}: no meaningful alignment"
        )
    stop = reg.GetOptimizerStopConditionDescription()
    log.info("[%s] metric %.4f | %s", label, metric, stop)
    return RegistrationResult(transform, float(metric), stop, reg.GetOptimizerIteration())
