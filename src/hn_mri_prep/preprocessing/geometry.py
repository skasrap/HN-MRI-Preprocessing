"""Orientation, reference grids and anisotropy-aware resampling.

Head-and-neck MRI is acquired as thick 2D stacks: in this cohort the median structural
voxel is 0.37-0.46 mm in-plane x 5 mm slice, and the EPI diffusion voxel 1.0 x 1.0 x 3.85 mm.
Every operation here therefore handles each axis separately; nothing assumes isotropy.
"""

from __future__ import annotations

import math

import numpy as np
import SimpleITK as sitk

_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def to_ras(image: sitk.Image) -> sitk.Image:
    """Reorient so that voxel axes point to patient Right, Anterior, Superior."""
    return sitk.DICOMOrient(image, "RAS")


def make_reference_grid(
    anchor: sitk.Image, spacing: tuple[float, float, float], preserve_extent: bool = True
) -> sitk.Image:
    """Empty grid with the anchor's orientation and field of view at a new spacing.

    With ``preserve_extent`` the first voxel centre is moved by half the spacing change, so
    the new grid covers exactly the anchor's physical extent. The thesis code kept the anchor
    origin, which shifts the grid by up to (new - old) / 2 along each axis.
    """
    old_spacing = np.asarray(anchor.GetSpacing(), dtype=float)
    new_spacing = np.asarray(spacing, dtype=float)
    size = np.maximum(1, np.round(np.asarray(anchor.GetSize()) * old_spacing / new_spacing)).astype(int)

    origin = np.asarray(anchor.GetOrigin(), dtype=float)
    if preserve_extent:
        direction = np.asarray(anchor.GetDirection(), dtype=float).reshape(3, 3)
        origin = origin + direction @ ((new_spacing - old_spacing) / 2.0)

    grid = sitk.Image([int(s) for s in size], sitk.sitkFloat32)
    grid.SetSpacing(new_spacing.tolist())
    grid.SetDirection(anchor.GetDirection())
    grid.SetOrigin(origin.tolist())
    return grid


def antialias_sigmas(source_spacing, target_spacing) -> list[float]:
    """Per-axis Gaussian sigma (mm) that brings the source resolution down to the target.

    Uses sigma = sqrt(t^2 - s^2) * FWHM-to-sigma. Axes that are not down-sampled (typically
    the 5 mm slice axis) get sigma 0, so through-plane resolution is never blurred further.
    """
    s = np.asarray(source_spacing, dtype=float)
    t = np.asarray(target_spacing, dtype=float)
    return (np.sqrt(np.clip(t**2 - s**2, 0.0, None)) * _FWHM_TO_SIGMA).tolist()


def smooth_anisotropic(image: sitk.Image, sigmas_mm: list[float]) -> sitk.Image:
    if not any(s > 0 for s in sigmas_mm):
        return image
    f = sitk.DiscreteGaussianImageFilter()
    f.SetVariance([s**2 for s in sigmas_mm])
    f.SetUseImageSpacing(True)
    f.SetMaximumKernelWidth(64)
    return f.Execute(sitk.Cast(image, sitk.sitkFloat32))


def resample(
    moving: sitk.Image,
    reference: sitk.Image,
    transform: sitk.Transform | None = None,
    interpolator: int = sitk.sitkLinear,
    antialias: bool = False,
    default_value: float = 0.0,
) -> sitk.Image:
    """Resample ``moving`` onto ``reference`` (transform maps reference -> moving points)."""
    if antialias and interpolator != sitk.sitkNearestNeighbor:
        moving = smooth_anisotropic(moving, antialias_sigmas(moving.GetSpacing(), reference.GetSpacing()))
    return sitk.Resample(
        moving,
        reference,
        transform if transform is not None else sitk.Transform(),
        interpolator,
        default_value,
        moving.GetPixelID() if interpolator == sitk.sitkNearestNeighbor else sitk.sitkFloat32,
    )


def coverage_mask(moving: sitk.Image, reference: sitk.Image, transform: sitk.Transform | None) -> sitk.Image:
    """Voxels of ``reference`` that fall inside the field of view of ``moving``."""
    ones = sitk.Image(moving.GetSize(), sitk.sitkUInt8)
    ones.CopyInformation(moving)
    ones = ones + 1
    return resample(ones, reference, transform, sitk.sitkNearestNeighbor)


def same_grid(a: sitk.Image, b: sitk.Image, tol: float = 1e-3) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol)
    )
