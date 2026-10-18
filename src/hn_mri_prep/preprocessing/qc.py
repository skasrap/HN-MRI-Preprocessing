"""Quality-control figures for preprocessed subjects.

Display convention (radiological): patient Right on the image Left, Anterior at the top.
Volumes are RAS-oriented, so ``array[z]`` has rows along +A and columns along +R; flipping
both axes gives the radiological view.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from hn_mri_prep.preprocessing.geometry import resample

_TXT_FX = [path_effects.withStroke(linewidth=2.5, foreground="black")]


def radiological(slice_2d: np.ndarray) -> np.ndarray:
    return np.flipud(np.fliplr(slice_2d))


def window(slc: np.ndarray, mask: np.ndarray, is_diff: bool = False) -> tuple[float, float]:
    valid = slc[mask > 0.5]
    if valid.size == 0:
        return -2.0, 2.0
    lo = np.percentile(valid, 2 if is_diff else 1)
    hi = np.percentile(valid, 95 if is_diff else 99)
    return (-2.0, 2.0) if hi - lo < 1e-8 else (float(lo), float(hi))


def to_unit(slc: np.ndarray, mask: np.ndarray, is_diff: bool = False) -> np.ndarray:
    lo, hi = window(slc, mask, is_diff)
    out = np.clip((slc - lo) / (hi - lo), 0, 1)
    out[mask < 0.5] = 0.0
    return out


def _on(image: sitk.Image, reference: sitk.Image, nearest: bool = False) -> np.ndarray:
    interp = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
    return sitk.GetArrayFromImage(resample(image, reference, None, interp))


def _orientation_labels(ax, shape) -> None:
    h, w = shape
    for txt, x, y, ha, va in (("A", w / 2, 15, "center", "top"), ("P", w / 2, h - 15, "center", "bottom"),
                              ("R", 15, h / 2, "left", "center"), ("L", w - 15, h / 2, "right", "center")):
        ax.text(x, y, txt, color="#00FFFF", fontsize=14, fontweight="bold", ha=ha, va=va, path_effects=_TXT_FX)


def save_compact_qc(images: dict[str, sitk.Image | None], struct_mask, diff_mask, subject: str, path: Path) -> None:
    """2 x 4 dashboard on the mid-diffusion slice: structural row, diffusion row, overlays."""
    ref = images["T2"]
    arrays = {k: (sitk.GetArrayFromImage(v) if k in ("T1c", "T1", "T2") else _on(v, ref)) if v is not None else None
              for k, v in images.items()}
    s_mask = sitk.GetArrayFromImage(struct_mask)
    d_mask = _on(diff_mask, ref, nearest=True)

    z_valid = np.where(np.any(arrays["B0"] > 1e-5, axis=(1, 2)))[0]
    z = int(z_valid[len(z_valid) // 2]) if z_valid.size else arrays["T2"].shape[0] // 2
    slc = {k: radiological(a[z]) if a is not None else None for k, a in arrays.items()}
    sm, dm = radiological(s_mask[z]), radiological(d_mask[z])

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.patch.set_facecolor("black")
    fig.suptitle(f"QC: {subject} (slice z={z})", color="white", fontsize=20, fontweight="bold")
    panels = [("T1c", "T1c (anchor)", sm, False), ("T1", "T1 (registered)", sm, False),
              ("T2", "T2 (registered)", sm, False), None,
              ("B0", "B0 (z-scored)", dm, True), ("DWI", "DWI (z-scored)", dm, True),
              ("ADC", "ADC", dm, False), None]
    for ax, panel in zip(axes.flat, panels, strict=False):
        ax.axis("off")
        if panel is None:
            continue
        key, title, mask, is_diff = panel
        if slc[key] is None:
            ax.text(0.5, 0.5, "Not available", color="gray", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, color="gray", fontsize=14)
            continue
        lo, hi = window(slc[key], mask, is_diff)
        ax.imshow(np.where(mask > 0.5, slc[key], lo), cmap="gray", vmin=lo, vmax=hi)
        ax.contour(mask, levels=[0.5], colors=["#FFFF00"], linewidths=1.2, alpha=0.7)
        _orientation_labels(ax, slc[key].shape)
        ax.set_title(title, color="white", fontsize=16, path_effects=_TXT_FX)

    def overlay(ax, red, cyan, title):
        rgb = np.stack([red, cyan, cyan], axis=-1)
        ax.imshow(rgb)
        _orientation_labels(ax, red.shape)
        ax.set_title(title, color="white", fontsize=16, path_effects=_TXT_FX)

    overlay(axes[0, 3], to_unit(slc["T1c"], sm), to_unit(slc["T2"], sm), "Red=T1c, Cyan=T2")
    overlay(axes[1, 3], to_unit(slc["T2"], sm), to_unit(slc["B0"], dm, True), "Red=T2, Cyan=B0")

    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_multislice_qc(images: dict[str, sitk.Image], struct_mask, diff_mask, subject: str, path: Path,
                       num_slices: int = 6) -> None:
    """Rows T1c / T2 / B0 / ADC / T2+B0 overlay across the diffusion slab, to check z alignment."""
    ref = images["T2"]
    t1c, t2 = sitk.GetArrayFromImage(images["T1c"]), sitk.GetArrayFromImage(ref)
    b0, adc = _on(images["B0"], ref), _on(images["ADC"], ref)
    s_mask, d_mask = sitk.GetArrayFromImage(struct_mask), _on(diff_mask, ref, nearest=True)

    z_valid = np.where(np.any(b0 > 1e-5, axis=(1, 2)))[0]
    if z_valid.size == 0:
        return
    zs = np.linspace(z_valid[0], z_valid[-1], num_slices, dtype=int)

    fig, axes = plt.subplots(5, num_slices, figsize=(3.5 * num_slices, 17.5), squeeze=False)
    fig.patch.set_facecolor("black")
    fig.suptitle(f"{subject} | slices {z_valid[0]}-{z_valid[-1]}", color="white", fontsize=18,
                 fontweight="bold", y=0.96)
    labels = ["T1c", "T2", "B0", "ADC", "T2+B0"]
    for i, z in enumerate(zs):
        sm, dm = radiological(s_mask[z]), radiological(d_mask[z])
        rows = [(radiological(t1c[z]), sm, False), (radiological(t2[z]), sm, False),
                (radiological(b0[z]), dm, True), (radiological(adc[z]), dm, False)]
        for r, (s, m, is_diff) in enumerate(rows):
            lo, hi = window(s, m, is_diff)
            axes[r, i].imshow(np.where(m > 0.5, s, lo), cmap="gray", vmin=lo, vmax=hi)
        red, cyan = to_unit(rows[1][0], sm), to_unit(rows[2][0], dm, True)
        axes[4, i].imshow(np.stack([red, cyan, cyan], axis=-1))
        axes[0, i].set_title(f"z={z}", color="white", fontsize=16)
        for r in range(5):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
            if i == 0:
                axes[r, i].set_ylabel(labels[r], color="white", fontsize=14, fontweight="bold")
    plt.subplots_adjust(top=0.92, wspace=0.05, hspace=0.05)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
