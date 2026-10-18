"""Expert QC sheet: native slices with the 64 mm extraction box drawn in radiological view."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from hn_mri_prep.roi.extraction import voxel_bounds


def radiological_slice(vol: np.ndarray, z: int) -> np.ndarray:
    return np.flip(np.rot90(vol[:, :, z], k=1), axis=1)


def draw_box(ax, bi: tuple[int, int], bj: tuple[int, int], shape: tuple[int, int], color: str) -> None:
    """Rectangle for RAS index bounds on a radiological display of a slice of ``shape`` (I, J)."""
    (si, ei), (sj, ej) = bi, bj
    w_i, w_j = shape
    ax.add_patch(patches.Rectangle((w_i - ei, w_j - ej), ei - si, ej - sj,
                                   linewidth=2, edgecolor=color, facecolor="none"))


def save_roi_qc(subject, t1c, dwi, b0, center, spacing, windows, path: Path, box_mm: float = 64.0, n_cols: int = 5):
    rows = [("T1c", t1c, "struct", "red", {}), ("DWI (high b)", dwi, "diff", "lime", {"vmin": -1.0, "vmax": 3.5}),
            ("B0 (low b)", b0, "diff", "cyan", {})]
    fig, axes = plt.subplots(3, n_cols, figsize=(22, 16), squeeze=False)
    fig.patch.set_facecolor("black")
    fig.suptitle(f"ROI QC | {subject} | {box_mm:g} mm box", fontsize=16, fontweight="bold", color="white")
    for r, (title, vol, kind, color, kw) in enumerate(rows):
        s, e = windows[kind]
        zs = np.linspace(s, e - 1, n_cols).astype(int)
        bi = voxel_bounds(center[kind][0], box_mm, spacing[kind][0])
        bj = voxel_bounds(center[kind][1], box_mm, spacing[kind][1])
        for c, z in enumerate(zs):
            ax = axes[r, c]
            ax.axis("off")
            if not 0 <= z < vol.shape[2]:
                ax.set_title(f"{title} z={z} (padding)", color="gray")
                continue
            slc = radiological_slice(vol, z)
            ax.imshow(np.ma.masked_where(np.abs(slc) <= 1e-5, slc), cmap="gray", **kw)
            draw_box(ax, bi, bj, vol.shape[:2], color)
            ax.set_title(f"{title} z={z}", color="white")
    plt.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    plt.close(fig)
