"""Leakage-safe ADC scaling for cross-validation.

ADC is the one channel kept in physical units after preprocessing, because its absolute
value carries diagnostic information (e.g. pleomorphic adenoma vs Warthin tumour). Mapping
it to [0, 1] needs a range, and a range estimated from the whole cohort would let
validation/test subjects influence the training inputs. The scaler is therefore fitted on
the training subjects of each fold only and applied unchanged to that fold's held-out subjects.

Methods:

* ``"minmax"``: training-fold minimum and maximum (the thesis approach). Simple, but the
  maximum of millions of voxels is set by a single extreme voxel (CSF, cyst, artefact), so
  the scale can change noticeably between folds.
* ``"percentile"``: training-fold percentiles of the pooled tissue voxels (default
  0.5 / 99.5). Same leakage guarantees, much less sensitive to outliers.
* ``"fixed"``: a physiological range fixed before seeing the data (e.g. 0-3000 x 1e-6 mm^2/s).
  Nothing is estimated, so there is nothing to leak. A good sensitivity analysis.

Example::

    folds = [(train_ids, val_ids) for train_ids, val_ids in my_splitter]
    scalers = fit_fold_scalers(adc_by_subject, folds, method="percentile")
    x_val = scalers[k].transform(adc_by_subject[sid], subject_id=sid)
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

Method = Literal["minmax", "percentile", "fixed"]


class DataLeakageError(RuntimeError):
    """Raised when a scaler is applied to an evaluation subject it was fitted on."""


@dataclass
class FoldwiseADCScaler:
    method: Method = "percentile"
    percentiles: tuple[float, float] = (0.5, 99.5)
    fixed_range: tuple[float, float] = (0.0, 3000.0)
    clip: bool = True
    # Exact zeros are masked background after preprocessing, not tissue.
    ignore_zeros: bool = True
    n_bins: int = 65536
    low_: float | None = None
    high_: float | None = None
    fitted_ids_: frozenset[str] = field(default_factory=frozenset)

    def _voxels(self, vol: np.ndarray) -> np.ndarray:
        v = np.asarray(vol, dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        return v[v != 0] if self.ignore_zeros else v

    def fit(self, volumes: Mapping[str, np.ndarray]) -> FoldwiseADCScaler:
        """Fit on ``{subject_id: adc_volume}`` of the TRAINING subjects only."""
        if not volumes:
            raise ValueError("fit() needs at least one training volume")
        self.fitted_ids_ = frozenset(map(str, volumes))
        if self.method == "fixed":
            self.low_, self.high_ = map(float, self.fixed_range)
            return self

        # Pass 1: global range (also the answer for min/max).
        lo, hi = np.inf, -np.inf
        for vol in volumes.values():
            v = self._voxels(vol)
            if v.size:
                lo, hi = min(lo, v.min()), max(hi, v.max())
        if not np.isfinite(lo):
            raise ValueError("training volumes contain no usable voxels")
        if self.method == "minmax" or hi == lo:
            self.low_, self.high_ = float(lo), float(hi)
            return self

        # Pass 2: streaming histogram -> percentiles, error <= (hi - lo) / n_bins.
        edges = np.linspace(lo, hi, self.n_bins + 1)
        counts = np.zeros(self.n_bins, dtype=np.int64)
        for vol in volumes.values():
            counts += np.histogram(self._voxels(vol), bins=edges)[0]
        cdf = np.cumsum(counts) / counts.sum()
        p_lo, p_hi = (np.asarray(self.percentiles) / 100.0).tolist()
        self.low_ = float(edges[np.searchsorted(cdf, p_lo)])
        self.high_ = float(edges[min(np.searchsorted(cdf, p_hi) + 1, self.n_bins)])
        return self

    def check_disjoint(self, subject_ids: Iterable[str]) -> None:
        overlap = self.fitted_ids_ & set(map(str, subject_ids))
        if overlap:
            raise DataLeakageError(f"scaler was fitted on evaluation subjects: {sorted(overlap)[:10]}")

    def transform(self, volume: np.ndarray, subject_id: str | None = None, evaluation: bool = True) -> np.ndarray:
        """Scale one volume to [0, 1].

        Passing ``subject_id`` for a held-out subject (``evaluation=True``) verifies that the
        scaler never saw it. Pass ``evaluation=False`` when transforming training data.
        """
        if self.low_ is None or self.high_ is None:
            raise RuntimeError("scaler is not fitted")
        if subject_id is not None and evaluation:
            self.check_disjoint([subject_id])
        vol = np.asarray(volume, dtype=np.float32)
        out = (vol - self.low_) / max(self.high_ - self.low_, 1e-8)
        if self.clip:
            out = np.clip(out, 0.0, 1.0)
        if self.ignore_zeros:
            out[vol == 0] = 0.0
        return out.astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "method": self.method, "percentiles": list(self.percentiles), "fixed_range": list(self.fixed_range),
            "clip": self.clip, "ignore_zeros": self.ignore_zeros, "n_bins": self.n_bins,
            "low": self.low_, "high": self.high_, "fitted_ids": sorted(self.fitted_ids_),
        }

    @classmethod
    def from_dict(cls, d: dict) -> FoldwiseADCScaler:
        s = cls(method=d["method"], percentiles=tuple(d["percentiles"]), fixed_range=tuple(d["fixed_range"]),
                clip=d["clip"], ignore_zeros=d["ignore_zeros"], n_bins=d["n_bins"])
        s.low_, s.high_, s.fitted_ids_ = d["low"], d["high"], frozenset(d["fitted_ids"])
        return s

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> FoldwiseADCScaler:
        return cls.from_dict(json.loads(Path(path).read_text()))


def assert_group_disjoint(
    train_ids: Iterable[str], eval_ids: Iterable[str], groups: Mapping[str, str] | None = None
) -> None:
    """Raise if a subject, or a patient (``groups`` maps subject -> patient), is on both sides."""
    train, evals = set(map(str, train_ids)), set(map(str, eval_ids))
    if train & evals:
        raise DataLeakageError(f"subjects in train and evaluation: {sorted(train & evals)[:10]}")
    if groups is not None:
        tg = {groups.get(s, s) for s in train}
        eg = {groups.get(s, s) for s in evals}
        if tg & eg:
            raise DataLeakageError(f"patients in train and evaluation: {sorted(tg & eg)[:10]}")


def fit_fold_scalers(
    adc_by_subject: Mapping[str, np.ndarray],
    folds: Sequence[tuple[Iterable[str], Iterable[str]]],
    groups: Mapping[str, str] | None = None,
    **scaler_kwargs,
) -> list[FoldwiseADCScaler]:
    """One scaler per ``(train_ids, eval_ids)`` fold, each fitted on that fold's training subjects."""
    scalers = []
    for train_ids, eval_ids in folds:
        train_ids, eval_ids = list(map(str, train_ids)), list(map(str, eval_ids))
        assert_group_disjoint(train_ids, eval_ids, groups)
        scaler = FoldwiseADCScaler(**scaler_kwargs).fit({s: adc_by_subject[s] for s in train_ids})
        scaler.check_disjoint(eval_ids)
        scalers.append(scaler)
    return scalers
