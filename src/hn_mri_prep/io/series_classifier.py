"""Map SeriesDescription / ImageType to a sequence type.

Naming varies a lot between vendors and sites (``t1_tse_tra_fs_ce``, ``T1W_TSE_GD``,
``Ax T1 FS +C``, ...), so the rules are simple and every result carries the rule that fired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class SeriesType(str, Enum):
    T1 = "T1"
    T1C = "T1c"
    T2 = "T2"
    ADC = "ADC"
    DIFFUSION = "DIFFUSION_RAW"
    SKIP = "SKIP"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Classification:
    series_type: SeriesType
    rule: str


# vendor-computed maps, not used
_SYNTHETIC = ("eadc", "edwi", "exp", "sdwi")
_CONTRAST_SUBSTRINGS = ("gd", "gad", "post", "+c", "c+", "contrast")
# whole tokens only ("ce" is inside space, sequence, slice, ...)
_CONTRAST_TOKENS = frozenset({"ce", "c"})
_ADC_MARKERS = ("ddwibs", "dadc", "adc")
_DIFFUSION_MARKERS = ("dwi", "b0", "trace", "iso", "diff")


def tokenize(description: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", description.lower()) if t]


def _has_contrast_marker(desc: str, tokens: list[str], legacy: bool) -> str | None:
    if legacy:
        for sub in ("gd", "+c", "post", "gad", "ce"):
            if sub in desc:
                return f"substring '{sub}' (legacy)"
        return None
    for sub in _CONTRAST_SUBSTRINGS:
        if sub in desc:
            return f"substring '{sub}'"
    for tok in tokens:
        if tok in _CONTRAST_TOKENS:
            return f"token '{tok}'"
    return None


def classify_series(
    description: str,
    image_type: list[str] | tuple[str, ...] = (),
    has_bval: bool = False,
    legacy: bool = False,
) -> Classification:
    """Classify one series.

    Args:
        description: DICOM SeriesDescription (or ProtocolName fallback).
        image_type: DICOM ImageType values, as exported by dcm2niix in the JSON sidecar.
        has_bval: whether dcm2niix wrote a ``.bval`` file for the series.
        legacy: reproduce the thesis (v5) substring-based contrast detection.
    """
    desc = (description or "").lower()
    tokens = tokenize(desc)
    image_type_upper = {str(v).upper() for v in image_type}

    for sub in _SYNTHETIC:
        if sub in desc:
            return Classification(SeriesType.SKIP, f"synthetic map '{sub}'")

    if "t2" in desc and "dwi" not in desc:
        if "et2" in desc:
            return Classification(SeriesType.SKIP, "synthetic 'et2'")
        return Classification(SeriesType.T2, "'t2' in description")

    if "t1" in desc:
        if not legacy and "et1" in desc:
            return Classification(SeriesType.SKIP, "synthetic 'et1'")
        marker = _has_contrast_marker(desc, tokens, legacy)
        if marker:
            return Classification(SeriesType.T1C, f"'t1' + contrast {marker}")
        if "et1" in desc:
            return Classification(SeriesType.SKIP, "synthetic 'et1'")
        return Classification(SeriesType.T1, "'t1' without contrast marker")

    for sub in _ADC_MARKERS:
        if sub in desc:
            return Classification(SeriesType.ADC, f"'{sub}' in description")
    if "ADC" in image_type_upper:
        return Classification(SeriesType.ADC, "ImageType contains ADC")

    if has_bval:
        return Classification(SeriesType.DIFFUSION, ".bval present")
    for sub in _DIFFUSION_MARKERS:
        if (legacy and sub == "diff") or sub not in desc:
            continue
        return Classification(SeriesType.DIFFUSION, f"'{sub}' in description")

    return Classification(SeriesType.UNKNOWN, "no rule matched")
