"""Provenance records: what code, which config and which library versions produced an output."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hn_mri_prep import __version__


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_info() -> dict[str, Any]:
    import nibabel
    import numpy
    import scipy
    import SimpleITK

    return {
        "hn_mri_prep": __version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "SimpleITK": SimpleITK.Version_VersionString(),
        "nibabel": nibabel.__version__,
    }


def write_provenance(path: Path, config_fingerprint: str, **record: Any) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_fingerprint": config_fingerprint,
        "environment": environment_info(),
        **record,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
