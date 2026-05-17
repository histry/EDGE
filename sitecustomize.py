"""Auto-install optional EDGE freeze-aware motion patch.

Place this file in the EDGE repository root.  Python automatically imports
sitecustomize during interpreter startup when the current directory is on
sys.path.  The patch only installs when EDGE_FREEZE_AWARE_MOTION=1, so copying
this file is safe for old experiments.
"""

from __future__ import annotations

import os

_TRUE = {"1", "true", "yes", "y", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in _TRUE


if _env_bool("EDGE_FREEZE_AWARE_MOTION", False) and _env_bool("EDGE_FREEZE_AWARE_AUTO_INSTALL", True):
    try:
        from freeze_aware_motion_patch import install_freeze_aware_motion_patch

        install_freeze_aware_motion_patch(
            verbose=_env_bool("EDGE_FREEZE_AWARE_INSTALL_VERBOSE", True)
        )
    except Exception as exc:
        print(f"⚠️ freeze-aware motion auto-install failed: {exc}")
