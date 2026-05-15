from __future__ import annotations

import os
import torch
from torch import nn

_TRUE = {"1", "true", "yes", "y", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in _TRUE


class GaitPhaseTrajectoryProjection(nn.Module):
    """
    Compatibility placeholder.

    The old experimental wrapper changed trajectory_projection structure even when
    gait was disabled. That caused checkpoint mismatch:
      train/replay model != generate model.

    This class is kept only so old imports do not fail. The installer below will
    not wrap anything unless gait is explicitly enabled.
    """

    def __init__(self, base=None, *args, **kwargs):
        super().__init__()
        self.base = base if base is not None else nn.Identity()

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)


def install_gait_phase_trajectory_patch(*args, **kwargs) -> bool:
    """
    Safe installer.

    Default behavior: no-op.
    Enabled only when EDGE_GAIT_PHASE_COND=1 and EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER!=1.

    For the single-unit reconstruction experiments, keep:
      EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER=1
      EDGE_GAIT_PHASE_COND=0
    """
    if _env_bool("EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER", True):
        print("✅ Gait trajectory wrapper disabled by EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER=1")
        return True

    if not _env_bool("EDGE_GAIT_PHASE_COND", False):
        print("✅ Gait trajectory wrapper skipped: EDGE_GAIT_PHASE_COND=0")
        return True

    print(
        "⚠️ Gait trajectory wrapper requested, but this safe replacement does not "
        "modify trajectory_projection. Restore the original patch only for gait experiments."
    )
    return True


def install_gait_phase_patch(*args, **kwargs) -> bool:
    return install_gait_phase_trajectory_patch(*args, **kwargs)


def install(*args, **kwargs) -> bool:
    return install_gait_phase_trajectory_patch(*args, **kwargs)
