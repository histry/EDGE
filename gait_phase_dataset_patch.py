"""Runtime dataset patch for gait phase + sparse waypoint trajectory condition.

Env gates:
  EDGE_GAIT_PHASE_COND=1         -> attach cond['gait_phase'] [T,6]
  EDGE_TRAJ_SPARSE_WAYPOINT=1    -> attach cond['trajectory_mask'] [T,1]
  EDGE_TRAJ_BEV_COND=1           -> optional cond['bev_map'] from trajectory

No dataset cache format changes are required.
"""
from __future__ import annotations

import os
from functools import wraps

import numpy as np

_TRUE = {"1", "true", "yes", "y", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in _TRUE


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def install_gait_phase_dataset_patch(verbose: bool = True) -> bool:
    try:
        from dataset import dance_dataset as dd
        from footstep_phase_utils import gait_phase_from_motion
        from trajectory_representation_utils import build_sparse_waypoint_mask, trajectory_to_bev_heatmap
    except Exception as exc:
        if verbose:
            print(f"⚠️ Gait/trajectory dataset patch skipped: {exc}")
        return False

    if getattr(dd, "_edge_gait_phase_dataset_patch_installed", False):
        return True

    def patch_cls(cls_name: str):
        cls = getattr(dd, cls_name, None)
        if cls is None or getattr(cls, "_edge_gait_phase_getitem_patched", False):
            return False
        original_getitem = cls.__getitem__

        @wraps(original_getitem)
        def patched_getitem(self, idx):
            pose, cond, filename, wav = original_getitem(self, idx)
            if not isinstance(cond, dict):
                cond = {"audio": cond}

            if _env_bool("EDGE_GAIT_PHASE_COND", False) and "gait_phase" not in cond:
                cond["gait_phase"] = gait_phase_from_motion(
                    pose,
                    normalizer=getattr(self, "normalizer", None),
                    speed_threshold=_env_float("EDGE_GAIT_PHASE_SPEED_THRESHOLD", 0.01),
                    stride_length=_env_float("EDGE_GAIT_PHASE_STRIDE_LENGTH", 0.35),
                )

            traj = cond.get("trajectory", None)
            if _env_bool("EDGE_TRAJ_SPARSE_WAYPOINT", False) and traj is not None and "trajectory_mask" not in cond:
                T = int(traj.shape[0]) if hasattr(traj, "shape") else int(getattr(self, "seq_len", 150))
                cond["trajectory_mask"] = build_sparse_waypoint_mask(T)

            if _env_bool("EDGE_TRAJ_BEV_COND", False) and traj is not None and "bev_map" not in cond:
                try:
                    cond["bev_map"] = trajectory_to_bev_heatmap(
                        np.asarray(traj, dtype=np.float32),
                        size=_env_int("EDGE_TRAJ_BEV_SIZE", 32),
                        sigma=_env_float("EDGE_TRAJ_BEV_SIGMA", 1.5),
                    )
                except Exception as exc:
                    if _env_bool("EDGE_TRAJ_BEV_STRICT", False):
                        raise
                    if _env_bool("EDGE_TRAJ_BEV_VERBOSE", False):
                        print(f"⚠️ bev_map skipped for {filename}: {exc}")

            return pose, cond, filename, wav

        cls.__getitem__ = patched_getitem
        cls._edge_gait_phase_getitem_patched = True
        return True

    patched = []
    for name in ["DunhuangDataset", "AISTPPDataset"]:
        if patch_cls(name):
            patched.append(name)
    dd._edge_gait_phase_dataset_patch_installed = True
    if verbose:
        print(
            "✅ Installed gait/advanced-trajectory dataset patch: "
            f"gait={_env_bool('EDGE_GAIT_PHASE_COND', False)}, "
            f"sparse={_env_bool('EDGE_TRAJ_SPARSE_WAYPOINT', False)}, "
            f"bev={_env_bool('EDGE_TRAJ_BEV_COND', False)}, classes={patched}"
        )
    return True


def install():
    return install_gait_phase_dataset_patch(verbose=True)
