"""Runtime dataset patch: attach cond['gait_phase'] for EDGE training.

Enable with:
  EDGE_GAIT_PHASE_COND=1

It patches DunhuangDataset/AISTPPDataset __getitem__ after they are imported but
before DataLoader construction. No dataset cache format changes are required.
"""
from __future__ import annotations

import os
from functools import wraps

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


def install_gait_phase_dataset_patch(verbose: bool = True) -> bool:
    try:
        from dataset import dance_dataset as dd
        from footstep_phase_utils import gait_phase_from_motion
    except Exception as exc:
        if verbose:
            print(f"⚠️ Gait phase dataset patch skipped: {exc}")
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
            if _env_bool("EDGE_GAIT_PHASE_COND", False):
                if not isinstance(cond, dict):
                    cond = {"audio": cond}
                if "gait_phase" not in cond:
                    cond["gait_phase"] = gait_phase_from_motion(
                        pose,
                        normalizer=getattr(self, "normalizer", None),
                        speed_threshold=_env_float("EDGE_GAIT_PHASE_SPEED_THRESHOLD", 0.01),
                        stride_length=_env_float("EDGE_GAIT_PHASE_STRIDE_LENGTH", 0.35),
                    )
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
        print(f"✅ Installed gait phase dataset patch: enabled={_env_bool('EDGE_GAIT_PHASE_COND', False)}, classes={patched}")
    return True


def install():
    return install_gait_phase_dataset_patch(verbose=True)
