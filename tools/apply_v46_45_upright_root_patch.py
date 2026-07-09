#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V46.45 Upright Root Patch
=========================
Patch tools/v46_motionrag_diff.py after V46.44.

Purpose
-------
Chang-E BVH root/Hips rotation may contain pitch/roll or dataset coordinate-frame
calibration. Mapping that full root rotation directly into EDGE/SMPL joint-0 can
rotate the entire 24-joint skeleton and produce full-body rolling/flipping.

This patch adds a BVH-loader guard:
  V46_45_BVH_ROOT_ROT_MODE=yaw      (default) keep only root yaw
  V46_45_BVH_ROOT_ROT_MODE=identity force identity root rotation
  V46_45_BVH_ROOT_ROT_MODE=raw      keep original root rotation

The change is applied at BVH loading time, so the Event-RAG DB must be rebuilt
and V44/V45/V46 must be retrained.
"""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "tools" / "v46_motionrag_diff.py"

HELPER = r'''

# ===== V46.45 UPRIGHT BVH ROOT PATCH START =====
def _v46_45_env_bool(name: str, default: bool = False) -> bool:
    try:
        return bool(int(os.environ.get(name, "1" if default else "0")))
    except Exception:
        return str(os.environ.get(name, str(default))).strip().lower() in {"true", "yes", "on"}


def _v46_45_root_yaw_only_matrix_np(root_r: np.ndarray) -> np.ndarray:
    """Project arbitrary root rotations to yaw-only rotations around world Y.

    V46 stores rotations in column-concat 6D.  rot6d_to_matrix_np() returns a
    matrix whose third column is the local forward axis.  We keep the horizontal
    facing direction and remove pitch/roll so BVH coordinate calibration cannot
    roll the whole SMPL body.
    """
    R = np.asarray(root_r, dtype=np.float32)
    if R.ndim == 2:
        R = R[None]
    out = np.tile(np.eye(3, dtype=np.float32), (R.shape[0], 1, 1))
    forward = R[:, :, 2].astype(np.float32)
    yaw = np.arctan2(forward[:, 0], forward[:, 2]).astype(np.float32)
    bad = (~np.isfinite(yaw)) | (np.linalg.norm(forward[:, [0, 2]], axis=1) < 1e-6)
    yaw[bad] = 0.0
    c = np.cos(yaw).astype(np.float32)
    s = np.sin(yaw).astype(np.float32)
    out[:, 0, 0] = c
    out[:, 0, 2] = s
    out[:, 1, 0] = 0.0
    out[:, 1, 1] = 1.0
    out[:, 1, 2] = 0.0
    out[:, 2, 0] = -s
    out[:, 2, 1] = 0.0
    out[:, 2, 2] = c
    return out.astype(np.float32)


def _v46_45_apply_bvh_root_upright_guard(target_local: np.ndarray, source_hint: str = "") -> tuple[np.ndarray, dict]:
    """Remove non-yaw root rotations for BVH-derived EDGE motions.

    mode=yaw      : keep facing direction, remove pitch/roll.  Recommended.
    mode=identity : remove all root rotation.  Useful for ablation/debug.
    mode=raw/off  : no change.
    """
    x = np.asarray(target_local, dtype=np.float32).copy()
    mode = str(os.environ.get("V46_45_BVH_ROOT_ROT_MODE", os.environ.get("V46_BVH_ROOT_ROT_MODE", "yaw"))).strip().lower()
    if mode in {"0", "false", "off", "raw", "none", "disable", "disabled"}:
        return x, {"enabled": False, "mode": mode, "source_hint": str(source_hint)}
    root_before = x[:, 0].copy()
    up = root_before[:, :, 1]
    up_norm = np.maximum(np.linalg.norm(up, axis=1), 1e-8)
    cos_up = np.clip(up[:, 1] / up_norm, -1.0, 1.0)
    tilt_before_deg = np.degrees(np.arccos(cos_up)).astype(np.float32)
    if mode in {"identity", "id"}:
        x[:, 0] = np.eye(3, dtype=np.float32)[None]
    else:
        mode = "yaw"
        x[:, 0] = _v46_45_root_yaw_only_matrix_np(root_before)
    up2 = x[:, 0, :, 1]
    up2_norm = np.maximum(np.linalg.norm(up2, axis=1), 1e-8)
    cos_up2 = np.clip(up2[:, 1] / up2_norm, -1.0, 1.0)
    tilt_after_deg = np.degrees(np.arccos(cos_up2)).astype(np.float32)
    rep = {
        "enabled": True,
        "version": "v46_45_bvh_root_upright_guard",
        "mode": mode,
        "source_hint": str(source_hint),
        "root_tilt_before_deg_p95": float(np.nanpercentile(tilt_before_deg, 95)) if tilt_before_deg.size else 0.0,
        "root_tilt_before_deg_max": float(np.nanmax(tilt_before_deg)) if tilt_before_deg.size else 0.0,
        "root_tilt_after_deg_p95": float(np.nanpercentile(tilt_after_deg, 95)) if tilt_after_deg.size else 0.0,
        "root_tilt_after_deg_max": float(np.nanmax(tilt_after_deg)) if tilt_after_deg.size else 0.0,
    }
    return x.astype(np.float32), rep
# ===== V46.45 UPRIGHT BVH ROOT PATCH END =====
'''


def backup(path: Path) -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = path.with_suffix(path.suffix + f".v46_45_upright_backup_{ts}")
    shutil.copy2(path, dst)
    print(f"[BACKUP] {dst}")


def main() -> int:
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)
    s = TARGET.read_text(encoding="utf-8")
    backup(TARGET)

    if "V46.45 UPRIGHT BVH ROOT PATCH START" not in s:
        anchor = "def _norm_joint_name(name: str) -> str:\n"
        idx = s.find(anchor)
        if idx < 0:
            raise RuntimeError("Cannot find insertion anchor def _norm_joint_name")
        s = s[:idx] + HELPER + "\n" + s[idx:]
        print("[PATCH] inserted V46.45 upright root helper")
    else:
        print("[SKIP] helper already present")

    pattern = (
        "    target_idx = _bvh_target_joint_indices([str(j[\"name\"]) for j in joints])\n"
        "    target_local = np.tile(np.eye(3, dtype=np.float32), (data.shape[0], NUM_JOINTS, 1, 1))\n"
        "    for tgt, src in enumerate(target_idx):\n"
        "        if 0 <= src < len(joints):\n"
        "            target_local[:, tgt] = local_all[:, src]\n\n"
        "    out = np.zeros((data.shape[0], EDGE_DIM), dtype=np.float32)\n"
    )
    replacement = (
        "    target_idx = _bvh_target_joint_indices([str(j[\"name\"]) for j in joints])\n"
        "    target_local = np.tile(np.eye(3, dtype=np.float32), (data.shape[0], NUM_JOINTS, 1, 1))\n"
        "    for tgt, src in enumerate(target_idx):\n"
        "        if 0 <= src < len(joints):\n"
        "            target_local[:, tgt] = local_all[:, src]\n"
        "    target_local, root_upright_report = _v46_45_apply_bvh_root_upright_guard(target_local, source_hint=str(p))\n\n"
        "    out = np.zeros((data.shape[0], EDGE_DIM), dtype=np.float32)\n"
    )
    if pattern in s:
        s = s.replace(pattern, replacement, 1)
        print("[PATCH] load_bvh_file applies root upright guard")
    elif "root_upright_report = _v46_45_apply_bvh_root_upright_guard" in s:
        print("[SKIP] load_bvh_file already patched")
    else:
        raise RuntimeError("Cannot find load_bvh_file target_local block")

    # Store the report in channels 2/3 in a harmless bounded diagnostic way only if desired?
    # We intentionally do not write audit numbers into EDGE channels to avoid contact pollution.

    TARGET.write_text(s, encoding="utf-8")
    print("[DONE] V46.45 upright root patch applied")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
