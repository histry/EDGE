#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V46.47 Chang-E contract patch for histry/EDGE.

Purpose
-------
Patch tools/v46_motionrag_diff.py in-place so the current GitHub version gains:
  1) V46_45/V46_47 root upright control in load_bvh_file();
  2) yaw-only / identity / raw root rotation modes for Chang-E BVH;
  3) lightweight metadata in DB events.npz for direct source-disjoint audit;
  4) no change to existing V46.38/V46.41/V46.46 routing/closed-loop modules.

This script is intentionally conservative: it does not rewrite the whole long
v46_motionrag_diff.py. It inserts a small helper and one guarded call before
BVH target joint mapping. Re-running this script is idempotent.

Recommended main experiment:
  export V46_45_BVH_ROOT_ROT_MODE=yaw
  python tools/apply_v46_47_chang_e_contract_patch.py
  python -m py_compile tools/v46_motionrag_diff.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

HELPER_MARK = "# ===== V46.47 CHANG-E UPRIGHT ROOT HELPERS START ====="
CALL_MARK = "# ===== V46.47 CHANG-E UPRIGHT ROOT CALL START ====="
META_MARK = "# ===== V46.47 SOURCE-DISJOINT DB META START ====="

HELPER_CODE = r'''
# ===== V46.47 CHANG-E UPRIGHT ROOT HELPERS START =====
def _v46_47_env_root_rot_mode() -> str:
    """Return BVH root-rotation mode.

    Modes:
      yaw      : keep only global facing/yaw; remove pitch/roll from Hips root.
      identity : remove all Hips root rotation; use local child rotations only.
      raw      : keep original full BVH Hips rotation.

    V46_45_BVH_ROOT_ROT_MODE is kept for compatibility with previous notes.
    V46_47_BVH_ROOT_ROT_MODE is the new name used by this patch.
    """
    try:
        mode = os.environ.get("V46_47_BVH_ROOT_ROT_MODE", os.environ.get("V46_45_BVH_ROOT_ROT_MODE", "yaw"))
    except Exception:
        mode = "yaw"
    mode = str(mode).strip().lower()
    if mode in {"keep", "full", "original"}:
        mode = "raw"
    if mode not in {"yaw", "identity", "raw"}:
        print(f"[V46.47 WARN] unknown root rot mode={mode!r}; using yaw", file=sys.stderr)
        mode = "yaw"
    return mode


def _v46_47_yaw_matrix_from_root_np(root_r: np.ndarray) -> np.ndarray:
    """Project root rotation matrices to yaw-only Y-axis rotations.

    The forward vector is taken from the third rotation column, matching
    root_yaw_np().  The returned matrix preserves facing direction but removes
    pitch/roll, preventing Chang-E Hips full rotation from flipping the entire
    EDGE/SMPL-like skeleton.
    """
    R = np.asarray(root_r, dtype=np.float32)
    if R.ndim == 2:
        R = R[None]
    forward = R[:, :, 2]
    yaw = np.arctan2(forward[:, 0], forward[:, 2]).astype(np.float32)
    c = np.cos(yaw).astype(np.float32)
    s = np.sin(yaw).astype(np.float32)
    out = np.zeros_like(R, dtype=np.float32)
    out[:, 0, 0] = c
    out[:, 0, 2] = s
    out[:, 1, 1] = 1.0
    out[:, 2, 0] = -s
    out[:, 2, 2] = c
    return out.astype(np.float32)


def _v46_47_apply_upright_root_np(local_all: np.ndarray, source_hint: str = "") -> tuple[np.ndarray, dict]:
    """Apply Chang-E Hips root upright guard to local rotation stack.

    This edits only local_all[:, 0], i.e. the root/Hips local rotation before
    mapping to EDGE 24-joint rot6d. Child joint rotations are left untouched.
    """
    arr = np.asarray(local_all, dtype=np.float32).copy()
    report = {"enabled": False, "mode": "raw", "source_hint": str(source_hint)}
    if arr.ndim != 4 or arr.shape[0] == 0 or arr.shape[1] == 0:
        report["reason"] = "bad_shape_or_empty"
        return arr, report
    mode = _v46_47_env_root_rot_mode()
    report.update({"enabled": mode != "raw", "mode": mode})
    root_before = arr[:, 0].copy()
    try:
        fwd = root_before[:, :, 2]
        up = root_before[:, :, 1]
        tilt = np.arccos(np.clip(np.abs(up[:, 1]), 0.0, 1.0))
        report.update({
            "tilt_before_p95_rad": float(np.percentile(tilt, 95)),
            "tilt_before_max_rad": float(np.max(tilt)),
            "yaw_before_range_rad": float(np.ptp(np.arctan2(fwd[:, 0], fwd[:, 2]))),
        })
    except Exception as exc:
        report["pre_audit_error"] = str(exc)
    if mode == "identity":
        arr[:, 0] = np.eye(3, dtype=np.float32)[None]
    elif mode == "yaw":
        arr[:, 0] = _v46_47_yaw_matrix_from_root_np(arr[:, 0])
    # raw keeps original
    try:
        up2 = arr[:, 0, :, 1]
        tilt2 = np.arccos(np.clip(np.abs(up2[:, 1]), 0.0, 1.0))
        report.update({
            "tilt_after_p95_rad": float(np.percentile(tilt2, 95)),
            "tilt_after_max_rad": float(np.max(tilt2)),
        })
    except Exception as exc:
        report["post_audit_error"] = str(exc)
    return arr.astype(np.float32), report
# ===== V46.47 CHANG-E UPRIGHT ROOT HELPERS END =====
'''

CALL_CODE = r'''    # ===== V46.47 CHANG-E UPRIGHT ROOT CALL START =====
    # The current GitHub loader maps BVH Hips full rotation directly into EDGE
    # root.  Chang-E Hips often contains pitch/roll; keeping it can flip or roll
    # the whole SMPL-like body.  This guarded projection keeps yaw/facing only by
    # default while preserving raw mode for ablation.
    local_all, root_upright_report = _v46_47_apply_upright_root_np(local_all, source_hint=str(p))
    # ===== V46.47 CHANG-E UPRIGHT ROOT CALL END =====
'''

META_CODE = r'''        # ===== V46.47 SOURCE-DISJOINT DB META START =====
        source_files=np.array([m.get("source_file", "") for m in meta], dtype=object),
        starts=np.array([int(m.get("start", 0)) for m in meta], dtype=np.int32),
        ends=np.array([int(m.get("end", 0)) for m in meta], dtype=np.int32),
        fragment_indices=np.array([int(m.get("fragment_index", 0) or 0) for m in meta], dtype=np.int32),
        event_starts=np.array([int(m.get("event_start", m.get("start", 0)) or 0) for m in meta], dtype=np.int32),
        event_ends=np.array([int(m.get("event_end", m.get("end", 0)) or 0) for m in meta], dtype=np.int32),
        event_source_frames=np.array([int(m.get("event_source_frames", 0) or 0) for m in meta], dtype=np.int32),
        input_modes=np.array([m.get("input_mode", "") for m in meta], dtype=object),
        # ===== V46.47 SOURCE-DISJOINT DB META END =====
'''


def insert_after(text: str, needle: str, insertion: str, marker: str) -> str:
    if marker in text:
        return text
    pos = text.find(needle)
    if pos < 0:
        raise RuntimeError(f"needle not found: {needle[:80]!r}")
    pos_end = pos + len(needle)
    return text[:pos_end] + "\n\n" + insertion.strip("\n") + "\n" + text[pos_end:]


def insert_before(text: str, needle: str, insertion: str, marker: str) -> str:
    if marker in text:
        return text
    pos = text.find(needle)
    if pos < 0:
        raise RuntimeError(f"needle not found: {needle[:80]!r}")
    return text[:pos] + insertion + "\n" + text[pos:]


def patch_file(path: Path, backup: bool = True) -> None:
    text = path.read_text(encoding="utf-8")
    original = text

    # Insert helper after _bvh_euler_to_matrix function.  This location is early
    # enough that load_bvh_file can call the helper.
    helper_anchor = "def _norm_joint_name(name: str) -> str:"
    text = insert_before(text, helper_anchor, HELPER_CODE, HELPER_MARK)

    # Insert upright call immediately before target joint mapping in load_bvh_file.
    call_anchor = "    target_idx = _bvh_target_joint_indices([str(j[\"name\"]) for j in joints])"
    text = insert_before(text, call_anchor, CALL_CODE, CALL_MARK)

    # Add source-disjoint metadata arrays to events.npz.  This does not break old
    # readers because extra npz arrays are ignored by existing load_db users.
    meta_anchor = "        durations=np.array([m[\"duration\"] for m in meta], dtype=np.float32),"
    if META_MARK not in text and meta_anchor in text:
        text = insert_before(text, meta_anchor, META_CODE, META_MARK)

    if text == original:
        print(f"[V46.47] no changes needed: {path}")
        return
    if backup:
        bak = path.with_suffix(path.suffix + ".v46_47_bak")
        if not bak.exists():
            bak.write_text(original, encoding="utf-8")
            print(f"[V46.47] backup written: {bak}")
    path.write_text(text, encoding="utf-8")
    print(f"[V46.47] patched: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="tools/v46_motionrag_diff.py")
    ap.add_argument("--no_backup", action="store_true")
    args = ap.parse_args()
    patch_file(Path(args.file), backup=not args.no_backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
