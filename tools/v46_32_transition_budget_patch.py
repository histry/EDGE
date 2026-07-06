#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V46.32 transition-budgeted inbetweening patch for tools/v46_motionrag_diff.py.

Purpose
-------
Patch the current V46.31/V46.x MotionRAG-Diff pipeline so that whole-song
composition uses:
  1) music-slot duration allocation;
  2) light resampling only on the retrieved core motion;
  3) explicit transition budget at slot boundaries;
  4) kinematic inbetweening in EDGE-151D motion space;
  5) seam/transition masks for V45 refiner and V46 diffusion;
  6) final V43 lower-body IK unchanged.

This file is intentionally a source patcher rather than a monkey patch at
runtime: it rewrites the local tools/v46_motionrag_diff.py once, keeps a backup,
and is safe to run repeatedly.  It does not depend on README assumptions.
"""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import time

TARGET = Path("tools/v46_motionrag_diff.py")
MARK = "# ===== V46.32 TRANSITION-BUDGET PATCH START ====="
END_MARK = "# ===== V46.32 TRANSITION-BUDGET PATCH END ====="


def find_func_span(text: str, name: str):
    m = re.search(rf"^def\s+{re.escape(name)}\s*\(", text, flags=re.M)
    if not m:
        return None
    start = m.start()
    m2 = re.search(r"^(def\s+|class\s+)", text[m.end():], flags=re.M)
    end = m.end() + m2.start() if m2 else len(text)
    return start, end


def backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + f".v46_32_transition_budget_{time.strftime('%Y%m%d_%H%M%S')}.bak")
    shutil.copy2(path, bak)
    print("[BAK]", bak)
    return bak


PATCH_BLOCK = r'''
# ===== V46.32 TRANSITION-BUDGET PATCH START =====
def _v46_env_bool(name: str, default: bool = False) -> bool:
    try:
        return bool(int(os.environ.get(name, "1" if default else "0")))
    except Exception:
        return bool(default)


def _v46_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _v46_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def quat_slerp_np(q0: np.ndarray, q1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Vectorized shortest-path quaternion SLERP / nlerp fallback.

    q0, q1: [...,4], w: broadcastable [...,1]. Returns normalized [...,4].
    This function is intentionally NumPy-only so it is usable during concat.
    """
    q0 = normalize_quat_np(np.asarray(q0, dtype=np.float32))
    q1 = normalize_quat_np(np.asarray(q1, dtype=np.float32))
    w = np.asarray(w, dtype=np.float32)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    # Use normalized linear interpolation near zero angle; true SLERP elsewhere.
    near = dot > 0.9995
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin0 = np.sin(theta0)
    s0 = np.sin((1.0 - w) * theta0) / np.maximum(sin0, 1e-8)
    s1 = np.sin(w * theta0) / np.maximum(sin0, 1e-8)
    qs = s0 * q0 + s1 * q1
    ql = (1.0 - w) * q0 + w * q1
    out = np.where(near, ql, qs)
    return normalize_quat_np(out).astype(np.float32)


def _v46_root_velocity(m: np.ndarray, at_end: bool) -> np.ndarray:
    if m.shape[0] < 2:
        return np.zeros(3, dtype=np.float32)
    if at_end:
        return (m[-1, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] - m[-2, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]).astype(np.float32)
    return (m[1, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] - m[0, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]).astype(np.float32)


def transition_len_for_boundary(prev: np.ndarray, curr: np.ndarray, target_len: int, cfg: V46Config) -> int:
    """Choose a boundary transition budget for the *incoming* slot.

    The length is risk-aware but capped by the current music slot so that the
    retrieved core motion remains dominant.  This implements the paper position:
    real events preserve cultural vocabulary, local generation repairs only seams.
    """
    min_t = _v46_env_int("V46_TRANSITION_MIN_FRAMES", 8)
    max_t = _v46_env_int("V46_TRANSITION_MAX_FRAMES", 24)
    ratio = _v46_env_float("V46_TRANSITION_RATIO", 0.18)
    min_core = _v46_env_int("V46_TRANSITION_MIN_CORE_FRAMES", max(18, int(getattr(cfg, "min_event_frames", 36) * 0.55)))
    base = int(round(float(target_len) * float(ratio)))

    try:
        exit_j = fk_24_np(prev[-min(len(prev), 3):])[-1]
        entry_j = fk_24_np(curr[:min(len(curr), 3)])[0]
        pose_gap = float(np.linalg.norm(exit_j - entry_j, axis=-1).mean())
    except Exception:
        pose_gap = 0.0
    try:
        yaw_gap = abs(float(np.arctan2(np.sin(root_yaw_np(prev[-1:])[0] - root_yaw_np(curr[:1])[0]),
                                      np.cos(root_yaw_np(prev[-1:])[0] - root_yaw_np(curr[:1])[0]))))
    except Exception:
        yaw_gap = 0.0
    # Small risk schedule: larger pose/yaw gap gets a longer bridge.
    risk_extra = int(round(np.clip(pose_gap * 16.0 + yaw_gap * 4.0, 0.0, 10.0)))
    L = int(np.clip(base + risk_extra, min_t, max_t))
    L = min(L, max(0, int(target_len) - int(min_core)))
    return int(max(0, L))


def motion_inbetween_np(left_ctx: np.ndarray, right_ctx: np.ndarray, length: int, cfg: V46Config,
                        source_hint: str = "v46_32_inbetween") -> np.ndarray:
    """Generate a kinematic transition in EDGE-151D space.

    The bridge interpolates root trajectory with cubic Hermite and rotations with
    quaternion shortest-path interpolation.  Contact channels are rebuilt by FK in
    enforce_edge151_contract_np, so no invalid gray contacts are preserved.
    """
    L = int(length)
    if L <= 0:
        return np.zeros((0, EDGE_DIM), dtype=np.float32)
    a = np.asarray(left_ctx[-1], dtype=np.float32).copy()
    b = np.asarray(right_ctx[0], dtype=np.float32).copy()
    out = np.repeat(a[None, :], L, axis=0).astype(np.float32)

    # Phase excludes exact endpoints to avoid duplicating previous last or next first.
    u = (np.arange(1, L + 1, dtype=np.float32) / float(L + 1))[:, None]
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2

    p0 = a[[ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].astype(np.float32)
    p1 = b[[ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].astype(np.float32)
    v0 = _v46_root_velocity(left_ctx, at_end=True)
    v1 = _v46_root_velocity(right_ctx, at_end=False)
    # Limit velocity to prevent a transition budget from launching the body.
    vmax = _v46_env_float("V46_TRANSITION_ROOT_VEL_CLAMP_MPF", 0.055)
    v0 = np.clip(v0, -vmax, vmax)
    v1 = np.clip(v1, -vmax, vmax)
    root = h00 * p0[None] + h10 * (L * v0[None]) + h01 * p1[None] + h11 * (L * v1[None])
    out[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = root.astype(np.float32)

    # Rotation SLERP for all joints.
    Ra = rot6d_to_matrix_np(a[ROT6D_START:ROT6D_END].reshape(1, NUM_JOINTS, 6))[0]
    Rb = rot6d_to_matrix_np(b[ROT6D_START:ROT6D_END].reshape(1, NUM_JOINTS, 6))[0]
    qa = matrix_to_quat_np(Ra)[None, :, :]
    qb = matrix_to_quat_np(Rb)[None, :, :]
    q = quat_slerp_np(np.repeat(qa, L, axis=0), np.repeat(qb, L, axis=0), u[:, None, :])
    R = quat_to_matrix_np(q)
    out[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(R).reshape(L, -1)

    # Contacts are not linearly interpolated. Rebuild from FK/contact thresholds.
    out[:, 0:4] = 0.0
    out, _ = enforce_edge151_contract_np(out, cfg, source_hint=source_hint, derive_contact=True, project_rot=True)
    return out.astype(np.float32)


def align_event_core_to_prev_np(prev: np.ndarray, curr: np.ndarray, cfg: V46Config) -> Tuple[np.ndarray, dict]:
    """Yaw + XZ align the incoming event core to the previous exit."""
    out = np.asarray(curr, dtype=np.float32).copy()
    rep: Dict[str, object] = {"mode": "none"}
    if prev.shape[0] == 0 or out.shape[0] == 0:
        return out, rep
    try:
        yaw_ref = float(root_yaw_np(prev[-1:])[0])
        yaw_m = float(root_yaw_np(out[:1])[0])
        dyaw = float(np.arctan2(np.sin(yaw_ref - yaw_m), np.cos(yaw_ref - yaw_m)))
    except Exception:
        yaw_ref, yaw_m, dyaw = 0.0, 0.0, 0.0
    out = rotate_motion_around_y_np(out, dyaw, pivot_xz=out[0, [ROOT_X_IDX, ROOT_Z_IDX]])
    delta_xz = prev[-1, [ROOT_X_IDX, ROOT_Z_IDX]] - out[0, [ROOT_X_IDX, ROOT_Z_IDX]]
    out[:, ROOT_X_IDX] += float(delta_xz[0])
    out[:, ROOT_Z_IDX] += float(delta_xz[1])
    out, contract = enforce_edge151_contract_np(out, cfg, source_hint="v46_32_align_event_core_to_prev", derive_contact=True, project_rot=True)
    rep = {"mode": "yaw_xz_entry_to_prev_exit", "yaw_ref": yaw_ref, "yaw_incoming_before": yaw_m,
           "dyaw_applied": dyaw, "delta_xz_applied": [float(delta_xz[0]), float(delta_xz[1])],
           "root_y_ramp_applied": False, "contract": contract}
    return out.astype(np.float32), rep


def concat_events(event_paths: Sequence[str], target_durations: Sequence[float], cfg: V46Config) -> Tuple[np.ndarray, List[dict]]:
    """V46.32 transition-budgeted concatenation.

    When V46_TRANSITION_BUDGET_ENABLE=1, every music slot keeps its exact frame
    budget.  For slot i>0, the first frames are a learned/refinable transition
    budget bridging previous exit and current entry; the remaining frames are the
    retrieved core motion after light resampling.  If disabled, fall back to the
    preserved V46.31 overlap implementation.
    """
    if not _v46_env_bool("V46_TRANSITION_BUDGET_ENABLE", True):
        return concat_events_v46_31_overlap(event_paths, target_durations, cfg)

    pieces: List[np.ndarray] = []
    rep: List[dict] = []
    cursor = 0
    target_lens = [max(int(getattr(cfg, "min_event_frames", 36)), int(round(float(d) * float(cfg.fps)))) for d in target_durations]
    core_warp_min = _v46_env_float("V46_CORE_WARP_MIN", 0.70)
    core_warp_max = _v46_env_float("V46_CORE_WARP_MAX", 1.45)
    min_core = _v46_env_int("V46_TRANSITION_MIN_CORE_FRAMES", max(18, int(getattr(cfg, "min_event_frames", 36) * 0.55)))

    prev_tail: Optional[np.ndarray] = None
    for i, (p, dur) in enumerate(zip(event_paths, target_durations)):
        target_len = int(target_lens[i])
        m_raw = np.load(str(p)).astype(np.float32)
        m, pre_report = enforce_edge151_contract_np(m_raw, cfg, source_hint=f"v46_32_concat_load:{p}", derive_contact=True, project_rot=True)

        # Incoming transition budget is charged to the current music slot, so
        # the global music boundary remains locked and total frames are exact.
        transition_len = 0
        align_report = None
        transition_report = {"enabled": False, "frames": 0}
        if prev_tail is not None:
            # First make a provisional core to estimate boundary risk.
            provisional_core_len = max(min_core, target_len - _v46_env_int("V46_TRANSITION_MIN_FRAMES", 8))
            provisional_core = resample_motion_np(m, provisional_core_len).astype(np.float32)
            provisional_core, _ = enforce_edge151_contract_np(provisional_core, cfg, source_hint=f"v46_32_provisional_core:{p}", derive_contact=True, project_rot=True)
            provisional_core, _ = align_event_core_to_prev_np(prev_tail, provisional_core, cfg)
            transition_len = transition_len_for_boundary(prev_tail, provisional_core, target_len, cfg)
        core_len = int(max(min_core, target_len - transition_len))
        if core_len + transition_len != target_len:
            # Preserve exact slot budget after respecting min_core.
            transition_len = int(max(0, target_len - core_len))

        src_len = max(1, int(m.shape[0]))
        warp = float(core_len / src_len)
        # Clamp only the core warp; if extreme, still keep target by transition
        # budget but report the violation.  This avoids hard failures in low-resource data.
        warp_clamped = False
        if bool(_v46_env_bool("V46_CORE_WARP_CLAMP_ENABLE", True)) and src_len > 1:
            desired = int(round(np.clip(core_len / src_len, core_warp_min, core_warp_max) * src_len))
            if desired != core_len and desired >= min_core and desired <= target_len:
                core_len = int(desired)
                transition_len = int(max(0, target_len - core_len))
                warp = float(core_len / src_len)
                warp_clamped = True

        core = resample_motion_np(m, core_len).astype(np.float32)
        core, core_report = enforce_edge151_contract_np(core, cfg, source_hint=f"v46_32_core_resample:{p}", derive_contact=True, project_rot=True)

        span_start = cursor
        bridge = np.zeros((0, EDGE_DIM), dtype=np.float32)
        if prev_tail is not None:
            core, align_report = align_event_core_to_prev_np(prev_tail, core, cfg)
            if transition_len > 0 and _v46_env_bool("V46_TRANSITION_INBETWEEN_ENABLE", True):
                bridge = motion_inbetween_np(prev_tail, core, transition_len, cfg, source_hint=f"v46_32_transition:{Path(str(p)).name}")
                transition_report = {"enabled": True, "mode": "kinematic_root_hermite_rot_slerp_contact_rebuild",
                                     "frames": int(transition_len), "span": [int(cursor), int(cursor + transition_len)],
                                     "mask_value": 1.0}
            elif transition_len > 0:
                bridge = resample_motion_np(np.stack([prev_tail[-1], core[0]], axis=0), transition_len).astype(np.float32)
                bridge, _ = enforce_edge151_contract_np(bridge, cfg, source_hint=f"v46_32_linear_transition_fallback:{p}", derive_contact=True, project_rot=True)
                transition_report = {"enabled": True, "mode": "linear_fallback_contract_projected", "frames": int(transition_len),
                                     "span": [int(cursor), int(cursor + transition_len)], "mask_value": 1.0}

        piece = np.concatenate([bridge, core], axis=0).astype(np.float32) if bridge.shape[0] else core.astype(np.float32)
        # Exact per-slot terminal guard.
        if piece.shape[0] != target_len:
            if piece.shape[0] < target_len:
                pad = np.repeat(piece[-1:, :], target_len - piece.shape[0], axis=0).astype(np.float32)
                piece = np.concatenate([piece, pad], axis=0)
                timing_fix = "slot_terminal_hold_pad"
            else:
                piece = piece[:target_len]
                timing_fix = "slot_terminal_trim"
        else:
            timing_fix = "none"
        piece, piece_report = enforce_edge151_contract_np(piece, cfg, source_hint=f"v46_32_piece_final:{p}", derive_contact=True, project_rot=True)
        pieces.append(piece.astype(np.float32))
        span_end = cursor + int(piece.shape[0])
        prev_tail = piece[-min(len(piece), max(3, int(transition_len) + 3)):].copy()

        rep.append({
            "path": str(p),
            "target_frames": int(target_len),
            "source_frames": int(m_raw.shape[0]),
            "core_frames": int(core_len),
            "transition_in_frames": int(transition_len),
            "transition_span": transition_report.get("span", []),
            "slot_span": [int(span_start), int(span_end)],
            "warp": float(warp),
            "core_warp_clamped": bool(warp_clamped),
            "boundary_blend_mode": "transition_budgeted_inbetweening" if transition_len > 0 else "none",
            "transition_budget_mode": "v46_32_slot_budget_core_resample_boundary_inbetween",
            "contract_pre": pre_report,
            "contract_core": core_report,
            "contract_after_align": align_report,
            "contract_piece": piece_report,
            "transition_report": transition_report,
            "slot_timing_fix": timing_fix,
        })
        cursor = span_end

    final = np.concatenate(pieces, axis=0).astype(np.float32) if pieces else np.zeros((0, EDGE_DIM), dtype=np.float32)
    total_target_frames = int(sum(target_lens))
    timing_report = {
        "target_total_frames": int(total_target_frames),
        "frames_before_terminal_guard": int(final.shape[0]),
        "timing_frame_delta_before_terminal_guard": int(total_target_frames - final.shape[0]),
        "timing_compensation_applied": False,
        "timing_compensation_mode": "v46_32_slot_local_transition_budget_exact",
        "global_resample_applied": False,
    }
    if total_target_frames > 0 and final.shape[0] != total_target_frames:
        delta = int(total_target_frames - final.shape[0])
        if delta > 0:
            final = np.concatenate([final, np.repeat(final[-1:, :], delta, axis=0)], axis=0).astype(np.float32)
            mode = "terminal_hold_last_frame_pad_no_global_resample"
        else:
            final = final[:total_target_frames].astype(np.float32)
            mode = "terminal_trim_no_global_resample"
        timing_report.update({"timing_compensation_applied": True, "timing_compensation_mode": mode, "terminal_delta_frames": int(delta)})
    timing_report["frames_after_terminal_guard"] = int(final.shape[0])
    final, final_report = enforce_edge151_contract_np(final, cfg, source_hint="v46_32_concat_final", derive_contact=True, project_rot=True)
    if rep:
        rep[-1]["concat_timing_compensation"] = timing_report
        rep[-1]["concat_final_contract"] = final_report
    return final.astype(np.float32), rep


def transition_mask_from_concat_report(T: int, concat_report: Sequence[dict], default_width: int = 24) -> Tuple[np.ndarray, List[int], List[dict]]:
    """Build refiner/diffusion mask from explicit V46.32 transition spans.

    Falls back to seam-center halos if a report comes from the old overlap path.
    """
    mask = np.zeros((int(T), 1), dtype=np.float32)
    centers: List[int] = []
    spans: List[dict] = []
    halo = _v46_env_int("V46_TRANSITION_MASK_HALO", 4)
    for r in concat_report:
        sp = r.get("transition_span", []) if isinstance(r, dict) else []
        if isinstance(sp, (list, tuple)) and len(sp) == 2:
            a = max(0, int(sp[0]) - halo)
            b = min(int(T), int(sp[1]) + halo)
            if b > a:
                mask[a:b, 0] = 1.0
                centers.append((a + b) // 2)
                spans.append({"start": int(a), "end": int(b), "source": "transition_span"})
    if not centers:
        acc = 0
        for r in list(concat_report)[:-1]:
            tf = int(r.get("target_frames", 0)) if isinstance(r, dict) else 0
            acc += tf
            centers.append(acc)
        mask = make_boundary_mask(int(T), centers, width=default_width)
        spans = [{"center": int(c), "start": int(max(0, c - default_width)), "end": int(min(T, c + default_width)), "source": "fallback_seam_center"} for c in centers]
    return mask.astype(np.float32), centers, spans
# ===== V46.32 TRANSITION-BUDGET PATCH END =====
'''


def main() -> int:
    if not TARGET.exists():
        raise SystemExit(f"[ERROR] missing {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    backup(TARGET)

    # Remove a previous V46.32 block if present.
    if MARK in text and END_MARK in text:
        text = re.sub(re.escape(MARK) + r".*?" + re.escape(END_MARK) + r"\n*", "", text, flags=re.S)

    # Preserve original concat once as fallback.
    if "def concat_events_v46_31_overlap(" not in text:
        span = find_func_span(text, "concat_events")
        if span is None:
            raise SystemExit("[ERROR] Cannot find def concat_events in tools/v46_motionrag_diff.py")
        original = text[span[0]:span[1]]
        original_renamed = original.replace("def concat_events(", "def concat_events_v46_31_overlap(", 1)
        text = text[:span[0]] + original_renamed + "\n\n" + PATCH_BLOCK + "\n" + text[span[1]:]
    else:
        # Insert new concat block before make_boundary_mask if fallback already exists.
        span_make = find_func_span(text, "make_boundary_mask")
        if span_make is None:
            raise SystemExit("[ERROR] Cannot find def make_boundary_mask in tools/v46_motionrag_diff.py")
        text = text[:span_make[0]] + PATCH_BLOCK + "\n" + text[span_make[0]:]

    # Replace generate seam-mask construction so V45/V46 target explicit transition spans.
    old = '''    motion, concat_report = concat_events(selected_paths, [s["duration"] for s in slots], cfg)
    seam_positions = []
    acc = 0
    for r in concat_report[:-1]:
        acc += int(r["target_frames"] - min(cfg.overlap, r["target_frames"] // 3))
        seam_positions.append(acc)
    seam_mask = make_boundary_mask(motion.shape[0], seam_positions, width=24)
'''
    new = '''    motion, concat_report = concat_events(selected_paths, [s["duration"] for s in slots], cfg)
    seam_mask, seam_positions, transition_spans = transition_mask_from_concat_report(motion.shape[0], concat_report, default_width=24)
'''
    if old in text:
        text = text.replace(old, new, 1)
    else:
        # More robust regex for already-edited local files.
        pattern = (r'    motion, concat_report = concat_events\(selected_paths, \[s\["duration"\] for s in slots\], cfg\)\n'
                   r'    seam_positions = \[\]\n'
                   r'    acc = 0\n'
                   r'    for r in concat_report\[:-1\]:\n'
                   r'        acc \+= int\(r\["target_frames"\] - min\(cfg\.overlap, r\["target_frames"\] // 3\)\)\n'
                   r'        seam_positions\.append\(acc\)\n'
                   r'    seam_mask = make_boundary_mask\(motion\.shape\[0\], seam_positions, width=24\)\n')
        text2, n = re.subn(pattern, new, text, count=1)
        if n == 0 and "transition_mask_from_concat_report" not in text:
            raise SystemExit("[ERROR] Could not patch generate() seam mask block.")
        text = text2

    # Add transition spans to stage reports when the standard line exists.
    old_stage = '    stage_reports = {"retrieval": retrieval_report, "concat": concat_report, "seams": seam_positions}\n'
    new_stage = '    stage_reports = {"retrieval": retrieval_report, "concat": concat_report, "seams": seam_positions, "transition_spans": transition_spans, "transition_budget_enabled": _v46_env_bool("V46_TRANSITION_BUDGET_ENABLE", True)}\n'
    if old_stage in text:
        text = text.replace(old_stage, new_stage, 1)

    TARGET.write_text(text, encoding="utf-8")
    print("[OK] Applied V46.32 transition-budgeted inbetweening patch to", TARGET)
    print("[INFO] Environment switch: V46_TRANSITION_BUDGET_ENABLE=1 enables new concat; 0 falls back to V46.31 overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
