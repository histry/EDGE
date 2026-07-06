#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V46.33 Reference-Conditioned Transition-Masked MotionRAG-Diff patch.

Drop this file into tools/ and run it after the existing V46.31 research
contract patch / JSON safety hotfix. It modifies tools/v46_motionrag_diff.py
in-place, with timestamped backup.

Design goal:
  RAG + transition-budget inbetweening -> motion_ref
  motion_ref is a strong reference trajectory for refiner/diffusion
  diffusion/refiner only edits transition-mask regions by default
  V43 IK finalizes foot-ground contact.
"""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

TARGET = Path("tools/v46_motionrag_diff.py")


def find_func_span(text: str, name: str):
    pat = rf"^def\s+{re.escape(name)}\s*\("
    m = re.search(pat, text, flags=re.M)
    if not m:
        return None
    start = m.start()
    m2 = re.search(r"^(def\s+|class\s+|@dataclasses\.dataclass)", text[m.end():], flags=re.M)
    end = m.end() + m2.start() if m2 else len(text)
    return start, end


def replace_func(text: str, name: str, new_src: str) -> str:
    span = find_func_span(text, name)
    if span is None:
        raise RuntimeError(f"Cannot locate function {name}() in {TARGET}")
    return text[:span[0]] + new_src.rstrip() + "\n\n\n" + text[span[1]:]


CONCAT_AND_HELPERS = r'''
# === V46.33 reference-conditioned transition budget begin ===
def _v46_33_env_bool(name: str, default: bool) -> bool:
    if name in os.environ:
        try:
            return bool(int(os.environ[name]))
        except Exception:
            return str(os.environ[name]).strip().lower() in {"true", "yes", "on"}
    return bool(default)


def _v46_33_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _v46_33_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _v46_33_cfg_bool(cfg: V46Config, attr: str, env: str, default: bool) -> bool:
    return _v46_33_env_bool(env, bool(getattr(cfg, attr, default)))


def _v46_33_cfg_int(cfg: V46Config, attr: str, env: str, default: int) -> int:
    return _v46_33_env_int(env, int(getattr(cfg, attr, default)))


def _v46_33_cfg_float(cfg: V46Config, attr: str, env: str, default: float) -> float:
    return _v46_33_env_float(env, float(getattr(cfg, attr, default)))


def _v46_33_slerp_quat_np(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Vectorized quaternion SLERP. q0/q1: [J,4], t: [T,1,1]."""
    q0 = normalize_quat_np(np.asarray(q0, dtype=np.float32))
    q1 = normalize_quat_np(np.asarray(q1, dtype=np.float32))
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    tt = np.asarray(t, dtype=np.float32)
    q0b = q0[None]
    q1b = q1[None]
    dotb = dot[None]
    thetab = theta[None]
    sinb = sin_theta[None]
    lerp = normalize_quat_np((1.0 - tt) * q0b + tt * q1b)
    s0 = np.sin((1.0 - tt) * thetab) / np.maximum(sinb, 1e-6)
    s1 = np.sin(tt * thetab) / np.maximum(sinb, 1e-6)
    slerp = normalize_quat_np(s0 * q0b + s1 * q1b)
    use_lerp = (dotb > 0.9995) | (np.abs(sinb) < 1e-6)
    return np.where(use_lerp, lerp, slerp).astype(np.float32)


def v46_33_motion_inbetween_np(prev_tail: np.ndarray, curr_head: np.ndarray, n_frames: int, cfg: V46Config) -> np.ndarray:
    """Kinematic inbetweening in EDGE-151D: root Hermite + per-joint rotation SLERP.

    prev_tail and curr_head are short clips. The generated bridge excludes both
    endpoints, so it can be inserted between previous core and current core
    without duplicating boundary frames.
    """
    n = int(n_frames)
    if n <= 0:
        return np.zeros((0, EDGE_DIM), dtype=np.float32)
    a_clip = np.asarray(prev_tail, dtype=np.float32)
    b_clip = np.asarray(curr_head, dtype=np.float32)
    a = a_clip[-1].copy()
    b = b_clip[0].copy()
    out = np.zeros((n, EDGE_DIM), dtype=np.float32)
    phase = (np.arange(n, dtype=np.float32) + 1.0) / float(n + 1)
    s = phase[:, None]
    smooth = (s * s * (3.0 - 2.0 * s)).astype(np.float32)

    # Contact channels are re-derived after FK; keep them as conservative blends here.
    out[:, 0:4] = ((1.0 - smooth) * a[None, 0:4] + smooth * b[None, 0:4]).astype(np.float32)

    # Root position: C1 Hermite using local endpoint velocities.
    p0 = a[[ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].astype(np.float32)
    p1 = b[[ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].astype(np.float32)
    v0 = np.zeros(3, dtype=np.float32)
    v1 = np.zeros(3, dtype=np.float32)
    if a_clip.shape[0] >= 2:
        v0 = (a_clip[-1, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] - a_clip[-2, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]).astype(np.float32)
    if b_clip.shape[0] >= 2:
        v1 = (b_clip[1, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] - b_clip[0, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]).astype(np.float32)
    # Bound bridge tangents to avoid long-range root launches at mismatched clips.
    max_step = _v46_33_cfg_float(cfg, "transition_root_tangent_max_mpf", "V46_TRANSITION_ROOT_TANGENT_MAX_MPF", 0.045)
    for vv in (v0, v1):
        norm = float(np.linalg.norm(vv[[0, 2]]))
        if norm > max_step:
            vv[[0, 2]] *= max_step / max(norm, 1e-8)
    tt = phase[:, None]
    h00 = 2 * tt ** 3 - 3 * tt ** 2 + 1
    h10 = tt ** 3 - 2 * tt ** 2 + tt
    h01 = -2 * tt ** 3 + 3 * tt ** 2
    h11 = tt ** 3 - tt ** 2
    scale = float(n + 1)
    root = h00 * p0[None] + h10 * (v0[None] * scale) + h01 * p1[None] + h11 * (v1[None] * scale)
    out[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = root.astype(np.float32)

    # Rotation: joint-wise quaternion SLERP, then convert back to legal Rot6D.
    Ra = rot6d_to_matrix_np(a[ROT6D_START:ROT6D_END].reshape(1, NUM_JOINTS, 6))[0]
    Rb = rot6d_to_matrix_np(b[ROT6D_START:ROT6D_END].reshape(1, NUM_JOINTS, 6))[0]
    qa = matrix_to_quat_np(Ra)
    qb = matrix_to_quat_np(Rb)
    q = _v46_33_slerp_quat_np(qa, qb, phase.reshape(n, 1, 1))
    R = quat_to_matrix_np(q)
    out[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(R).reshape(n, -1)

    out, _ = enforce_edge151_contract_np(out, cfg, source_hint="v46_33_motion_inbetween", derive_contact=True, project_rot=True)
    return out.astype(np.float32)


def _v46_33_align_core_to_prev(prev_piece: np.ndarray, core: np.ndarray, cfg: V46Config) -> Tuple[np.ndarray, dict]:
    """Align current core to previous endpoint in yaw and XZ only."""
    out = core.copy().astype(np.float32)
    report: Dict[str, object] = {"mode": "yaw_xz_to_previous_endpoint_no_root_y_ramp"}
    if prev_piece.size == 0 or out.size == 0:
        return out, report
    try:
        yaw_prev = float(root_yaw_np(prev_piece[-1:])[0])
        yaw_core = float(root_yaw_np(out[:1])[0])
        dyaw = float(np.arctan2(np.sin(yaw_prev - yaw_core), np.cos(yaw_prev - yaw_core)))
    except Exception:
        yaw_prev, yaw_core, dyaw = 0.0, 0.0, 0.0
    out = rotate_motion_around_y_np(out, dyaw, pivot_xz=out[0, [ROOT_X_IDX, ROOT_Z_IDX]])
    delta = prev_piece[-1, [ROOT_X_IDX, ROOT_Z_IDX]] - out[0, [ROOT_X_IDX, ROOT_Z_IDX]]
    out[:, ROOT_X_IDX] += float(delta[0])
    out[:, ROOT_Z_IDX] += float(delta[1])
    out, contract = enforce_edge151_contract_np(out, cfg, source_hint="v46_33_align_core_to_prev", derive_contact=True, project_rot=True)
    report.update({
        "yaw_prev": float(yaw_prev),
        "yaw_core_before": float(yaw_core),
        "dyaw_applied": float(dyaw),
        "delta_xz_applied": [float(delta[0]), float(delta[1])],
        "root_y_ramp_applied": False,
        "contract": contract,
    })
    return out.astype(np.float32), report


def _v46_33_choose_core_and_transition_lengths(source_len: int, target_len: int, has_prev: bool, cfg: V46Config) -> Tuple[int, int, dict]:
    """Return (core_len, transition_in_len) while preserving target_len exactly."""
    target_len = max(1, int(target_len))
    source_len = max(1, int(source_len))
    if not has_prev:
        return target_len, 0, {"reason": "first_slot_no_transition", "core_warp": float(target_len / source_len)}

    min_trans = _v46_33_cfg_int(cfg, "transition_min_frames", "V46_TRANSITION_MIN_FRAMES", 10)
    max_trans = _v46_33_cfg_int(cfg, "transition_max_frames", "V46_TRANSITION_MAX_FRAMES", 28)
    ratio = _v46_33_cfg_float(cfg, "transition_ratio", "V46_TRANSITION_RATIO", 0.18)
    min_core = _v46_33_cfg_int(cfg, "transition_min_core_frames", "V46_TRANSITION_MIN_CORE_FRAMES", 30)
    warp_min = _v46_33_cfg_float(cfg, "core_warp_min", "V46_CORE_WARP_MIN", 0.72)
    warp_max = _v46_33_cfg_float(cfg, "core_warp_max", "V46_CORE_WARP_MAX", 1.38)

    if target_len <= min_core + 2:
        return target_len, 0, {"reason": "slot_too_short_for_transition", "core_warp": float(target_len / source_len)}

    trans = int(round(target_len * ratio))
    trans = max(min_trans, min(max_trans, trans))
    trans = min(trans, max(0, target_len - min_core))
    core = max(min_core, target_len - trans)

    # Prefer natural core duration, but never violate total slot length.
    lower = max(min_core, int(round(source_len * warp_min)))
    upper = max(lower, int(round(source_len * warp_max)))
    desired = int(np.clip(core, lower, upper))
    desired = min(max(min_core, desired), target_len - max(1, min_trans))
    if desired > 0:
        core = desired
        trans = target_len - core

    if trans < 0:
        trans = 0
        core = target_len
    info = {
        "target_len": int(target_len),
        "source_len": int(source_len),
        "transition_frames": int(trans),
        "core_frames": int(core),
        "core_warp": float(core / max(1, source_len)),
        "warp_min": float(warp_min),
        "warp_max": float(warp_max),
        "ratio": float(ratio),
    }
    return int(core), int(trans), info


def concat_events(event_paths: Sequence[str], target_durations: Sequence[float], cfg: V46Config) -> Tuple[np.ndarray, List[dict]]:
    """V46.33 reference-conditioned transition-budget concatenation.

    This constructs a strong reference motion stream (motion_ref): each music
    slot contributes exactly target_frames.  For non-first slots, part of the
    slot is reserved as transition budget; the core event is lightly resampled,
    aligned in yaw/XZ, and connected through root-Hermite + rotation-SLERP
    inbetweening.  The generated transition spans are reported so generate()
    can build a precise transition mask for V45/V46.
    """
    if not _v46_33_cfg_bool(cfg, "transition_budget_enable", "V46_TRANSITION_BUDGET_ENABLE", True):
        if "concat_events_v46_31_overlap" in globals():
            return concat_events_v46_31_overlap(event_paths, target_durations, cfg)

    pieces: List[np.ndarray] = []
    rep: List[dict] = []
    target_lens = [max(cfg.min_event_frames, int(round(float(d) * cfg.fps))) for d in target_durations]
    cursor = 0
    transition_spans_global: List[Tuple[int, int]] = []

    for i, (p, dur) in enumerate(zip(event_paths, target_durations)):
        m_raw = np.load(str(p)).astype(np.float32)
        m, pre_report = enforce_edge151_contract_np(
            m_raw, cfg, source_hint=f"v46_33_concat_load:{p}", derive_contact=True, project_rot=True
        )
        target_len = int(target_lens[i])
        has_prev = bool(pieces)
        core_len, trans_len, length_info = _v46_33_choose_core_and_transition_lengths(m.shape[0], target_len, has_prev, cfg)
        core = resample_motion_np(m, int(core_len)).astype(np.float32)
        core, core_report = enforce_edge151_contract_np(
            core, cfg, source_hint=f"v46_33_core_resample:{p}", derive_contact=True, project_rot=True
        )
        align_report = None
        bridge_report: Dict[str, object] = {"enabled": False, "frames": 0}
        transition_span = None

        if has_prev and trans_len > 0 and _v46_33_cfg_bool(cfg, "transition_inbetween_enable", "V46_TRANSITION_INBETWEEN_ENABLE", True):
            core, align_report = _v46_33_align_core_to_prev(pieces[-1], core, cfg)
            prev_tail_n = min(max(2, trans_len // 2), len(pieces[-1]))
            curr_head_n = min(max(2, trans_len // 2), len(core))
            bridge = v46_33_motion_inbetween_np(pieces[-1][-prev_tail_n:], core[:curr_head_n], trans_len, cfg)
            start = cursor
            end = cursor + int(bridge.shape[0])
            transition_span = [int(start), int(end)]
            transition_spans_global.append((int(start), int(end)))
            pieces.append(bridge.astype(np.float32))
            cursor += int(bridge.shape[0])
            bridge_report = {
                "enabled": True,
                "mode": "root_hermite_rotation_slerp_motion_space_inbetweening",
                "frames": int(bridge.shape[0]),
                "span": transition_span,
                "prev_tail_frames": int(prev_tail_n),
                "curr_head_frames": int(curr_head_n),
            }
        elif has_prev:
            core, align_report = _v46_33_align_core_to_prev(pieces[-1], core, cfg)

        pieces.append(core.astype(np.float32))
        core_span = [int(cursor), int(cursor + core.shape[0])]
        cursor += int(core.shape[0])
        rep.append({
            "version": "v46_33_reference_conditioned_transition_budget",
            "path": str(p),
            "target_frames": int(target_len),
            "source_frames": int(m_raw.shape[0]),
            "core_frames": int(core.shape[0]),
            "transition_in_frames": int(trans_len if has_prev else 0),
            "slot_total_frames": int((trans_len if has_prev else 0) + core.shape[0]),
            "core_span": core_span,
            "transition_span": transition_span,
            "transition_spans": [transition_span] if transition_span else [],
            "core_warp": float(core.shape[0] / max(1, m_raw.shape[0])),
            "length_policy": length_info,
            "contract_pre": pre_report,
            "contract_core": core_report,
            "contract_after_align": align_report,
            "boundary_inbetween": bridge_report,
            "reference_conditioning": {
                "motion_ref_role": "strong_reference_trajectory",
                "diffusion_should_edit": "transition_mask_regions_only_by_default",
                "core_motion_preservation": True,
            },
        })

    if pieces:
        final = np.concatenate(pieces, axis=0).astype(np.float32)
    else:
        final = np.zeros((0, EDGE_DIM), dtype=np.float32)
    total_target_frames = int(sum(target_lens))
    timing_report = {
        "target_total_frames": int(total_target_frames),
        "frames_before_terminal_guard": int(final.shape[0]),
        "timing_frame_delta_before_terminal_guard": int(total_target_frames - final.shape[0]),
        "timing_compensation_applied": False,
        "timing_compensation_mode": "v46_33_slot_exact_transition_budget_no_global_resample",
        "global_resample_applied": False,
        "transition_spans_global": [[int(a), int(b)] for a, b in transition_spans_global],
    }
    if total_target_frames > 0 and int(final.shape[0]) != int(total_target_frames):
        delta = int(total_target_frames - final.shape[0])
        if delta > 0:
            pad = np.repeat(final[-1:, :], delta, axis=0).astype(np.float32)
            final = np.concatenate([final, pad], axis=0).astype(np.float32)
            mode = "terminal_hold_last_frame_pad_no_global_resample"
        else:
            final = final[:total_target_frames].astype(np.float32)
            mode = "terminal_trim_no_global_resample"
        timing_report.update({
            "timing_compensation_applied": True,
            "timing_compensation_mode": mode,
            "terminal_delta_frames": int(delta),
        })
    timing_report["frames_after_terminal_guard"] = int(final.shape[0])
    final, final_report = enforce_edge151_contract_np(
        final, cfg, source_hint="v46_33_concat_final_motion_ref", derive_contact=True, project_rot=True
    )
    if rep:
        rep[-1]["concat_timing_compensation"] = timing_report
        rep[-1]["concat_final_contract"] = final_report
    return final.astype(np.float32), rep


def make_transition_budget_mask(T: int, transition_spans: Sequence[Sequence[int]], cfg: V46Config) -> np.ndarray:
    """Build precise transition mask with optional halo and low core mask."""
    core_val = _v46_33_cfg_float(cfg, "transition_core_mask_value", "V46_TRANSITION_CORE_MASK_VALUE", 0.0)
    halo = _v46_33_cfg_int(cfg, "transition_mask_halo", "V46_TRANSITION_MASK_HALO", 6)
    mask = np.full((int(T), 1), float(core_val), dtype=np.float32)
    for sp in transition_spans:
        if sp is None or len(sp) < 2:
            continue
        a, b = int(sp[0]), int(sp[1])
        a0 = max(0, a - halo)
        b0 = min(int(T), b + halo)
        if b0 <= a0:
            continue
        # Raised plateau: transition core = 1, halo ramps down to core_val.
        mask[a:b, 0] = 1.0
        if halo > 0:
            la = max(0, a - halo)
            if a > la:
                ramp = np.linspace(float(core_val), 1.0, a - la, endpoint=False, dtype=np.float32)
                mask[la:a, 0] = np.maximum(mask[la:a, 0], ramp)
            rb = min(int(T), b + halo)
            if rb > b:
                ramp = np.linspace(1.0, float(core_val), rb - b, endpoint=False, dtype=np.float32)
                mask[b:rb, 0] = np.maximum(mask[b:rb, 0], ramp)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)
# === V46.33 reference-conditioned transition budget end ===
'''


DEGRADE = r'''
def degrade_for_refiner(clean: np.ndarray, severity: float = 0.06, cfg: Optional[V46Config] = None) -> Tuple[np.ndarray, np.ndarray]:
    """V46.33 transition-masked corruption for V45/V46 training.

    Instead of arbitrary global drift only, corrupt a local transition region by
    replacing it with a weak root-Hermite / rotation-SLERP inbetweening path plus
    noise. This matches the inference-time transition-budget mask: the model
    learns to repair motion_ref only near boundaries while preserving core clips.
    """
    cfg = cfg or V46Config()
    x = np.asarray(clean, dtype=np.float32).copy()
    T, D = x.shape
    seam = np.zeros((T, 1), dtype=np.float32)
    if T <= 12:
        x, _ = enforce_edge151_contract_np(x, cfg, source_hint="v46_33_degrade_too_short", derive_contact=True, project_rot=True)
        return x.astype(np.float32), seam

    min_w = _v46_33_cfg_int(cfg, "transition_train_min_frames", "V46_TRANSITION_TRAIN_MIN_FRAMES", 10)
    max_w = _v46_33_cfg_int(cfg, "transition_train_max_frames", "V46_TRANSITION_TRAIN_MAX_FRAMES", 28)
    halo = _v46_33_cfg_int(cfg, "transition_mask_halo", "V46_TRANSITION_MASK_HALO", 6)
    max_w = max(min_w, min(max_w, max(4, T // 3)))
    w = random.randint(max(4, min_w), max_w)
    c = random.randint(max(2, T // 5), max(3, 4 * T // 5))
    a = max(1, c - w // 2)
    b = min(T - 1, a + w)
    a = max(1, b - w)
    if b - a >= 3:
        prev_tail = x[max(0, a - 4):a]
        curr_head = x[b:min(T, b + 4)]
        if prev_tail.shape[0] >= 1 and curr_head.shape[0] >= 1:
            bridge = v46_33_motion_inbetween_np(prev_tail, curr_head, b - a, cfg)
            # Add light residual corruption mainly in root/rot channels; contacts rebuilt later.
            noise = np.zeros_like(bridge, dtype=np.float32)
            noise[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = np.random.normal(0, severity * 0.18, size=(bridge.shape[0], 3)).astype(np.float32)
            noise[:, ROT6D_START:ROT6D_END] = np.random.normal(0, severity * 0.08, size=(bridge.shape[0], ROT6D_END - ROT6D_START)).astype(np.float32)
            x[a:b] = bridge + noise
            seam[max(0, a - halo):min(T, b + halo), 0] = 0.35
            seam[a:b, 0] = 1.0
        # Soft post-boundary drift to simulate mismatched retrieval alignment.
        offset = np.zeros(D, dtype=np.float32)
        offset[[ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = np.random.normal(0, severity * 0.45, size=3)
        offset[ROT6D_START:ROT6D_END] = np.random.normal(0, severity * 0.16, size=ROT6D_END - ROT6D_START)
        tail = T - b
        if tail > 0:
            decay = np.linspace(1.0, 0.0, tail, dtype=np.float32)[:, None]
            x[b:] += decay * offset[None]

    # Tiny background noise keeps denoising stable without encouraging core rewrite.
    noise = np.zeros_like(x, dtype=np.float32)
    noise[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = np.random.normal(0, severity * 0.025, size=(T, 3)).astype(np.float32)
    noise[:, ROT6D_START:ROT6D_END] = np.random.normal(0, severity * 0.012, size=(T, ROT6D_END - ROT6D_START)).astype(np.float32)
    x += noise
    x, _ = enforce_edge151_contract_np(x, cfg, source_hint="v46_33_degrade_for_transition_refiner", derive_contact=True, project_rot=True)
    return x.astype(np.float32), np.clip(seam, 0.0, 1.0).astype(np.float32)
'''


APPLY_REFINER = r'''
def apply_refiner_model(motion: np.ndarray, cond: np.ndarray, seam_mask: np.ndarray, ckpt_path: Optional[str], cfg: V46Config) -> np.ndarray:
    """Apply V45 as reference-conditioned transition residual refiner.

    Core regions are strongly locked.  By default only a tiny residual is allowed
    outside transition masks; transition regions receive the full correction.
    """
    core_strength = _v46_33_cfg_float(cfg, "refiner_core_strength", "V46_REFINER_CORE_STRENGTH", 0.02)
    trans_strength = _v46_33_cfg_float(cfg, "refiner_transition_strength", "V46_REFINER_TRANSITION_STRENGTH", 1.00)
    if torch is None or not ckpt_path or not Path(ckpt_path).exists():
        seam_centers = []
        for a, b in contiguous_regions(seam_mask[:, 0] > 0.5):
            seam_centers.append((a + b) // 2)
        refined = analytic_residual_refine(motion, seam_centers)
        # Blend analytic fallback back to the reference outside transition mask.
        w = np.clip(core_strength + (trans_strength - core_strength) * seam_mask.astype(np.float32), 0.0, 1.0)
        refined = motion.astype(np.float32) * (1.0 - w) + refined.astype(np.float32) * w
        refined, _ = enforce_edge151_contract_np(
            refined, cfg, source_hint="apply_refiner_model:v46_33_reference_analytic", derive_contact=True, project_rot=True
        )
        return refined.astype(np.float32)

    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model = TemporalRefiner(EDGE_DIM, 32).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    T = int(motion.shape[0])
    win = int(cfg.window_len)
    hop = max(1, min(int(getattr(cfg, "hop_len", win)), win))
    accum, weight_sum, rot_quat_accum, rot_quat_weight = init_motion_window_accumulators(T, EDGE_DIM)

    with torch.no_grad():
        for st, ed in sliding_window_ranges(T, win, hop):
            chunk = motion[st:ed]
            mask = seam_mask[st:ed]
            orig_len = len(chunk)
            if orig_len < win:
                chunk_in = resample_motion_np(chunk, win)
                mask_in = resample_motion_np(mask, win)
            else:
                chunk_in = chunk
                mask_in = mask
            chunk_in, _ = enforce_edge151_contract_np(
                chunk_in, cfg, source_hint="apply_refiner_model:v46_33_input_chunk", derive_contact=True, project_rot=True
            )
            x = torch.from_numpy(chunk_in[None]).float().to(cfg.device)
            c = torch.from_numpy(cond[None].astype(np.float32)).float().to(cfg.device)
            sm = torch.from_numpy(mask_in[None].astype(np.float32)).float().to(cfg.device)
            delta = model(x, c, sm)
            strength = torch.clamp(float(core_strength) + (float(trans_strength) - float(core_strength)) * sm, 0.0, 1.0)
            y = x + delta * strength
            y_np = y[0].detach().cpu().numpy()
            if orig_len < win:
                y_np = resample_motion_np(y_np, orig_len)
            y_np, _ = enforce_edge151_contract_np(
                y_np, cfg, source_hint="apply_refiner_model:v46_33_output_chunk", derive_contact=True, project_rot=True
            )
            w = overlap_add_weight_np(orig_len, st, T, hop, win)
            accumulate_motion_window_np(accum, weight_sum, rot_quat_accum, rot_quat_weight, y_np, w, st, ed)

    out, _ = finalize_motion_window_accum_np(
        accum, weight_sum, rot_quat_accum, rot_quat_weight, cfg, source_hint="apply_refiner_model:v46_33_final"
    )
    # Hard blend with original reference according to the exact transition mask.
    w = np.clip(core_strength + (trans_strength - core_strength) * seam_mask.astype(np.float32), 0.0, 1.0)
    out = motion.astype(np.float32) * (1.0 - w) + out.astype(np.float32) * w
    out, _ = enforce_edge151_contract_np(out, cfg, source_hint="apply_refiner_model:v46_33_reference_blend", derive_contact=True, project_rot=True)
    return out.astype(np.float32)
'''


APPLY_DIFFUSION = r'''
def apply_diffusion_model(motion: np.ndarray, cond: np.ndarray, seam_mask: np.ndarray, ckpt_path: Optional[str], cfg: V46Config) -> np.ndarray:
    """Apply V46 diffusion as transition-masked residual generation.

    The input motion is motion_ref / motion_refined and is treated as a strong
    retrieval/reference condition.  Core frames are locked by mask, while
    transition frames are allowed to be regenerated as residual motion.
    """
    if torch is None or not ckpt_path or not Path(ckpt_path).exists():
        motion, _ = enforce_edge151_contract_np(
            motion, cfg, source_hint="apply_diffusion_model:v46_33_disabled", derive_contact=True, project_rot=True
        )
        return motion.astype(np.float32)

    core_strength = _v46_33_cfg_float(cfg, "diffusion_core_strength", "V46_DIFFUSION_CORE_STRENGTH", 0.00)
    trans_strength = _v46_33_cfg_float(cfg, "diffusion_transition_strength", "V46_DIFFUSION_TRANSITION_STRENGTH", 0.72)
    noise_scale = _v46_33_cfg_float(cfg, "diffusion_reference_noise_scale", "V46_DIFFUSION_REFERENCE_NOISE_SCALE", 0.03)

    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    Tdiff = int(ckpt.get("diffusion_steps", cfg.diffusion_steps))
    model = DiffusionDenoiser(EDGE_DIM, 32).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    betas, alphas, abar = make_beta_schedule(Tdiff, torch.device(cfg.device))

    T = int(motion.shape[0])
    win = int(cfg.window_len)
    hop = max(1, min(int(getattr(cfg, "hop_len", win)), win))
    accum, weight_sum, rot_quat_accum, rot_quat_weight = init_motion_window_accumulators(T, EDGE_DIM)

    with torch.no_grad():
        for st, ed in sliding_window_ranges(T, win, hop):
            retr_np = motion[st:ed]
            mask_np = seam_mask[st:ed]
            orig_len = len(retr_np)
            if orig_len < win:
                retr_in = resample_motion_np(retr_np, win)
                mask_in = resample_motion_np(mask_np, win)
            else:
                retr_in = retr_np
                mask_in = mask_np
            retr_in, _ = enforce_edge151_contract_np(
                retr_in, cfg, source_hint="apply_diffusion_model:v46_33_retrieval_chunk", derive_contact=True, project_rot=True
            )
            retr = torch.from_numpy(retr_in[None]).float().to(cfg.device)
            raw_mask = torch.from_numpy(mask_in[None].astype(np.float32)).float().to(cfg.device)
            mask = torch.clamp(float(core_strength) + (float(trans_strength) - float(core_strength)) * raw_mask, 0.0, 1.0)
            c = torch.from_numpy(cond[None].astype(np.float32)).float().to(cfg.device)
            x = retr + float(noise_scale) * torch.randn_like(retr) * (0.15 + 0.85 * mask)
            for ti in reversed(range(Tdiff)):
                t = torch.full((1,), ti, device=cfg.device, dtype=torch.long)
                eps = model(x, retr, c, raw_mask, t)
                beta = betas[ti]
                alpha = alphas[ti]
                ab = abar[ti]
                mean = (1 / torch.sqrt(alpha)) * (x - beta / torch.sqrt(1 - ab).clamp_min(1e-6) * eps)
                if ti > 0:
                    x = mean + torch.sqrt(beta) * torch.randn_like(x) * 0.35
                else:
                    x = mean
                # Strong reference lock: core mask=0 returns exactly retr.
                x = retr * (1.0 - mask) + x * mask
            y = x[0].detach().cpu().numpy()
            if orig_len < win:
                y = resample_motion_np(y, orig_len)
            y, _ = enforce_edge151_contract_np(
                y, cfg, source_hint="apply_diffusion_model:v46_33_output_chunk", derive_contact=True, project_rot=True
            )
            w = overlap_add_weight_np(orig_len, st, T, hop, win)
            accumulate_motion_window_np(accum, weight_sum, rot_quat_accum, rot_quat_weight, y, w, st, ed)

    out, _ = finalize_motion_window_accum_np(
        accum, weight_sum, rot_quat_accum, rot_quat_weight, cfg, source_hint="apply_diffusion_model:v46_33_final"
    )
    # Final exact reference blend in the original-length mask coordinates.
    w = np.clip(core_strength + (trans_strength - core_strength) * seam_mask.astype(np.float32), 0.0, 1.0)
    out = motion.astype(np.float32) * (1.0 - w) + out.astype(np.float32) * w
    out, _ = enforce_edge151_contract_np(out, cfg, source_hint="apply_diffusion_model:v46_33_reference_blend", derive_contact=True, project_rot=True)
    return out.astype(np.float32)
'''


GENERATE = r'''
def generate(args: argparse.Namespace) -> int:
    cfg = V46Config.from_json(args.config).apply_env()
    sem_dirs = getattr(args, "music_semantic_dirs", None)
    if sem_dirs:
        cfg.external_music_semantic_dirs = os.pathsep.join([str(x) for x in sem_dirs])
    if getattr(args, "external_music_semantic_cmd", None):
        cfg.external_music_semantic_cmd = str(args.external_music_semantic_cmd)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch is not None:
        torch.manual_seed(cfg.seed)
    db = load_db(args.db)
    contrastive = load_contrastive(args.contrastive, cfg)
    slots, slot_feat = audio_slots(args.audio, cfg, args.slot_seconds, args.slots_json)
    path_idx, retrieval_report = retrieve_schedule(slots, slot_feat, db, cfg, contrastive)
    paths = np.asarray(db["paths"], dtype=object)
    selected_paths = [str(paths[i]) for i in path_idx]

    motion_ref, concat_report = concat_events(selected_paths, [s["duration"] for s in slots], cfg)

    transition_spans: List[List[int]] = []
    for r in concat_report:
        for sp in r.get("transition_spans", []):
            if sp is not None and len(sp) >= 2:
                transition_spans.append([int(sp[0]), int(sp[1])])
    if transition_spans:
        seam_mask = make_transition_budget_mask(motion_ref.shape[0], transition_spans, cfg)
        seam_positions = [int((a + b) // 2) for a, b in transition_spans]
        mask_policy = "v46_33_transition_budget_spans"
    else:
        seam_positions = []
        acc = 0
        for r in concat_report[:-1]:
            acc += int(r.get("target_frames", 0))
            seam_positions.append(acc)
        seam_mask = make_boundary_mask(motion_ref.shape[0], seam_positions, width=24)
        mask_policy = "fallback_boundary_mask_no_transition_spans"

    cond = np.mean(slot_feat, axis=0).astype(np.float32)
    cond = (cond - np.asarray(db["desc_mean"], dtype=np.float32)[0]) / np.asarray(db["desc_std"], dtype=np.float32)[0]

    stage_reports = {
        "retrieval": retrieval_report,
        "concat": concat_report,
        "seams": seam_positions,
        "transition_spans": transition_spans,
        "seam_mask_policy": mask_policy,
        "seam_mask_stats": {
            "shape": list(seam_mask.shape),
            "mean": float(np.mean(seam_mask)) if seam_mask.size else 0.0,
            "max": float(np.max(seam_mask)) if seam_mask.size else 0.0,
            "transition_frame_ratio": float(np.mean(seam_mask[:, 0] > 0.5)) if seam_mask.size else 0.0,
        },
        "v46_33_reference_conditioning": {
            "motion_ref_as_strong_reference": True,
            "diffusion_edit_policy": "transition_masked_residual_generation",
            "ik_finalization": bool(cfg.ik_enable),
            "env": {
                "V46_TRANSITION_BUDGET_ENABLE": os.environ.get("V46_TRANSITION_BUDGET_ENABLE", "1"),
                "V46_TRANSITION_INBETWEEN_ENABLE": os.environ.get("V46_TRANSITION_INBETWEEN_ENABLE", "1"),
                "V46_REFINER_CORE_STRENGTH": os.environ.get("V46_REFINER_CORE_STRENGTH", "0.02"),
                "V46_DIFFUSION_CORE_STRENGTH": os.environ.get("V46_DIFFUSION_CORE_STRENGTH", "0.00"),
                "V46_DIFFUSION_TRANSITION_STRENGTH": os.environ.get("V46_DIFFUSION_TRANSITION_STRENGTH", "0.72"),
            },
        },
    }
    pre_audit = audit_motion_np(motion_ref, cfg)
    motion = motion_ref.astype(np.float32)

    if cfg.refiner_enable:
        motion = apply_refiner_model(motion, cond, seam_mask, args.refiner, cfg)
        stage_reports["v45_refiner_audit"] = audit_motion_np(motion, cfg)
    if cfg.diffusion_enable:
        motion = apply_diffusion_model(motion, cond, seam_mask, args.diffusion, cfg)
        stage_reports["v46_diffusion_audit"] = audit_motion_np(motion, cfg)

    ik_report = {"enabled": False}
    if cfg.ik_enable:
        motion, ik_report = true_lower_body_ik(motion, cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, motion.astype(np.float32))
    # Save reference motion next to final output for ablation and paper figures.
    motion_ref_path = str(out).replace(".npy", ".motion_ref.npy")
    np.save(motion_ref_path, motion_ref.astype(np.float32))
    mask_path = str(out).replace(".npy", ".transition_mask.npy")
    np.save(mask_path, seam_mask.astype(np.float32))

    report = {
        "version": "v46_33_reference_conditioned_transition_masked_motionrag_diff",
        "audio": args.audio,
        "db": args.db,
        "config": dataclasses.asdict(cfg),
        "fk_tree_source": FK_TREE_SOURCE,
        "selected_event_indices": path_idx,
        "selected_event_paths": selected_paths,
        "slots": slots,
        "motion_ref_path": motion_ref_path,
        "transition_mask_path": mask_path,
        "pre_refine_audit": pre_audit,
        "stage_reports": stage_reports,
        "v43_true_ik": ik_report,
        "final_audit": audit_motion_np(motion, cfg),
    }
    json_path = args.json or str(out).replace(".npy", ".v46_33_report.json")
    save_json(report, json_path)
    if args.render_output:
        render_if_possible(str(out), args.audio, args.render_output, args.render_script)
    print(json.dumps({"motion": str(out), "motion_ref": motion_ref_path, "transition_mask": mask_path, "json": json_path, "frames": int(motion.shape[0]), "final_audit": report["final_audit"]}, ensure_ascii=False, indent=2))
    return 0
'''


def main() -> int:
    if not TARGET.exists():
        raise SystemExit(f"[ERROR] Missing {TARGET}. Run from EDGE repo root.")
    s = TARGET.read_text(encoding="utf-8")
    bak = TARGET.with_suffix(TARGET.suffix + f".v46_33_reference_transition_{time.strftime('%Y%m%d_%H%M%S')}.bak")
    shutil.copy2(TARGET, bak)
    print("[BAK]", bak)

    # Preserve old concat as fallback once.
    if "def concat_events_v46_31_overlap" not in s:
        span = find_func_span(s, "concat_events")
        if span is None:
            raise SystemExit("[ERROR] Cannot locate concat_events().")
        old = s[span[0]:span[1]]
        old = old.replace("def concat_events(", "def concat_events_v46_31_overlap(", 1)
        s = s[:span[0]] + old + "\n\n" + CONCAT_AND_HELPERS + s[span[1]:]
    else:
        # Replace current concat_events with V46.33 implementation only.
        s = replace_func(s, "concat_events", CONCAT_AND_HELPERS)

    s = replace_func(s, "degrade_for_refiner", DEGRADE)
    s = replace_func(s, "apply_refiner_model", APPLY_REFINER)
    s = replace_func(s, "apply_diffusion_model", APPLY_DIFFUSION)
    s = replace_func(s, "generate", GENERATE)

    TARGET.write_text(s, encoding="utf-8")
    print("[OK] Applied V46.33 reference-conditioned transition-masked patch to", TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
