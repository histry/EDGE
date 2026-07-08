#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply V46.41 Stage-Anchored Guided Temporal Generative Transactions patch.

Run after V46.38 MSSD/AESD routing patch.  This patch targets the current
V46 motion pipeline code, whose generation path is:
retrieve_schedule -> concat_events -> apply_refiner_model -> apply_diffusion_model -> true_lower_body_ik.

Added mechanisms:
1. Macroscopic Stage Anchoring (MSA): a low-frequency global root-XZ prior that
   keeps the long sequence inside a bounded stage domain.
2. Temporal Generative Transactions (TGT): local transition windows are atomic
   commit/rollback units.
3. Kinematic Barrier Oracle (KBO): non-differentiable physical/topological
   checks before a candidate can be committed.
4. Diffusion early-abort: inspect intermediate denoising candidates and abort to
   deterministic degradation if KBO is already violated.
5. Hard-negative audit persistence: rejected candidates/snapshots can be saved
   for later HN-DPO-style preference fine-tuning.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

TARGET = Path("tools/v46_motionrag_diff.py")
START = "# ===== V46.41 STAGE-ANCHORED GUIDED TGT PATCH START ====="
END = "# ===== V46.41 STAGE-ANCHORED GUIDED TGT PATCH END ====="
OLD_GUARDS = [
    ("# ===== V46.40 TEMPORAL GENERATIVE TRANSACTIONS PATCH START =====", "# ===== V46.40 TEMPORAL GENERATIVE TRANSACTIONS PATCH END ====="),
    ("# ===== V46.39 TRANSACTIONAL ISOLATION PATCH START =====", "# ===== V46.39 TRANSACTIONAL ISOLATION PATCH END ====="),
    ("# ===== V46.38 STAGE SAFETY GUARD PATCH START =====", "# ===== V46.38 STAGE SAFETY GUARD PATCH END ====="),
    ("# ===== V46.34 LONG-SONG SAFETY GUARD PATCH START =====", "# ===== V46.34 LONG-SONG SAFETY GUARD PATCH END ====="),
]

PATCH = r'''
# ===== V46.41 STAGE-ANCHORED GUIDED TGT PATCH START =====
# V46.41: Macroscopic Stage Anchoring + KBO-guided Temporal Generative Transactions.
# This layer is intentionally generation-time only. It preserves the V46.38
# MSSD/AESD routing objective and protects V45/V46/IK from long-horizon drift.

_v46_41_orig_concat_events = concat_events
_v46_41_orig_apply_refiner_model = apply_refiner_model
_v46_41_orig_apply_diffusion_model = apply_diffusion_model
_v46_41_orig_true_lower_body_ik = true_lower_body_ik
_v46_41_orig_generate = generate

_V46_41_AUDIT_TOKENS = []
_V46_41_STAGE_PRIOR_XZ = None
_V46_41_STAGE_PRIOR_META = {}


def _v46_41_env_bool(name, default=True):
    try:
        return bool(int(os.environ.get(name, "1" if default else "0")))
    except Exception:
        return bool(default)


def _v46_41_env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _v46_41_env_int(name, default):
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def _v46_41_jsonable(x):
    try:
        return _v46_json_safe(x)
    except Exception:
        if isinstance(x, dict):
            return {str(k): _v46_41_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_v46_41_jsonable(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        return x if isinstance(x, (str, int, float, bool)) or x is None else str(x)


def _v46_41_reset_audit():
    global _V46_41_AUDIT_TOKENS
    _V46_41_AUDIT_TOKENS = []


def _v46_41_add_token(item):
    global _V46_41_AUDIT_TOKENS
    if len(_V46_41_AUDIT_TOKENS) < _v46_41_env_int("V46_41_AUDIT_MAX_RECORDS", 4000):
        _V46_41_AUDIT_TOKENS.append(_v46_41_jsonable(dict(item)))


def _v46_41_trusted_torch_load(path, map_location=None):
    if torch is None:
        raise RuntimeError("PyTorch is required")
    if "_v46_trusted_torch_load" in globals():
        return _v46_trusted_torch_load(path, map_location=map_location)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _v46_41_build_stage_prior_xz(motion, target_durations=None, cfg=None):
    """Build a low-frequency root-XZ anchor prior from the retrieved motion.

    The prior is conservative: it keeps the macro route but pulls it back into a
    bounded stage radius.  It is intentionally not a learned generator here;
    the MSSD slot durations provide the temporal scaffold, while the retrieved
    motion supplies the cultural trajectory skeleton.
    """
    m = np.asarray(motion, dtype=np.float32)
    T = int(m.shape[0])
    if T <= 1:
        return np.zeros((T, 2), dtype=np.float32), {"enabled": False, "reason": "too_short"}
    xz = m[:, [ROOT_X_IDX, ROOT_Z_IDX]].astype(np.float32)
    center = np.median(xz, axis=0, keepdims=True).astype(np.float32)
    rel = xz - center
    if ndi is not None:
        sigma = max(8.0, T / max(8.0, _v46_41_env_float("V46_41_MSA_SMOOTH_DIV", 72.0)))
        rel_s = ndi.gaussian_filter1d(rel, sigma=float(sigma), axis=0, mode="nearest")
    else:
        rel_s = rel
    radius = _v46_41_env_float("V46_41_STAGE_RADIUS_M", 1.80)
    norm = np.linalg.norm(rel_s, axis=1, keepdims=True)
    rel_clamped = rel_s * np.minimum(1.0, radius / np.maximum(norm, 1e-6))
    prior = (center + rel_clamped).astype(np.float32)
    meta = {
        "enabled": True,
        "version": "v46_41_macroscopic_stage_anchoring",
        "stage_radius_m": float(radius),
        "prior_root_xz_range_before": (xz.max(axis=0) - xz.min(axis=0)).tolist(),
        "prior_root_xz_range_after": (prior.max(axis=0) - prior.min(axis=0)).tolist(),
        "num_target_durations": int(len(target_durations) if target_durations is not None else 0),
    }
    return prior, meta


def _v46_41_apply_stage_prior(motion, cfg, strength=None):
    global _V46_41_STAGE_PRIOR_XZ
    if not _v46_41_env_bool("V46_41_MSA_ENABLE", True):
        return np.asarray(motion, dtype=np.float32), {"enabled": False}
    m = np.asarray(motion, dtype=np.float32).copy()
    prior = _V46_41_STAGE_PRIOR_XZ
    if prior is None or len(prior) != len(m):
        prior, meta = _v46_41_build_stage_prior_xz(m, None, cfg)
    else:
        meta = dict(_V46_41_STAGE_PRIOR_META)
    alpha = _v46_41_env_float("V46_41_MSA_COMMIT_STRENGTH", 0.16) if strength is None else float(strength)
    max_delta = _v46_41_env_float("V46_41_MSA_MAX_DELTA_M", 0.06)
    delta = np.clip(prior - m[:, [ROOT_X_IDX, ROOT_Z_IDX]], -max_delta, max_delta)
    m[:, ROOT_X_IDX] = m[:, ROOT_X_IDX] + float(alpha) * delta[:, 0]
    m[:, ROOT_Z_IDX] = m[:, ROOT_Z_IDX] + float(alpha) * delta[:, 1]
    m, _ = enforce_edge151_contract_np(m, cfg, source_hint="v46_41_msa_apply_stage_prior", derive_contact=True, project_rot=True)
    meta.update({"applied": True, "strength": float(alpha), "max_delta_m": float(max_delta)})
    return m.astype(np.float32), meta


def concat_events(event_paths, target_durations, cfg):
    global _V46_41_STAGE_PRIOR_XZ, _V46_41_STAGE_PRIOR_META
    motion, rep = _v46_41_orig_concat_events(event_paths, target_durations, cfg)
    if _v46_41_env_bool("V46_41_MSA_ENABLE", True):
        _V46_41_STAGE_PRIOR_XZ, _V46_41_STAGE_PRIOR_META = _v46_41_build_stage_prior_xz(motion, target_durations, cfg)
        motion2, meta = _v46_41_apply_stage_prior(motion, cfg, strength=_v46_41_env_float("V46_41_MSA_REFERENCE_STRENGTH", 0.10))
        _V46_41_STAGE_PRIOR_META.update(meta)
        if isinstance(rep, list) and rep:
            rep[-1].setdefault("v46_41_macroscopic_stage_anchor", _V46_41_STAGE_PRIOR_META)
        _v46_41_add_token({"mechanism": "MSA", "stage": "concat", "commit_state": "anchor_applied", "meta": _V46_41_STAGE_PRIOR_META})
        return motion2.astype(np.float32), rep
    return motion, rep


def _v46_41_kinematic_stats(motion, cfg):
    m = np.asarray(motion, dtype=np.float32)
    stats = {"finite": bool(np.isfinite(m).all()), "shape": list(m.shape)}
    if m.ndim != 2 or m.shape[0] < 2 or m.shape[1] < EDGE_DIM:
        stats["valid"] = False
        return stats
    try:
        joints = fk_24_np(m)
        stats["fk_finite"] = bool(np.isfinite(joints).all())
        foot = joints[:, list(DEFAULT_FOOT_JOINTS)]
        foot_y = foot[..., 1]
        stats["floor_y"] = float(np.percentile(foot_y.reshape(-1), 5))
        stats["foot_penetration_min_m"] = float(np.min(foot_y - stats["floor_y"]))
        if joints.shape[0] >= 4:
            vel = np.diff(joints, axis=0)
            acc = np.diff(joints, n=2, axis=0)
            jerk = np.diff(joints, n=3, axis=0)
            stats["joint_vel_p95"] = float(np.percentile(np.linalg.norm(vel, axis=-1).mean(axis=-1), 95))
            stats["joint_acc_max"] = float(np.max(np.linalg.norm(acc, axis=-1).mean(axis=-1)))
            stats["joint_jerk_max"] = float(np.max(np.linalg.norm(jerk, axis=-1).mean(axis=-1)))
            stats["joint_jerk_p95"] = float(np.percentile(np.linalg.norm(jerk, axis=-1).mean(axis=-1), 95))
        bone_vars = []
        for j in range(1, min(NUM_JOINTS, len(PARENTS))):
            pa = int(PARENTS[j])
            if pa < 0 or pa >= NUM_JOINTS:
                continue
            L = np.linalg.norm(joints[:, j] - joints[:, pa], axis=-1)
            bone_vars.append(float(np.max(np.abs(L - np.median(L)))))
        stats["bone_length_violation_max_m"] = float(max(bone_vars) if bone_vars else 0.0)
    except Exception as exc:
        stats["fk_finite"] = False
        stats["fk_error"] = str(exc)
    try:
        stats.update(audit_motion_np(m, cfg))
    except Exception as exc:
        stats["audit_error"] = str(exc)
    stats["root_y_range_m"] = float(np.max(m[:, ROOT_Y_IDX]) - np.min(m[:, ROOT_Y_IDX]))
    xz = m[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    stats["root_xz_radius_p95_m"] = float(np.percentile(np.linalg.norm(xz - np.median(xz, axis=0, keepdims=True), axis=-1), 95))
    stats["valid"] = True
    return stats


def _v46_41_anchor_error(candidate, a0=0):
    global _V46_41_STAGE_PRIOR_XZ
    cand = np.asarray(candidate, dtype=np.float32)
    if _V46_41_STAGE_PRIOR_XZ is None:
        return 0.0
    b0 = int(a0) + len(cand)
    if int(a0) < 0 or b0 > len(_V46_41_STAGE_PRIOR_XZ):
        return 0.0
    prior = _V46_41_STAGE_PRIOR_XZ[int(a0):b0]
    return float(np.percentile(np.linalg.norm(cand[:, [ROOT_X_IDX, ROOT_Z_IDX]] - prior, axis=-1), 95))


def _v46_41_kbo(candidate, reference, cfg, stage="stage", global_start=0):
    cand = np.asarray(candidate, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    reasons = []
    if cand.shape != ref.shape:
        return False, ["shape_changed"], {"candidate_shape": list(cand.shape), "reference_shape": list(ref.shape)}
    c = _v46_41_kinematic_stats(cand, cfg)
    r = _v46_41_kinematic_stats(ref, cfg)
    if not c.get("finite", False) or not c.get("fk_finite", False):
        reasons.append("nan_or_inf_or_fk_invalid")
    if float(c.get("root_y_range_m", 0.0)) > _v46_41_env_float("V46_41_KBO_ROOT_RANGE_ABS_MAX_M", 2.50):
        reasons.append("root_y_range_abs_exceeded")
    if abs(float(c.get("floor_y", 0.0)) - float(r.get("floor_y", 0.0))) > _v46_41_env_float("V46_41_KBO_FLOOR_SHIFT_MAX_M", 1.50):
        reasons.append("floor_shift_exceeded")
    if float(c.get("bone_length_violation_max_m", 0.0)) > _v46_41_env_float("V46_41_KBO_BONE_LENGTH_EPS_M", 0.02):
        reasons.append("bone_length_violation")
    if float(c.get("joint_acc_max", 0.0)) > _v46_41_env_float("V46_41_KBO_ACC_MAX", 3.0):
        reasons.append("acceleration_spike")
    if float(c.get("joint_jerk_max", 0.0)) > _v46_41_env_float("V46_41_KBO_JERK_MAX", 3.0):
        reasons.append("jerk_spike")
    if float(c.get("mean_joint_jerk_p95", c.get("joint_jerk_p95", 0.0))) > max(
        float(r.get("mean_joint_jerk_p95", r.get("joint_jerk_p95", 0.0))) * _v46_41_env_float("V46_41_KBO_JERK_RATIO", 2.5),
        float(r.get("mean_joint_jerk_p95", r.get("joint_jerk_p95", 0.0))) + _v46_41_env_float("V46_41_KBO_JERK_MARGIN", 0.15),
    ):
        reasons.append("jerk_p95_worse")
    if float(c.get("foot_skate_p95_mpf", 0.0)) > max(
        float(r.get("foot_skate_p95_mpf", 0.0)) * _v46_41_env_float("V46_41_KBO_SKATE_RATIO", 2.5),
        float(r.get("foot_skate_p95_mpf", 0.0)) + _v46_41_env_float("V46_41_KBO_SKATE_MARGIN", 0.06),
    ):
        reasons.append("skate_p95_worse")
    if float(c.get("foot_penetration_min_m", 0.0)) < float(r.get("foot_penetration_min_m", 0.0)) - _v46_41_env_float("V46_41_KBO_PENETRATION_MARGIN_M", 0.20):
        reasons.append("penetration_worse")
    if _v46_41_env_bool("V46_41_KBO_STAGE_ANCHOR_ENABLE", True):
        ae = _v46_41_anchor_error(cand, global_start)
        if ae > _v46_41_env_float("V46_41_KBO_ANCHOR_P95_MAX_M", 0.85):
            reasons.append("stage_anchor_deviation")
        c["stage_anchor_error_p95_m"] = ae
    return len(reasons) == 0, reasons, {"candidate": c, "reference": r, "stage": stage, "global_start": int(global_start)}


def _v46_41_save_hn_pair(stage, tx_id, snapshot, rejected, accepted, reasons, global_span):
    if not _v46_41_env_bool("V46_41_HN_DPO_SAVE_PAIRS", True):
        return {}
    root = Path(os.environ.get("V46_41_HN_DPO_DIR", "output/v46_41_hn_dpo_pairs"))
    root.mkdir(parents=True, exist_ok=True)
    tag = f"{stage}_tx{int(tx_id):04d}_{int(time.time()*1000)}"
    snap_p = root / f"{tag}_snapshot.npy"
    rej_p = root / f"{tag}_rejected.npy"
    acc_p = root / f"{tag}_accepted.npy"
    np.save(snap_p, np.asarray(snapshot, dtype=np.float32))
    np.save(rej_p, np.asarray(rejected, dtype=np.float32))
    np.save(acc_p, np.asarray(accepted, dtype=np.float32))
    meta = {"stage": stage, "transaction_id": int(tx_id), "span": list(map(int, global_span)), "snapshot": str(snap_p), "rejected": str(rej_p), "accepted": str(acc_p), "reasons": list(map(str, reasons))}
    with open(root / "pairs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(_v46_41_jsonable(meta), ensure_ascii=False) + "\n")
    return meta


def _v46_41_safe_residual(candidate, reference, seam_mask, cfg, stage="stage", global_start=0):
    cand = np.asarray(candidate, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    if cand.shape != ref.shape:
        return ref.astype(np.float32)
    sm = np.asarray(seam_mask, dtype=np.float32)
    if sm.ndim == 1:
        sm = sm[:, None]
    if sm.shape[0] != ref.shape[0]:
        sm = resample_motion_np(sm, ref.shape[0])
    core = _v46_41_env_float(f"V46_41_{stage.upper()}_CORE_COMMIT", 0.0)
    trans_default = 0.18 if stage == "refiner" else 0.12
    trans = _v46_41_env_float(f"V46_41_{stage.upper()}_TRANSITION_COMMIT", trans_default)
    w = np.clip(core + (trans - core) * sm.astype(np.float32), 0.0, 1.0)
    delta = cand - ref
    out = ref.copy().astype(np.float32)
    root_xz_max = _v46_41_env_float("V46_41_ROOT_XZ_DELTA_MAX_M", 0.05)
    root_y_max = _v46_41_env_float("V46_41_ROOT_Y_DELTA_MAX_M", 0.02)
    rot_max = _v46_41_env_float("V46_41_ROT6D_DELTA_MAX", 0.12)
    for idx, mx in [(ROOT_X_IDX, root_xz_max), (ROOT_Y_IDX, root_y_max), (ROOT_Z_IDX, root_xz_max)]:
        out[:, idx] = ref[:, idx] + np.clip(delta[:, idx], -mx, mx) * w[:, 0]
    out[:, ROT6D_START:ROT6D_END] = ref[:, ROT6D_START:ROT6D_END] + np.clip(delta[:, ROT6D_START:ROT6D_END], -rot_max, rot_max) * w
    out, _ = enforce_edge151_contract_np(out, cfg, source_hint=f"v46_41_safe_residual:{stage}", derive_contact=True, project_rot=True)
    out, _ = _v46_41_apply_stage_prior(out, cfg, strength=_v46_41_env_float("V46_41_MSA_TRANSACTION_STRENGTH", 0.08))
    ok, reasons, detail = _v46_41_kbo(out, ref, cfg, stage=f"{stage}_bounded_residual", global_start=global_start)
    if not ok:
        _v46_41_add_token({"mechanism": "KBO", "stage": stage, "event": "bounded_residual_rejected", "barrier_violations": reasons, "detail": detail, "hard_negative": True})
        return ref.astype(np.float32)
    return out.astype(np.float32)


def _v46_41_deterministic_bridge(reference, seam_mask, cfg, stage="fallback", global_start=0):
    ref = np.asarray(reference, dtype=np.float32).copy()
    if ref.shape[0] < 4:
        return ref.astype(np.float32), {"mode": "snapshot_too_short", "committed": False}
    sm = np.asarray(seam_mask, dtype=np.float32)
    if sm.ndim == 1:
        sm = sm[:, None]
    active = sm[:, 0] > _v46_41_env_float("V46_41_TGT_ACTIVE_THRESHOLD", 0.05)
    regs = contiguous_regions(active)
    if not regs:
        return ref.astype(np.float32), {"mode": "no_active_mask", "committed": False}
    out = ref.copy().astype(np.float32)
    fallback_strength = _v46_41_env_float("V46_41_DETERMINISTIC_FALLBACK_STRENGTH", 0.35)
    reports = []
    for a, b in regs:
        a = max(1, int(a)); b = min(int(b), ref.shape[0] - 1)
        if b - a < 2:
            continue
        n = b - a
        try:
            if "v46_33_motion_inbetween_np" in globals():
                bridge = v46_33_motion_inbetween_np(ref[max(0, a-2):a], ref[b:min(ref.shape[0], b+2)], n, cfg)
            else:
                raise RuntimeError("v46_33_motion_inbetween_np unavailable")
        except Exception:
            left = ref[a - 1].copy(); right = ref[b].copy()
            x = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
            cubic = x * x * (3.0 - 2.0 * x)
            bridge = (1.0 - cubic) * left[None] + cubic * right[None]
        idxs = [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX] + list(range(ROT6D_START, ROT6D_END))
        w = np.clip(sm[a:b], 0.0, 1.0) * float(fallback_strength)
        out[a:b, idxs] = out[a:b, idxs] * (1.0 - w) + bridge[:, idxs] * w
        reports.append({"span": [int(a), int(b)], "frames": int(n)})
    out, _ = enforce_edge151_contract_np(out, cfg, source_hint=f"v46_41_deterministic_bridge:{stage}", derive_contact=True, project_rot=True)
    out, _ = _v46_41_apply_stage_prior(out, cfg, strength=_v46_41_env_float("V46_41_MSA_FALLBACK_STRENGTH", 0.10))
    ok, reasons, detail = _v46_41_kbo(out, ref, cfg, stage=f"{stage}_deterministic_bridge", global_start=global_start)
    if not ok:
        return ref.astype(np.float32), {"mode": "deterministic_bridge_rejected", "committed": False, "reasons": reasons, "detail": detail}
    return out.astype(np.float32), {"mode": "deterministic_root_rotation_bridge", "committed": True, "regions": reports}


def _v46_41_regions(seam_mask, T):
    sm = np.asarray(seam_mask, dtype=np.float32)
    if sm.ndim == 1:
        sm = sm[:, None]
    active = sm[:, 0] > _v46_41_env_float("V46_41_TGT_ACTIVE_THRESHOLD", 0.05)
    raw = contiguous_regions(active)
    if not raw:
        return []
    halo = _v46_41_env_int("V46_41_TGT_HALO", 12)
    min_len = _v46_41_env_int("V46_41_TGT_MIN_FRAMES", 16)
    max_len = _v46_41_env_int("V46_41_TGT_MAX_FRAMES", 96)
    out = []
    for a, b in raw:
        a = max(0, int(a) - halo); b = min(int(T), int(b) + halo)
        if b - a < min_len:
            mid = (a + b) // 2
            a = max(0, mid - min_len // 2)
            b = min(int(T), a + min_len)
            a = max(0, b - min_len)
        while b - a > max_len:
            out.append((a, a + max_len))
            a = a + max_len - halo
        out.append((a, b))
    out.sort()
    merged = []
    for a, b in out:
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    return [(int(a), int(b)) for a, b in merged if int(b) > int(a)]


def _v46_41_diffusion_window_proposal(snapshot, cond, sm_win, ckpt_path, cfg, global_start=0):
    if torch is None or not ckpt_path or not Path(ckpt_path).exists():
        return _v46_41_orig_apply_diffusion_model(snapshot, cond, sm_win, ckpt_path, cfg)
    core_strength = _v46_41_env_float("V46_DIFFUSION_CORE_STRENGTH", 0.00)
    trans_strength = _v46_41_env_float("V46_DIFFUSION_TRANSITION_STRENGTH", 0.25)
    noise_scale = _v46_41_env_float("V46_DIFFUSION_REFERENCE_NOISE_SCALE", 0.01)
    ckpt = _v46_41_trusted_torch_load(ckpt_path, map_location=cfg.device)
    Tdiff = int(ckpt.get("diffusion_steps", cfg.diffusion_steps))
    model = DiffusionDenoiser(EDGE_DIM, 32).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    betas, alphas, abar = make_beta_schedule(Tdiff, torch.device(cfg.device))
    retr_in, _ = enforce_edge151_contract_np(np.asarray(snapshot, dtype=np.float32), cfg, source_hint="v46_41_diffusion_window_retrieval", derive_contact=True, project_rot=True)
    mask_in = np.asarray(sm_win, dtype=np.float32)
    if mask_in.ndim == 1:
        mask_in = mask_in[:, None]
    if mask_in.shape[0] != retr_in.shape[0]:
        mask_in = resample_motion_np(mask_in, retr_in.shape[0])
    abort_fraction = _v46_41_env_float("V46_41_DIFFUSION_EARLY_ABORT_FRACTION", 0.50)
    abort_t = int(round(Tdiff * abort_fraction))
    with torch.no_grad():
        retr = torch.from_numpy(retr_in[None]).float().to(cfg.device)
        raw_mask = torch.from_numpy(mask_in[None].astype(np.float32)).float().to(cfg.device)
        mask = torch.clamp(float(core_strength) + (float(trans_strength) - float(core_strength)) * raw_mask, 0.0, 1.0)
        c = torch.from_numpy(cond[None].astype(np.float32)).float().to(cfg.device)
        x = retr + float(noise_scale) * torch.randn_like(retr) * (0.15 + 0.85 * mask)
        checked = False
        for ti in reversed(range(Tdiff)):
            t = torch.full((1,), ti, device=cfg.device, dtype=torch.long)
            eps = model(x, retr, c, raw_mask, t)
            beta = betas[ti]; alpha = alphas[ti]; ab = abar[ti]
            mean = (1 / torch.sqrt(alpha)) * (x - beta / torch.sqrt(1 - ab).clamp_min(1e-6) * eps)
            if ti > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x) * 0.35
            else:
                x = mean
            x = retr * (1.0 - mask) + x * mask
            if (not checked) and ti <= abort_t:
                probe = x[0].detach().cpu().numpy().astype(np.float32)
                probe, _ = enforce_edge151_contract_np(probe, cfg, source_hint="v46_41_diffusion_early_abort_probe", derive_contact=True, project_rot=True)
                probe = _v46_41_safe_residual(probe, retr_in, mask_in, cfg, stage="diffusion", global_start=global_start)
                ok, reasons, detail = _v46_41_kbo(probe, retr_in, cfg, stage="diffusion_early_abort_probe", global_start=global_start)
                checked = True
                if not ok:
                    _v46_41_add_token({"mechanism": "early_abort", "stage": "diffusion", "commit_state": "abort_to_ccd", "barrier_violations": reasons, "detail": detail, "hard_negative": True})
                    raise RuntimeError("diffusion_early_abort:" + ",".join(reasons))
        y = x[0].detach().cpu().numpy().astype(np.float32)
    y, _ = enforce_edge151_contract_np(y, cfg, source_hint="v46_41_diffusion_window_output", derive_contact=True, project_rot=True)
    return y.astype(np.float32)


def _v46_41_apply_stage(stage, orig_func, motion, cond, seam_mask, ckpt_path, cfg):
    if not _v46_41_env_bool("V46_41_TGT_ENABLE", True):
        cand = orig_func(motion, cond, seam_mask, ckpt_path, cfg)
        return _v46_41_safe_residual(cand, motion, seam_mask, cfg, stage=stage, global_start=0)
    ref_all = np.asarray(motion, dtype=np.float32)
    out = ref_all.copy().astype(np.float32)
    regions = _v46_41_regions(seam_mask, ref_all.shape[0])
    if not regions:
        _v46_41_add_token({"mechanism": "TGT", "stage": stage, "event": "no_transaction_regions", "commit_state": "return_reference"})
        return out.astype(np.float32)
    for tx_id, (a, b) in enumerate(regions):
        snapshot = out[a:b].copy().astype(np.float32)
        sm_win = np.asarray(seam_mask[a:b], dtype=np.float32).copy()
        token = {"mechanism": "TGT+KBO", "stage": stage, "temporal_transaction_id": int(tx_id), "atomic_window": [int(a), int(b)], "frames": int(b-a), "commit_state": "pending"}
        rejected_candidate = None
        try:
            if stage == "diffusion" and _v46_41_env_bool("V46_41_DIFFUSION_EARLY_ABORT_ENABLE", True):
                cand = _v46_41_diffusion_window_proposal(snapshot.copy(), cond, sm_win, ckpt_path, cfg, global_start=a)
            else:
                cand = orig_func(snapshot.copy(), cond, sm_win, ckpt_path, cfg)
            rejected_candidate = np.asarray(cand, dtype=np.float32)
            cand = _v46_41_safe_residual(cand, snapshot, sm_win, cfg, stage=stage, global_start=a)
            ok, reasons, detail = _v46_41_kbo(cand, snapshot, cfg, stage=f"{stage}_neural_commit", global_start=a)
            if ok:
                out[a:b] = cand.astype(np.float32)
                token.update({"commit_state": "committed", "fallback_level": "neural_bounded_commit", "kbo_status": "pass", "hard_negative": False})
            else:
                token.update({"commit_state": "neural_rejected", "kbo_status": "fail", "barrier_violations": reasons, "detail": detail, "hard_negative": True})
                raise RuntimeError("kbo_reject:" + ",".join(reasons))
        except Exception as exc:
            token["neural_exception"] = str(exc)[:500]
            fb, fb_report = _v46_41_deterministic_bridge(snapshot, sm_win, cfg, stage=stage, global_start=a)
            if fb_report.get("committed"):
                out[a:b] = fb.astype(np.float32)
                token.update({"commit_state": "committed", "fallback_level": "deterministic_root_rotation_prior", "kbo_status": "fallback_pass", "fallback_report": fb_report, "hard_negative": True})
                if rejected_candidate is not None:
                    token["hn_dpo_pair"] = _v46_41_save_hn_pair(stage, tx_id, snapshot, rejected_candidate, fb, token.get("barrier_violations", [str(exc)]), [a, b])
            else:
                out[a:b] = snapshot.astype(np.float32)
                token.update({"commit_state": "rolled_back", "fallback_level": "snapshot_rollback", "kbo_status": "fallback_fail", "fallback_report": fb_report, "hard_negative": True})
                if rejected_candidate is not None:
                    token["hn_dpo_pair"] = _v46_41_save_hn_pair(stage, tx_id, snapshot, rejected_candidate, snapshot, token.get("barrier_violations", [str(exc)]), [a, b])
        _v46_41_add_token(token)
    out, _ = enforce_edge151_contract_np(out, cfg, source_hint=f"v46_41_tgt_final:{stage}", derive_contact=True, project_rot=True)
    out, _ = _v46_41_apply_stage_prior(out, cfg, strength=_v46_41_env_float("V46_41_MSA_STAGE_FINAL_STRENGTH", 0.08))
    ok, reasons, detail = _v46_41_kbo(out, ref_all, cfg, stage=f"{stage}_whole_stage_guard", global_start=0)
    if not ok:
        _v46_41_add_token({"mechanism": "KBO", "stage": stage, "event": "whole_stage_rollback", "commit_state": "rolled_back", "barrier_violations": reasons, "detail": detail, "hard_negative": True})
        return ref_all.astype(np.float32)
    return out.astype(np.float32)


def apply_refiner_model(motion, cond, seam_mask, ckpt_path, cfg):
    return _v46_41_apply_stage("refiner", _v46_41_orig_apply_refiner_model, motion, cond, seam_mask, ckpt_path, cfg)


def apply_diffusion_model(motion, cond, seam_mask, ckpt_path, cfg):
    return _v46_41_apply_stage("diffusion", _v46_41_orig_apply_diffusion_model, motion, cond, seam_mask, ckpt_path, cfg)


def true_lower_body_ik(motion, cfg):
    if not _v46_41_env_bool("V46_41_IK_TGT_ENABLE", True):
        return _v46_41_orig_true_lower_body_ik(motion, cfg)
    snapshot = np.asarray(motion, dtype=np.float32).copy()
    try:
        out, report = _v46_41_orig_true_lower_body_ik(snapshot.copy(), cfg)
        out, _ = _v46_41_apply_stage_prior(out, cfg, strength=_v46_41_env_float("V46_41_MSA_IK_STRENGTH", 0.04))
        ok, reasons, detail = _v46_41_kbo(out, snapshot, cfg, stage="ik_final", global_start=0)
        if ok:
            _v46_41_add_token({"mechanism": "IK_TGT", "stage": "ik", "commit_state": "committed", "fallback_level": "ik_commit", "kbo_status": "pass", "frames": int(snapshot.shape[0])})
            return out.astype(np.float32), report
        _v46_41_add_token({"mechanism": "IK_TGT", "stage": "ik", "commit_state": "rolled_back", "fallback_level": "fk_snapshot_rollback", "barrier_violations": reasons, "detail": detail, "hard_negative": True})
        try:
            report = dict(report)
            report["v46_41_ik_rollback_to_fk"] = True
            report["v46_41_rollback_reasons"] = reasons
        except Exception:
            pass
        return snapshot.astype(np.float32), report
    except Exception as exc:
        _v46_41_add_token({"mechanism": "IK_TGT", "stage": "ik", "commit_state": "rolled_back", "fallback_level": "ik_exception_to_fk", "exception": str(exc)[:500], "hard_negative": True})
        return snapshot.astype(np.float32), {"enabled": True, "v46_41_ik_exception_to_fk": True, "exception": str(exc)[:500]}


def _v46_41_summary(records):
    out = {"version": "v46_41_stage_anchored_guided_tgt_kbo", "num_records": int(len(records)), "by_stage": {}, "fallback_counts": {}, "hard_negatives": 0}
    for r in records:
        st = str(r.get("stage", "unknown"))
        out["by_stage"].setdefault(st, {"records": 0, "committed": 0, "rolled_back": 0})
        out["by_stage"][st]["records"] += 1
        cs = str(r.get("commit_state", ""))
        if cs == "committed":
            out["by_stage"][st]["committed"] += 1
        elif cs in ("rolled_back", "neural_rejected"):
            out["by_stage"][st]["rolled_back"] += 1
        fb = str(r.get("fallback_level", r.get("commit_state", "unknown")))
        out["fallback_counts"][fb] = out["fallback_counts"].get(fb, 0) + 1
        if bool(r.get("hard_negative", False)):
            out["hard_negatives"] += 1
    out["stage_anchor"] = _v46_41_jsonable(_V46_41_STAGE_PRIOR_META)
    return out


def generate(args):
    _v46_41_reset_audit()
    rc = int(_v46_41_orig_generate(args))
    try:
        out_path = Path(args.out)
        json_path = Path(args.json or str(out_path).replace(".npy", ".v46_33_report.json"))
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            report.setdefault("stage_reports", {})["v46_41_temporal_generative_transactions"] = _V46_41_AUDIT_TOKENS
            report["v46_41_tgt_kbo_summary"] = _v46_41_summary(_V46_41_AUDIT_TOKENS)
            report["v46_41_scientific_mechanism"] = {
                "name": "Stage-Anchored KBO-guided Temporal Generative Transactions",
                "problem": "long-horizon covariate shift and topological fragility",
                "mechanisms": ["Macroscopic Stage Anchoring", "Temporal Generative Transactions", "Kinematic Barrier Oracle", "Confidence-aware Cascaded Degradation", "Diffusion Early-Abort", "Hard-negative Audit Tokens"],
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(_v46_41_jsonable(report), f, ensure_ascii=False, indent=2)
            print(json.dumps({"v46_41_tgt_kbo_summary": report["v46_41_tgt_kbo_summary"], "json_updated": str(json_path)}, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[V46.41 WARN] failed to append audit tokens: {exc}", file=sys.stderr)
    return rc
# ===== V46.41 STAGE-ANCHORED GUIDED TGT PATCH END =====
'''


def _strip_block(text: str, start: str, end: str) -> str:
    while start in text and end in text:
        pre = text.split(start, 1)[0]
        post = text.split(end, 1)[1]
        text = pre + post
    return text


def main() -> int:
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)
    text = TARGET.read_text(encoding="utf-8")
    for s, e in OLD_GUARDS:
        text = _strip_block(text, s, e)
    text = _strip_block(text, START, END)
    marker = 'if __name__ == "__main__":'
    idx = text.rfind(marker)
    if idx < 0:
        raise RuntimeError('Could not find final if __name__ == "__main__" marker')
    new_text = text[:idx] + PATCH + "\n\n" + text[idx:]
    backup = TARGET.with_suffix(TARGET.suffix + f".v46_41_tgt_{time.strftime('%Y%m%d_%H%M%S')}.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(new_text, encoding="utf-8")
    print(f"[BAK] {backup}")
    print(f"[OK] patched {TARGET} with V46.41 stage-anchored guided TGT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
