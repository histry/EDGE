#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply V46.42 stability-alignment fixes on top of V46.41.

Run order:
  1) apply_v46_38_complete_routing_patch.py
  2) apply_v46_41_stage_anchor_guided_tgt_patch.py
  3) apply_v46_42_stability_alignment_patch.py

V46.42 addresses three scientific/engineering loopholes:
- Tweedie jitter false positives: early-abort KBO uses low-pass probe and
  relaxed early thresholds instead of final strict jerk thresholds.
- MSA rubber-band effect: stage anchoring is dynamically weakened for high-energy
  / climax / percussive / leap-like windows and for high root-velocity snippets.
- HN-DPO static mode collapse: handled by v46_42_train_hn_dpo_diffusion.py with
  kinetic-energy preservation.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

TARGET = Path("tools/v46_motionrag_diff.py")
START = "# ===== V46.42 STABILITY ALIGNMENT PATCH START ====="
END = "# ===== V46.42 STABILITY ALIGNMENT PATCH END ====="

PATCH = r'''
# ===== V46.42 STABILITY ALIGNMENT PATCH START =====
# Fixes for V46.41 scientific loopholes:
# 1) Tweedie jitter false positives: early-abort probe is low-pass filtered and
#    checked with relaxed early thresholds.
# 2) Rubber-band MSA: stage-anchor strength is modulated by MSSD energy/role and
#    local root velocity; high-energy leaps are not over-constrained.
# 3) Audit exposes V46.42 policy; kinetic HN-DPO is implemented in the separate
#    v46_42_train_hn_dpo_diffusion.py tool.

_v46_42_orig_generate = generate
_v46_42_orig_apply_stage_prior = _v46_41_apply_stage_prior
_v46_42_orig_safe_residual = _v46_41_safe_residual
_v46_42_orig_deterministic_bridge = _v46_41_deterministic_bridge
_v46_42_orig_diffusion_window_proposal = _v46_41_diffusion_window_proposal
_v46_42_orig_anchor_error = _v46_41_anchor_error

_V46_42_FRAME_MSA_WEIGHT = None
_V46_42_MSSD_WEIGHT_META = {}


def _v46_42_env_bool(name, default=True):
    try:
        return bool(int(os.environ.get(name, "1" if default else "0")))
    except Exception:
        return bool(default)


def _v46_42_env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _v46_42_env_int(name, default):
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def _v46_42_slot_energy_weight(slot):
    """Return anchor weight in [min,max]; lower weight means freer movement."""
    label = " ".join([
        str(slot.get("music_event", "")),
        str(slot.get("music_alignment_label", "")),
        str(slot.get("music_semantic_top_label", "")),
        str(slot.get("role", "")),
        str(slot.get("slot_role", "")),
        str(slot.get("energy_label", "")),
        str(slot.get("predicted_motion_event", "")),
    ]).lower()
    high_words = ("climax", "turn", "percussive", "accent", "footwork", "leap", "jump", "build", "high")
    calm_words = ("calm", "meditative", "pose", "hold", "sustain", "resolution", "intro")
    energy = float(slot.get("energy", slot.get("boundary_accent_strength", 0.0)) or 0.0)
    tension = float(slot.get("tension", 0.0) or 0.0)
    speed = float(slot.get("music_speed_factor", 1.0) or 1.0)
    w = _v46_42_env_float("V46_42_MSA_CALM_WEIGHT", 1.0)
    if any(x in label for x in high_words):
        w *= _v46_42_env_float("V46_42_MSA_HIGH_ENERGY_SCALE", 0.22)
    elif any(x in label for x in calm_words):
        w *= _v46_42_env_float("V46_42_MSA_CALM_SCALE", 1.00)
    # Continuous attenuation from music dynamics.
    dyn = max(0.0, min(1.0, 0.60 * energy + 0.30 * tension + 0.10 * max(0.0, speed - 1.0)))
    w *= (1.0 - dyn * _v46_42_env_float("V46_42_MSA_DYNAMIC_ATTENUATION", 0.75))
    return float(np.clip(w, _v46_42_env_float("V46_42_MSA_MIN_WEIGHT", 0.05), _v46_42_env_float("V46_42_MSA_MAX_WEIGHT", 1.0)))


def _v46_42_load_mssd_stage_weights(slots_json, total_frames_hint=0):
    global _V46_42_FRAME_MSA_WEIGHT, _V46_42_MSSD_WEIGHT_META
    _V46_42_FRAME_MSA_WEIGHT = None
    _V46_42_MSSD_WEIGHT_META = {"enabled": False, "reason": "no_slots_json"}
    if not slots_json:
        return
    try:
        p = Path(slots_json)
        if not p.exists():
            _V46_42_MSSD_WEIGHT_META = {"enabled": False, "reason": f"missing:{slots_json}"}
            return
        obj = json.load(open(p, "r", encoding="utf-8"))
        slots = obj.get("slots", []) if isinstance(obj, dict) else []
        if not isinstance(slots, list) or not slots:
            _V46_42_MSSD_WEIGHT_META = {"enabled": False, "reason": "no_slots"}
            return
        total = int(obj.get("total_target_frames", 0) or 0)
        if total <= 0:
            total = max(int(s.get("end_frame", 0) or 0) for s in slots)
        if total <= 0:
            total = int(total_frames_hint or 0)
        if total <= 0:
            _V46_42_MSSD_WEIGHT_META = {"enabled": False, "reason": "invalid_total_frames"}
            return
        w = np.ones((total,), dtype=np.float32)
        hist = {}
        for i, s in enumerate(slots):
            a = int(s.get("start_frame", 0) or 0)
            b = int(s.get("end_frame", a + int(s.get("target_frames", 0) or 0)) or a)
            if b <= a:
                b = a + int(s.get("target_frames", 1) or 1)
            a = max(0, min(total, a)); b = max(a, min(total, b))
            sw = _v46_42_slot_energy_weight(s)
            if b > a:
                w[a:b] = sw
            key = str(s.get("music_semantic_top_label", s.get("music_event", "unknown")))
            hist[key] = hist.get(key, 0) + 1
        if ndi is not None and len(w) > 7:
            w = ndi.gaussian_filter1d(w, sigma=float(_v46_42_env_float("V46_42_MSA_WEIGHT_SMOOTH_SIGMA", 3.0)), mode="nearest").astype(np.float32)
        _V46_42_FRAME_MSA_WEIGHT = np.clip(w, _v46_42_env_float("V46_42_MSA_MIN_WEIGHT", 0.05), 1.0).astype(np.float32)
        _V46_42_MSSD_WEIGHT_META = {
            "enabled": True,
            "source": str(p),
            "total_frames": int(total),
            "min": float(np.min(_V46_42_FRAME_MSA_WEIGHT)),
            "mean": float(np.mean(_V46_42_FRAME_MSA_WEIGHT)),
            "p95": float(np.percentile(_V46_42_FRAME_MSA_WEIGHT, 95)),
            "semantic_histogram": hist,
            "interpretation": "lower weights indicate high-energy/climax windows where MSA is relaxed",
        }
    except Exception as exc:
        _V46_42_FRAME_MSA_WEIGHT = None
        _V46_42_MSSD_WEIGHT_META = {"enabled": False, "reason": str(exc)}


def _v46_42_frame_weights(T, global_start=0):
    if _V46_42_FRAME_MSA_WEIGHT is None or T <= 0:
        return np.ones((int(T), 1), dtype=np.float32)
    a = int(global_start)
    b = a + int(T)
    if a < 0 or b > len(_V46_42_FRAME_MSA_WEIGHT):
        # Defensive resize for rare report/motion length drifts.
        idx = np.linspace(0, len(_V46_42_FRAME_MSA_WEIGHT) - 1, int(T)).clip(0, len(_V46_42_FRAME_MSA_WEIGHT) - 1).astype(int)
        return _V46_42_FRAME_MSA_WEIGHT[idx, None].astype(np.float32)
    return _V46_42_FRAME_MSA_WEIGHT[a:b, None].astype(np.float32)


def _v46_42_velocity_gate(motion):
    m = np.asarray(motion, dtype=np.float32)
    if m.shape[0] < 3:
        return np.ones((m.shape[0], 1), dtype=np.float32)
    v = np.linalg.norm(np.diff(m[:, [ROOT_X_IDX, ROOT_Z_IDX]], axis=0), axis=-1)
    v = np.concatenate([[v[0]], v]).astype(np.float32)
    thr = _v46_42_env_float("V46_42_MSA_ROOT_SPEED_RELAX_THRESH", 0.045)
    if thr <= 0:
        return np.ones((m.shape[0], 1), dtype=np.float32)
    # High root speed means possible leap/large travel; attenuate anchor.
    g = 1.0 / (1.0 + (v / max(thr, 1e-6)) ** 2)
    g = np.clip(g, _v46_42_env_float("V46_42_MSA_VELOCITY_MIN_GATE", 0.12), 1.0)
    if ndi is not None and len(g) > 7:
        g = ndi.gaussian_filter1d(g, sigma=2.0, mode="nearest")
    return g[:, None].astype(np.float32)


def _v46_41_apply_stage_prior(motion, cfg, strength=None, global_start=0):
    """Dynamic MSA: high-energy/leap windows receive weaker anchoring."""
    global _V46_41_STAGE_PRIOR_XZ
    if not _v46_41_env_bool("V46_41_MSA_ENABLE", True):
        return np.asarray(motion, dtype=np.float32), {"enabled": False}
    m = np.asarray(motion, dtype=np.float32).copy()
    prior = _V46_41_STAGE_PRIOR_XZ
    if prior is None or len(prior) < int(global_start) + len(m):
        prior_local, meta = _v46_41_build_stage_prior_xz(m, None, cfg)
    else:
        prior_local = prior[int(global_start):int(global_start)+len(m)]
        meta = dict(_V46_41_STAGE_PRIOR_META)
    base_alpha = _v46_41_env_float("V46_41_MSA_COMMIT_STRENGTH", 0.16) if strength is None else float(strength)
    frame_w = _v46_42_frame_weights(len(m), global_start=global_start)
    vel_gate = _v46_42_velocity_gate(m)
    dyn_w = np.clip(frame_w * vel_gate, _v46_42_env_float("V46_42_MSA_MIN_WEIGHT", 0.05), 1.0)
    max_delta = _v46_41_env_float("V46_41_MSA_MAX_DELTA_M", 0.06)
    delta = np.clip(prior_local - m[:, [ROOT_X_IDX, ROOT_Z_IDX]], -max_delta, max_delta)
    m[:, ROOT_X_IDX] = m[:, ROOT_X_IDX] + float(base_alpha) * dyn_w[:, 0] * delta[:, 0]
    m[:, ROOT_Z_IDX] = m[:, ROOT_Z_IDX] + float(base_alpha) * dyn_w[:, 0] * delta[:, 1]
    m, _ = enforce_edge151_contract_np(m, cfg, source_hint="v46_42_dynamic_msa_apply_stage_prior", derive_contact=True, project_rot=True)
    meta.update({
        "applied": True,
        "version": "v46_42_dynamic_music_energy_msa",
        "base_strength": float(base_alpha),
        "effective_strength_mean": float(base_alpha * float(np.mean(dyn_w))),
        "effective_strength_min": float(base_alpha * float(np.min(dyn_w))),
        "max_delta_m": float(max_delta),
        "mssd_weight_meta": _v46_42_jsonable(_V46_42_MSSD_WEIGHT_META) if "_v46_42_jsonable" in globals() else str(_V46_42_MSSD_WEIGHT_META),
    })
    return m.astype(np.float32), meta


def _v46_41_anchor_error(candidate, a0=0):
    """Music/velocity weighted anchor error to avoid high-energy rubber-band rejection."""
    global _V46_41_STAGE_PRIOR_XZ
    cand = np.asarray(candidate, dtype=np.float32)
    if _V46_41_STAGE_PRIOR_XZ is None:
        return 0.0
    a = int(a0); b = a + len(cand)
    if a < 0 or b > len(_V46_41_STAGE_PRIOR_XZ):
        return 0.0
    prior = _V46_41_STAGE_PRIOR_XZ[a:b]
    err = np.linalg.norm(cand[:, [ROOT_X_IDX, ROOT_Z_IDX]] - prior, axis=-1)
    w = _v46_42_frame_weights(len(cand), a)[:, 0]
    vg = _v46_42_velocity_gate(cand)[:, 0]
    weighted = err * np.clip(w * vg, _v46_42_env_float("V46_42_MSA_MIN_WEIGHT", 0.05), 1.0)
    return float(np.percentile(weighted, 95))


def _v46_42_jsonable(x):
    try:
        return _v46_41_jsonable(x)
    except Exception:
        if isinstance(x, dict):
            return {str(k): _v46_42_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_v46_42_jsonable(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        return x if isinstance(x, (str, int, float, bool)) or x is None else str(x)


def _v46_42_lowpass_motion_for_kbo(motion, cfg, sigma=None):
    """Low-pass a Tweedie/intermediate probe before high-order KBO.

    This prevents high-frequency residual noise from creating false positive jerk
    spikes during early-abort checks. The committed sample is not replaced by
    this smoothed probe; smoothing is only for the oracle decision.
    """
    m = np.asarray(motion, dtype=np.float32).copy()
    if sigma is None:
        sigma = _v46_42_env_float("V46_42_EARLY_ABORT_KBO_SMOOTH_SIGMA", 1.35)
    if ndi is not None and m.shape[0] > 5 and float(sigma) > 0:
        idx = [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX] + list(range(ROT6D_START, ROT6D_END))
        m[:, idx] = ndi.gaussian_filter1d(m[:, idx], sigma=float(sigma), axis=0, mode="nearest")
    m, _ = enforce_edge151_contract_np(m, cfg, source_hint="v46_42_lowpass_tweedie_probe", derive_contact=True, project_rot=True)
    return m.astype(np.float32)


def _v46_42_kbo_early_abort(candidate, reference, cfg, stage="diffusion_early_abort_probe", global_start=0):
    """Relaxed KBO for intermediate diffusion probes.

    Final KBO remains strict. Early probes are smoothed and use a larger barrier
    margin because x_t / provisional x0 contains residual denoising jitter.
    """
    raw = np.asarray(candidate, dtype=np.float32)
    smooth = _v46_42_lowpass_motion_for_kbo(raw, cfg)
    ref = np.asarray(reference, dtype=np.float32)
    reasons = []
    relax = _v46_42_env_float("V46_42_EARLY_ABORT_KBO_RELAX", 3.0)
    c = _v46_41_kinematic_stats(smooth, cfg)
    r = _v46_41_kinematic_stats(ref, cfg)
    if not c.get("finite", False) or not c.get("fk_finite", False):
        reasons.append("nan_or_inf_or_fk_invalid")
    if float(c.get("root_y_range_m", 0.0)) > _v46_41_env_float("V46_41_KBO_ROOT_RANGE_ABS_MAX_M", 2.50) * max(1.0, relax * 0.75):
        reasons.append("root_y_range_abs_exceeded")
    if abs(float(c.get("floor_y", 0.0)) - float(r.get("floor_y", 0.0))) > _v46_41_env_float("V46_41_KBO_FLOOR_SHIFT_MAX_M", 1.50) * max(1.0, relax):
        reasons.append("floor_shift_exceeded")
    if float(c.get("bone_length_violation_max_m", 0.0)) > _v46_41_env_float("V46_41_KBO_BONE_LENGTH_EPS_M", 0.02) * max(1.0, relax):
        reasons.append("bone_length_violation")
    if float(c.get("joint_acc_max", 0.0)) > _v46_41_env_float("V46_41_KBO_ACC_MAX", 3.0) * max(1.0, relax):
        reasons.append("acceleration_spike")
    if float(c.get("joint_jerk_max", 0.0)) > _v46_41_env_float("V46_41_KBO_JERK_MAX", 3.0) * max(1.0, relax):
        reasons.append("jerk_spike")
    # Anchor check is also weighted by dynamic MSA; do not reject high-energy windows solely due to anchor.
    if _v46_41_env_bool("V46_41_KBO_STAGE_ANCHOR_ENABLE", True):
        ae = _v46_41_anchor_error(smooth, global_start)
        if ae > _v46_41_env_float("V46_41_KBO_ANCHOR_P95_MAX_M", 0.85) * max(1.0, relax):
            reasons.append("stage_anchor_deviation")
        c["stage_anchor_error_p95_m"] = ae
    detail = {"candidate_smoothed": c, "reference": r, "raw_probe_shape": list(raw.shape), "kbo_mode": "early_abort_lowpass_relaxed", "relax": float(relax), "stage": stage, "global_start": int(global_start)}
    return len(reasons) == 0, reasons, detail


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
    out, _ = enforce_edge151_contract_np(out, cfg, source_hint=f"v46_42_safe_residual:{stage}", derive_contact=True, project_rot=True)
    out, _ = _v46_41_apply_stage_prior(out, cfg, strength=_v46_41_env_float("V46_41_MSA_TRANSACTION_STRENGTH", 0.08), global_start=global_start)
    ok, reasons, detail = _v46_41_kbo(out, ref, cfg, stage=f"{stage}_bounded_residual", global_start=global_start)
    if not ok:
        _v46_41_add_token({"mechanism": "KBO", "version": "v46_42", "stage": stage, "event": "bounded_residual_rejected", "barrier_violations": reasons, "detail": detail, "hard_negative": True})
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
    out, _ = enforce_edge151_contract_np(out, cfg, source_hint=f"v46_42_deterministic_bridge:{stage}", derive_contact=True, project_rot=True)
    out, _ = _v46_41_apply_stage_prior(out, cfg, strength=_v46_41_env_float("V46_41_MSA_FALLBACK_STRENGTH", 0.10), global_start=global_start)
    ok, reasons, detail = _v46_41_kbo(out, ref, cfg, stage=f"{stage}_deterministic_bridge", global_start=global_start)
    if not ok:
        return ref.astype(np.float32), {"mode": "deterministic_bridge_rejected", "committed": False, "reasons": reasons, "detail": detail}
    return out.astype(np.float32), {"mode": "deterministic_root_rotation_bridge", "committed": True, "regions": reports, "v46_42_dynamic_msa": True}


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
    retr_in, _ = enforce_edge151_contract_np(np.asarray(snapshot, dtype=np.float32), cfg, source_hint="v46_42_diffusion_window_retrieval", derive_contact=True, project_rot=True)
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
                probe, _ = enforce_edge151_contract_np(probe, cfg, source_hint="v46_42_diffusion_early_abort_probe_raw", derive_contact=True, project_rot=True)
                # Apply bounded residual first, then low-pass/relaxed KBO to avoid Tweedie jitter false positives.
                probe_bounded = _v46_41_safe_residual(probe, retr_in, mask_in, cfg, stage="diffusion", global_start=global_start)
                ok, reasons, detail = _v46_42_kbo_early_abort(probe_bounded, retr_in, cfg, stage="diffusion_early_abort_probe", global_start=global_start)
                checked = True
                if not ok:
                    _v46_41_add_token({"mechanism": "early_abort", "version": "v46_42_lowpass_relaxed", "stage": "diffusion", "commit_state": "abort_to_ccd", "barrier_violations": reasons, "detail": detail, "hard_negative": True})
                    raise RuntimeError("diffusion_early_abort_v46_42:" + ",".join(reasons))
        y = x[0].detach().cpu().numpy().astype(np.float32)
    y, _ = enforce_edge151_contract_np(y, cfg, source_hint="v46_42_diffusion_window_output", derive_contact=True, project_rot=True)
    return y.astype(np.float32)


def generate(args):
    try:
        _v46_42_load_mssd_stage_weights(getattr(args, "slots_json", None))
    except Exception as exc:
        print(f"[V46.42 WARN] failed to load MSSD dynamic stage weights: {exc}", file=sys.stderr)
    rc = int(_v46_42_orig_generate(args))
    try:
        out_path = Path(args.out)
        json_path = Path(args.json or str(out_path).replace(".npy", ".v46_33_report.json"))
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            report["v46_42_stability_alignment"] = {
                "version": "v46_42_lowpass_early_abort_dynamic_msa_kinetic_hn_dpo",
                "fixes": [
                    "low-pass relaxed KBO for early-abort probes",
                    "music-energy and root-velocity adaptive macroscopic stage anchoring",
                    "kinetic-energy preserving HN-DPO fine-tuning tool",
                ],
                "mssd_stage_weight_meta": _v46_42_jsonable(_V46_42_MSSD_WEIGHT_META),
                "early_abort_relax": float(_v46_42_env_float("V46_42_EARLY_ABORT_KBO_RELAX", 3.0)),
                "early_abort_smooth_sigma": float(_v46_42_env_float("V46_42_EARLY_ABORT_KBO_SMOOTH_SIGMA", 1.35)),
            }
            mech = report.get("v46_41_scientific_mechanism", {})
            if isinstance(mech, dict):
                mech.setdefault("v46_42_fixes", report["v46_42_stability_alignment"]["fixes"])
                report["v46_41_scientific_mechanism"] = mech
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(_v46_42_jsonable(report), f, ensure_ascii=False, indent=2)
            print(json.dumps({"v46_42_stability_alignment": report["v46_42_stability_alignment"], "json_updated": str(json_path)}, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[V46.42 WARN] failed to append stability-alignment metadata: {exc}", file=sys.stderr)
    return rc
# ===== V46.42 STABILITY ALIGNMENT PATCH END =====
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
    if "# ===== V46.41 STAGE-ANCHORED GUIDED TGT PATCH START =====" not in text:
        raise RuntimeError("V46.41 patch block was not found. Run apply_v46_41_stage_anchor_guided_tgt_patch.py first.")
    text = _strip_block(text, START, END)
    marker = 'if __name__ == "__main__":'
    idx = text.rfind(marker)
    if idx < 0:
        raise RuntimeError('Could not find final if __name__ == "__main__" marker')
    new_text = text[:idx] + PATCH + "\n\n" + text[idx:]
    backup = TARGET.with_suffix(TARGET.suffix + f".v46_42_stability_{time.strftime('%Y%m%d_%H%M%S')}.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(new_text, encoding="utf-8")
    print(f"[BAK] {backup}")
    print(f"[OK] patched {TARGET} with V46.42 stability-alignment fixes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
