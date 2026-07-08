#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply V46.43 Physics-Consistent Stability patch.

Run order:
  1) apply_v46_38_complete_routing_patch.py
  2) apply_v46_41_stage_anchor_guided_tgt_patch.py
  3) apply_v46_42_stability_alignment_patch.py
  4) apply_v46_43_physics_consistent_stability_patch.py

V46.43 is a stricter scientific correction to V46.42:
- Derivative-safe early-abort: low-pass + robust derivative oracle; derivative-only
  spikes cannot abort unless accompanied by low-frequency/fatal barriers; optional
  consecutive-probe requirement avoids Tweedie noise false positives.
- Velocity-preserving stage anchoring: MSA applies only low-frequency drift
  correction, gates out leap/high-root-speed windows, and caps correction velocity,
  preventing rubber-band moonwalk.
- Kinetic/motion-density preserving HN-DPO is implemented in
  v46_43_train_hn_dpo_diffusion.py.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

TARGET = Path("tools/v46_motionrag_diff.py")
START = "# ===== V46.43 PHYSICS-CONSISTENT STABILITY PATCH START ====="
END = "# ===== V46.43 PHYSICS-CONSISTENT STABILITY PATCH END ====="

PATCH = r'''
# ===== V46.43 PHYSICS-CONSISTENT STABILITY PATCH START =====
# Physics-consistent fixes after V46.42.
# This block intentionally redefines V46.41/V46.42 runtime functions because the
# generation code resolves them by global name at call time.

_v46_43_orig_generate = generate

_V46_43_EARLY_ABORT_TRACE = []


def _v46_43_env_bool(name, default=True):
    try:
        return bool(int(os.environ.get(name, "1" if default else "0")))
    except Exception:
        return bool(default)


def _v46_43_env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _v46_43_env_int(name, default):
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def _v46_43_jsonable(x):
    try:
        return _v46_42_jsonable(x)
    except Exception:
        if isinstance(x, dict):
            return {str(k): _v46_43_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_v46_43_jsonable(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        return x if isinstance(x, (str, int, float, bool)) or x is None else str(x)


def _v46_43_lowpass_channels(motion, cfg, sigma=2.25):
    """Only used for oracle decisions, never as committed sample."""
    m = np.asarray(motion, dtype=np.float32).copy()
    if ndi is not None and m.ndim == 2 and m.shape[0] > 7 and float(sigma) > 0:
        idx = [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX] + list(range(ROT6D_START, ROT6D_END))
        m[:, idx] = ndi.gaussian_filter1d(m[:, idx], sigma=float(sigma), axis=0, mode="nearest")
    m, _ = enforce_edge151_contract_np(m, cfg, source_hint="v46_43_derivative_safe_lowpass_probe", derive_contact=True, project_rot=True)
    return m.astype(np.float32)


def _v46_43_robust_derivative_stats(motion, cfg):
    """Robust derivative statistics after low-pass filtering.

    Uses p95/p99 rather than raw max to avoid one-frame Tweedie jitter causing
    false positive early-abort. Raw max remains available for diagnostics only.
    """
    m = np.asarray(motion, dtype=np.float32)
    st = {"finite": bool(np.isfinite(m).all()), "shape": list(m.shape)}
    if m.ndim != 2 or m.shape[0] < 4 or m.shape[1] < EDGE_DIM:
        st["valid"] = False
        return st
    try:
        joints = fk_24_np(m)
        st["fk_finite"] = bool(np.isfinite(joints).all())
        vel = np.diff(joints, axis=0)
        acc = np.diff(joints, n=2, axis=0)
        jerk = np.diff(joints, n=3, axis=0)
        acc_n = np.linalg.norm(acc, axis=-1).mean(axis=-1) if acc.size else np.zeros((1,), dtype=np.float32)
        jerk_n = np.linalg.norm(jerk, axis=-1).mean(axis=-1) if jerk.size else np.zeros((1,), dtype=np.float32)
        st["joint_acc_p95"] = float(np.percentile(acc_n, 95))
        st["joint_acc_p99"] = float(np.percentile(acc_n, 99))
        st["joint_acc_max_diag"] = float(np.max(acc_n))
        st["joint_jerk_p95"] = float(np.percentile(jerk_n, 95))
        st["joint_jerk_p99"] = float(np.percentile(jerk_n, 99))
        st["joint_jerk_max_diag"] = float(np.max(jerk_n))
        # Bone-length variance should be extremely small for a valid FK skeleton.
        bone_vars = []
        for j in range(1, min(NUM_JOINTS, len(PARENTS))):
            pa = int(PARENTS[j])
            if pa < 0 or pa >= NUM_JOINTS:
                continue
            L = np.linalg.norm(joints[:, j] - joints[:, pa], axis=-1)
            if L.size:
                bone_vars.append(float(np.max(np.abs(L - np.median(L)))))
        st["bone_length_violation_max_m"] = float(max(bone_vars) if bone_vars else 0.0)
    except Exception as exc:
        st["fk_finite"] = False
        st["fk_error"] = str(exc)
    try:
        st.update(audit_motion_np(m, cfg))
    except Exception as exc:
        st["audit_error"] = str(exc)
    st["root_y_range_m"] = float(np.max(m[:, ROOT_Y_IDX]) - np.min(m[:, ROOT_Y_IDX])) if m.size else 0.0
    st["valid"] = True
    return st


def _v46_43_early_abort_oracle(candidate, reference, cfg, stage="diffusion_early_abort_probe", global_start=0):
    """Derivative-safe early-abort oracle.

    It deliberately separates fatal low-frequency barriers from derivative-only
    barriers. A derivative spike on a Tweedie/intermediate probe cannot abort by
    itself, because differentiation amplifies high-frequency noise.
    """
    raw = np.asarray(candidate, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    sigma = _v46_43_env_float("V46_43_EARLY_ABORT_LOWPASS_SIGMA", 2.25)
    relax = _v46_43_env_float("V46_43_EARLY_ABORT_RELAX", 4.0)
    smooth = _v46_43_lowpass_channels(raw, cfg, sigma=sigma)
    c = _v46_43_robust_derivative_stats(smooth, cfg)
    r = _v46_43_robust_derivative_stats(ref, cfg)
    fatal = []
    soft = []

    if not c.get("finite", False) or not c.get("fk_finite", False):
        fatal.append("non_finite_or_fk_invalid")
    if float(c.get("root_y_range_m", 0.0)) > _v46_41_env_float("V46_41_KBO_ROOT_RANGE_ABS_MAX_M", 2.50) * max(1.0, relax * 0.75):
        fatal.append("root_y_range_abs_exceeded")
    if abs(float(c.get("floor_y", 0.0)) - float(r.get("floor_y", 0.0))) > _v46_41_env_float("V46_41_KBO_FLOOR_SHIFT_MAX_M", 1.50) * max(1.0, relax):
        fatal.append("floor_shift_exceeded")
    if float(c.get("bone_length_violation_max_m", 0.0)) > _v46_41_env_float("V46_41_KBO_BONE_LENGTH_EPS_M", 0.02) * max(1.0, relax):
        fatal.append("bone_length_violation")

    # Derivative barriers are soft in early-abort mode. They need co-occurring
    # fatal evidence, or a very large robust p99 excursion when the user enables it.
    acc_thr = _v46_41_env_float("V46_41_KBO_ACC_MAX", 3.0) * max(1.0, relax)
    jerk_thr = _v46_41_env_float("V46_41_KBO_JERK_MAX", 3.0) * max(1.0, relax)
    if float(c.get("joint_acc_p99", 0.0)) > acc_thr:
        soft.append("robust_acc_p99_spike")
    if float(c.get("joint_jerk_p99", 0.0)) > jerk_thr:
        soft.append("robust_jerk_p99_spike")

    if _v46_41_env_bool("V46_41_KBO_STAGE_ANCHOR_ENABLE", True):
        ae = _v46_41_anchor_error(smooth, global_start)
        # Anchor is a soft early signal; high-energy windows already get low
        # weights through V46.43 anchor_error.
        c["stage_anchor_error_p95_m"] = float(ae)
        if ae > _v46_41_env_float("V46_41_KBO_ANCHOR_P95_MAX_M", 0.85) * max(1.0, relax):
            soft.append("weighted_stage_anchor_deviation")

    derivative_only_abort = _v46_43_env_bool("V46_43_EARLY_ABORT_ALLOW_DERIVATIVE_ONLY_FATAL", False)
    if fatal:
        ok = False
        reasons = fatal + soft
    elif derivative_only_abort and len(soft) >= _v46_43_env_int("V46_43_EARLY_ABORT_MIN_SOFT_BARRIERS", 2):
        ok = False
        reasons = soft
    else:
        ok = True
        reasons = soft  # diagnostic only

    detail = {
        "kbo_mode": "v46_43_derivative_safe_early_abort",
        "lowpass_sigma": float(sigma),
        "relax": float(relax),
        "fatal_barriers": fatal,
        "soft_barriers": soft,
        "candidate_lowpass": c,
        "reference": r,
        "raw_probe_shape": list(raw.shape),
        "global_start": int(global_start),
        "interpretation": "soft derivative barriers alone do not abort Tweedie probes",
    }
    return ok, reasons, detail


# Preserve the V46.42 function name used by diffusion proposal, but replace its logic.
def _v46_42_kbo_early_abort(candidate, reference, cfg, stage="diffusion_early_abort_probe", global_start=0):
    return _v46_43_early_abort_oracle(candidate, reference, cfg, stage=stage, global_start=global_start)


def _v46_43_anchor_weight_for_motion(motion, global_start=0):
    m = np.asarray(motion, dtype=np.float32)
    T = len(m)
    if T <= 0:
        return np.ones((0, 1), dtype=np.float32)
    try:
        frame_w = _v46_42_frame_weights(T, global_start=global_start)
    except Exception:
        frame_w = np.ones((T, 1), dtype=np.float32)
    try:
        vel_gate = _v46_42_velocity_gate(m)
    except Exception:
        vel_gate = np.ones((T, 1), dtype=np.float32)
    # Harder leap gate: if root speed is high, do not pull the body back to a
    # low-pass prior. Dilate high-speed regions to include takeoff/landing.
    if T >= 3:
        v = np.linalg.norm(np.diff(m[:, [ROOT_X_IDX, ROOT_Z_IDX]], axis=0), axis=-1)
        v = np.concatenate([[v[0]], v]).astype(np.float32)
        leap_thr = _v46_43_env_float("V46_43_MSA_LEAP_SPEED_THRESH", 0.070)
        leap = v > leap_thr
        if ndi is not None and np.any(leap):
            leap = ndi.binary_dilation(leap.astype(bool), iterations=_v46_43_env_int("V46_43_MSA_LEAP_DILATE", 4))
        leap_gate = np.where(leap, _v46_43_env_float("V46_43_MSA_LEAP_MIN_GATE", 0.0), 1.0).astype(np.float32)[:, None]
    else:
        leap_gate = np.ones((T, 1), dtype=np.float32)
    w = frame_w * vel_gate * leap_gate
    return np.clip(w, _v46_42_env_float("V46_42_MSA_MIN_WEIGHT", 0.05), 1.0).astype(np.float32)


def _v46_41_apply_stage_prior(motion, cfg, strength=None, global_start=0):
    """Velocity-preserving MSA.

    Instead of dragging root to the low-frequency prior frame-by-frame, correct
    only low-frequency drift with capped, smoothed offsets. Leap/high-speed
    frames are gated out to avoid moonwalk/airborne rubber-band artifacts.
    """
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
    w = _v46_43_anchor_weight_for_motion(m, global_start=global_start)
    raw_corr = prior_local - m[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    sigma = _v46_43_env_float("V46_43_MSA_CORRECTION_LOWPASS_SIGMA", 10.0)
    if ndi is not None and len(raw_corr) > 7 and sigma > 0:
        corr = ndi.gaussian_filter1d(raw_corr, sigma=float(sigma), axis=0, mode="nearest")
    else:
        corr = raw_corr
    # Capping the correction magnitude and its frame-to-frame velocity preserves
    # local foot/root dynamics and prevents rubber-band deceleration.
    max_delta = _v46_43_env_float("V46_43_MSA_MAX_OFFSET_DELTA_M", _v46_41_env_float("V46_41_MSA_MAX_DELTA_M", 0.06))
    corr = np.clip(corr, -max_delta, max_delta)
    max_corr_vel = _v46_43_env_float("V46_43_MSA_MAX_CORRECTION_VEL_MPF", 0.006)
    if len(corr) > 1 and max_corr_vel > 0:
        smooth_corr = corr.copy()
        for t in range(1, len(smooth_corr)):
            step = np.clip(smooth_corr[t] - smooth_corr[t-1], -max_corr_vel, max_corr_vel)
            smooth_corr[t] = smooth_corr[t-1] + step
        corr = smooth_corr
    alpha = float(base_alpha) * w[:, 0]
    m[:, ROOT_X_IDX] = m[:, ROOT_X_IDX] + alpha * corr[:, 0]
    m[:, ROOT_Z_IDX] = m[:, ROOT_Z_IDX] + alpha * corr[:, 1]
    m, _ = enforce_edge151_contract_np(m, cfg, source_hint="v46_43_velocity_preserving_msa", derive_contact=True, project_rot=True)
    meta.update({
        "applied": True,
        "version": "v46_43_velocity_preserving_dynamic_msa",
        "base_strength": float(base_alpha),
        "effective_strength_mean": float(base_alpha * float(np.mean(w))) if len(w) else 0.0,
        "effective_strength_min": float(base_alpha * float(np.min(w))) if len(w) else 0.0,
        "correction_lowpass_sigma": float(sigma),
        "max_offset_delta_m": float(max_delta),
        "max_correction_velocity_mpf": float(max_corr_vel),
        "interpretation": "low-frequency drift correction only; leap/high-root-speed frames are released",
    })
    return m.astype(np.float32), meta


def _v46_41_anchor_error(candidate, a0=0):
    """Anchor error for KBO, with leap/high-energy weighting.

    High-energy or high-root-speed windows are not rejected only because they
    deviate from the low-frequency stage prior.
    """
    global _V46_41_STAGE_PRIOR_XZ
    cand = np.asarray(candidate, dtype=np.float32)
    if _V46_41_STAGE_PRIOR_XZ is None:
        return 0.0
    a = int(a0); b = a + len(cand)
    if a < 0 or b > len(_V46_41_STAGE_PRIOR_XZ):
        return 0.0
    prior = _V46_41_STAGE_PRIOR_XZ[a:b]
    err = np.linalg.norm(cand[:, [ROOT_X_IDX, ROOT_Z_IDX]] - prior, axis=-1)
    w = _v46_43_anchor_weight_for_motion(cand, global_start=a)[:, 0]
    weighted = err * np.clip(w, _v46_42_env_float("V46_42_MSA_MIN_WEIGHT", 0.05), 1.0)
    return float(np.percentile(weighted, 95))


def _v46_41_diffusion_window_proposal(snapshot, cond, sm_win, ckpt_path, cfg, global_start=0):
    """V46.43 diffusion proposal with consecutive robust early probes."""
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
    retr_in, _ = enforce_edge151_contract_np(np.asarray(snapshot, dtype=np.float32), cfg, source_hint="v46_43_diffusion_window_retrieval", derive_contact=True, project_rot=True)
    mask_in = np.asarray(sm_win, dtype=np.float32)
    if mask_in.ndim == 1:
        mask_in = mask_in[:, None]
    if mask_in.shape[0] != retr_in.shape[0]:
        mask_in = resample_motion_np(mask_in, retr_in.shape[0])
    # Multiple probe points reduce single-step false positives.
    probe_fracs = os.environ.get("V46_43_EARLY_ABORT_PROBE_FRACTIONS", "0.66,0.50,0.33")
    probe_ts = set()
    for part in probe_fracs.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            probe_ts.add(int(round(Tdiff * float(part))))
        except Exception:
            pass
    consecutive_needed = max(1, _v46_43_env_int("V46_43_EARLY_ABORT_CONSECUTIVE_FATAL", 2))
    fatal_streak = 0
    with torch.no_grad():
        retr = torch.from_numpy(retr_in[None]).float().to(cfg.device)
        raw_mask = torch.from_numpy(mask_in[None].astype(np.float32)).float().to(cfg.device)
        mask = torch.clamp(float(core_strength) + (float(trans_strength) - float(core_strength)) * raw_mask, 0.0, 1.0)
        c = torch.from_numpy(cond[None].astype(np.float32)).float().to(cfg.device)
        x = retr + float(noise_scale) * torch.randn_like(retr) * (0.15 + 0.85 * mask)
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
            if ti in probe_ts:
                probe = x[0].detach().cpu().numpy().astype(np.float32)
                probe, _ = enforce_edge151_contract_np(probe, cfg, source_hint="v46_43_diffusion_early_probe_raw", derive_contact=True, project_rot=True)
                # Important: do NOT call strict safe_residual before early KBO.
                # Bound channel residuals lightly without running final KBO.
                delta = probe - retr_in
                bounded = retr_in.copy().astype(np.float32)
                root_xz = _v46_41_env_float("V46_41_ROOT_XZ_DELTA_MAX_M", 0.05) * _v46_43_env_float("V46_43_EARLY_ABORT_BOUND_RELAX", 2.0)
                root_y = _v46_41_env_float("V46_41_ROOT_Y_DELTA_MAX_M", 0.02) * _v46_43_env_float("V46_43_EARLY_ABORT_BOUND_RELAX", 2.0)
                rot = _v46_41_env_float("V46_41_ROT6D_DELTA_MAX", 0.12) * _v46_43_env_float("V46_43_EARLY_ABORT_BOUND_RELAX", 2.0)
                for idx, mx in [(ROOT_X_IDX, root_xz), (ROOT_Y_IDX, root_y), (ROOT_Z_IDX, root_xz)]:
                    bounded[:, idx] = retr_in[:, idx] + np.clip(delta[:, idx], -mx, mx) * np.clip(mask_in[:, 0], 0.0, 1.0)
                bounded[:, ROT6D_START:ROT6D_END] = retr_in[:, ROT6D_START:ROT6D_END] + np.clip(delta[:, ROT6D_START:ROT6D_END], -rot, rot) * np.clip(mask_in, 0.0, 1.0)
                bounded, _ = enforce_edge151_contract_np(bounded, cfg, source_hint="v46_43_diffusion_early_probe_bounded_no_strict_kbo", derive_contact=True, project_rot=True)
                ok, reasons, detail = _v46_43_early_abort_oracle(bounded, retr_in, cfg, stage="diffusion_early_abort_probe", global_start=global_start)
                _V46_43_EARLY_ABORT_TRACE.append({"ti": int(ti), "ok": bool(ok), "reasons": reasons, "detail": detail})
                # Only fatal barriers count toward abort; soft derivative-only
                # barriers are diagnostics in _v46_43_early_abort_oracle.
                fatal_now = bool(detail.get("fatal_barriers"))
                fatal_streak = fatal_streak + 1 if fatal_now else 0
                if fatal_streak >= consecutive_needed:
                    _v46_41_add_token({
                        "mechanism": "early_abort",
                        "version": "v46_43_derivative_safe_consecutive",
                        "stage": "diffusion",
                        "commit_state": "abort_to_ccd",
                        "barrier_violations": reasons,
                        "detail": detail,
                        "hard_negative": True,
                    })
                    raise RuntimeError("diffusion_early_abort_v46_43:" + ",".join(reasons))
        y = x[0].detach().cpu().numpy().astype(np.float32)
    y, _ = enforce_edge151_contract_np(y, cfg, source_hint="v46_43_diffusion_window_output", derive_contact=True, project_rot=True)
    return y.astype(np.float32)


def generate(args):
    global _V46_43_EARLY_ABORT_TRACE
    _V46_43_EARLY_ABORT_TRACE = []
    rc = int(_v46_43_orig_generate(args))
    try:
        out_path = Path(args.out)
        json_path = Path(args.json or str(out_path).replace(".npy", ".v46_33_report.json"))
        if json_path.exists():
            report = json.load(open(json_path, "r", encoding="utf-8"))
            report["v46_43_physics_consistent_stability"] = {
                "version": "v46_43_derivative_safe_msa_velocity_preserving_kinetic_dpo",
                "fixes": [
                    "early-abort uses low-pass robust derivative oracle",
                    "derivative-only Tweedie jitter cannot abort by default",
                    "multiple early probes require consecutive fatal low-frequency barriers",
                    "stage anchoring preserves local velocity and releases leap/high-speed windows",
                    "HN-DPO training uses kinetic and motion-density preservation",
                ],
                "early_abort_probe_trace_count": int(len(_V46_43_EARLY_ABORT_TRACE)),
                "early_abort_probe_trace_preview": _v46_43_jsonable(_V46_43_EARLY_ABORT_TRACE[:20]),
                "env": {
                    "V46_43_EARLY_ABORT_LOWPASS_SIGMA": _v46_43_env_float("V46_43_EARLY_ABORT_LOWPASS_SIGMA", 2.25),
                    "V46_43_EARLY_ABORT_RELAX": _v46_43_env_float("V46_43_EARLY_ABORT_RELAX", 4.0),
                    "V46_43_EARLY_ABORT_CONSECUTIVE_FATAL": _v46_43_env_int("V46_43_EARLY_ABORT_CONSECUTIVE_FATAL", 2),
                    "V46_43_MSA_LEAP_SPEED_THRESH": _v46_43_env_float("V46_43_MSA_LEAP_SPEED_THRESH", 0.070),
                    "V46_43_MSA_MAX_CORRECTION_VEL_MPF": _v46_43_env_float("V46_43_MSA_MAX_CORRECTION_VEL_MPF", 0.006),
                },
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(_v46_43_jsonable(report), f, ensure_ascii=False, indent=2)
            print(json.dumps({"v46_43_physics_consistent_stability": report["v46_43_physics_consistent_stability"], "json_updated": str(json_path)}, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[V46.43 WARN] failed to append metadata: {exc}", file=sys.stderr)
    return rc
# ===== V46.43 PHYSICS-CONSISTENT STABILITY PATCH END =====
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
    if "# ===== V46.42 STABILITY ALIGNMENT PATCH START =====" not in text:
        raise RuntimeError("V46.42 patch block was not found. Run apply_v46_42_stability_alignment_patch.py first.")
    text = _strip_block(text, START, END)
    marker = 'if __name__ == "__main__":'
    idx = text.rfind(marker)
    if idx < 0:
        raise RuntimeError('Could not find final if __name__ == "__main__" marker')
    new_text = text[:idx] + PATCH + "\n\n" + text[idx:]
    backup = TARGET.with_suffix(TARGET.suffix + f".v46_43_physics_{time.strftime('%Y%m%d_%H%M%S')}.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(new_text, encoding="utf-8")
    print(f"[BAK] {backup}")
    print(f"[OK] patched {TARGET} with V46.43 physics-consistent stability fixes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
