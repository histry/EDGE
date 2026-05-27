#!/usr/bin/env python3
import os

# Disable optional heavy context branches before importing generate_controlled.
os.environ.setdefault("EDGE_ENABLE_TEXT_CONTEXT_RAG", "0")
os.environ.setdefault("EDGE_ENABLE_RAG_SUMMARY_TOKEN", "0")
os.environ.setdefault("EDGE_UNIT_SOFT_PRIOR", "0")
os.environ.setdefault("EDGE_ENERGY_COND", "0")
os.environ.setdefault("EDGE_DISABLE_TRAJ_COND", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from generate_controlled import extract_audio_feature, resample_feature, to_numpy
from EDGE import EDGE

CONTACT = slice(0, 4)
ROOT_X = 4
ROOT_Y = 5
ROOT_Z = 6
ROT = slice(7, 151)


def load_motion(path):
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        obj = arr.item()
        arr = obj.get("motion", obj.get("motion_151", obj.get("pose", arr)))

    arr = np.asarray(arr, dtype=np.float32)
    squeeze = False

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
        squeeze = True

    if arr.ndim != 2 or arr.shape[1] != 151:
        raise ValueError(f"{path}: expected [T,151] or [1,T,151], got {arr.shape}")

    return arr.astype(np.float32), squeeze


def save_motion(path, motion, squeeze=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = motion[None] if squeeze else motion
    np.save(path, out.astype(np.float32))


def normalize_motion(normalizer, motion_np, device):
    x = torch.from_numpy(motion_np).float().unsqueeze(0).to(device)
    return normalizer.normalize(x)


def unnormalize_motion(normalizer, motion_t):
    x = normalizer.unnormalize(motion_t)
    return to_numpy(x)[0].astype(np.float32)


def parse_int_list(text):
    if not text:
        return []
    return [int(float(x.strip())) for x in text.replace(";", ",").split(",") if x.strip()]


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_temporal_rewrite_weight(
    frames,
    seams,
    core_radius,
    buffer_radius,
    global_floor,
    max_rewrite_weight,
):
    """
    Returns [T] weight:
      0 means preserve prior
      1 means model may rewrite

    We use a soft bell around each seam, plus optional tiny global floor.
    """
    w = np.full((frames,), float(global_floor), dtype=np.float32)

    core_radius = max(0, int(core_radius))
    buffer_radius = max(core_radius + 1, int(buffer_radius))

    for seam in seams:
        for f in range(max(0, seam - buffer_radius), min(frames, seam + buffer_radius + 1)):
            d = abs(f - seam)
            if d <= core_radius:
                val = 1.0
            else:
                val = 1.0 - (d - core_radius) / max(buffer_radius - core_radius, 1)
            val = float(smoothstep(val))
            w[f] = max(w[f], val)

    w = np.clip(w * float(max_rewrite_weight), 0.0, 1.0)
    return w.astype(np.float32)


def build_feature_mask(
    temporal_w,
    lower_scale=0.20,
    torso_scale=0.55,
    upper_scale=0.80,
    root_y_scale=0.0,
):
    """
    Feature-wise soft rewrite mask [1,T,151].

    Contacts and root X/Z are always preserved.
    Rotation channels are grouped with approximate 24-joint layout:
      joints 0:8   lower/root-ish
      joints 8:14  torso
      joints 14:24 upper/arms
    """
    T = len(temporal_w)
    mask = np.zeros((T, 151), dtype=np.float32)

    # root_y optional tiny smoothing; root_x/root_z and contacts remain zero.
    if root_y_scale > 0:
        mask[:, ROOT_Y] = temporal_w * float(root_y_scale)

    # rotation channels
    for j in range(24):
        s = 7 + j * 6
        e = s + 6
        if j < 8:
            scale = lower_scale
        elif j < 14:
            scale = torso_scale
        else:
            scale = upper_scale
        mask[:, s:e] = temporal_w[:, None] * float(scale)

    mask[:, CONTACT] = 0.0
    mask[:, ROOT_X] = 0.0
    mask[:, ROOT_Z] = 0.0
    return mask[None].astype(np.float32)


def build_per_frame_t(temporal_w, base_t, seam_t):
    t = float(base_t) + temporal_w * (float(seam_t) - float(base_t))
    t = np.round(np.clip(t, 0, 999)).astype(np.int64)
    return t[None, :, None]


def q_sample_variable_t(diffusion, x_start, t_frame, noise):
    """
    x_start/noise: [B,T,D]
    t_frame: [B,T,1] long
    """
    t_frame = t_frame.to(device=x_start.device, dtype=torch.long)
    sqrt_a = diffusion.sqrt_alphas_cumprod[t_frame].to(dtype=x_start.dtype)
    sqrt_om = diffusion.sqrt_one_minus_alphas_cumprod[t_frame].to(dtype=x_start.dtype)
    return sqrt_a * x_start + sqrt_om * noise


def metrics(m):
    root = m[:, [ROOT_X, ROOT_Z]]
    rot = m[:, ROT]
    droot = np.linalg.norm(root[1:] - root[:-1], axis=1)
    drot = np.linalg.norm(rot[1:] - rot[:-1], axis=1)
    out = {
        "frames": int(len(m)),
        "root_max_radius": float(np.linalg.norm(root - root[:1], axis=1).max()),
        "global_root_jump_p95": float(np.percentile(droot, 95)) if len(droot) else 0.0,
        "global_rot_jump_p95": float(np.percentile(drot, 95)) if len(drot) else 0.0,
        "segment_activity_mean": float(drot.mean()) if len(drot) else 0.0,
    }
    for b in [35, 45, 70, 74, 90, 105, 108, 135, 140, 142]:
        if 2 <= b < len(m) - 2:
            lo = max(1, b - 2)
            hi = min(len(m), b + 3)
            local = np.linalg.norm(rot[lo:hi] - rot[lo - 1:hi - 1], axis=1)
            out[f"boundary_{b}_local_rot_jump_max"] = float(local.max())
    return out


@torch.no_grad()
def hybrid_ddim_sdedt(
    diffusion,
    prior_norm,
    cond,
    t_frame,
    feature_mask,
    sampling_steps=30,
    eta=0.0,
):
    """
    Hybrid soft-mask SDEdit.

    prior_norm: [1,T,151]
    t_frame: [1,T,1] per-frame forward noise timestep
    feature_mask: [1,T,151], 0 preserve prior, 1 allow model
    """
    device = diffusion.betas.device
    prior_norm = prior_norm.to(device)
    t_frame = t_frame.to(device)
    feature_mask = feature_mask.to(device=device, dtype=prior_norm.dtype)

    B = prior_norm.shape[0]
    max_t = int(t_frame.max().item())
    max_t = max(1, min(max_t, diffusion.n_timestep - 1))

    noise = torch.randn_like(prior_norm)
    x = q_sample_variable_t(diffusion, prior_norm, t_frame, noise)

    times = torch.linspace(-1, max_t, steps=int(sampling_steps) + 1, device=device)
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))

    cond = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cond.items()}

    for time, time_next in tqdm(time_pairs, desc="hybrid-sdedt ddim"):
        time_cond = torch.full((B,), time, device=device, dtype=torch.long)

        pred_noise, x_start = diffusion.model_predictions(
            x,
            cond,
            time_cond,
            clip_x_start=diffusion.clip_denoised,
            constraint=None,
        )

        if time_next < 0:
            x_model = x_start
            prior_next = prior_norm
        else:
            alpha = diffusion.alphas_cumprod[time]
            alpha_next = diffusion.alphas_cumprod[time_next]

            sigma = eta * (
                (1.0 - alpha / alpha_next)
                * (1.0 - alpha_next)
                / (1.0 - alpha)
            ).sqrt()
            c = (1.0 - alpha_next - sigma ** 2).sqrt()

            step_noise = torch.randn_like(x)
            x_model = x_start * alpha_next.sqrt() + c * pred_noise + sigma * step_noise

            # Preserve prior in protected areas at the corresponding lower noise level.
            next_t = torch.minimum(
                t_frame,
                torch.full_like(t_frame, max(int(time_next), 0)),
            )
            prior_next = q_sample_variable_t(diffusion, prior_norm, next_t, noise)

        x = feature_mask * x_model + (1.0 - feature_mask) * prior_next

    # Final safety blend: protected areas exactly return to prior.
    x = feature_mask * x + (1.0 - feature_mask) * prior_norm
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--feature_type", default="hybrid", choices=["hybrid", "baseline", "jukebox"])
    ap.add_argument("--audio_dim", type=int, default=803)
    ap.add_argument("--mixed_precision", default="fp16", choices=["no", "fp16", "bf16"])
    ap.add_argument("--no_ema", action="store_true")

    ap.add_argument("--seams", default="35,74,105,108")
    ap.add_argument("--base_t", type=int, default=25)
    ap.add_argument("--seam_t", type=int, default=160)
    ap.add_argument("--sampling_steps", type=int, default=30)
    ap.add_argument("--eta", type=float, default=0.0)

    ap.add_argument("--core_radius", type=int, default=4)
    ap.add_argument("--buffer_radius", type=int, default=14)
    ap.add_argument("--global_floor", type=float, default=0.00)
    ap.add_argument("--max_rewrite_weight", type=float, default=0.55)

    ap.add_argument("--lower_scale", type=float, default=0.12)
    ap.add_argument("--torso_scale", type=float, default=0.45)
    ap.add_argument("--upper_scale", type=float, default=0.70)
    ap.add_argument("--root_y_scale", type=float, default=0.0)

    ap.add_argument("--prior_space", default="physical", choices=["physical", "normalized"])
    args = ap.parse_args()

    prior_np, squeeze = load_motion(args.prior)
    T = len(prior_np)

    print(f"prior shape={prior_np.shape}, squeeze={squeeze}")

    audio_feature = extract_audio_feature(args.audio, args.feature_type)
    audio_feature = resample_feature(audio_feature, T)

    model = EDGE(
        feature_type=args.feature_type,
        checkpoint_path=args.checkpoint,
        EMA=not bool(args.no_ema),
        audio_dim=args.audio_dim,
        seq_len=T,
        mixed_precision=args.mixed_precision,
    )
    model.eval()

    device = model.accelerator.device
    normalizer = model.normalizer

    if args.prior_space == "physical":
        prior_norm = normalize_motion(normalizer, prior_np, device)
    else:
        prior_norm = torch.from_numpy(prior_np).float().unsqueeze(0).to(device)

    cond = {
        "audio": torch.from_numpy(audio_feature[None]).to(device=device, dtype=torch.float32)
    }

    seams = parse_int_list(args.seams)
    temporal_w = build_temporal_rewrite_weight(
        frames=T,
        seams=seams,
        core_radius=args.core_radius,
        buffer_radius=args.buffer_radius,
        global_floor=args.global_floor,
        max_rewrite_weight=args.max_rewrite_weight,
    )

    feature_mask_np = build_feature_mask(
        temporal_w,
        lower_scale=args.lower_scale,
        torso_scale=args.torso_scale,
        upper_scale=args.upper_scale,
        root_y_scale=args.root_y_scale,
    )
    t_frame_np = build_per_frame_t(temporal_w, args.base_t, args.seam_t)

    feature_mask = torch.from_numpy(feature_mask_np).float().to(device)
    t_frame = torch.from_numpy(t_frame_np).long().to(device)

    print(
        "Hybrid SDEdit config: "
        f"base_t={args.base_t}, seam_t={args.seam_t}, "
        f"max_t={int(t_frame.max().item())}, "
        f"mask_mean={float(feature_mask.mean().item()):.4f}, "
        f"mask_max={float(feature_mask.max().item()):.4f}, "
        f"seams={seams}"
    )

    with torch.no_grad():
        refined_norm = hybrid_ddim_sdedt(
            diffusion=model.diffusion,
            prior_norm=prior_norm,
            cond=cond,
            t_frame=t_frame,
            feature_mask=feature_mask,
            sampling_steps=args.sampling_steps,
            eta=args.eta,
        )

    if args.prior_space == "physical":
        refined_np = unnormalize_motion(normalizer, refined_norm)
    else:
        refined_np = to_numpy(refined_norm)[0].astype(np.float32)

    # Absolute safety: preserve contacts + root X/Z exactly from prior.
    refined_np[:, CONTACT] = prior_np[:, CONTACT]
    refined_np[:, ROOT_X] = prior_np[:, ROOT_X]
    refined_np[:, ROOT_Z] = prior_np[:, ROOT_Z]

    save_motion(args.out, refined_np, squeeze=True)

    report = {
        "checkpoint": args.checkpoint,
        "prior": args.prior,
        "audio": args.audio,
        "out": args.out,
        "seams": seams,
        "base_t": args.base_t,
        "seam_t": args.seam_t,
        "sampling_steps": args.sampling_steps,
        "core_radius": args.core_radius,
        "buffer_radius": args.buffer_radius,
        "global_floor": args.global_floor,
        "max_rewrite_weight": args.max_rewrite_weight,
        "lower_scale": args.lower_scale,
        "torso_scale": args.torso_scale,
        "upper_scale": args.upper_scale,
        "root_y_scale": args.root_y_scale,
        "metrics_prior": metrics(prior_np),
        "metrics_refined": metrics(refined_np),
    }

    report_path = Path(args.out).with_suffix(".hybrid_sdedt_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved motion: {args.out}")
    print(f"saved report: {report_path}")


if __name__ == "__main__":
    main()
