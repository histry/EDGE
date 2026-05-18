#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4_audio_rhythm_utils import (
    audio_rhythm_curve,
    warp_motion_by_speed_curve,
    save_rhythm_diagnostics,
)


def load_motion(path):
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        return arr
    if path.suffix == ".pkl":
        d = pickle.load(open(path, "rb"))
        for k in ["motion", "motion_151", "poses", "unit_motions_physical"]:
            if k in d:
                arr = np.asarray(d[k], dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr[0]
                return arr.astype(np.float32)
    raise ValueError(f"Cannot load motion from {path}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prior", required=True, help="retrieved GT / RAG unit pkl or npy")
    ap.add_argument("--audio", required=True, help="music wav used for rhythm adaptation")
    ap.add_argument("--out", required=True)

    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=45)
    ap.add_argument("--audio_dim", type=int, default=803)
    ap.add_argument("--feature_type", default="hybrid")
    ap.add_argument("--seed", type=int, default=1234)

    # V3D/V4 prior schedule.
    ap.add_argument("--strength", type=float, default=0.35)
    ap.add_argument("--start_frac", type=float, default=0.45)
    ap.add_argument("--gamma", type=float, default=1.2)
    ap.add_argument("--prior_max_blend", type=float, default=0.85)

    # V4 rhythm adaptation.
    ap.add_argument("--warp_strength", type=float, default=1.0)
    ap.add_argument("--min_speed", type=float, default=0.55)
    ap.add_argument("--max_speed", type=float, default=1.85)
    ap.add_argument("--rhythm_min_gain", type=float, default=0.65)
    ap.add_argument("--rhythm_max_gain", type=float, default=1.35)
    ap.add_argument("--rms_weight", type=float, default=0.55)
    ap.add_argument("--flux_weight", type=float, default=0.45)
    ap.add_argument("--smooth_radius", type=int, default=2)

    # Spatial guidance mode.
    ap.add_argument(
        "--body_part",
        default="upper_safe_plus",
        choices=[
            "upper_safe_plus",
            "arms",
            "arms_hands",
            "hands",
            "torso",
            "torso_only",
            "style_fullbody",
            "fullbody_style",
            "dunhuang_style",
            "all_rot",
            "full_rot",
        ],
        help=(
            "Which feature group to guide. "
            "Use style_fullbody for Dunhuang three-bend silhouette guidance."
        ),
    )

    # Upper/spine scales.
    ap.add_argument("--torso_scale", type=float, default=1.00)
    ap.add_argument("--neck_head_scale", type=float, default=1.00)
    ap.add_argument("--arms_scale", type=float, default=1.00)

    # Style-fullbody scales.
    ap.add_argument("--root_y_scale", type=float, default=0.25)
    ap.add_argument("--root_xz_scale", type=float, default=0.00)
    ap.add_argument("--contact_scale", type=float, default=0.00)
    ap.add_argument("--pelvis_scale", type=float, default=0.35)
    ap.add_argument("--hips_scale", type=float, default=0.30)
    ap.add_argument("--knees_scale", type=float, default=0.18)
    ap.add_argument("--ankles_feet_scale", type=float, default=0.08)
    ap.add_argument("--lower_style_scale", type=float, default=1.00)

    ap.add_argument("--save_warped_prior", default="")
    ap.add_argument("--save_diag", default="")
    args = ap.parse_args()

    # Clean V3/V4 inference environment.
    os.environ.setdefault("EDGE_TRAIN_PROFILE", "v3_unit_recon")
    os.environ.setdefault("EDGE_V3_UNIT_RECON", "1")
    os.environ.setdefault("EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER", "1")
    os.environ.setdefault("EDGE_TRAJ_EVENT_COND", "0")
    os.environ.setdefault("EDGE_BEAT_GUIDANCE", "0")
    os.environ.setdefault("EDGE_UNIT_SOFT_PRIOR", "0")
    os.environ.setdefault("EDGE_ENABLE_TEXT_CONTEXT_RAG", "0")

    # V3D/V4 soft prior.
    os.environ["EDGE_V3D_UPPER_SOFT_PRIOR"] = "1"
    os.environ["EDGE_V3D_UPPER_PRIOR_STRENGTH"] = str(args.strength)
    os.environ["EDGE_V3D_UPPER_PRIOR_START_FRAC"] = str(args.start_frac)
    os.environ["EDGE_V3D_UPPER_PRIOR_GAMMA"] = str(args.gamma)
    os.environ["EDGE_V3D_PRIOR_MAX_BLEND"] = str(args.prior_max_blend)
    os.environ.setdefault("EDGE_V3D_UPPER_PRIOR_DEBUG", "1")

    # Body-part / style-fullbody mask.
    os.environ["EDGE_V3D_UPPER_PRIOR_BODY_PART"] = str(args.body_part)
    os.environ["EDGE_V3D_TORSO_PRIOR_SCALE"] = str(args.torso_scale)
    os.environ["EDGE_V3D_NECK_HEAD_PRIOR_SCALE"] = str(args.neck_head_scale)
    os.environ["EDGE_V3D_ARMS_PRIOR_SCALE"] = str(args.arms_scale)

    os.environ["EDGE_V3D_ROOT_Y_PRIOR_SCALE"] = str(args.root_y_scale)
    os.environ["EDGE_V3D_ROOT_XZ_PRIOR_SCALE"] = str(args.root_xz_scale)
    os.environ["EDGE_V3D_CONTACT_PRIOR_SCALE"] = str(args.contact_scale)
    os.environ["EDGE_V3D_PELVIS_PRIOR_SCALE"] = str(args.pelvis_scale)
    os.environ["EDGE_V3D_HIPS_PRIOR_SCALE"] = str(args.hips_scale)
    os.environ["EDGE_V3D_KNEES_PRIOR_SCALE"] = str(args.knees_scale)
    os.environ["EDGE_V3D_ANKLES_FEET_PRIOR_SCALE"] = str(args.ankles_feet_scale)
    os.environ["EDGE_V3D_LOWER_STYLE_PRIOR_SCALE"] = str(args.lower_style_scale)

    # V4 rhythm adaptive per-frame prior strength.
    os.environ["EDGE_V4_RHYTHM_ADAPTIVE_PRIOR"] = "1"
    os.environ["EDGE_V4_RHYTHM_MIN_GAIN"] = str(args.rhythm_min_gain)
    os.environ["EDGE_V4_RHYTHM_MAX_GAIN"] = str(args.rhythm_max_gain)
    os.environ.setdefault("EDGE_V4_RHYTHM_DEBUG", "1")

    from unit_reconstruction_patch import install_v3_unit_reconstruction_patch
    from v3d_upper_soft_prior_patch import install_v3d_upper_soft_prior_patch

    install_v3_unit_reconstruction_patch(verbose=True)
    install_v3d_upper_soft_prior_patch(verbose=True)

    from EDGE import EDGE

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prior_phys_raw = load_motion(args.prior)
    if prior_phys_raw.ndim != 2 or prior_phys_raw.shape[-1] != 151:
        raise ValueError(f"prior must be [T,151], got {prior_phys_raw.shape}")

    rhythm = audio_rhythm_curve(
        args.audio,
        target_len=args.seq_len,
        rms_weight=args.rms_weight,
        flux_weight=args.flux_weight,
        smooth_radius=args.smooth_radius,
    )

    warped_prior, phase = warp_motion_by_speed_curve(
        prior_phys_raw,
        rhythm["speed_curve"],
        warp_strength=args.warp_strength,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        preserve_ends=True,
    )

    rhythm_curve = np.asarray(rhythm["rhythm_curve"], dtype=np.float32)
    speed_curve = np.asarray(rhythm["speed_curve"], dtype=np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.save_warped_prior:
        warped_path = Path(args.save_warped_prior)
    else:
        warped_path = out.with_name(out.stem + "_warped_prior.npy")
    warped_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(warped_path, warped_prior.astype(np.float32))

    diag = {
        "checkpoint": args.checkpoint,
        "prior": args.prior,
        "audio": args.audio,
        "out": args.out,
        "warped_prior": str(warped_path),
        "seq_len": args.seq_len,
        "seed": args.seed,
        "strength": args.strength,
        "start_frac": args.start_frac,
        "gamma": args.gamma,
        "prior_max_blend": args.prior_max_blend,
        "body_part": args.body_part,
        "root_y_scale": args.root_y_scale,
        "root_xz_scale": args.root_xz_scale,
        "contact_scale": args.contact_scale,
        "pelvis_scale": args.pelvis_scale,
        "hips_scale": args.hips_scale,
        "knees_scale": args.knees_scale,
        "ankles_feet_scale": args.ankles_feet_scale,
        "lower_style_scale": args.lower_style_scale,
        "torso_scale": args.torso_scale,
        "neck_head_scale": args.neck_head_scale,
        "arms_scale": args.arms_scale,
        "warp_strength": args.warp_strength,
        "min_speed": args.min_speed,
        "max_speed": args.max_speed,
        "rhythm_min_gain": args.rhythm_min_gain,
        "rhythm_max_gain": args.rhythm_max_gain,
        "phase": phase,
        "rhythm_curve": rhythm_curve,
        "speed_curve": speed_curve,
        "audio_sample_rate": rhythm["sample_rate"],
        "audio_num_samples": rhythm["audio_num_samples"],
    }

    diag_path = Path(args.save_diag) if args.save_diag else out.with_name(out.stem + "_rhythm_diag.json")
    save_rhythm_diagnostics(diag_path, diag)

    edge = EDGE(
        feature_type=args.feature_type,
        checkpoint_path=args.checkpoint,
        EMA=True,
        audio_dim=args.audio_dim,
        seq_len=args.seq_len,
        mixed_precision="bf16",
        cond_drop_prob=0.0,
        audio_pairing_mode="none",
        mmr_loss_weight=0.0,
        keyframe_condition_prob=0.0,
        keyframe_loss_weight=0.0,
        mid_keyframe_condition_prob=0.0,
        mid_keyframe_count=0,
        trajectory_loss_weight=0.0,
        trajectory_velocity_loss_weight=0.0,
        beat_guidance_weight=0.0,
        sync_loss_weight=0.0,
        energy_loss_weight=0.0,
        root_lower_coupling_loss_weight=0.0,
        contact_loss_weight=0.02,
        foot_loss_weight=0.02,
        train_stage="full",
    )

    edge.eval()
    device = edge.accelerator.device

    prior_t = torch.tensor(warped_prior, dtype=torch.float32, device=device).unsqueeze(0)

    if edge.normalizer is not None:
        prior_norm = edge.normalizer.normalize(prior_t.detach().cpu()).to(device=device, dtype=torch.float32)
    else:
        prior_norm = prior_t

    prior_norm = prior_norm.repeat(args.batch_size, 1, 1)

    rhythm_t = torch.tensor(rhythm_curve, dtype=torch.float32, device=device).view(1, args.seq_len, 1)
    rhythm_t = rhythm_t.repeat(args.batch_size, 1, 1)

    cond = {
        "audio": torch.zeros((args.batch_size, args.seq_len, args.audio_dim), device=device, dtype=torch.float32),
        "retrieved_prior_motion": prior_norm,
        "rhythm_weight": rhythm_t,
    }

    shape = (args.batch_size, args.seq_len, 151)

    print("==== V4 music-adaptive retrieved prior sampling ====")
    print("checkpoint:", args.checkpoint)
    print("prior:", args.prior)
    print("audio:", args.audio)
    print("out:", args.out)
    print("warped_prior:", warped_path)
    print("diag:", diag_path)
    print("shape:", shape)
    print("body_part:", args.body_part)
    print("strength:", args.strength)
    print("start_frac:", args.start_frac)
    print("gamma:", args.gamma)
    print("prior_max_blend:", args.prior_max_blend)
    print("root_y/root_xz/contact:", args.root_y_scale, args.root_xz_scale, args.contact_scale)
    print("pelvis/hips/knees/ankles_feet:", args.pelvis_scale, args.hips_scale, args.knees_scale, args.ankles_feet_scale)
    print("torso/neck/arms:", args.torso_scale, args.neck_head_scale, args.arms_scale)
    print("warp_strength:", args.warp_strength)
    print("min_speed/max_speed:", args.min_speed, args.max_speed)
    print("rhythm gain:", args.rhythm_min_gain, args.rhythm_max_gain)
    print("rhythm mean/min/max:", float(rhythm_curve.mean()), float(rhythm_curve.min()), float(rhythm_curve.max()))
    print("phase first/mid/last:", float(phase[0]), float(phase[len(phase)//2]), float(phase[-1]))

    with torch.no_grad():
        samples_norm = edge.diffusion.p_sample_loop(
            shape,
            cond,
            constraint=None,
            use_tto=False,
        )

    if edge.normalizer is not None:
        samples = edge.normalizer.unnormalize(samples_norm.detach().cpu())
        if torch.is_tensor(samples):
            samples = samples.detach().cpu().numpy()
    else:
        samples = samples_norm.detach().cpu().numpy()

    np.save(out, samples.astype(np.float32))
    print("saved:", out, samples.shape)


if __name__ == "__main__":
    main()
