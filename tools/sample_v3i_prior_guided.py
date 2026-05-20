#!/usr/bin/env python3
import argparse
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("EDGE_TRAIN_PROFILE", "v3_unit_recon")
os.environ.setdefault("EDGE_V3_UNIT_RECON", "1")
os.environ.setdefault("EDGE_TRAJECTORY_PLANE", "xz")
os.environ.setdefault("EDGE_DUNHUANG_ASSERT_TRAJ_MATCH", "0")

os.environ.setdefault("EDGE_V3_BASE_LOSS_STABILITY", "1")
os.environ.setdefault("EDGE_V3_DISABLE_RAW_PHYSICAL_LOSSES", "1")
os.environ.setdefault("EDGE_V3_LOSS_STABILITY", "1")
os.environ.setdefault("EDGE_V3_CAP_TOTAL_LOSS", "0")
os.environ.setdefault("EDGE_X0_RECON_LOSS", "0")
os.environ.setdefault("EDGE_V3_TEMPORAL_WEIGHT", "0.0")
os.environ.setdefault("EDGE_V3C_VISIBLE_FK", "0")
os.environ.setdefault("EDGE_V3F_BODY_CENTERED", "0")
os.environ.setdefault("EDGE_V3H_SUPPORT_CHAIN_LOSS", "1")
os.environ.setdefault("EDGE_V3H_SUPPORT_CHAIN_WEIGHT", "0.035")

from EDGE import EDGE


def load_motion(path, seq_len=45):
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p, allow_pickle=True)
    else:
        obj = pickle.load(open(p, "rb"))
        for k in ["motion_151", "motion", "unit_motion", "unit_motions_physical"]:
            if isinstance(obj, dict) and k in obj:
                arr = obj[k]
                break
        else:
            raise KeyError(f"No motion key in {p}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape[-1] != 151:
        raise ValueError(f"Expected [T,151], got {arr.shape}: {p}")
    if arr.shape[0] < seq_len:
        pad = np.repeat(arr[-1:], seq_len - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    return arr[:seq_len].astype(np.float32)


def make_mask(mode, device):
    mask = torch.zeros((1, 1, 151), dtype=torch.float32, device=device)

    if mode == "upper":
        # approximate upper-body dims: torso/head/arms/hands rotations
        joints = [3, 6, 9, 12, 15, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23]
        for j in joints:
            s = 7 + 6 * j
            mask[..., s:s+6] = 1.0

    elif mode == "torso_upper":
        joints = [0, 3, 6, 9, 12, 15, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23]
        for j in joints:
            s = 7 + 6 * j
            mask[..., s:s+6] = 1.0

    elif mode == "full_rot":
        mask[..., 7:151] = 1.0

    elif mode == "body_no_rootxz":
        mask[..., 5:6] = 1.0
        mask[..., 7:151] = 1.0

    elif mode == "full_no_contact":
        mask[..., 4:151] = 1.0

    elif mode == "full":
        mask[..., :] = 1.0

    else:
        raise ValueError(f"Unknown mask mode: {mode}")

    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="prior_guided")
    ap.add_argument("--seq_len", type=int, default=45)
    ap.add_argument("--audio_dim", type=int, default=803)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--strength", type=float, default=0.45)
    ap.add_argument("--start_frac", type=float, default=0.65)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--mask", default="full_rot",
                    choices=["upper", "torso_upper", "full_rot", "body_no_rootxz", "full_no_contact", "full"])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("V3I prior-guided sampling")
    print(f"ckpt={args.ckpt}")
    print(f"prior={args.prior}")
    print(f"out={out}")
    print(f"strength={args.strength} start_frac={args.start_frac} gamma={args.gamma} mask={args.mask}")
    print("============================================================")

    edge = EDGE(
        feature_type="hybrid",
        checkpoint_path=args.ckpt,
        EMA=True,
        learning_rate=1e-6,
        weight_decay=0.02,
        audio_dim=args.audio_dim,
        seq_len=args.seq_len,
        mixed_precision="no",
        gradient_checkpointing=False,
        cond_drop_prob=0.0,
        audio_pairing_mode="none",
        mmr_loss_weight=0.0,
        keyframe_condition_prob=0.0,
        keyframe_loss_weight=0.0,
        mid_keyframe_condition_prob=0.0,
        mid_keyframe_count=0,
        beat_guidance_weight=0.0,
        trajectory_loss_weight=0.0,
        trajectory_velocity_loss_weight=0.0,
        energy_loss_weight=0.0,
        root_lower_coupling_loss_weight=0.0,
        hard_keyframe_project=False,
        train_stage="full",
        enable_rag_summary_token=False,
    )
    edge.eval()

    device = edge.accelerator.device
    diffusion = edge.diffusion
    n = diffusion.n_timestep

    prior_phys = load_motion(args.prior, seq_len=args.seq_len)
    prior = torch.from_numpy(prior_phys).float().to(device).unsqueeze(0)
    prior_norm = edge.normalizer.normalize(prior)
    prior_norm = prior_norm.repeat(args.num_samples, 1, 1)

    mask = make_mask(args.mask, device).repeat(args.num_samples, args.seq_len, 1)

    cond = {
        "audio": torch.zeros((args.num_samples, args.seq_len, args.audio_dim), dtype=torch.float32, device=device)
    }

    x = torch.randn((args.num_samples, args.seq_len, 151), device=device)

    apply_t_threshold = int(n * float(args.start_frac))

    with torch.no_grad():
        for i in reversed(range(n)):
            t = torch.full((args.num_samples,), i, device=device, dtype=torch.long)
            x, _ = diffusion.p_sample(x, cond, t, use_tto=False)

            # Apply only in later denoising phase.
            if i <= apply_t_threshold:
                progress = 1.0 - (float(i) / max(1.0, float(apply_t_threshold)))
                w = float(args.strength) * (progress ** float(args.gamma))
                noisy_prior = diffusion.q_sample(prior_norm, t)
                x = x * (1.0 - w * mask) + noisy_prior * (w * mask)

    motion = edge.normalizer.unnormalize(x).detach().cpu().numpy().astype(np.float32)
    np.save(out, motion)
    print(f"✅ saved {out}, shape={motion.shape}")


if __name__ == "__main__":
    main()
