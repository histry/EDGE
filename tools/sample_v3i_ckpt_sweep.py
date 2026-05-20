#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep the same clean V3 unit-recon profile used during training.
os.environ.setdefault("EDGE_TRAIN_PROFILE", "v3_unit_recon")
os.environ.setdefault("EDGE_V3_UNIT_RECON", "1")
os.environ.setdefault("EDGE_TRAJECTORY_PLANE", "xz")
os.environ.setdefault("EDGE_DUNHUANG_ASSERT_TRAJ_MATCH", "0")

# Same conservative V3I switches.
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
os.environ.setdefault("EDGE_V3_MOTION_ENERGY_LOSS_CAP", "80")
os.environ.setdefault("EDGE_V3H_SAMPLE_LOSS_CAP", "8")
os.environ.setdefault("EDGE_V3H_TERM_CAP", "8")

from EDGE import EDGE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--num_samples", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=45)
    ap.add_argument("--audio_dim", type=int, default=803)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("============================================================")
    print("V3I checkpoint sampling")
    print(f"ckpt={args.ckpt}")
    print(f"out_dir={out_dir}")
    print(f"label={args.label}")
    print(f"num_samples={args.num_samples}")
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
    cond = {
        "audio": torch.zeros(
            (args.num_samples, args.seq_len, args.audio_dim),
            dtype=torch.float32,
            device=device,
        )
    }

    with torch.no_grad():
        motion = edge.diffusion.render_sample(
            shape=(args.num_samples, args.seq_len, 151),
            cond=cond,
            normalizer=edge.normalizer,
            epoch=args.label,
            render_out=str(out_dir),
            name=args.label,
            sound=False,
            mode="normal",
            render=False,
            use_tto=False,
        )

    # render_sample already saves one .npy. Save a deterministic alias too.
    motion = np.asarray(motion, dtype=np.float32)
    alias = out_dir / f"{args.label}_samples.npy"
    np.save(alias, motion)
    print(f"✅ saved: {alias}, shape={motion.shape}")


if __name__ == "__main__":
    main()
