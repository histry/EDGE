#!/usr/bin/env python3
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=45)
    ap.add_argument("--audio_dim", type=int, default=803)
    ap.add_argument("--feature_type", default="hybrid")
    ap.add_argument("--strength", type=float, default=0.22)
    ap.add_argument("--start_frac", type=float, default=0.55)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    # V3 clean inference environment.
    os.environ.setdefault("EDGE_TRAIN_PROFILE", "v3_unit_recon")
    os.environ.setdefault("EDGE_V3_UNIT_RECON", "1")
    os.environ.setdefault("EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER", "1")
    os.environ.setdefault("EDGE_TRAJ_EVENT_COND", "0")
    os.environ.setdefault("EDGE_BEAT_GUIDANCE", "0")
    os.environ.setdefault("EDGE_UNIT_SOFT_PRIOR", "0")
    os.environ.setdefault("EDGE_ENABLE_TEXT_CONTEXT_RAG", "0")

    # V3D soft prior.
    os.environ["EDGE_V3D_UPPER_SOFT_PRIOR"] = "1"
    os.environ["EDGE_V3D_UPPER_PRIOR_STRENGTH"] = str(args.strength)
    os.environ["EDGE_V3D_UPPER_PRIOR_START_FRAC"] = str(args.start_frac)
    os.environ["EDGE_V3D_UPPER_PRIOR_GAMMA"] = str(args.gamma)
    os.environ.setdefault("EDGE_V3D_UPPER_PRIOR_DEBUG", "1")
    os.environ.setdefault("EDGE_V3D_TORSO_PRIOR_SCALE", "0.65")
    os.environ.setdefault("EDGE_V3D_NECK_HEAD_PRIOR_SCALE", "0.85")
    os.environ.setdefault("EDGE_V3D_ARMS_PRIOR_SCALE", "1.00")

    from unit_reconstruction_patch import install_v3_unit_reconstruction_patch
    from v3d_upper_soft_prior_patch import install_v3d_upper_soft_prior_patch

    install_v3_unit_reconstruction_patch(verbose=True)
    install_v3d_upper_soft_prior_patch(verbose=True)

    from EDGE import EDGE

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    prior_phys = load_motion(args.prior)[: args.seq_len]
    if prior_phys.shape != (args.seq_len, 151):
        raise ValueError(f"prior must be [{args.seq_len},151], got {prior_phys.shape}")

    prior_t = torch.tensor(prior_phys, dtype=torch.float32, device=device).unsqueeze(0)

    if edge.normalizer is not None:
        prior_norm = edge.normalizer.normalize(prior_t.detach().cpu()).to(device=device, dtype=torch.float32)
    else:
        prior_norm = prior_t

    prior_norm = prior_norm.repeat(args.batch_size, 1, 1)

    cond = {
        "audio": torch.zeros((args.batch_size, args.seq_len, args.audio_dim), device=device, dtype=torch.float32),
        "retrieved_prior_motion": prior_norm,
    }

    shape = (args.batch_size, args.seq_len, 151)

    print("==== V3D upper soft prior sampling ====")
    print("checkpoint:", args.checkpoint)
    print("prior:", args.prior)
    print("out:", args.out)
    print("shape:", shape)
    print("strength:", args.strength)
    print("start_frac:", args.start_frac)
    print("gamma:", args.gamma)

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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, samples.astype(np.float32))

    print("saved:", out, samples.shape)


if __name__ == "__main__":
    main()
