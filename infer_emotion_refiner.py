#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emotion-conditioned Refiner 推理脚本。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model.emotion_conditioned_refiner import EmotionConditionedSeamRefiner
from tools.build_emotion_refiner_dataset import build_seam_mask, resize_feature, load_motion


ROOT_X = 4
ROOT_Z = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".npy file or directory")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--music_npy", default="")
    ap.add_argument("--audio", default="")
    ap.add_argument("--starts", default="0,32,64,96")
    ap.add_argument("--seam_radius", type=int, default=8)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    in_path = Path(args.input)
    files = sorted(in_path.glob("*.npy")) if in_path.is_dir() else [in_path]
    if not files:
        raise RuntimeError(f"No input npy found: {in_path}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt.get("config", {})
    model = EmotionConditionedSeamRefiner(
        hidden_dim=int(config.get("hidden_dim", 256)),
        num_layers=int(config.get("num_layers", 4)),
        num_heads=int(config.get("num_heads", 4)),
        dropout=float(config.get("dropout", 0.0)),
        residual_scale=float(config.get("residual_scale", 0.20)),
    )
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    music_feat = None
    if args.music_npy:
        music_feat = np.load(args.music_npy).astype(np.float32)
    elif args.audio:
        from tools.extract_music_emotion_features import extract_music_emotion_features
        music_feat, _ = extract_music_emotion_features(args.audio, num_frames=150, fps=args.fps)

    starts = [int(float(x)) for x in args.starts.replace(";", ",").split(",") if x.strip()]

    for f in files:
        motion = load_motion(f)
        T = len(motion)
        if music_feat is None:
            music = np.zeros((T, 8), dtype=np.float32)
        else:
            music = resize_feature(music_feat, T)

        seam = build_seam_mask(T, starts, radius=args.seam_radius)

        with torch.no_grad():
            m = torch.from_numpy(motion[None]).float().to(device)
            mu = torch.from_numpy(music[None]).float().to(device)
            sm = torch.from_numpy(seam[None]).float().to(device)
            pred = model(m, mu, sm).detach().cpu().numpy()[0]

        pred[:, ROOT_X] = motion[:, ROOT_X]
        pred[:, ROOT_Z] = motion[:, ROOT_Z]

        out = out_dir / f"{f.stem}_emotion_refined.npy"
        np.save(out, pred[None].astype(np.float32))
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
