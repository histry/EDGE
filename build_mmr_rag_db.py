"""Build clip-level Dunhuang MMR-RAG index.

Compared with pose-level rag_index.npz, this index stores MotionEncoder embeddings
for continuous Dunhuang motion clips. During inference, Chinese music is encoded
by AudioEncoder and retrieves these motion embeddings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from model.mmr_encoder import load_mmr_model
from mmr_data_utils import (
    iter_motion_files,
    load_motion_151,
    load_normalizer_from_checkpoint,
    normalize_motion_if_needed,
    resample_sequence,
    rhythm_summary_from_motion,
    ROOT_X_IDX,
    ROOT_Z_IDX,
)


def contact_stability(motion: np.ndarray) -> float:
    contacts = motion[:, 0:4] > 0.5
    if len(motion) < 2 or not contacts[:-1].any():
        return 0.5
    # Higher is more stable, based on contact continuity.
    pairs = contacts[1:] & contacts[:-1]
    return float(pairs.mean())


def root_direction(motion: np.ndarray) -> np.ndarray:
    root = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    if len(root) < 2:
        return np.zeros((2,), dtype=np.float32)
    v = root[-1] - root[0]
    n = np.linalg.norm(v)
    if n <= 1e-8:
        return np.zeros((2,), dtype=np.float32)
    return (v / n).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mmr_checkpoint", required=True)
    p.add_argument("--motion_dir", required=True)
    p.add_argument("--out", default="data/dunhuang_rag_db/mmr_rag_index.npz")
    p.add_argument("--edge_checkpoint", default="", help="Optional EDGE checkpoint normalizer when motion files are physical-space.")
    p.add_argument("--pose_space", default="normalized", choices=["normalized", "physical"])
    p.add_argument("--seq_len", type=int, default=150)
    p.add_argument("--window", type=int, default=150)
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_mmr_model(args.mmr_checkpoint, device=device)
    normalizer = load_normalizer_from_checkpoint(args.edge_checkpoint) if args.edge_checkpoint else None

    embeddings, poses, sources, source_frames, dirs = [], [], [], [], []
    energies, contacts, rhythm_json = [], [], []
    files = list(iter_motion_files(args.motion_dir))
    for path in tqdm(files, desc="build mmr rag"):
        try:
            motion = load_motion_151(path)
            motion = normalize_motion_if_needed(motion, normalizer, args.pose_space)
        except Exception as exc:
            print("skip", path, exc)
            continue
        if len(motion) < 2:
            continue
        starts = [0] if len(motion) < args.window else list(range(0, len(motion) - args.window + 1, args.stride))
        for start in starts:
            clip = motion[start : start + args.window]
            if len(clip) < 2:
                continue
            clip_rs = resample_sequence(clip, args.seq_len)
            with torch.no_grad():
                z = model.encode_motion(torch.from_numpy(clip_rs[None]).float().to(device))[0]
            z = z.detach().cpu().numpy().astype(np.float32)
            center = clip_rs[len(clip_rs) // 2].astype(np.float32)
            summ = rhythm_summary_from_motion(clip_rs)
            embeddings.append(z)
            poses.append(center)
            sources.append(str(path))
            source_frames.append(int(start + len(clip) // 2))
            dirs.append(root_direction(clip_rs))
            energies.append(float(summ["motion_energy_mean"]))
            contacts.append(contact_stability(clip_rs))
            rhythm_json.append(summ)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        motion_embeddings=np.asarray(embeddings, dtype=np.float32),
        poses=np.asarray(poses, dtype=np.float32),
        source=np.asarray(sources),
        source_frame=np.asarray(source_frames, dtype=np.int32),
        root_vel=np.asarray(dirs, dtype=np.float32),
        motion_energy=np.asarray(energies, dtype=np.float32),
        contact_stability=np.asarray(contacts, dtype=np.float32),
    )
    meta = {
        "type": "mmr_clip_rag_index",
        "mmr_checkpoint": args.mmr_checkpoint,
        "motion_dir": args.motion_dir,
        "clip_count": len(embeddings),
        "seq_len": args.seq_len,
        "window": args.window,
        "stride": args.stride,
        "pose_space": args.pose_space,
    }
    json.dump(meta, open(out.with_suffix(".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("✅ saved MMR-RAG index:", out)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
