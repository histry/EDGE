import argparse
import glob
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from dataset.quaternion import ax_to_6v
from dataset.preprocess import vectorize_many, Normalizer
from vis import SMPLSkeleton


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_normalizer(checkpoint_path: str):
    if not checkpoint_path:
        return None
    ckpt = torch_load(checkpoint_path)
    norm_data = ckpt.get("normalizer") if isinstance(ckpt, dict) else None
    if norm_data is None:
        return None

    if hasattr(norm_data, "mean") and hasattr(norm_data, "std"):
        return norm_data

    if isinstance(norm_data, dict) and "mean" in norm_data and "std" in norm_data:
        normalizer = Normalizer(torch.zeros((1, 1, 151)))
        normalizer.mean = np.asarray(norm_data["mean"], dtype=np.float32)
        normalizer.std = np.asarray(norm_data["std"], dtype=np.float32)
        return normalizer

    return None


def pkl_to_motion_151(path: str) -> np.ndarray:
    data = pickle.load(open(path, "rb"))
    if "pos" not in data or "q" not in data:
        raise ValueError(f"{path} missing pos/q")

    pos = torch.tensor(data["pos"], dtype=torch.float32).unsqueeze(0)
    q = torch.tensor(data["q"], dtype=torch.float32).unsqueeze(0)
    q = q.reshape(q.shape[0], q.shape[1], -1, 3)

    smpl = SMPLSkeleton()
    with torch.no_grad():
        joints = smpl.forward(q, pos)
        feet = joints[:, :, [7, 8, 10, 11]]
        feetv = torch.zeros(feet.shape[:3])
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).to(q)
        q_6v = ax_to_6v(q)
        motion = vectorize_many([contacts, pos, q_6v])

    return motion[0].detach().cpu().numpy().astype(np.float32)


def normalize_motion(motion: np.ndarray, normalizer):
    if normalizer is None:
        return motion.astype(np.float32)
    motion_t = torch.from_numpy(motion[None]).float()
    out = normalizer.normalize(motion_t)
    if torch.is_tensor(out):
        return out.detach().cpu().numpy()[0].astype(np.float32)
    return np.asarray(out)[0].astype(np.float32)


def build_index(
    data_path: str,
    checkpoint: str,
    out: str,
    sample_stride: int = 5,
    max_files: int = 0,
):
    normalizer = load_normalizer(checkpoint)
    if normalizer is None:
        print("⚠️ checkpoint 中没有 normalizer：将输出 physical-space poses。")
    else:
        print("✅ loaded normalizer from checkpoint")

    files = sorted(glob.glob(str(Path(data_path) / "*.pkl")))
    if not files:
        files = sorted(glob.glob(str(Path(data_path) / "processed" / "*.pkl")))
    if not files:
        raise FileNotFoundError(f"No .pkl files found under {data_path}")

    if max_files and max_files > 0:
        files = files[:max_files]

    poses = []
    source = []
    source_frame = []
    root_vel = []

    for path in files:
        try:
            motion = pkl_to_motion_151(path)
        except Exception as exc:
            print(f"⚠️ skip {path}: {exc}")
            continue

        motion = normalize_motion(motion, normalizer)

        root_xz = motion[:, [4, 6]].astype(np.float32)
        vel = np.zeros_like(root_xz)
        if len(root_xz) > 1:
            vel[1:] = root_xz[1:] - root_xz[:-1]

        for frame in range(0, len(motion), max(1, int(sample_stride))):
            poses.append(motion[frame].astype(np.float32))
            source.append(str(path))
            source_frame.append(int(frame))
            root_vel.append(vel[frame].astype(np.float32))

    if not poses:
        raise RuntimeError("No valid poses collected.")

    poses = np.asarray(poses, dtype=np.float32)
    source = np.asarray(source, dtype=object)
    source_frame = np.asarray(source_frame, dtype=np.int32)
    root_vel = np.asarray(root_vel, dtype=np.float32)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        poses=poses,
        source=source,
        source_frame=source_frame,
        root_vel=root_vel,
    )

    meta = {
        "data_path": data_path,
        "checkpoint": checkpoint,
        "pose_count": int(len(poses)),
        "source_file_count": int(len(files)),
        "sample_stride": int(sample_stride),
        "pose_space": "normalized" if normalizer is not None else "physical",
        "out": str(out_path),
    }
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ saved RAG index: {out_path}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Build a lightweight Dunhuang RAG pose index.")
    parser.add_argument("--data_path", default="data/dunhuang_bvh/processed")
    parser.add_argument("--checkpoint", default="runs/best/dunhuang_stage1_best_exp17_train30.pt")
    parser.add_argument("--out", default="data/dunhuang_rag_db/rag_index.npz")
    parser.add_argument("--sample_stride", type=int, default=5)
    parser.add_argument("--max_files", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(
        data_path=args.data_path,
        checkpoint=args.checkpoint,
        out=args.out,
        sample_stride=args.sample_stride,
        max_files=args.max_files,
    )
