"""Build ChoreoRAG motion-unit database for EDGE.

This implements the "motion-unit RAG instead of single-pose RAG" layer.
It creates a .npz file directly readable by the replacement
auto_keyframe_planner.py.

Example:
python build_choreo_unit_rag_db.py \
  --input_dir data/dunhuang_bvh/processed \
  --out data/dunhuang_choreo_unit_rag/index.npz \
  --checkpoint runs/train/exp17_dunhuang_stage1_from_aist300/weights/train-30.pt \
  --pose_space normalized \
  --unit_len 45 \
  --stride 15
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from dataset.quaternion import ax_to_6v
    from dataset.preprocess import vectorize_many, Normalizer
    from vis import SMPLSkeleton
except Exception:
    ax_to_6v = None
    vectorize_many = None
    Normalizer = None
    SMPLSkeleton = None

try:
    from model.text_bridge_encoder import TextBridgeEncoder
except Exception:
    from text_bridge_encoder import TextBridgeEncoder  # type: ignore


ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
CONTACT_SLICE = slice(0, 4)
ROT_SLICE = slice(7, 151)


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_normalizer_from_checkpoint(checkpoint_path: str):
    if not checkpoint_path:
        return None
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    norm_data = ckpt.get("normalizer") if isinstance(ckpt, dict) else None
    if norm_data is None:
        return None
    if hasattr(norm_data, "mean") and hasattr(norm_data, "std"):
        return norm_data
    if isinstance(norm_data, dict) and "mean" in norm_data and "std" in norm_data:
        if Normalizer is None:
            class DummyNormalizer:
                pass
            normalizer = DummyNormalizer()
        else:
            normalizer = Normalizer(torch.zeros((1, 1, 151)))
        normalizer.mean = np.asarray(norm_data["mean"], dtype=np.float32)
        normalizer.std = np.asarray(norm_data["std"], dtype=np.float32)
        return normalizer
    return None


def normalize_motion(motion: np.ndarray, normalizer):
    if normalizer is None:
        raise ValueError("--pose_space normalized requires --checkpoint with normalizer")
    mt = torch.from_numpy(np.asarray(motion, dtype=np.float32)).float()
    if mt.ndim == 2:
        return _to_numpy(normalizer.normalize(mt[None]))[0].astype(np.float32)
    if mt.ndim == 3:
        return _to_numpy(normalizer.normalize(mt)).astype(np.float32)
    raise ValueError(f"Unsupported motion ndim={mt.ndim}")


def pkl_to_motion_151(path: Path) -> Optional[np.ndarray]:
    if SMPLSkeleton is None or ax_to_6v is None or vectorize_many is None:
        raise ImportError("Run this inside EDGE repo; dataset.quaternion/preprocess/vis are required.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "pos" not in data or "q" not in data:
        print(f"⚠️ Skip {path}: missing pos/q")
        return None

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


def root_dir(unit: np.ndarray, start: int, end: int) -> np.ndarray:
    start = int(np.clip(start, 0, len(unit) - 1))
    end = int(np.clip(end, 0, len(unit) - 1))
    v = unit[end, [ROOT_X_IDX, ROOT_Z_IDX]] - unit[start, [ROOT_X_IDX, ROOT_Z_IDX]]
    n = float(np.linalg.norm(v))
    return np.zeros((2,), dtype=np.float32) if n <= 1e-8 else (v / n).astype(np.float32)


def unit_stats(unit: np.ndarray) -> Dict[str, float]:
    if len(unit) <= 1:
        return dict(motion_energy=0.0, root_speed=0.0, upper_activity=0.0, lower_activity=0.0, contact_stability=1.0, spatial_range=0.0, turning=0.0)

    diff = unit[1:] - unit[:-1]
    rot_diff = diff[:, ROT_SLICE]
    root = unit[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    root_vel = root[1:] - root[:-1]

    # Coarse joint regions in 24 joints * 6D.
    lower_slice = slice(7 + 6 * 1, 7 + 6 * 12)
    upper_slice = slice(7 + 6 * 12, 7 + 6 * 22)

    contacts = (unit[:, CONTACT_SLICE] > 0.5).astype(np.float32)
    contact_changes = float(np.abs(contacts[1:] - contacts[:-1]).mean()) if len(contacts) > 1 else 0.0
    contact_stability = 1.0 - float(np.clip(contact_changes, 0.0, 1.0))

    if len(root_vel) > 2:
        v1, v2 = root_vel[:-1], root_vel[1:]
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        cos = np.sum(v1 * v2, axis=1) / np.clip(n1 * n2, 1e-8, None)
        turning = float(np.mean(1.0 - np.clip(cos, -1.0, 1.0)))
    else:
        turning = 0.0

    return {
        "motion_energy": float(np.sqrt(np.mean(rot_diff ** 2))),
        "root_speed": float(np.linalg.norm(root_vel, axis=1).mean()) if len(root_vel) else 0.0,
        "upper_activity": float(np.sqrt(np.mean(diff[:, upper_slice] ** 2))),
        "lower_activity": float(np.sqrt(np.mean(diff[:, lower_slice] ** 2))),
        "contact_stability": contact_stability,
        "spatial_range": float(np.linalg.norm(root.max(axis=0) - root.min(axis=0))),
        "turning": turning,
    }


def robust_ranges(records: List[Dict]) -> Dict[str, Tuple[float, float]]:
    keys = ["motion_energy", "root_speed", "upper_activity", "lower_activity", "spatial_range", "turning"]
    out = {}
    for k in keys:
        vals = np.asarray([float(r["stats"][k]) for r in records], dtype=np.float32)
        lo, hi = np.percentile(vals, [10, 90])
        if float(hi - lo) <= 1e-8:
            lo, hi = float(vals.min()), float(vals.max() + 1e-6)
        out[k] = (float(lo), float(hi))
    return out


def norm_value(v: float, lo: float, hi: float) -> float:
    return float(np.clip((float(v) - float(lo)) / max(float(hi) - float(lo), 1e-8), 0.0, 1.0))


def bucket(v: float, lo: float, hi: float, labels) -> str:
    x = norm_value(v, lo, hi)
    if x < 0.33:
        return labels[0]
    if x < 0.66:
        return labels[1]
    return labels[2]


def caption_from_stats(stats: Dict[str, float], ranges: Dict[str, Tuple[float, float]]) -> str:
    energy = bucket(stats["motion_energy"], *ranges["motion_energy"], ("低能量", "中等能量", "高能量"))
    speed = bucket(stats["root_speed"], *ranges["root_speed"], ("缓慢移动", "平稳移动", "快速移动"))
    upper = bucket(stats["upper_activity"], *ranges["upper_activity"], ("上肢含蓄", "上肢舒展", "上肢大幅展开"))
    lower = bucket(stats["lower_activity"], *ranges["lower_activity"], ("下肢稳定", "步伐变化", "下肢活跃"))
    space = bucket(stats["spatial_range"], *ranges["spatial_range"], ("小空间", "中等空间", "大空间"))
    turn = "带有旋转" if norm_value(stats["turning"], *ranges["turning"]) > 0.55 else "方向平稳"
    contact = "重心稳定" if stats["contact_stability"] >= 0.70 else "脚步切换明显"
    return f"{energy}，{speed}，{upper}，{lower}，{space}，{turn}，{contact}，敦煌舞，飞天风格"


def collect_units(input_dir: Path, unit_len: int, stride: int, min_unit_len: int) -> List[Dict]:
    files = sorted(input_dir.glob("*.pkl"))
    if not files and (input_dir / "processed").is_dir():
        files = sorted((input_dir / "processed").glob("*.pkl"))
    records = []

    for file in files:
        motion = pkl_to_motion_151(file)
        if motion is None or len(motion) < min_unit_len:
            continue
        L = min(int(unit_len), len(motion))
        for start in range(0, len(motion) - L + 1, int(stride)):
            end = start + L
            unit = motion[start:end].astype(np.float32)
            c = len(unit) // 2
            stats = unit_stats(unit)
            records.append({
                "source": str(file),
                "unit_start": int(start),
                "unit_center": int(start + c),
                "unit_end": int(end - 1),
                "unit_motion_physical": unit,
                "entry_pose_physical": unit[0].copy(),
                "center_pose_physical": unit[c].copy(),
                "exit_pose_physical": unit[-1].copy(),
                "contact_entry": (unit[0, CONTACT_SLICE] > 0.5).astype(np.float32),
                "contact_center": (unit[c, CONTACT_SLICE] > 0.5).astype(np.float32),
                "contact_exit": (unit[-1, CONTACT_SLICE] > 0.5).astype(np.float32),
                "root_dir_entry": root_dir(unit, 0, min(5, len(unit) - 1)),
                "root_dir_exit": root_dir(unit, max(0, len(unit) - 6), len(unit) - 1),
                "root_dir_full": root_dir(unit, 0, len(unit) - 1),
                "stats": stats,
            })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/dunhuang_bvh/processed")
    parser.add_argument("--out", type=str, default="data/dunhuang_choreo_unit_rag/index.npz")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--pose_space", choices=["physical", "normalized"], default="normalized")
    parser.add_argument("--unit_len", type=int, default=45)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--min_unit_len", type=int, default=30)
    parser.add_argument("--text_model", type=str, default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--text_device", type=str, default="cpu")
    parser.add_argument("--fallback_dim", type=int, default=384)
    parser.add_argument("--max_units", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    normalizer = load_normalizer_from_checkpoint(args.checkpoint) if args.pose_space == "normalized" else None
    records = collect_units(Path(args.input_dir), args.unit_len, args.stride, args.min_unit_len)
    if args.max_units > 0:
        records = records[: args.max_units]
    if not records:
        raise RuntimeError(f"No valid motion units found in {args.input_dir}")

    ranges = robust_ranges(records)
    captions = [caption_from_stats(r["stats"], ranges) for r in records]

    encoder = TextBridgeEncoder(model_name=args.text_model, device=args.text_device, fallback_dim=args.fallback_dim)
    emb = encoder.encode(captions).astype(np.float32)

    unit_physical = np.stack([r["unit_motion_physical"] for r in records], axis=0).astype(np.float32)
    poses_physical = np.stack([r["center_pose_physical"] for r in records], axis=0).astype(np.float32)
    entry_physical = np.stack([r["entry_pose_physical"] for r in records], axis=0).astype(np.float32)
    exit_physical = np.stack([r["exit_pose_physical"] for r in records], axis=0).astype(np.float32)

    if args.pose_space == "normalized":
        unit_motions = normalize_motion(unit_physical, normalizer)
        poses = normalize_motion(poses_physical, normalizer)
        entry_poses = normalize_motion(entry_physical, normalizer)
        exit_poses = normalize_motion(exit_physical, normalizer)
    else:
        unit_motions, poses, entry_poses, exit_poses = unit_physical, poses_physical, entry_physical, exit_physical

    stats = {k: np.asarray([float(r["stats"][k]) for r in records], dtype=np.float32)
             for k in ["motion_energy", "root_speed", "upper_activity", "lower_activity", "contact_stability", "spatial_range", "turning"]}

    np.savez_compressed(
        out,
        db_type=np.asarray(["choreo_unit_rag"]),
        pose_space=np.asarray([args.pose_space]),
        unit_len=np.asarray([args.unit_len], dtype=np.int64),
        poses=poses.astype(np.float32),
        entry_poses=entry_poses.astype(np.float32),
        exit_poses=exit_poses.astype(np.float32),
        unit_motions=unit_motions.astype(np.float32),
        unit_motions_physical=unit_physical.astype(np.float32),
        source=np.asarray([r["source"] for r in records]),
        source_frame=np.asarray([r["unit_center"] for r in records], dtype=np.int64),
        unit_start=np.asarray([r["unit_start"] for r in records], dtype=np.int64),
        unit_center=np.asarray([r["unit_center"] for r in records], dtype=np.int64),
        unit_end=np.asarray([r["unit_end"] for r in records], dtype=np.int64),
        root_vel=np.stack([r["root_dir_full"] for r in records], axis=0).astype(np.float32),
        root_dir_entry=np.stack([r["root_dir_entry"] for r in records], axis=0).astype(np.float32),
        root_dir_exit=np.stack([r["root_dir_exit"] for r in records], axis=0).astype(np.float32),
        contact_entry=np.stack([r["contact_entry"] for r in records], axis=0).astype(np.float32),
        contact_center=np.stack([r["contact_center"] for r in records], axis=0).astype(np.float32),
        contact_exit=np.stack([r["contact_exit"] for r in records], axis=0).astype(np.float32),
        motion_text=np.asarray(captions),
        motion_text_embedding=emb,
        motion_embedding=emb,  # backward-compatible field
        text_model=np.asarray([args.text_model]),
        text_backend=np.asarray([encoder.backend]),
        **stats,
    )

    meta = {
        "db_type": "choreo_unit_rag",
        "count": len(records),
        "input_dir": args.input_dir,
        "out": str(out),
        "checkpoint": args.checkpoint,
        "pose_space": args.pose_space,
        "unit_len": args.unit_len,
        "stride": args.stride,
        "text_model": args.text_model,
        "text_backend": encoder.backend,
        "stat_ranges": ranges,
    }
    with open(out.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Choreo-unit RAG saved: {out}")
    print(f"   units={len(records)} unit_len={args.unit_len} stride={args.stride} pose_space={args.pose_space}")
    print(f"   text_backend={encoder.backend} embedding_dim={emb.shape[-1]}")
    print(f"   meta={out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
