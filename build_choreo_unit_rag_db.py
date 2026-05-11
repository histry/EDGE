"""Build footstep-aware ChoreoRAG motion-unit database for EDGE.

Drop-in replacement for build_choreo_unit_rag_db.py.

Adds the original expressiveness_score plus:
  locomotion_score = root/lower/spatial/turning motion
  footstep_score   = contact switching + left/right alternation + root/lower sync
  mobile_score     = blended locomotion + footstep score

These fields support velocity-conditioned routing during inference without
retraining the diffusion model.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from footstep_phase_utils import (
    CONTACT_SLICE,
    ROOT_X_IDX,
    ROOT_Z_IDX,
    add_dual_scores,
    robust_norm,
    unit_basic_stats,
)

try:
    from dataset.quaternion import ax_to_6v
    from dataset.preprocess import vectorize_many, Normalizer
    from vis import SMPLSkeleton
except Exception:  # pragma: no cover
    ax_to_6v = None
    vectorize_many = None
    Normalizer = None
    SMPLSkeleton = None

try:
    from model.text_bridge_encoder import TextBridgeEncoder
except Exception:  # pragma: no cover
    from text_bridge_encoder import TextBridgeEncoder  # type: ignore


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
            stats = unit_basic_stats(unit)
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


def compute_stats_arrays(records: List[Dict]) -> Dict[str, np.ndarray]:
    raw_keys = [
        "motion_energy", "root_speed", "upper_activity", "lower_activity",
        "spatial_range", "turning", "contact_stability", "contact_switch",
        "alternating_foot_phase", "root_lower_sync",
    ]
    stats = {k: np.asarray([float(r["stats"].get(k, 0.0)) for r in records], dtype=np.float32) for k in raw_keys}
    # Backward-compatible alias used by v10 planner.
    stats["unit_energy"] = stats["motion_energy"].copy()
    for k in ["motion_energy", "unit_energy", "root_speed", "upper_activity", "lower_activity", "spatial_range", "turning", "contact_switch", "root_lower_sync"]:
        stats[f"{k}_norm"] = robust_norm(stats[k])[0]
    add_dual_scores(stats)
    return stats


def bucket(v: float, arr: np.ndarray, labels) -> str:
    lo, hi = np.percentile(arr.astype(np.float32), [10, 90]) if len(arr) else (0.0, 1.0)
    x = np.clip((float(v) - float(lo)) / max(float(hi - lo), 1e-8), 0.0, 1.0)
    if x < 0.33:
        return labels[0]
    if x < 0.66:
        return labels[1]
    return labels[2]


def captions_from_stats(records: List[Dict], stats: Dict[str, np.ndarray]) -> List[str]:
    captions = []
    for r in records:
        s = r["stats"]
        energy = bucket(s["motion_energy"], stats["motion_energy"], ("低能量", "中等能量", "高能量"))
        speed = bucket(s["root_speed"], stats["root_speed"], ("缓慢移动", "平稳移动", "快速移动"))
        upper = bucket(s["upper_activity"], stats["upper_activity"], ("上肢含蓄", "上肢舒展", "上肢大幅展开"))
        lower = bucket(s["lower_activity"], stats["lower_activity"], ("下肢稳定", "步伐变化", "下肢活跃"))
        space = bucket(s["spatial_range"], stats["spatial_range"], ("小空间", "中等空间", "大空间"))
        turn = "带有旋转" if s.get("turning", 0.0) >= np.percentile(stats["turning"], 70) else "方向平稳"
        contact = "脚步切换明显" if s.get("contact_switch", 0.0) >= np.percentile(stats["contact_switch"], 65) else "重心稳定"
        mobile = "移动步态明显" if s.get("root_speed", 0.0) >= np.percentile(stats["root_speed"], 65) else "原地舞姿"
        captions.append(f"{energy}，{speed}，{upper}，{lower}，{space}，{turn}，{contact}，{mobile}，敦煌舞，飞天风格")
    return captions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/dunhuang_bvh/processed")
    parser.add_argument("--out", type=str, default="data/dunhuang_choreo_unit_rag/index_footstep.npz")
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

    stats = compute_stats_arrays(records)
    captions = captions_from_stats(records, stats)
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

    np.savez_compressed(
        out,
        db_type=np.asarray(["footstep_aware_choreo_unit_rag"]),
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
        motion_embedding=emb,
        text_model=np.asarray([args.text_model]),
        text_backend=np.asarray([encoder.backend]),
        **stats,
    )

    meta = {
        "db_type": "footstep_aware_choreo_unit_rag",
        "count": len(records),
        "input_dir": args.input_dir,
        "out": str(out),
        "checkpoint": args.checkpoint,
        "pose_space": args.pose_space,
        "unit_len": args.unit_len,
        "stride": args.stride,
        "text_model": args.text_model,
        "text_backend": encoder.backend,
        "score_fields": ["expressiveness_score", "locomotion_score", "footstep_score", "mobile_score"],
        "score_formula": {
            "locomotion_score": "0.45 root_speed + 0.35 lower_activity + 0.15 spatial_range + 0.05 turning",
            "footstep_score": "0.35 contact_switch + 0.30 alternating_foot_phase + 0.20 root_lower_sync + 0.15 contact_stability",
            "mobile_score": "0.60 locomotion_score + 0.40 footstep_score",
        },
    }
    with open(out.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Footstep-aware Choreo-unit RAG saved: {out}")
    print(f"   units={len(records)} unit_len={args.unit_len} stride={args.stride} pose_space={args.pose_space}")
    print(f"   expressiveness mean={float(stats['expressiveness_score'].mean()):.3f} p90={float(np.percentile(stats['expressiveness_score'], 90)):.3f}")
    print(f"   mobile mean={float(stats['mobile_score'].mean()):.3f} p90={float(np.percentile(stats['mobile_score'], 90)):.3f}")
    print(f"   footstep mean={float(stats['footstep_score'].mean()):.3f} p90={float(np.percentile(stats['footstep_score'], 90)):.3f}")
    print(f"   meta={out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
