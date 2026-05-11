#!/usr/bin/env python3
"""Audit Dunhuang Train/Val source isolation and X/Z trajectory contract.

Run from the EDGE repository root after replacing dataset/dance_dataset.py:

  EDGE_DUNHUANG_SPLIT_REPORT_DIR=output/split_audit \
  python tools/audit_dunhuang_split_xz_contract.py \
    --data_path data/dunhuang_bvh/processed \
    --seq_len 150 \
    --split_ratio 0.9 \
    --split_seed 42

The script intentionally imports DunhuangDataset and constructs both train and
validation datasets. It fails if:
  - train/val original source ids overlap;
  - validation is empty in strict mode;
  - cond['trajectory'] differs from normalized motion root X/Z dims [4,6].
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure repository root is importable when running:
#   python tools/audit_dunhuang_split_xz_contract.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dance_dataset import DunhuangDataset, TRAJ_ROOT_XZ_IDXS


def _tensor_to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def audit_dataset(ds: DunhuangDataset, name: str, max_items: int = 64) -> dict:
    n = len(ds)
    if n == 0:
        raise RuntimeError(f"{name} dataset is empty")
    max_delta = 0.0
    checked = min(n, max_items)
    for i in range(checked):
        pose, cond, window_id, audio_source = ds[i]
        if "trajectory" not in cond:
            continue
        root_xz = _tensor_to_numpy(pose)[:, TRAJ_ROOT_XZ_IDXS]
        traj = _tensor_to_numpy(cond["trajectory"])
        delta = float(np.max(np.abs(root_xz - traj)))
        max_delta = max(max_delta, delta)
        if delta > 1e-4:
            raise RuntimeError(
                f"{name} X/Z trajectory mismatch at idx={i}, window_id={window_id}, max_delta={delta:.6g}"
            )
    return {
        "name": name,
        "windows": int(n),
        "sources": list(getattr(ds, "selected_source_ids", [])),
        "num_sources": int(len(getattr(ds, "selected_source_ids", []))),
        "checked_items": int(checked),
        "max_abs_normalized_root_xz_minus_traj": float(max_delta),
        "split_report": getattr(ds, "split_report", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/dunhuang_bvh/processed")
    parser.add_argument("--seq_len", type=int, default=150)
    parser.add_argument("--audio_dim", type=int, default=803)
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--return_traj", action="store_true", default=True)
    parser.add_argument("--out", default="output/dunhuang_split_xz_audit.json")
    parser.add_argument("--max_items", type=int, default=64)
    args = parser.parse_args()

    train = DunhuangDataset(
        data_path=args.data_path,
        train=True,
        seq_len=args.seq_len,
        audio_dim=args.audio_dim,
        overlap=args.overlap,
        normalizer=None,
        return_traj=True,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        audio_pairing_mode="none",
        audio_sample_mode="zero",
        traj_aug_prob=0.0,
    )
    val = DunhuangDataset(
        data_path=args.data_path,
        train=False,
        seq_len=args.seq_len,
        audio_dim=args.audio_dim,
        overlap=args.overlap,
        normalizer=train.normalizer,
        return_traj=True,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        audio_pairing_mode="none",
        audio_sample_mode="zero",
        traj_aug_prob=0.0,
    )

    train_sources = set(getattr(train, "selected_source_ids", []))
    val_sources = set(getattr(val, "selected_source_ids", []))
    overlap_sources = sorted(train_sources & val_sources)
    if overlap_sources:
        raise RuntimeError(f"CRITICAL DATA LEAKAGE: overlapping source ids: {overlap_sources[:50]}")

    report = {
        "ok": True,
        "data_path": args.data_path,
        "seq_len": args.seq_len,
        "trajectory_plane": "xz",
        "trajectory_root_indices_151d": TRAJ_ROOT_XZ_IDXS,
        "train": audit_dataset(train, "train", max_items=args.max_items),
        "val": audit_dataset(val, "val", max_items=args.max_items),
        "overlap_sources": overlap_sources,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Audit passed. Report saved: {out}")
    print(f"   train_sources={len(train_sources)} val_sources={len(val_sources)}")
    print(f"   train_windows={len(train)} val_windows={len(val)}")


if __name__ == "__main__":
    main()
