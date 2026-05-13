#!/usr/bin/env python3
"""Train Turn-aware Event Refiner v2.

Improvements over the first proof-of-concept:
  1. multi pseudo-target training
  2. event-weighted body-part losses
  3. small bounded residual on top of no-train turn-aware anchor
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from turn_aware_event_refiner import RefinerConfig, TurnAwareEventRefiner, make_checkpoint, refiner_loss
from turn_aware_event_utils import TurnEventConfig, event_feature_matrix


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def load_motion(path: str) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[-1] != 151:
        raise ValueError(f"Expected [T,151], got {arr.shape}: {path}")
    return arr


def load_rows(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out_ckpt", default="runs/turn_event_refiner/turn_event_refiner_v2.pt")
    ap.add_argument("--epochs", type=int, default=env_int("EDGE_TURN_REFINER_EPOCHS", 400))
    ap.add_argument("--lr", type=float, default=env_float("EDGE_TURN_REFINER_LR", 7e-4))
    ap.add_argument("--hidden", type=int, default=env_int("EDGE_TURN_REFINER_HIDDEN", 256))
    ap.add_argument("--depth", type=int, default=env_int("EDGE_TURN_REFINER_DEPTH", 3))
    ap.add_argument("--dropout", type=float, default=env_float("EDGE_TURN_REFINER_DROPOUT", 0.05))
    ap.add_argument("--max_delta", type=float, default=env_float("EDGE_TURN_REFINER_MAX_DELTA", 0.14))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true", help="Allow training without EDGE_TURN_EVENT_REFINER_TRAIN=1")
    args = ap.parse_args()

    if os.environ.get("EDGE_TURN_EVENT_REFINER_TRAIN", "0") != "1" and not args.force:
        raise SystemExit("Set EDGE_TURN_EVENT_REFINER_TRAIN=1 or pass --force to train.")

    rows = load_rows(args.pairs)
    first = rows[0]
    cfg_ev = TurnEventConfig.from_env(seq_len=int(first.get("seq_len", 150)), count=int(first.get("count", 5)))
    event_np, feature_names, ev_report = event_feature_matrix(first["trajectory"], cfg_ev)

    cfg = RefinerConfig(
        event_dim=int(event_np.shape[-1]),
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
        max_delta=args.max_delta,
    )
    device = torch.device(args.device)
    model = TurnAwareEventRefiner(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Cache tensors.
    samples = []
    for row in rows:
        ev_np, _, _ = event_feature_matrix(row["trajectory"], cfg_ev)
        base = load_motion(row["base"])
        anchor = load_motion(row["anchor"])
        target = load_motion(row["target"])
        T = min(len(base), len(anchor), len(target), len(ev_np))
        sample = {
            "base": torch.from_numpy(base[:T]).float().to(device),
            "anchor": torch.from_numpy(anchor[:T]).float().to(device),
            "target": torch.from_numpy(target[:T]).float().to(device),
            "event": torch.from_numpy(ev_np[:T]).float().to(device),
            "weight": float(row.get("weight", 1.0)),
            "target_path": row["target"],
        }
        samples.append(sample)

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        total = torch.zeros((), device=device)
        parts_acc: Dict[str, float] = {}
        wsum = 0.0
        for s in samples:
            pred = model(s["base"], s["anchor"], s["event"])
            loss, parts = refiner_loss(pred, s["target"], s["base"], s["anchor"], s["event"], cfg)
            w = float(s["weight"])
            total = total + loss * w
            wsum += w
            for k, v in parts.items():
                parts_acc[k] = parts_acc.get(k, 0.0) + v * w
        total = total / max(wsum, 1e-8)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            msg = f"epoch={epoch:04d} loss={float(total.detach().cpu()):.6f}"
            print(msg)
            history.append({"epoch": epoch, "loss": float(total.detach().cpu()), **{k: v / max(wsum, 1e-8) for k, v in parts_acc.items()}})

    ckpt = make_checkpoint(
        model,
        cfg,
        feature_names,
        extra={
            "pairs": rows,
            "history": history,
            "event_report": {
                "event_centers": ev_report["event_centers"],
                "support_frames": ev_report["support_frames"],
                "expressive_frames": ev_report["expressive_frames"],
            },
        },
    )
    torch.save(ckpt, out_ckpt)
    print(f"✅ saved refiner checkpoint: {out_ckpt}")


if __name__ == "__main__":
    main()
