#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from turn_aware_event_refiner import TurnAwareEventRefiner, TurnEventRefinerConfig, save_checkpoint
from turn_aware_event_utils import detect_turn_events


ROOT_X_IDX = 4
ROOT_Z_IDX = 6
CONTACT_SLICE = slice(0, 4)
LOWER_SL = slice(7 + 6 * 0, 7 + 6 * 12)  # rough lower/pelvis-heavy range fallback
TORSO_SL = slice(7 + 6 * 12, 7 + 6 * 18)
UPPER_SL = slice(7 + 6 * 18, 151)


def parse_pairs(path: str) -> List[Dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No pairs in {path}")
    return rows


class PairDataset(Dataset):
    def __init__(self, rows: List[Dict], seq_len: int = 150):
        self.rows = rows
        self.seq_len = seq_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        base = np.load(r["base"], allow_pickle=True).astype(np.float32)
        target = np.load(r["target"], allow_pickle=True).astype(np.float32)
        T = min(len(base), len(target), self.seq_len)
        base = base[:T]
        target = target[:T]
        if "event_features" in r and r["event_features"]:
            event = np.load(r["event_features"], allow_pickle=True).astype(np.float32)[:T]
        else:
            rep = detect_turn_events(r["trajectory"], seq_len=T, count=int(r.get("count", 5)))
            event = rep["event_features"].astype(np.float32)
        return torch.from_numpy(base), torch.from_numpy(target), torch.from_numpy(event)


def masked_losses(pred, target, base, event, residual):
    # event dims from turn_aware_event_utils:
    # 6 turn_gate, 7 support_gate, 8 expressive_gate, 9 settle_gate
    turn_gate = event[..., 6:7]
    support_gate = event[..., 7:8]
    expr_gate = event[..., 8:9]
    body_mask = torch.ones_like(pred)
    body_mask[..., ROOT_X_IDX] = 0.0
    body_mask[..., ROOT_Z_IDX] = 0.0

    recon = F.smooth_l1_loss(pred * body_mask, target * body_mask)
    support = F.smooth_l1_loss(pred[..., CONTACT_SLICE] * support_gate, target[..., CONTACT_SLICE] * support_gate)
    lower = F.smooth_l1_loss(pred[..., LOWER_SL] * support_gate, target[..., LOWER_SL] * support_gate)
    expr = F.smooth_l1_loss(pred[..., TORSO_SL] * expr_gate, target[..., TORSO_SL] * expr_gate) + F.smooth_l1_loss(
        pred[..., UPPER_SL] * expr_gate, target[..., UPPER_SL] * expr_gate
    )
    # smooth residual, not motion itself, to avoid jitter
    if residual.shape[1] > 2:
        jerk = (residual[:, 2:] - 2 * residual[:, 1:-1] + residual[:, :-2]).pow(2).mean()
    else:
        jerk = residual.pow(2).mean() * 0.0
    root = F.mse_loss(pred[..., [ROOT_X_IDX, ROOT_Z_IDX]], base[..., [ROOT_X_IDX, ROOT_Z_IDX]])
    total = recon + 0.8 * support + 0.8 * lower + 1.0 * expr + 0.05 * jerk + 5.0 * root
    return total, {"recon": recon, "support": support, "lower": lower, "expr": expr, "jerk": jerk, "root": root}


def collate(batch):
    # current examples are all T=150; keep a simple collate
    base, target, event = zip(*batch)
    return torch.stack(base), torch.stack(target), torch.stack(event)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="JSONL with base,target,trajectory[,event_features]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rows = parse_pairs(args.pairs)
    ds = PairDataset(rows)
    sample_event = ds[0][2]
    cfg = TurnEventRefinerConfig(event_dim=int(sample_event.shape[-1]), hidden_dim=args.hidden_dim, layers=args.layers)
    model = TurnAwareEventRefiner(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for base, target, event in loader:
            base = base.to(args.device)
            target = target.to(args.device)
            event = event.to(args.device)
            pred, residual = model(base, event)
            loss, parts = masked_losses(pred, target, base, event, residual)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": ep, "loss": float(np.mean(losses))}
        log.append(row)
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            print(f"epoch={ep:04d} loss={row['loss']:.6f}")
    save_checkpoint(args.out, model, cfg, {"pairs": args.pairs, "epochs": args.epochs, "log": log})
    Path(args.out).with_suffix(".log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"✅ saved refiner checkpoint: {args.out}")


if __name__ == "__main__":
    main()
