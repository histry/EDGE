#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a lightweight hyperbolic hierarchy encoder for V27-HG.

This directly addresses the "math washing" criticism: hierarchy embeddings are
not only hand-scaled into a Poincare ball.  A small encoder learns the tangent
direction with self-supervised hierarchy contrast, while an explicit radius
regularizer preserves the coarse-to-fine interpretation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tools.schedule_v21_multi_music import load_shared_index
from tools.v26_hierarchical_graph_scheduler import build_hierarchy_features


class HyperbolicEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int | None = None) -> None:
        super().__init__()
        out_dim = int(out_dim or in_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def info_nce_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    z = F.normalize(z, dim=-1)
    logits = z @ z.t() / max(float(temperature), 1e-6)
    eye = torch.eye(len(z), device=z.device, dtype=torch.bool)
    labels = labels.reshape(-1, 1)
    positives = (labels == labels.t()) & (~eye)
    logits = logits.masked_fill(eye, -1e4)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = positives.sum(dim=1).clamp_min(1)
    loss = -(log_prob * positives.float()).sum(dim=1) / pos_count
    return loss.mean()


def make_batches(n: int, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    rng.shuffle(order)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.12)
    parser.add_argument("--radius_weight", type=float, default=0.20)
    parser.add_argument("--center_weight", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, arrays, items = load_shared_index(Path(args.index_json), Path(args.duration_index_npz))
    hierarchy = build_hierarchy_features(arrays, items, hyperbolic_ckpt=None)
    raw = torch.from_numpy(np.asarray(hierarchy["hierarchy_raw"], dtype=np.float32))
    body = torch.from_numpy(np.asarray(hierarchy["body_code"], dtype=np.int64))
    center = torch.from_numpy(np.asarray(hierarchy["center_code"], dtype=np.int64))
    gesture = torch.from_numpy(np.asarray(hierarchy["gesture_code"], dtype=np.int64))
    radius = torch.from_numpy(np.asarray(hierarchy["hierarchy_radius"], dtype=np.float32))
    specificity = torch.from_numpy(np.asarray(hierarchy["specificity"], dtype=np.float32))

    labels = body * 100 + center * 10 + gesture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = raw.to(device)
    body = body.to(device)
    labels = labels.to(device)
    radius = radius.to(device)
    specificity = specificity.to(device)

    model = HyperbolicEncoder(raw.shape[1], hidden_dim=args.hidden_dim, out_dim=raw.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_loss = float("inf")
    best_path = out_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in make_batches(len(raw), args.batch_size, args.seed + epoch):
            idx = torch.from_numpy(batch).to(device)
            x = raw[idx]
            z = model(x)
            # Coarse positives: same body/center/gesture hierarchy.
            loss_h = info_nce_loss(z, labels[idx], args.temperature)
            # Body-level auxiliary contrast prevents tiny classes from becoming
            # isolated without a parent relation.
            loss_body = info_nce_loss(z, body[idx], args.temperature * 1.35)
            z_norm = torch.linalg.norm(z, dim=-1)
            target_tangent_norm = torch.atanh(radius[idx].clamp(0.05, 0.95))
            loss_radius = F.mse_loss(z_norm, target_tangent_norm)
            # Coarser motions should stay closer to the origin than highly
            # specific turn/gesture events.
            coarse_target = specificity[idx]
            loss_center = F.mse_loss(torch.sigmoid(z_norm - z_norm.mean()), coarse_target)
            loss = loss_h + 0.35 * loss_body + args.radius_weight * loss_radius + args.center_weight * loss_center
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else 0.0
        record: Dict[str, float | int] = {"epoch": epoch, "loss": epoch_loss}
        history.append(record)
        print(f"[v27 hyperbolic] epoch={epoch:04d} loss={epoch_loss:.6f}", flush=True)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "in_dim": int(raw.shape[1]),
                        "hidden_dim": int(args.hidden_dim),
                        "out_dim": int(raw.shape[1]),
                        "temperature": float(args.temperature),
                        "radius_weight": float(args.radius_weight),
                        "center_weight": float(args.center_weight),
                    },
                    "best_loss": float(best_loss),
                    "epoch": int(epoch),
                },
                best_path,
            )

    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "BEST_V27_HYPERBOLIC_CKPT.txt").write_text(str(best_path), encoding="utf-8")
    print(f"[SAVED] {best_path}")


if __name__ == "__main__":
    main()
