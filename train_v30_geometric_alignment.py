#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train V30 hyperbolic music-motion geometric alignment."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from tools.v30_geometric_alignment import (
    GeometricAlignmentConfig,
    V30GeometricAligner,
    geometric_alignment_loss,
)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: str | Path) -> None:
        z = np.load(path, allow_pickle=True)
        self.values = {
            key: np.asarray(z[key])
            for key in (
                "music_rule", "music_clap", "clap_valid",
                "positive_raw", "positive_mmr",
                "negative_raw", "negative_mmr",
                "hierarchy_label", "target_radius", "positive_id", "sample_weight",
            )
        }
        self.group = np.asarray(z["group"], dtype=object)
        self.meta = json.loads(str(z["meta"].item()))

    def __len__(self) -> int:
        return len(self.values["music_rule"])

    def __getitem__(self, index: int):
        row = {key: value[index] for key, value in self.values.items()}
        row["index"] = index
        return row


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            if torch.is_floating_point(value):
                self.shadow[key].lerp_(value.detach(), 1.0 - self.decay)
            else:
                self.shadow[key].copy_(value)


def split_groups(
    groups: np.ndarray, ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    unique = np.asarray(sorted(set(str(x) for x in groups)), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    count = max(1, min(len(unique) - 1, int(round(len(unique) * ratio))))
    validation = set(str(x) for x in unique[:count])
    val_idx = np.asarray([i for i, g in enumerate(groups) if str(g) in validation], np.int64)
    train_idx = np.asarray([i for i, g in enumerate(groups) if str(g) not in validation], np.int64)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("Cannot create non-empty group-disjoint split")
    return train_idx, val_idx, {
        "num_groups": int(len(unique)),
        "validation_groups": sorted(validation),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
    }


def run_epoch(
    model: V30GeometricAligner,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    ema: EMA | None,
    args: argparse.Namespace,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, list[float]] = {}
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            tensors = {
                key: torch.as_tensor(value, device=device)
                for key, value in batch.items() if key != "index"
            }
            loss, metrics = geometric_alignment_loss(
                model,
                tensors["music_rule"].float(),
                tensors["music_clap"].float(),
                tensors["clap_valid"].float(),
                tensors["positive_raw"].float(),
                tensors["positive_mmr"].float(),
                tensors["negative_raw"].float(),
                tensors["negative_mmr"].float(),
                tensors["hierarchy_label"].long(),
                tensors["target_radius"].float(),
                positive_id=tensors["positive_id"].long(),
                preference_margin=args.preference_margin,
                hierarchy_weight=args.hierarchy_weight,
                radius_weight=args.radius_weight,
                preference_weight=args.preference_weight,
            )
            # Training exposure is controlled by a weighted sampler.  Do not
            # scale the entire batch loss by its mean weight, which would couple
            # unrelated explicit and weak samples.
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if ema is not None:
                    ema.update(model)
            totals.setdefault("total", []).append(float(loss.detach().cpu()))
            for key, value in metrics.items():
                totals.setdefault(key, []).append(float(value.detach().cpu()))
    return {key: float(np.mean(value)) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=192)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--preference_margin", type=float, default=0.20)
    parser.add_argument("--preference_weight", type=float, default=0.55)
    parser.add_argument("--hierarchy_weight", type=float, default=0.20)
    parser.add_argument("--radius_weight", type=float, default=0.12)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = Dataset(args.data)
    train_idx, val_idx, split = split_groups(dataset.group, args.val_ratio, args.seed)
    train_subset = torch.utils.data.Subset(dataset, train_idx.tolist())
    train_weights = torch.as_tensor(
        dataset.values["sample_weight"][train_idx], dtype=torch.double
    ).clamp_min(0.02)
    train_sampler = torch.utils.data.WeightedRandomSampler(
        train_weights, num_samples=len(train_weights), replacement=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, val_idx.tolist()),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    config = GeometricAlignmentConfig(
        rule_dim=int(dataset.values["music_rule"].shape[1]),
        clap_dim=int(dataset.values["music_clap"].shape[1]),
        motion_raw_dim=int(dataset.values["positive_raw"].shape[1]),
        motion_mmr_dim=int(dataset.values["positive_mmr"].shape[1]),
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        curvature=args.curvature,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V30GeometricAligner(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.08
    )
    ema = EMA(model, args.ema_decay)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / "best.pt"
    history = []
    best = float("inf")
    bad = 0
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, device, optimizer, ema, args)
        val = run_epoch(model, val_loader, device, None, None, args)
        scheduler.step()
        row = {"epoch": epoch, "train": train, "val": val, "val_loss": val["total"]}
        history.append(row)
        print(
            f"[V30 alignment] epoch={epoch:04d} "
            f"train={train['total']:.6f} val={val['total']:.6f} "
            f"pos={val.get('positive_distance', 0):.4f} "
            f"neg={val.get('negative_distance', 0):.4f}",
            flush=True,
        )
        if val["total"] < best:
            best = val["total"]
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema_model": ema.shadow,
                    "config": asdict(config),
                    "dataset_meta": dataset.meta,
                    "split": split,
                    "epoch": epoch,
                    "best_val_loss": best,
                    "val_metrics": val,
                },
                best_path,
            )
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[EARLY STOP] patience={args.patience}")
                break

    (out / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "BEST_V30_ALIGNMENT_CKPT.txt").write_text(
        str(best_path), encoding="utf-8"
    )
    print(f"[SAVED] {best_path}")


if __name__ == "__main__":
    main()
