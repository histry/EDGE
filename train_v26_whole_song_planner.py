#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the optional V26 whole-song phrase-sequence planner."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model.v26_whole_song_planner import V26WholeSongPlanner
from tools.v21_common import EVENT_TYPES


class PlannerDataset(Dataset):
    def __init__(self, path: str | Path, indices: np.ndarray) -> None:
        with np.load(path, allow_pickle=True) as z:
            self.features = list(z["features"])
            self.events = list(z["event_labels"])
            self.durations = list(z["duration_targets"])
            self.transitions = list(z["transition_labels"])
            self.activities = list(z["activity_targets"])
            self.keys = list(z["song_keys"])
            self.transition_lengths = tuple(int(x) for x in z["transition_lengths"])
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        return {
            "features": np.asarray(self.features[index], dtype=np.float32),
            "events": np.asarray(self.events[index], dtype=np.int64),
            "durations": np.asarray(self.durations[index], dtype=np.float32),
            "transitions": np.asarray(self.transitions[index], dtype=np.int64),
            "activities": np.asarray(self.activities[index], dtype=np.float32),
            "key": str(self.keys[index]),
        }


def collate(batch):
    max_len = max(len(row["features"]) for row in batch)
    batch_size = len(batch)
    feature_dim = batch[0]["features"].shape[1]
    features = torch.zeros((batch_size, max_len, feature_dim), dtype=torch.float32)
    events = torch.full((batch_size, max_len), -100, dtype=torch.long)
    durations = torch.zeros((batch_size, max_len), dtype=torch.float32)
    transitions = torch.full((batch_size, max_len), -100, dtype=torch.long)
    activities = torch.zeros((batch_size, max_len), dtype=torch.float32)
    padding = torch.ones((batch_size, max_len), dtype=torch.bool)
    valid = torch.zeros((batch_size, max_len), dtype=torch.bool)
    keys = []
    for i, row in enumerate(batch):
        length = len(row["features"])
        features[i, :length] = torch.from_numpy(row["features"])
        events[i, :length] = torch.from_numpy(row["events"])
        durations[i, :length] = torch.from_numpy(row["durations"])
        transitions[i, :length] = torch.from_numpy(row["transitions"])
        activities[i, :length] = torch.from_numpy(row["activities"])
        padding[i, :length] = False
        valid[i, :length] = True
        keys.append(row["key"])
    return {
        "features": features,
        "events": events,
        "durations": durations,
        "transitions": transitions,
        "activities": activities,
        "padding": padding,
        "valid": valid,
        "keys": keys,
    }


def run_epoch(model, loader, optimizer, device, train: bool) -> Dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "event": 0.0, "duration": 0.0, "transition": 0.0, "activity": 0.0, "count": 0.0}
    correct_event = 0
    correct_transition = 0
    duration_abs = 0.0
    valid_count = 0
    for batch in loader:
        features = batch["features"].to(device)
        events = batch["events"].to(device)
        durations = batch["durations"].to(device)
        transitions = batch["transitions"].to(device)
        activities = batch["activities"].to(device)
        padding = batch["padding"].to(device)
        valid = batch["valid"].to(device)

        with torch.set_grad_enabled(train):
            output = model(features, padding_mask=padding)
            event_loss = F.cross_entropy(
                output["event_logits"].reshape(-1, output["event_logits"].shape[-1]),
                events.reshape(-1),
                ignore_index=-100,
            )
            log_target = torch.log(durations.clamp_min(1.0))
            duration_loss = F.smooth_l1_loss(output["log_duration"][valid], log_target[valid])
            transition_loss = F.cross_entropy(
                output["transition_logits"].reshape(-1, output["transition_logits"].shape[-1]),
                transitions.reshape(-1),
                ignore_index=-100,
            )
            activity_loss = F.smooth_l1_loss(output["activity"][valid], activities[valid])
            # Slow sequence regularity discourages frame-to-frame phrase-duration oscillation.
            if output["log_duration"].shape[1] > 1:
                pair_valid = valid[:, 1:] & valid[:, :-1]
                smooth = torch.abs(
                    output["log_duration"][:, 1:] - output["log_duration"][:, :-1]
                )[pair_valid].mean() if pair_valid.any() else torch.zeros((), device=device)
            else:
                smooth = torch.zeros((), device=device)
            loss = event_loss + 1.15 * duration_loss + 0.35 * transition_loss + 0.25 * activity_loss + 0.03 * smooth
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        count = int(valid.sum().item())
        totals["loss"] += float(loss.item()) * count
        totals["event"] += float(event_loss.item()) * count
        totals["duration"] += float(duration_loss.item()) * count
        totals["transition"] += float(transition_loss.item()) * count
        totals["activity"] += float(activity_loss.item()) * count
        totals["count"] += count
        correct_event += int((output["event_logits"].argmax(-1)[valid] == events[valid]).sum().item())
        correct_transition += int((output["transition_logits"].argmax(-1)[valid] == transitions[valid]).sum().item())
        duration_abs += float(torch.abs(output["duration_frames"][valid] - durations[valid]).sum().item())
        valid_count += count

    denom = max(totals["count"], 1.0)
    return {
        "loss": totals["loss"] / denom,
        "event_loss": totals["event"] / denom,
        "duration_loss": totals["duration"] / denom,
        "transition_loss": totals["transition"] / denom,
        "activity_loss": totals["activity"] / denom,
        "event_accuracy": correct_event / max(valid_count, 1),
        "transition_accuracy": correct_transition / max(valid_count, 1),
        "duration_mae_frames": duration_abs / max(valid_count, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.data, allow_pickle=True) as z:
        n = len(z["song_keys"])
        transition_lengths = tuple(int(x) for x in z["transition_lengths"])
        feature_dim = int(np.asarray(z["features"][0]).shape[1])
    order = np.random.default_rng(args.seed).permutation(n)
    val_count = max(1, int(round(n * args.val_ratio)))
    val_indices = order[:val_count]
    train_indices = order[val_count:]

    train_set = PlannerDataset(args.data, train_indices)
    val_set = PlannerDataset(args.data, val_indices)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "feature_dim": feature_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "num_event_types": len(EVENT_TYPES),
        "transition_lengths": transition_lengths,
        "seed": args.seed,
    }
    model = V26WholeSongPlanner(**{k: config[k] for k in ("feature_dim", "hidden_dim", "num_layers", "num_heads", "dropout", "num_event_types", "transition_lengths")}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.03)

    history = []
    best_score = float("inf")
    best_epoch = -1
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, None, device, train=False)
        scheduler.step()
        score = (
            val_metrics["duration_mae_frames"] / 20.0
            + (1.0 - val_metrics["event_accuracy"])
            + 0.35 * (1.0 - val_metrics["transition_accuracy"])
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "score": score}
        history.append(row)
        print(
            f"epoch={epoch:03d} val_loss={val_metrics['loss']:.5f} "
            f"event={val_metrics['event_accuracy']:.3f} "
            f"duration_mae={val_metrics['duration_mae_frames']:.2f} "
            f"transition={val_metrics['transition_accuracy']:.3f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "selection_score": score,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            stale += 1
        if stale >= args.patience:
            break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "BEST_V26_PLANNER_CKPT.txt").write_text(str(checkpoint_dir / "best.pt") + "\n", encoding="utf-8")
    (out_dir / "split.json").write_text(
        json.dumps({"train_indices": train_indices.tolist(), "val_indices": val_indices.tolist(), "seed": args.seed}, indent=2),
        encoding="utf-8",
    )
    print(f"best_epoch={best_epoch} best_score={best_score:.6f}")


if __name__ == "__main__":
    main()
