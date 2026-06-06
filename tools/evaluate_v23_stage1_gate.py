#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a Stage-1 duration checkpoint with continuous-duration metrics.

The auxiliary ordinal argmax is reported, but scientific gating follows the
final calibrated continuous duration because that is what Stage 2 and runtime
actually consume.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model.v23_monotonic_duration import load_v23_checkpoint
from train_v23_monotonic_duration import (
    dataset_arrays,
    duration_group_metrics,
    duration_stratified_group_split,
    load_duration_edges,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=-1.0)
    parser.add_argument("--split_seed", type=int, default=-1)
    parser.add_argument("--split_trials", type=int, default=-1)
    args = parser.parse_args()

    arrays = dataset_arrays(args.data)
    duration_edges = load_duration_edges(args.data, arrays)
    bundle = load_v23_checkpoint(args.checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    model = bundle["model"]
    config = bundle["config"]
    device = next(model.parameters()).device

    val_ratio = float(config.get("val_ratio", 0.15)) if args.val_ratio < 0 else args.val_ratio
    split_seed = int(config.get("split_seed", 20260620)) if args.split_seed < 0 else args.split_seed
    split_trials = int(config.get("split_trials", 4096)) if args.split_trials < 0 else args.split_trials
    _, val_indices = duration_stratified_group_split(
        arrays["source_id"],
        arrays["duration_bin"],
        arrays["is_identity"],
        arrays["target_duration_frames"],
        val_ratio,
        split_seed,
        split_trials,
    )

    stores: dict[str, list[np.ndarray]] = {
        "predicted": [], "target": [], "target_bin": [],
        "continuous_bin": [], "ordinal_bin": [], "edit_probability": [],
        "edit_target": [], "event_uid": [], "bin_probability": [],
    }
    model.eval()
    for start in range(0, len(val_indices), args.batch_size):
        idx = val_indices[start:start + args.batch_size]
        motion = torch.from_numpy(arrays["corrupted"][idx].astype(np.float32)).to(device)
        mask = torch.from_numpy(arrays["edit_mask"][idx].astype(np.float32)).to(device)
        condition = torch.from_numpy(arrays["condition"][idx].astype(np.float32)).to(device)
        with torch.no_grad():
            result = model.predict_duration(motion, mask, condition, use_hard_duration=False)
        stores["predicted"].append(result["duration_soft_frames"].cpu().numpy())
        stores["target"].append(arrays["target_duration_frames"][idx].astype(np.float32))
        stores["target_bin"].append(arrays["duration_bin"][idx].astype(np.int64))
        stores["continuous_bin"].append(result["duration_continuous_bin_index"].cpu().numpy())
        stores["ordinal_bin"].append(result["duration_ordinal_bin_index"].cpu().numpy())
        stores["edit_probability"].append(result["edit_probability"].cpu().numpy())
        stores["edit_target"].append((1.0 - arrays["is_identity"][idx]).astype(np.float32))
        stores["event_uid"].append(arrays["event_uid"][idx].astype(np.int64))
        stores["bin_probability"].append(result["duration_bin_probabilities"].cpu().numpy())

    merged = {key: np.concatenate(value) for key, value in stores.items()}
    metrics = duration_group_metrics(
        merged["predicted"],
        merged["target"],
        merged["target_bin"],
        merged["continuous_bin"],
        merged["edit_probability"],
        merged["edit_target"],
        len(duration_edges) - 1,
        event_uid=merged["event_uid"],
        bin_probability=merged["bin_probability"],
        duration_edges=duration_edges,
        ordinal_predicted_bin=merged["ordinal_bin"],
    )
    report = {
        "version": "V23-v2.5-continuous-gate",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(bundle.get("epoch", -1)),
        "data": str(Path(args.data).resolve()),
        "validation_samples": int(len(val_indices)),
        "validation_sources": int(len(np.unique(arrays["source_id"][val_indices]))),
        "validation_events": int(len(np.unique(arrays["event_uid"][val_indices]))),
        "split_seed": split_seed,
        "split_trials": split_trials,
        "duration_edges": duration_edges.tolist(),
        "metrics": metrics,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 88)
    print("V23-v2.5 STAGE-1 CONTINUOUS-DURATION EVALUATION")
    print("=" * 88)
    keys = [
        "event_duration_mae_frames", "event_duration_long_mae",
        "event_duration_medium_mae", "event_duration_correlation",
        "event_duration_continuous_bin_accuracy",
        "event_duration_continuous_within_one_bin_accuracy",
        "event_duration_ordinal_bin_accuracy",
        "event_duration_ordinal_within_one_bin_accuracy",
        "event_duration_p90_error", "event_duration_quantile_calibration_mae",
        "edit_accuracy", "edit_balanced_accuracy", "edit_auroc",
        "edit_optimal_threshold", "edit_optimal_balanced_accuracy",
    ]
    for key in keys:
        print(f"{key:52s} = {metrics.get(key, float('nan')):.8f}")
    print("saved:", out)


if __name__ == "__main__":
    main()
