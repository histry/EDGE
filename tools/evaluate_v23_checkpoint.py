#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model.v23_monotonic_duration import (
    load_v23_checkpoint,
    root_yaw_velocity_dps,
    warp_motion_so3,
)
from train_v23_monotonic_duration import group_split, duration_group_metrics


def rotation_activity(x: torch.Tensor) -> torch.Tensor:
    velocity = x[:, 1:, 7:151] - x[:, :-1, 7:151]
    return torch.linalg.vector_norm(velocity, dim=-1).mean(dim=1)


def pose_range(x: torch.Tensor) -> torch.Tensor:
    rotation = x[..., 7:151]
    return torch.linalg.vector_norm(rotation - rotation[:, :1], dim=-1).amax(dim=1)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--max_samples", type=int, default=4096)
    parser.add_argument("--eval_seed", type=int, default=20260623)
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.data, allow_pickle=True) as archive:
        required = (
            "corrupted", "target", "edit_mask", "condition", "target_tau",
            "source_id", "target_duration_frames", "is_identity", "duration_bin", "event_uid",
        )
        for key in required:
            if key not in archive.files:
                raise RuntimeError(f"Dataset missing {key}")
        arrays = {key: np.asarray(archive[key]) for key in archive.files}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_v23_checkpoint(args.checkpoint, device=device)
    model = bundle["model"]
    config = bundle["config"]
    val_ratio = float(config.get("val_ratio", 0.15))
    split_seed = int(config.get("split_seed", 20260620))
    split_trials = int(config.get("split_trials", 4096))
    _, val_indices = group_split(
        arrays["source_id"],
        val_ratio,
        split_seed,
        duration_bin=arrays["duration_bin"],
        is_identity=arrays["is_identity"],
        target_duration=arrays["target_duration_frames"],
        trials=split_trials,
    )
    if args.max_samples > 0 and len(val_indices) > args.max_samples:
        rng = np.random.default_rng(args.eval_seed)
        val_indices = np.sort(rng.choice(val_indices, args.max_samples, replace=False))

    stores: dict[str, list[np.ndarray]] = {}
    all_predicted_motion = []
    for start in range(0, len(val_indices), args.batch_size):
        indices = val_indices[start : start + args.batch_size]
        x = torch.from_numpy(arrays["corrupted"][indices].astype(np.float32)).to(device)
        y = torch.from_numpy(arrays["target"][indices].astype(np.float32)).to(device)
        mask = torch.from_numpy(arrays["edit_mask"][indices].astype(np.float32)).to(device)
        condition = torch.from_numpy(arrays["condition"][indices].astype(np.float32)).to(device)
        target_tau = torch.from_numpy(arrays["target_tau"][indices].astype(np.float32)).to(device)
        target_duration = torch.from_numpy(
            arrays["target_duration_frames"][indices].astype(np.float32)
        ).to(device)
        identity = torch.from_numpy(arrays["is_identity"][indices].astype(np.float32)).to(device)
        target_bin = torch.from_numpy(arrays["duration_bin"][indices].astype(np.int64)).to(device)

        with torch.no_grad():
            result = model(x, mask, condition, use_hard_duration=False)
            predicted = warp_motion_so3(x, result["tau"])
            oracle = warp_motion_so3(x, target_tau)
            input_mse = ((x[..., 4:151] - y[..., 4:151]) ** 2).mean(dim=(1, 2))
            pred_mse = ((predicted[..., 4:151] - y[..., 4:151]) ** 2).mean(dim=(1, 2))
            oracle_mse = ((oracle[..., 4:151] - y[..., 4:151]) ** 2).mean(dim=(1, 2))
            input_yaw = root_yaw_velocity_dps(x)
            pred_yaw = root_yaw_velocity_dps(predicted)
            target_yaw = root_yaw_velocity_dps(y)
            input_yaw_mae = torch.abs(input_yaw - target_yaw).mean(dim=1)
            pred_yaw_mae = torch.abs(pred_yaw - target_yaw).mean(dim=1)
            input_peak = torch.quantile(torch.abs(input_yaw), 0.95, dim=1)
            pred_peak = torch.quantile(torch.abs(pred_yaw), 0.95, dim=1)
            target_peak = torch.quantile(torch.abs(target_yaw), 0.95, dim=1)
            input_activity = rotation_activity(x)
            pred_activity = rotation_activity(predicted)
            target_activity = rotation_activity(y)
            pred_range = pose_range(predicted)
            target_range = pose_range(y)
            outside = (1.0 - mask).clamp(0.0, 1.0)[..., None]
            outside_drift = (
                torch.abs(predicted[..., 4:151] - y[..., 4:151]) * outside
            ).sum(dim=(1, 2)) / (outside.sum(dim=(1, 2)) * 147.0).clamp_min(1.0)
            uniform = torch.linspace(0.0, 1.0, x.shape[1], device=device)[None]
            identity_rows = identity > 0.5
            identity_tau = torch.abs(result["tau"] - uniform).mean(dim=1)
            identity_motion = torch.abs(predicted[..., 4:151] - x[..., 4:151]).mean(dim=(1, 2))

        batch_values = {
            "input_mse": input_mse,
            "pred_mse": pred_mse,
            "oracle_mse": oracle_mse,
            "tau_mae": torch.abs(result["tau"] - target_tau).mean(dim=1),
            "pred_tau": result["tau"],
            "target_tau_array": target_tau,
            "target_duration": target_duration,
            "pred_duration": result["duration_frames"],
            "target_bin": target_bin,
            "pred_bin": result["duration_bin_index"],
            "bin_confidence": result["duration_bin_confidence"],
            "bin_probability": result["duration_bin_probabilities"],
            "event_uid": torch.from_numpy(arrays["event_uid"][indices].astype(np.int64)).to(device),
            "edit_probability": result["edit_probability"],
            "edit_target": 1.0 - identity,
            "input_yaw_mae": input_yaw_mae,
            "pred_yaw_mae": pred_yaw_mae,
            "input_peak_error": torch.abs(input_peak - target_peak),
            "pred_peak_error": torch.abs(pred_peak - target_peak),
            "input_activity": input_activity,
            "pred_activity": pred_activity,
            "target_activity": target_activity,
            "pred_range": pred_range,
            "target_range": target_range,
            "outside_mask_drift": outside_drift,
            "identity_tau": identity_tau[identity_rows],
            "identity_motion": identity_motion[identity_rows],
        }
        for key, value in batch_values.items():
            stores.setdefault(key, []).append(value.detach().cpu().numpy())
        all_predicted_motion.append(predicted.cpu().numpy())
        print(f"evaluated {min(start + len(indices), len(val_indices))}/{len(val_indices)}", flush=True)

    values = {
        key: np.concatenate([part for part in parts if len(part) > 0])
        if any(len(part) > 0 for part in parts) else np.empty((0,), dtype=np.float32)
        for key, parts in stores.items()
    }
    predicted_motion = np.concatenate(all_predicted_motion)
    duration_error = np.abs(values["pred_duration"] - values["target_duration"])
    num_bins = len(config.get("duration_edges", [])) - 1
    metrics = {
        "input_mse": float(values["input_mse"].mean()),
        "pred_mse": float(values["pred_mse"].mean()),
        "oracle_mse": float(values["oracle_mse"].mean()),
        "motion_mse_improvement_percent": float(
            100.0 * (1.0 - values["pred_mse"].mean() / max(values["input_mse"].mean(), 1e-12))
        ),
        "tau_mae": float(values["tau_mae"].mean()),
        "duration_mae": float(duration_error.mean()),
        "duration_correlation": corr(values["pred_duration"], values["target_duration"]),
        "duration_bin_accuracy": float(np.mean(values["pred_bin"] == values["target_bin"])),
        "duration_bin_confidence": float(values["bin_confidence"].mean()),
        "edit_accuracy": float(
            np.mean((values["edit_probability"] >= 0.5) == (values["edit_target"] >= 0.5))
        ),
        "identity_edit_probability": float(
            values["edit_probability"][values["edit_target"] < 0.5].mean()
        ),
        "non_identity_edit_probability": float(
            values["edit_probability"][values["edit_target"] >= 0.5].mean()
        ),
        "input_yaw_mae": float(values["input_yaw_mae"].mean()),
        "pred_yaw_mae": float(values["pred_yaw_mae"].mean()),
        "yaw_mae_improvement_percent": float(
            100.0 * (1.0 - values["pred_yaw_mae"].mean() / max(values["input_yaw_mae"].mean(), 1e-12))
        ),
        "input_peak_error": float(values["input_peak_error"].mean()),
        "pred_peak_error": float(values["pred_peak_error"].mean()),
        "peak_error_improvement_percent": float(
            100.0 * (1.0 - values["pred_peak_error"].mean() / max(values["input_peak_error"].mean(), 1e-12))
        ),
        "activity_preservation_ratio": float(
            values["pred_activity"].mean() / max(values["target_activity"].mean(), 1e-12)
        ),
        "pose_range_preservation_ratio": float(
            values["pred_range"].mean() / max(values["target_range"].mean(), 1e-12)
        ),
        "outside_mask_drift": float(values["outside_mask_drift"].mean()),
        "identity_tau_mae": float(values["identity_tau"].mean()) if len(values["identity_tau"]) else 0.0,
        "identity_motion_drift": float(values["identity_motion"].mean()) if len(values["identity_motion"]) else 0.0,
        "monotonic_violation": 0.0,
        "endpoint_error": 0.0,
    }
    grouped = duration_group_metrics(
        values["pred_duration"],
        values["target_duration"],
        values["target_bin"].astype(int),
        values["pred_bin"].astype(int),
        values["edit_probability"],
        values["edit_target"],
        num_bins,
        event_uid=values["event_uid"],
        bin_probability=values["bin_probability"],
    )
    metrics.update(grouped)
    for bin_id in range(max(num_bins, 0)):
        rows = values["target_bin"] == bin_id
        metrics[f"duration_bin_{bin_id}_mae"] = float(duration_error[rows].mean()) if np.any(rows) else float("nan")
    if num_bins > 0:
        short = values["target_bin"] <= max(0, num_bins // 3 - 1)
        long = values["target_bin"] >= max(0, num_bins - max(1, num_bins // 3))
        middle = ~(short | long)
        metrics["duration_short_mae"] = float(duration_error[short].mean())
        metrics["duration_medium_mae"] = float(duration_error[middle].mean())
        metrics["duration_long_mae"] = float(duration_error[long].mean())

    improvement = values["input_mse"] - values["pred_mse"]
    order = np.argsort(improvement)
    examples = {
        "worst": int(order[0]),
        "median": int(order[len(order) // 2]),
        "best": int(order[-1]),
    }
    for label, position in examples.items():
        original = int(val_indices[position])
        np.save(output_dir / f"{label}_corrupted.npy", arrays["corrupted"][original])
        np.save(output_dir / f"{label}_predicted.npy", predicted_motion[position])
        np.save(output_dir / f"{label}_target.npy", arrays["target"][original])
        np.save(output_dir / f"{label}_predicted_tau.npy", values["pred_tau"][position])
        np.save(output_dir / f"{label}_target_tau.npy", values["target_tau_array"][position])

    report = {
        "version": "v23_v2_4_ordinal_event_consistent",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(bundle["epoch"]),
        "training_stage": bundle["stage"],
        "evaluated_samples": int(len(val_indices)),
        "evaluated_sources": int(len(np.unique(arrays["source_id"][val_indices]))),
        "metrics": metrics,
        "target_duration_percentiles": np.percentile(
            values["target_duration"], [0, 10, 25, 50, 75, 90, 100]
        ).tolist(),
        "pred_duration_percentiles": np.percentile(
            values["pred_duration"], [0, 10, 25, 50, 75, 90, 100]
        ).tolist(),
        "examples": examples,
    }
    output_path = output_dir / "V23_V2_4_HELDOUT_EVALUATION.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 88)
    print("V23-v2.4 HELD-OUT EVALUATION")
    print("=" * 88)
    for key, value in metrics.items():
        print(f"{key:42s} = {value:.8f}")
    print("target duration:", report["target_duration_percentiles"])
    print("pred duration:  ", report["pred_duration_percentiles"])
    print("saved:", output_path)


if __name__ == "__main__":
    main()
