#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from model.v23_monotonic_duration import load_v23_checkpoint, root_yaw_velocity_dps, warp_motion_so3
from train_v23_monotonic_duration import group_split


def motion_activity(x: torch.Tensor) -> torch.Tensor:
    velocity = x[:, 1:, 7:151] - x[:, :-1, 7:151]
    return torch.linalg.vector_norm(velocity, dim=-1).mean(dim=1)


def motion_range(x: torch.Tensor) -> torch.Tensor:
    centered = x[..., 7:151] - x[..., 7:151].mean(dim=1, keepdim=True)
    return torch.linalg.vector_norm(centered, dim=-1).mean(dim=1)


def per_sample_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a[..., 4:151] - b[..., 4:151]) ** 2).mean(dim=(1, 2))


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=4096)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--eval_seed", type=int, default=20260610)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    match = re.search(r"seed_(\d+)", str(checkpoint_path))
    if not match:
        raise RuntimeError("Checkpoint path must contain seed_<integer>")
    training_seed = int(match.group(1))

    with np.load(args.data, allow_pickle=True) as archive:
        required = (
            "corrupted", "target", "edit_mask", "condition", "target_tau", "source_id",
            "target_duration_frames", "speed_factor", "is_identity", "duration_bin",
        )
        for key in required:
            if key not in archive.files:
                raise RuntimeError(f"Dataset missing {key}")
        arrays = {key: np.asarray(archive[key]) for key in archive.files}

    _, val_indices = group_split(arrays["source_id"], args.val_ratio, training_seed)
    if args.max_samples > 0 and len(val_indices) > args.max_samples:
        rng = np.random.default_rng(args.eval_seed)
        # Preserve identity and duration-bin coverage in the evaluation subset.
        selected = []
        strata = arrays["is_identity"].astype(np.int64) * (int(arrays["duration_bin"].max()) + 1) + arrays["duration_bin"].astype(np.int64)
        per_stratum = max(1, args.max_samples // max(1, len(np.unique(strata[val_indices]))))
        for stratum in np.unique(strata[val_indices]):
            candidates = val_indices[strata[val_indices] == stratum]
            take = min(len(candidates), per_stratum)
            selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
        if len(selected) < args.max_samples:
            remaining = np.setdiff1d(val_indices, np.asarray(selected, dtype=np.int64), assume_unique=False)
            take = min(len(remaining), args.max_samples - len(selected))
            if take > 0:
                selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
        val_indices = np.asarray(sorted(selected[: args.max_samples]), dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_v23_checkpoint(checkpoint_path, device=device)
    model = bundle["model"]

    collected = {
        "input_mse": [], "pred_mse": [], "oracle_mse": [], "tau_mae": [],
        "duration_mae": [], "input_yaw_mae": [], "pred_yaw_mae": [],
        "input_peak_error": [], "pred_peak_error": [], "target_activity": [],
        "input_activity": [], "pred_activity": [], "target_range": [], "pred_range": [],
        "outside_mask_drift": [], "identity_tau_mae": [], "identity_motion_drift": [],
        "edit_probability": [], "edit_accuracy": [], "monotonic_violation": [], "endpoint_error": [],
        "predicted_duration": [], "target_duration": [], "duration_bin": [], "is_identity": [],
    }
    all_predictions = []
    all_tau = []
    uniform_tau = torch.linspace(0.0, 1.0, arrays["corrupted"].shape[1], device=device)[None]

    for start in range(0, len(val_indices), args.batch_size):
        indices = val_indices[start : start + args.batch_size]
        corrupted = torch.from_numpy(arrays["corrupted"][indices].astype(np.float32)).to(device)
        target = torch.from_numpy(arrays["target"][indices].astype(np.float32)).to(device)
        mask = torch.from_numpy(arrays["edit_mask"][indices].astype(np.float32)).to(device)
        condition = torch.from_numpy(arrays["condition"][indices].astype(np.float32)).to(device)
        target_tau = torch.from_numpy(arrays["target_tau"][indices].astype(np.float32)).to(device)
        target_duration = torch.from_numpy(arrays["target_duration_frames"][indices].astype(np.float32)).to(device)
        identity = torch.from_numpy(arrays["is_identity"][indices].astype(np.float32)).to(device)

        with torch.no_grad():
            result = model(corrupted, mask, condition)
            predicted_tau = result["tau"]
            predicted = warp_motion_so3(corrupted, predicted_tau)
            oracle = warp_motion_so3(corrupted, target_tau)

            input_mse = per_sample_mse(corrupted, target)
            pred_mse = per_sample_mse(predicted, target)
            oracle_mse = per_sample_mse(oracle, target)
            tau_mae = torch.abs(predicted_tau - target_tau).mean(dim=1)
            duration_mae = torch.abs(result["duration_frames"] - target_duration)

            input_yaw = root_yaw_velocity_dps(corrupted)
            pred_yaw = root_yaw_velocity_dps(predicted)
            target_yaw = root_yaw_velocity_dps(target)
            input_yaw_mae = torch.abs(input_yaw - target_yaw).mean(dim=1)
            pred_yaw_mae = torch.abs(pred_yaw - target_yaw).mean(dim=1)
            input_peak_error = torch.abs(torch.quantile(torch.abs(input_yaw), 0.95, dim=1) - torch.quantile(torch.abs(target_yaw), 0.95, dim=1))
            pred_peak_error = torch.abs(torch.quantile(torch.abs(pred_yaw), 0.95, dim=1) - torch.quantile(torch.abs(target_yaw), 0.95, dim=1))

            target_activity = motion_activity(target)
            input_activity = motion_activity(corrupted)
            pred_activity = motion_activity(predicted)
            target_range = motion_range(target)
            pred_range = motion_range(predicted)

            outside = (1.0 - mask).clamp(0.0, 1.0)[..., None]
            outside_drift = (torch.abs(predicted[..., 4:151] - target[..., 4:151]) * outside).sum(dim=(1, 2))
            outside_drift = outside_drift / (outside.sum(dim=(1, 2)) * 147.0).clamp_min(1.0)
            identity_tau = torch.abs(predicted_tau - uniform_tau).mean(dim=1)
            identity_motion = torch.abs(predicted[..., 4:151] - corrupted[..., 4:151]).mean(dim=(1, 2))
            edit_target = 1.0 - identity
            edit_accuracy = ((result["edit_probability"] >= 0.5) == (edit_target >= 0.5)).float()
            monotonic = (predicted_tau[:, 1:] < predicted_tau[:, :-1] - 1e-7).any(dim=1).float()
            endpoint = torch.abs(predicted_tau[:, 0]) + torch.abs(predicted_tau[:, -1] - 1.0)

        values = {
            "input_mse": input_mse, "pred_mse": pred_mse, "oracle_mse": oracle_mse,
            "tau_mae": tau_mae, "duration_mae": duration_mae,
            "input_yaw_mae": input_yaw_mae, "pred_yaw_mae": pred_yaw_mae,
            "input_peak_error": input_peak_error, "pred_peak_error": pred_peak_error,
            "target_activity": target_activity, "input_activity": input_activity,
            "pred_activity": pred_activity, "target_range": target_range, "pred_range": pred_range,
            "outside_mask_drift": outside_drift, "identity_tau_mae": identity_tau,
            "identity_motion_drift": identity_motion, "edit_probability": result["edit_probability"],
            "edit_accuracy": edit_accuracy, "monotonic_violation": monotonic, "endpoint_error": endpoint,
            "predicted_duration": result["duration_frames"], "target_duration": target_duration,
        }
        for key, value in values.items():
            collected[key].append(value.detach().cpu().numpy())
        collected["duration_bin"].append(arrays["duration_bin"][indices].astype(np.float32))
        collected["is_identity"].append(arrays["is_identity"][indices].astype(np.float32))
        all_predictions.append(predicted.cpu().numpy())
        all_tau.append(predicted_tau.cpu().numpy())
        print(f"evaluated {min(start + len(indices), len(val_indices))}/{len(val_indices)}", flush=True)

    flat = {key: np.concatenate(parts) for key, parts in collected.items()}
    prediction = np.concatenate(all_predictions)
    predicted_tau = np.concatenate(all_tau)
    identity_rows = flat["is_identity"] > 0.5
    non_identity_rows = ~identity_rows
    rare_rows = flat["target_duration"] < float(bundle["config"].get("duration_max_frames", 56.0)) - 2.0

    metrics = {
        "input_mse": float(flat["input_mse"].mean()),
        "pred_mse": float(flat["pred_mse"].mean()),
        "oracle_mse": float(flat["oracle_mse"].mean()),
        "motion_mse_improvement_percent": float(100.0 * (1.0 - flat["pred_mse"].mean() / max(flat["input_mse"].mean(), 1e-12))),
        "tau_mae": float(flat["tau_mae"].mean()),
        "duration_mae": float(flat["duration_mae"].mean()),
        "rare_duration_mae": float(flat["duration_mae"][rare_rows].mean()) if rare_rows.any() else float("nan"),
        "duration_correlation": safe_corrcoef(flat["target_duration"], flat["predicted_duration"]),
        "input_yaw_mae": float(flat["input_yaw_mae"].mean()),
        "pred_yaw_mae": float(flat["pred_yaw_mae"].mean()),
        "yaw_mae_improvement_percent": float(100.0 * (1.0 - flat["pred_yaw_mae"].mean() / max(flat["input_yaw_mae"].mean(), 1e-12))),
        "input_peak_error": float(flat["input_peak_error"].mean()),
        "pred_peak_error": float(flat["pred_peak_error"].mean()),
        "peak_error_improvement_percent": float(100.0 * (1.0 - flat["pred_peak_error"].mean() / max(flat["input_peak_error"].mean(), 1e-12))),
        "activity_preservation_ratio": float(flat["pred_activity"].mean() / max(flat["target_activity"].mean(), 1e-12)),
        "pose_range_preservation_ratio": float(flat["pred_range"].mean() / max(flat["target_range"].mean(), 1e-12)),
        "outside_mask_drift": float(flat["outside_mask_drift"].mean()),
        "identity_tau_mae": float(flat["identity_tau_mae"][identity_rows].mean()) if identity_rows.any() else float("nan"),
        "identity_motion_drift": float(flat["identity_motion_drift"][identity_rows].mean()) if identity_rows.any() else float("nan"),
        "identity_edit_probability": float(flat["edit_probability"][identity_rows].mean()) if identity_rows.any() else float("nan"),
        "non_identity_edit_probability": float(flat["edit_probability"][non_identity_rows].mean()) if non_identity_rows.any() else float("nan"),
        "edit_accuracy": float(flat["edit_accuracy"].mean()),
        "monotonic_violation": float(flat["monotonic_violation"].mean()),
        "endpoint_error": float(flat["endpoint_error"].mean()),
    }

    improvement = flat["input_mse"] - flat["pred_mse"]
    order = np.argsort(improvement)
    example_positions = {"worst": int(order[0]), "median": int(order[len(order) // 2]), "best": int(order[-1])}
    for label, position in example_positions.items():
        original_index = int(val_indices[position])
        np.save(output_dir / f"{label}_corrupted.npy", arrays["corrupted"][original_index])
        np.save(output_dir / f"{label}_predicted.npy", prediction[position])
        np.save(output_dir / f"{label}_target.npy", arrays["target"][original_index])
        np.save(output_dir / f"{label}_predicted_tau.npy", predicted_tau[position])
        np.save(output_dir / f"{label}_target_tau.npy", arrays["target_tau"][original_index])

    duration_bins = {}
    for bin_id in sorted(np.unique(flat["duration_bin"].astype(np.int64)).tolist()):
        rows = flat["duration_bin"].astype(np.int64) == bin_id
        duration_bins[str(bin_id)] = {
            "samples": int(rows.sum()),
            "target_mean": float(flat["target_duration"][rows].mean()),
            "predicted_mean": float(flat["predicted_duration"][rows].mean()),
            "mae": float(flat["duration_mae"][rows].mean()),
        }

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(bundle.get("epoch", -1)),
        "checkpoint_val_loss": float(bundle.get("val_loss", float("nan"))),
        "checkpoint_selection_score": float(bundle.get("selection_score", float("nan"))),
        "training_seed": training_seed,
        "evaluated_samples": int(len(val_indices)),
        "evaluated_sources": int(len(np.unique(arrays["source_id"][val_indices]))),
        "metrics": metrics,
        "duration_bins": duration_bins,
        "target_duration_percentiles": np.percentile(flat["target_duration"], [0, 10, 25, 50, 75, 90, 100]).tolist(),
        "predicted_duration_percentiles": np.percentile(flat["predicted_duration"], [0, 10, 25, 50, 75, 90, 100]).tolist(),
        "examples": example_positions,
    }
    report_path = output_dir / "V23_V2_HELDOUT_EVALUATION.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("V23-v2 HELD-OUT EVALUATION")
    print("=" * 88)
    for key, value in metrics.items():
        print(f"{key:38s} = {value:.8f}")
    print("target duration:", report["target_duration_percentiles"])
    print("pred duration:  ", report["predicted_duration_percentiles"])
    print("saved:", report_path)


if __name__ == "__main__":
    main()
