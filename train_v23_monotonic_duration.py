#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-stage training for V23-v2.3.

Stage ``duration`` trains bin classification, intra-bin residual regression and
edit classification.  Stage ``timewarp`` loads/fixes the duration branch and
trains the monotonic tau branch with scheduled duration teacher forcing.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from model.v23_monotonic_duration import (
    V23MonotonicDurationNet,
    root_yaw_velocity_dps,
    soft_turn_duration_ratio,
    warp_motion_so3,
)


class V23Dataset(Dataset):
    KEYS = (
        "corrupted",
        "target",
        "edit_mask",
        "condition",
        "target_tau",
        "target_duration_frames",
        "turn_input_start",
        "turn_input_end",
        "is_identity",
        "duration_bin",
    )

    def __init__(self, arrays: Dict[str, np.ndarray]):
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays["corrupted"])

    def __getitem__(self, index: int):
        values = []
        for key in self.KEYS:
            value = np.asarray(self.arrays[key][index])
            if key == "duration_bin":
                values.append(torch.tensor(value, dtype=torch.long))
            else:
                values.append(torch.from_numpy(np.asarray(value, dtype=np.float32)))
        return tuple(values)


def _split_score(
    indices: np.ndarray,
    total_count: int,
    val_ratio: float,
    duration_bin: np.ndarray,
    is_identity: np.ndarray,
    target_duration: np.ndarray,
    global_hist: np.ndarray,
    global_identity: float,
    global_mean: float,
    duration_range: float,
) -> float:
    if len(indices) == 0:
        return float("inf")
    bins = duration_bin[indices]
    hist = np.bincount(bins, minlength=len(global_hist)).astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    sample_ratio_error = abs(len(indices) / max(total_count, 1) - val_ratio)
    hist_error = float(np.abs(hist - global_hist).sum())
    identity_error = abs(float(np.mean(is_identity[indices] > 0.5)) - global_identity)
    mean_error = abs(float(np.mean(target_duration[indices])) - global_mean) / max(duration_range, 1.0)
    p90_global = float(np.percentile(target_duration, 90))
    p90_error = abs(float(np.percentile(target_duration[indices], 90)) - p90_global) / max(duration_range, 1.0)
    return 3.0 * sample_ratio_error + hist_error + identity_error + mean_error + 0.5 * p90_error


def duration_stratified_group_split(
    source_id: np.ndarray,
    duration_bin: np.ndarray,
    is_identity: np.ndarray,
    target_duration: np.ndarray,
    val_ratio: float,
    seed: int,
    trials: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Source-level split selected to preserve duration/identity distributions."""
    source_id = np.asarray(source_id)
    duration_bin = np.asarray(duration_bin).astype(np.int64)
    is_identity = np.asarray(is_identity)
    target_duration = np.asarray(target_duration, dtype=np.float32)
    unique_sources = np.unique(source_id)
    n_val = max(1, int(round(len(unique_sources) * float(val_ratio))))
    if n_val >= len(unique_sources):
        raise RuntimeError("Invalid validation ratio")

    num_bins = int(duration_bin.max()) + 1
    global_hist = np.bincount(duration_bin, minlength=num_bins).astype(np.float64)
    global_hist /= max(global_hist.sum(), 1.0)
    global_identity = float(np.mean(is_identity > 0.5))
    global_mean = float(np.mean(target_duration))
    duration_range = float(np.max(target_duration) - np.min(target_duration))
    source_to_indices = {int(source): np.flatnonzero(source_id == source) for source in unique_sources}

    rng = np.random.default_rng(seed)
    best_sources = None
    best_score = float("inf")
    trials = max(int(trials), 256)
    for _ in range(trials):
        chosen = rng.choice(unique_sources, size=n_val, replace=False)
        indices = np.concatenate([source_to_indices[int(source)] for source in chosen])
        score = _split_score(
            indices,
            len(source_id),
            val_ratio,
            duration_bin,
            is_identity,
            target_duration,
            global_hist,
            global_identity,
            global_mean,
            duration_range,
        )
        if score < best_score:
            best_score = score
            best_sources = set(int(value) for value in chosen.tolist())
    if best_sources is None:
        raise RuntimeError("Could not build source-level split")
    val = np.asarray([i for i, source in enumerate(source_id) if int(source) in best_sources], dtype=np.int64)
    train = np.asarray([i for i, source in enumerate(source_id) if int(source) not in best_sources], dtype=np.int64)
    if len(train) == 0 or len(val) == 0:
        raise RuntimeError("Invalid source-level split")
    return train, val


def group_split(
    source_id: np.ndarray,
    val_ratio: float,
    seed: int,
    duration_bin: np.ndarray | None = None,
    is_identity: np.ndarray | None = None,
    target_duration: np.ndarray | None = None,
    trials: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible split helper used by evaluation scripts."""
    if duration_bin is not None and is_identity is not None and target_duration is not None:
        return duration_stratified_group_split(
            source_id, duration_bin, is_identity, target_duration, val_ratio, seed, trials
        )
    unique = np.unique(source_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_ratio)))
    val_sources = set(int(value) for value in unique[:n_val].tolist())
    val = np.asarray([i for i, source in enumerate(source_id) if int(source) in val_sources], dtype=np.int64)
    train = np.asarray([i for i, source in enumerate(source_id) if int(source) not in val_sources], dtype=np.int64)
    return train, val


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    multiplier = 1
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
        multiplier *= value.shape[-1]
    return (value * mask).sum() / (mask.sum() * multiplier + 1e-8)


def rotation_activity(motion: torch.Tensor) -> torch.Tensor:
    velocity = motion[:, 1:, 7:151] - motion[:, :-1, 7:151]
    return torch.linalg.vector_norm(velocity, dim=-1).mean(dim=1)


def safe_corrcoef_tensor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denominator = torch.sqrt(x.square().sum() * y.square().sum()).clamp_min(1e-8)
    return (x * y).sum() / denominator


def safe_corrcoef_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def duration_rank_loss(predicted: torch.Tensor, target: torch.Tensor, margin_frames: float = 2.0) -> torch.Tensor:
    if len(predicted) < 2:
        return predicted.sum() * 0.0
    permutation = torch.randperm(len(predicted), device=predicted.device)
    target_delta = target - target[permutation]
    predicted_delta = predicted - predicted[permutation]
    valid = torch.abs(target_delta) >= 6.0
    if not torch.any(valid):
        return predicted.sum() * 0.0
    sign = torch.sign(target_delta[valid])
    return F.relu(float(margin_frames) - sign * predicted_delta[valid]).mean() / max(float(target.max()), 1.0)


def balanced_binary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive = target >= 0.5
    negative = ~positive
    terms = []
    if torch.any(positive):
        terms.append(losses[positive].mean())
    if torch.any(negative):
        terms.append(losses[negative].mean())
    return torch.stack(terms).mean() if terms else losses.mean()


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def load_duration_edges(data_path: str, arrays: Dict[str, np.ndarray]) -> np.ndarray:
    if "duration_edges" in arrays:
        return np.asarray(arrays["duration_edges"], dtype=np.float32)
    metadata_path = Path(data_path).with_suffix(".metadata.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "duration_edges" in metadata:
            return np.asarray(metadata["duration_edges"], dtype=np.float32)
    bins = np.asarray(arrays["duration_bin"], dtype=np.int64)
    duration = np.asarray(arrays["target_duration_frames"], dtype=np.float32)
    edges = [float(duration.min())]
    for bin_id in range(int(bins.max()) + 1):
        values = duration[bins == bin_id]
        if len(values):
            edges.append(float(values.max()) + 1.0)
    edges = np.maximum.accumulate(np.asarray(edges, dtype=np.float32))
    if np.any(np.diff(edges) <= 0):
        raise RuntimeError("Could not infer duration edges")
    return edges


def dataset_arrays(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        required = set(V23Dataset.KEYS) | {"source_id", "speed_factor"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(f"Dataset missing: {missing}")
        return {key: np.asarray(archive[key]) for key in archive.files}


def make_sampler(
    arrays: Dict[str, np.ndarray],
    train_indices: np.ndarray,
    enabled: bool,
) -> WeightedRandomSampler | None:
    if not enabled:
        return None
    bins = np.asarray(arrays["duration_bin"])[train_indices].astype(np.int64)
    identity = (np.asarray(arrays["is_identity"])[train_indices] > 0.5).astype(np.int64)
    class_id = identity * (int(bins.max()) + 1) + bins
    counts = np.bincount(class_id, minlength=int(class_id.max()) + 1).astype(np.float64)
    weights = 1.0 / np.sqrt(np.maximum(counts[class_id], 1.0))
    weights /= max(weights.mean(), 1e-8)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_indices),
        replacement=True,
    )


def teacher_forcing_ratio(epoch: int, start: float, end: float, decay_epochs: int) -> float:
    if decay_epochs <= 1:
        return float(end)
    progress = np.clip((epoch - 1) / float(decay_epochs - 1), 0.0, 1.0)
    return float(start + (end - start) * progress)


def duration_group_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    target_bin: np.ndarray,
    predicted_bin: np.ndarray,
    edit_probability: np.ndarray,
    edit_target: np.ndarray,
    num_bins: int,
) -> Dict[str, float]:
    error = np.abs(predicted - target)
    metrics: Dict[str, float] = {
        "duration_mae_frames": float(np.mean(error)),
        "duration_correlation": safe_corrcoef_np(predicted, target),
        "duration_bin_accuracy": float(np.mean(predicted_bin == target_bin)),
        "edit_accuracy": float(np.mean((edit_probability >= 0.5) == (edit_target >= 0.5))),
    }
    bin_maes = []
    for bin_id in range(num_bins):
        rows = target_bin == bin_id
        value = float(np.mean(error[rows])) if np.any(rows) else float("nan")
        metrics[f"duration_bin_{bin_id}_mae"] = value
        if np.isfinite(value):
            bin_maes.append(value)
    short_rows = target_bin <= max(0, num_bins // 3 - 1)
    long_rows = target_bin >= max(0, num_bins - max(1, num_bins // 3))
    middle_rows = ~(short_rows | long_rows)
    for name, rows in (("short", short_rows), ("medium", middle_rows), ("long", long_rows)):
        metrics[f"duration_{name}_mae"] = float(np.mean(error[rows])) if np.any(rows) else float("nan")
    metrics["rare_duration_mae_frames"] = float(np.mean(error[target < np.percentile(target, 90)]))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--stage", choices=["duration", "timewarp", "joint"], required=True)
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--min_lr", type=float, default=5e-7)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=20260620)
    parser.add_argument("--split_trials", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--lr_patience", type=int, default=8)
    parser.add_argument("--balanced_sampler", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260610)

    # Stage-1 duration losses.
    parser.add_argument("--lambda_bin", type=float, default=1.0)
    parser.add_argument("--lambda_residual", type=float, default=1.0)
    parser.add_argument("--lambda_relative", type=float, default=1.2)
    parser.add_argument("--lambda_log_duration", type=float, default=0.8)
    parser.add_argument("--lambda_linear_duration", type=float, default=0.5)
    parser.add_argument("--lambda_duration_rank", type=float, default=0.15)
    parser.add_argument("--lambda_edit", type=float, default=0.30)
    parser.add_argument("--long_duration_weight", type=float, default=1.25)

    # Stage-2 time-warp losses.
    parser.add_argument("--teacher_forcing_start", type=float, default=1.0)
    parser.add_argument("--teacher_forcing_end", type=float, default=0.0)
    parser.add_argument("--teacher_forcing_decay_epochs", type=int, default=60)
    parser.add_argument("--lambda_tau", type=float, default=2.0)
    parser.add_argument("--lambda_duration_consistency", type=float, default=0.8)
    parser.add_argument("--lambda_motion", type=float, default=0.9)
    parser.add_argument("--lambda_context", type=float, default=0.30)
    parser.add_argument("--lambda_velocity", type=float, default=0.35)
    parser.add_argument("--lambda_activity", type=float, default=0.25)
    parser.add_argument("--lambda_yaw", type=float, default=0.45)
    parser.add_argument("--lambda_peak_yaw", type=float, default=0.18)
    parser.add_argument("--lambda_smooth", type=float, default=0.05)
    parser.add_argument("--lambda_identity_tau", type=float, default=0.55)
    parser.add_argument("--lambda_identity_motion", type=float, default=0.40)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    arrays = dataset_arrays(args.data)
    n, time_steps, motion_dim = arrays["corrupted"].shape
    if motion_dim != 151 or arrays["target"].shape != (n, time_steps, motion_dim):
        raise ValueError("Invalid motion arrays")
    duration_edges = load_duration_edges(args.data, arrays)
    num_bins = len(duration_edges) - 1
    if int(np.max(arrays["duration_bin"])) >= num_bins:
        raise ValueError("duration_bin is incompatible with duration_edges")

    train_indices, val_indices = duration_stratified_group_split(
        arrays["source_id"],
        arrays["duration_bin"],
        arrays["is_identity"],
        arrays["target_duration_frames"],
        args.val_ratio,
        args.split_seed,
        args.split_trials,
    )
    dataset = V23Dataset(arrays)
    sampler = make_sampler(arrays, train_indices, bool(args.balanced_sampler))
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V23MonotonicDurationNet(
        motion_dim=151,
        condition_dim=int(arrays["condition"].shape[1]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        duration_edges=duration_edges.tolist(),
        window_len=time_steps,
    ).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.set_train_stage(args.stage)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"No trainable parameters for stage {args.stage}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=args.min_lr
    )
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_bin_counts = np.bincount(
        np.asarray(arrays["duration_bin"])[train_indices].astype(np.int64), minlength=num_bins
    ).astype(np.float32)
    class_weights = 1.0 / np.sqrt(np.maximum(train_bin_counts, 1.0))
    class_weights /= max(float(np.mean(class_weights)), 1e-8)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    split = {
        "split_strategy": "duration_stratified_source",
        "split_seed": int(args.split_seed),
        "samples": int(n),
        "train_samples": int(len(train_indices)),
        "val_samples": int(len(val_indices)),
        "train_sources": np.unique(arrays["source_id"][train_indices]).astype(int).tolist(),
        "val_sources": np.unique(arrays["source_id"][val_indices]).astype(int).tolist(),
        "train_duration_percentiles": np.percentile(
            arrays["target_duration_frames"][train_indices], [0, 10, 25, 50, 75, 90, 100]
        ).tolist(),
        "val_duration_percentiles": np.percentile(
            arrays["target_duration_frames"][val_indices], [0, 10, 25, 50, 75, 90, 100]
        ).tolist(),
        "train_bin_counts": np.bincount(
            arrays["duration_bin"][train_indices].astype(int), minlength=num_bins
        ).tolist(),
        "val_bin_counts": np.bincount(
            arrays["duration_bin"][val_indices].astype(int), minlength=num_bins
        ).tolist(),
    }
    (out_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    def unpack(batch):
        return [value.to(device, non_blocking=True) for value in batch]

    def duration_loss_batch(batch):
        (
            corrupted,
            _target,
            mask,
            condition,
            _target_tau,
            target_duration,
            _turn_start,
            _turn_end,
            is_identity,
            duration_bin,
        ) = unpack(batch)
        result = model.predict_duration(corrupted, mask, condition, use_hard_duration=False)
        predicted = result["duration_soft_frames"]
        target_range = max(float(duration_edges[-1] - duration_edges[0]), 1.0)
        normalized_target = ((target_duration - float(duration_edges[0])) / target_range).clamp(0.0, 1.0)
        sample_weight = 0.75 + args.long_duration_weight * normalized_target

        bin_loss = F.cross_entropy(result["duration_bin_logits"], duration_bin, weight=class_weights_t)
        true_residual_logit = result["duration_residual_logits"].gather(1, duration_bin[:, None]).squeeze(1)
        lower = model.duration_edges[:-1][duration_bin]
        upper = model.duration_edges[1:][duration_bin] - 1.0
        target_residual = ((target_duration - lower) / (upper - lower).clamp_min(1.0)).clamp(0.0, 1.0)
        residual_per_sample = F.smooth_l1_loss(
            torch.sigmoid(true_residual_logit), target_residual, reduction="none"
        )
        residual_loss = weighted_mean(residual_per_sample, sample_weight)
        relative_loss = weighted_mean(
            torch.abs(predicted - target_duration) / target_duration.clamp_min(1.0), sample_weight
        )
        log_loss = weighted_mean(
            F.smooth_l1_loss(
                torch.log(predicted.clamp_min(1.0)),
                torch.log(target_duration.clamp_min(1.0)),
                reduction="none",
            ),
            sample_weight,
        )
        linear_loss = weighted_mean(
            F.smooth_l1_loss(predicted / float(time_steps), target_duration / float(time_steps), reduction="none"),
            sample_weight,
        )
        rank_loss = duration_rank_loss(predicted, target_duration)
        edit_target = 1.0 - is_identity.clamp(0.0, 1.0)
        edit_loss = balanced_binary_loss(result["edit_logit"], edit_target)
        total = (
            args.lambda_bin * bin_loss
            + args.lambda_residual * residual_loss
            + args.lambda_relative * relative_loss
            + args.lambda_log_duration * log_loss
            + args.lambda_linear_duration * linear_loss
            + args.lambda_duration_rank * rank_loss
            + args.lambda_edit * edit_loss
        )
        return total, {
            "loss": total,
            "bin": bin_loss,
            "residual": residual_loss,
            "relative": relative_loss,
            "log_duration": log_loss,
            "linear_duration": linear_loss,
            "rank": rank_loss,
            "edit": edit_loss,
        }, {
            "pred_duration": predicted.detach(),
            "target_duration": target_duration.detach(),
            "pred_bin": torch.argmax(result["duration_bin_logits"], dim=-1).detach(),
            "target_bin": duration_bin.detach(),
            "edit_probability": result["edit_probability"].detach(),
            "edit_target": edit_target.detach(),
        }

    def timewarp_loss_batch(batch, tf_ratio: float):
        (
            corrupted,
            target,
            mask,
            condition,
            target_tau,
            target_duration,
            turn_start,
            turn_end,
            is_identity,
            duration_bin,
        ) = unpack(batch)
        del duration_bin
        with torch.no_grad():
            duration_result = model.predict_duration(corrupted, mask, condition, use_hard_duration=True)
            predicted_duration = duration_result["duration_hard_frames"]
            duration_for_tau = float(tf_ratio) * target_duration + (1.0 - float(tf_ratio)) * predicted_duration
        tau_result = model.predict_tau(corrupted, mask, condition, duration_for_tau)
        tau = tau_result["tau"]
        predicted_motion = warp_motion_so3(corrupted, tau)

        tau_weight = 0.15 + 1.85 * mask
        tau_loss = ((tau - target_tau).abs() * tau_weight).sum() / tau_weight.sum().clamp_min(1.0)
        soft_duration = soft_turn_duration_ratio(tau, turn_start, turn_end) * float(time_steps)
        duration_consistency = F.smooth_l1_loss(
            soft_duration / float(time_steps), target_duration / float(time_steps)
        )
        motion_weight = 0.15 + 1.85 * mask
        motion_loss = masked_mean((predicted_motion[..., 4:151] - target[..., 4:151]) ** 2, motion_weight)
        outside = (1.0 - mask).clamp(0.0, 1.0)
        context_loss = masked_mean(
            F.smooth_l1_loss(predicted_motion[..., 4:151], target[..., 4:151], reduction="none"),
            0.10 + outside,
        )
        predicted_velocity = predicted_motion[:, 1:, 7:151] - predicted_motion[:, :-1, 7:151]
        target_velocity = target[:, 1:, 7:151] - target[:, :-1, 7:151]
        velocity_mask = torch.maximum(mask[:, 1:], mask[:, :-1])
        velocity_loss = masked_mean(
            F.smooth_l1_loss(predicted_velocity, target_velocity, reduction="none"),
            0.20 + velocity_mask,
        )
        predicted_activity = rotation_activity(predicted_motion)
        target_activity = rotation_activity(target)
        activity_loss = F.smooth_l1_loss(
            torch.log1p(predicted_activity), torch.log1p(target_activity)
        )
        predicted_yaw = root_yaw_velocity_dps(predicted_motion)
        target_yaw = root_yaw_velocity_dps(target)
        yaw_mask = torch.maximum(mask[:, 1:], mask[:, :-1])
        yaw_loss = masked_mean(
            F.smooth_l1_loss(predicted_yaw / 180.0, target_yaw / 180.0, reduction="none"),
            0.15 + yaw_mask,
        )
        predicted_peak = torch.quantile(torch.abs(predicted_yaw), 0.95, dim=1)
        target_peak = torch.quantile(torch.abs(target_yaw), 0.95, dim=1)
        peak_yaw_loss = F.smooth_l1_loss(predicted_peak / 180.0, target_peak / 180.0)
        increments = tau[:, 1:] - tau[:, :-1]
        smooth_loss = torch.mean((increments[:, 1:] - increments[:, :-1]) ** 2)

        uniform_tau = torch.linspace(0.0, 1.0, time_steps, device=device, dtype=tau.dtype)[None]
        identity_rows = is_identity > 0.5
        identity_count = identity_rows.float().sum().clamp_min(1.0)
        identity_tau_per_sample = torch.abs(tau - uniform_tau).mean(dim=1)
        identity_tau_loss = (identity_tau_per_sample * identity_rows.float()).sum() / identity_count
        identity_motion_per_sample = torch.abs(
            predicted_motion[..., 4:151] - corrupted[..., 4:151]
        ).mean(dim=(1, 2))
        identity_motion_loss = (identity_motion_per_sample * identity_rows.float()).sum() / identity_count
        total = (
            args.lambda_tau * tau_loss
            + args.lambda_duration_consistency * duration_consistency
            + args.lambda_motion * motion_loss
            + args.lambda_context * context_loss
            + args.lambda_velocity * velocity_loss
            + args.lambda_activity * activity_loss
            + args.lambda_yaw * yaw_loss
            + args.lambda_peak_yaw * peak_yaw_loss
            + args.lambda_smooth * smooth_loss
            + args.lambda_identity_tau * identity_tau_loss
            + args.lambda_identity_motion * identity_motion_loss
        )
        input_mse = ((corrupted[..., 4:151] - target[..., 4:151]) ** 2).mean(dim=(1, 2))
        pred_mse = ((predicted_motion[..., 4:151] - target[..., 4:151]) ** 2).mean(dim=(1, 2))
        input_yaw_mae = torch.abs(root_yaw_velocity_dps(corrupted) - target_yaw).mean(dim=1)
        pred_yaw_mae = torch.abs(predicted_yaw - target_yaw).mean(dim=1)
        edit_target = 1.0 - is_identity.clamp(0.0, 1.0)
        return total, {
            "loss": total,
            "tau": tau_loss,
            "duration_consistency": duration_consistency,
            "motion": motion_loss,
            "context": context_loss,
            "velocity": velocity_loss,
            "activity": activity_loss,
            "yaw": yaw_loss,
            "peak_yaw": peak_yaw_loss,
            "smooth": smooth_loss,
            "identity_tau": identity_tau_loss,
            "identity_motion": identity_motion_loss,
        }, {
            "pred_duration": predicted_duration.detach(),
            "target_duration": target_duration.detach(),
            "pred_bin": duration_result["duration_bin_index"].detach(),
            "target_bin": torch.zeros_like(duration_result["duration_bin_index"]),
            "edit_probability": duration_result["edit_probability"].detach(),
            "edit_target": edit_target.detach(),
            "tau_mae": torch.abs(tau - target_tau).mean(dim=1).detach(),
            "input_mse": input_mse.detach(),
            "pred_mse": pred_mse.detach(),
            "input_yaw_mae": input_yaw_mae.detach(),
            "pred_yaw_mae": pred_yaw_mae.detach(),
            "activity_ratio": (predicted_activity / target_activity.clamp_min(1e-6)).detach(),
            "identity_tau": identity_tau_per_sample[identity_rows].detach(),
            "identity_motion": identity_motion_per_sample[identity_rows].detach(),
        }

    def run_epoch(loader, training: bool, epoch: int) -> Dict[str, float]:
        model.train(training)
        if args.stage == "timewarp":
            # The calibrated duration branch is frozen and must also stay in
            # evaluation mode so dropout does not inject duration noise.
            model.duration_encoder.eval()
            model.duration_bin_head.eval()
            model.duration_residual_head.eval()
            model.edit_head.eval()
            model.tau_encoder.train(training)
            model.duration_embedding.train(training)
            model.tau_increment_head.train(training)
        elif args.stage == "duration":
            model.tau_encoder.eval()
            model.duration_embedding.eval()
            model.tau_increment_head.eval()
        if not training:
            model.eval()
        loss_totals: Dict[str, float] = {}
        sample_count = 0
        collected: Dict[str, list[np.ndarray]] = {}
        tf_ratio = teacher_forcing_ratio(
            epoch,
            args.teacher_forcing_start,
            args.teacher_forcing_end,
            args.teacher_forcing_decay_epochs,
        ) if args.stage != "duration" else 0.0

        for batch in loader:
            with torch.set_grad_enabled(training):
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    if args.stage == "duration":
                        loss, parts, values = duration_loss_batch(batch)
                    else:
                        loss, parts, values = timewarp_loss_batch(batch, tf_ratio)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
            batch_size = len(batch[0])
            sample_count += batch_size
            for key, value in parts.items():
                loss_totals[key] = loss_totals.get(key, 0.0) + float(value.detach()) * batch_size
            for key, value in values.items():
                array = value.detach().cpu().numpy()
                collected.setdefault(key, []).append(array)

        metrics = {key: value / max(sample_count, 1) for key, value in loss_totals.items()}
        merged = {key: np.concatenate(parts) if parts else np.empty((0,)) for key, parts in collected.items()}
        target_bin = np.asarray(arrays["duration_bin"])[val_indices if not training else train_indices]
        # Sampler changes training order/distribution, therefore use collected target bins there.
        if args.stage == "duration":
            duration_metrics = duration_group_metrics(
                merged["pred_duration"],
                merged["target_duration"],
                merged["target_bin"].astype(int),
                merged["pred_bin"].astype(int),
                merged["edit_probability"],
                merged["edit_target"],
                num_bins,
            )
        else:
            true_bins = np.digitize(
                merged["target_duration"], duration_edges[1:-1], right=False
            ).astype(int)
            duration_metrics = duration_group_metrics(
                merged["pred_duration"],
                merged["target_duration"],
                true_bins,
                merged["pred_bin"].astype(int),
                merged["edit_probability"],
                merged["edit_target"],
                num_bins,
            )
            metrics.update({
                "tau_mae": float(np.mean(merged["tau_mae"])),
                "motion_mse_ratio": float(np.mean(merged["pred_mse"]) / max(np.mean(merged["input_mse"]), 1e-12)),
                "yaw_mae_ratio": float(np.mean(merged["pred_yaw_mae"]) / max(np.mean(merged["input_yaw_mae"]), 1e-12)),
                "activity_ratio": float(np.mean(merged["activity_ratio"])),
                "identity_tau_mae": float(np.mean(merged["identity_tau"])) if len(merged["identity_tau"]) else 0.0,
                "identity_motion_drift": float(np.mean(merged["identity_motion"])) if len(merged["identity_motion"]) else 0.0,
                "teacher_forcing_ratio": float(tf_ratio),
            })
        metrics.update(duration_metrics)
        return metrics

    config = {
        "version": "v23_v2_3_bin_residual_two_stage",
        "stage": args.stage,
        "motion_dim": 151,
        "condition_dim": int(arrays["condition"].shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "window_len": int(time_steps),
        "fps": 30.0,
        "duration_edges": duration_edges.tolist(),
        "duration_dilations": [1, 2, 4, 8, 16, 32],
        "tau_dilations": [1, 2, 4, 8, 16, 32],
        "split_seed": int(args.split_seed),
        "split_trials": int(args.split_trials),
        "val_ratio": float(args.val_ratio),
    }

    best_score = float("inf")
    best_epoch = -1
    best_val = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(train_loader, True, epoch)
        val_metrics = run_epoch(val_loader, False, epoch)
        if args.stage == "duration":
            selection_score = (
                0.10 * val_metrics["loss"]
                + val_metrics["duration_mae_frames"] / float(time_steps)
                + 0.50 * val_metrics["duration_long_mae"] / float(time_steps)
                + 0.20 * max(0.0, 0.90 - val_metrics["duration_correlation"])
                + 0.10 * (1.0 - val_metrics["duration_bin_accuracy"])
                + 0.10 * (1.0 - val_metrics["edit_accuracy"])
            )
        else:
            selection_score = (
                2.0 * val_metrics["tau_mae"]
                + 0.25 * val_metrics["motion_mse_ratio"]
                + 0.25 * val_metrics["yaw_mae_ratio"]
                + 0.10 * abs(1.0 - val_metrics["activity_ratio"])
                + 0.50 * val_metrics["identity_tau_mae"]
                + 0.10 * (1.0 - val_metrics["edit_accuracy"])
                + 0.10 * val_metrics["duration_mae_frames"] / float(time_steps)
            )
        scheduler.step(selection_score)
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "selection_score": selection_score,
            "lr": optimizer.param_groups[0]["lr"],
        })
        if selection_score < best_score - 1e-7:
            best_score = float(selection_score)
            best_epoch = epoch
            best_val = float(val_metrics["loss"])
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "stage": args.stage,
                    "epoch": epoch,
                    "val_loss": best_val,
                    "selection_score": best_score,
                    "val_metrics": val_metrics,
                    "split": split,
                    "init_checkpoint": args.init_checkpoint,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            if args.stage == "duration":
                print(
                    f"stage=duration epoch={epoch:04d} train={train_metrics['loss']:.6f} "
                    f"val={val_metrics['loss']:.6f} score={selection_score:.6f} "
                    f"dur={val_metrics['duration_mae_frames']:.3f} "
                    f"short={val_metrics['duration_short_mae']:.3f} "
                    f"mid={val_metrics['duration_medium_mae']:.3f} "
                    f"long={val_metrics['duration_long_mae']:.3f} "
                    f"corr={val_metrics['duration_correlation']:.3f} "
                    f"bin={val_metrics['duration_bin_accuracy']:.3f} "
                    f"edit={val_metrics['edit_accuracy']:.3f} best={best_score:.6f}@{best_epoch}",
                    flush=True,
                )
            else:
                print(
                    f"stage=timewarp epoch={epoch:04d} train={train_metrics['loss']:.6f} "
                    f"val={val_metrics['loss']:.6f} score={selection_score:.6f} "
                    f"tf={val_metrics['teacher_forcing_ratio']:.3f} "
                    f"dur={val_metrics['duration_mae_frames']:.3f} "
                    f"tau={val_metrics['tau_mae']:.5f} "
                    f"motion_ratio={val_metrics['motion_mse_ratio']:.3f} "
                    f"yaw_ratio={val_metrics['yaw_mae_ratio']:.3f} "
                    f"activity={val_metrics['activity_ratio']:.3f} "
                    f"id_tau={val_metrics['identity_tau_mae']:.5f} "
                    f"edit={val_metrics['edit_accuracy']:.3f} best={best_score:.6f}@{best_epoch}",
                    flush=True,
                )
        if args.patience > 0 and stale >= args.patience:
            print(f"early_stop stage={args.stage} epoch={epoch} best={best_score:.6f}@{best_epoch}", flush=True)
            break

    final_payload = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "stage": args.stage,
        "epoch": history[-1]["epoch"],
        "val_loss": history[-1]["val"]["loss"],
        "selection_score": history[-1]["selection_score"],
        "val_metrics": history[-1]["val"],
        "split": split,
        "init_checkpoint": args.init_checkpoint,
    }
    torch.save(final_payload, checkpoint_dir / "final.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "BEST_V23_CKPT.txt").write_text(
        str((checkpoint_dir / "best.pt").resolve()) + "\n", encoding="utf-8"
    )
    print("best:", checkpoint_dir / "best.pt")
    print("best val loss:", best_val, "selection score:", best_score, "epoch:", best_epoch)


if __name__ == "__main__":
    main()
