#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V23-v2.4 event-consistent ordinal duration and two-stage time-warp training.

Stage 1 operates on natural-event groups instead of treating synthetic views as
independent samples.  Every training item contains two views of the same event
(preferably one identity and one compressed view).  The natural-duration branch
is trained to be invariant across views, while the edit classifier is trained
separately and remains pace-sensitive.

Stage 2 freezes the calibrated duration branch and learns monotonic tau with
scheduled duration teacher forcing.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

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


SAMPLE_KEYS = (
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
    "event_uid",
)

DURATION_KEYS = (
    "corrupted",
    "edit_mask",
    "condition",
    "target_duration_frames",
    "is_identity",
    "duration_bin",
    "event_uid",
)


class V23SampleDataset(Dataset):
    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        indices: Sequence[int] | None = None,
        keys: Sequence[str] = SAMPLE_KEYS,
    ):
        self.arrays = arrays
        self.keys = tuple(keys)
        self.indices = np.arange(len(arrays["corrupted"]), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def _sample(self, raw_index: int):
        values = []
        for key in self.keys:
            value = np.asarray(self.arrays[key][raw_index])
            if key in {"duration_bin", "event_uid"}:
                values.append(torch.tensor(value, dtype=torch.long))
            else:
                values.append(torch.from_numpy(np.asarray(value, dtype=np.float32)))
        return tuple(values)

    def __getitem__(self, index: int):
        return self._sample(int(self.indices[index]))


class V23EventPairDataset(V23SampleDataset):
    """One item per natural event, returning two independently corrupted views."""

    def __init__(self, arrays: Dict[str, np.ndarray], indices: Sequence[int], seed: int):
        super().__init__(arrays, indices, keys=DURATION_KEYS)
        groups: Dict[int, list[int]] = defaultdict(list)
        for raw_index in self.indices.tolist():
            groups[int(arrays["event_uid"][raw_index])].append(int(raw_index))
        self.event_uids = np.asarray(sorted(groups), dtype=np.int64)
        self.groups = {uid: np.asarray(groups[int(uid)], dtype=np.int64) for uid in self.event_uids}
        self.identity_groups = {
            int(uid): indices_[np.asarray(arrays["is_identity"])[indices_] > 0.5]
            for uid, indices_ in self.groups.items()
        }
        self.nonidentity_groups = {
            int(uid): indices_[np.asarray(arrays["is_identity"])[indices_] <= 0.5]
            for uid, indices_ in self.groups.items()
        }
        self.event_bins = np.asarray(
            [int(np.asarray(arrays["duration_bin"])[self.groups[int(uid)][0]]) for uid in self.event_uids],
            dtype=np.int64,
        )
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.event_uids)

    def __getitem__(self, index: int):
        uid = int(self.event_uids[index])
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch + 7919 * uid)
        identity = self.identity_groups[uid]
        nonidentity = self.nonidentity_groups[uid]
        all_indices = self.groups[uid]
        if len(identity) and len(nonidentity):
            first = int(identity[rng.integers(0, len(identity))])
            second = int(nonidentity[rng.integers(0, len(nonidentity))])
            if (self.epoch + index) % 2:
                first, second = second, first
        elif len(all_indices) >= 2:
            pair = rng.choice(all_indices, size=2, replace=False)
            first, second = int(pair[0]), int(pair[1])
        else:
            first = second = int(all_indices[0])
        return self._sample(first), self._sample(second)


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
    quantile_error = 0.0
    for percentile in (10, 50, 75, 90):
        quantile_error += abs(
            float(np.percentile(target_duration[indices], percentile))
            - float(np.percentile(target_duration, percentile))
        ) / max(duration_range, 1.0)
    return 3.0 * sample_ratio_error + hist_error + identity_error + mean_error + 0.25 * quantile_error


def duration_stratified_group_split(
    source_id: np.ndarray,
    duration_bin: np.ndarray,
    is_identity: np.ndarray,
    target_duration: np.ndarray,
    val_ratio: float,
    seed: int,
    trials: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
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
    for _ in range(max(int(trials), 256)):
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
    valid = torch.abs(target_delta) >= 8.0
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


def symmetric_kl(probability_a: torch.Tensor, probability_b: torch.Tensor) -> torch.Tensor:
    a = probability_a.clamp_min(1e-7)
    b = probability_b.clamp_min(1e-7)
    mean = 0.5 * (a + b)
    return 0.5 * (
        torch.sum(a * (torch.log(a) - torch.log(mean)), dim=-1)
        + torch.sum(b * (torch.log(b) - torch.log(mean)), dim=-1)
    ).mean()


def load_duration_edges(data_path: str, arrays: Dict[str, np.ndarray]) -> np.ndarray:
    if "duration_edges" in arrays:
        return np.asarray(arrays["duration_edges"], dtype=np.float32)
    metadata_path = Path(data_path).with_suffix(".metadata.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "duration_edges" in metadata:
            return np.asarray(metadata["duration_edges"], dtype=np.float32)
    raise RuntimeError("Dataset does not contain duration_edges")


def dataset_arrays(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        required = set(SAMPLE_KEYS) | {
            "source_id",
            "speed_factor",
            "corrupted_duration_frames",
            "augmentation_id",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(
                f"Dataset missing V23-v2.4 fields: {missing}. Rebuild the database with the v2.4 builder."
            )
        return {key: np.asarray(archive[key]) for key in archive.files}


def make_sample_sampler(
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


def make_event_sampler(dataset: V23EventPairDataset, enabled: bool) -> WeightedRandomSampler | None:
    if not enabled:
        return None
    counts = np.bincount(dataset.event_bins, minlength=int(dataset.event_bins.max()) + 1).astype(np.float64)
    weights = 1.0 / np.sqrt(np.maximum(counts[dataset.event_bins], 1.0))
    weights /= max(weights.mean(), 1e-8)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
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
    event_uid: np.ndarray | None = None,
    bin_probability: np.ndarray | None = None,
) -> Dict[str, float]:
    error = np.abs(predicted - target)
    bin_distance = np.abs(predicted_bin - target_bin)
    metrics: Dict[str, float] = {
        "duration_mae_frames": float(np.mean(error)),
        "duration_correlation": safe_corrcoef_np(predicted, target),
        "duration_bin_accuracy": float(np.mean(predicted_bin == target_bin)),
        "duration_within_one_bin_accuracy": float(np.mean(bin_distance <= 1)),
        "duration_ordinal_mae_bins": float(np.mean(bin_distance)),
        "edit_accuracy": float(np.mean((edit_probability >= 0.5) == (edit_target >= 0.5))),
    }
    for bin_id in range(num_bins):
        rows = target_bin == bin_id
        metrics[f"duration_bin_{bin_id}_mae"] = float(np.mean(error[rows])) if np.any(rows) else float("nan")
    short_rows = target_bin <= max(0, num_bins // 3 - 1)
    long_rows = target_bin >= max(0, num_bins - max(1, num_bins // 3))
    medium_rows = ~(short_rows | long_rows)
    for name, rows in (("short", short_rows), ("medium", medium_rows), ("long", long_rows)):
        metrics[f"duration_{name}_mae"] = float(np.mean(error[rows])) if np.any(rows) else float("nan")

    if event_uid is not None:
        event_uid = np.asarray(event_uid, dtype=np.int64)
        unique = np.unique(event_uid)
        event_pred, event_target, event_target_bin, event_pred_bin = [], [], [], []
        for uid in unique:
            rows = event_uid == uid
            event_pred.append(float(np.mean(predicted[rows])))
            event_target.append(float(np.mean(target[rows])))
            event_target_bin.append(int(Counter(target_bin[rows].tolist()).most_common(1)[0][0]))
            if bin_probability is not None:
                event_pred_bin.append(int(np.argmax(np.mean(bin_probability[rows], axis=0))))
            else:
                event_pred_bin.append(int(Counter(predicted_bin[rows].tolist()).most_common(1)[0][0]))
        event_pred = np.asarray(event_pred)
        event_target = np.asarray(event_target)
        event_target_bin = np.asarray(event_target_bin)
        event_pred_bin = np.asarray(event_pred_bin)
        event_error = np.abs(event_pred - event_target)
        event_distance = np.abs(event_pred_bin - event_target_bin)
        metrics.update(
            {
                "event_duration_mae_frames": float(np.mean(event_error)),
                "event_duration_correlation": safe_corrcoef_np(event_pred, event_target),
                "event_duration_bin_accuracy": float(np.mean(event_pred_bin == event_target_bin)),
                "event_duration_within_one_bin_accuracy": float(np.mean(event_distance <= 1)),
                "event_duration_ordinal_mae_bins": float(np.mean(event_distance)),
            }
        )
        event_long = event_target_bin >= max(0, num_bins - max(1, num_bins // 3))
        event_medium = (event_target_bin >= max(1, num_bins // 3)) & (~event_long)
        metrics["event_duration_long_mae"] = float(np.mean(event_error[event_long])) if np.any(event_long) else float("nan")
        metrics["event_duration_medium_mae"] = float(np.mean(event_error[event_medium])) if np.any(event_medium) else float("nan")
    return metrics


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self, model: torch.nn.Module) -> None:
        self.backup = {}
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.backup:
                parameter.copy_(self.backup[name])
        self.backup = {}


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    minimum_ratio: float,
):
    def schedule(epoch_index: int) -> float:
        epoch = epoch_index + 1
        if epoch <= max(1, warmup_epochs):
            return max(0.05, epoch / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--stage", choices=["duration", "timewarp", "joint"], required=True)
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min_lr", type=float, default=3e-7)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.24)
    parser.add_argument("--slow_feature_span", type=int, default=10)
    parser.add_argument("--ordinal_blend", type=float, default=0.82)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=20260620)
    parser.add_argument("--split_trials", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--event_num_workers", type=int, default=0)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--balanced_sampler", type=int, default=1)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--condition_noise_std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260610)

    # Stage-1 ordinal/event-consistent duration losses.
    parser.add_argument("--lambda_ordinal", type=float, default=1.0)
    parser.add_argument("--lambda_residual", type=float, default=0.8)
    parser.add_argument("--lambda_relative", type=float, default=1.0)
    parser.add_argument("--lambda_log_duration", type=float, default=0.6)
    parser.add_argument("--lambda_direct", type=float, default=0.35)
    parser.add_argument("--lambda_underestimate", type=float, default=0.8)
    parser.add_argument("--lambda_duration_rank", type=float, default=0.18)
    parser.add_argument("--lambda_pair_duration", type=float, default=0.9)
    parser.add_argument("--lambda_pair_distribution", type=float, default=0.35)
    parser.add_argument("--lambda_moment", type=float, default=0.30)
    parser.add_argument("--lambda_edit", type=float, default=0.25)
    parser.add_argument("--long_duration_weight", type=float, default=1.5)

    # Stage-2 time-warp losses.
    parser.add_argument("--teacher_forcing_start", type=float, default=1.0)
    parser.add_argument("--teacher_forcing_end", type=float, default=0.0)
    parser.add_argument("--teacher_forcing_decay_epochs", type=int, default=90)
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
    sample_count, time_steps, motion_dim = arrays["corrupted"].shape
    if motion_dim != 151 or arrays["target"].shape != (sample_count, time_steps, motion_dim):
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

    sample_dataset = V23SampleDataset(arrays)
    duration_sample_dataset = V23SampleDataset(arrays, keys=DURATION_KEYS)
    validation_dataset = duration_sample_dataset if args.stage == "duration" else sample_dataset
    val_loader = DataLoader(
        Subset(validation_dataset, val_indices.tolist()),
        batch_size=max(args.batch_size, 64),
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    if args.stage == "duration":
        event_dataset = V23EventPairDataset(arrays, train_indices, args.seed)
        event_sampler = make_event_sampler(event_dataset, bool(args.balanced_sampler))
        train_loader = DataLoader(
            event_dataset,
            batch_size=args.batch_size,
            shuffle=event_sampler is None,
            sampler=event_sampler,
            drop_last=True,
            num_workers=args.event_num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        )
    else:
        train_subset = Subset(sample_dataset, train_indices.tolist())
        sample_sampler = make_sample_sampler(arrays, train_indices, bool(args.balanced_sampler))
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=sample_sampler is None,
            sampler=sample_sampler,
            drop_last=True,
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
        slow_feature_span=args.slow_feature_span,
        ordinal_blend=args.ordinal_blend,
    ).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.set_train_stage(args.stage)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"No trainable parameters for stage {args.stage}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = cosine_warmup_scheduler(
        optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        minimum_ratio=max(args.min_lr / max(args.lr, 1e-12), 1e-4),
    )
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ModelEMA(model, args.ema_decay) if 0.0 < args.ema_decay < 1.0 else None

    train_event_bins = []
    seen_uids = set()
    for index in train_indices:
        uid = int(arrays["event_uid"][index])
        if uid not in seen_uids:
            seen_uids.add(uid)
            train_event_bins.append(int(arrays["duration_bin"][index]))
    train_event_bins_np = np.asarray(train_event_bins, dtype=np.int64)
    threshold_positive = np.asarray(
        [np.mean(train_event_bins_np > threshold) for threshold in range(num_bins - 1)],
        dtype=np.float32,
    )
    ordinal_pos_weight = np.clip((1.0 - threshold_positive) / np.maximum(threshold_positive, 1e-4), 0.35, 3.0)
    ordinal_pos_weight_t = torch.tensor(ordinal_pos_weight, dtype=torch.float32, device=device)

    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    split = {
        "split_strategy": "duration_stratified_source_event_grouped",
        "split_seed": int(args.split_seed),
        "samples": int(sample_count),
        "unique_events": int(len(np.unique(arrays["event_uid"]))),
        "train_samples": int(len(train_indices)),
        "val_samples": int(len(val_indices)),
        "train_events": int(len(np.unique(arrays["event_uid"][train_indices]))),
        "val_events": int(len(np.unique(arrays["event_uid"][val_indices]))),
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
        "ordinal_pos_weight": ordinal_pos_weight.tolist(),
    }
    (out_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    def unpack(sample_batch):
        return [value.to(device, non_blocking=True) for value in sample_batch]

    def duration_view_loss(sample_batch, training: bool):
        (
            corrupted,
            mask,
            condition,
            target_duration,
            is_identity,
            duration_bin,
            event_uid,
        ) = unpack(sample_batch)
        if training and args.condition_noise_std > 0.0:
            condition = condition + torch.randn_like(condition) * float(args.condition_noise_std)
        result = model.predict_duration(corrupted, mask, condition, use_hard_duration=False)
        predicted = result["duration_soft_frames"]
        target_range = max(float(duration_edges[-1] - duration_edges[0]), 1.0)
        normalized_target = ((target_duration - float(duration_edges[0])) / target_range).clamp(0.0, 1.0)
        sample_weight = 0.75 + args.long_duration_weight * normalized_target.square()

        ordinal_target = (
            duration_bin[:, None] > torch.arange(num_bins - 1, device=device)[None]
        ).to(predicted.dtype)
        ordinal_per_element = F.binary_cross_entropy_with_logits(
            result["duration_ordinal_logits"],
            ordinal_target,
            reduction="none",
            pos_weight=ordinal_pos_weight_t,
        )
        ordinal_loss = ordinal_per_element.mean()

        true_residual_logit = result["duration_residual_logits"].gather(1, duration_bin[:, None]).squeeze(1)
        lower = model.duration_edges[:-1][duration_bin]
        upper = model.duration_edges[1:][duration_bin] - 1.0
        target_residual = ((target_duration - lower) / (upper - lower).clamp_min(1.0)).clamp(0.0, 1.0)
        residual_loss = weighted_mean(
            F.smooth_l1_loss(torch.sigmoid(true_residual_logit), target_residual, reduction="none"),
            sample_weight,
        )
        relative_loss = weighted_mean(
            torch.abs(predicted - target_duration) / target_duration.clamp_min(1.0),
            sample_weight,
        )
        log_loss = weighted_mean(
            F.smooth_l1_loss(
                torch.log(predicted.clamp_min(1.0)),
                torch.log(target_duration.clamp_min(1.0)),
                reduction="none",
            ),
            sample_weight,
        )
        direct_loss = weighted_mean(
            torch.abs(result["duration_direct_frames"] - target_duration) / target_duration.clamp_min(1.0),
            sample_weight,
        )
        underestimate_weight = 0.25 + 1.75 * normalized_target.square()
        underestimate_loss = weighted_mean(
            F.relu(target_duration - predicted) / target_duration.clamp_min(1.0),
            underestimate_weight,
        )
        edit_target = 1.0 - is_identity.clamp(0.0, 1.0)
        edit_loss = balanced_binary_loss(result["edit_logit"], edit_target)
        return {
            "result": result,
            "predicted": predicted,
            "target_duration": target_duration,
            "duration_bin": duration_bin,
            "event_uid": event_uid,
            "edit_target": edit_target,
            "ordinal": ordinal_loss,
            "residual": residual_loss,
            "relative": relative_loss,
            "log_duration": log_loss,
            "direct": direct_loss,
            "underestimate": underestimate_loss,
            "edit": edit_loss,
        }

    def duration_pair_loss(batch, training: bool):
        view_a, view_b = batch
        a = duration_view_loss(view_a, training)
        b = duration_view_loss(view_b, training)
        event_prediction = 0.5 * (a["predicted"] + b["predicted"])
        event_target = 0.5 * (a["target_duration"] + b["target_duration"])
        target_range = max(float(duration_edges[-1] - duration_edges[0]), 1.0)
        rank_loss = duration_rank_loss(event_prediction, event_target)
        duration_consistency = torch.abs(a["predicted"] - b["predicted"]).mean() / target_range
        distribution_consistency = symmetric_kl(
            a["result"]["duration_bin_probabilities"],
            b["result"]["duration_bin_probabilities"],
        )
        moment_loss = (
            torch.abs(event_prediction.mean() - event_target.mean())
            + torch.abs(event_prediction.std(unbiased=False) - event_target.std(unbiased=False))
        ) / target_range
        total = (
            args.lambda_ordinal * 0.5 * (a["ordinal"] + b["ordinal"])
            + args.lambda_residual * 0.5 * (a["residual"] + b["residual"])
            + args.lambda_relative * 0.5 * (a["relative"] + b["relative"])
            + args.lambda_log_duration * 0.5 * (a["log_duration"] + b["log_duration"])
            + args.lambda_direct * 0.5 * (a["direct"] + b["direct"])
            + args.lambda_underestimate * 0.5 * (a["underestimate"] + b["underestimate"])
            + args.lambda_duration_rank * rank_loss
            + args.lambda_pair_duration * duration_consistency
            + args.lambda_pair_distribution * distribution_consistency
            + args.lambda_moment * moment_loss
            + args.lambda_edit * 0.5 * (a["edit"] + b["edit"])
        )
        merged = {}
        for key in ("predicted", "target_duration", "duration_bin", "event_uid", "edit_target"):
            merged[key] = torch.cat([a[key], b[key]], dim=0)
        merged["pred_bin"] = torch.cat(
            [a["result"]["duration_bin_index"], b["result"]["duration_bin_index"]], dim=0
        )
        merged["edit_probability"] = torch.cat(
            [a["result"]["edit_probability"], b["result"]["edit_probability"]], dim=0
        )
        merged["bin_probability"] = torch.cat(
            [a["result"]["duration_bin_probabilities"], b["result"]["duration_bin_probabilities"]], dim=0
        )
        parts = {
            "loss": total,
            "ordinal": 0.5 * (a["ordinal"] + b["ordinal"]),
            "residual": 0.5 * (a["residual"] + b["residual"]),
            "relative": 0.5 * (a["relative"] + b["relative"]),
            "log_duration": 0.5 * (a["log_duration"] + b["log_duration"]),
            "direct": 0.5 * (a["direct"] + b["direct"]),
            "underestimate": 0.5 * (a["underestimate"] + b["underestimate"]),
            "rank": rank_loss,
            "pair_duration": duration_consistency,
            "pair_distribution": distribution_consistency,
            "moment": moment_loss,
            "edit": 0.5 * (a["edit"] + b["edit"]),
        }
        return total, parts, {
            "pred_duration": merged["predicted"].detach(),
            "target_duration": merged["target_duration"].detach(),
            "pred_bin": merged["pred_bin"].detach(),
            "target_bin": merged["duration_bin"].detach(),
            "edit_probability": merged["edit_probability"].detach(),
            "edit_target": merged["edit_target"].detach(),
            "event_uid": merged["event_uid"].detach(),
            "bin_probability": merged["bin_probability"].detach(),
        }

    def duration_single_loss(batch, training: bool):
        view = duration_view_loss(batch, training)
        total = (
            args.lambda_ordinal * view["ordinal"]
            + args.lambda_residual * view["residual"]
            + args.lambda_relative * view["relative"]
            + args.lambda_log_duration * view["log_duration"]
            + args.lambda_direct * view["direct"]
            + args.lambda_underestimate * view["underestimate"]
            + args.lambda_edit * view["edit"]
        )
        result = view["result"]
        return total, {"loss": total}, {
            "pred_duration": view["predicted"].detach(),
            "target_duration": view["target_duration"].detach(),
            "pred_bin": result["duration_bin_index"].detach(),
            "target_bin": view["duration_bin"].detach(),
            "edit_probability": result["edit_probability"].detach(),
            "edit_target": view["edit_target"].detach(),
            "event_uid": view["event_uid"].detach(),
            "bin_probability": result["duration_bin_probabilities"].detach(),
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
            event_uid,
        ) = unpack(batch)
        with torch.no_grad():
            duration_result = model.predict_duration(corrupted, mask, condition, use_hard_duration=False)
            predicted_duration = duration_result["duration_soft_frames"]
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
        activity_loss = F.smooth_l1_loss(torch.log1p(predicted_activity), torch.log1p(target_activity))
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
            "target_bin": duration_bin.detach(),
            "edit_probability": duration_result["edit_probability"].detach(),
            "edit_target": edit_target.detach(),
            "event_uid": event_uid.detach(),
            "bin_probability": duration_result["duration_bin_probabilities"].detach(),
            "tau_mae": torch.abs(tau - target_tau).mean(dim=1).detach(),
            "input_mse": input_mse.detach(),
            "pred_mse": pred_mse.detach(),
            "input_yaw_mae": input_yaw_mae.detach(),
            "pred_yaw_mae": pred_yaw_mae.detach(),
            "activity_ratio": (predicted_activity / target_activity.clamp_min(1e-6)).detach(),
            "identity_tau": identity_tau_per_sample[identity_rows].detach(),
            "identity_motion": identity_motion_per_sample[identity_rows].detach(),
        }

    def evaluate_duration(loader) -> Dict[str, float]:
        model.eval()
        totals: Dict[str, float] = {}
        count = 0
        collected: Dict[str, list[np.ndarray]] = defaultdict(list)
        with torch.no_grad():
            for batch in loader:
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    loss, parts, values = duration_single_loss(batch, training=False)
                batch_size = len(batch[0])
                count += batch_size
                totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * batch_size
                for key, value in values.items():
                    collected[key].append(value.detach().cpu().numpy())
        metrics = {key: value / max(count, 1) for key, value in totals.items()}
        merged = {key: np.concatenate(parts) for key, parts in collected.items()}
        metrics.update(
            duration_group_metrics(
                merged["pred_duration"],
                merged["target_duration"],
                merged["target_bin"].astype(int),
                merged["pred_bin"].astype(int),
                merged["edit_probability"],
                merged["edit_target"],
                num_bins,
                event_uid=merged["event_uid"],
                bin_probability=merged["bin_probability"],
            )
        )
        return metrics

    def run_epoch(loader, training: bool, epoch: int) -> Dict[str, float]:
        if args.stage == "duration" and not training:
            return evaluate_duration(loader)
        model.train(training)
        if args.stage == "timewarp":
            model.duration_encoder.eval()
            model.ordinal_head.eval()
            model.duration_residual_head.eval()
            model.direct_duration_head.eval()
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
        if args.stage == "duration" and training:
            event_dataset.set_epoch(epoch)

        loss_totals: Dict[str, float] = {}
        sample_count_epoch = 0
        collected: Dict[str, list[np.ndarray]] = defaultdict(list)
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
                        loss, parts, values = duration_pair_loss(batch, training=True)
                        batch_size = len(batch[0][0]) * 2
                    else:
                        loss, parts, values = timewarp_loss_batch(batch, tf_ratio)
                        batch_size = len(batch[0])
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, 0.7)
                    scaler.step(optimizer)
                    scaler.update()
                    if ema is not None:
                        ema.update(model)
            sample_count_epoch += batch_size
            for key, value in parts.items():
                loss_totals[key] = loss_totals.get(key, 0.0) + float(value.detach()) * batch_size
            for key, value in values.items():
                collected[key].append(value.detach().cpu().numpy())

        metrics = {key: value / max(sample_count_epoch, 1) for key, value in loss_totals.items()}
        merged = {key: np.concatenate(parts) if parts else np.empty((0,)) for key, parts in collected.items()}
        metrics.update(
            duration_group_metrics(
                merged["pred_duration"],
                merged["target_duration"],
                merged["target_bin"].astype(int),
                merged["pred_bin"].astype(int),
                merged["edit_probability"],
                merged["edit_target"],
                num_bins,
                event_uid=merged["event_uid"],
                bin_probability=merged["bin_probability"],
            )
        )
        if args.stage != "duration":
            metrics.update(
                {
                    "tau_mae": float(np.mean(merged["tau_mae"])),
                    "motion_mse_ratio": float(np.mean(merged["pred_mse"]) / max(np.mean(merged["input_mse"]), 1e-12)),
                    "yaw_mae_ratio": float(np.mean(merged["pred_yaw_mae"]) / max(np.mean(merged["input_yaw_mae"]), 1e-12)),
                    "activity_ratio": float(np.mean(merged["activity_ratio"])),
                    "identity_tau_mae": float(np.mean(merged["identity_tau"])) if len(merged["identity_tau"]) else 0.0,
                    "identity_motion_drift": float(np.mean(merged["identity_motion"])) if len(merged["identity_motion"]) else 0.0,
                    "teacher_forcing_ratio": float(tf_ratio),
                }
            )
        return metrics

    config = {
        "version": "v23_v2_4_ordinal_event_consistent",
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
        "slow_feature_span": int(args.slow_feature_span),
        "ordinal_blend": float(args.ordinal_blend),
        "split_seed": int(args.split_seed),
        "split_trials": int(args.split_trials),
        "seed": int(args.seed),
    }

    def selection_score(metrics: Dict[str, float]) -> float:
        duration_range = max(float(duration_edges[-1] - duration_edges[0]), 1.0)
        if args.stage == "duration":
            event_mae = float(metrics.get("event_duration_mae_frames", metrics["duration_mae_frames"]))
            event_long = float(metrics.get("event_duration_long_mae", metrics["duration_long_mae"]))
            event_medium = float(metrics.get("event_duration_medium_mae", metrics["duration_medium_mae"]))
            event_corr = float(metrics.get("event_duration_correlation", metrics["duration_correlation"]))
            ordinal_mae = float(metrics.get("event_duration_ordinal_mae_bins", metrics["duration_ordinal_mae_bins"]))
            within_one = float(metrics.get("event_duration_within_one_bin_accuracy", metrics["duration_within_one_bin_accuracy"]))
            return (
                event_mae / duration_range
                + 0.55 * event_long / duration_range
                + 0.30 * event_medium / duration_range
                + 0.22 * (1.0 - event_corr)
                + 0.10 * ordinal_mae / max(num_bins - 1, 1)
                + 0.10 * (1.0 - within_one)
                + 0.08 * (1.0 - float(metrics["edit_accuracy"]))
            )
        return (
            float(metrics["tau_mae"])
            + 0.45 * float(metrics["motion_mse_ratio"])
            + 0.20 * float(metrics["yaw_mae_ratio"])
            + 0.15 * max(0.0, 0.82 - float(metrics["activity_ratio"]))
            + 0.20 * float(metrics["identity_tau_mae"])
            + 0.08 * (1.0 - float(metrics["edit_accuracy"]))
        )

    best_score = float("inf")
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(train_loader, True, epoch)
        if ema is not None:
            ema.apply(model)
        val_metrics = run_epoch(val_loader, False, epoch)
        score = selection_score(val_metrics)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "selection_score": score,
                "lr": current_lr,
            }
        )
        improved = score < best_score - 1e-7
        if improved:
            best_score = score
            best_val = float(val_metrics["loss"])
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": best_val,
                    "selection_score": best_score,
                    "val_metrics": val_metrics,
                    "split": split,
                    "stage": args.stage,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            stale += 1
        if ema is not None:
            ema.restore(model)
        scheduler.step()

        if epoch == 1 or epoch % 5 == 0 or improved or epoch == args.epochs:
            if args.stage == "duration":
                print(
                    f"stage=duration epoch={epoch:04d} train={train_metrics['loss']:.6f} "
                    f"val={val_metrics['loss']:.6f} score={score:.6f} "
                    f"event_mae={val_metrics.get('event_duration_mae_frames', float('nan')):.3f} "
                    f"long={val_metrics.get('event_duration_long_mae', float('nan')):.3f} "
                    f"corr={val_metrics.get('event_duration_correlation', float('nan')):.3f} "
                    f"within1={val_metrics.get('event_duration_within_one_bin_accuracy', float('nan')):.3f} "
                    f"edit={val_metrics['edit_accuracy']:.3f} lr={current_lr:.2e} "
                    f"best={best_score:.6f}@{best_epoch}",
                    flush=True,
                )
            else:
                print(
                    f"stage=timewarp epoch={epoch:04d} train={train_metrics['loss']:.6f} "
                    f"val={val_metrics['loss']:.6f} score={score:.6f} "
                    f"tf={val_metrics['teacher_forcing_ratio']:.3f} "
                    f"dur={val_metrics['duration_mae_frames']:.3f} tau={val_metrics['tau_mae']:.5f} "
                    f"motion_ratio={val_metrics['motion_mse_ratio']:.3f} "
                    f"yaw_ratio={val_metrics['yaw_mae_ratio']:.3f} activity={val_metrics['activity_ratio']:.3f} "
                    f"id_tau={val_metrics['identity_tau_mae']:.5f} lr={current_lr:.2e} "
                    f"best={best_score:.6f}@{best_epoch}",
                    flush=True,
                )
        if args.patience > 0 and stale >= args.patience:
            print(f"early_stop stage={args.stage} epoch={epoch} best={best_score:.6f}@{best_epoch}", flush=True)
            break

    if ema is not None:
        ema.apply(model)
    final_metrics = run_epoch(val_loader, False, history[-1]["epoch"])
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": history[-1]["epoch"],
            "val_loss": float(final_metrics["loss"]),
            "selection_score": selection_score(final_metrics),
            "val_metrics": final_metrics,
            "split": split,
            "stage": args.stage,
        },
        checkpoint_dir / "final.pt",
    )
    if ema is not None:
        ema.restore(model)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "BEST_V23_CKPT.txt").write_text(str((checkpoint_dir / "best.pt").resolve()) + "\n", encoding="utf-8")
    print("best:", checkpoint_dir / "best.pt")
    print("best val loss:", best_val, "selection score:", best_score, "epoch:", best_epoch)


if __name__ == "__main__":
    main()
