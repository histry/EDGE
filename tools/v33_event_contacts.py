#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Event-level contact reconstruction and cache utilities for EDGE V33.

This module deliberately reconstructs contact labels on each complete indexed
motion event before any transition window is sampled.  Every downstream window
therefore slices the same event-level contact array, so an original event frame
has one deterministic label regardless of how many overlapping windows contain
it.

The cache stores variable-length labels in concatenated arrays plus offsets.  A
SHA1 motion fingerprint guards against accidentally pairing a cache with a
changed event database.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from tools.v21_common import CONTACT, load_motion
from tools.v29_motion_geometry import motion_to_joint_positions_torch

# Established EDGE evaluator mapping: left ankle, right ankle, left toe,
# right toe.  The contact channels follow the same order.
FOOT_JOINTS = (7, 8, 10, 11)
CONTACT_CHANNELS = 4
CACHE_VERSION = "v33_event_level_contact_cache"
LABEL_SOURCE = "event_level_kinematic_pseudo_contact_v33"


@dataclass
class ContactCalibration:
    fps: float = 30.0
    target_rates: Tuple[float, float, float, float] = (0.42, 0.42, 0.38, 0.38)
    height_weight: float = 1.0
    horizontal_speed_weight: float = 0.65
    vertical_speed_weight: float = 0.20
    transition_penalty: float = 1.40
    min_run: int = 2
    max_gap: int = 2
    probability_temperature: float = 0.20
    min_confidence: float = 0.05
    preserve_min_rate: float = 0.01
    preserve_max_rate: float = 0.90


@dataclass
class GlobalThresholds:
    height_scale: List[float]
    horizontal_speed_scale: List[float]
    vertical_speed_scale: List[float]
    score_threshold: List[float]


@dataclass
class EventFeatures:
    height_relative: np.ndarray
    horizontal_speed: np.ndarray
    vertical_speed: np.ndarray


def event_identifier(item: Mapping[str, Any], index: int) -> str:
    value = item.get("event_id")
    if value not in (None, ""):
        return str(value)
    return f"event_index:{index}"


def event_path(item: Mapping[str, Any]) -> str:
    return str(item.get("pkl", item.get("path", "")))


def motion_fingerprint(motion: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(motion, dtype=np.float32))
    digest = hashlib.sha1()
    digest.update(str(array.shape).encode("utf-8"))
    # Contact channels are excluded because the cache may be built precisely
    # to replace missing/invalid contacts.  Root and rotations define identity.
    payload = np.ascontiguousarray(array[:, 4:])
    digest.update(payload.view(np.uint8))
    return digest.hexdigest()


def _moving_average(values: np.ndarray, radius: int = 2) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if len(x) <= 1 or radius <= 0:
        return x.copy()
    kernel = np.asarray([1, 2, 3, 2, 1], dtype=np.float32)
    if radius != 2:
        kernel = np.ones((2 * radius + 1,), dtype=np.float32)
    kernel /= kernel.sum()
    padded = np.pad(x, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    result = np.empty_like(x)
    for channel in range(x.shape[1]):
        for coordinate in range(x.shape[2]):
            result[:, channel, coordinate] = np.convolve(
                padded[:, channel, coordinate], kernel, mode="valid"
            )
    return result


def foot_features(foot_positions: np.ndarray, fps: float) -> EventFeatures:
    feet = np.asarray(foot_positions, dtype=np.float32)
    if feet.ndim != 3 or feet.shape[1:] != (CONTACT_CHANNELS, 3):
        raise ValueError(f"Expected [T,4,3] feet, got {feet.shape}")
    if len(feet) == 0:
        empty = np.zeros((0, CONTACT_CHANNELS), dtype=np.float32)
        return EventFeatures(empty, empty, empty)

    smooth = _moving_average(feet, radius=2)
    velocity = np.zeros_like(smooth)
    if len(smooth) > 1:
        velocity[0] = (smooth[1] - smooth[0]) * float(fps)
        velocity[-1] = (smooth[-1] - smooth[-2]) * float(fps)
    if len(smooth) > 2:
        velocity[1:-1] = (smooth[2:] - smooth[:-2]) * (0.5 * float(fps))

    height = smooth[..., 1]
    ground = np.quantile(height, 0.05, axis=0)
    relative = np.maximum(height - ground[None], 0.0)
    horizontal = np.linalg.norm(velocity[..., (0, 2)], axis=-1)
    vertical = np.abs(velocity[..., 1])
    return EventFeatures(
        height_relative=relative.astype(np.float32),
        horizontal_speed=horizontal.astype(np.float32),
        vertical_speed=vertical.astype(np.float32),
    )


def _robust_scale(values: np.ndarray, quantile: float, floor: float) -> np.ndarray:
    result = np.quantile(values, quantile, axis=0).astype(np.float32)
    return np.maximum(result, float(floor)).astype(np.float32)


def calibrate_global_thresholds(
    feature_rows: Sequence[EventFeatures],
    config: ContactCalibration,
) -> GlobalThresholds:
    valid = [row for row in feature_rows if len(row.height_relative)]
    if not valid:
        raise RuntimeError("No valid event foot features for contact calibration")
    height = np.concatenate([row.height_relative for row in valid], axis=0)
    horizontal = np.concatenate([row.horizontal_speed for row in valid], axis=0)
    vertical = np.concatenate([row.vertical_speed for row in valid], axis=0)

    height_scale = _robust_scale(height, 0.40, 0.018)
    horizontal_scale = _robust_scale(horizontal, 0.40, 0.035)
    vertical_scale = _robust_scale(vertical, 0.40, 0.025)

    score = (
        float(config.height_weight) * (height / height_scale[None]) ** 2
        + float(config.horizontal_speed_weight)
        * (horizontal / horizontal_scale[None]) ** 2
        + float(config.vertical_speed_weight)
        * (vertical / vertical_scale[None]) ** 2
    )
    target = np.asarray(config.target_rates, dtype=np.float32)
    if target.shape != (CONTACT_CHANNELS,):
        raise ValueError("target_rates must contain four values")
    if np.any(target <= 0.0) or np.any(target >= 1.0):
        raise ValueError("target_rates must lie strictly in (0,1)")
    thresholds = np.asarray(
        [np.quantile(score[:, channel], float(target[channel]))
         for channel in range(CONTACT_CHANNELS)],
        dtype=np.float32,
    )
    thresholds = np.maximum(thresholds, 1e-4)
    return GlobalThresholds(
        height_scale=height_scale.astype(float).tolist(),
        horizontal_speed_scale=horizontal_scale.astype(float).tolist(),
        vertical_speed_scale=vertical_scale.astype(float).tolist(),
        score_threshold=thresholds.astype(float).tolist(),
    )


def contact_score(
    features: EventFeatures,
    thresholds: GlobalThresholds,
    config: ContactCalibration,
) -> np.ndarray:
    height_scale = np.asarray(thresholds.height_scale, dtype=np.float32)
    horizontal_scale = np.asarray(
        thresholds.horizontal_speed_scale, dtype=np.float32
    )
    vertical_scale = np.asarray(
        thresholds.vertical_speed_scale, dtype=np.float32
    )
    return (
        float(config.height_weight)
        * (features.height_relative / height_scale[None]) ** 2
        + float(config.horizontal_speed_weight)
        * (features.horizontal_speed / horizontal_scale[None]) ** 2
        + float(config.vertical_speed_weight)
        * (features.vertical_speed / vertical_scale[None]) ** 2
    ).astype(np.float32)


def score_probability(
    score: np.ndarray,
    thresholds: GlobalThresholds,
    config: ContactCalibration,
) -> np.ndarray:
    boundary = np.asarray(thresholds.score_threshold, dtype=np.float32)
    scale = np.maximum(
        np.abs(boundary) * float(config.probability_temperature), 1e-3
    )
    logits = (boundary[None] - score) / scale[None]
    logits = np.clip(logits, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def _viterbi_binary(probability: np.ndarray, penalty: float) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    length = len(p)
    if length == 0:
        return np.zeros((0,), dtype=np.bool_)
    emission = np.stack([-np.log(1.0 - p), -np.log(p)], axis=1)
    cost = np.full((length, 2), np.inf, dtype=np.float64)
    back = np.zeros((length, 2), dtype=np.int8)
    cost[0] = emission[0]
    for time in range(1, length):
        for state in (0, 1):
            stay = cost[time - 1, state]
            switch = cost[time - 1, 1 - state] + float(penalty)
            if stay <= switch:
                cost[time, state] = stay + emission[time, state]
                back[time, state] = state
            else:
                cost[time, state] = switch + emission[time, state]
                back[time, state] = 1 - state
    output = np.zeros((length,), dtype=np.bool_)
    state = int(np.argmin(cost[-1]))
    output[-1] = bool(state)
    for time in range(length - 1, 0, -1):
        state = int(back[time, state])
        output[time - 1] = bool(state)
    return output


def _runs(binary: np.ndarray) -> Iterable[Tuple[int, int, bool]]:
    values = np.asarray(binary, dtype=np.bool_)
    if len(values) == 0:
        return
    start = 0
    state = bool(values[0])
    for index in range(1, len(values) + 1):
        if index == len(values) or bool(values[index]) != state:
            yield start, index, state
            if index < len(values):
                start = index
                state = bool(values[index])


def clean_contact_runs(
    binary: np.ndarray,
    min_run: int,
    max_gap: int,
) -> np.ndarray:
    result = np.asarray(binary, dtype=np.bool_).copy()
    if len(result) == 0:
        return result
    for start, end, state in list(_runs(result)):
        if (
            not state
            and end - start <= int(max_gap)
            and start > 0
            and end < len(result)
            and result[start - 1]
            and result[end]
        ):
            result[start:end] = True
    for start, end, state in list(_runs(result)):
        if state and end - start < int(min_run):
            result[start:end] = False
    return result


def infer_from_features(
    features: EventFeatures,
    thresholds: GlobalThresholds,
    config: ContactCalibration,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = contact_score(features, thresholds, config)
    probability = score_probability(score, thresholds, config)
    hard = np.zeros_like(probability, dtype=np.float32)
    for channel in range(CONTACT_CHANNELS):
        sequence = _viterbi_binary(
            probability[:, channel], float(config.transition_penalty)
        )
        sequence = clean_contact_runs(
            sequence, int(config.min_run), int(config.max_gap)
        )
        hard[:, channel] = sequence.astype(np.float32)
    confidence = np.clip(
        np.abs(probability - 0.5) * 2.0,
        float(config.min_confidence),
        1.0,
    ).astype(np.float32)
    return hard, confidence, probability


def existing_contacts_are_plausible(
    motion: np.ndarray,
    config: ContactCalibration,
) -> bool:
    contact = np.asarray(motion[:, CONTACT], dtype=np.float32)
    if len(contact) == 0 or not np.isfinite(contact).all():
        return False
    rate = float(np.clip(contact, 0.0, 1.0).mean())
    return float(config.preserve_min_rate) <= rate <= float(config.preserve_max_rate)


def parse_target_rates(value: str | Sequence[float]) -> Tuple[float, float, float, float]:
    if isinstance(value, str):
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    else:
        values = [float(part) for part in value]
    if len(values) != 4:
        raise ValueError("Expected four comma-separated target contact rates")
    return tuple(values)  # type: ignore[return-value]


class EventContactCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        data = np.load(self.path, allow_pickle=True)
        required = {
            "event_id", "event_path", "event_length", "event_valid",
            "motion_fingerprint", "offsets", "contact_hard",
            "contact_confidence", "meta",
        }
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(f"Contact cache missing arrays: {sorted(missing)}")
        self.event_id = np.asarray(data["event_id"], dtype=object)
        self.event_path = np.asarray(data["event_path"], dtype=object)
        self.event_length = np.asarray(data["event_length"], dtype=np.int32)
        self.event_valid = np.asarray(data["event_valid"], dtype=np.bool_)
        self.fingerprint = np.asarray(data["motion_fingerprint"], dtype=object)
        self.offsets = np.asarray(data["offsets"], dtype=np.int64)
        self.hard = np.asarray(data["contact_hard"], dtype=np.float32)
        self.confidence = np.asarray(data["contact_confidence"], dtype=np.float32)
        self.meta = json.loads(str(data["meta"].item()))
        if self.meta.get("version") != CACHE_VERSION:
            raise RuntimeError(
                f"Unsupported contact cache version: {self.meta.get('version')}"
            )
        thresholds = self.meta.get("thresholds", {})
        calibration = self.meta.get("calibration", {})
        self.thresholds = GlobalThresholds(**thresholds)
        self.calibration = ContactCalibration(**calibration)

    def __len__(self) -> int:
        return len(self.event_id)

    def get(
        self,
        index: int,
        item: Mapping[str, Any],
        motion: np.ndarray,
        strict: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if not bool(self.event_valid[index]):
            raise RuntimeError(f"No contact cache entry for event index {index}")
        expected_id = event_identifier(item, index)
        expected_path = event_path(item)
        problems: List[str] = []
        if str(self.event_id[index]) != expected_id:
            problems.append(
                f"event_id cache={self.event_id[index]!r} index={expected_id!r}"
            )
        if str(self.event_path[index]) != expected_path:
            problems.append(
                f"event_path cache={self.event_path[index]!r} index={expected_path!r}"
            )
        if int(self.event_length[index]) != len(motion):
            problems.append(
                f"length cache={self.event_length[index]} motion={len(motion)}"
            )
        current_fingerprint = motion_fingerprint(motion)
        if str(self.fingerprint[index]) != current_fingerprint:
            problems.append("motion fingerprint mismatch")
        if problems and strict:
            raise RuntimeError(
                f"Contact cache mismatch at event {index}: " + "; ".join(problems)
            )
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        hard = self.hard[start:end]
        confidence = self.confidence[start:end]
        if len(hard) != len(motion):
            raise RuntimeError(
                f"Contact cache slice length {len(hard)} != motion length {len(motion)}"
            )
        origin = f"event:{index}:{expected_id}"
        return hard.copy(), confidence.copy(), origin

    def infer_sequence(
        self, motion: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            tensor = torch.from_numpy(
                np.asarray(motion, dtype=np.float32)
            ).unsqueeze(0)
            positions = motion_to_joint_positions_torch(tensor)
            feet = positions[0, :, FOOT_JOINTS].cpu().numpy()
        features = foot_features(feet, float(self.calibration.fps))
        hard, confidence, _ = infer_from_features(
            features, self.thresholds, self.calibration
        )
        return hard, confidence


def write_cache(
    output: Path,
    event_ids: Sequence[str],
    event_paths: Sequence[str],
    lengths: Sequence[int],
    valid: Sequence[bool],
    fingerprints: Sequence[str],
    contacts: Sequence[np.ndarray],
    confidences: Sequence[np.ndarray],
    metadata: Mapping[str, Any],
) -> None:
    offsets = [0]
    hard_rows: List[np.ndarray] = []
    confidence_rows: List[np.ndarray] = []
    for is_valid, length, hard, confidence in zip(
        valid, lengths, contacts, confidences
    ):
        if is_valid:
            hard_array = np.asarray(hard, dtype=np.float32)
            confidence_array = np.asarray(confidence, dtype=np.float32)
            if hard_array.shape != (int(length), CONTACT_CHANNELS):
                raise ValueError(f"Invalid contact shape {hard_array.shape}")
            if confidence_array.shape != hard_array.shape:
                raise ValueError("Confidence/contact shape mismatch")
            hard_rows.append(hard_array)
            confidence_rows.append(confidence_array)
            offsets.append(offsets[-1] + int(length))
        else:
            offsets.append(offsets[-1])
    hard_concat = (
        np.concatenate(hard_rows, axis=0)
        if hard_rows else np.zeros((0, CONTACT_CHANNELS), np.float32)
    )
    confidence_concat = (
        np.concatenate(confidence_rows, axis=0)
        if confidence_rows else np.zeros((0, CONTACT_CHANNELS), np.float32)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        event_id=np.asarray(event_ids, dtype=object),
        event_path=np.asarray(event_paths, dtype=object),
        event_length=np.asarray(lengths, dtype=np.int32),
        event_valid=np.asarray(valid, dtype=np.bool_),
        motion_fingerprint=np.asarray(fingerprints, dtype=object),
        offsets=np.asarray(offsets, dtype=np.int64),
        contact_hard=hard_concat.astype(np.float32),
        contact_confidence=confidence_concat.astype(np.float32),
        meta=np.asarray(json.dumps(dict(metadata), ensure_ascii=False), dtype=object),
    )
