#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build adjacent-boundary supervision for V21 DPN and transition refiner.

Pairs come from neighbouring dynamic events in the same physical source motion.
The target transition is taken from the original continuous 151D sequence, while
the input rough transition is endpoint interpolation. This gives the refiner a
real reconstruction objective instead of learning synthetic cross-fades.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from model.v21_transition import TRANSITION_LENGTHS
from tools.v21_common import CONTACT, ROOT_X, ROOT_Y, ROOT_Z, ROT, load_json_items, load_motion, make_linear_transition, motion_descriptor_raw, robust_scale


def proxy_transition_length(prev: np.ndarray, nxt: np.ndarray) -> int:
    pose_jump = float(np.linalg.norm(prev[-1, ROT] - nxt[0, ROT]) / np.sqrt(144.0))
    pv = prev[-1, ROT] - prev[-2, ROT] if len(prev) > 1 else np.zeros((144,), dtype=np.float32)
    nv = nxt[1, ROT] - nxt[0, ROT] if len(nxt) > 1 else np.zeros((144,), dtype=np.float32)
    vel_jump = float(np.linalg.norm(pv - nv) / np.sqrt(144.0))
    raw = 4.0 + 18.0 * np.clip(pose_jump, 0.0, 0.5) + 10.0 * np.clip(vel_jump, 0.0, 0.4)
    return int(min(TRANSITION_LENGTHS, key=lambda k: abs(k - raw)))


def transition_features(prev: np.ndarray, nxt: np.ndarray, music_query: np.ndarray) -> np.ndarray:
    pose_diff = prev[-1] - nxt[0]
    pv = prev[-1] - prev[-2] if len(prev) > 1 else np.zeros((151,), dtype=np.float32)
    nv = nxt[1] - nxt[0] if len(nxt) > 1 else np.zeros((151,), dtype=np.float32)
    feat = np.asarray(
        [
            np.linalg.norm(pose_diff[ROT]) / np.sqrt(144.0),
            np.linalg.norm(pv[ROT] - nv[ROT]) / np.sqrt(144.0),
            abs(float(pose_diff[ROOT_Y])),
            float(np.abs(pose_diff[CONTACT]).mean()),
            float(np.linalg.norm(prev[-1, ROT] - prev[0, ROT]) / np.sqrt(144.0)),
            float(np.linalg.norm(nxt[-1, ROT] - nxt[0, ROT]) / np.sqrt(144.0)),
            float(len(prev) / 72.0),
            float(len(nxt) / 72.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate([feat, np.asarray(music_query, dtype=np.float32).reshape(12)], axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_pairs", type=int, default=30000)
    ap.add_argument("--max_gap", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260605)
    args = ap.parse_args()

    _, items = load_json_items(args.event_db)
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        source = str(item.get("source_file", ""))
        if source and Path(source).is_file():
            by_source[source].append(item)
    rng = np.random.default_rng(args.seed)

    rough_list: List[np.ndarray] = []
    target_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    start_list: List[np.ndarray] = []
    end_list: List[np.ndarray] = []
    music_list: List[np.ndarray] = []
    dpn_feat_list: List[np.ndarray] = []
    label_list: List[int] = []
    max_k = max(TRANSITION_LENGTHS)

    for source, group in sorted(by_source.items()):
        group.sort(key=lambda x: int(x.get("source_start", 0)))
        source_motion = load_motion(source)
        for a, b in zip(group[:-1], group[1:]):
            a_end = int(a.get("source_end", 0))
            b_start = int(b.get("source_start", 0))
            if b_start < a_end or b_start - a_end > args.max_gap:
                continue
            pa = Path(str(a.get("pkl", a.get("path", ""))))
            pb = Path(str(b.get("pkl", b.get("path", ""))))
            if not pa.is_file() or not pb.is_file():
                continue
            prev = load_motion(pa)
            nxt = load_motion(pb)
            k = proxy_transition_length(prev, nxt)
            half = k // 2
            center = int(round((a_end + b_start) / 2.0))
            lo = max(0, center - half)
            hi = min(len(source_motion), lo + k)
            lo = max(0, hi - k)
            target = source_motion[lo:hi].astype(np.float32)
            if len(target) < 4:
                continue
            k = len(target)
            start_pose = target[0].copy()
            end_pose = target[-1].copy()
            rough = make_linear_transition(target[:1], target[-1:], k)

            prev_desc = motion_descriptor_raw(prev)
            next_desc = motion_descriptor_raw(nxt)
            # Weak music/event condition: average normalized dynamics of adjacent phrases.
            music_query = 0.5 * (prev_desc + next_desc)
            music_query[11] = np.clip(k / 16.0, 0.0, 1.0)

            rough_pad = np.repeat(rough[-1:], max_k, axis=0)
            target_pad = np.repeat(target[-1:], max_k, axis=0)
            rough_pad[:k] = rough
            target_pad[:k] = target
            mask = np.zeros((max_k,), dtype=np.float32)
            mask[:k] = 1.0

            rough_list.append(rough_pad)
            target_list.append(target_pad)
            mask_list.append(mask)
            start_list.append(start_pose)
            end_list.append(end_pose)
            music_list.append(music_query.astype(np.float32))
            dpn_feat_list.append(transition_features(prev, nxt, music_query))
            nearest = int(np.argmin(np.abs(np.asarray(TRANSITION_LENGTHS) - k)))
            label_list.append(nearest)
            if len(rough_list) >= args.max_pairs:
                break
        if len(rough_list) >= args.max_pairs:
            break

    if not rough_list:
        raise RuntimeError("No adjacent transition pairs were built")

    dpn_raw = np.stack(dpn_feat_list).astype(np.float32)
    dpn_scaled, dpn_lo, dpn_hi = robust_scale(dpn_raw)
    order = rng.permutation(len(rough_list))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        rough=np.stack(rough_list)[order].astype(np.float32),
        target=np.stack(target_list)[order].astype(np.float32),
        mask=np.stack(mask_list)[order].astype(np.float32),
        start=np.stack(start_list)[order].astype(np.float32),
        end=np.stack(end_list)[order].astype(np.float32),
        music=np.stack(music_list)[order].astype(np.float32),
        dpn_features=dpn_scaled[order].astype(np.float32),
        dpn_features_raw=dpn_raw[order].astype(np.float32),
        dpn_lo=dpn_lo.astype(np.float32),
        dpn_hi=dpn_hi.astype(np.float32),
        dpn_label=np.asarray(label_list, dtype=np.int64)[order],
        transition_lengths=np.asarray(TRANSITION_LENGTHS, dtype=np.int64),
    )
    print("saved:", out)
    print("pairs:", len(rough_list))
    print("label_counts:", {int(i): int(np.sum(np.asarray(label_list) == i)) for i in range(len(TRANSITION_LENGTHS))})


if __name__ == "__main__":
    main()
