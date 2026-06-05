#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build dataset for endpoint-conditioned transition refiner."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

from tools.v20_motion_utils import MOTION_DIM, ROOT_X, ROOT_Z, ROT, linear_transition, load_json, load_motion_any, localize_root, velocity_at, write_json


def load_event(item: Dict) -> Dict:
    with open(item["pkl"], "rb") as f:
        obj = pickle.load(f)
    return {**item, **obj}


def music_event_zero(k: int, dim: int = 12) -> np.ndarray:
    return np.zeros((k, dim), dtype=np.float32)


def pad_to(x: np.ndarray, max_len: int, value: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    mask = np.zeros((max_len, 1), dtype=np.float32)
    out = np.zeros((max_len, x.shape[-1]), dtype=np.float32) + value
    L = min(len(x), max_len)
    if L:
        out[:L] = x[:L]
        mask[:L] = 1.0
        if L < max_len:
            out[L:] = out[L - 1:L]
    return out, mask


def target_transition_from_source(prev: Dict, nxt: Dict, max_transition_len: int) -> np.ndarray:
    # Use source frames around the boundary when possible.
    try:
        if prev.get("source_file") == nxt.get("source_file"):
            raw, _ = load_motion_any(prev["source_file"])
            raw = localize_root(raw)
            boundary = int(prev.get("source_end", 0))
            k = min(max_transition_len, max(4, int(nxt.get("source_start", boundary)) - boundary + 8))
            half = k // 2
            lo = max(0, boundary - half)
            hi = min(len(raw), lo + k)
            lo = max(0, hi - k)
            seg = raw[lo:hi]
            if len(seg) >= 4:
                return seg.astype(np.float32)
    except Exception:
        pass
    k = min(max_transition_len, 10)
    return linear_transition(prev["motion"][-1], nxt["motion"][0], k)


def corrupt(rough: np.ndarray, mask: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = rough.copy()
    noise = rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)
    ch = np.zeros((1, MOTION_DIM), dtype=np.float32); ch[:, 5] = 0.25; ch[:, ROT] = 1.0
    out += noise * mask * ch
    out[:, ROOT_X] = rough[:, ROOT_X]
    out[:, ROOT_Z] = rough[:, ROOT_Z]
    return out.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_db", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=30000)
    ap.add_argument("--max_transition_len", type=int, default=20)
    ap.add_argument("--noise_std", type=float, default=0.012)
    args = ap.parse_args()

    db = load_json(args.event_db)
    events = [load_event(x) for x in db.get("items", [])]
    by_source: Dict[int, List[Dict]] = {}
    for e in events:
        by_source.setdefault(int(e.get("source_id", -1)), []).append(e)
    for sid in by_source:
        by_source[sid] = sorted(by_source[sid], key=lambda x: int(x.get("source_start", 0)))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []; count = 0
    for sid, evs in by_source.items():
        for prev, nxt in zip(evs[:-1], evs[1:]):
            if count >= args.max_samples:
                break
            target = target_transition_from_source(prev, nxt, args.max_transition_len)
            rough = linear_transition(prev["motion"][-1], nxt["motion"][0], len(target))
            target_pad, mask = pad_to(target, args.max_transition_len)
            rough_pad, _ = pad_to(rough, args.max_transition_len)
            noisy = corrupt(rough_pad, mask, args.noise_std, seed=200000 + count)
            music = music_event_zero(args.max_transition_len)
            out = out_dir / f"transition_{count:06d}.npz"
            np.savez_compressed(
                out,
                rough=noisy.astype(np.float32),
                rough_clean=rough_pad.astype(np.float32),
                target=target_pad.astype(np.float32),
                valid_mask=mask.astype(np.float32),
                exit_pose=np.asarray(prev["motion"][-1], dtype=np.float32),
                entry_pose=np.asarray(nxt["motion"][0], dtype=np.float32),
                music_event=music.astype(np.float32),
                prev_event=prev["event_id"],
                next_event=nxt["event_id"],
            )
            rows.append({"path": str(out), "prev": prev["event_id"], "next": nxt["event_id"], "length": int(mask.sum())})
            count += 1
    write_json({"num_samples": count, "max_transition_len": args.max_transition_len, "samples": rows}, out_dir / "index.json")
    print(f"saved dataset: {out_dir}")
    print(f"samples: {count}")


if __name__ == "__main__":
    main()
