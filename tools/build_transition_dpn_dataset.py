#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build dataset for TransitionDurationPredictor."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

from tools.v20_motion_utils import CONTACT, ROT, load_json, pose_distance, velocity_at, write_json
from model.transition_duration_predictor import TRANSITION_BINS


def load_event(item: Dict) -> Dict:
    with open(item["pkl"], "rb") as f:
        obj = pickle.load(f)
    return {**item, **obj}


def desc_vec(e: Dict) -> np.ndarray:
    d = e.get("descriptor", {})
    keys = ["upper_activity", "torso_activity", "lower_activity", "full_activity", "style_tension", "smoothness", "contact_switch", "root_y_range", "safety_score", "quality_score"]
    return np.asarray([float(d.get(k, e.get(k, 0.0))) for k in keys], dtype=np.float32)


def make_feature(prev: Dict, nxt: Dict, music_dim: int = 12) -> np.ndarray:
    a = np.asarray(prev["exit_pose"], dtype=np.float32)
    b = np.asarray(nxt["entry_pose"], dtype=np.float32)
    va = np.asarray(prev.get("exit_velocity", velocity_at(prev["motion"], at_exit=True)), dtype=np.float32)
    vb = np.asarray(nxt.get("entry_velocity", velocity_at(nxt["motion"], at_exit=False)), dtype=np.float32)
    # Keep feature compact: endpoint diffs + velocities + descriptors + simple costs.
    rot_diff = (a[ROT] - b[ROT]).astype(np.float32)
    root_contact = np.concatenate([[a[4] - b[4], a[5] - b[5], a[6] - b[6]], a[CONTACT] - b[CONTACT]], axis=0).astype(np.float32)
    f = np.concatenate([
        rot_diff,
        va[ROT],
        vb[ROT],
        root_contact,
        desc_vec(prev),
        desc_vec(nxt),
        np.asarray([pose_distance(a, b), float(prev.get("length", 0)), float(nxt.get("length", 0))], dtype=np.float32),
        np.zeros((music_dim,), dtype=np.float32),
    ], axis=0)
    return np.nan_to_num(f.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def label_transition_len(prev: Dict, nxt: Dict) -> int:
    d = pose_distance(np.asarray(prev["exit_pose"]), np.asarray(nxt["entry_pose"]))
    base = 8
    if d > 1.4:
        base = 16
    elif d > 0.9:
        base = 12
    elif d < 0.45:
        base = 6
    # Support changes get slightly longer transition.
    if prev.get("event_type") == "support_shift" or nxt.get("event_type") == "support_shift":
        base += 2
    base = int(max(min(TRANSITION_BINS), min(max(TRANSITION_BINS), round(base / 2) * 2)))
    return min(TRANSITION_BINS, key=lambda x: abs(x - base))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_db", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_pairs", type=int, default=50000)
    ap.add_argument("--same_source_only", type=int, default=1)
    args = ap.parse_args()

    db = load_json(args.event_db)
    items = db.get("items", [])
    events = [load_event(x) for x in items]
    by_source: Dict[int, List[Dict]] = {}
    for e in events:
        by_source.setdefault(int(e.get("source_id", -1)), []).append(e)
    for k in by_source:
        by_source[k] = sorted(by_source[k], key=lambda x: int(x.get("source_start", 0)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    count = 0
    # Prefer adjacent same-source pairs for natural duration proxy.
    for sid, evs in by_source.items():
        for a, b in zip(evs[:-1], evs[1:]):
            if count >= args.max_pairs:
                break
            x = make_feature(a, b)
            y_len = label_transition_len(a, b)
            y_cls = TRANSITION_BINS.index(y_len)
            out = out_dir / f"pair_{count:06d}.npz"
            np.savez_compressed(out, x=x, y_cls=np.asarray(y_cls, dtype=np.int64), y_len=np.asarray(y_len, dtype=np.float32), prev_event=a["event_id"], next_event=b["event_id"])
            rows.append({"path": str(out), "prev": a["event_id"], "next": b["event_id"], "y_len": y_len, "y_cls": y_cls})
            count += 1
    write_json({"num_samples": count, "transition_bins": TRANSITION_BINS, "samples": rows}, out_dir / "index.json")
    print(f"saved dataset: {out_dir}")
    print(f"samples: {count}")


if __name__ == "__main__":
    main()
