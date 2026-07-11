#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build source-disjoint splits/folds from a V46 Event-RAG DB.

Why
---
Chang-E BVH files are cut into many overlapping/sequential events. Random event
splits leak source motion into validation/test.  This script groups by
source_group/source_uid/source_bvh and produces source-disjoint folds plus audit
statistics for dance_key/gender/support/label coverage.

Outputs
-------
  source_disjoint_splits.json
  source_disjoint_splits.csv
  source_disjoint_split_audit.json

Example:
  python tools/make_v46_47_source_disjoint_splits.py \
    --db output/v46_47_db \
    --out_dir output/v46_47_db/splits \
    --folds 3 --group_key source_group
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def j(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [j(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def load_db_arrays(db_dir: Path) -> Dict[str, np.ndarray]:
    p = db_dir / "events.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=True)
    n = len(z["paths"])
    def get(name: str, default: str = "unknown") -> np.ndarray:
        if name in z.files:
            return np.asarray(z[name], dtype=object)
        return np.array([default] * n, dtype=object)
    return {
        "paths": get("paths", ""),
        "source_group": get("source_groups", "unknown"),
        "source_uid": get("source_uids", "unknown"),
        "source_bvh": get("source_bvh", "unknown"),
        "label": get("labels", "unknown"),
        "parent_label": get("parent_labels", "unknown"),
        "dance_key": get("dance_keys", "unknown"),
        "gender": get("genders", "unknown"),
        "support_label": get("support_labels", "unknown"),
        "music_alignment_label": get("music_alignment_labels", "unknown"),
        "event_family": get("event_families", "unknown"),
        "duration": np.asarray(z["durations"], dtype=np.float32) if "durations" in z.files else np.ones(n, dtype=np.float32),
    }


def group_events(arr: Dict[str, np.ndarray], group_key: str) -> Dict[str, List[int]]:
    if group_key not in arr:
        raise ValueError(f"Unknown group_key={group_key}; choices={list(arr)}")
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, g in enumerate(arr[group_key]):
        groups[str(g)].append(i)
    return dict(groups)


def group_profile(arr: Dict[str, np.ndarray], indices: List[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"num_events": len(indices), "duration_total_s": float(np.sum(arr["duration"][indices])) if indices else 0.0}
    for key in ["label", "dance_key", "gender", "support_label", "music_alignment_label", "event_family"]:
        out[key + "_counts"] = dict(Counter(str(arr[key][i]) for i in indices))
    return out


def make_folds(arr: Dict[str, np.ndarray], groups: Dict[str, List[int]], folds: int) -> List[Dict[str, Any]]:
    # Greedy balanced by number of events and broad dance_key coverage.
    group_items = []
    for g, idxs in groups.items():
        prof = group_profile(arr, idxs)
        dance_keys = set(str(arr["dance_key"][i]) for i in idxs)
        group_items.append((g, idxs, prof, len(idxs), dance_keys))
    group_items.sort(key=lambda x: (-x[3], x[0]))
    bins = [{"groups": [], "indices": [], "dance_keys": Counter()} for _ in range(folds)]
    for g, idxs, prof, size, dks in group_items:
        # Prefer fold with smallest events, then one where this dance_key is rare.
        best = None
        best_score = None
        for k, b in enumerate(bins):
            dk_overlap = sum(b["dance_keys"].get(x, 0) for x in dks)
            score = (len(b["indices"]), dk_overlap, k)
            if best is None or score < best_score:
                best = k
                best_score = score
        b = bins[best]
        b["groups"].append(g)
        b["indices"].extend(idxs)
        for x in dks:
            b["dance_keys"][x] += 1
    out = []
    for k, b in enumerate(bins):
        idxs = sorted(b["indices"])
        out.append({"fold": k, "groups": sorted(b["groups"]), "event_indices": idxs, "profile": group_profile(arr, idxs)})
    return out


def split_from_folds(folds: List[Dict[str, Any]], test_fold: int = 0, val_fold: int = 1) -> Dict[str, Any]:
    test = folds[test_fold % len(folds)]
    val = folds[val_fold % len(folds)] if len(folds) > 1 else {"groups": [], "event_indices": []}
    train_idxs: List[int] = []
    train_groups: List[str] = []
    for f in folds:
        if f["fold"] not in {test["fold"], val.get("fold", -1)}:
            train_idxs.extend(f["event_indices"])
            train_groups.extend(f["groups"])
    return {
        "train": {"groups": sorted(train_groups), "event_indices": sorted(train_idxs)},
        "val": {"groups": val.get("groups", []), "event_indices": val.get("event_indices", [])},
        "test": {"groups": test.get("groups", []), "event_indices": test.get("event_indices", [])},
    }


def assert_disjoint(split: Dict[str, Any]) -> Dict[str, Any]:
    names = ["train", "val", "test"]
    overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ga = set(split[a]["groups"])
            gb = set(split[b]["groups"])
            ia = set(split[a]["event_indices"])
            ib = set(split[b]["event_indices"])
            overlaps[f"{a}_{b}_group_overlap"] = sorted(ga & gb)
            overlaps[f"{a}_{b}_event_overlap"] = sorted(ia & ib)
    overlaps["ok"] = all(len(v) == 0 for k, v in overlaps.items() if k != "ok")
    return overlaps


def write_csv(arr: Dict[str, np.ndarray], folds: List[Dict[str, Any]], path: Path) -> None:
    fold_of = {}
    group_of = {}
    for f in folds:
        for idx in f["event_indices"]:
            fold_of[idx] = f["fold"]
            group_of[idx] = next((g for g in f["groups"] if str(arr["source_group"][idx]) == g or str(arr["source_uid"][idx]) == g or str(arr["source_bvh"][idx]) == g), "")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["event_id", "fold", "path", "source_group", "source_uid", "source_bvh", "label", "dance_key", "gender", "support_label", "music_alignment_label", "duration"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for i in range(len(arr["paths"])):
            w.writerow({
                "event_id": i,
                "fold": fold_of.get(i, -1),
                "path": str(arr["paths"][i]),
                "source_group": str(arr["source_group"][i]),
                "source_uid": str(arr["source_uid"][i]),
                "source_bvh": str(arr["source_bvh"][i]),
                "label": str(arr["label"][i]),
                "dance_key": str(arr["dance_key"][i]),
                "gender": str(arr["gender"][i]),
                "support_label": str(arr["support_label"][i]),
                "music_alignment_label": str(arr["music_alignment_label"][i]),
                "duration": float(arr["duration"][i]),
            })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--group_key", default="source_group", choices=["source_group", "source_uid", "source_bvh"])
    ap.add_argument("--test_fold", type=int, default=0)
    ap.add_argument("--val_fold", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arr = load_db_arrays(Path(args.db))
    groups = group_events(arr, args.group_key)
    folds = make_folds(arr, groups, max(2, int(args.folds)))
    split = split_from_folds(folds, args.test_fold, args.val_fold)
    overlap = assert_disjoint(split)
    report = {
        "version": "v46_47_source_disjoint_splits",
        "db": str(args.db),
        "group_key": args.group_key,
        "num_events": int(len(arr["paths"])),
        "num_groups": int(len(groups)),
        "folds": folds,
        "default_split": split,
        "disjoint_audit": overlap,
        "group_profiles": {g: group_profile(arr, idxs) for g, idxs in groups.items()},
    }
    (out / "source_disjoint_splits.json").write_text(json.dumps(j(report), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "source_disjoint_split_audit.json").write_text(json.dumps(j({"disjoint_audit": overlap, "folds": folds}), ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(arr, folds, out / "source_disjoint_splits.csv")
    print(json.dumps(j({"num_events": report["num_events"], "num_groups": report["num_groups"], "disjoint_audit": overlap}), ensure_ascii=False, indent=2))
    if not overlap.get("ok", False):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
