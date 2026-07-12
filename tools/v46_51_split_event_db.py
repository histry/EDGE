#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create source-disjoint train/val/test Event-RAG databases.

The assignment is deterministic and made at source_uid level.  All events from
one source remain in exactly one split.  This avoids patch-level/event-level
leakage while preserving the full heading-aware V46.50 NPZ schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA = "v46_51_source_disjoint_event_db_split"


def jsonable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(jsonable(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def stable_int(text: str, seed: int) -> int:
    payload = f"{seed}::{text}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def dominant_label(values: Sequence[str]) -> str:
    if not values:
        return "unknown"
    counts = Counter(str(x) for x in values)
    return counts.most_common(1)[0][0]


def assign_sources(
    source_to_label: Mapping[str, str],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, str]:
    """Stratified deterministic assignment with train coverage priority."""
    total = float(train_ratio + val_ratio + test_ratio)
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value")
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    by_label: Dict[str, List[str]] = defaultdict(list)
    for source, label in source_to_label.items():
        by_label[str(label)].append(str(source))

    assignment: Dict[str, str] = {}
    toggle = 0
    for label in sorted(by_label):
        sources = sorted(
            by_label[label],
            key=lambda s: stable_int(f"{label}::{s}", seed),
        )
        n = len(sources)
        if n == 1:
            n_train, n_val, n_test = 1, 0, 0
        elif n == 2:
            n_train = 1
            # Alternate the held-out destination across categories.
            if toggle % 2 == 0:
                n_val, n_test = 1, 0
            else:
                n_val, n_test = 0, 1
            toggle += 1
        else:
            n_val = max(1, int(round(n * val_ratio)))
            n_test = max(1, int(round(n * test_ratio)))
            while n_val + n_test >= n:
                if n_val >= n_test and n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break
            n_train = n - n_val - n_test

        for source in sources[:n_train]:
            assignment[source] = "train"
        for source in sources[n_train : n_train + n_val]:
            assignment[source] = "val"
        for source in sources[n_train + n_val :]:
            assignment[source] = "test"

    return assignment


def subset_payload(
    db: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    split: str,
    manifest_sha256: str,
) -> Dict[str, np.ndarray]:
    n = int(len(mask))
    out: Dict[str, np.ndarray] = {}
    for key, arr0 in db.items():
        arr = np.asarray(arr0)
        if arr.ndim >= 1 and arr.shape[0] == n:
            out[key] = arr[mask]
        else:
            out[key] = arr
    out["research_split"] = np.asarray(split, dtype=object)
    out["source_split_manifest_sha256"] = np.asarray(
        manifest_sha256,
        dtype=object,
    )
    out["source_split_schema"] = np.asarray(SCHEMA, dtype=object)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, default=20260712)
    ap.add_argument("--train_ratio", type=float, default=0.80)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--test_ratio", type=float, default=0.10)
    ap.add_argument("--min_train_sources", type=int, default=2)
    args = ap.parse_args(argv)

    src = Path(args.db)
    data = np.load(src, allow_pickle=True)
    db = {k: data[k] for k in data.files}
    if "paths" not in db or "source_uids" not in db:
        raise RuntimeError(
            "Event DB requires paths and source_uids arrays"
        )
    n = int(len(db["paths"]))
    source_uids = np.asarray(db["source_uids"], dtype=object).astype(str)
    if len(source_uids) != n:
        raise RuntimeError("source_uids length mismatch")
    dance_keys = np.asarray(
        db.get(
            "dance_keys",
            np.asarray(["unknown"] * n, dtype=object),
        ),
        dtype=object,
    ).astype(str)

    source_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, source in enumerate(source_uids.tolist()):
        source_to_indices[str(source)].append(int(i))

    source_to_label = {
        source: dominant_label(dance_keys[idxs].tolist())
        for source, idxs in source_to_indices.items()
    }
    assignment = assign_sources(
        source_to_label,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    manifest_base = {
        "schema": SCHEMA,
        "source_db": str(src.resolve()),
        "seed": int(args.seed),
        "ratios_requested": {
            "train": float(args.train_ratio),
            "val": float(args.val_ratio),
            "test": float(args.test_ratio),
        },
        "source_assignment": assignment,
        "source_labels": source_to_label,
    }
    manifest_bytes = json.dumps(
        jsonable(manifest_base),
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    split_reports: Dict[str, Any] = {}
    split_sources: Dict[str, set] = {}

    meta_path = src.parent / "events_meta.json"
    meta_rows = None
    if meta_path.is_file():
        try:
            obj = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(obj, list) and len(obj) == n:
                meta_rows = obj
        except Exception:
            meta_rows = None

    for split in ("train", "val", "test"):
        mask = np.asarray(
            [assignment[str(s)] == split for s in source_uids],
            dtype=bool,
        )
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_db = split_dir / "events.npz"
        payload = subset_payload(
            db,
            mask,
            split=split,
            manifest_sha256=manifest_sha,
        )
        np.savez_compressed(split_db, **payload)

        indices = np.where(mask)[0].tolist()
        if meta_rows is not None:
            save_json(
                [meta_rows[i] for i in indices],
                split_dir / "events_meta.json",
            )

        sources = set(source_uids[mask].tolist())
        split_sources[split] = sources
        labels = dance_keys[mask].tolist()
        split_reports[split] = {
            "db": str(split_db),
            "events": int(mask.sum()),
            "sources": int(len(sources)),
            "source_uids": sorted(sources),
            "dance_key_histogram": {
                label: int(sum(x == label for x in labels))
                for label in sorted(set(labels))
            },
        }

    overlap = {
        "train_val": sorted(
            split_sources["train"] & split_sources["val"]
        ),
        "train_test": sorted(
            split_sources["train"] & split_sources["test"]
        ),
        "val_test": sorted(
            split_sources["val"] & split_sources["test"]
        ),
    }
    reasons: List[str] = []
    if any(overlap.values()):
        reasons.append("source_overlap_between_splits")
    if len(split_sources["train"]) < int(args.min_train_sources):
        reasons.append("too_few_train_sources")
    if split_reports["train"]["events"] <= 0:
        reasons.append("empty_train_split")
    if len(source_to_indices) >= 3:
        if split_reports["val"]["events"] <= 0:
            reasons.append("empty_val_split")
        if split_reports["test"]["events"] <= 0:
            reasons.append("empty_test_split")

    manifest = {
        **manifest_base,
        "manifest_sha256": manifest_sha,
        "ok": not reasons,
        "reasons": reasons,
        "num_events": n,
        "num_sources": int(len(source_to_indices)),
        "splits": split_reports,
        "overlap": overlap,
        "policy": {
            "assignment_unit": "source_uid",
            "event_slicing_leakage": "prohibited",
            "training_db": "train only",
            "val_test_usage": "evaluation only",
            "all_change_db_usage": "qualitative upper-bound only",
        },
    }
    manifest_path = out_root / "source_split_manifest.json"
    save_json(manifest, manifest_path)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "ok": manifest["ok"],
                "reasons": reasons,
                "num_sources": manifest["num_sources"],
                "splits": {
                    k: {
                        "events": v["events"],
                        "sources": v["sources"],
                        "db": v["db"],
                    }
                    for k, v in split_reports.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if manifest["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
