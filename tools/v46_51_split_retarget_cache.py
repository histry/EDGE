#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split V46.49.4 retarget caches by source before event slicing.

Every retargeted source sequence is assigned to train/val/test before
`v46_50_build_event_heading_db.py` performs adaptive event segmentation.
This directly enforces the paper's source-disjoint protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.v46_motionrag_diff as v46  # noqa: E402


SCHEMA = "v46_51_pre_event_source_disjoint_cache_split"


def jsonable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
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
    return int(
        hashlib.sha256(f"{seed}::{text}".encode("utf-8")).hexdigest()[:16],
        16,
    )


def assign_sources(
    source_to_label: Mapping[str, str],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, str]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    by_label: Dict[str, List[str]] = defaultdict(list)
    for source, label in source_to_label.items():
        by_label[str(label)].append(str(source))

    assignment: Dict[str, str] = {}
    heldout_toggle = 0
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
            if heldout_toggle % 2 == 0:
                n_val, n_test = 1, 0
            else:
                n_val, n_test = 0, 1
            heldout_toggle += 1
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


def report_path_for_motion(path: Path) -> Path:
    return path.with_suffix(".retarget.json")


def source_record(cache_root: Path, motion_path: Path) -> Dict[str, Any]:
    report_path = report_path_for_motion(motion_path)
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Retarget report missing for {motion_path}: {report_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not bool(report.get("ok", False)):
        raise RuntimeError(f"Non-OK retarget cache: {motion_path}")

    rel = motion_path.relative_to(cache_root)
    original = str(
        report.get("source")
        or report.get("source_relative")
        or rel.with_suffix(".bvh")
    )
    semantic = v46.parse_change_bvh_semantics(original)
    source_uid = str(
        semantic.get("source_uid")
        or Path(original).stem
    )
    dance_key = str(
        semantic.get("dance_key")
        or semantic.get("dance_category")
        or "unknown"
    )
    return {
        "motion": str(motion_path.resolve()),
        "report": str(report_path.resolve()),
        "relative_motion": str(rel),
        "relative_report": str(
            report_path.relative_to(cache_root)
        ),
        "original_source": original,
        "source_uid": source_uid,
        "dance_key": dance_key,
        "semantic": semantic,
    }


def materialize(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy_fallback"
    try:
        os.symlink(src.resolve(), dst)
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy_fallback"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, default=20260712)
    ap.add_argument("--train_ratio", type=float, default=0.80)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--test_ratio", type=float, default=0.10)
    ap.add_argument(
        "--mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cache_root = Path(args.cache_root).resolve()
    out_root = Path(args.out_root).resolve()
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = []
    for p in sorted(cache_root.rglob("*.npy")):
        name = p.name.lower()
        if any(
            token in name
            for token in (
                "motion_ref",
                "transition_mask",
                "single_test",
                "spin_interval",
                "jitter",
            )
        ):
            continue
        files.append(p)
    if not files:
        raise RuntimeError(f"No retarget cache NPY files in {cache_root}")

    records = [source_record(cache_root, p) for p in files]
    uid_counts = Counter(r["source_uid"] for r in records)
    duplicate_uids = sorted(k for k, v in uid_counts.items() if v > 1)
    if duplicate_uids:
        raise RuntimeError(
            "source_uid must identify one complete source sequence before "
            f"event slicing; duplicates={duplicate_uids[:20]}"
        )

    source_to_label = {
        r["source_uid"]: r["dance_key"] for r in records
    }
    assignment = assign_sources(
        source_to_label,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    split_records: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    actual_modes = Counter()
    for record in records:
        split = assignment[record["source_uid"]]
        motion_src = Path(record["motion"])
        report_src = Path(record["report"])
        motion_dst = out_root / split / record["relative_motion"]
        report_dst = out_root / split / record["relative_report"]
        actual_modes[materialize(motion_src, motion_dst, args.mode)] += 1
        actual_modes[materialize(report_src, report_dst, args.mode)] += 1
        copied = dict(record)
        copied["split"] = split
        copied["split_motion"] = str(motion_dst)
        copied["split_report"] = str(report_dst)
        split_records[split].append(copied)

    source_sets = {
        k: set(r["source_uid"] for r in rows)
        for k, rows in split_records.items()
    }
    overlap = {
        "train_val": sorted(source_sets["train"] & source_sets["val"]),
        "train_test": sorted(source_sets["train"] & source_sets["test"]),
        "val_test": sorted(source_sets["val"] & source_sets["test"]),
    }
    reasons: List[str] = []
    if any(overlap.values()):
        reasons.append("source_overlap")
    if not split_records["train"]:
        reasons.append("empty_train")
    if len(records) >= 3 and not split_records["val"]:
        reasons.append("empty_val")
    if len(records) >= 3 and not split_records["test"]:
        reasons.append("empty_test")

    report = {
        "schema": SCHEMA,
        "ok": not reasons,
        "reasons": reasons,
        "cache_root": str(cache_root),
        "out_root": str(out_root),
        "seed": args.seed,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "assignment_unit": "source_uid_before_event_slicing",
        "materialization_requested": args.mode,
        "materialization_actual": dict(actual_modes),
        "num_sources": len(records),
        "splits": {
            split: {
                "sources": len(rows),
                "source_uids": sorted(
                    r["source_uid"] for r in rows
                ),
                "dance_key_histogram": dict(
                    Counter(r["dance_key"] for r in rows)
                ),
                "records": rows,
            }
            for split, rows in split_records.items()
        },
        "overlap": overlap,
        "policy": {
            "split_before_event_slicing": True,
            "train_motion_only": "training_and_retrieval",
            "val_test_motion": "evaluation_only",
            "all_change": "qualitative_upper_bound_only",
        },
    }
    report_path = out_root / "source_split_manifest.json"
    save_json(report, report_path)
    print(
        json.dumps(
            {
                "manifest": str(report_path),
                "ok": report["ok"],
                "reasons": reasons,
                "num_sources": len(records),
                "train_sources": len(split_records["train"]),
                "val_sources": len(split_records["val"]),
                "test_sources": len(split_records["test"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
