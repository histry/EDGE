#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact-cardinality source-disjoint split for low-resource retarget caches.

The public V46.51 splitter rounds within each dance label independently, which can
leave a globally empty validation/test split. This replacement first computes exact
global capacities (all non-empty for n>=3), then performs deterministic label-aware
assignment without source leakage.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.v46_motionrag_diff as v46

SCHEMA = "v46_53_1_exact_source_disjoint_cache_split"
SPLITS = ("train", "val", "test")


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
    p.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def stable_int(text: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}::{text}".encode("utf-8")).hexdigest()[:16], 16)


def exact_split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, int]:
    if n < 3:
        raise ValueError(f"source-disjoint train/val/test requires at least 3 sources, got {n}")
    ratios = [float(train_ratio), float(val_ratio), float(test_ratio)]
    total = sum(ratios)
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    ratios = [r / total for r in ratios]

    ideal = [n * r for r in ratios]
    counts = [int(x) for x in ideal]
    left = n - sum(counts)
    remainders = [ideal[i] - counts[i] for i in range(3)]
    order = sorted(range(3), key=lambda i: (remainders[i], ratios[i], -i), reverse=True)
    for k in range(left):
        counts[order[k % 3]] += 1

    # Guarantee a real held-out validation and test set for every n>=3.
    for receiver in (1, 2, 0):
        if counts[receiver] == 0:
            donor = max((i for i in range(3) if counts[i] > 1), key=lambda i: counts[i])
            counts[donor] -= 1
            counts[receiver] += 1
    return dict(zip(SPLITS, counts))


def assign_sources(
    source_to_label: Mapping[str, str], *, seed: int, train_ratio: float, val_ratio: float, test_ratio: float
) -> Dict[str, str]:
    sources = list(source_to_label)
    target = exact_split_counts(len(sources), train_ratio, val_ratio, test_ratio)
    remaining = dict(target)
    label_total = Counter(str(source_to_label[s]) for s in sources)
    # Rare labels first, then deterministic hash, to distribute scarce categories.
    ordered = sorted(sources, key=lambda s: (label_total[str(source_to_label[s])], stable_int(s, seed)))
    label_counts: Dict[str, Counter] = {sp: Counter() for sp in SPLITS}
    assignment: Dict[str, str] = {}

    for source in ordered:
        label = str(source_to_label[source])
        candidates = [sp for sp in SPLITS if remaining[sp] > 0]
        if not candidates:
            raise RuntimeError("split capacity exhausted before all sources were assigned")

        def cost(sp: str) -> tuple:
            # Prefer splits with low representation of this label and ample capacity.
            label_load = label_counts[sp][label] / max(1, target[sp])
            total_load = (target[sp] - remaining[sp]) / max(1, target[sp])
            heldout_bonus = 0.0 if sp == "train" else -0.02
            tie = stable_int(f"{source}::{sp}", seed)
            return (label_load + 0.30 * total_load + heldout_bonus, tie)

        chosen = min(candidates, key=cost)
        assignment[source] = chosen
        remaining[chosen] -= 1
        label_counts[chosen][label] += 1

    actual = Counter(assignment.values())
    if any(actual[sp] != target[sp] for sp in SPLITS):
        raise RuntimeError(f"exact split count mismatch: target={target}, actual={dict(actual)}")
    return assignment


def report_path_for_motion(path: Path) -> Path:
    return path.with_suffix(".retarget.json")


def source_record(cache_root: Path, motion_path: Path) -> Dict[str, Any]:
    report_path = report_path_for_motion(motion_path)
    if not report_path.is_file():
        raise FileNotFoundError(f"Retarget report missing for {motion_path}: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not bool(report.get("ok", False)) or not bool(report.get("source_gate_ok", report.get("anatomy_ok", False))):
        raise RuntimeError(f"Non-OK source-safe retarget cache: {motion_path}")

    rel = motion_path.relative_to(cache_root)
    original = str(report.get("source_used") or report.get("source") or report.get("source_relative") or rel.with_suffix(".bvh"))
    semantic = v46.parse_change_bvh_semantics(original)
    source_uid = str(semantic.get("source_uid") or Path(original).stem)
    dance_key = str(semantic.get("dance_key") or semantic.get("dance_category") or "unknown")
    return {
        "motion": str(motion_path.resolve()),
        "report": str(report_path.resolve()),
        "relative_motion": str(rel),
        "relative_report": str(report_path.relative_to(cache_root)),
        "original_source": original,
        "source_uid": source_uid,
        "dance_key": dance_key,
        "source_anatomy_quality": float(report.get("anatomy", {}).get("anatomy_quality", 0.0)),
        "source_gate_reasons": list(report.get("source_gate_reasons", [])),
        "semantic": semantic,
    }


def materialize(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst); return "copy"
    if mode == "hardlink":
        try:
            os.link(src, dst); return "hardlink"
        except OSError:
            shutil.copy2(src, dst); return "copy_fallback"
    try:
        os.symlink(src.resolve(), dst); return "symlink"
    except OSError:
        shutil.copy2(src, dst); return "copy_fallback"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument("--train_ratio", type=float, default=0.67)
    ap.add_argument("--val_ratio", type=float, default=0.165)
    ap.add_argument("--test_ratio", type=float, default=0.165)
    ap.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cache_root = Path(args.cache_root).resolve()
    out_root = Path(args.out_root).resolve()
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = []
    for p in sorted(cache_root.rglob("*.npy")):
        if any(token in p.name.lower() for token in ("motion_ref", "transition_mask", "single_test", "spin_interval", "jitter")):
            continue
        files.append(p)
    if not files:
        raise RuntimeError(f"No retarget cache NPY files in {cache_root}")

    records = [source_record(cache_root, p) for p in files]
    uid_counts = Counter(r["source_uid"] for r in records)
    duplicates = sorted(k for k, v in uid_counts.items() if v > 1)
    if duplicates:
        raise RuntimeError(f"source_uid must identify one complete source sequence; duplicates={duplicates[:20]}")

    target_counts = exact_split_counts(len(records), args.train_ratio, args.val_ratio, args.test_ratio)
    assignment = assign_sources(
        {r["source_uid"]: r["dance_key"] for r in records}, seed=args.seed,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
    )

    split_records: Dict[str, List[Dict[str, Any]]] = {sp: [] for sp in SPLITS}
    actual_modes = Counter()
    for record in records:
        split = assignment[record["source_uid"]]
        motion_src, report_src = Path(record["motion"]), Path(record["report"])
        motion_dst = out_root / split / record["relative_motion"]
        report_dst = out_root / split / record["relative_report"]
        actual_modes[materialize(motion_src, motion_dst, args.mode)] += 1
        actual_modes[materialize(report_src, report_dst, args.mode)] += 1
        row = dict(record)
        row.update({"split": split, "split_motion": str(motion_dst), "split_report": str(report_dst)})
        split_records[split].append(row)

    source_sets = {sp: {r["source_uid"] for r in rows} for sp, rows in split_records.items()}
    overlap = {
        "train_val": sorted(source_sets["train"] & source_sets["val"]),
        "train_test": sorted(source_sets["train"] & source_sets["test"]),
        "val_test": sorted(source_sets["val"] & source_sets["test"]),
    }
    reasons: List[str] = []
    if any(overlap.values()): reasons.append("source_overlap")
    for sp in SPLITS:
        if len(split_records[sp]) != target_counts[sp]: reasons.append(f"count_mismatch_{sp}")
        if not split_records[sp]: reasons.append(f"empty_{sp}")

    report = {
        "schema": SCHEMA,
        "ok": not reasons,
        "reasons": reasons,
        "cache_root": str(cache_root),
        "out_root": str(out_root),
        "seed": args.seed,
        "split_ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "target_counts": target_counts,
        "assignment_unit": "source_uid_before_event_slicing",
        "assignment_algorithm": "exact_global_capacity_label_aware_deterministic_greedy",
        "materialization_requested": args.mode,
        "materialization_actual": dict(actual_modes),
        "num_sources": len(records),
        "splits": {
            sp: {
                "sources": len(rows),
                "source_uids": sorted(r["source_uid"] for r in rows),
                "dance_key_histogram": dict(Counter(r["dance_key"] for r in rows)),
                "records": rows,
            } for sp, rows in split_records.items()
        },
        "overlap": overlap,
        "policy": {
            "split_before_event_slicing": True,
            "train_motion_only": "training_and_retrieval",
            "val_test_motion": "evaluation_only",
            "all_splits_nonempty": True,
        },
    }
    manifest = out_root / "source_split_manifest.json"
    save_json(report, manifest)
    print(json.dumps({
        "manifest": str(manifest), "ok": report["ok"], "reasons": reasons,
        "num_sources": len(records),
        "train_sources": len(split_records["train"]),
        "val_sources": len(split_records["val"]),
        "test_sources": len(split_records["test"]),
    }, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
