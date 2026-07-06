#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit official Chang-E source-level split manifests.

This is a lightweight guard before building train/val/test RAG DBs.  It checks:
- manifest files exist and are non-empty;
- source_id does not overlap across train/val/test;
- source_bvh/fragment_file rows do not duplicate across splits;
- label distributions are reported for the paper and experiment log.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def clean(x) -> str:
    if x is None:
        return ""
    s = str(x).strip().strip('"').strip("'")
    if s.lower() in {"none", "null", "nan"}:
        return ""
    return s


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [{str(k): clean(v) for k, v in row.items() if k is not None} for row in csv.DictReader(f)]
    if not rows:
        raise RuntimeError(f"Empty split manifest: {path}")
    return rows


def row_identity(row: Dict[str, str]) -> str:
    return clean(row.get("fragment_file")) or clean(row.get("source_bvh")) or clean(row.get("source_id"))


def counts(rows: List[Dict[str, str]], key: str):
    return Counter(clean(r.get(key)) or "unknown" for r in rows).most_common()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", default="change/splits_official")
    ap.add_argument("--json", default="")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    split_dir = Path(args.split_dir)
    rows = {s: read_csv(split_dir / f"{s}_manifest.csv") for s in ["train", "val", "test"]}
    sources = {s: {clean(r.get("source_id")) for r in rows[s] if clean(r.get("source_id"))} for s in rows}
    ids = {s: {row_identity(r) for r in rows[s] if row_identity(r)} for s in rows}

    leakage = {
        "source_train_val": sorted(sources["train"] & sources["val"]),
        "source_train_test": sorted(sources["train"] & sources["test"]),
        "source_val_test": sorted(sources["val"] & sources["test"]),
        "row_train_val": sorted(ids["train"] & ids["val"]),
        "row_train_test": sorted(ids["train"] & ids["test"]),
        "row_val_test": sorted(ids["val"] & ids["test"]),
    }
    bad = {k: v for k, v in leakage.items() if v}
    summary = {
        "version": "v46_23_official_enriched_split_audit",
        "split_dir": str(split_dir),
        "num_records": {s: len(rows[s]) for s in rows},
        "num_sources": {s: len(sources[s]) for s in sources},
        "leakage": leakage,
        "ok": not bool(bad),
        "per_split": {
            s: {
                "dance_key_counts": counts(rows[s], "dance_key"),
                "gender_counts": counts(rows[s], "gender"),
                "music_alignment_counts": counts(rows[s], "music_alignment_label"),
                "source_ids": sorted(sources[s]),
            }
            for s in ["train", "val", "test"]
        },
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    out = Path(args.json) if args.json else split_dir / "split_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    if args.strict and bad:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
