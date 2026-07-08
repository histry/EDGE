#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract hard-negative preference pairs from V46.42/V46.41 audit reports.

The generation patch can save per-transaction snapshot/rejected/accepted .npy
triples.  This script combines those records with audit tokens into a clean JSONL
file for HN-DPO-style diffusion preference fine-tuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=None)
    ap.add_argument("--pair_dir", default="output/v46_42_hn_dpo_pairs")
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    records = []
    pair_dir = Path(args.pair_dir)
    records.extend(read_jsonl(pair_dir / "pairs.jsonl"))

    if args.report:
        r = json.load(open(args.report, "r", encoding="utf-8"))
        for tok in r.get("stage_reports", {}).get("v46_41_temporal_generative_transactions", []):
            pair = tok.get("hn_dpo_pair") if isinstance(tok, dict) else None
            if isinstance(pair, dict):
                records.append(pair)

    # Deduplicate by rejected path.
    seen = set()
    clean = []
    for rec in records:
        key = rec.get("rejected")
        if not key or key in seen:
            continue
        required = [rec.get("snapshot"), rec.get("rejected"), rec.get("accepted")]
        if not all(x and Path(x).exists() for x in required):
            continue
        seen.add(key)
        clean.append({
            "snapshot": rec["snapshot"],
            "preferred": rec["accepted"],
            "rejected": rec["rejected"],
            "stage": rec.get("stage", "unknown"),
            "span": rec.get("span", []),
            "reasons": rec.get("reasons", []),
            "preference": "preferred_is_kbo_safe_or_snapshot; rejected_triggered_kbo",
        })

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in clean:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({"out_jsonl": str(out), "pairs": len(clean)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
