#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit gravity/body-frame validity for cached motions, Event-RAG DB, or output."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from tools.v46_49_gravity_contract import (
    GravityThresholds,
    evaluate_gravity_contract,
    gravity_metrics_np,
)


def audit_one(path: Path, thresholds: GravityThresholds) -> dict:
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    metrics = gravity_metrics_np(arr)
    ok, reasons = evaluate_gravity_contract(metrics, thresholds)
    return {
        "path": str(path),
        "shape": list(arr.shape),
        "ok": bool(ok),
        "reasons": reasons,
        **metrics,
    }


def collect_paths(args) -> List[Path]:
    paths: List[Path] = []
    if args.input:
        paths.append(Path(args.input))
    if args.motion_dir:
        paths.extend(sorted(Path(args.motion_dir).rglob("*.npy")))
    if args.db:
        p = Path(args.db)
        db_file = p / "events.npz" if p.is_dir() else p
        data = np.load(db_file, allow_pickle=True)
        if "paths" not in data.files:
            raise RuntimeError(f"{db_file} has no paths array")
        paths.extend(Path(str(x)) for x in data["paths"].tolist())
    # Ordered unique paths.
    out, seen = [], set()
    for p in paths:
        s = str(p)
        if s not in seen and p.is_file():
            seen.add(s)
            out.append(p)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--motion_dir", default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--allow_failed", action="store_true")
    ap.add_argument("--torso_p05_min", type=float, default=0.45)
    ap.add_argument("--torso_median_min", type=float, default=0.70)
    ap.add_argument("--head_ratio_min", type=float, default=0.92)
    ap.add_argument("--feet_ratio_min", type=float, default=0.90)
    ap.add_argument("--horizontal_ratio_max", type=float, default=0.10)
    args = ap.parse_args(argv)

    th = GravityThresholds(
        torso_up_cos_p05_min=args.torso_p05_min,
        torso_up_cos_median_min=args.torso_median_min,
        head_above_pelvis_ratio_min=args.head_ratio_min,
        feet_below_pelvis_ratio_min=args.feet_ratio_min,
        horizontal_body_ratio_max=args.horizontal_ratio_max,
    )
    paths = collect_paths(args)
    if not paths:
        raise RuntimeError("No motions found")

    rows = []
    for i, p in enumerate(paths, 1):
        row = audit_one(p, th)
        rows.append(row)
        print(
            f"[{i}/{len(paths)}] ok={row['ok']} "
            f"torso_p05={row['torso_up_cos_p05']:.3f} "
            f"horizontal={row['horizontal_body_ratio']:.3f} {p}",
            flush=True,
        )

    failed = [r for r in rows if not r["ok"]]
    summary = {
        "version": "v46_49_gravity_contract_audit",
        "thresholds": th.to_dict(),
        "num_motions": len(rows),
        "num_passed": len(rows) - len(failed),
        "num_failed": len(failed),
        "all_ok": not failed,
        "torso_up_cos_p05_min_observed": min(r["torso_up_cos_p05"] for r in rows),
        "torso_up_cos_median_min_observed": min(r["torso_up_cos_median"] for r in rows),
        "horizontal_body_ratio_max_observed": max(r["horizontal_body_ratio"] for r in rows),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.csv:
        cp = Path(args.csv)
        cp.parent.mkdir(parents=True, exist_ok=True)
        keys = [
            "path", "ok", "frames", "torso_up_cos_p05", "torso_up_cos_median",
            "horizontal_body_ratio", "head_above_pelvis_ratio",
            "feet_below_pelvis_ratio", "root_up_cos_p05", "nonfinite_count", "reasons",
        ]
        with cp.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for r in rows:
                q = {k: r.get(k) for k in keys}
                q["reasons"] = " | ".join(r.get("reasons", []))
                wr.writerow(q)

    print(json.dumps({
        "out": str(out),
        "num_motions": len(rows),
        "num_failed": len(failed),
        "all_ok": not failed,
    }, ensure_ascii=False, indent=2))
    if failed and not args.allow_failed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
