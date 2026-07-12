#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-build a gravity-valid EDGE151D cache from Chang-E BVH files."""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from tools.chang_e_edge_retarget import RetargetConfig, retarget_bvh


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="change")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--glob", default="**/*.bvh")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob(args.glob))
    if not files:
        raise RuntimeError(f"No BVH files found in {in_dir} with {args.glob!r}")

    cfg = RetargetConfig.from_env()
    if args.device:
        cfg.device = args.device

    reports = []
    failures = []
    for idx, src in enumerate(files, 1):
        rel = src.relative_to(in_dir)
        dst = (out_dir / rel).with_suffix(".npy")
        rep_path = dst.with_suffix(".retarget.json")
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[V46.49 RETARGET {idx}/{len(files)}] {src} -> {dst}", flush=True)
        try:
            if dst.exists() and rep_path.exists() and not args.overwrite:
                rep = json.loads(rep_path.read_text(encoding="utf-8"))
                if bool(rep.get("ok")):
                    print("[SKIP] existing valid cache", flush=True)
                    reports.append(rep)
                    continue
            motion, rep = retarget_bvh(src, cfg)
            np.save(dst, motion.astype(np.float32))
            rep["output"] = str(dst)
            rep["source_relative"] = str(rel)
            rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
            reports.append(rep)
        except Exception as exc:
            failure = {
                "source": str(src),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(f"[FAILED] {src}: {exc}", flush=True)
            if not args.allow_partial:
                summary = {
                    "version": "v46_49_retarget_cache",
                    "in_dir": str(in_dir),
                    "out_dir": str(out_dir),
                    "num_inputs": len(files),
                    "num_ok": len(reports),
                    "num_failed": len(failures),
                    "reports": reports,
                    "failures": failures,
                }
                (out_dir / "v46_49_retarget_cache_report.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                raise

    summary = {
        "version": "v46_49_retarget_cache",
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "num_inputs": len(files),
        "num_ok": len(reports),
        "num_failed": len(failures),
        "all_ok": len(reports) == len(files) and not failures,
        "fit_rmse_p95_max_m": max(
            [float(r.get("fit", {}).get("fit_rmse_p95_m", 0.0)) for r in reports],
            default=0.0,
        ),
        "torso_up_cos_p05_min": min(
            [float(r.get("gravity", {}).get("torso_up_cos_p05", 1.0)) for r in reports],
            default=1.0,
        ),
        "horizontal_body_ratio_max": max(
            [float(r.get("gravity", {}).get("horizontal_body_ratio", 0.0)) for r in reports],
            default=0.0,
        ),
        "reports": reports,
        "failures": failures,
    }
    report_path = out_dir / "v46_49_retarget_cache_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "num_inputs": len(files),
        "num_ok": len(reports),
        "num_failed": len(failures),
        "all_ok": summary["all_ok"],
    }, ensure_ascii=False, indent=2))
    if failures and not args.allow_partial:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
