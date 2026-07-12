#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a strict V46.49.4 retarget cache for the V46.50 Event-RAG database.

Unlike the earlier cache builder, an existing file is skipped only when its
report proves all formal contracts:
- non-root position mode = ignore;
- absolute heading mode = stabilize;
- target root orientation = absolute_reference_lock;
- gravity and fit gates passed.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.chang_e_edge_retarget import RetargetConfig, retarget_bvh  # noqa: E402


def report_is_formal(rep: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not bool(rep.get("ok", False)):
        reasons.append("not_ok")
    pos = rep.get("source_position_contract", {})
    if str(pos.get("nonroot_position_mode", "")) != "ignore":
        reasons.append("nonroot_position_mode_not_ignore")
    fit = rep.get("fit", {})
    heading = fit.get("heading_contract", {})
    if str(heading.get("mode", "")) != "stabilize":
        reasons.append("heading_mode_not_stabilize")
    root = fit.get("root_orientation_contract", {})
    if str(root.get("mode", "")) != "absolute_reference_lock":
        reasons.append("root_orientation_not_absolute_reference_lock")
    if not bool(rep.get("fit_ok", False)):
        reasons.append("fit_not_ok")
    if not bool(rep.get("gravity_ok", False)):
        reasons.append("gravity_not_ok")
    return not reasons, reasons


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

    reports: List[dict] = []
    failures: List[dict] = []
    stale_rebuilt: List[dict] = []

    for idx, src in enumerate(files, 1):
        rel = src.relative_to(in_dir)
        dst = (out_dir / rel).with_suffix(".npy")
        rep_path = dst.with_suffix(".retarget.json")
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[V46.50 RETARGET {idx}/{len(files)}] {src} -> {dst}", flush=True)

        try:
            if dst.exists() and rep_path.exists() and not args.overwrite:
                old = json.loads(rep_path.read_text(encoding="utf-8"))
                valid, reasons = report_is_formal(old)
                if valid:
                    print("[SKIP] existing formal V46.49.4 cache", flush=True)
                    reports.append(old)
                    continue
                stale_rebuilt.append({
                    "source": str(src),
                    "output": str(dst),
                    "reasons": reasons,
                })
                print(f"[REBUILD STALE] reasons={reasons}", flush=True)

            motion, rep = retarget_bvh(src, cfg)
            rep["output"] = str(dst)
            rep["source_relative"] = str(rel)
            rep["v46_50_cache_contract"] = {
                "schema": "v46_50_strict_retarget_cache",
                "requires_nonroot_position_ignore": True,
                "requires_heading_stabilize": True,
                "requires_absolute_root_orientation_lock": True,
            }
            formal, reasons = report_is_formal(rep)
            if not formal:
                raise RuntimeError(
                    f"Retarget returned non-formal report for {src}: {reasons}"
                )

            np.save(dst, motion.astype(np.float32))
            rep_path.write_text(
                json.dumps(rep, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
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
                break

    summary = {
        "schema": "v46_50_strict_retarget_cache",
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "num_inputs": int(len(files)),
        "num_ok": int(len(reports)),
        "num_failed": int(len(failures)),
        "num_stale_rebuilt": int(len(stale_rebuilt)),
        "all_ok": bool(len(reports) == len(files) and not failures),
        "fit_rmse_p95_max_m": max(
            [
                float(r.get("fit", {}).get("fit_rmse_p95_m", 0.0))
                for r in reports
            ],
            default=0.0,
        ),
        "torso_up_cos_p05_min": min(
            [
                float(r.get("gravity", {}).get("torso_up_cos_p05", 1.0))
                for r in reports
            ],
            default=1.0,
        ),
        "horizontal_body_ratio_max": max(
            [
                float(r.get("gravity", {}).get("horizontal_body_ratio", 0.0))
                for r in reports
            ],
            default=0.0,
        ),
        "stale_rebuilt": stale_rebuilt,
        "reports": reports,
        "failures": failures,
    }
    report_path = out_dir / "v46_50_retarget_cache_report.json"
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "report": str(report_path),
        "num_inputs": len(files),
        "num_ok": len(reports),
        "num_failed": len(failures),
        "num_stale_rebuilt": len(stale_rebuilt),
        "all_ok": summary["all_ok"],
    }, ensure_ascii=False, indent=2))
    return 0 if summary["all_ok"] or args.allow_partial else 2


if __name__ == "__main__":
    raise SystemExit(main())
