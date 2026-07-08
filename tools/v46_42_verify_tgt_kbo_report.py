#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify V46.42 Stage-Anchored Guided TGT/KBO report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--max_root_y_range", type=float, default=3.0)
    ap.add_argument("--max_abs_floor_y", type=float, default=10.0)
    ap.add_argument("--max_jerk_p95", type=float, default=5.0)
    ap.add_argument("--max_skate_p95", type=float, default=5.0)
    ap.add_argument("--require_v46_38_routing", action="store_true")
    ap.add_argument("--require_v46_41_tokens", action="store_true")
    ap.add_argument("--require_v46_42_metadata", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    p = Path(args.report)
    r = json.load(open(p, "r", encoding="utf-8"))
    fa = r.get("final_audit", {})
    errors = []
    if float(fa.get("root_y_range_m", 0.0)) > args.max_root_y_range:
        errors.append(f"root_y_range_m too large: {fa.get('root_y_range_m')}")
    if abs(float(fa.get("floor_y", 0.0))) > args.max_abs_floor_y:
        errors.append(f"abs(floor_y) too large: {fa.get('floor_y')}")
    if float(fa.get("mean_joint_jerk_p95", 0.0)) > args.max_jerk_p95:
        errors.append(f"mean_joint_jerk_p95 too large: {fa.get('mean_joint_jerk_p95')}")
    if float(fa.get("foot_skate_p95_mpf", 0.0)) > args.max_skate_p95:
        errors.append(f"foot_skate_p95_mpf too large: {fa.get('foot_skate_p95_mpf')}")

    tx = r.get("stage_reports", {}).get("v46_41_temporal_generative_transactions", [])
    summ = r.get("v46_41_tgt_kbo_summary", {})
    if args.require_v46_41_tokens:
        if not isinstance(tx, list):
            errors.append("v46_41_temporal_generative_transactions is not a list")
        if not summ:
            errors.append("missing v46_41_tgt_kbo_summary")

    meta42 = r.get("v46_42_stability_alignment", {})
    if args.require_v46_42_metadata and not meta42:
        errors.append("missing v46_42_stability_alignment metadata")

    if args.require_v46_38_routing:
        retrieval = r.get("stage_reports", {}).get("retrieval", [])
        policies = [str(x.get("routing_policy", "")) for x in retrieval if isinstance(x, dict)]
        if not any("V46.38" in x and "MSSD-AESD" in x for x in policies):
            errors.append("retrieval report does not show V46.38 MSSD-AESD routing")

    out = {
        "report": str(p),
        "final_audit": fa,
        "transaction_records": len(tx) if isinstance(tx, list) else -1,
        "transaction_summary": summ,
        "v46_42_stability_alignment": meta42,
        "ok": not errors,
        "errors": errors,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if errors:
        raise RuntimeError("; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
