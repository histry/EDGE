#!/usr/bin/env python3
"""Augment an existing ChoreoRAG DB with functional choreography scores.

This is the v13 no-retrain upgrade:
  - support context decides when to shift weight / switch support / step;
  - expressive-mobile context decides how torso/arms respond during those events;
  - coupling metrics quantify whether these happen as one choreography event.

Input:
  data/dunhuang_choreo_unit_rag/index_v12_footstep_u45_s15.npz

Output:
  data/dunhuang_choreo_unit_rag/index_v13_functional_u45_s15.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from functional_choreo_metrics import (
    add_functional_scores,
    functional_choreo_stats,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--unit_key", default="", help="Auto by default: unit_motions_physical if exists else unit_motions")
    args = ap.parse_args()

    in_db = Path(args.in_db)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    db = np.load(in_db, allow_pickle=True)
    payload = {k: db[k] for k in db.files}

    if args.unit_key:
        unit_key = args.unit_key
    elif "unit_motions_physical" in payload:
        unit_key = "unit_motions_physical"
    elif "unit_motions" in payload:
        unit_key = "unit_motions"
    else:
        raise KeyError("DB must contain unit_motions or unit_motions_physical")

    units = np.asarray(payload[unit_key], dtype=np.float32)
    if units.ndim != 3 or units.shape[-1] != 151:
        raise ValueError(f"{unit_key} expected [N,L,151], got {units.shape}")

    stats_list = [functional_choreo_stats(u) for u in units]
    stat_keys = sorted(stats_list[0].keys())

    arrays = {}
    for k in stat_keys:
        arrays[k] = np.asarray([float(s.get(k, 0.0)) for s in stats_list], dtype=np.float32)

    arrays = add_functional_scores(arrays)

    # Keep backward DB fields and add functional fields.
    for k, v in arrays.items():
        payload[k] = v.astype(np.float32)

    old_text = payload.get("motion_text", np.asarray([""] * len(units))).astype(str)
    captions = []
    for i, s in enumerate(stats_list):
        support_label = "支撑切换明显" if arrays["support_context_score"][i] >= np.percentile(arrays["support_context_score"], 70) else "支撑稳定"
        expr_label = "移动中上身表达明显" if arrays["expressive_mobile_score"][i] >= np.percentile(arrays["expressive_mobile_score"], 70) else "上身表达含蓄"
        coupling_label = "走位-支撑-表达耦合强" if arrays["functional_coupling_score"][i] >= np.percentile(arrays["functional_coupling_score"], 70) else "耦合一般"
        mobile_expr_label = "移动表达单元" if arrays["mobile_expressive_score"][i] >= np.percentile(arrays["mobile_expressive_score"], 70) else "稳定单元"
        captions.append(f"{old_text[i]}，{support_label}，{expr_label}，{coupling_label}，{mobile_expr_label}")

    payload["motion_text_functional"] = np.asarray(captions)
    payload["db_type"] = np.asarray(["functional_choreo_unit_rag"])

    np.savez_compressed(out, **payload)

    meta = {
        "in_db": str(in_db),
        "out": str(out),
        "unit_key": unit_key,
        "count": int(len(units)),
        "new_score_fields": [
            "support_context_score",
            "expressive_mobile_score",
            "functional_coupling_score",
            "mobile_expressive_score",
        ],
        "new_metric_fields": stat_keys,
        "purpose": "functional collaboration between support context and expressive-mobile context",
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Functional ChoreoRAG DB saved: {out}")
    print(f"   units={len(units)} unit_key={unit_key}")
    for k in ["support_context_score", "expressive_mobile_score", "functional_coupling_score", "mobile_expressive_score"]:
        arr = arrays[k]
        print(f"   {k}: mean={float(arr.mean()):.4f} p70={float(np.percentile(arr,70)):.4f} p90={float(np.percentile(arr,90)):.4f}")
    print(f"   meta={out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
