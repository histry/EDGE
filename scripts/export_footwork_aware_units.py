#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path as _PathForSys

_REPO_ROOT = _PathForSys(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from functional_choreo_metrics import functional_choreo_stats, add_functional_scores


def pctl(x, q):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, q))


def safe_arr(arrays, name, n):
    if name in arrays:
        return np.nan_to_num(np.asarray(arrays[name], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.zeros((n,), dtype=np.float32)


def pick_top(mask, score, k, used):
    idx = np.where(mask)[0]
    idx = [int(i) for i in idx if int(i) not in used]
    idx = sorted(idx, key=lambda i: float(score[i]), reverse=True)
    return idx[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=45)
    ap.add_argument("--per_bucket", type=int, default=100)
    ap.add_argument("--max_total", type=int, default=420)
    ap.add_argument("--unit_key", default="", help="default: unit_motions_physical if exists else unit_motions")
    args = ap.parse_args()

    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = np.load(db_path, allow_pickle=True)
    keys = list(db.files)

    if args.unit_key:
        unit_key = args.unit_key
    elif "unit_motions_physical" in keys:
        unit_key = "unit_motions_physical"
    elif "unit_motions" in keys:
        unit_key = "unit_motions"
    else:
        raise KeyError(f"No unit_motions_physical/unit_motions in {db_path}")

    units = np.asarray(db[unit_key], dtype=np.float32)
    if units.ndim != 3 or units.shape[-1] != 151:
        raise ValueError(f"{unit_key} expected [N,T,151], got {units.shape}")

    valid = []
    for i, u in enumerate(units):
        if u.shape[0] >= args.seq_len:
            valid.append(i)

    if len(valid) == 0:
        raise RuntimeError(f"No valid unit with length >= {args.seq_len}")

    units = units[valid, :args.seq_len].astype(np.float32)
    orig_indices = np.asarray(valid, dtype=np.int64)
    n = len(units)

    print(f"Loaded DB: {db_path}")
    print(f"unit_key={unit_key}, valid_units={n}, seq_len={args.seq_len}")

    stats_list = [functional_choreo_stats(u) for u in units]
    stat_keys = sorted(stats_list[0].keys())

    arrays = {}
    for k in stat_keys:
        arrays[k] = np.asarray([float(s.get(k, 0.0)) for s in stats_list], dtype=np.float32)

    arrays = add_functional_scores(arrays)

    root_path = safe_arr(arrays, "root_path", n)
    root_speed = safe_arr(arrays, "root_speed_mean", n)
    turning = safe_arr(arrays, "turning_mean", n)
    lower = safe_arr(arrays, "lower_activity", n)
    torso = safe_arr(arrays, "torso_activity", n)
    upper = safe_arr(arrays, "upper_activity", n)
    expr = safe_arr(arrays, "expression_activity", n)
    contact = safe_arr(arrays, "contact_switch", n)

    support_score = safe_arr(arrays, "support_context_score", n)
    expressive_score = safe_arr(arrays, "expressive_mobile_score", n)
    coupling_score = safe_arr(arrays, "functional_coupling_score", n)
    mobile_expr_score = safe_arr(arrays, "mobile_expressive_score", n)

    combined = (
        0.30 * support_score
        + 0.25 * mobile_expr_score
        + 0.20 * coupling_score
        + 0.15 * expressive_score
        + 0.10 * lower / max(float(lower.max()), 1e-6)
    ).astype(np.float32)

    buckets = {
        "stationary_expr": {
            "mask": (root_path <= pctl(root_path, 55)) & (expr >= pctl(expr, 50)) & (upper >= pctl(upper, 45)),
            "score": 0.45 * expressive_score + 0.35 * coupling_score + 0.20 * upper,
        },
        "support_shift": {
            "mask": (root_path <= pctl(root_path, 70)) & (lower >= pctl(lower, 50)) & (contact >= pctl(contact, 45)),
            "score": 0.50 * support_score + 0.25 * lower + 0.25 * contact,
        },
        "turn_pivot": {
            "mask": (turning >= pctl(turning, 55)) & (support_score >= pctl(support_score, 40)),
            "score": 0.45 * coupling_score + 0.35 * turning + 0.20 * support_score,
        },
        "mobile_step": {
            "mask": (root_path >= pctl(root_path, 55)) & (lower >= pctl(lower, 45)) & (contact >= pctl(contact, 40)),
            "score": 0.45 * support_score + 0.30 * mobile_expr_score + 0.25 * root_speed,
        },
        "fullbody_coupled": {
            "mask": (coupling_score >= pctl(coupling_score, 60)) & (expr >= pctl(expr, 40)) & (lower >= pctl(lower, 40)),
            "score": 0.55 * coupling_score + 0.25 * expressive_score + 0.20 * support_score,
        },
    }

    selected = []
    bucket_of = {}
    used = set()

    for bucket, spec in buckets.items():
        take = pick_top(spec["mask"], spec["score"], args.per_bucket, used)
        for i in take:
            selected.append(i)
            bucket_of[i] = bucket
            used.add(i)
        print(f"{bucket}: picked={len(take)} candidates={int(np.asarray(spec['mask']).sum())}")

    if len(selected) < min(args.max_total, n):
        backfill = [int(i) for i in np.argsort(-combined) if int(i) not in used]
        for i in backfill:
            if len(selected) >= min(args.max_total, n):
                break
            selected.append(i)
            bucket_of[i] = "backfill"
            used.add(i)

    selected = selected[: min(args.max_total, len(selected))]
    print(f"TOTAL selected={len(selected)} / {n}")

    # Clean old pkl files from this export dir only.
    for old in out_dir.glob("*.pkl"):
        old.unlink()

    manifest_rows = []
    for rank, i in enumerate(selected):
        bucket = bucket_of.get(i, "unknown")
        src = f"footwork_{bucket}_unit_{int(orig_indices[i]):06d}"

        payload = {
            "motion": units[i].astype(np.float32),
            "motion_151": units[i].astype(np.float32),
            "original_filename": src,
            "source_file": src,
            "source_id": src,
            "unit_index": int(orig_indices[i]),
            "export_rank": int(rank),
            "footwork_bucket": bucket,
            "metrics": {k: float(arrays[k][i]) for k in arrays if np.asarray(arrays[k]).shape == (n,)},
            "source_db": str(db_path),
            "unit_key": unit_key,
        }

        name = f"{rank:04d}_{bucket}_u{int(orig_indices[i]):06d}.pkl"
        with open(out_dir / name, "wb") as f:
            pickle.dump(payload, f)

        manifest_rows.append({
            "rank": rank,
            "file": name,
            "orig_unit": int(orig_indices[i]),
            "bucket": bucket,
            "root_path": float(root_path[i]),
            "root_speed_mean": float(root_speed[i]),
            "turning_mean": float(turning[i]),
            "lower_activity": float(lower[i]),
            "torso_activity": float(torso[i]),
            "upper_activity": float(upper[i]),
            "expression_activity": float(expr[i]),
            "contact_switch": float(contact[i]),
            "support_context_score": float(support_score[i]),
            "expressive_mobile_score": float(expressive_score[i]),
            "functional_coupling_score": float(coupling_score[i]),
            "mobile_expressive_score": float(mobile_expr_score[i]),
            "combined": float(combined[i]),
        })

    manifest_csv = out_dir / "footwork_manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    bucket_counts = {}
    for r in manifest_rows:
        bucket_counts[r["bucket"]] = bucket_counts.get(r["bucket"], 0) + 1

    summary = {
        "db": str(db_path),
        "out_dir": str(out_dir),
        "unit_key": unit_key,
        "seq_len": args.seq_len,
        "selected": len(selected),
        "bucket_counts": bucket_counts,
        "manifest_csv": str(manifest_csv),
        "note": "Each exported pkl has unique source_file/source_id to avoid source-split collapsing to only a few train windows.",
    }

    summary_json = out_dir / "footwork_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Export done")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
