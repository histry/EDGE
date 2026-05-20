#!/usr/bin/env python3
import argparse, csv, json, pickle
import sys
from pathlib import Path as _PathForSys
_REPO_ROOT = _PathForSys(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathlib import Path
import numpy as np
from functional_choreo_metrics import functional_choreo_stats, add_functional_scores

def pctl(x, q):
    x = np.asarray(x, dtype=np.float32)
    return float(np.percentile(x, q)) if x.size else 0.0

def safe_arr(arrays, name, n):
    if name in arrays:
        return np.nan_to_num(np.asarray(arrays[name], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.zeros((n,), dtype=np.float32)

def pick_top(mask, score, k, used):
    idx = np.where(mask)[0]
    idx = [int(i) for i in idx if int(i) not in used]
    return sorted(idx, key=lambda i: float(score[i]), reverse=True)[:k]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=45)
    ap.add_argument("--per_bucket", type=int, default=120)
    ap.add_argument("--max_total", type=int, default=500)
    ap.add_argument("--hf_weight", type=float, default=0.35)
    args = ap.parse_args()

    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = np.load(db_path, allow_pickle=True)
    keys = list(db.files)

    if "unit_motions_physical" in keys:
        unit_key = "unit_motions_physical"
    elif "unit_motions" in keys:
        unit_key = "unit_motions"
    else:
        raise KeyError(f"No unit_motions_physical/unit_motions in {db_path}")

    if "hf_event_score" not in keys:
        raise KeyError("No hf_event_score in DB. Check HF-event export.")

    units_all = np.asarray(db[unit_key], dtype=np.float32)
    hf_all = np.asarray(db["hf_event_score"], dtype=np.float32)

    valid = [i for i, u in enumerate(units_all) if u.shape[0] >= args.seq_len]
    units = units_all[valid, :args.seq_len].astype(np.float32)
    hf = hf_all[valid].astype(np.float32)
    orig_indices = np.asarray(valid, dtype=np.int64)
    n = len(units)

    hf_norm = (hf - np.percentile(hf, 5)) / max(np.percentile(hf, 95) - np.percentile(hf, 5), 1e-6)
    hf_norm = np.clip(hf_norm, 0.0, 1.0).astype(np.float32)

    print(f"Loaded DB: {db_path}")
    print(f"unit_key={unit_key}, valid_units={n}, seq_len={args.seq_len}")
    print(f"hf_event_score min/mean/max = {hf.min():.6f}/{hf.mean():.6f}/{hf.max():.6f}")

    stats_list = [functional_choreo_stats(u) for u in units]
    arrays = {}
    for k in sorted(stats_list[0].keys()):
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

    w = float(args.hf_weight)
    buckets = {
        "hf_stationary_expr": {
            "mask": (root_path <= pctl(root_path, 60)) & (expr >= pctl(expr, 45)) & (upper >= pctl(upper, 40)),
            "score": (1-w) * (0.45 * expressive_score + 0.35 * coupling_score + 0.20 * upper) + w * hf_norm,
        },
        "hf_support_shift": {
            "mask": (root_path <= pctl(root_path, 75)) & (lower >= pctl(lower, 45)) & (contact >= pctl(contact, 40)),
            "score": (1-w) * (0.50 * support_score + 0.25 * lower + 0.25 * contact) + w * hf_norm,
        },
        "hf_turn_pivot": {
            "mask": (turning >= pctl(turning, 50)) & (support_score >= pctl(support_score, 35)),
            "score": (1-w) * (0.45 * coupling_score + 0.35 * turning + 0.20 * support_score) + w * hf_norm,
        },
        "hf_mobile_step": {
            "mask": (root_path >= pctl(root_path, 50)) & (lower >= pctl(lower, 40)) & (contact >= pctl(contact, 35)),
            "score": (1-w) * (0.45 * support_score + 0.30 * mobile_expr_score + 0.25 * root_speed) + w * hf_norm,
        },
        "hf_fullbody_coupled": {
            "mask": (coupling_score >= pctl(coupling_score, 55)) & (expr >= pctl(expr, 35)) & (lower >= pctl(lower, 35)),
            "score": (1-w) * (0.55 * coupling_score + 0.25 * expressive_score + 0.20 * support_score) + w * hf_norm,
        },
    }

    selected, bucket_of, used = [], {}, set()
    for bucket, spec in buckets.items():
        take = pick_top(spec["mask"], spec["score"], args.per_bucket, used)
        for i in take:
            selected.append(i)
            bucket_of[i] = bucket
            used.add(i)
        print(f"{bucket}: picked={len(take)} candidates={int(np.asarray(spec['mask']).sum())}")

    combined = (
        0.24 * support_score
        + 0.20 * mobile_expr_score
        + 0.18 * coupling_score
        + 0.13 * expressive_score
        + 0.10 * lower / max(float(lower.max()), 1e-6)
        + 0.15 * hf_norm
    ).astype(np.float32)

    if len(selected) < min(args.max_total, n):
        for i in [int(i) for i in np.argsort(-combined) if int(i) not in used]:
            if len(selected) >= min(args.max_total, n):
                break
            selected.append(i)
            bucket_of[i] = "hf_backfill"
            used.add(i)

    selected = selected[:min(args.max_total, len(selected))]
    print(f"TOTAL selected={len(selected)} / {n}")

    for old in out_dir.glob("*.pkl"):
        old.unlink()

    rows = []
    for rank, i in enumerate(selected):
        bucket = bucket_of.get(i, "unknown")
        src = f"hf_event_{bucket}_unit_{int(orig_indices[i]):06d}"
        payload = {
            "motion": units[i].astype(np.float32),
            "motion_151": units[i].astype(np.float32),
            "original_filename": src,
            "source_file": src,
            "source_id": src,
            "unit_index": int(orig_indices[i]),
            "export_rank": int(rank),
            "footwork_bucket": bucket,
            "hf_event_score": float(hf[i]),
            "hf_event_score_norm": float(hf_norm[i]),
            "metrics": {k: float(arrays[k][i]) for k in arrays if np.asarray(arrays[k]).shape == (n,)},
            "source_db": str(db_path),
            "unit_key": unit_key,
        }
        name = f"{rank:04d}_{bucket}_u{int(orig_indices[i]):06d}.pkl"
        with open(out_dir / name, "wb") as f:
            pickle.dump(payload, f)

        rows.append({
            "rank": rank,
            "file": name,
            "orig_unit": int(orig_indices[i]),
            "bucket": bucket,
            "hf_event_score": float(hf[i]),
            "hf_event_score_norm": float(hf_norm[i]),
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
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bucket_counts = {}
    for r in rows:
        bucket_counts[r["bucket"]] = bucket_counts.get(r["bucket"], 0) + 1

    summary = {
        "db": str(db_path),
        "out_dir": str(out_dir),
        "unit_key": unit_key,
        "seq_len": args.seq_len,
        "selected": len(selected),
        "hf_weight": args.hf_weight,
        "bucket_counts": bucket_counts,
        "manifest_csv": str(manifest_csv),
        "note": "HF-event boosted footwork-aware unit export for V3I training.",
    }
    (out_dir / "footwork_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ Export done")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
