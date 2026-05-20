#!/usr/bin/env python3
import argparse, csv, json, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from functional_choreo_metrics import functional_choreo_stats, add_functional_scores

def robust_norm(x):
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)

def frame_delta(m):
    if len(m) <= 1:
        return np.zeros((len(m),), dtype=np.float32)
    d = m[1:] - m[:-1]
    return np.sqrt(np.mean(d * d, axis=1)).astype(np.float32)

def tail_activity_ratio(m):
    d = frame_delta(m)
    if len(d) < 10:
        return 0.0
    first = float(d[:len(d)//2].mean()) + 1e-8
    second = float(d[len(d)//2:].mean())
    return float(second / first)

def jump_p95(m):
    d = frame_delta(m)
    return float(np.percentile(d, 95)) if len(d) else 0.0

def jerk_p95(m):
    if len(m) < 4:
        return 0.0
    j = m[3:] - 3*m[2:-1] + 3*m[1:-2] - m[:-3]
    v = np.sqrt(np.mean(j*j, axis=1))
    return float(np.percentile(v, 95))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out_sidecar", required=True)
    ap.add_argument("--out_pool", required=True)
    ap.add_argument("--seq_len", type=int, default=45)
    ap.add_argument("--top_k", type=int, default=120)
    ap.add_argument("--unit_key", default="")
    args = ap.parse_args()

    db_path = Path(args.db)
    out_sidecar = Path(args.out_sidecar)
    out_pool = Path(args.out_pool)
    out_sidecar.parent.mkdir(parents=True, exist_ok=True)
    out_pool.mkdir(parents=True, exist_ok=True)

    db = np.load(db_path, allow_pickle=True)
    keys = list(db.files)

    unit_key = args.unit_key
    if not unit_key:
        if "unit_motions_physical" in keys:
            unit_key = "unit_motions_physical"
        elif "unit_motions" in keys:
            unit_key = "unit_motions"
        else:
            raise KeyError("No unit_motions_physical / unit_motions found.")

    units = np.asarray(db[unit_key], dtype=np.float32)
    units = units[:, :args.seq_len].astype(np.float32)
    n = len(units)

    hf = np.asarray(db["hf_event_score"], dtype=np.float32) if "hf_event_score" in keys else np.zeros((n,), dtype=np.float32)
    hf_norm = robust_norm(hf)

    print(f"DB={db_path}")
    print(f"unit_key={unit_key}, units={n}, seq_len={args.seq_len}")
    print(f"hf_event_score min/mean/max={hf.min():.6f}/{hf.mean():.6f}/{hf.max():.6f}")

    stats_list = []
    tail = np.zeros((n,), dtype=np.float32)
    jump = np.zeros((n,), dtype=np.float32)
    jerk = np.zeros((n,), dtype=np.float32)

    for i, m in enumerate(units):
        if i % 5000 == 0:
            print(f"metrics {i}/{n}")
        stats_list.append(functional_choreo_stats(m))
        tail[i] = tail_activity_ratio(m)
        jump[i] = jump_p95(m)
        jerk[i] = jerk_p95(m)

    stat_keys = sorted(stats_list[0].keys())
    arrays = {}
    for k in stat_keys:
        arrays[k] = np.asarray([float(s.get(k, 0.0)) for s in stats_list], dtype=np.float32)
    arrays = add_functional_scores(arrays)

    root_path = arrays.get("root_path", np.zeros(n, dtype=np.float32))
    lower = arrays.get("lower_activity", np.zeros(n, dtype=np.float32))
    torso = arrays.get("torso_activity", np.zeros(n, dtype=np.float32))
    upper = arrays.get("upper_activity", np.zeros(n, dtype=np.float32))
    expr = arrays.get("expression_activity", np.zeros(n, dtype=np.float32))
    contact = arrays.get("contact_switch", np.zeros(n, dtype=np.float32))
    support_context = arrays.get("support_context_score", np.zeros(n, dtype=np.float32))
    coupling = arrays.get("functional_coupling_score", np.zeros(n, dtype=np.float32))
    mobile_expr = arrays.get("mobile_expressive_score", np.zeros(n, dtype=np.float32))

    root_lower_ratio = root_path / np.maximum(lower, 1e-8)

    lower_n = robust_norm(lower)
    torso_n = robust_norm(torso)
    upper_n = robust_norm(upper)
    expr_n = robust_norm(expr)
    contact_n = robust_norm(contact)
    tail_n = robust_norm(tail)
    jump_n = robust_norm(jump)
    jerk_n = robust_norm(jerk)
    root_lower_n = robust_norm(root_lower_ratio)

    tail_freeze = tail < 0.35
    root_drag = (root_lower_ratio > 2.5) & (contact < 0.02)
    upper_only = (lower < np.percentile(lower, 35)) & (upper > lower * 2.5)
    large_jump = jump > np.percentile(jump, 90)
    high_jerk = jerk > np.percentile(jerk, 90)

    penalty = (
        0.30 * tail_freeze.astype(np.float32)
        + 0.25 * root_drag.astype(np.float32)
        + 0.15 * upper_only.astype(np.float32)
        + 0.15 * large_jump.astype(np.float32)
        + 0.15 * high_jerk.astype(np.float32)
    )

    support_prior_quality = np.clip(
        0.24 * support_context
        + 0.20 * coupling
        + 0.16 * contact_n
        + 0.14 * lower_n
        + 0.10 * upper_n
        + 0.08 * expr_n
        + 0.08 * hf_norm
        + 0.08 * tail_n
        - penalty,
        0.0,
        1.0,
    ).astype(np.float32)

    good_gate = (
        (tail >= 0.35)
        & (~root_drag)
        & (~large_jump)
        & (~high_jerk)
        & ((contact > 0.0) | (lower_n > 0.35))
        & (support_prior_quality > np.percentile(support_prior_quality, 60))
    )

    np.savez(
        out_sidecar,
        source_db=str(db_path),
        unit_key=unit_key,
        support_prior_quality=support_prior_quality,
        good_gate=good_gate.astype(np.float32),
        hf_event_score=hf,
        hf_event_score_norm=hf_norm,
        tail_activity_ratio=tail,
        root_lower_ratio=root_lower_ratio,
        jump_p95=jump,
        jerk_p95=jerk,
        root_path=root_path,
        lower_activity=lower,
        torso_activity=torso,
        upper_activity=upper,
        expression_activity=expr,
        contact_switch=contact,
        support_context_score=support_context,
        functional_coupling_score=coupling,
        mobile_expressive_score=mobile_expr,
        tail_freeze=tail_freeze.astype(np.float32),
        root_drag=root_drag.astype(np.float32),
        upper_only=upper_only.astype(np.float32),
        large_jump=large_jump.astype(np.float32),
        high_jerk=high_jerk.astype(np.float32),
    )

    # 导出 top support prior pool，后续采样直接读 pkl
    for old in out_pool.glob("*.pkl"):
        old.unlink()

    candidate_idx = np.where(good_gate)[0]
    candidate_idx = sorted(candidate_idx, key=lambda i: float(support_prior_quality[i]), reverse=True)
    candidate_idx = candidate_idx[:args.top_k]

    rows = []
    for rank, i in enumerate(candidate_idx):
        name = f"{rank:04d}_supportq_u{i:06d}.pkl"
        payload = {
            "motion": units[i].astype(np.float32),
            "motion_151": units[i].astype(np.float32),
            "original_filename": f"supportq_u{i:06d}",
            "source_file": f"supportq_u{i:06d}",
            "source_id": f"supportq_u{i:06d}",
            "unit_index": int(i),
            "support_prior_quality": float(support_prior_quality[i]),
            "hf_event_score": float(hf[i]),
            "tail_activity_ratio": float(tail[i]),
            "root_lower_ratio": float(root_lower_ratio[i]),
            "jump_p95": float(jump[i]),
            "jerk_p95": float(jerk[i]),
            "metrics": {k: float(arrays[k][i]) for k in arrays if np.asarray(arrays[k]).shape == (n,)},
            "source_db": str(db_path),
            "unit_key": unit_key,
        }
        with open(out_pool / name, "wb") as f:
            pickle.dump(payload, f)

        rows.append({
            "rank": rank,
            "file": name,
            "unit_index": int(i),
            "support_prior_quality": float(support_prior_quality[i]),
            "hf_event_score": float(hf[i]),
            "tail_activity_ratio": float(tail[i]),
            "root_lower_ratio": float(root_lower_ratio[i]),
            "root_path": float(root_path[i]),
            "lower_activity": float(lower[i]),
            "torso_activity": float(torso[i]),
            "upper_activity": float(upper[i]),
            "contact_switch": float(contact[i]),
            "jump_p95": float(jump[i]),
            "jerk_p95": float(jerk[i]),
        })

    with open(out_pool / "support_quality_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "source_db": str(db_path),
        "out_sidecar": str(out_sidecar),
        "out_pool": str(out_pool),
        "unit_key": unit_key,
        "units": n,
        "good_gate_count": int(good_gate.sum()),
        "top_k_exported": len(rows),
        "note": "Support-prior quality gate sidecar and top prior pool for RAG rerank.",
    }
    (out_pool / "support_quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Done")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
