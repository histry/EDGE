import argparse, json, os
from pathlib import Path
import numpy as np

REPR_DIM = 151

def norm(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(x, [10, 90])
    if hi - lo < 1e-8:
        lo, hi = float(x.min()), float(x.max() + 1e-6)
    return np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1)

def load_npz(path):
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)

def get_field(*npzs, key, n, default=0.0):
    for z in npzs:
        if z is not None and key in z.files:
            arr = np.asarray(z[key])
            if arr.shape[0] >= n:
                return np.asarray(arr[:n], dtype=np.float32)
    return np.full((n,), float(default), dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag_db", required=True)
    ap.add_argument("--stats", default="")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--min_index_gap", type=int, default=3500)
    ap.add_argument("--source_max", type=int, default=1)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--env_out", required=True)
    ap.add_argument("--offset", type=int, default=0, help="skip top scored candidates to create alternative sets")
    args = ap.parse_args()

    rag = load_npz(args.rag_db)
    stats = load_npz(args.stats) if args.stats and Path(args.stats).exists() else None

    # 找 unit 数量
    unit_key = None
    units_arr = None
    for k in ["unit_motions", "unit_motions_physical", "motions", "motion_units", "clips", "units", "x"]:
        if k in rag.files:
            arr = np.asarray(rag[k])
            if arr.ndim == 3 and (arr.shape[-1] == REPR_DIM or arr.shape[1] == REPR_DIM):
                unit_key = k
                units_arr = arr
                break

    if units_arr is None:
        raise RuntimeError(f"No unit motion array found in {args.rag_db}. keys={rag.files}")

    n = units_arr.shape[0]
    idxs = np.arange(n)

    energy = get_field(stats, rag, key="unit_energy", n=n)
    if np.allclose(energy, 0):
        energy = get_field(stats, rag, key="motion_energy", n=n)

    upper = get_field(stats, rag, key="upper_activity", n=n)
    expr = get_field(stats, rag, key="expressiveness_score", n=n)
    pose_div = get_field(stats, rag, key="pose_diversity", n=n)
    turning = get_field(stats, rag, key="turning", n=n)
    root_speed = get_field(stats, rag, key="root_speed", n=n)

    score = (
        0.30 * norm(energy)
        + 0.30 * norm(upper)
        + 0.20 * norm(expr)
        + 0.12 * norm(pose_div)
        + 0.08 * norm(turning)
        - 0.05 * norm(root_speed)
    )

    source = None
    if "source" in rag.files and len(rag["source"]) >= n:
        source = np.asarray(rag["source"][:n]).astype(str)

    order = np.argsort(-score)
    if args.offset > 0:
        order = order[args.offset:]

    selected = []
    source_count = {}

    def ok_candidate(i, min_gap, source_max):
        if any(abs(int(i) - int(j)) < min_gap for j in selected):
            return False
        if source is not None and source_max > 0:
            s = str(source[i])
            if source_count.get(s, 0) >= source_max:
                return False
        return True

    # round 1: strict
    for i in order:
        if len(selected) >= args.count:
            break
        if ok_candidate(i, args.min_index_gap, args.source_max):
            selected.append(int(i))
            if source is not None:
                source_count[str(source[i])] = source_count.get(str(source[i]), 0) + 1

    # round 2: relax source, keep index gap
    if len(selected) < args.count:
        for i in order:
            if len(selected) >= args.count:
                break
            if int(i) in selected:
                continue
            if ok_candidate(i, args.min_index_gap, source_max=999):
                selected.append(int(i))

    # round 3: relax gap
    if len(selected) < args.count:
        relaxed_gap = max(500, args.min_index_gap // 2)
        for i in order:
            if len(selected) >= args.count:
                break
            if int(i) in selected:
                continue
            if all(abs(int(i) - int(j)) >= relaxed_gap for j in selected):
                selected.append(int(i))

    selected = selected[:args.count]

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "rag_db": args.rag_db,
        "stats": args.stats,
        "unit_key": unit_key,
        "count": args.count,
        "min_index_gap": args.min_index_gap,
        "source_max": args.source_max,
        "offset": args.offset,
        "selected_indices": selected,
        "selected_scores": [float(score[i]) for i in selected],
        "selected_energy": [float(energy[i]) for i in selected],
        "selected_upper": [float(upper[i]) for i in selected],
        "selected_expr": [float(expr[i]) for i in selected],
        "selected_source": [str(source[i]) if source is not None else "" for i in selected],
    }

    with open(args.out_prefix + "_selection.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    manual_units = ",".join(str(i) for i in selected)
    with open(args.env_out, "w", encoding="utf-8") as f:
        f.write(f'export EDGE_V10_MANUAL_UNITS="{manual_units}"\n')
        f.write(f'export EDGE_HARD_DEDUP_SELECTION_JSON="{args.out_prefix}_selection.json"\n')

    print("selected:", manual_units)
    print("saved:", args.out_prefix + "_selection.json")
    print("env:", args.env_out)

if __name__ == "__main__":
    main()
