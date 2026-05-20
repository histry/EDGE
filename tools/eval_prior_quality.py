#!/usr/bin/env python3
import csv, json, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from functional_choreo_metrics import functional_choreo_stats

def load_motion(path):
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p, allow_pickle=True)
    else:
        obj = pickle.load(open(p, "rb"))
        arr = obj.get("motion_151", obj.get("motion"))
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    return arr[:45]

def frame_delta(m):
    if len(m) <= 1:
        return np.zeros((len(m),), dtype=np.float32)
    d = m[1:] - m[:-1]
    return np.sqrt(np.mean(d*d, axis=1)).astype(np.float32)

def jump_p95(m):
    d = frame_delta(m)
    return float(np.percentile(d, 95)) if len(d) else 0.0

def jerk_p95(m):
    if len(m) < 4:
        return 0.0
    j = m[3:] - 3*m[2:-1] + 3*m[1:-2] - m[:-3]
    v = np.sqrt(np.mean(j*j, axis=1))
    return float(np.percentile(v, 95))

def tail_activity_ratio(m):
    d = frame_delta(m)
    if len(d) < 10:
        return 0.0
    first = float(d[:len(d)//2].mean()) + 1e-8
    second = float(d[len(d)//2:].mean())
    return float(second / first)

def main():
    manifest = Path("data/dunhuang_bvh/footwork_hfevent_v3i_u45/footwork_manifest.csv")
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))

    # 当前人工观察的五个 prior
    targets = [
        "0480_hf_fullbody_coupled_u045350.pkl",
        "0360_hf_mobile_step_u067710.pkl",
        "0266_hf_turn_pivot_u004889.pkl",
        "0120_hf_support_shift_u032781.pkl",
        "0000_hf_stationary_expr_u034029.pkl",
    ]

    out_rows = []
    for name in targets:
        path = Path("data/dunhuang_bvh/footwork_hfevent_v3i_u45") / name
        m = load_motion(path)
        s = functional_choreo_stats(m)

        root_path = float(s.get("root_path", 0.0))
        lower = float(s.get("lower_activity", 0.0))
        upper = float(s.get("upper_activity", 0.0))
        torso = float(s.get("torso_activity", 0.0))
        contact = float(s.get("contact_switch", 0.0))

        row = {
            "file": name,
            "root_path": root_path,
            "lower_activity": lower,
            "torso_activity": torso,
            "upper_activity": upper,
            "expression_activity": float(s.get("expression_activity", 0.0)),
            "contact_switch": contact,
            "tail_activity_ratio": tail_activity_ratio(m),
            "root_lower_ratio": root_path / max(lower, 1e-8),
            "jump_p95": jump_p95(m),
            "jerk_p95": jerk_p95(m),
        }

        # 简单规则标记
        flags = []
        if row["tail_activity_ratio"] < 0.35:
            flags.append("tail_freeze")
        if row["root_lower_ratio"] > 2.5 and contact < 0.02:
            flags.append("root_drag_without_support")
        if lower < 0.002 and upper > lower * 2.5:
            flags.append("upper_only")
        if row["jump_p95"] > 0.08:
            flags.append("large_jump")
        row["flags"] = ";".join(flags) if flags else "ok"

        out_rows.append(row)

    out_csv = Path("output/v3i_prior_debug/prior_quality.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(out_rows[0].keys())

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out_rows)

    Path("output/v3i_prior_debug/prior_quality.json").write_text(
        json.dumps(out_rows, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("✅ wrote", out_csv)
    for r in out_rows:
        print(
            r["file"],
            "root", f'{r["root_path"]:.5f}',
            "lower", f'{r["lower_activity"]:.5f}',
            "upper", f'{r["upper_activity"]:.5f}',
            "contact", f'{r["contact_switch"]:.5f}',
            "tail", f'{r["tail_activity_ratio"]:.3f}',
            "root/lower", f'{r["root_lower_ratio"]:.2f}',
            "flags", r["flags"],
        )

if __name__ == "__main__":
    main()
