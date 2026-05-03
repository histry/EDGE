import os
import csv
import glob
import pickle
import numpy as np
from pathlib import Path

def load_pkl(p):
    with open(p, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict) or "pos" not in d or "q" not in d:
        return None
    pos = np.asarray(d["pos"], dtype=np.float32)
    q = np.asarray(d["q"], dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3 or q.ndim != 2 or q.shape[1] != 72:
        return None
    return pos, q

def score_clip(pos, q, start, unit_len):
    end = start + unit_len
    pos_c = pos[start:end]
    q_c = q[start:end]

    root_speed = np.linalg.norm(np.diff(pos_c[:, [0, 2]], axis=0), axis=1).mean()
    root_range = np.linalg.norm(pos_c[-1, [0, 2]] - pos_c[0, [0, 2]])
    q_speed = np.linalg.norm(np.diff(q_c, axis=0), axis=1).mean()
    q_range = np.linalg.norm(q_c.max(axis=0) - q_c.min(axis=0))

    # 综合分：身体姿态变化为主，root 移动为辅
    score = 1.0 * q_speed + 0.15 * q_range + 0.4 * root_speed + 0.05 * root_range
    return score, root_speed, root_range, q_speed, q_range

def main():
    input_dir = "data/dunhuang_bvh/processed"
    out = "output/choreorag_final/report_assets/high_energy_units.csv"
    unit_len = 150
    stride = 30
    topk = 80

    rows = []
    for p in sorted(glob.glob(os.path.join(input_dir, "*.pkl"))):
        item = load_pkl(p)
        if item is None:
            continue
        pos, q = item
        T = min(len(pos), len(q))
        if T < unit_len:
            continue
        for s in range(0, T - unit_len + 1, stride):
            score, root_speed, root_range, q_speed, q_range = score_clip(pos, q, s, unit_len)
            rows.append({
                "source": p,
                "start": s,
                "center": s + unit_len // 2,
                "end": s + unit_len,
                "score": float(score),
                "root_speed": float(root_speed),
                "root_range": float(root_range),
                "q_speed": float(q_speed),
                "q_range": float(q_range),
            })

    rows = sorted(rows, key=lambda r: r["score"], reverse=True)[:topk]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("✅ saved:", out)
    for r in rows[:15]:
        print(r)

if __name__ == "__main__":
    main()
