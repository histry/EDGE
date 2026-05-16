import argparse
import csv
import numpy as np
from pathlib import Path

ROOT_X = 4
ROOT_Z = 6
ROT_START = 7
ROT_DIM = 6

TORSO_JOINTS = [3, 6, 9]
UPPER_JOINTS = [12,13,14,15,16,17,18,19,20,21,22,23]
LOWER_JOINTS = [1,2,4,5,7,8,10,11]

def joint_idx(joints):
    idx = []
    for j in joints:
        idx.extend(range(ROT_START + ROT_DIM*j, ROT_START + ROT_DIM*(j+1)))
    return np.array(idx, dtype=np.int64)

TORSO = joint_idx(TORSO_JOINTS)
UPPER = joint_idx(UPPER_JOINTS)
LOWER = joint_idx(LOWER_JOINTS)
BODY = np.concatenate([TORSO, UPPER])

def pick_key(db, names):
    for k in names:
        if k in db:
            return k
    return None

def robust_norm(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, [10, 90])
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)

def metrics(m):
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 3:
        m = m[0]
    m = m[:45]

    d = m[1:] - m[:-1]

    body_frame = np.sqrt(np.mean(d[:, BODY] ** 2, axis=1))
    upper_frame = np.sqrt(np.mean(d[:, UPPER] ** 2, axis=1))
    torso_frame = np.sqrt(np.mean(d[:, TORSO] ** 2, axis=1))
    lower_frame = np.sqrt(np.mean(d[:, LOWER] ** 2, axis=1))

    root = m[:, [ROOT_X, ROOT_Z]]
    root_range = float(np.linalg.norm(root.max(axis=0) - root.min(axis=0)))

    # 用自适应阈值判断 active，不被绝对尺度影响
    med = float(np.median(body_frame))
    p75 = float(np.percentile(body_frame, 75))
    thr = max(med + 0.25 * (p75 - med), 1e-5)

    active = body_frame > thr
    active_ratio = float(active.mean())

    early = body_frame[:8]
    mid = body_frame[8:30]
    late = body_frame[30:]

    early_energy = float(early.sum())
    mid_energy = float(mid.sum())
    late_energy = float(late.sum())
    total_energy = float(body_frame.sum() + 1e-8)

    early_ratio = early_energy / total_energy
    mid_ratio = mid_energy / total_energy
    late_ratio = late_energy / total_energy

    # 如果是“前面动一下后面不动”，late_ratio 和 late_active 会很低
    late_active_ratio = float((late > thr).mean())
    mid_active_ratio = float((mid > thr).mean())

    jump_max = float(body_frame.max())
    jump_mean = float(body_frame.mean() + 1e-8)
    jump_spike = jump_max / jump_mean

    if len(body_frame) >= 3:
        acc = body_frame[2:] - 2 * body_frame[1:-1] + body_frame[:-2]
        jerk_mean = float(np.mean(np.abs(acc)))
    else:
        jerk_mean = 0.0

    upper_mean = float(upper_frame.mean())
    torso_mean = float(torso_frame.mean())
    lower_mean = float(lower_frame.mean())
    body_mean = float(body_frame.mean())

    return {
        "body_mean": body_mean,
        "upper_mean": upper_mean,
        "torso_mean": torso_mean,
        "lower_mean": lower_mean,
        "root_range": root_range,
        "active_ratio": active_ratio,
        "mid_active_ratio": mid_active_ratio,
        "late_active_ratio": late_active_ratio,
        "early_ratio": early_ratio,
        "mid_ratio": mid_ratio,
        "late_ratio": late_ratio,
        "jump_spike": jump_spike,
        "jerk_mean": jerk_mean,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag_db", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--topk", type=int, default=120)
    ap.add_argument("--max_root_range", type=float, default=0.12)
    ap.add_argument("--min_active_ratio", type=float, default=0.30)
    ap.add_argument("--min_mid_active", type=float, default=0.25)
    ap.add_argument("--min_late_active", type=float, default=0.20)
    ap.add_argument("--max_early_ratio", type=float, default=0.45)
    ap.add_argument("--min_late_ratio", type=float, default=0.15)
    ap.add_argument("--max_jump_spike", type=float, default=6.0)
    ap.add_argument("--min_id_gap", type=int, default=12)
    args = ap.parse_args()

    db = np.load(args.rag_db, allow_pickle=True)
    motion_key = pick_key(db, ["unit_motions_physical", "unit_motions", "motions", "motion"])
    if motion_key is None:
        raise RuntimeError("No motion key found")

    motions = np.asarray(db[motion_key])
    unit_ids = np.arange(len(motions), dtype=int)

    rows = []
    for i in range(len(motions)):
        m = np.asarray(motions[i], dtype=np.float32)
        if m.ndim == 3:
            m = m[0]
        if m.ndim != 2 or m.shape[-1] != 151 or len(m) < 45:
            continue
        mm = metrics(m)

        # hard filter: 去掉前动后静 / padding / 单帧突变 / 大 root 平移
        if mm["root_range"] > args.max_root_range:
            continue
        if mm["active_ratio"] < args.min_active_ratio:
            continue
        if mm["mid_active_ratio"] < args.min_mid_active:
            continue
        if mm["late_active_ratio"] < args.min_late_active:
            continue
        if mm["early_ratio"] > args.max_early_ratio:
            continue
        if mm["late_ratio"] < args.min_late_ratio:
            continue
        if mm["jump_spike"] > args.max_jump_spike:
            continue

        # continuous expression score
        score = (
            2.0 * mm["upper_mean"]
            + 1.5 * mm["torso_mean"]
            + 0.8 * mm["active_ratio"]
            + 0.8 * mm["mid_active_ratio"]
            + 1.0 * mm["late_active_ratio"]
            + 0.6 * mm["late_ratio"]
            - 0.8 * mm["early_ratio"]
            - 0.5 * mm["root_range"]
            - 0.2 * mm["jerk_mean"]
        )

        rows.append((score, i, mm))

    rows.sort(key=lambda x: x[0], reverse=True)

    # 去重：避免选到相邻滑窗
    selected = []
    selected_ids = []
    for score, i, mm in rows:
        if any(abs(i - j) < args.min_id_gap for j in selected_ids):
            continue
        selected.append((score, i, mm))
        selected_ids.append(i)
        if len(selected) >= args.topk:
            break

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "continuous_stationary_candidates.csv"
    list_path = out / "candidate_units.txt"

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv, open(list_path, "w", encoding="utf-8") as flist:
        fieldnames = [
            "rank","unit_id","score",
            "body_mean","upper_mean","torso_mean","lower_mean",
            "root_range","active_ratio","mid_active_ratio","late_active_ratio",
            "early_ratio","mid_ratio","late_ratio","jump_spike","jerk_mean","npy"
        ]
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for rank, (score, uid, mm) in enumerate(selected):
            m = np.asarray(motions[uid], dtype=np.float32)
            if m.ndim == 3:
                m = m[0]
            m = m[:45]
            npy = out / f"rank{rank:03d}_unit{uid}.npy"
            np.save(npy, m)

            row = {"rank": rank, "unit_id": uid, "score": score, "npy": str(npy)}
            row.update(mm)
            writer.writerow(row)
            flist.write(f"{rank},{uid},{npy}\n")
            print(rank, uid, "score", score, "late_active", mm["late_active_ratio"], "early_ratio", mm["early_ratio"], "root", mm["root_range"])

    print("saved:", csv_path)
    print("saved:", list_path)
    print("selected:", len(selected))

if __name__ == "__main__":
    main()
