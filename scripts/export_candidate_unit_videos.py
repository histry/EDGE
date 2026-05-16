import argparse
import numpy as np
from pathlib import Path

def pick_motion_key(db):
    for k in ["unit_motions_physical", "unit_motions", "motions", "motion"]:
        if k in db:
            return k
    raise RuntimeError("No motion key found.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag_db", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--topk", type=int, default=80)
    args = ap.parse_args()

    db = np.load(args.rag_db, allow_pickle=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    motion_key = pick_motion_key(db)
    motions = np.asarray(db[motion_key])

    unit_ids = None
    for k in ["unit_ids", "ids", "unit_id"]:
        if k in db:
            unit_ids = np.asarray(db[k]).astype(int)
            break
    if unit_ids is None:
        unit_ids = np.arange(len(motions), dtype=int)

    # 尽量用已有 expressiveness / stationary 相关分数排序；没有就按前 topk。
    score = np.zeros(len(motions), dtype=np.float32)
    for k, w in [
        ("expressiveness_score", 1.0),
        ("upper_activity", 0.5),
        ("torso_activity", 0.5),
        ("mobile_score", -0.3),
        ("locomotion_score", -0.3),
    ]:
        if k in db:
            arr = np.asarray(db[k]).astype(np.float32)
            if arr.ndim > 1:
                arr = arr.reshape(len(arr), -1).mean(axis=1)
            score += w * arr[:len(score)]

    order = np.argsort(-score)[:args.topk]

    list_path = out / "candidate_units.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for rank, i in enumerate(order):
            uid = int(unit_ids[i])
            motion = np.asarray(motions[i], dtype=np.float32)
            if motion.ndim == 3:
                motion = motion[0]
            if motion.ndim != 2 or motion.shape[-1] != 151 or len(motion) < 45:
                continue
            motion = motion[:45]
            npy = out / f"rank{rank:03d}_unit{uid}.npy"
            np.save(npy, motion)
            f.write(f"{rank},{uid},{npy}\n")
            print(rank, uid, npy)

    print("candidate list:", list_path)

if __name__ == "__main__":
    main()
