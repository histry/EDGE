import argparse
import csv
import pickle
import numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag_db", required=True)
    ap.add_argument("--whitelist_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    db = np.load(args.rag_db, allow_pickle=True)
    out = Path(args.out_dir) / "processed"
    out.mkdir(parents=True, exist_ok=True)

    ids = []
    with open(args.whitelist_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(int(row["unit_id"]))

    motions = None
    for k in ["unit_motions_physical", "unit_motions", "motions", "motion"]:
        if k in db:
            motions = np.asarray(db[k])
            print("using motion key:", k, motions.shape)
            break
    if motions is None:
        raise RuntimeError("No unit motion array found.")

    unit_ids = None
    for k in ["unit_ids", "ids", "unit_id"]:
        if k in db:
            unit_ids = np.asarray(db[k]).astype(int)
            print("using id key:", k)
            break
    if unit_ids is None:
        print("No explicit unit_ids found; treating row index as unit_id.")
        unit_ids = np.arange(len(motions), dtype=int)

    id_to_idx = {int(uid): i for i, uid in enumerate(unit_ids)}

    saved = []
    for uid in ids:
        if uid not in id_to_idx:
            print("missing unit_id:", uid)
            continue

        i = id_to_idx[uid]
        motion = np.asarray(motions[i], dtype=np.float32)
        if motion.ndim == 3:
            motion = motion[0]
        if motion.ndim != 2 or motion.shape[-1] != 151:
            print("skip invalid:", uid, motion.shape)
            continue
        if len(motion) < 45:
            print("skip short:", uid, motion.shape)
            continue

        motion = motion[:45].astype(np.float32)

        path = out / f"unit_{uid}_gt45.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "motion": motion,
                "motion_151": motion,
                "source": f"stationary_whitelist_unit_{uid}",
                "unit_id": uid,
            }, f)

        saved.append(str(path))
        print("saved:", path, motion.shape)

    print("total saved:", len(saved))

if __name__ == "__main__":
    main()
