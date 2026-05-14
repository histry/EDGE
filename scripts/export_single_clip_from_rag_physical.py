import argparse
from pathlib import Path
import numpy as np

ROOT_X = 4
ROOT_Z = 6

def pick_db():
    candidates = [
        "data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_mobility.npz",
        "data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco.npz",
        "data/dunhuang_choreo_unit_rag/index_u45_s15_e10.npz",
    ]
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError("No RAG DB found in default candidate paths.")

def valid_mid_frames(T):
    # Avoid frame 0 and T-1. Use quarter points.
    candidates = [
        int(round(T * 0.25)),
        int(round(T * 0.50)),
        int(round(T * 0.75)),
    ]
    out = []
    for f in candidates:
        f = max(1, min(T - 2, f))
        if f not in out:
            out.append(f)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--unit_id", type=int, default=55370)
    ap.add_argument("--out_dir", default="output/single_clip_recon")
    ap.add_argument("--keyframe_dir", default="test_keyframes/single_clip_recon")
    ap.add_argument("--seq_len", type=int, default=45)
    args = ap.parse_args()

    db_path = Path(args.db) if args.db else pick_db()
    out_dir = Path(args.out_dir)
    key_dir = Path(args.keyframe_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(db_path, allow_pickle=True)
    print("✅ DB:", db_path)
    print("keys:", z.files)

    if "unit_motions_physical" in z.files:
        arr = z["unit_motions_physical"]
        key = "unit_motions_physical"
    elif "unit_motions" in z.files:
        arr = z["unit_motions"]
        key = "unit_motions"
        print("⚠️ unit_motions_physical not found, fallback to unit_motions")
    else:
        raise RuntimeError("No unit_motions_physical or unit_motions in DB.")

    print("selected key:", key, "shape:", arr.shape)

    uid = args.unit_id
    if uid < 0 or uid >= arr.shape[0]:
        raise IndexError(f"unit_id {uid} out of range 0..{arr.shape[0]-1}")

    clip = np.asarray(arr[uid]).astype(np.float32)
    print("raw clip shape:", clip.shape)

    if clip.ndim != 2 or clip.shape[-1] != 151:
        raise RuntimeError(f"Selected unit is not [T,151], got {clip.shape}")

    T = args.seq_len
    if clip.shape[0] >= T:
        clipT = clip[:T].copy()
    else:
        pad = np.repeat(clip[-1:], T - clip.shape[0], axis=0)
        clipT = np.concatenate([clip, pad], axis=0).astype(np.float32)
        print(f"⚠️ padded clip from {clip.shape[0]} to {T}")

    rootlock = clipT.copy()
    rootlock[:, ROOT_X] = rootlock[0, ROOT_X]
    rootlock[:, ROOT_Z] = rootlock[0, ROOT_Z]

    traj = clipT[:, [ROOT_X, ROOT_Z]].astype(np.float32)
    static_traj = np.repeat(traj[:1], T, axis=0).astype(np.float32)

    np.save(out_dir / "gt_clip.npy", clipT)
    np.save(out_dir / "gt_rootlock.npy", rootlock)
    np.save(out_dir / "gt_traj.npy", traj)
    np.save(out_dir / "static_traj.npy", static_traj)

    start_path = key_dir / "start.npy"
    end_path = key_dir / "end.npy"
    np.save(start_path, clipT[0])
    np.save(end_path, clipT[T - 1])

    mids = valid_mid_frames(T)
    mid_paths = []
    for f in mids:
        p = key_dir / f"mid_{f:03d}.npy"
        np.save(p, clipT[f])
        mid_paths.append(p)

    with open(out_dir / "asset_paths.env", "w", encoding="utf-8") as f:
        f.write(f"GT_CLIP={out_dir/'gt_clip.npy'}\n")
        f.write(f"GT_ROOTLOCK={out_dir/'gt_rootlock.npy'}\n")
        f.write(f"GT_TRAJ={out_dir/'gt_traj.npy'}\n")
        f.write(f"STATIC_TRAJ={out_dir/'static_traj.npy'}\n")
        f.write(f"START_POSE={start_path}\n")
        f.write(f"END_POSE={end_path}\n")
        f.write("MID_POSES=" + ",".join(str(p) for p in mid_paths) + "\n")
        f.write("MID_FRAMES=" + ",".join(str(x) for x in mids) + "\n")
        f.write(f"RAG_DB={db_path}\n")
        f.write(f"RAG_KEY={key}\n")
        f.write(f"UNIT_ID={uid}\n")
        f.write(f"SEQ_LEN={T}\n")

    print("✅ exported GT/keyframes from RAG physical unit")
    print("GT:", out_dir / "gt_clip.npy")
    print("ROOTLOCK:", out_dir / "gt_rootlock.npy")
    print("START:", start_path)
    print("END:", end_path)
    print("MID_FRAMES:", mids)
    print("MID_POSES:", ",".join(str(p) for p in mid_paths))
    print("ENV:", out_dir / "asset_paths.env")

if __name__ == "__main__":
    main()
