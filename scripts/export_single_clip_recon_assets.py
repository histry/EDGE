import argparse
import pickle
from pathlib import Path
import numpy as np

ROOT_X = 4
ROOT_Z = 6

def iter_arrays(obj, prefix="obj"):
    if isinstance(obj, np.ndarray):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_arrays(v, f"{prefix}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from iter_arrays(v, f"{prefix}[{i}]")

def coerce_motion(arr):
    arr = np.asarray(arr)
    if arr.dtype == object:
        return None
    if arr.ndim == 2 and arr.shape[-1] == 151:
        return arr.astype(np.float32)
    if arr.ndim >= 3 and arr.shape[-1] == 151:
        # flatten all leading sequence dimensions except time and feature
        arr = arr.reshape(-1, arr.shape[-2], 151)
        return arr[0].astype(np.float32)
    if arr.ndim == 1 and arr.shape[0] == 151:
        return arr[None].astype(np.float32)
    return None

def load_first_motion(data_dir):
    data_dir = Path(data_dir)
    candidates = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in {".npy", ".npz", ".pkl", ".pickle"}:
            continue
        try:
            if path.suffix.lower() == ".npy":
                obj = np.load(path, allow_pickle=True)
                if obj.ndim == 0:
                    try:
                        obj = obj.item()
                    except Exception:
                        pass
                for name, arr in iter_arrays(obj, path.name):
                    mot = coerce_motion(arr)
                    if mot is not None:
                        candidates.append((path, name, mot))
            elif path.suffix.lower() == ".npz":
                z = np.load(path, allow_pickle=True)
                for k in z.files:
                    mot = coerce_motion(z[k])
                    if mot is not None:
                        candidates.append((path, k, mot))
            else:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                for name, arr in iter_arrays(obj, path.name):
                    mot = coerce_motion(arr)
                    if mot is not None:
                        candidates.append((path, name, mot))
        except Exception as exc:
            print(f"skip unreadable {path}: {exc}")

    if not candidates:
        raise RuntimeError(f"No [T,151] motion found under {data_dir}")

    # Prefer clips with at least 150 frames.
    candidates.sort(key=lambda x: (x[2].shape[0] < 150, -x[2].shape[0], str(x[0])))
    return candidates[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--keyframe_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=150)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    key_dir = Path(args.keyframe_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)

    src_path, src_key, motion = load_first_motion(args.data_dir)
    print(f"✅ selected motion source: {src_path} :: {src_key}, shape={motion.shape}")

    T = args.seq_len
    if motion.shape[0] >= T:
        clip = motion[:T].copy()
    else:
        pad = np.repeat(motion[-1:], T - motion.shape[0], axis=0)
        clip = np.concatenate([motion, pad], axis=0).astype(np.float32)

    rootlock = clip.copy()
    rootlock[:, ROOT_X] = rootlock[0, ROOT_X]
    rootlock[:, ROOT_Z] = rootlock[0, ROOT_Z]

    np.save(out_dir / "gt_clip.npy", clip.astype(np.float32))
    np.save(out_dir / "gt_rootlock.npy", rootlock.astype(np.float32))

    traj_gt = clip[:, [ROOT_X, ROOT_Z]].astype(np.float32)
    traj_static = np.repeat(traj_gt[:1], T, axis=0).astype(np.float32)
    np.save(out_dir / "gt_traj.npy", traj_gt)
    np.save(out_dir / "static_traj.npy", traj_static)

    frames = [0, 30, 60, 90, 120, T - 1]
    names = ["start", "mid_030", "mid_060", "mid_090", "mid_120", "end"]
    for name, f in zip(names, frames):
        np.save(key_dir / f"{name}.npy", clip[f].astype(np.float32))

    with open(out_dir / "asset_paths.env", "w", encoding="utf-8") as f:
        f.write(f"GT_CLIP={out_dir / 'gt_clip.npy'}\n")
        f.write(f"GT_ROOTLOCK={out_dir / 'gt_rootlock.npy'}\n")
        f.write(f"STATIC_TRAJ={out_dir / 'static_traj.npy'}\n")
        f.write(f"START_POSE={key_dir / 'start.npy'}\n")
        f.write(f"END_POSE={key_dir / 'end.npy'}\n")
        f.write(f"MID_POSES={key_dir / 'mid_030.npy'},{key_dir / 'mid_060.npy'},{key_dir / 'mid_090.npy'},{key_dir / 'mid_120.npy'}\n")
        f.write("MID_FRAMES=30,60,90,120\n")

    print("✅ exported:")
    print(f"  GT_CLIP={out_dir / 'gt_clip.npy'}")
    print(f"  GT_ROOTLOCK={out_dir / 'gt_rootlock.npy'}")
    print(f"  START={key_dir / 'start.npy'}")
    print(f"  END={key_dir / 'end.npy'}")
    print(f"  MID=30,60,90,120")
    print(f"  ENV={out_dir / 'asset_paths.env'}")

if __name__ == "__main__":
    main()
