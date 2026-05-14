import pickle
import json
import shutil
from pathlib import Path

import numpy as np

DB = "data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_mobility.npz"
WL = "data/dunhuang_choreo_unit_rag/stationary_expr_whitelist.txt"
OUT = Path("data/dunhuang_bvh/stationary_expr_subset_e150")
OUT.mkdir(parents=True, exist_ok=True)

WINDOW = 150

z = np.load(DB, allow_pickle=True)

indices = []
for line in Path(WL).read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        indices.append(int(line))

def is_temporal_array(x, min_len):
    if isinstance(x, np.ndarray) and x.ndim >= 1 and x.shape[0] >= min_len:
        return True
    return False

def infer_T(obj):
    cand = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                if v.shape[0] > 200:
                    cand.append(v.shape[0])
    if not cand:
        return None
    # 取最常见 / 最大时间维
    return max(cand)

def crop_obj(obj, start, end, T):
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T:
                new[k] = v[start:end].copy()
            else:
                new[k] = v
        return new
    raise TypeError(type(obj))

manifest = []

for n, idx in enumerate(indices, 1):
    src = str(z["source"][idx])
    frame = int(z["source_frame"][idx])

    src_path = Path(src)
    if not src_path.exists():
        # 有些 source 可能是相对 EDGE 根目录
        src_path = Path("/home/disk/lsm/storage/EDGE") / src

    if not src_path.exists():
        print("missing source:", src)
        continue

    with open(src_path, "rb") as f:
        obj = pickle.load(f)

    T = infer_T(obj)
    if T is None:
        print("cannot infer T:", src)
        continue

    start = max(0, frame - WINDOW // 2)
    if start + WINDOW > T:
        start = max(0, T - WINDOW)
    end = start + WINDOW

    if end - start < WINDOW:
        print("too short:", src, T, frame)
        continue

    cropped = crop_obj(obj, start, end, T)

    # 写入一些元信息，不影响原有 key
    if isinstance(cropped, dict):
        cropped["_stationary_subset_meta"] = {
            "rag_idx": int(idx),
            "source": src,
            "source_frame": int(frame),
            "crop_start": int(start),
            "crop_end": int(end),
            "original_T": int(T),
            "motion_text": str(z["motion_text"][idx]) if "motion_text" in z else "",
            "upper_activity": float(z["upper_activity"][idx]) if "upper_activity" in z else None,
            "motion_energy": float(z["motion_energy"][idx]) if "motion_energy" in z else None,
        }

    out_name = f"stationary_expr_{n:04d}_idx{idx}_{src_path.stem}_f{frame}.pkl"
    out_path = OUT / out_name

    with open(out_path, "wb") as f:
        pickle.dump(cropped, f)

    manifest.append({
        "idx": int(idx),
        "source": src,
        "source_frame": int(frame),
        "crop_start": int(start),
        "crop_end": int(end),
        "out": str(out_path),
        "motion_text": str(z["motion_text"][idx]) if "motion_text" in z else "",
    })

(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print("saved subset:", OUT)
print("num clips:", len(manifest))
for m in manifest:
    print(m)
