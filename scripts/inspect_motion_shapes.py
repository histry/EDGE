import pickle
from pathlib import Path
import numpy as np

ROOT = Path("data/dunhuang_bvh/stationary_expr_subset_e150")

def walk(obj, name="obj", depth=0):
    if depth > 5:
        return
    if isinstance(obj, np.ndarray):
        print(f"{name}: ndarray shape={obj.shape} dtype={obj.dtype}")
    elif isinstance(obj, dict):
        print(f"{name}: dict keys={list(obj.keys())[:30]}")
        for k, v in obj.items():
            walk(v, f"{name}.{k}", depth + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{name}: {type(obj).__name__} len={len(obj)}")
        for i, v in enumerate(obj[:10]):
            walk(v, f"{name}[{i}]", depth + 1)

print("ROOT =", ROOT.resolve())
print("exists =", ROOT.exists())

files = sorted([p for p in ROOT.rglob("*") if p.is_file()])
print("num_files =", len(files))
for p in files[:200]:
    print("\nFILE", p, "size", p.stat().st_size)
    try:
        if p.suffix == ".npy":
            obj = np.load(p, allow_pickle=True)
            if obj.ndim == 0:
                try:
                    obj = obj.item()
                except Exception:
                    pass
            walk(obj, p.name)
        elif p.suffix == ".npz":
            z = np.load(p, allow_pickle=True)
            print("npz keys:", z.files)
            for k in z.files:
                walk(z[k], f"{p.name}.{k}")
        elif p.suffix in [".pkl", ".pickle"]:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            walk(obj, p.name)
        else:
            print("skip suffix", p.suffix)
    except Exception as e:
        print("ERROR:", repr(e))
