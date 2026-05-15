import sys
import argparse
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

# 重要：pickle 里的 Normalizer / DummyNormalizer 可能引用 dataset 模块
import dataset.dance_dataset  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("--motion", required=True)
ap.add_argument("--normalizer", required=True)
ap.add_argument("--gt_phys", required=True)
ap.add_argument("--gt_norm", required=True)
ap.add_argument("--out_unnorm", required=True)
args = ap.parse_args()

m = np.load(args.motion).astype("float32")
gt_phys = np.load(args.gt_phys).astype("float32")
gt_norm = np.load(args.gt_norm).astype("float32")

with open(args.normalizer, "rb") as f:
    normalizer = pickle.load(f)

u = normalizer.unnormalize(torch.from_numpy(m[None])).numpy()[0].astype("float32")

Path(args.out_unnorm).parent.mkdir(parents=True, exist_ok=True)
np.save(args.out_unnorm, u)

n = min(len(m), len(gt_phys), len(gt_norm), len(u))

def root_path(x):
    return float(np.linalg.norm(np.diff(x[:n, [4, 6]], axis=0), axis=1).sum())

print("motion shape:", m.shape)
print("direct MSE output vs GT_NORM:", float(np.mean((m[:n] - gt_norm[:n]) ** 2)))
print("direct MSE output vs GT_PHYS:", float(np.mean((m[:n] - gt_phys[:n]) ** 2)))
print("unnorm MSE output vs GT_PHYS:", float(np.mean((u[:n] - gt_phys[:n]) ** 2)))
print("direct rot MSE vs GT_NORM:", float(np.mean((m[:n, 7:151] - gt_norm[:n, 7:151]) ** 2)))
print("unnorm rot MSE vs GT_PHYS:", float(np.mean((u[:n, 7:151] - gt_phys[:n, 7:151]) ** 2)))
print("direct root path:", root_path(m))
print("unnorm root path:", root_path(u))
print("GT_NORM root path:", root_path(gt_norm))
print("GT_PHYS root path:", root_path(gt_phys))
print("saved unnormalized:", args.out_unnorm)
