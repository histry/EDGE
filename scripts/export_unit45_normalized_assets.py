import sys
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from dataset.dance_dataset import DunhuangDataset

DATA_DIR = "data/dunhuang_bvh/single_unit45_recon_physical"
GT_PATH = "output/single_unit45_recon/gt_clip.npy"

OUT_DIR = Path("output/single_unit45_recon_norm")
KF_DIR = Path("test_keyframes/single_unit45_recon_norm")
OUT_DIR.mkdir(parents=True, exist_ok=True)
KF_DIR.mkdir(parents=True, exist_ok=True)

gt = np.load(GT_PATH).astype("float32")

ds = DunhuangDataset(
    DATA_DIR,
    train=True,
    seq_len=45,
    audio_dim=803,
    return_traj=False,
    audio_pairing_mode="none",
)

x, cond, fn, wav = ds[0]
x = x.numpy().astype("float32")

ux = ds.normalizer.unnormalize(torch.from_numpy(x[None])).numpy()[0]

print("gt shape:", gt.shape)
print("dataset normalized x shape:", x.shape)
print("MSE normalized_x vs physical_gt:", float(np.mean((x - gt) ** 2)))
print("MSE unnormalized_x vs physical_gt:", float(np.mean((ux - gt) ** 2)))

assert gt.shape == (45, 151), gt.shape
assert x.shape == (45, 151), x.shape
assert np.mean((ux - gt) ** 2) < 1e-10

np.save(OUT_DIR / "gt_clip_normalized.npy", x.astype("float32"))
np.save(OUT_DIR / "gt_clip_physical.npy", gt.astype("float32"))

rootlock_norm = x.copy()
rootlock_norm[:, 4] = rootlock_norm[0, 4]
rootlock_norm[:, 6] = rootlock_norm[0, 6]
np.save(OUT_DIR / "gt_rootlock_normalized.npy", rootlock_norm.astype("float32"))

frames = [0, 11, 22, 34, 44]
names = ["start", "mid_011", "mid_022", "mid_034", "end"]

for name, frame in zip(names, frames):
    np.save(KF_DIR / f"{name}.npy", x[frame].astype("float32"))

with open(OUT_DIR / "normalizer.pkl", "wb") as f:
    pickle.dump(ds.normalizer, f)

with open(OUT_DIR / "asset_paths.env", "w", encoding="utf-8") as f:
    f.write(f"GT_NORM={OUT_DIR / 'gt_clip_normalized.npy'}\n")
    f.write(f"GT_PHYS={OUT_DIR / 'gt_clip_physical.npy'}\n")
    f.write(f"GT_ROOTLOCK_NORM={OUT_DIR / 'gt_rootlock_normalized.npy'}\n")
    f.write(f"NORMALIZER={OUT_DIR / 'normalizer.pkl'}\n")
    f.write(f"START_POSE={KF_DIR / 'start.npy'}\n")
    f.write(f"END_POSE={KF_DIR / 'end.npy'}\n")
    f.write(
        "MID_POSES="
        + ",".join([
            str(KF_DIR / "mid_011.npy"),
            str(KF_DIR / "mid_022.npy"),
            str(KF_DIR / "mid_034.npy"),
        ])
        + "\n"
    )
    f.write("MID_FRAMES=11,22,34\n")
    f.write("SEQ_LEN=45\n")

print("✅ exported normalized assets")
print("env:", OUT_DIR / "asset_paths.env")
print("start:", KF_DIR / "start.npy")
print("end:", KF_DIR / "end.npy")
print("mid:", KF_DIR / "mid_011.npy", KF_DIR / "mid_022.npy", KF_DIR / "mid_034.npy")
