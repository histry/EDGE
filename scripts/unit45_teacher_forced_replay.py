import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import os
import numpy as np
import torch

from EDGE import EDGE
from dataset.dance_dataset import DunhuangDataset

CKPT = "runs/train_nextgen/strict_single_unit45_recon_v11_3000steps_b1/weights/train-3000.pt"
DATA_DIR = "data/dunhuang_bvh/single_unit45_recon_physical"
GT_PHYS = "output/single_unit45_recon/gt_clip.npy"

torch.manual_seed(1234)
np.random.seed(1234)

model = EDGE(
    feature_type="hybrid",
    checkpoint_path=CKPT,
    EMA=True,
    audio_dim=803,
    seq_len=45,
    mixed_precision="bf16",
    enable_rag_summary_token=True,
)
model.eval()

device = model.accelerator.device
diff = model.diffusion
net = diff.model

ds = DunhuangDataset(
    DATA_DIR,
    train=True,
    seq_len=45,
    audio_dim=803,
    return_traj=False,
    audio_pairing_mode="none",
)

x, cond, fn, wav = ds[0]
x = x[None].to(device=device, dtype=torch.float32)

# Move cond to device and remove trajectory for this replay.
if isinstance(cond, dict):
    cond2 = {}
    for k, v in cond.items():
        if k == "trajectory":
            continue
        cond2[k] = v[None].to(device=device, dtype=torch.float32) if torch.is_tensor(v) else v
    cond = cond2
else:
    cond = cond[None].to(device=device, dtype=torch.float32)

print("sample:", fn)
print("x normalized shape:", tuple(x.shape))
print("x normalized mean/std:", float(x.mean()), float(x.std()))

# Verify normalizer unnormalizes dataset x back to GT physical.
gt = np.load(GT_PHYS).astype("float32")
with torch.no_grad():
    ux = model.normalizer.unnormalize(x).detach().cpu().numpy()[0]
print("MSE model.normalizer.unnormalize(dataset_x) vs GT_PHYS:", float(np.mean((ux - gt) ** 2)))

# Optional keyframe constraint exactly matching the training/inference setting.
frames = [0, 11, 22, 34, 44]
constraint_value = torch.zeros_like(x)
constraint_mask = torch.zeros_like(x)
for f in frames:
    constraint_value[:, f] = x[:, f]
    constraint_mask[:, f, :] = 1.0
constraint = {"value": constraint_value, "mask": constraint_mask}

timesteps = [0, 1, 5, 10, 25, 50, 100, 200, 500, 750, 999]

print("\n=== teacher-forced denoise replay ===")
with torch.no_grad():
    for tval in timesteps:
        t = torch.full((1,), tval, device=device, dtype=torch.long)
        noise = torch.randn_like(x)
        x_t = diff.q_sample(x_start=x, t=t, noise=noise)

        pred_noise, pred_x0 = diff.model_predictions(
            x_t,
            cond,
            t,
            clip_x_start=False,
            constraint=constraint,
        )

        mse_norm = torch.mean((pred_x0 - x) ** 2).item()
        rot_mse_norm = torch.mean((pred_x0[..., 7:151] - x[..., 7:151]) ** 2).item()
        kf_mse_norm = torch.mean((pred_x0[:, frames] - x[:, frames]) ** 2).item()

        pred_phys = model.normalizer.unnormalize(pred_x0).detach().cpu().numpy()[0]
        mse_phys = float(np.mean((pred_phys - gt) ** 2))
        rot_mse_phys = float(np.mean((pred_phys[:, 7:151] - gt[:, 7:151]) ** 2))

        print(
            f"t={tval:03d} | "
            f"norm_mse={mse_norm:.6f} rot_norm={rot_mse_norm:.6f} "
            f"kf_norm={kf_mse_norm:.6f} | "
            f"phys_mse={mse_phys:.6f} rot_phys={rot_mse_phys:.6f}"
        )
