import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import torch

try:
    from edge_recon_contract_patch import install_recon_contract_patch
    install_recon_contract_patch(verbose=True)
except Exception as exc:
    print("⚠️ recon patch install failed:", exc)

from EDGE import EDGE
from dataset.dance_dataset import DunhuangDataset

CKPT = "runs/train_nextgen/strict_single_unit45_recon_v14_x0w50_lr1e4_1000steps_b1/weights/train-1000.pt"
DATA_DIR = "data/dunhuang_bvh/single_unit45_recon_physical"
GT_PHYS = "output/single_unit45_recon/gt_clip.npy"

torch.manual_seed(1234)
np.random.seed(1234)


def run_replay(use_ema: bool):
    print("\n" + "=" * 80)
    print(f"REPLAY use_ema={use_ema}")
    print("=" * 80)

    model = EDGE(
        feature_type="hybrid",
        checkpoint_path=CKPT,
        EMA=use_ema,
        audio_dim=803,
        seq_len=45,
        mixed_precision="bf16",
        enable_rag_summary_token=True,
    )
    model.eval()

    device = model.accelerator.device
    diff = model.diffusion

    ds = DunhuangDataset(
        DATA_DIR,
        train=True,
        seq_len=45,
        audio_dim=803,
        normalizer=model.normalizer,
        return_traj=False,
        audio_pairing_mode="none",
    )

    x, cond, fn, wav = ds[0]
    x = x[None].to(device=device, dtype=torch.float32)

    cond2 = {}
    if isinstance(cond, dict):
        for k, v in cond.items():
            if k == "trajectory":
                continue
            cond2[k] = v[None].to(device=device, dtype=torch.float32) if torch.is_tensor(v) else v
    else:
        cond2 = cond[None].to(device=device, dtype=torch.float32)
    cond = model._maybe_attach_rag_summary(x, cond2, training=False)

    gt = np.load(GT_PHYS).astype("float32")
    with torch.no_grad():
        ux = model.normalizer.unnormalize(x).detach().cpu().numpy()[0]

    print("sample:", fn)
    print("MSE normalizer.unnormalize(dataset_x) vs GT_PHYS:", float(np.mean((ux - gt) ** 2)))
    print("cond keys:", sorted(cond.keys()) if isinstance(cond, dict) else type(cond))

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


run_replay(use_ema=True)
run_replay(use_ema=False)
