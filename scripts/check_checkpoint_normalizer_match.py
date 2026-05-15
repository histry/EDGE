import pickle
import numpy as np
import torch

CKPT = "runs/train_nextgen/strict_single_unit45_recon_v11_3000steps_b1/weights/train-3000.pt"
LOCAL_NORM = "output/single_unit45_recon_norm/normalizer.pkl"

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)

ckpt_norm = ckpt.get("normalizer", None)
if ckpt_norm is None:
    raise RuntimeError("checkpoint has no normalizer")

with open(LOCAL_NORM, "rb") as f:
    local_norm = pickle.load(f)

def get_mean_std(norm):
    if isinstance(norm, dict):
        return np.asarray(norm["mean"]), np.asarray(norm["std"])
    return np.asarray(norm.mean), np.asarray(norm.std)

cm, cs = get_mean_std(ckpt_norm)
lm, ls = get_mean_std(local_norm)

print("ckpt mean/std shape:", cm.shape, cs.shape)
print("local mean/std shape:", lm.shape, ls.shape)
print("mean MSE:", float(np.mean((cm - lm) ** 2)))
print("std  MSE:", float(np.mean((cs - ls) ** 2)))
print("mean max abs:", float(np.max(np.abs(cm - lm))))
print("std  max abs:", float(np.max(np.abs(cs - ls))))
