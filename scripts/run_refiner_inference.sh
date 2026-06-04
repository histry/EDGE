#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}

INPUT_DIR=${INPUT_DIR:-output/night_v16d_refiner_latest/v16c_collected_npy}
OUT_DIR=${OUT_DIR:-output/night_v16d_refiner_latest/v16d_refined}
PHRASE_REFINER_MODEL=${PHRASE_REFINER_MODEL:-model/phrase_refiner.pt}
USE_ONSET=${USE_ONSET:-0}

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "V16D Phrase Refiner Inference"
echo "INPUT_DIR=$INPUT_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "PHRASE_REFINER_MODEL=$PHRASE_REFINER_MODEL"
echo "USE_ONSET=$USE_ONSET"
echo "============================================================"

if [ ! -d "$INPUT_DIR" ]; then
  echo "ERROR: INPUT_DIR not found: $INPUT_DIR"
  exit 1
fi

if [ ! -f "$PHRASE_REFINER_MODEL" ]; then
  echo "ERROR: PHRASE_REFINER_MODEL not found: $PHRASE_REFINER_MODEL"
  exit 2
fi

python - <<'PY'
from pathlib import Path
import os
import sys
import numpy as np
import torch

from model.phrase_refiner import PhraseRefiner


INPUT_DIR = Path(os.environ.get("INPUT_DIR", ""))
OUT_DIR = Path(os.environ.get("OUT_DIR", ""))
CKPT = Path(os.environ.get("PHRASE_REFINER_MODEL", "model/phrase_refiner.pt"))

OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_motion_shape(arr: np.ndarray, path: Path) -> np.ndarray:
    """
    Convert loaded npy motion into [B,T,151].

    Supported:
      [T,151]
      [1,T,151]
      [B,T,151]
      [T,1,151]
      [1,1,T,151]
      [B,1,T,151]
      [B,T,151,1]
      and other shapes that can be squeezed safely.
    """
    x = np.asarray(arr, dtype=np.float32)

    # Remove useless singleton dimensions first.
    x = np.squeeze(x)

    if x.ndim == 2:
        # [T,151]
        if x.shape[-1] == 151:
            x = x[None, :, :]
        # [151,T]
        elif x.shape[0] == 151:
            x = x.T[None, :, :]
        else:
            raise ValueError(f"{path}: unsupported 2D shape {arr.shape} -> squeezed {x.shape}")

    elif x.ndim == 3:
        # [B,T,151]
        if x.shape[-1] == 151:
            pass
        # [B,151,T]
        elif x.shape[1] == 151:
            x = np.transpose(x, (0, 2, 1))
        # [T,151,B] rare
        elif x.shape[1] != 151 and x.shape[-2] == 151:
            x = np.transpose(x, (2, 0, 1))
        else:
            raise ValueError(f"{path}: unsupported 3D shape {arr.shape} -> squeezed {x.shape}")

    else:
        raise ValueError(f"{path}: unsupported ndim {arr.ndim}, shape={arr.shape}, squeezed={x.shape}")

    if x.ndim != 3 or x.shape[-1] != 151:
        raise ValueError(f"{path}: normalize failed, got {x.shape}")

    return x.astype(np.float32)


def load_refiner(ckpt_path: Path):
    model = PhraseRefiner()
    obj = torch.load(str(ckpt_path), map_location="cpu")

    # Support both raw state_dict and checkpoint dict.
    if isinstance(obj, dict) and "model" in obj:
        state = obj["model"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj

    # Some checkpoints may have module. prefix.
    if isinstance(state, dict):
        fixed = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            fixed[nk] = v
        state = fixed

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] missing keys:", missing[:10])
    if unexpected:
        print("[WARN] unexpected keys:", unexpected[:10])

    model.eval()
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_refiner(CKPT).to(device)

files = sorted(INPUT_DIR.glob("*.npy"))
if not files:
    print(f"ERROR: no npy files found in {INPUT_DIR}")
    sys.exit(3)

print(f"Found {len(files)} npy files")

saved = 0

for f in files:
    try:
        raw = np.load(f)
        x_np = normalize_motion_shape(raw, f)
        x = torch.from_numpy(x_np).float().to(device)

        with torch.no_grad():
            y = model(x)

        if isinstance(y, (tuple, list)):
            y = y[0]

        y_np = y.detach().cpu().numpy().astype(np.float32)

        # Safety: ensure output is [B,T,151]
        y_np = normalize_motion_shape(y_np, f)

        # Keep root X/Z from input to avoid introducing drift.
        y_np[:, :, 4] = x_np[:, :, 4]
        y_np[:, :, 6] = x_np[:, :, 6]

        out = OUT_DIR / f"{f.stem}_v16d_refined.npy"
        np.save(out, y_np)

        print(f"[OK] {f.name}: raw={raw.shape} normalized={x_np.shape} saved={out.name} out={y_np.shape}")
        saved += 1

    except Exception as e:
        print(f"[FAIL] {f}: {repr(e)}", file=sys.stderr)

print(f"saved_count={saved}")
if saved == 0:
    sys.exit(4)
PY

echo "DONE: $OUT_DIR"
