import argparse
from pathlib import Path
import numpy as np

ROOT_X = 4
ROOT_Z = 6

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--freeze_y", action="store_true")
    args = ap.parse_args()

    x = np.load(args.input)
    y = np.array(x, copy=True)

    # 支持 [T,D], [1,T,D], [B,T,D]
    if y.ndim == 2:
        y[:, ROOT_X] = y[0, ROOT_X]
        y[:, ROOT_Z] = y[0, ROOT_Z]
        if args.freeze_y:
            y[:, 5] = y[0, 5]
    elif y.ndim == 3:
        y[:, :, ROOT_X] = y[:, :1, ROOT_X]
        y[:, :, ROOT_Z] = y[:, :1, ROOT_Z]
        if args.freeze_y:
            y[:, :, 5] = y[:, :1, 5]
    else:
        raise ValueError(f"Unsupported shape: {y.shape}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, y)
    print(f"saved: {args.output}, shape={y.shape}")

if __name__ == "__main__":
    main()
