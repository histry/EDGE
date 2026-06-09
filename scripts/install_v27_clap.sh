#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import sys
try:
    import torch
    print("[torch]", torch.__version__, "cuda=", torch.cuda.is_available())
except Exception as exc:
    raise SystemExit(f"[ERROR] PyTorch is not available in this environment: {exc}")
print("[python]", sys.version)
PY

python -m pip install -U pip setuptools wheel

# Keep the existing PyTorch installation intact.  LAION-CLAP declares torch in
# some dependency sets; --no-deps prevents pip from replacing the working CUDA
# build inside the EDGE conda environment.
python -m pip install --no-deps laion-clap

# Runtime dependencies commonly needed by LAION-CLAP inference.  These are kept
# separate from torch so the CUDA stack remains stable.
python -m pip install \
  librosa \
  soundfile \
  ftfy \
  regex \
  tqdm \
  braceexpand \
  webdataset \
  wget \
  h5py \
  pandas \
  scikit-learn \
  transformers \
  progressbar

python - <<'PY'
import importlib
for name in ["laion_clap", "librosa", "soundfile", "torch"]:
    module = importlib.import_module(name)
    print("[OK import]", name, getattr(module, "__version__", "unknown"))
PY

echo "[DONE] CLAP runtime installed. If strict CLAP still fails, set V27_CLAP_CKPT to a local LAION-CLAP checkpoint."
