#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${EDGE_ENV_NAME:-edge}"
PYTHON_VERSION="${EDGE_PYTHON_VERSION:-3.9}"
WITH_JUKEBOX="${EDGE_WITH_JUKEBOX:-0}"
SKIP_PYTORCH3D="${EDGE_SKIP_PYTORCH3D:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash setup_edge_env.sh [--env-name NAME] [--with-jukebox] [--skip-pytorch3d]

Environment variables:
  EDGE_ENV_NAME          Conda env name, default: edge
  EDGE_PYTHON_VERSION    Python version, default: 3.9
  EDGE_WITH_JUKEBOX      Set to 1 to install optional legacy Jukebox dependencies
  EDGE_SKIP_PYTORCH3D    Set to 1 to skip PyTorch3D source install
  MAX_JOBS               Build parallelism for PyTorch3D, default: 4
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --with-jukebox)
      WITH_JUKEBOX=1
      shift
      ;;
    --skip-pytorch3d)
      SKIP_PYTORCH3D=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [ -n "${CONDA_EXE:-}" ]; then
  CONDA_BIN="$CONDA_EXE"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [ -x "/home/disk/lsm/miniconda3/bin/conda" ]; then
  CONDA_BIN="/home/disk/lsm/miniconda3/bin/conda"
else
  echo "conda not found. Install Miniconda/Anaconda or export CONDA_EXE=/path/to/conda." >&2
  exit 1
fi

CONDA_BASE="$("$CONDA_BIN" info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[EDGE env] Using existing conda env: $ENV_NAME"
else
  echo "[EDGE env] Creating conda env: $ENV_NAME"
  conda create -y -n "$ENV_NAME" -c conda-forge -c defaults \
    "python=$PYTHON_VERSION" pip ffmpeg libsndfile
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel

echo "[EDGE env] Installing PyTorch CUDA 12.8 wheels"
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

echo "[EDGE env] Installing core EDGE dependencies"
python -m pip install -r requirements-edge.txt

if [ "$SKIP_PYTORCH3D" != "1" ]; then
  if python - <<'PY'
from pytorch3d.transforms import axis_angle_to_matrix
print("PyTorch3D already importable")
PY
  then
    :
  else
    echo "[EDGE env] Installing PyTorch3D from the commit used by the verified env"
    MAX_JOBS="${MAX_JOBS:-4}" python -m pip install \
      "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@b6a77ad7aaf41ed90fca80ce6a2bac3c462a7881"
  fi
fi

if [ "$WITH_JUKEBOX" = "1" ]; then
  echo "[EDGE env] Installing optional legacy Jukebox dependencies"
  python -m pip install -r requirements-jukebox-optional.txt
fi

echo "[EDGE env] Running import health check"
python - <<'PY'
import importlib
import torch

required = [
    "accelerate",
    "wandb",
    "numpy",
    "scipy",
    "librosa",
    "soundfile",
    "matplotlib",
    "tqdm",
    "einops",
    "p_tqdm",
    "transformers",
    "cv2",
    "gradio",
    "fastdtw",
]

missing = []
for module_name in required:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        missing.append(f"{module_name}: {exc}")

try:
    from pytorch3d.transforms import axis_angle_to_matrix  # noqa: F401
except Exception as exc:
    missing.append(f"pytorch3d.transforms: {exc}")

print(f"torch={torch.__version__}, cuda_runtime={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")

if missing:
    print("Missing or broken modules:")
    for item in missing:
        print(f"  - {item}")
    raise SystemExit(1)

print("EDGE environment health check passed.")
PY

echo
echo "Done. Activate with:"
echo "  conda activate $ENV_NAME"
