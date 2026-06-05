#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/disk/lsm/storage/EDGE
cd "$ROOT"

export PATH="/home/disk/lsm/conda_envs/edge/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN="${V22_OVERNIGHT_ROOT:?V22_OVERNIGHT_ROOT is required}"
DATA="${V22_TURN_DATASET:-data/v22_turn_pace_dataset.npz}"
TRAIN_OUT="${V22_TURN_TRAIN_OUT:-$RUN/train}"

mkdir -p "$RUN" "$TRAIN_OUT"

exec > >(tee -a "$RUN/overnight.log") 2>&1

on_error() {
    code=$?

    echo
    echo "============================================================"
    echo "[V22 ERROR]"
    echo "exit_code=$code"
    echo "line=${BASH_LINENO[0]:-unknown}"
    echo "command=${BASH_COMMAND:-unknown}"
    echo "run=$RUN"
    echo "============================================================"

    exit "$code"
}

trap on_error ERR

export V22_TURN_DATASET="$DATA"
export V22_TURN_TRAIN_OUT="$TRAIN_OUT"

export V22_MOTION_GLOB="${V22_MOTION_GLOB:-data/dunhuang_151d_physical/*.npy}"
export V22_DATA_MAX_SAMPLES="${V22_DATA_MAX_SAMPLES:-12000}"

export V22_TURN_EPOCHS="${V22_TURN_EPOCHS:-320}"
export V22_TURN_BATCH_SIZE="${V22_TURN_BATCH_SIZE:-96}"

echo "============================================================"
echo "V22 Stage 2/3 resume"
echo "root=$ROOT"
echo "run=$RUN"
echo "dataset=$DATA"
echo "train_out=$TRAIN_OUT"
echo "motion_glob=$V22_MOTION_GLOB"
echo "epochs=$V22_TURN_EPOCHS"
echo "batch_size=$V22_TURN_BATCH_SIZE"
echo "python=$(command -v python)"
echo "============================================================"

required_files=(
    "tools/build_v22_turn_pace_dataset.py"
    "train_v22_turn_pace.py"
    "model/v22_turn_pace.py"
    "scripts/run_v22_build_turn_dataset.sh"
    "scripts/run_v22_train_turn_pace.sh"
    "data/dunhuang_dynamic_event_rag_physical/v22_turn_aware_event_index.json"
    "data/dunhuang_dynamic_event_rag_physical/v22_turn_aware_event_index.npz"
)

for path in "${required_files[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "[ERROR] missing required file: $path"
        exit 2
    fi
done

python -m py_compile \
    tools/build_v22_turn_pace_dataset.py \
    train_v22_turn_pace.py \
    model/v22_turn_pace.py

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_devices:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

dataset_is_valid=0

if [[ -s "$DATA" ]]; then
    echo
    echo "[CHECK] Existing dataset found: $DATA"

    if python - "$DATA" <<'PY'
import sys
import numpy as np

path = sys.argv[1]

with np.load(path, allow_pickle=True) as data:
    required = (
        "corrupted",
        "target",
        "edit_mask",
        "condition",
        "source_id",
    )

    for key in required:
        if key not in data.files:
            raise RuntimeError(f"missing array: {key}")

    corrupted = np.asarray(data["corrupted"])
    target = np.asarray(data["target"])
    mask = np.asarray(data["edit_mask"])
    condition = np.asarray(data["condition"])
    source_id = np.asarray(data["source_id"])

    if corrupted.ndim != 3 or corrupted.shape[-1] != 151:
        raise RuntimeError(
            f"invalid corrupted shape: {corrupted.shape}"
        )

    if corrupted.shape != target.shape:
        raise RuntimeError(
            f"corrupted/target mismatch: "
            f"{corrupted.shape} vs {target.shape}"
        )

    if mask.shape != corrupted.shape[:2]:
        raise RuntimeError(
            f"invalid mask: {mask.shape}"
        )

    n = len(corrupted)

    if len(condition) != n or len(source_id) != n:
        raise RuntimeError("dataset length mismatch")

    if n < 300:
        raise RuntimeError(
            f"too few samples: {n}"
        )

    for key in (
        "corrupted",
        "target",
        "edit_mask",
        "condition",
    ):
        if not np.isfinite(data[key]).all():
            raise RuntimeError(
                f"{key} contains NaN or Inf"
            )

    print("existing dataset is valid")
    print("samples:", n)
    print("motion shape:", corrupted.shape)
    print("condition shape:", condition.shape)
    print("sources:", len(np.unique(source_id)))
PY
    then
        dataset_is_valid=1
    else
        echo "[WARNING] Existing dataset is incomplete or invalid."
        dataset_is_valid=0
    fi
fi

if [[ "$dataset_is_valid" != "1" ]]; then
    echo
    echo "============================================================"
    echo "Stage 2/3: build turn-pace dataset"
    echo "============================================================"

    rm -f "$DATA"
    rm -f "${DATA%.npz}.metadata.json"

    bash scripts/run_v22_build_turn_dataset.sh
fi

echo
echo "============================================================"
echo "Validate newly built dataset"
echo "============================================================"

python - "$DATA" <<'PY'
import sys
from collections import Counter

import numpy as np

path = sys.argv[1]

with np.load(path, allow_pickle=True) as data:
    corrupted = np.asarray(
        data["corrupted"],
        dtype=np.float32,
    )

    target = np.asarray(
        data["target"],
        dtype=np.float32,
    )

    mask = np.asarray(
        data["edit_mask"],
        dtype=np.float32,
    )

    condition = np.asarray(
        data["condition"],
        dtype=np.float32,
    )

    source_id = np.asarray(
        data["source_id"],
        dtype=np.int32,
    )

    target_peak = np.asarray(
        data["target_peak_dps"],
        dtype=np.float32,
    )

    corrupted_peak = np.asarray(
        data["corrupted_peak_dps"],
        dtype=np.float32,
    )

    speed_factor = np.asarray(
        data["speed_factor"],
        dtype=np.float32,
    )

print("dataset:", path)
print("samples:", len(corrupted))
print("motion:", corrupted.shape)
print("mask:", mask.shape)
print("condition:", condition.shape)
print("sources:", len(np.unique(source_id)))

print(
    "target peak p10/p50/p90/max:",
    np.percentile(
        target_peak,
        [10, 50, 90, 100],
    ),
)

print(
    "corrupted peak p10/p50/p90/max:",
    np.percentile(
        corrupted_peak,
        [10, 50, 90, 100],
    ),
)

print(
    "speed factor p10/p50/p90/max:",
    np.percentile(
        speed_factor,
        [10, 50, 90, 100],
    ),
)

assert corrupted.shape == target.shape
assert corrupted.ndim == 3
assert corrupted.shape[1] == 72
assert corrupted.shape[2] == 151
assert mask.shape == corrupted.shape[:2]
assert len(corrupted) >= 300
assert len(np.unique(source_id)) >= 2

for array in (
    corrupted,
    target,
    mask,
    condition,
):
    assert np.isfinite(array).all()

print("[PASS] V22 dataset is healthy.")
PY

echo
echo "============================================================"
echo "Stage 3/3: train turn-pace refiner"
echo "============================================================"

bash scripts/run_v22_train_turn_pace.sh

BEST_FILE="$TRAIN_OUT/BEST_TURN_PACE_CKPT.txt"

if [[ ! -s "$BEST_FILE" ]]; then
    echo "[ERROR] missing: $BEST_FILE"
    exit 5
fi

BEST_CKPT="$(cat "$BEST_FILE")"

if [[ ! -s "$BEST_CKPT" ]]; then
    echo "[ERROR] missing checkpoint: $BEST_CKPT"
    exit 5
fi

cp "$BEST_FILE" \
   "$RUN/BEST_TURN_PACE_CKPT.txt"

echo
echo "============================================================"
echo "V22 Stage 2/3 finished"
echo "run=$RUN"
echo "dataset=$DATA"
echo "best_checkpoint=$BEST_CKPT"
echo "============================================================"
