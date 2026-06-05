#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

MUSIC_DIR="${V21_ROUTER_MUSIC_DIR:-data/v21_router_music_999/splits/train}"
FEATURE_DIR="${V21_ROUTER_FEATURE_DIR:-data/v21_music_features_router_985_train}"
INDEX_PREFIX="${V21_INDEX_PREFIX:-data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index}"
RUN_ROOT="${V21_ROUTER_RUN_ROOT:-output/v21_music_router_985songs_$(date +%Y%m%d_%H%M%S)}"

NUM_FRAMES="${V21_ROUTER_NUM_FRAMES:-150}"
PHRASES="${V21_ROUTER_PHRASES:-3}"

EPOCHS="${V21_ROUTER_EPOCHS:-180}"
BATCH_SIZE="${V21_ROUTER_BATCH_SIZE:-512}"
LR="${V21_ROUTER_LR:-1e-4}"
WEIGHT_DECAY="${V21_ROUTER_WEIGHT_DECAY:-1e-4}"
HIDDEN_DIM="${V21_ROUTER_HIDDEN:-192}"
LATENT_DIM="${V21_ROUTER_LATENT:-96}"
DROPOUT="${V21_ROUTER_DROPOUT:-0.15}"
MARGIN_WEIGHT="${V21_ROUTER_MARGIN_WEIGHT:-0.80}"

POSITIVES="${V21_ROUTER_POSITIVES:-6}"
NEGATIVES="${V21_ROUTER_NEGATIVES:-4}"

FORCE_REEXTRACT="${V21_FORCE_REEXTRACT:-0}"

mkdir -p "$RUN_ROOT"
mkdir -p "$FEATURE_DIR"

exec > >(tee -a "$RUN_ROOT/overnight.log") 2>&1

on_error() {
    local code=$?
    echo
    echo "============================================================"
    echo "[V21 ROUTER ERROR]"
    echo "exit_code=$code"
    echo "line=${BASH_LINENO[0]:-unknown}"
    echo "command=${BASH_COMMAND:-unknown}"
    echo "run_root=$RUN_ROOT"
    echo "============================================================"
    exit "$code"
}

trap on_error ERR

echo "============================================================"
echo "V21 calibrated Music-Motion Router training"
echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
echo "music_dir=$MUSIC_DIR"
echo "feature_dir=$FEATURE_DIR"
echo "index_prefix=$INDEX_PREFIX"
echo "run_root=$RUN_ROOT"
echo "epochs=$EPOCHS"
echo "batch_size=$BATCH_SIZE"
echo "lr=$LR"
echo "positive=$POSITIVES"
echo "negative=$NEGATIVES"
echo "============================================================"

# ------------------------------------------------------------
# 0. 环境与文件检查
# ------------------------------------------------------------
required_files=(
    "tools/extract_v21_music_features.py"
    "tools/build_v21_router_dataset.py"
    "tools/v21_music_event_calibrated.py"
    "train_v21_music_router.py"
    "model/v21_music_router.py"
    "${INDEX_PREFIX}.json"
    "${INDEX_PREFIX}.npz"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "[ERROR] missing required file: $file"
        exit 2
    fi
done

if [[ ! -d "$MUSIC_DIR" ]]; then
    echo "[ERROR] missing music directory: $MUSIC_DIR"
    exit 2
fi

NUM_MUSIC="$(
    find "$MUSIC_DIR" \
      -maxdepth 1 \
      -type f \
      -iname "*.wav" \
      | wc -l
)"

echo "num_music=$NUM_MUSIC"

if [[ "$NUM_MUSIC" -lt 100 ]]; then
    echo "[ERROR] too few training songs: $NUM_MUSIC"
    exit 2
fi

"$PYTHON_BIN" -m py_compile \
    tools/extract_v21_music_features.py \
    tools/build_v21_router_dataset.py \
    tools/v21_music_event_calibrated.py \
    train_v21_music_router.py \
    model/v21_music_router.py

"$PYTHON_BIN" - <<'PY'
import torch
import numpy as np

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_devices:", torch.cuda.device_count())
print("numpy:", np.__version__)

if not torch.cuda.is_available():
    print("[WARNING] CUDA is unavailable; training will run on CPU.")
PY

# ------------------------------------------------------------
# 1. 提取所有训练音乐特征
# ------------------------------------------------------------
echo
echo "============================================================"
echo "Stage 1/4: extracting music features"
echo "============================================================"

if [[ "$FORCE_REEXTRACT" == "1" ]]; then
    rm -rf "$FEATURE_DIR"
    mkdir -p "$FEATURE_DIR"
fi

EXTRACTED=0
SKIPPED=0
FAILED=0

while IFS= read -r -d '' AUDIO; do
    BASE="$(basename "$AUDIO")"
    STEM="${BASE%.*}"

    OUT_NPY="$FEATURE_DIR/${STEM}_v21_music.npy"
    OUT_JSON="$FEATURE_DIR/${STEM}_v21_music.json"

    if [[ -s "$OUT_NPY" && -s "$OUT_JSON" && "$FORCE_REEXTRACT" != "1" ]]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "[FEATURE] $AUDIO"

    if "$PYTHON_BIN" tools/extract_v21_music_features.py \
        --audio "$AUDIO" \
        --out_npy "$OUT_NPY" \
        --out_json "$OUT_JSON" \
        --num_frames "$NUM_FRAMES"; then
        EXTRACTED=$((EXTRACTED + 1))
    else
        echo "[FEATURE FAIL] $AUDIO"
        FAILED=$((FAILED + 1))
    fi
done < <(
    find "$MUSIC_DIR" \
      -maxdepth 1 \
      -type f \
      -iname "*.wav" \
      -print0 \
      | sort -z
)

NUM_FEATURES="$(
    find "$FEATURE_DIR" \
      -maxdepth 1 \
      -type f \
      -name "*_v21_music.npy" \
      | wc -l
)"

echo "feature_extracted=$EXTRACTED"
echo "feature_skipped=$SKIPPED"
echo "feature_failed=$FAILED"
echo "feature_total=$NUM_FEATURES"

if [[ "$FAILED" -gt 0 ]]; then
    echo "[ERROR] some audio files failed feature extraction"
    exit 3
fi

if [[ "$NUM_FEATURES" -ne "$NUM_MUSIC" ]]; then
    echo "[ERROR] feature count mismatch: music=$NUM_MUSIC features=$NUM_FEATURES"
    exit 3
fi

# ------------------------------------------------------------
# 2. 构建弱监督 Music-Motion Router 数据
# ------------------------------------------------------------
echo
echo "============================================================"
echo "Stage 2/4: building weak contrastive router dataset"
echo "============================================================"

DATASET="$RUN_ROOT/router_dataset.npz"

"$PYTHON_BIN" tools/build_v21_router_dataset.py \
    --index_json "${INDEX_PREFIX}.json" \
    --index_npz "${INDEX_PREFIX}.npz" \
    --music_glob "$FEATURE_DIR/*_v21_music.npy" \
    --out "$DATASET" \
    --phrases "$PHRASES" \
    --positives_per_phrase "$POSITIVES" \
    --negatives_per_positive "$NEGATIVES" \
    --seed 20260605

"$PYTHON_BIN" - "$DATASET" <<'PY'
import sys
from collections import Counter

import numpy as np

path = sys.argv[1]
data = np.load(path, allow_pickle=True)

print("dataset:", path)
print("keys:", data.files)

for key in data.files:
    arr = data[key]
    print(key, arr.shape, arr.dtype)

for key in ("music", "positive", "negative"):
    if key not in data.files:
        raise RuntimeError(f"missing dataset array: {key}")

    if not np.isfinite(data[key]).all():
        raise RuntimeError(f"{key} contains NaN or Inf")

labels = [str(x) for x in data["label"]] if "label" in data.files else []

events = Counter(
    label.rsplit(":", 1)[-1]
    for label in labels
)

print("event_counts:", events)
print("event_classes:", len(events))
print("samples:", len(data["music"]))

if len(data["music"]) < 10_000:
    raise RuntimeError(
        f"Too few router samples: {len(data['music'])}"
    )

if labels and len(events) < 4:
    raise RuntimeError(
        f"Too few event classes: {events}"
    )

print("[PASS] router dataset is healthy")
PY

# ------------------------------------------------------------
# 3. 三个随机种子训练
# ------------------------------------------------------------
echo
echo "============================================================"
echo "Stage 3/4: training three seeds"
echo "============================================================"

SEEDS=(20260605 20260606 20260607)

for SEED in "${SEEDS[@]}"; do
    OUT_DIR="$RUN_ROOT/seed_${SEED}"
    LOG_FILE="$RUN_ROOT/train_seed_${SEED}.log"

    echo
    echo "------------------------------------------------------------"
    echo "training seed=$SEED"
    echo "output=$OUT_DIR"
    echo "------------------------------------------------------------"

    "$PYTHON_BIN" train_v21_music_router.py \
        --data "$DATASET" \
        --out_dir "$OUT_DIR" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --hidden_dim "$HIDDEN_DIM" \
        --latent_dim "$LATENT_DIM" \
        --dropout "$DROPOUT" \
        --margin_weight "$MARGIN_WEIGHT" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_FILE"

    if [[ ! -f "$OUT_DIR/checkpoints/best.pt" ]]; then
        echo "[ERROR] best checkpoint missing for seed=$SEED"
        exit 4
    fi
done

# ------------------------------------------------------------
# 4. 自动选择验证损失最低的模型
# ------------------------------------------------------------
echo
echo "============================================================"
echo "Stage 4/4: selecting best checkpoint"
echo "============================================================"

"$PYTHON_BIN" - "$RUN_ROOT" <<'PY'
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
rows = []

for checkpoint in sorted(root.glob("seed_*/checkpoints/best.pt")):
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    val_loss = float(
        payload.get(
            "val_loss",
            payload.get(
                "best_val_loss",
                1e9,
            ),
        )
    )

    epoch = int(
        payload.get(
            "epoch",
            payload.get("best_epoch", -1),
        )
    )

    rows.append(
        (val_loss, epoch, checkpoint)
    )

if not rows:
    raise RuntimeError("No best checkpoints were found")

rows.sort(key=lambda row: row[0])

print("checkpoint ranking:")

for rank, (val_loss, epoch, checkpoint) in enumerate(rows, 1):
    print(
        f"{rank}. val_loss={val_loss:.8f} "
        f"epoch={epoch} "
        f"path={checkpoint}"
    )

best = rows[0][2].resolve()

(root / "BEST_ROUTER_CKPT.txt").write_text(
    str(best) + "\n",
    encoding="utf-8",
)

summary = "\n".join(
    f"{val_loss:.10f}\t{epoch}\t{checkpoint}"
    for val_loss, epoch, checkpoint in rows
)

(root / "CHECKPOINT_RANKING.tsv").write_text(
    "val_loss\tepoch\tcheckpoint\n"
    + summary
    + "\n",
    encoding="utf-8",
)

print("BEST_ROUTER_CKPT:", best)
PY

echo
echo "============================================================"
echo "V21 Router training finished"
echo "run_root=$RUN_ROOT"
echo "best_checkpoint=$(cat "$RUN_ROOT/BEST_ROUTER_CKPT.txt")"
echo "============================================================"
