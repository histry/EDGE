#!/usr/bin/env bash
set -uo pipefail

cd /home/disk/lsm/storage/EDGE

# ============================================================
# Basic env
# ============================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/home/disk/lsm/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/home/disk/lsm/miniconda3/etc/profile.d/conda.sh"
fi

conda activate edge || conda activate /home/disk/lsm/conda_envs/edge || true

export PYTHONPATH=$PWD:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p logs/stage_AE_night

MASTER_LOG="logs/stage_AE_night/master_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "============================================================"
echo "Stage A-E overnight pipeline started"
echo "time=$(date)"
echo "MASTER_LOG=$MASTER_LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

# ============================================================
# Config
# ============================================================
V3H_RUN_DIR="${V3H_RUN_DIR:-runs/train_nextgen/v3h_footwork_day07_v3h_only_full_e300}"
V3I_RUN_DIR="${V3I_RUN_DIR:-runs/train_nextgen/v3i_hfevent_support_day08_full_e240}"

HF_RUN_ID="${HF_RUN_ID:-hf_event_day01}"
HF_OUT_DIR="${HF_OUT_DIR:-checkpoints/hf_event_contrastive/${HF_RUN_ID}}"
HF_ENCODER="${HF_ENCODER:-${HF_OUT_DIR}/hf_event_encoder.pt}"

HF_DATA_DIR="${HF_DATA_DIR:-data/dunhuang_bvh/footwork_v3h_u45}"
MUSIC_GLOB="${MUSIC_GLOB:-test_music_bank/*.wav}"
V13_DB="${V13_DB:-data/dunhuang_choreo_unit_rag/index_v13_functional_u45_s15.npz}"
V14_DB="${V14_DB:-data/dunhuang_choreo_unit_rag/index_v14_hf_event_u45_s15.npz}"

SUPPORT_SIDECAR="${SUPPORT_SIDECAR:-data/dunhuang_choreo_unit_rag/index_v14_hf_event_u45_s15.support_quality_gate.npz}"
SUPPORT_POOL="${SUPPORT_POOL:-data/dunhuang_bvh/support_quality_prior_pool_v15_u45}"
SUPPORT_TOP_K="${SUPPORT_TOP_K:-500}"

V15_OUT="output/v15_support_rerank_sampling"
V15_LOG="logs/v15_support_rerank_sampling"

# Stage E training
STAGE_E_RUN_ID="${STAGE_E_RUN_ID:-v3j_hfevent_loss_day09}"
STAGE_E_EPOCHS="${STAGE_E_EPOCHS:-160}"
STAGE_E_SAVE_INTERVAL="${STAGE_E_SAVE_INTERVAL:-40}"
STAGE_E_BATCH_SIZE="${STAGE_E_BATCH_SIZE:-10}"
STAGE_E_SMOKE_BATCH="${STAGE_E_SMOKE_BATCH:-6}"
STAGE_E_LR="${STAGE_E_LR:-2e-6}"
STAGE_E_HF_WEIGHT="${STAGE_E_HF_WEIGHT:-0.0025}"
STAGE_E_DATA="${STAGE_E_DATA:-${SUPPORT_POOL}}"

# 如果 V3I e120 存在，默认从 V3I e120 继续；否则从 Stage A 选出的 V3H ckpt 继续。
DEFAULT_V3I_BASE="${V3I_RUN_DIR}/weights/train-120.pt"

echo "Config:"
echo "  V3H_RUN_DIR=$V3H_RUN_DIR"
echo "  V3I_RUN_DIR=$V3I_RUN_DIR"
echo "  HF_ENCODER=$HF_ENCODER"
echo "  V14_DB=$V14_DB"
echo "  SUPPORT_POOL=$SUPPORT_POOL"
echo "  STAGE_E_RUN_ID=$STAGE_E_RUN_ID"
echo "  STAGE_E_DATA=$STAGE_E_DATA"
echo "============================================================"

# ============================================================
# Helper
# ============================================================
count_files() {
  local pattern="$1"
  find $(dirname "$pattern") -name "$(basename "$pattern")" 2>/dev/null | wc -l
}

# ============================================================
# Stage A: day07 V3H support-chain checkpoint selection
# ============================================================
echo
echo "==================== Stage A ===================="
echo "Stage A: select day07 V3H support-chain checkpoint"

mkdir -p logs/stage_AE_night/stageA

if [ ! -d "${V3H_RUN_DIR}/weights" ]; then
  echo "ERROR: V3H weights dir not found: ${V3H_RUN_DIR}/weights"
  exit 2
fi

echo "Available V3H checkpoints:"
find "${V3H_RUN_DIR}/weights" -name "train-*.pt" | sort

# 保守规则：
# - 如果 V3I e120 已经存在，Stage E 使用 V3I e120；
# - 否则 Stage A 选 V3H train-300；
# - 如果 train-300 不存在，选最大 epoch。
SELECTED_V3H_CKPT="${V3H_RUN_DIR}/weights/train-300.pt"
if [ ! -f "$SELECTED_V3H_CKPT" ]; then
  SELECTED_V3H_CKPT="$(find "${V3H_RUN_DIR}/weights" -name "train-*.pt" | sort -V | tail -n 1)"
fi

if [ -z "$SELECTED_V3H_CKPT" ] || [ ! -f "$SELECTED_V3H_CKPT" ]; then
  echo "ERROR: no V3H checkpoint found"
  exit 3
fi

if [ -f "$DEFAULT_V3I_BASE" ]; then
  BASE_CKPT_FOR_E="${BASE_CKPT_FOR_E:-$DEFAULT_V3I_BASE}"
else
  BASE_CKPT_FOR_E="${BASE_CKPT_FOR_E:-$SELECTED_V3H_CKPT}"
fi

cat > logs/stage_AE_night/stageA/selected_ckpt.env <<EOF
SELECTED_V3H_CKPT=$SELECTED_V3H_CKPT
BASE_CKPT_FOR_E=$BASE_CKPT_FOR_E
EOF

echo "SELECTED_V3H_CKPT=$SELECTED_V3H_CKPT"
echo "BASE_CKPT_FOR_E=$BASE_CKPT_FOR_E"

# ============================================================
# Stage B + C: train HF event encoder and export V14 DB
# ============================================================
echo
echo "==================== Stage B/C ===================="
echo "Stage B: offline train HF event encoder"
echo "Stage C: export V14 HF-event RAG DB"

if [ -f "$HF_ENCODER" ] && [ -f "$V14_DB" ]; then
  echo "Skip Stage B/C: existing encoder and V14 DB found."
  ls -lh "$HF_ENCODER" "$V14_DB"
else
  echo "Running scripts/run_hf_event_stage.sh ..."
  RUN_ID="$HF_RUN_ID" \
  OUT_DIR="$HF_OUT_DIR" \
  DATA_DIR="$HF_DATA_DIR" \
  MUSIC_GLOB="$MUSIC_GLOB" \
  IN_DB="$V13_DB" \
  OUT_DB="$V14_DB" \
  EPOCHS="${HF_EPOCHS:-80}" \
  BATCH_SIZE="${HF_BATCH_SIZE:-64}" \
  bash scripts/run_hf_event_stage.sh

  if [ ! -f "$HF_ENCODER" ] || [ ! -f "$V14_DB" ]; then
    echo "ERROR: Stage B/C failed. Missing $HF_ENCODER or $V14_DB"
    exit 4
  fi
fi

echo "HF encoder: $HF_ENCODER"
echo "V14 DB: $V14_DB"

# ============================================================
# Stage D: HF/support quality RAG rerank
# ============================================================
echo
echo "==================== Stage D ===================="
echo "Stage D: build support-prior quality gate and run V15 RAG rerank sampling"

mkdir -p "$V15_OUT/npy" "$V15_OUT/videos" "$V15_LOG"

# 需要你之前写入过 tools/build_support_prior_quality_gate.py。
# 如果不存在，直接退出，避免半夜静默跑错。
if [ ! -f tools/build_support_prior_quality_gate.py ]; then
  echo "ERROR: tools/build_support_prior_quality_gate.py not found."
  echo "请先写入之前给你的 V15 support-prior quality gate 脚本。"
  exit 5
fi

POOL_COUNT=0
if [ -d "$SUPPORT_POOL" ]; then
  POOL_COUNT=$(find "$SUPPORT_POOL" -name "*.pkl" | wc -l)
fi

if [ -f "$SUPPORT_SIDECAR" ] && [ "$POOL_COUNT" -ge 300 ]; then
  echo "Skip building support quality pool: sidecar exists and pool_count=$POOL_COUNT"
else
  echo "Building support quality sidecar and top-${SUPPORT_TOP_K} pool..."
  python tools/build_support_prior_quality_gate.py \
    --db "$V14_DB" \
    --out_sidecar "$SUPPORT_SIDECAR" \
    --out_pool "$SUPPORT_POOL" \
    --seq_len 45 \
    --top_k "$SUPPORT_TOP_K" \
    2>&1 | tee "$V15_LOG/build_support_prior_quality_gate.log"
fi

echo "Support pool summary:"
cat "$SUPPORT_POOL/support_quality_summary.json" || true
head -n 20 "$SUPPORT_POOL/support_quality_manifest.csv" || true

# top12 prior list
python - <<'PY' > output/v15_support_rerank_sampling/top12_prior_list.txt
from pathlib import Path
pool = Path("data/dunhuang_bvh/support_quality_prior_pool_v15_u45")
items = sorted(pool.glob("*.pkl"))[:12]
if len(items) < 12:
    raise SystemExit(f"ERROR: only {len(items)} support-quality priors found")
for p in items:
    print(p)
PY

echo "Top12 prior list:"
cat output/v15_support_rerank_sampling/top12_prior_list.txt

# V15 guided sampling: 12 priors × e80/e120 × 2 masks = 48
for E in 80 120; do
  CKPT="${V3I_RUN_DIR}/weights/train-${E}.pt"
  if [ ! -f "$CKPT" ]; then
    echo "WARN: missing $CKPT, fallback to $BASE_CKPT_FOR_E"
    CKPT="$BASE_CKPT_FOR_E"
  fi

  IDX=0
  while read PRIOR; do
    for MASK in body_no_rootxz torso_upper; do
      LABEL="v15_e${E}_q${IDX}_${MASK}_s0.35"
      OUT_NPY="$V15_OUT/npy/${LABEL}.npy"

      if [ -f "$OUT_NPY" ]; then
        echo "skip existing sample: $OUT_NPY"
      else
        echo "===== sampling $LABEL ====="
        python tools/sample_v3i_prior_guided.py \
          --ckpt "$CKPT" \
          --prior "$PRIOR" \
          --out "$OUT_NPY" \
          --label "$LABEL" \
          --num_samples 1 \
          --seq_len 45 \
          --mask "$MASK" \
          --strength 0.35 \
          --start_frac 0.70 \
          --gamma 1.3 \
          --seed $((20260522 + E + IDX)) \
          2>&1 | tee "$V15_LOG/sample_${LABEL}.log"
      fi
    done
    IDX=$((IDX+1))
  done < output/v15_support_rerank_sampling/top12_prior_list.txt
done

echo "V15 npy count:"
find "$V15_OUT/npy" -name "*.npy" | wc -l

# Render Stage D; render failure should not stop Stage E.
for NPY in $(find "$V15_OUT/npy" -name "*.npy" | sort); do
  STEM=$(basename "$NPY" .npy)
  OUT_MP4="$V15_OUT/videos/${STEM}_fixed.mp4"

  if [ -f "$OUT_MP4" ]; then
    echo "skip rendered: $STEM"
  else
    echo "===== rendering $STEM ====="
    python render_from_npy.py \
      --motion "$NPY" \
      --audio test_music_bank/dunhuangwu2.wav \
      --output "$OUT_MP4" \
      --camera_mode fixed \
      2>&1 | tee "$V15_LOG/render_${STEM}.log" || true
  fi
done

echo "V15 rendered videos:"
find "$V15_OUT/videos" -name "*.mp4" | wc -l

# ============================================================
# Stage E pre-patch: make V3 keep audio only for HF-event training
# ============================================================
echo
echo "==================== Stage E prep ===================="
echo "Patch V3 to keep proxy audio only when EDGE_HF_EVENT_CONTRASTIVE=1"

python - <<'PY'
from pathlib import Path

# Patch unit_reconstruction_patch.py: keep audio when HF event loss is enabled.
p = Path("unit_reconstruction_patch.py")
txt = p.read_text(encoding="utf-8")
backup = Path("unit_reconstruction_patch.py.bak_before_hf_keep_audio")
if not backup.exists():
    backup.write_text(txt, encoding="utf-8")

old = '''    audio = cond.get("audio", None)
    if torch.is_tensor(audio):
        cleaned["audio"] = torch.zeros_like(audio)
'''

new = '''    audio = cond.get("audio", None)
    if torch.is_tensor(audio):
        # V3 normally zeros audio to avoid accidental music supervision.
        # For Stage-E HF-event contrastive training, keep weak/proxy audio explicitly.
        if _env_bool("EDGE_V3_KEEP_AUDIO_FOR_HF", False) and _env_bool("EDGE_HF_EVENT_CONTRASTIVE", False):
            cleaned["audio"] = audio
        else:
            cleaned["audio"] = torch.zeros_like(audio)
'''

if "EDGE_V3_KEEP_AUDIO_FOR_HF" not in txt:
    if old not in txt:
        raise SystemExit("ERROR: expected audio-zeroing block not found in unit_reconstruction_patch.py")
    txt = txt.replace(old, new)
    p.write_text(txt, encoding="utf-8")
    print("patched unit_reconstruction_patch.py")
else:
    print("unit_reconstruction_patch.py already patched")

# Patch train.py: do not force audio_pairing_mode=none in V3 when HF-event contrastive is explicitly enabled.
p = Path("train.py")
txt = p.read_text(encoding="utf-8")
backup = Path("train.py.bak_before_hf_keep_audio")
if not backup.exists():
    backup.write_text(txt, encoding="utf-8")

old = '''        opt.audio_pairing_mode = "none"
'''
new = '''        if _truthy("EDGE_V3_KEEP_AUDIO_FOR_HF", False) and _truthy("EDGE_HF_EVENT_CONTRASTIVE", False):
            opt.audio_pairing_mode = os.environ.get("EDGE_V3_AUDIO_PAIRING_MODE", "proxy")
            print(f"🎵 V3 HF-event mode: keeping audio_pairing_mode={opt.audio_pairing_mode}")
        else:
            opt.audio_pairing_mode = "none"
'''

if "EDGE_V3_AUDIO_PAIRING_MODE" not in txt:
    if old not in txt:
        raise SystemExit("ERROR: expected audio_pairing_mode override not found in train.py")
    txt = txt.replace(old, new, 1)
    p.write_text(txt, encoding="utf-8")
    print("patched train.py")
else:
    print("train.py already patched")
PY

# Proxy audio check: if no proxy audio exists, HF term may be zero.
echo "Proxy audio precheck:"
python - <<'PY'
from pathlib import Path
n_rag = len(list(Path("data/dunhuang_rag_db").glob("*.npy"))) if Path("data/dunhuang_rag_db").exists() else 0
weak = Path("data/proxy_weak_pairs/weak_pairs.csv")
print("data/dunhuang_rag_db npy count:", n_rag)
print("weak_pairs exists:", weak.exists(), weak)
if n_rag == 0 and not weak.exists():
    print("WARNING: no proxy audio source found. Stage E can run, but HF Event Loss may stay near zero.")
PY

# ============================================================
# Stage E: Small-weight EDGE_HF_EVENT_CONTRASTIVE=1 training
# ============================================================
echo
echo "==================== Stage E ===================="
echo "Stage E: small-weight EDGE_HF_EVENT_CONTRASTIVE=1 new training"

if [ ! -f "$BASE_CKPT_FOR_E" ]; then
  echo "ERROR: BASE_CKPT_FOR_E not found: $BASE_CKPT_FOR_E"
  exit 6
fi

if [ ! -d "$STAGE_E_DATA" ]; then
  echo "ERROR: STAGE_E_DATA not found: $STAGE_E_DATA"
  exit 7
fi

if [ ! -f "$HF_ENCODER" ]; then
  echo "ERROR: HF_ENCODER not found: $HF_ENCODER"
  exit 8
fi

echo "Stage E base ckpt: $BASE_CKPT_FOR_E"
echo "Stage E data: $STAGE_E_DATA"
echo "Stage E HF encoder: $HF_ENCODER"

export EDGE_TRAIN_PROFILE=v3_unit_recon
export EDGE_V3_UNIT_RECON=1
export EDGE_TRAJECTORY_PLANE=xz
export EDGE_DUNHUANG_SPLIT_MODE=source_file
export EDGE_DUNHUANG_STRICT_SPLIT=1
export EDGE_DUNHUANG_ASSERT_TRAJ_MATCH=0

# Keep proxy audio only for HF-event loss.
export EDGE_V3_KEEP_AUDIO_FOR_HF=1
export EDGE_V3_AUDIO_PAIRING_MODE=proxy

# V3 stable support-chain config
export EDGE_V3_BASE_LOSS_STABILITY=1
export EDGE_V3_DISABLE_RAW_PHYSICAL_LOSSES=1
export EDGE_V3_LOSS_STABILITY=1
export EDGE_V3_CAP_TOTAL_LOSS=0
export EDGE_X0_RECON_LOSS=0
export EDGE_V3_TEMPORAL_WEIGHT=0.0

export EDGE_V3C_VISIBLE_FK=0
export EDGE_V3F_BODY_CENTERED=0
export EDGE_V3H_SUPPORT_CHAIN_LOSS=1
export EDGE_V3H_SUPPORT_CHAIN_WEIGHT="${EDGE_V3H_SUPPORT_CHAIN_WEIGHT:-0.025}"
export EDGE_V3_MOTION_ENERGY_LOSS_CAP=80
export EDGE_V3H_SAMPLE_LOSS_CAP=8
export EDGE_V3H_TERM_CAP=8

# Small HF event contrastive loss
export EDGE_HF_EVENT_CONTRASTIVE=1
export EDGE_HF_EVENT_ENCODER_CKPT="$HF_ENCODER"
export EDGE_HF_EVENT_WEIGHT="$STAGE_E_HF_WEIGHT"
export EDGE_HF_EVENT_WARMUP_START="${EDGE_HF_EVENT_WARMUP_START:-10}"
export EDGE_HF_EVENT_WARMUP_END="${EDGE_HF_EVENT_WARMUP_END:-60}"
export EDGE_HF_EVENT_TEMPERATURE="${EDGE_HF_EVENT_TEMPERATURE:-0.1}"
export EDGE_HF_EVENT_USE_SUPCON="${EDGE_HF_EVENT_USE_SUPCON:-1}"
export EDGE_HF_EVENT_APPEND_METRIC=1
export EDGE_HF_EVENT_DEBUG="${EDGE_HF_EVENT_DEBUG:-1}"

# Smoke first
echo "-------------------- Stage E smoke --------------------"
python train.py \
  --project runs/train_nextgen \
  --exp_name "${STAGE_E_RUN_ID}_smoke_e5" \
  --data_path "$STAGE_E_DATA" \
  --checkpoint "$BASE_CKPT_FOR_E" \
  --feature_type hybrid \
  --audio_dim 803 \
  --seq_len 45 \
  --batch_size "$STAGE_E_SMOKE_BATCH" \
  --epochs 5 \
  --save_interval 5 \
  --learning_rate "$STAGE_E_LR" \
  --weight_decay 0.02 \
  --audio_pairing_mode proxy \
  --mmr_loss_weight 0.0 \
  --disable_traj_cond \
  --keyframe_condition_prob 0.0 \
  --keyframe_loss_weight 0.0 \
  --mid_keyframe_condition_prob 0.0 \
  --mid_keyframe_count 0 \
  --trajectory_loss_weight 0.0 \
  --trajectory_velocity_loss_weight 0.0 \
  --beat_guidance_weight 0.0 \
  --root_lower_coupling_loss_weight 0.0 \
  --energy_loss_weight 0.0 \
  --val_batches 2 \
  --train_num_workers 2 \
  --val_num_workers 1 \
  2>&1 | tee "logs/stage_AE_night/${STAGE_E_RUN_ID}_smoke.log"

echo "-------------------- Stage E full --------------------"
python train.py \
  --project runs/train_nextgen \
  --exp_name "${STAGE_E_RUN_ID}_full_e${STAGE_E_EPOCHS}" \
  --data_path "$STAGE_E_DATA" \
  --checkpoint "$BASE_CKPT_FOR_E" \
  --feature_type hybrid \
  --audio_dim 803 \
  --seq_len 45 \
  --batch_size "$STAGE_E_BATCH_SIZE" \
  --epochs "$STAGE_E_EPOCHS" \
  --save_interval "$STAGE_E_SAVE_INTERVAL" \
  --learning_rate "$STAGE_E_LR" \
  --weight_decay 0.02 \
  --audio_pairing_mode proxy \
  --mmr_loss_weight 0.0 \
  --disable_traj_cond \
  --keyframe_condition_prob 0.0 \
  --keyframe_loss_weight 0.0 \
  --mid_keyframe_condition_prob 0.0 \
  --mid_keyframe_count 0 \
  --trajectory_loss_weight 0.0 \
  --trajectory_velocity_loss_weight 0.0 \
  --beat_guidance_weight 0.0 \
  --root_lower_coupling_loss_weight 0.0 \
  --energy_loss_weight 0.0 \
  --val_batches 4 \
  --train_num_workers 4 \
  --val_num_workers 2 \
  2>&1 | tee "logs/stage_AE_night/${STAGE_E_RUN_ID}_full.log"

echo
echo "============================================================"
echo "Stage A-E overnight pipeline finished"
echo "time=$(date)"
echo "Stage E checkpoints:"
find runs/train_nextgen -path "*${STAGE_E_RUN_ID}_full_e${STAGE_E_EPOCHS}*/weights/train-*.pt" | sort || true
echo "============================================================"
