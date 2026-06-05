#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/disk/lsm/storage/EDGE
source /home/disk/lsm/conda_envs/edge/bin/activate 2>/dev/null || true
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
RUN_ROOT=${RUN_ROOT:-output/v20_transition_train_$(date +%Y%m%d_%H%M%S)}
EVENT_DB=${EVENT_DB:-data/dunhuang_dynamic_event_rag/index_dynamic_event.json}
mkdir -p "$RUN_ROOT"
python tools/build_transition_dpn_dataset.py \
  --event_db "$EVENT_DB" \
  --out_dir "$RUN_ROOT/dpn_dataset" \
  --max_pairs ${V20_DPN_MAX_PAIRS:-50000} \
  2>&1 | tee "$RUN_ROOT/build_dpn_dataset.log"
python train_transition_duration_predictor.py \
  --data_dir "$RUN_ROOT/dpn_dataset" \
  --out_dir "$RUN_ROOT/dpn" \
  --epochs ${V20_DPN_EPOCHS:-500} \
  --batch_size ${V20_DPN_BATCH_SIZE:-128} \
  --lr ${V20_DPN_LR:-3e-4} \
  2>&1 | tee "$RUN_ROOT/train_dpn.log"
python tools/build_transition_refiner_dataset.py \
  --event_db "$EVENT_DB" \
  --out_dir "$RUN_ROOT/transition_refiner_dataset" \
  --max_samples ${V20_TRANS_MAX_SAMPLES:-30000} \
  --max_transition_len ${V20_MAX_TRANSITION_LEN:-20} \
  --noise_std ${V20_TRANS_NOISE_STD:-0.012} \
  2>&1 | tee "$RUN_ROOT/build_transition_refiner_dataset.log"
python train_endpoint_transition_refiner.py \
  --data_dir "$RUN_ROOT/transition_refiner_dataset" \
  --out_dir "$RUN_ROOT/transition_refiner" \
  --epochs ${V20_TRANS_EPOCHS:-800} \
  --batch_size ${V20_TRANS_BATCH_SIZE:-16} \
  --lr ${V20_TRANS_LR:-2e-4} \
  --lambda_endpoint ${V20_TRANS_LAMBDA_ENDPOINT:-2.0} \
  --lambda_vel ${V20_TRANS_LAMBDA_VEL:-0.8} \
  --lambda_acc ${V20_TRANS_LAMBDA_ACC:-0.2} \
  --lambda_style_preserve ${V20_TRANS_LAMBDA_PRESERVE:-0.35} \
  --lambda_root ${V20_TRANS_LAMBDA_ROOT:-5.0} \
  2>&1 | tee "$RUN_ROOT/train_transition_refiner.log"
echo "DONE: $RUN_ROOT"
