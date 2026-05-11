#!/usr/bin/env bash
set -euo pipefail

# Stage 2: light adapter training for gait/footstep phase conditioning.
# Keep this adapter-only first. Add --adapter_train_decoder only in a second small-LR run.

export EDGE_GAIT_PHASE_COND=${EDGE_GAIT_PHASE_COND:-1}
export EDGE_GAIT_PHASE_DIM=${EDGE_GAIT_PHASE_DIM:-6}
export EDGE_GAIT_PHASE_DROP_PROB=${EDGE_GAIT_PHASE_DROP_PROB:-0.10}
export EDGE_GAIT_CONTACT_LOSS=${EDGE_GAIT_CONTACT_LOSS:-1}
export EDGE_GAIT_CONTACT_LOSS_WEIGHT=${EDGE_GAIT_CONTACT_LOSS_WEIGHT:-0.60}

# Recommended with current repo's nextgen DCL patch.
export EDGE_DIFF_CONTACT_LOSS=${EDGE_DIFF_CONTACT_LOSS:-1}
export EDGE_DCL_CONTACT_SOURCE=${EDGE_DCL_CONTACT_SOURCE:-auto}
export EDGE_DCL_MAX_TARGET_CONTACT_RATIO=${EDGE_DCL_MAX_TARGET_CONTACT_RATIO:-0.85}
export EDGE_DCL_FALLBACK_CONTACT_SOURCE=${EDGE_DCL_FALLBACK_CONTACT_SOURCE:-pred_fk_height}

python train.py \
  --data_path ${DATA_PATH:-data/dunhuang_bvh/processed} \
  --checkpoint ${CHECKPOINT:-runs/train_stage45/v10_text_context_rag_adapter_e8_fix2_full/weights/train-4.pt} \
  --project ${PROJECT:-runs/train_footstep_phase} \
  --exp_name ${EXP_NAME:-gait_phase_adapter_v1} \
  --train_stage adapter \
  --batch_size ${BATCH_SIZE:-4} \
  --epochs ${EPOCHS:-20} \
  --learning_rate ${LR:-1e-4} \
  --trajectory_loss_weight ${TRAJ_LOSS:-1.0} \
  --trajectory_velocity_loss_weight ${TRAJ_VEL_LOSS:-0.35} \
  --foot_loss_weight ${FOOT_LOSS:-0.4} \
  --contact_loss_weight ${CONTACT_LOSS:-0.4} \
  --sync_loss_weight ${SYNC_LOSS:-1.0} \
  --save_interval ${SAVE_INTERVAL:-5} \
  --val_batches ${VAL_BATCHES:-10} \
  "$@"
