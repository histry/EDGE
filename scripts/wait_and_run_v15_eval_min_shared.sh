#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/disk/lsm/storage/EDGE

# ===== 共享服务器安全设置 =====
GPU_ID=${GPU_ID:-0}
MIN_GPU_FREE_MB=${MIN_GPU_FREE_MB:-18000}     # minimal eval 至少等 18GB 空闲
CHECK_INTERVAL_SEC=${CHECK_INTERVAL_SEC:-300} # 每 5 分钟检查一次
MAX_WAIT_HOURS=${MAX_WAIT_HOURS:-12}

LOG_DIR="logs/shared_gpu_wait_v15_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

exec > >(stdbuf -oL -eL tee -a "$LOG_DIR/wait.log") 2>&1

echo "============================================================"
echo "Shared GPU safe wait: V15 minimal eval"
echo "GPU_ID=$GPU_ID"
echo "MIN_GPU_FREE_MB=$MIN_GPU_FREE_MB"
echo "CHECK_INTERVAL_SEC=$CHECK_INTERVAL_SEC"
echo "MAX_WAIT_HOURS=$MAX_WAIT_HOURS"
echo "LOG_DIR=$LOG_DIR"
echo "============================================================"

if [ ! -f scripts/eval_v15_ck260_minimal.sh ]; then
  echo "ERROR: scripts/eval_v15_ck260_minimal.sh not found"
  echo "请先创建 minimal eval 脚本。"
  exit 1
fi

bash -n scripts/eval_v15_ck260_minimal.sh

START_TS=$(date +%s)
MAX_WAIT_SEC=$((MAX_WAIT_HOURS * 3600))

while true; do
  NOW_TS=$(date +%s)
  ELAPSED=$((NOW_TS - START_TS))

  GPU_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | head -1 | tr -d ' ')
  GPU_USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | head -1 | tr -d ' ')

  echo
  echo "[$(date '+%F %T')] GPU used=${GPU_USED_MB}MiB free=${GPU_FREE_MB}MiB elapsed=${ELAPSED}s"

  nvidia-smi | sed -n '/Processes:/,$p'

  if [ "$GPU_FREE_MB" -ge "$MIN_GPU_FREE_MB" ]; then
    echo
    echo "✅ GPU memory enough. Start minimal eval."
    break
  fi

  if [ "$ELAPSED" -ge "$MAX_WAIT_SEC" ]; then
    echo
    echo "❌ Timeout: waited ${MAX_WAIT_HOURS} hours but GPU is still busy."
    exit 2
  fi

  echo "GPU busy. Do not kill other users. Sleep ${CHECK_INTERVAL_SEC}s..."
  sleep "$CHECK_INTERVAL_SEC"
done

# 二次确认，避免刚好有人抢占
sleep 10
GPU_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | head -1 | tr -d ' ')
echo "Second check: GPU free=${GPU_FREE_MB}MiB"

if [ "$GPU_FREE_MB" -lt "$MIN_GPU_FREE_MB" ]; then
  echo "GPU was occupied again, exit safely."
  exit 3
fi

export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled

# 低优先级运行，避免影响服务器交互
nice -n 10 bash scripts/eval_v15_ck260_minimal.sh

echo
echo "DONE"
echo "LOG_DIR=$LOG_DIR"
echo "LATEST_EVAL=$(ls -td output/eval_v15_ck260_minimal_* 2>/dev/null | head -1)"
