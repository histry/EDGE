#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/disk/lsm/storage/EDGE

# ===== 阈值，可按需调整 =====
MIN_GPU_FREE_MB=${MIN_GPU_FREE_MB:-22000}   # RTX4090 跑全量评估建议至少空出 22GB
MIN_RAM_FREE_MB=${MIN_RAM_FREE_MB:-16000}   # 系统可用内存至少 16GB
MIN_DISK_FREE_GB=${MIN_DISK_FREE_GB:-30}    # 磁盘至少 30GB
GPU_ID=${CUDA_VISIBLE_DEVICES:-0}

echo "============================================================"
echo "Preflight check before V15 full checkpoint evaluation"
echo "GPU_ID=$GPU_ID"
echo "MIN_GPU_FREE_MB=$MIN_GPU_FREE_MB"
echo "MIN_RAM_FREE_MB=$MIN_RAM_FREE_MB"
echo "MIN_DISK_FREE_GB=$MIN_DISK_FREE_GB"
echo "============================================================"

echo
echo "[1/5] Check scripts..."
if [ ! -f scripts/eval_v15_onset_ckpts.sh ]; then
  echo "ERROR: scripts/eval_v15_onset_ckpts.sh not found"
  exit 1
fi
bash -n scripts/eval_v15_onset_ckpts.sh
echo "✅ eval script syntax OK"

echo
echo "[2/5] Check GPU memory..."
GPU_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | head -1 | tr -d ' ')
GPU_USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | head -1 | tr -d ' ')
GPU_TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU_ID" | head -1 | tr -d ' ')

echo "GPU memory: used=${GPU_USED_MB}MiB free=${GPU_FREE_MB}MiB total=${GPU_TOTAL_MB}MiB"

if [ "$GPU_FREE_MB" -lt "$MIN_GPU_FREE_MB" ]; then
  echo
  echo "❌ GPU free memory is not enough."
  echo "Need >= ${MIN_GPU_FREE_MB}MiB, current free=${GPU_FREE_MB}MiB"
  echo
  echo "Current GPU processes:"
  nvidia-smi
  echo
  echo "Related python processes:"
  ps -ef | grep -E "generate_controlled.py|eval_v15|train.py|python" | grep -v grep || true
  echo
  echo "先确认这些进程是否可以杀掉，例如："
  echo "  kill <PID>"
  echo "或者强制："
  echo "  kill -9 <PID>"
  exit 2
fi
echo "✅ GPU memory enough"

echo
echo "[3/5] Check system RAM..."
RAM_AVAIL_MB=$(free -m | awk '/Mem:/ {print $7}')
echo "System RAM available: ${RAM_AVAIL_MB}MiB"

if [ "$RAM_AVAIL_MB" -lt "$MIN_RAM_FREE_MB" ]; then
  echo "❌ System RAM is not enough. Need >= ${MIN_RAM_FREE_MB}MiB"
  free -h
  exit 3
fi
echo "✅ System RAM enough"

echo
echo "[4/5] Check disk space..."
DISK_FREE_GB=$(df -BG . | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "Disk free: ${DISK_FREE_GB}GB"

if [ "$DISK_FREE_GB" -lt "$MIN_DISK_FREE_GB" ]; then
  echo "❌ Disk free space is not enough. Need >= ${MIN_DISK_FREE_GB}GB"
  df -h .
  exit 4
fi
echo "✅ Disk space enough"

echo
echo "[5/5] Check old tmux sessions..."
tmux ls 2>/dev/null || true

echo
echo "============================================================"
echo "✅ Preflight passed. Starting full evaluation..."
echo "============================================================"

tmux kill-session -t v15_eval_ckpts 2>/dev/null || true

tmux new -d -s v15_eval_ckpts "bash -lc '
  cd /home/disk/lsm/storage/EDGE
  source /home/disk/lsm/conda_envs/edge/bin/activate
  export PYTHONPATH=\$PWD:\${PYTHONPATH:-}
  export CUDA_VISIBLE_DEVICES=${GPU_ID}
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export WANDB_MODE=disabled

  bash scripts/eval_v15_onset_ckpts.sh
  STATUS=\$?
  echo
  echo ===== EVAL EXIT STATUS: \$STATUS =====
  echo 最新评估目录: \$(ls -td output/eval_v15_onset_ckpts_* 2>/dev/null | head -1)
  exec bash
'"

sleep 2
tmux ls
echo
echo "进入查看："
echo "tmux attach -t v15_eval_ckpts"
