#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/EDGE
mkdir -p logs

echo "== Step4 Greedy =="
EDGE_V10_SEARCH_METHOD=greedy \
EDGE_V10_OUT_PREFIX=output/v10_eval/v10_step4_greedy_wu2 \
bash scripts/run_v10_step4_auto_multiunit.sh 2>&1 | tee logs/v10_step4_greedy.log

echo "== Step4 Beam =="
EDGE_V10_SEARCH_METHOD=beam \
EDGE_V10_OUT_PREFIX=output/v10_eval/v10_step4_beam_wu2 \
bash scripts/run_v10_step4_auto_multiunit.sh 2>&1 | tee logs/v10_step4_beam.log
