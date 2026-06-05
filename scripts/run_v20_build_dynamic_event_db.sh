#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/disk/lsm/storage/EDGE
source /home/disk/lsm/conda_envs/edge/bin/activate 2>/dev/null || true
export PYTHONPATH=$PWD:${PYTHONPATH:-}
RUN_ROOT=${RUN_ROOT:-output/v20_dynamic_event_db_$(date +%Y%m%d_%H%M%S)}
INPUT_DIR=${INPUT_DIR:-data/dunhuang_bvh/processed}
OUT_DIR=${OUT_DIR:-data/dunhuang_dynamic_event_rag}
mkdir -p "$RUN_ROOT"
python tools/build_dynamic_rhythm_event_db.py \
  --input_dir "$INPUT_DIR" \
  --out_dir "$OUT_DIR" \
  --report "$RUN_ROOT/dynamic_event_db_report.json" \
  --min_len ${V20_MIN_LEN:-24} \
  --ideal_len ${V20_IDEAL_LEN:-48} \
  --max_len ${V20_MAX_LEN:-72} \
  --boundary_min_gap ${V20_BOUNDARY_MIN_GAP:-18} \
  --energy_smooth ${V20_ENERGY_SMOOTH:-7} \
  --save_canonical_len ${V20_CANONICAL_LEN:-48} \
  --quality_top_k ${V20_QUALITY_TOP_K:-0} \
  2>&1 | tee "$RUN_ROOT/build_dynamic_event_db.log"
echo "DONE: $OUT_DIR"
