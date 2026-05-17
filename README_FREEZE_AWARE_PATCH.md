# EDGE freeze-aware motion coverage patch

This patch addresses the v2b failure mode:

```text
mid-hard snapping is reduced, but some endpoint-continuous units collapse into:
fast transition -> static hold
```

`unit_76207` is a typical bad case: low jerk/jump but visually frozen.

## Files

Copy these files into the EDGE repository root:

```text
freeze_aware_motion_patch.py
sitecustomize.py
tools/diagnose_static_hold_units.py
tools/make_nostatic_unit_list.py
tools/export_units_from_rag_db.py
scripts/run_train_v2d_nostatic_endpoint.sh
```

No existing core file needs to be overwritten. `sitecustomize.py` auto-installs
the patch only when `EDGE_FREEZE_AWARE_MOTION=1`.

## Stop unstable overnight run

```bash
cd /home/disk/lsm/storage/EDGE
PID=$(cat logs/overnight_v2b_v2c_master.pid 2>/dev/null || true)
[ -n "$PID" ] && kill "$PID"
```

## Mark unit_76207 as bad

```bash
mkdir -p output/stationary_whitelist_v2b_endpoint_eval/qc
cat > output/stationary_whitelist_v2b_endpoint_eval/qc/bad_units_visual.txt <<'EOF'
76207
EOF
```

## Diagnose static hold

Find generated npy files first:

```bash
find output/stationary_whitelist_v2b_endpoint_eval -name "*.npy" | head -50
```

Then run:

```bash
python tools/diagnose_static_hold_units.py \
  --pred_dir output/stationary_whitelist_v2b_endpoint_eval/e200 \
  --out_csv output/stationary_whitelist_v2b_endpoint_eval/qc/static_hold_diag_e200.csv
```

Adjust `--pred_dir` to the actual directory containing generated `[T,151]` files.

## Build no-static unit list

```bash
python tools/make_nostatic_unit_list.py \
  --score_csv output/stationary_whitelist_v2b_endpoint_eval/best_units_by_score_verbose.csv \
  --static_csv output/stationary_whitelist_v2b_endpoint_eval/qc/static_hold_diag_e200.csv \
  --bad_units_txt output/stationary_whitelist_v2b_endpoint_eval/qc/bad_units_visual.txt \
  --step 200 \
  --max_units 5 \
  --out output/stationary_whitelist_v2b_endpoint_eval/qc/keep_units_no_static_hold.txt
```

## Export selected units from RAG DB

Example:

```bash
python tools/export_units_from_rag_db.py \
  --rag_db data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_mobility.npz \
  --unit_list output/stationary_whitelist_v2b_endpoint_eval/qc/keep_units_no_static_hold.txt \
  --out_dir data/dunhuang_bvh/stationary_whitelist_v2d_nostatic
```

## Train freeze-aware endpoint-continuous v2d

```bash
bash scripts/run_train_v2d_nostatic_endpoint.sh
```

Or override paths:

```bash
DATA_PATH=data/dunhuang_bvh/stationary_whitelist_v2d_nostatic \
CKPT=runs/train_nextgen/stationary_whitelist_v2b_endpoint_continuous_from_v16_27units_b2_fix/weights/train-200.pt \
EXP_NAME=stationary_v2d_nostatic_endpoint_freezeaware_x0w05 \
bash scripts/run_train_v2d_nostatic_endpoint.sh
```

## Key switches

```bash
export EDGE_FREEZE_AWARE_MOTION=1
export EDGE_FREEZE_AWARE_FEATURE_MODE=upper_torso
export EDGE_X0_RECON_LOSS=1
export EDGE_X0_RECON_LOSS_WEIGHT=0.5
export EDGE_MOTION_COVERAGE_WEIGHT=6.0
export EDGE_MOTION_TAIL_WEIGHT=6.0
export EDGE_MOTION_ACTIVE_WEIGHT=3.0
export EDGE_ANTI_FREEZE_LOSS_SCALE=4.0
```

Disable trajectory/RAG/beat during this stationary debugging phase:

```bash
export EDGE_WEAK_TRAJ_ENERGY=0
export EDGE_TRAJ_EVENT_COND=0
export EDGE_BEAT_GUIDANCE=0
export EDGE_UNIT_SOFT_PRIOR=0
export EDGE_ENABLE_TEXT_CONTEXT_RAG=0
```
