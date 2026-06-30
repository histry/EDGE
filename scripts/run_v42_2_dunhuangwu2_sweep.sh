#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
EDGE_ROOT="$(pwd)"
export PYTHONPATH="$EDGE_ROOT:${PYTHONPATH:-}"

FINAL_DIR="$(cat output/LATEST_FINAL_DUNHUANG_ACCEPTED.txt)"
INPUT="$FINAL_DIR/dunhuangwu2_v40_ACCEPTED_ref.npy"
if [[ ! -f "$INPUT" ]]; then
  INPUT="output/v38_source_aware_full_train_20260625_212818/v34_contact_inr/dunhuangwu2_v40_ACCEPTED_ref.npy"
fi
if [[ ! -f "$INPUT" ]]; then
  echo "[ERROR] input motion not found: $INPUT" >&2
  exit 2
fi

AUDIO="test_music_bank/dunhuangwu2.wav"
CFG="configs/v42_2_physics_config.json"
MASTER="$FINAL_DIR/v42_2_dunhuangwu2_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER"
echo "$MASTER" > output/LATEST_V42_2_DUNHUANGWU2_SWEEP.txt

echo "[V42.2 SWEEP START] $(date)"
echo "[MASTER] $MASTER"

# root_xz_strength contact_high contact_low max_corr root_y_strength height_margin speed_gate
RUNS=(
  "0.45 0.58 0.38 0.055 0.20 0.040 0.030"
  "0.55 0.56 0.36 0.065 0.25 0.045 0.035"
  "0.65 0.55 0.35 0.075 0.35 0.045 0.035"
  "0.75 0.54 0.34 0.085 0.40 0.050 0.040"
  "0.85 0.52 0.32 0.095 0.45 0.055 0.045"
)

i=0
for r in "${RUNS[@]}"; do
  i=$((i+1))
  read -r sx hi lo mc sy hm sg <<< "$r"
  TAG="${i}-sx${sx}_hi${hi}_lo${lo}_mc${mc}_sy${sy}_hm${hm}_sg${sg}"
  TAG="${TAG//./p}"
  RUN_DIR="$MASTER/$TAG"
  mkdir -p "$RUN_DIR"
  OUT="$RUN_DIR/dunhuangwu2_v42_2_ROOT_FOOTPLANT_ref.npy"
  JSON_OUT="$RUN_DIR/dunhuangwu2_v42_2_ROOT_FOOTPLANT_ref.v42_2_physics.json"
  TARGETS="$RUN_DIR/dunhuangwu2_v42_2_ROOT_FOOTPLANT_ref.v42_2_targets.npz"

  echo "============================================================"
  echo "[V42.2 RUN] $TAG"
  echo "  root_xz_strength=$sx contact_high=$hi contact_low=$lo max_corr=$mc root_y_strength=$sy height_margin=$hm speed_gate=$sg"
  echo "============================================================"

  python tools/v42_root_footplant_physics_optimizer.py \
    --input "$INPUT" \
    --output "$OUT" \
    --json "$JSON_OUT" \
    --targets "$TARGETS" \
    --config "$CFG" \
    --root_xz_strength "$sx" \
    --contact_high "$hi" \
    --contact_low "$lo" \
    --max_correction "$mc" \
    --root_y_strength "$sy" \
    --height_margin "$hm" \
    --speed_gate_mpf "$sg"
done

python - <<'PY'
import json, glob, os, shutil
final_dir = open('output/LATEST_FINAL_DUNHUANG_ACCEPTED.txt').read().strip()
master = open('output/LATEST_V42_2_DUNHUANGWU2_SWEEP.txt').read().strip()
rows=[]
for f in glob.glob(master+'/*/*.v42_2_physics.json'):
    d=json.load(open(f))
    post=d.get('post_audit',{})
    pf=d.get('planner_feedback',{})
    rb=d.get('rollback',{})
    rows.append({
        'json': f,
        'dir': os.path.dirname(f),
        'accepted': bool(pf.get('accepted')),
        'reject': pf.get('reject_reasons'),
        'rollback': bool(rb.get('triggered')),
        'skate': float(post.get('foot_skate_p95_mpf', 999)),
        'pen': float(post.get('foot_penetration_min_m', -999)),
        'jerk': float(post.get('mean_joint_jerk_p95', 999)),
        'rootxz': float(d.get('root_xz_delta_max', 999)),
    })
if not rows:
    print('[NO RESULTS]')
    raise SystemExit(1)
rows.sort(key=lambda r: (
    not r['accepted'],
    r['rollback'],
    r['skate'],
    abs(r['pen']),
    r['jerk'],
))
best=rows[0]
print('[BEST]', json.dumps(best, ensure_ascii=False, indent=2))
base='dunhuangwu2_v42_2_ROOT_FOOTPLANT_ref'
for ext, dstext in [('.npy','.npy'),('.v42_2_physics.json','.v42_2_physics.json'),('.v42_2_targets.npz','.v42_2_targets.npz')]:
    src=os.path.join(best['dir'], base+ext)
    dst=os.path.join(final_dir, 'dunhuangwu2_v42_2_ROOT_FOOTPLANT_BEST_ref'+dstext)
    if os.path.exists(src):
        shutil.copy2(src,dst)
        print('[COPIED]', dst)
summary=os.path.join(master,'V42_2_SWEEP_SUMMARY.json')
json.dump({'master':master,'best':best,'rows':rows}, open(summary,'w'), ensure_ascii=False, indent=2)
print('[SUMMARY]', summary)
PY

BEST_NPY="$FINAL_DIR/dunhuangwu2_v42_2_ROOT_FOOTPLANT_BEST_ref.npy"
BEST_MP4="$FINAL_DIR/dunhuangwu2_v42_2_ROOT_FOOTPLANT_BEST_ref.mp4"
if [[ -f "$BEST_NPY" && -f "$AUDIO" && -f render_from_npy.py ]]; then
  python render_from_npy.py \
    --motion "$BEST_NPY" \
    --audio "$AUDIO" \
    --output "$BEST_MP4" \
    --camera_mode follow \
    --render_smooth_window 5
fi

echo "[V42.2 SWEEP DONE] $(date)"
