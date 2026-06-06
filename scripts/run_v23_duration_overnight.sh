#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/disk/lsm/storage/EDGE
cd "$ROOT"
export PATH="/home/disk/lsm/conda_envs/edge/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN="${V23_RUN_ROOT:?V23_RUN_ROOT required}"
DATA="${V23_DATASET:-data/v23_v2_3_slowaware_w120_d88.npz}"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/overnight.log") 2>&1
trap 'code=$?; echo "[V23-v2.3 ERROR] code=$code line=${BASH_LINENO[0]} cmd=$BASH_COMMAND"; exit $code' ERR

python -m py_compile \
  model/v23_monotonic_duration.py \
  tools/v23_duration_utils.py \
  tools/build_v23_monotonic_duration_dataset.py \
  train_v23_monotonic_duration.py \
  tools/evaluate_v23_checkpoint.py \
  tools/apply_v23_monotonic_duration.py \
  tools/smoke_v23_monotonic_duration.py
python tools/smoke_v23_monotonic_duration.py \
  --window_len "${V23_WINDOW_LEN:-120}" \
  --duration_edges "${V23_SMOKE_DURATION_EDGES:-12,24,37,50,63,76,89}"

if [[ ! -s "$DATA" || "${V23_REBUILD_DATASET:-0}" == "1" ]]; then
  rm -f "$DATA" "${DATA%.npz}.metadata.json"
  bash scripts/run_v23_build_dataset.sh
fi

python - "$DATA" <<'PY'
import sys
import numpy as np
path=sys.argv[1]
with np.load(path, allow_pickle=True) as z:
    required=['corrupted','target','condition','target_tau','source_id','target_duration_frames',
              'speed_factor','is_identity','duration_bin']
    for key in required:
        assert key in z.files, key
    n=len(z['corrupted'])
    d=z['target_duration_frames']
    bins=z['duration_bin'].astype(int)
    identity=z['is_identity']>0.5
    print('samples=',n,'motion=',z['corrupted'].shape,'sources=',len(np.unique(z['source_id'])))
    print('identity=',identity.mean(),'duration=',np.percentile(d,[0,10,25,50,75,90,100]))
    print('bins=',np.bincount(bins),'factor=',np.percentile(z['speed_factor'],[0,10,50,90,100]))
    assert n >= 5000
    assert z['condition'].shape[1] == 17
    assert 0.18 <= identity.mean() <= 0.32
    assert len(np.bincount(bins)) >= 4
    assert np.bincount(bins).max()/n <= 0.45
    assert np.all(np.diff(z['target_tau'],axis=1)>=-1e-5)
print('[PASS] V23-v2.3 dataset')
PY

SEEDS="${V23_SEEDS:-20260610 20260611 20260612}"
for SEED in $SEEDS; do
  export V23_SEED="$SEED"
  export V23_OUT_DIR="$RUN/seed_${SEED}"
  bash scripts/run_v23_train_one_seed.sh 2>&1 | tee "$RUN/train_seed_${SEED}.log"
done

python - "$RUN" <<'PY'
import sys
from pathlib import Path
import torch
root=Path(sys.argv[1])
rows=[]
for path in sorted(root.glob('seed_*/stage2_timewarp/checkpoints/best.pt')):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    metrics=ckpt.get('val_metrics',{})
    rows.append((float(ckpt.get('selection_score',1e9)),float(ckpt.get('val_loss',1e9)),
                 int(ckpt.get('epoch',-1)),path,metrics))
if not rows:
    raise RuntimeError('No stage-2 checkpoints')
rows.sort(key=lambda row:row[0])
header='rank\tselection_score\tval_loss\tepoch\tduration_mae\tduration_corr\ttau_mae\tmotion_ratio\tyaw_ratio\tcheckpoint\n'
lines=[]
for rank,(score,val,epoch,path,m) in enumerate(rows,1):
    print(rank,score,val,epoch,m.get('duration_mae_frames'),m.get('duration_correlation'),path)
    lines.append(f"{rank}\t{score:.10f}\t{val:.10f}\t{epoch}\t"
                 f"{float(m.get('duration_mae_frames',float('nan'))):.6f}\t"
                 f"{float(m.get('duration_correlation',float('nan'))):.6f}\t"
                 f"{float(m.get('tau_mae',float('nan'))):.6f}\t"
                 f"{float(m.get('motion_mse_ratio',float('nan'))):.6f}\t"
                 f"{float(m.get('yaw_mae_ratio',float('nan'))):.6f}\t{path}")
best=rows[0][3].resolve()
(root/'BEST_V23_CKPT.txt').write_text(str(best)+'\n',encoding='utf-8')
(root/'CHECKPOINT_RANKING.tsv').write_text(header+'\n'.join(lines)+'\n',encoding='utf-8')
print('BEST_V23_CKPT=',best)
PY

BEST=$(cat "$RUN/BEST_V23_CKPT.txt")
python tools/evaluate_v23_checkpoint.py \
  --data "$DATA" \
  --checkpoint "$BEST" \
  --out_dir "$RUN/heldout_eval_best" \
  --batch_size "${V23_EVAL_BATCH_SIZE:-48}" \
  --max_samples "${V23_EVAL_MAX_SAMPLES:-4096}"

echo "DONE: $RUN"
echo "BEST: $BEST"
echo "EVAL: $RUN/heldout_eval_best/V23_V2_3_HELDOUT_EVALUATION.json"
