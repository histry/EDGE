#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/disk/lsm/storage/EDGE
cd "$ROOT"
export PATH="/home/disk/lsm/conda_envs/edge/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN="${V23_RUN_ROOT:?V23_RUN_ROOT required}"
DATA="${V23_DATASET:-data/v23_monotonic_duration_dataset.npz}"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/overnight.log") 2>&1
trap 'code=$?; echo "[V23 ERROR] code=$code line=${BASH_LINENO[0]} cmd=$BASH_COMMAND"; exit $code' ERR

python -m py_compile \
  model/v23_monotonic_duration.py \
  tools/build_v23_monotonic_duration_dataset.py \
  train_v23_monotonic_duration.py \
  tools/smoke_v23_monotonic_duration.py
python tools/smoke_v23_monotonic_duration.py

if [[ ! -s "$DATA" || "${V23_REBUILD_DATASET:-1}" == "1" ]]; then
  rm -f "$DATA" "${DATA%.npz}.metadata.json"
  bash scripts/run_v23_build_dataset.sh
fi

python - "$DATA" <<'PY'
import sys, numpy as np
p=sys.argv[1]
with np.load(p, allow_pickle=True) as z:
    n=len(z['corrupted'])
    print('samples=', n)
    print('motion=', z['corrupted'].shape)
    print('condition=', z['condition'].shape)
    print('sources=', len(np.unique(z['source_id'])))
    print('target duration p10/p50/p90=', np.percentile(z['target_duration_frames'], [10,50,90]))
    print('corrupt duration p10/p50/p90=', np.percentile(z['corrupted_duration_frames'], [10,50,90]))
    print('factor p10/p50/p90/max=', np.percentile(z['speed_factor'], [10,50,90,100]))
    assert n >= 500
    assert z['target_tau'].shape == z['edit_mask'].shape
    assert np.isfinite(z['target_tau']).all()
    assert np.all(np.diff(z['target_tau'], axis=1) >= -1e-5)
print('[PASS] V23 dataset')
PY

for SEED in 20260606 20260607 20260608; do
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
for p in sorted(root.glob('seed_*/checkpoints/best.pt')):
    x=torch.load(p,map_location='cpu',weights_only=False)
    rows.append((float(x.get('val_loss',1e9)),int(x.get('epoch',-1)),p))
if not rows: raise RuntimeError('No checkpoints')
rows.sort(key=lambda x:x[0])
for i,(v,e,p) in enumerate(rows,1): print(i,v,e,p)
best=rows[0][2].resolve()
(root/'BEST_V23_CKPT.txt').write_text(str(best)+'\n',encoding='utf-8')
(root/'CHECKPOINT_RANKING.tsv').write_text('val_loss\tepoch\tcheckpoint\n'+'\n'.join(f'{v:.10f}\t{e}\t{p}' for v,e,p in rows)+'\n',encoding='utf-8')
print('BEST_V23_CKPT=',best)
PY

echo "DONE: $RUN"
echo "BEST: $(cat "$RUN/BEST_V23_CKPT.txt")"
