#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/disk/lsm/storage/EDGE
cd "$ROOT"
export PATH="/home/disk/lsm/conda_envs/edge/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN="${V23_RUN_ROOT:?V23_RUN_ROOT required}"
DATA="${V23_DATASET:-data/v23_v2_natural_duration_dataset.npz}"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/overnight.log") 2>&1
trap 'code=$?; echo "[V23-v2 ERROR] code=$code line=${BASH_LINENO[0]} cmd=$BASH_COMMAND"; exit $code' ERR

python -m py_compile \
  model/v23_monotonic_duration.py \
  tools/v23_duration_utils.py \
  tools/build_v23_monotonic_duration_dataset.py \
  train_v23_monotonic_duration.py \
  tools/evaluate_v23_checkpoint.py \
  tools/apply_v23_monotonic_duration.py \
  tools/smoke_v23_monotonic_duration.py
python tools/smoke_v23_monotonic_duration.py

if [[ ! -s "$DATA" || "${V23_REBUILD_DATASET:-1}" == "1" ]]; then
  rm -f "$DATA" "${DATA%.npz}.metadata.json"
  bash scripts/run_v23_build_dataset.sh
fi

python - "$DATA" <<'PY'
import json
import os
import sys
import numpy as np

path = sys.argv[1]
expected = int(os.environ.get('V23_MAX_SAMPLES', '12000'))
metadata_path = path.replace('.npz', '.metadata.json')
with np.load(path, allow_pickle=True) as z:
    required = [
        'corrupted', 'target', 'condition', 'target_tau', 'source_id',
        'target_duration_frames', 'corrupted_duration_frames', 'speed_factor',
        'is_identity', 'duration_bin',
    ]
    for key in required:
        assert key in z.files, key
    n = len(z['corrupted'])
    duration = z['target_duration_frames']
    identity = z['is_identity'] > 0.5
    bins = z['duration_bin'].astype(int)
    counts = np.bincount(bins, minlength=bins.max() + 1)
    percentiles = np.percentile(duration, [0,10,25,50,75,90,100])
    print('samples=', n, 'expected=', expected)
    print('motion=', z['corrupted'].shape)
    print('condition=', z['condition'].shape)
    print('sources=', len(np.unique(z['source_id'])))
    print('identity ratio=', identity.mean())
    print('duration p0/p10/p25/p50/p75/p90/p100=', percentiles)
    print('duration bins=', counts)
    print('factor p0/p10/p50/p90/max=', np.percentile(z['speed_factor'], [0,10,50,90,100]))
    if os.path.isfile(metadata_path):
        meta = json.load(open(metadata_path, 'r', encoding='utf-8'))
        print('duration edges=', meta.get('duration_edges'))
        print('raw event duration=', meta.get('raw_event_duration_percentiles'))
        print('raw event bins=', meta.get('raw_event_bin_counts'))
    assert n == expected, (n, expected)
    assert z['condition'].shape[1] == 17
    assert 0.18 <= identity.mean() <= 0.32, identity.mean()
    assert len(np.unique(duration)) >= 10, np.unique(duration)
    nonempty = counts[counts > 0]
    assert len(nonempty) >= 4, counts
    assert nonempty.max() / n <= 0.50, counts
    assert percentiles[5] - percentiles[1] >= 8.0, percentiles
    assert percentiles[-1] - percentiles[0] >= 12.0, percentiles
    assert np.isfinite(z['target_tau']).all()
    assert np.all(np.diff(z['target_tau'], axis=1) >= -1e-5)
print('[PASS] V23-v2.1 balanced full-body natural-duration dataset')
PY
for SEED in 20260610 20260611 20260612; do
  export V23_SEED="$SEED"
  export V23_OUT_DIR="$RUN/seed_${SEED}"
  bash scripts/run_v23_train_one_seed.sh 2>&1 | tee "$RUN/train_seed_${SEED}.log"
done

python - "$RUN" <<'PY'
import sys
from pathlib import Path
import torch

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob('seed_*/checkpoints/best.pt')):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    score = float(checkpoint.get('selection_score', 1e9))
    val_loss = float(checkpoint.get('val_loss', 1e9))
    epoch = int(checkpoint.get('epoch', -1))
    metrics = checkpoint.get('val_metrics', {})
    rows.append((score, val_loss, epoch, path, metrics))
if not rows:
    raise RuntimeError('No checkpoints')
rows.sort(key=lambda row: row[0])
for rank, (score, val_loss, epoch, path, metrics) in enumerate(rows, 1):
    print(rank, score, val_loss, epoch, metrics.get('duration_mae_frames'), metrics.get('duration_correlation'), path)
best = rows[0][3].resolve()
(root / 'BEST_V23_CKPT.txt').write_text(str(best) + '\n', encoding='utf-8')
header = 'selection_score\tval_loss\tepoch\tduration_mae\tduration_corr\ttau_mae\tactivity_ratio\tcheckpoint\n'
lines = []
for score, val_loss, epoch, path, metrics in rows:
    lines.append(
        f"{score:.10f}\t{val_loss:.10f}\t{epoch}\t"
        f"{float(metrics.get('duration_mae_frames', float('nan'))):.6f}\t"
        f"{float(metrics.get('duration_correlation', float('nan'))):.6f}\t"
        f"{float(metrics.get('tau_mae', float('nan'))):.6f}\t"
        f"{float(metrics.get('activity_ratio', float('nan'))):.6f}\t{path}"
    )
(root / 'CHECKPOINT_RANKING.tsv').write_text(header + '\n'.join(lines) + '\n', encoding='utf-8')
print('BEST_V23_CKPT=', best)
PY

BEST=$(cat "$RUN/BEST_V23_CKPT.txt")
python tools/evaluate_v23_checkpoint.py \
  --data "$DATA" \
  --checkpoint "$BEST" \
  --out_dir "$RUN/heldout_eval_best" \
  --batch_size "${V23_EVAL_BATCH_SIZE:-64}" \
  --max_samples "${V23_EVAL_MAX_SAMPLES:-4096}"

echo "DONE: $RUN"
echo "BEST: $BEST"
echo "EVAL: $RUN/heldout_eval_best/V23_V2_HELDOUT_EVALUATION.json"
