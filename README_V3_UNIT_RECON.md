# V3 Temporal Unit Reconstruction / Unit Prior

## Why this patch exists

V2B/V2D/V2E/V2F showed that sparse endpoint/keyframe-driven training can improve some metrics while still producing bad videos. The model learned shortcuts: early endpoint arrival, endpoint jitter, and burst transitions.

V3 changes the primary problem:

```text
old: pose A -> pose B under sparse anchors
new: reconstruct the full 45-frame Dunhuang motion unit
```

The goal is to learn a temporal motion prior before adding endpoint, RAG, beat, or trajectory control.

## Files

```text
train.py
unit_reconstruction_patch.py
scripts/run_v3_unit_recon_best5.sh
scripts/run_v3_unit_recon_whitelist.sh
tools/eval_v3_unit_recon_metrics.py
README_V3_UNIT_RECON.md
```

## Install

From this pack, copy files into the EDGE repository root:

```bash
cp train.py /home/disk/lsm/storage/EDGE/train.py
cp unit_reconstruction_patch.py /home/disk/lsm/storage/EDGE/unit_reconstruction_patch.py
cp scripts/run_v3_unit_recon_best5.sh /home/disk/lsm/storage/EDGE/scripts/
cp scripts/run_v3_unit_recon_whitelist.sh /home/disk/lsm/storage/EDGE/scripts/
cp tools/eval_v3_unit_recon_metrics.py /home/disk/lsm/storage/EDGE/tools/
chmod +x /home/disk/lsm/storage/EDGE/scripts/run_v3_unit_recon_*.sh
chmod +x /home/disk/lsm/storage/EDGE/tools/eval_v3_unit_recon_metrics.py
```

Back up your old train.py first:

```bash
cp /home/disk/lsm/storage/EDGE/train.py /home/disk/lsm/storage/EDGE/train.py.bak_v2f
```

## Run V3-A best5 sanity

```bash
cd /home/disk/lsm/storage/EDGE
tmux new -s v3_unit_recon_best5
bash scripts/run_v3_unit_recon_best5.sh
```

Override checkpoint/data when needed:

```bash
BASE_CKPT=runs/train_nextgen/strict_single_unit45_recon_v16_smooth_dense8_from_v15/weights/train-5000.pt \
DATA_PATH=data/dunhuang_bvh/stationary_whitelist_v2d_nostatic_best5 \
bash scripts/run_v3_unit_recon_best5.sh
```

## Run V3-B 20-50 unit whitelist

```bash
cd /home/disk/lsm/storage/EDGE
tmux new -s v3_unit_recon_whitelist
DATA_PATH=data/dunhuang_bvh/stationary_whitelist_v2d_30_to_50_units \
BATCH_SIZE=4 \
bash scripts/run_v3_unit_recon_whitelist.sh
```

## Important switches

```bash
export EDGE_TRAIN_PROFILE=v3_unit_recon
export EDGE_V3_UNIT_RECON=1

export EDGE_X0_RECON_LOSS=1
export EDGE_X0_RECON_LOSS_WEIGHT=0.8

export EDGE_V3_TEMPORAL_WEIGHT=0.20
export EDGE_V3_DCT_KEEP=6
export EDGE_V3_TEMPORAL_FEATURES=no_contact
```

`EDGE_V3_TEMPORAL_FEATURES` options:

```text
all
no_contact
rot
upper_torso
rootxz
```

Default `no_contact` uses root xyz + body rotations, excluding contact channels.

## What is deliberately disabled in V3

```text
hard keyframe projection
start/end/mid keyframe training
trajectory condition/loss
trajectory-event condition
weak trajectory energy guidance
beat guidance
Text/Pose Context RAG
unit soft prior
freeze-aware / progress / burst-safe V2 loss patches
MMR audio-motion loss
```

## Evaluation principle

Do not evaluate endpoint generation first. V3 success means:

```text
reconstruction sample is natural
unconditional / weak-conditioned sample does not burst
full 45-frame unit keeps Dunhuang upper/torso line
root remains stable for stationary units
jump/jerk are not only numerically low but visually smooth
```

Use the metric helper:

```bash
python tools/eval_v3_unit_recon_metrics.py \
  --pred output/some_v3_sample.npy \
  --gt data/some_gt_unit.npy
```

For a directory:

```bash
python tools/eval_v3_unit_recon_metrics.py \
  --pred_dir output/v3_eval \
  --out_csv output/v3_eval/v3_metrics.csv
```

## Recommended route

```text
1. V3-A best5 full-unit reconstruction sanity
2. V3-B 20-50 high-quality stationary unit reconstruction
3. Only after videos are natural: soft endpoint / soft anchor
4. Then DCT unit prior / Text-Pose RAG
5. Finally weak beat/music and trajectory-event control
```
