# TEA-MotionAdapter Patch for EDGE

This patch implements the retraining-first plan:

1. Native trajectory adapter training
   - Uses existing `ZeroInitTrajectoryAdapter`, `trajectory_encoder`, `traj_modulate`, and `root_generator`.
   - Adds `--train_stage adapter`.
   - Freezes the pretrained motion prior and trains only lightweight control branches by default.

2. Energy-conditioned CFG
   - Dataset returns `cond["energy"]`.
   - `DanceDecoder` maps energy scalar through an MLP and adds it to timestep/global conditioning.
   - Inference can set:
     - `EDGE_ENERGY_COND=1`
     - `EDGE_ENERGY_LEVEL=0.0..1.0`
     - `EDGE_ENERGY_CFG_SCALE=...`

3. Root-lower coupling
   - Replaces broad kinematic sync with lower-body-focused coupling when root X/Z velocity is high.
   - Uses existing `sync_loss_weight` plus:
     - `--root_lower_coupling_loss_weight`
     - `--root_lower_speed_threshold`
     - `--root_lower_min_motion`

## Install

```bash
cd /home/disk/lsm/storage/EDGE
python install_tea_motion_adapter_patch.py
python -m py_compile args.py train.py EDGE.py dataset/dance_dataset.py model/model.py model/diffusion.py generate_controlled.py
```

## Stage A training

Edit checkpoint path in `env_tea_adapter_stageA.sh`, then:

```bash
bash env_tea_adapter_stageA.sh
```

## Energy-conditioned inference

```bash
source env_tea_inference_energy.sh
python generate_controlled.py ...
```

## Recommended ablation order

A. Adapter only:
- `--train_stage adapter`
- `--energy_condition_prob 0`
- root-lower coupling on.

B. Adapter + energy:
- `--energy_condition_prob 0.7`
- use `EDGE_ENERGY_LEVEL` in inference.

C. Future:
- RAG context token only after A/B are stable.
