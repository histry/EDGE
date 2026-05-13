# Turn-aware Event Refiner v2

This patch implements the next step after validating turn-aware event conditioning:

1. Multi pseudo-target training
2. Event-weighted body-part losses
3. Small bounded residual on top of the no-train turn-aware compositor output

The model now learns:

```text
motion_final = no_train_turn_event + small_delta(event, base, no_train_turn_event)
```

instead of:

```text
motion_final = base + full_residual
```

## Main commands

```bash
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export EDGE_TURN_EVENT_REFINER_TRAIN=1
export EDGE_TURN_REFINER_EPOCHS=500
export EDGE_TURN_REFINER_MAX_DELTA=0.14
bash scripts/run_train_turn_event_refiner_v2.sh
```

Render:

```bash
bash scripts/render_turn_event_refiner_v2.sh
```

Optional grid:

```bash
bash scripts/run_turn_event_refiner_v2_grid.sh
```

## Expected input files

The v2 script uses any existing targets from:

```text
output/v13_frame_sweep_hybrid/dhw4_v13_f35_60_85_110_135_mild.npy
output/v13_frame_sweep_hybrid/dhw4_v13_f40_65_90_115_140_mild.npy
output/v13_turn_event_hybrid/dhw4_turn_event_v13.npy
output/v13_functional_hybrid_sweep/dhw4_v13_mild.npy
```

At least two of them should exist.
