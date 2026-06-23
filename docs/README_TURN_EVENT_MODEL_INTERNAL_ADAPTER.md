# Turn-aware Event Conditioning inside EDGE Decoder

This patch moves the turn-aware event idea from external residual postprocess into a model-internal, environment-gated EDGE decoder adapter.

## What it adds

1. `turn_aware_event_utils.py`
   - trajectory event detection
   - event feature matrix `[x,z,speed,heading,curvature,turn_gate,support_gate,expressive_gate,...]`

2. `turn_event_model_adapter_patch.py`
   - patches `model.model.DanceDecoder`
   - computes event features internally from `cond["trajectory"]`
   - injects event features into trajectory tokens via zero-init residual projection
   - optionally loads an output adapter checkpoint into the decoder forward pass

3. `tools/train_turn_event_internal_adapter.py`
   - trains the same internal output adapter class on multi pseudo-targets
   - anchor is the no-train turn-aware motion
   - loss is event-weighted and body-part aware

4. `sitecustomize.py`
   - installs the patch automatically, but all behavior remains disabled unless env flags are set.

## Main env switches

```bash
export EDGE_TURN_EVENT_MODEL_ADAPTER=1      # enable model-side event adapter
export EDGE_TURN_EVENT_TRAJ_TOKEN=1         # inject event features into trajectory tokens
export EDGE_TURN_EVENT_OUTPUT_ADAPTER=1     # use optional output adapter checkpoint
export EDGE_TURN_EVENT_ADAPTER_CKPT=runs/turn_event_internal_adapter/turn_event_internal_adapter.pt
export EDGE_TURN_EVENT_FREEZE_BACKBONE=1    # when using normal train.py adapter-only training
export EDGE_DYNAMIC_TRAJ_CFG=0              # recommended safe default
```

Trajectory event timing:

```bash
export EDGE_TURN_SUPPORT_LAG=8
export EDGE_TURN_EXPR_LAG=4
export EDGE_TURN_MIN_GAP=18
export EDGE_TURN_GATE_SIGMA=5.0
```

## Fast low-risk training

This trains only the internal output adapter on pseudo targets:

```bash
bash scripts/run_train_turn_event_internal_adapter.sh
bash scripts/render_turn_event_internal_adapter.sh
```

## Use inside generate_controlled.py

```bash
export EDGE_TURN_EVENT_MODEL_ADAPTER=1
export EDGE_TURN_EVENT_TRAJ_TOKEN=1
export EDGE_TURN_EVENT_OUTPUT_ADAPTER=1
export EDGE_TURN_EVENT_ADAPTER_CKPT=runs/turn_event_internal_adapter/turn_event_internal_adapter.pt
export EDGE_DYNAMIC_TRAJ_CFG=0

bash scripts/run_generate_with_turn_event_internal_adapter.sh
```

## Full/native adapter training with existing train.py

If your existing training script supports trajectory conditions, enable:

```bash
export EDGE_TURN_EVENT_MODEL_ADAPTER=1
export EDGE_TURN_EVENT_TRAJ_TOKEN=1
export EDGE_TURN_EVENT_OUTPUT_ADAPTER=0
export EDGE_TURN_EVENT_FREEZE_BACKBONE=1
export EDGE_DYNAMIC_TRAJ_CFG=0
```

Then run your normal Stage2/trajectory training script. The patch computes event features from the trajectory automatically, so no dataset column is required.

