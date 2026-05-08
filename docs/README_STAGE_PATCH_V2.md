# ChoreoRAG Stage Patch v2

一次性补丁包，用环境变量按阶段打开，不改变默认行为。

## 安装

```bash
cd /home/disk/lsm/storage/EDGE
python /path/to/apply_choreorag_stage_patch_v2.py
python -m py_compile generate_controlled.py model/model.py trajectory_native_control.py choreorag_unit_prior.py
git diff --stat
```

## Stage B：Energy / Expressiveness-aware Retrieval

```bash
export EDGE_MOTION_UNIT_MODE=on
export EDGE_TEXT_BRIDGE_WEIGHT=0.80
export EDGE_UNIT_ENTRY_WEIGHT=0.30
export EDGE_UNIT_EXIT_WEIGHT=0.30
export EDGE_UNIT_CONTACT_PHASE_WEIGHT=0.35

export EDGE_UNIT_MIN_EXPRESSIVENESS=0.45
export EDGE_UNIT_EXPRESSIVENESS_BONUS=1.50
export EDGE_UNIT_MIN_ENERGY=0.35
export EDGE_UNIT_ENERGY_BONUS=1.20
export EDGE_TENSION_AWARE_PLANNER=1
```

生成 high-energy plan：

```bash
python make_highenergy_choreo_plan.py   --in output/choreo_plan/dunhuangwu2_plan.json   --out output/choreo_plan/dunhuangwu2_demo_highenergy_plan.json   --frame 115
```

## Stage B2：Energy-Speed Coherent CFG

```bash
export EDGE_ENERGY_COND=1
export EDGE_ENERGY_TIME_DEPENDENT=1
export EDGE_ENERGY_CFG_SCALE=0.8
export EDGE_ENERGY_LEVEL=0.55
export EDGE_ENERGY_MIN=0.20
export EDGE_ENERGY_MAX=0.85
export EDGE_ENERGY_TRAJ_SPEED_WEIGHT=0.55
export EDGE_ENERGY_AUDIO_WEIGHT=0.30
export EDGE_ENERGY_CURVATURE_WEIGHT=0.15
```

## Stage C：Frequency-Decoupled Retrieved Unit Soft Prior

定量安全版：

```bash
export EDGE_UNIT_SOFT_PRIOR=1
export EDGE_UNIT_PRIOR_FEATURES=upper
export EDGE_UNIT_PRIOR_STRENGTH=0.025
export EDGE_UNIT_PRIOR_MAX_LEN=45
export EDGE_UNIT_PRIOR_DECAY_GAMMA=1.5
export EDGE_UNIT_PRIOR_DCT=1
export EDGE_UNIT_PRIOR_LOW_FREQ_K=6
```

定性展示版：

```bash
export EDGE_UNIT_SOFT_PRIOR=1
export EDGE_UNIT_PRIOR_FEATURES=loco_upper
export EDGE_UNIT_PRIOR_STRENGTH=0.040
export EDGE_UNIT_PRIOR_MAX_LEN=45
export EDGE_UNIT_PRIOR_DECAY_GAMMA=1.2
export EDGE_UNIT_PRIOR_DCT=1
export EDGE_UNIT_PRIOR_LOW_FREQ_K=8
```

## Stage A：Explicit Root-Lower Velocity Matching

训练时启用：

```bash
export EDGE_EXPLICIT_LOWER_VEL_LOSS_WEIGHT=1.0
export EDGE_EXPLICIT_LOWER_ROOT_SPEED_THRESHOLD=0.006
export EDGE_EXPLICIT_LOWER_RELVEL_RATIO=0.75
export EDGE_EXPLICIT_LOWER_MIN_MOTION=0.003
```

配合：

```bash
--train_stage adapter --adapter_train_decoder
```

## 回滚

```bash
cp _backup_choreorag_stage_patch_v2/generate_controlled.py generate_controlled.py
cp _backup_choreorag_stage_patch_v2/model.py model/model.py
cp _backup_choreorag_stage_patch_v2/trajectory_native_control.py trajectory_native_control.py
cp _backup_choreorag_stage_patch_v2/choreorag_unit_prior.py choreorag_unit_prior.py
```
