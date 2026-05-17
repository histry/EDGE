# EDGE V2F Burst-Safe Patch

This patch targets the reward-hacking failure found after V2E: endpoint collapse
metrics become good, but videos show repeated rapid twitches. The uploaded log
shows V2E has `endpoint_collapse_bad=False` for e100/e200, while
`burst_jitter_bad=True` for all units with `burst_count=8` and high top-4
velocity share.

Files:
- freeze_aware_motion_patch.py
- scripts/run_train_v2f_burstsafe_best5.sh
- scripts/eval_v2f_burstsafe_best5.sh
- tools/diagnose_bursty_jitter_units.py

Install:
```bash
cd /home/disk/lsm/storage/EDGE
mkdir -p backup_before_v2f
cp freeze_aware_motion_patch.py backup_before_v2f/
cp tools/diagnose_bursty_jitter_units.py backup_before_v2f/ 2>/dev/null || true
unzip /mnt/data/edge_v2f_burstsafe_patch.zip -d /tmp/edge_v2f_patch
cp -r /tmp/edge_v2f_patch/* .
chmod +x scripts/run_train_v2f_burstsafe_best5.sh
chmod +x scripts/eval_v2f_burstsafe_best5.sh
chmod +x tools/diagnose_bursty_jitter_units.py
python -m py_compile freeze_aware_motion_patch.py tools/diagnose_bursty_jitter_units.py
```

Verify:
```bash
export EDGE_FREEZE_AWARE_MOTION=1
export EDGE_BURST_SAFE_PROGRESS=1
export EDGE_KINEMATIC_SMOOTHNESS=1
export EDGE_DIRECTION_CONSISTENCY=1
export EDGE_LOWPASS_TEMPORAL_PRIOR=1
python - <<'PY'
import inspect
import freeze_aware_motion_patch as p
from freeze_aware_motion_patch import install_freeze_aware_motion_patch
install_freeze_aware_motion_patch(verbose=True)
from model.diffusion import GaussianDiffusion
src = inspect.getsource(GaussianDiffusion._motion_energy_loss)
print("saturate:", "_saturate_energy" in inspect.getsource(p._temporal_progress_loss))
print("kinematic:", "_kinematic_smoothness_loss" in src)
print("direction:", "_directional_consistency_loss" in src)
print("lowpass:", "_lowpass_temporal_loss" in src)
PY
```

Train:
```bash
tmux new -s v2f_burstsafe
bash scripts/run_train_v2f_burstsafe_best5.sh
```

Evaluate after train-50.pt:
```bash
export EXP_NAME=stationary_v2f_best5_burstsafe_x0w030
export STEPS="50 100 150 200 250 300"
bash scripts/eval_v2f_burstsafe_best5.sh
```

Success criteria:
- burst_jitter_bad decreases from 5/5.
- burst_count < 4.
- top4_velocity_share < 0.55.
- unit_69373 no longer shows pose1 -> pose2 -> four rapid twitches.
