# V46.32 transition-budgeted inbetweening usage

Copy the `tools/` and `scripts/` folders into `/home/disk/lsm/storage/EDGE`, then run:

```bash
cd /home/disk/lsm/storage/EDGE
cp /path/to/tools/v46_32_transition_budget_patch.py tools/
cp /path/to/tools/v46_32_relabel_change_event_db.py tools/
cp /path/to/scripts/run_v46_32_alltrain_transition_budget_overnight.sh scripts/
bash scripts/run_v46_32_alltrain_transition_budget_overnight.sh
```

Main environment switches:

```bash
V46_TRANSITION_BUDGET_ENABLE=1      # enable new transition-budgeted concat
V46_TRANSITION_BUDGET_ENABLE=0      # fall back to preserved V46.31 overlap concat
V46_TRANSITION_MIN_FRAMES=10
V46_TRANSITION_MAX_FRAMES=28
V46_TRANSITION_RATIO=0.18
V46_TRANSITION_MIN_CORE_FRAMES=30
V46_CORE_WARP_MIN=0.72
V46_CORE_WARP_MAX=1.38
```

The run script rebuilds the all-train DB, relabels event semantics, trains V44/V45/V46, and generates both:

- `dunhuangwu2_v46_32_transition_refiner_ik.mp4`
- `dunhuangwu2_v46_32_transition_diffusion_ik.mp4`
