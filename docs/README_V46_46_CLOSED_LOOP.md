# V46.46 Closed-Loop Boundary-Safe Scheduler

Copy these files into the EDGE root:

```bash
cp tools/v46_46_boundary_closed_loop.py /home/disk/lsm/storage/EDGE/tools/
cp scripts/run_v46_46_closed_loop_after_v46_45.sh /home/disk/lsm/storage/EDGE/scripts/
cp configs/v46_46_closed_loop.env.example /home/disk/lsm/storage/EDGE/configs/
```

Recommended run after V46.44/V46.45 contract fixes:

```bash
cd /home/disk/lsm/storage/EDGE
source configs/v46_46_closed_loop.env.example
bash scripts/run_v46_46_closed_loop_after_v46_45.sh
```

Outputs include:

- `*.npy`: final motion
- `*.motion_ref.npy`: closed-loop reference motion before refiner/diffusion/IK
- `*.transition_mask.npy`: transition mask
- `*.report.json`: full generation report
- `*.boundary_audit.csv`: paper-ready boundary-level audit table
- `*.boundary_audit.json`: full per-boundary risk details

Core research changes:

1. search-time lightweight boundary simulation;
2. risk-adaptive transition length;
3. local candidate reselection for unsafe boundaries;
4. unified boundary audit table.
