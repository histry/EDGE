# V10 Env Isolation Fix

## Fixed bug

If `EDGE_V10_MANUAL_UNITS` was exported for Step2, Step1/Step3/Step4 could
accidentally reuse the manual units. Step4 has three frames but the manual
example often has two units, causing:

```text
ValueError: planned_units count 2 != frames count 3
```

## Permanent fixes

1. `v10_choreo_planner.py`
   - Manual env variables are honored only when `EDGE_V10_MODE=manual_multiunit`.
   - Non-manual modes print an informational message and ignore stale manual env.

2. Non-manual scripts
   - `run_v10_step1_dual_auto_mid.sh`
   - `run_v10_step3_upperdance_rag.sh`
   - `run_v10_step4_auto_multiunit.sh`
   - `run_v10_ablation_beam_vs_greedy.sh`

   These scripts now explicitly unset:
   - `EDGE_V10_MANUAL_UNITS`
   - `EDGE_V10_MANUAL_MID_POSES`
   - `EDGE_V10_MANUAL_MID_FRAMES`

3. Ablation output prefixes
   - Step4 now respects caller-provided `EDGE_V10_OUT_PREFIX` and `OUT_PATH`,
     so greedy/beam outputs no longer overwrite the default Step4 path.

## Quick test

```bash
python -m py_compile v10_choreo_planner.py generate_v10_choreo.py

export EDGE_V10_MANUAL_UNITS="1042,56"

bash scripts/run_v10_step1_dual_auto_mid.sh 2>&1 | tee logs/fix_step1.log
bash scripts/run_v10_step4_auto_multiunit.sh 2>&1 | tee logs/fix_step4.log

grep -E "Ignoring manual|manual_units=|ValueError|Traceback" -n logs/fix_step1.log logs/fix_step4.log
```

Expected:
- Step1/Step4 should show `manual_units=[]`.
- No `ValueError`.
