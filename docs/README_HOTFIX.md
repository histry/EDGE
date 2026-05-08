# EDGE ChoreoRAG hotfix: OOM + TextBridge compatibility

## Install

```bash
unzip edge_choreorag_hotfix_oom_textbridge.zip
cp edge_choreorag_hotfix/patch_v10_textbridge_compat.py .
cp edge_choreorag_hotfix/scripts/run_energy_temporal_prior_ablation_safe.sh scripts/
chmod +x scripts/run_energy_temporal_prior_ablation_safe.sh

python patch_v10_textbridge_compat.py
python -m py_compile v10_choreo_planner.py
```

## Clear GPU memory first

```bash
nvidia-smi
ps -fp 34811 36214 37513
# If they are stale jobs you own:
kill 34811 36214 37513
sleep 5
nvidia-smi
# If still alive and confirmed stale:
kill -9 34811 36214 37513
```

Or run on another free GPU:

```bash
export CUDA_VISIBLE_DEVICES=1
```

## Run safe ablation

```bash
export CHECKPOINT="runs/train_stage45/v10_text_context_rag_adapter_e8_fix2_full/weights/train-4.pt"
export EDGE_V10_RAG_DB="data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco.npz"
export EDGE_RAG_STATS_CACHE="data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_stats.npz"

export MUSIC="test_music_bank/dunhuangwu2.wav"
export START_POSE="test_keyframes/demo_dyl002_start.npy"
export END_POSE="test_keyframes/demo_dyl002_end.npy"
export TRAJECTORY="0,0;0.5,0.7;-0.3,1.2;0,1.6"

bash scripts/run_energy_temporal_prior_ablation_safe.sh
```
