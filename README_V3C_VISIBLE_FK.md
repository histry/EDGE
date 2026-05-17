# EDGE-Dunhuang V3C Visible-FK Patch Pack

This pack fixes the failure mode where 151D feature-space speed looks reasonable
but FK-rendered 3D hand/wrist motion remains almost static.

## Install from EDGE root

```bash
cp v3c_visible_fk_patch.py /home/disk/lsm/storage/EDGE/
cp render_from_npy.py /home/disk/lsm/storage/EDGE/
cp scripts/run_v3c_visible_fk.sh /home/disk/lsm/storage/EDGE/scripts/
cp tools/eval_fk_visible_motion.py /home/disk/lsm/storage/EDGE/tools/
python patches/patch_train_for_v3c_visible_fk.py
chmod +x scripts/run_v3c_visible_fk.sh tools/eval_fk_visible_motion.py
```

## Smoke run

```bash
DATA_PATH=data/dunhuang_bvh/stationary_whitelist_v3_27units EXP_NAME=smoke_v3c_visibleFK_x0w03_e50 EPOCHS=50 SAVE_INTERVAL=50 EDGE_X0_RECON_LOSS_WEIGHT=0.3 EDGE_V3C_VISIBLE_FK_WEIGHT=10.0 bash scripts/run_v3c_visible_fk.sh
```

## Evaluate

```bash
SAMPLE_NPY=$(find runs/train_nextgen/smoke_v3c_visibleFK_x0w03_e50 -type f -name "*.npy" | head -1)

python tools/eval_fk_visible_motion.py   --pred "$SAMPLE_NPY"   --gt_dir data/dunhuang_bvh/stationary_whitelist_v3_27units
```

## Render

```bash
python render_from_npy.py   --motion "$SAMPLE_NPY"   --audio test_music_bank/dunhuangwu2.wav   --output output/v3c_eval/v3c_visibleFK_e50_fixed.mp4   --camera_mode fixed
```

Batch input writes `_b00.mp4`, `_b01.mp4`, etc.

## Success criteria

V3B baseline:
- hand/wrist FK range ≈ 0.009–0.010
- hand/wrist speed_mean ≈ 0.0013–0.0016

V3C target:
- hand/wrist FK range >= 0.04–0.08
- hand/wrist speed_mean >= 0.006–0.012
- no V2F-style burst artifacts
