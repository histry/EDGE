#!/usr/bin/env bash
set -euo pipefail

CKPT="runs/best/dunhuang_stage1_best_exp17_train30.pt"
AUDIO="/home/lsm/disk_space/storage/EDGE/test_music_bank/dunhuangwu2.wav"
TARGET="output/check_mmr_rag/dyl002_mmr_auto3_plannerv3_target_traj.npy"
RAW="output/check_mmr_rag/dyl002_mmr_auto3_plannerv3_raw.npy"
START="test_keyframes/dyl002_600_1800_start.npy"
END="test_keyframes/dyl002_600_1800_end.npy"
MID_POSES="output/check_mmr_rag/dyl002_mmr_auto3_plannerv3_auto_mid1_f045.npy,output/check_mmr_rag/dyl002_mmr_auto3_plannerv3_auto_mid2_f091.npy,output/check_mmr_rag/dyl002_mmr_auto3_plannerv3_auto_mid3_f121.npy"
MID_FRAMES="45,91,121"

for NAME in \
  dyl002_mmr_auto3_plannerv3_raw \
  dyl002_mmr_auto3_plannerv3_anchorv2 \
  dyl002_mmr_auto3_plannerv3_anchorv2_s045_c010 \
  dyl002_mmr_auto3_plannerv3_anchorv2_s060_c015 \
  dyl002_mmr_auto3_plannerv3_anchorv2_s070_c020
do
  MOTION="output/check_mmr_rag/${NAME}.npy"

  python eval_quantitative.py \
    --motion "$MOTION" \
    --raw_motion "$RAW" \
    --post_motion "$MOTION" \
    --checkpoint "$CKPT" \
    --audio "$AUDIO" \
    --target_traj "$TARGET" \
    --start_pose "$START" \
    --end_pose "$END" \
    --mid_poses "$MID_POSES" \
    --mid_pose_frames "$MID_FRAMES" \
    --keyframe_space normalized \
    --out_json "output/check_mmr_rag/${NAME}_metrics_5kf.json" \
    --out_csv "output/check_mmr_rag/${NAME}_metrics_5kf.csv"
done
