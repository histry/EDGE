#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/EDGE
mkdir -p output/final_v10/videos

for name in v10_step1_dual_wu2 v10_step2_manual_wu2 v10_step3_upperdance_wu2 v10_step4_auto_multiunit_wu2; do
  motion="output/v10_eval/${name}.npy"
  plan="output/v10_eval/${name}_v10_plan.json"
  traj="output/v10_eval/${name}_target_traj.npy"
  diag="output/v10_eval/${name}_diag.json"

  if [ -f "$motion" ] && [ -f "$traj" ]; then
    if [ -f "$plan" ]; then
      python planner_edit_loop.py --motion "$motion" --plan_json "$plan" --target_traj "$traj" --out "$diag" || true
    fi
    python render_choreorag_results.py --motion "$motion" --music "${MUSIC:-/home/disk/lsm/storage/EDGE/test_music_bank/dunhuangwu2.wav}" --out "output/final_v10/videos/${name}_follow.mp4" --camera_mode follow --sixd_layout rows --smooth_window 7 || true
    python render_choreorag_results.py --motion "$motion" --music "${MUSIC:-/home/disk/lsm/storage/EDGE/test_music_bank/dunhuangwu2.wav}" --out "output/final_v10/videos/${name}_fixed.mp4" --camera_mode fixed --sixd_layout rows --smooth_window 7 || true
  fi
done

grep -R "generated_motion_energy\|generated_upper_activity\|transition_jerk\|contact_phase_break\|trajectory_ade_m\|action" output/v10_eval/*_diag.json 2>/dev/null || true
