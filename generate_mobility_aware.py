#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mobility-aware wrapper around generate_controlled_v9.py.

Modes:
  stationary:
    - no --trajectory
    - select stationary / turn-in-place units only
    - optional root-lock after generation

  trajectory:
    - uses --trajectory
    - select mobile / landing units only
    - does NOT allow stationary expressive units to be dragged along the path

This wrapper is deliberately small: it delegates generation to generate_controlled_v9.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from mobility_aware_selector import load_db, select_indices, export_selected
from mobility_motion_utils import load_motion, freeze_root_xz, metrics


def run(cmd: List[str], log_path: Optional[str] = None) -> int:
    print(" ".join(shlex.quote(x) for x in cmd))
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert p.stdout is not None
            for line in p.stdout:
                print(line, end="")
                f.write(line)
            return p.wait()
    return subprocess.call(cmd)


def clear_traj_env():
    for k in list(os.environ.keys()):
        if k.startswith("EDGE_TURN"):
            os.environ.pop(k, None)
    os.environ["EDGE_DYNAMIC_TRAJ_CFG"] = "0"
    os.environ["EDGE_GAIT_PHASE_COND"] = "0"
    os.environ["EDGE_GAIT_CONTACT_LOSS"] = "0"
    os.environ["EDGE_TRAJ_PHYSICS_FEATURES"] = "0"
    os.environ["EDGE_TRAJ_FOURIER_FEATURES"] = "0"
    os.environ["EDGE_TRAJ_SPARSE_WAYPOINT"] = "0"


def select_mid_poses(db_path: str, intent: str, count: int, out_prefix: str) -> List[str]:
    data = load_db(db_path)
    idxs = select_indices(data, intent=intent, count=count, min_gap=10)
    report = export_selected(data, idxs, out_prefix)
    return report.get("mid_poses", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stationary", "trajectory"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--music", required=True)
    ap.add_argument("--start_pose", required=True)
    ap.add_argument("--end_pose", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mobility_db", required=True)
    ap.add_argument("--trajectory", default=None)
    ap.add_argument("--sampler", default="ddim")
    ap.add_argument("--feature_type", default="hybrid")
    ap.add_argument("--endpoint_keyframe_strength", type=float, default=0.3)
    ap.add_argument("--mid_count", type=int, default=1)
    ap.add_argument("--mid_keyframe_strength", type=float, default=0.08)
    ap.add_argument("--beat_weight", type=float, default=0.0)
    ap.add_argument("--energy_scale", type=float, default=0.5)
    ap.add_argument("--context_scale", type=float, default=0.5)
    ap.add_argument("--unit_prior_strength", type=float, default=0.012)
    ap.add_argument("--unit_prior_features", default="upper")
    ap.add_argument("--root_lock_after", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    clear_traj_env()

    os.environ["EDGE_MOBILITY_AWARE"] = "1"
    os.environ["EDGE_MOBILITY_MODE"] = args.mode
    os.environ["EDGE_CHECKPOINT_COMPAT_CPU_MERGE"] = "1"
    os.environ["EDGE_AUDIO_DEVICE"] = "cpu"
    os.environ["EDGE_EXPERIMENT_PROFILE"] = "v10"
    os.environ["EDGE_ENABLE_TEXT_CONTEXT_RAG"] = "1"
    os.environ["EDGE_ENABLE_RAG_SUMMARY_TOKEN"] = "1"

    os.environ["EDGE_AUDIO_ENERGY_AS_COND"] = "1"
    os.environ["EDGE_MUSIC_TENSION_AS_ENERGY"] = "1"
    os.environ["EDGE_ENERGY_CFG_SCALE"] = str(args.energy_scale)

    os.environ["EDGE_CONTEXT_RAG_ENHANCE"] = "1"
    os.environ["EDGE_CONTEXT_RAG_SCALE"] = str(args.context_scale)

    os.environ["EDGE_UNIT_SOFT_PRIOR"] = "1"
    os.environ["EDGE_UNIT_PRIOR_REQUIRED"] = "0"
    os.environ["EDGE_UNIT_PRIOR_TEMPORAL"] = "1"
    os.environ["EDGE_UNIT_PRIOR_DCT"] = "1"
    os.environ["EDGE_UNIT_PRIOR_DCT_DECAY"] = "soft_exp"
    os.environ["EDGE_UNIT_PRIOR_LOW_FREQ_K"] = "4"
    os.environ["EDGE_UNIT_PRIOR_FEATURES"] = args.unit_prior_features
    os.environ["EDGE_UNIT_PRIOR_STRENGTH"] = str(args.unit_prior_strength)

    if args.beat_weight > 0:
        os.environ["EDGE_BEAT_GUIDANCE"] = "1"
        os.environ["EDGE_BEAT_GUIDANCE_WEIGHT"] = str(args.beat_weight)
        os.environ["EDGE_BEAT_GUIDANCE_TARGET"] = "1.35"
        os.environ["EDGE_BEAT_GUIDANCE_FEATURES"] = "all"
    else:
        os.environ["EDGE_BEAT_GUIDANCE"] = "0"
        os.environ["EDGE_BEAT_GUIDANCE_WEIGHT"] = "0"

    intent = "stationary_expressive" if args.mode == "stationary" else "mobile"
    prefix = str(Path(args.out).with_suffix("")) + "_mobility"
    mid_poses = []
    if args.mid_count > 0:
        mid_poses = select_mid_poses(args.mobility_db, intent, args.mid_count, prefix)

    cmd = [
        sys.executable,
        "generate_controlled_v9.py",
        "--checkpoint", args.checkpoint,
        "--music", args.music,
        "--start_pose", args.start_pose,
        "--end_pose", args.end_pose,
        "--out", args.out,
        "--feature_type", args.feature_type,
        "--sampler", args.sampler,
        "--endpoint_keyframe_strength", str(args.endpoint_keyframe_strength),
        "--no_tto",
    ]

    if mid_poses:
        frames = []
        if len(mid_poses) == 1:
            frames = ["75"]
        elif len(mid_poses) == 2:
            frames = ["50", "100"]
        else:
            # evenly spaced excluding endpoints
            frames = [str(int(round((i + 1) * 150 / (len(mid_poses) + 1)))) for i in range(len(mid_poses))]
        cmd += [
            "--mid_poses", ",".join(mid_poses),
            "--mid_pose_frames", ",".join(frames),
            "--mid_keyframe_strength", str(args.mid_keyframe_strength),
        ]

    if args.mode == "trajectory":
        if not args.trajectory:
            raise ValueError("--trajectory is required in trajectory mode")
        cmd += ["--trajectory", args.trajectory]

    status = run(cmd, args.log)
    if status != 0:
        raise SystemExit(status)

    final_out = args.out
    if args.mode == "stationary" and args.root_lock_after:
        x = load_motion(args.out)
        y = freeze_root_xz(x)
        rootlock_out = str(Path(args.out).with_suffix("")) + "_rootlock.npy"
        np.save(rootlock_out, y)
        final_out = rootlock_out
        report = {
            "raw": metrics(x),
            "root_locked": metrics(y),
            "rootlock_motion": rootlock_out,
        }
        Path(str(Path(args.out).with_suffix("")) + "_rootlock_eval.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )

    if args.render:
        audio = args.music
        base = str(Path(final_out).with_suffix(""))
        run([
            sys.executable, "render_from_npy.py",
            "--motion", final_out,
            "--audio", audio,
            "--output", base + "_follow.mp4",
            "--camera_mode", "follow",
        ])
        run([
            sys.executable, "render_from_npy.py",
            "--motion", final_out,
            "--audio", audio,
            "--output", base + "_fixed.mp4",
            "--camera_mode", "fixed",
        ])


if __name__ == "__main__":
    main()
