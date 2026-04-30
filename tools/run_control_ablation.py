import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


METRIC_COLUMNS = [
    "keyframe_mpjpe_m_mean",
    "keyframe_rot_err_deg_mean",
    "trajectory_ade_m",
    "trajectory_rmse_m",
    "foot_slide_rate",
    "foot_contact_speed_p95_mps",
    "beatalign_symmetric",
    "raw_trajectory_ade_m",
    "post_trajectory_ade_m",
    "post_minus_raw_trajectory_ade_m",
    "raw_foot_slide_rate",
    "post_foot_slide_rate",
    "post_minus_raw_foot_slide_rate",
    "raw_beatalign_symmetric",
    "post_beatalign_symmetric",
    "post_minus_raw_beatalign_symmetric",
]


def run(cmd):
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def add_value(cmd, flag, value):
    if value not in [None, ""]:
        cmd.extend([flag, str(value)])


def add_flag(cmd, flag, enabled):
    if enabled:
        cmd.append(flag)


def build_infer_cmd(args, variant):
    out_name = variant["name"]
    cmd = [
        args.python,
        "infer_controlled.py",
        "--checkpoint", args.checkpoint,
        "--music", args.music,
        "--feature_type", args.feature_type,
        "--audio_dim", str(args.audio_dim),
        "--frames", str(args.frames),
        "--model_seq_len", str(args.model_seq_len),
        "--out_dir", str(Path(args.out_dir) / out_name),
        "--out_name", out_name,
        "--keyframe_space", args.keyframe_space,
    ]

    if variant.get("keyframe", False):
        add_value(cmd, "--start_pose", args.start_pose)
        add_value(cmd, "--end_pose", args.end_pose)
        add_value(cmd, "--mid_poses", args.mid_poses)
        add_value(cmd, "--mid_pose_frames", args.mid_pose_frames)
        add_value(cmd, "--keyframe_width", args.keyframe_width)
        add_flag(cmd, "--hard_keyframe_project", args.hard_keyframe_project)

    if variant.get("trajectory", False):
        add_value(cmd, "--trajectory", args.trajectory)
        add_value(cmd, "--target_traj", args.target_traj)

    if variant.get("postprocess_trajectory", False):
        cmd.append("--postprocess_trajectory")
        cmd.extend(["--postprocess_strength", str(args.postprocess_strength)])

    if variant.get("foot_lock", False):
        cmd.append("--foot_lock_postprocess")
        cmd.extend(["--foot_lock_strength", str(args.foot_lock_strength)])
        cmd.extend(["--foot_lock_contact_source", args.foot_lock_contact_source])

    beat_weight = variant.get("beat_guidance_weight", 0.0)
    if beat_weight > 0:
        cmd.extend(["--beat_guidance_weight", str(beat_weight)])

    if args.no_render:
        cmd.append("--no_render")

    return cmd


def build_eval_cmd(args, variant):
    name = variant["name"]
    variant_dir = Path(args.out_dir) / name
    raw_motion = variant_dir / f"{name}_raw_model.npy"
    final_motion = variant_dir / f"{name}_final_system.npy"
    out_json = variant_dir / f"{name}_metrics.json"
    out_csv = variant_dir / f"{name}_metrics.csv"

    cmd = [
        args.python,
        "eval_quantitative.py",
        "--motion", str(final_motion),
        "--raw_motion", str(raw_motion),
        "--post_motion", str(final_motion),
        "--checkpoint", args.checkpoint,
        "--audio", args.music,
        "--fps", str(args.fps),
        "--device", args.eval_device,
        "--contact_source", args.contact_source,
        "--out_json", str(out_json),
        "--out_csv", str(out_csv),
    ]

    add_value(cmd, "--trajectory", args.trajectory)
    add_value(cmd, "--target_traj", args.target_traj)

    if variant.get("keyframe", False):
        add_value(cmd, "--start_pose", args.start_pose)
        add_value(cmd, "--end_pose", args.end_pose)
        add_value(cmd, "--mid_poses", args.mid_poses)
        add_value(cmd, "--mid_pose_frames", args.mid_pose_frames)
        cmd.extend(["--keyframe_space", args.keyframe_space])

    return cmd, out_json


def read_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("metrics", payload)


def main():
    parser = argparse.ArgumentParser(description="Run controlled inference ablations.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--music", required=True)
    parser.add_argument("--out_dir", default="output/ablation")

    parser.add_argument("--feature_type", default="hybrid")
    parser.add_argument("--audio_dim", type=int, default=803)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--model_seq_len", type=int, default=150)
    parser.add_argument("--fps", type=float, default=30.0)

    parser.add_argument("--start_pose", default="")
    parser.add_argument("--end_pose", default="")
    parser.add_argument("--mid_poses", default="")
    parser.add_argument("--mid_pose_frames", default="")
    parser.add_argument("--keyframe_space", default="normalized")
    parser.add_argument("--keyframe_width", type=int, default=3)
    parser.add_argument("--hard_keyframe_project", action="store_true")

    parser.add_argument("--trajectory", default="")
    parser.add_argument("--target_traj", default="")
    parser.add_argument("--postprocess_strength", type=float, default=1.0)

    parser.add_argument("--foot_lock_strength", type=float, default=0.7)
    parser.add_argument("--foot_lock_contact_source", default="auto")

    parser.add_argument("--beat_guidance_weight", type=float, default=0.5)

    parser.add_argument("--eval_device", default="cpu")
    parser.add_argument("--contact_source", default="auto")
    parser.add_argument("--no_render", action="store_true")

    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    variants = [
        {
            "name": "00_base",
            "keyframe": False,
            "trajectory": False,
            "postprocess_trajectory": False,
            "foot_lock": False,
            "beat_guidance_weight": 0.0,
        },
        {
            "name": "01_keyframe",
            "keyframe": True,
            "trajectory": False,
            "postprocess_trajectory": False,
            "foot_lock": False,
            "beat_guidance_weight": 0.0,
        },
        {
            "name": "02_keyframe_trajectory_raw",
            "keyframe": True,
            "trajectory": True,
            "postprocess_trajectory": False,
            "foot_lock": False,
            "beat_guidance_weight": 0.0,
        },
        {
            "name": "03_keyframe_trajectory_system",
            "keyframe": True,
            "trajectory": True,
            "postprocess_trajectory": True,
            "foot_lock": False,
            "beat_guidance_weight": 0.0,
        },
        {
            "name": "04_keyframe_trajectory_footlock",
            "keyframe": True,
            "trajectory": True,
            "postprocess_trajectory": True,
            "foot_lock": True,
            "beat_guidance_weight": 0.0,
        },
        {
            "name": "05_keyframe_trajectory_footlock_beat",
            "keyframe": True,
            "trajectory": True,
            "postprocess_trajectory": True,
            "foot_lock": True,
            "beat_guidance_weight": args.beat_guidance_weight,
        },
    ]

    summary_rows = []

    for variant in variants:
        run(build_infer_cmd(args, variant))
        eval_cmd, metrics_path = build_eval_cmd(args, variant)
        run(eval_cmd)

        metrics = read_metrics(metrics_path)
        row = {"variant": variant["name"]}
        for key in METRIC_COLUMNS:
            row[key] = metrics.get(key, "")
        summary_rows.append(row)

    summary_csv = Path(args.out_dir) / "ablation_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant"] + METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n✅ Ablation summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()