import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


METRIC_KEYS = [
    "keyframe_mpjpe_m_mean",
    "keyframe_rot_err_deg_mean",
    "trajectory_ade_m",
    "trajectory_rmse_m",
    "trajectory_final_error_m",
    "foot_slide_rate",
    "foot_contact_speed_mean_mps",
    "beatalign_symmetric",
]


def run_cmd(cmd):
    print("\n" + "=" * 80)
    print(" ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("metrics", payload)


def main():
    parser = argparse.ArgumentParser("Run controlled generation ablation")

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--music", required=True)
    parser.add_argument("--start_pose", required=True)
    parser.add_argument("--end_pose", required=True)
    parser.add_argument("--trajectory", default="")
    parser.add_argument("--target_traj", default="")

    parser.add_argument("--feature_type", default="hybrid")
    parser.add_argument("--audio_dim", type=int, default=803)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--keyframe_space", default="normalized")

    parser.add_argument("--out_dir", default="output/ablation")
    parser.add_argument("--no_render", action="store_true")

    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if not args.trajectory and not args.target_traj:
        raise ValueError("Ablation needs --trajectory or --target_traj for trajectory metrics")

    cases = [
        {
            "name": "audio_only",
            "use_keyframe": False,
            "use_traj": False,
            "use_tto": False,
            "post": False,
        },
        {
            "name": "keyframe",
            "use_keyframe": True,
            "use_traj": False,
            "use_tto": False,
            "post": False,
        },
        {
            "name": "keyframe_traj",
            "use_keyframe": True,
            "use_traj": True,
            "use_tto": False,
            "post": False,
        },
        {
            "name": "keyframe_traj_tto_post",
            "use_keyframe": True,
            "use_traj": True,
            "use_tto": True,
            "post": True,
        },
    ]

    rows = []

    for case in cases:
        case_dir = os.path.join(args.out_dir, case["name"])
        Path(case_dir).mkdir(parents=True, exist_ok=True)

        infer_cmd = [
            sys.executable,
            "infer_controlled.py",
            "--checkpoint",
            args.checkpoint,
            "--music",
            args.music,
            "--feature_type",
            args.feature_type,
            "--audio_dim",
            str(args.audio_dim),
            "--frames",
            str(args.frames),
            "--keyframe_space",
            args.keyframe_space,
            "--out_dir",
            case_dir,
            "--out_name",
            case["name"],
        ]

        if args.no_render:
            infer_cmd.append("--no_render")

        if case["use_keyframe"]:
            infer_cmd.extend(
                [
                    "--start_pose",
                    args.start_pose,
                    "--end_pose",
                    args.end_pose,
                    "--hard_keyframe_project",
                ]
            )

        if case["use_traj"]:
            if args.target_traj:
                infer_cmd.extend(["--target_traj", args.target_traj])
            else:
                infer_cmd.extend(["--trajectory", args.trajectory])

        if not case["use_tto"]:
            infer_cmd.append("--no_tto")

        if case["post"]:
            infer_cmd.append("--postprocess_trajectory")

        run_cmd(infer_cmd)

        motion_path = os.path.join(case_dir, f"{case['name']}.npy")
        raw_motion_path = os.path.join(case_dir, f"{case['name']}_raw.npy")
        metrics_json = os.path.join(case_dir, f"{case['name']}_metrics.json")
        metrics_csv = os.path.join(case_dir, f"{case['name']}_metrics.csv")

        eval_cmd = [
            sys.executable,
            "eval_quantitative.py",
            "--motion",
            motion_path,
            "--raw_motion",
            raw_motion_path,
            "--post_motion",
            motion_path,
            "--checkpoint",
            args.checkpoint,
            "--audio",
            args.music,
            "--start_pose",
            args.start_pose,
            "--end_pose",
            args.end_pose,
            "--out_json",
            metrics_json,
            "--out_csv",
            metrics_csv,
        ]

        if args.target_traj:
            eval_cmd.extend(["--target_traj", args.target_traj])
        else:
            eval_cmd.extend(["--trajectory", args.trajectory])

        run_cmd(eval_cmd)

        metrics = load_metrics(metrics_json)
        row = {"case": case["name"]}
        for key in METRIC_KEYS:
            row[key] = metrics.get(key, "")
        rows.append(row)

    summary_csv = os.path.join(args.out_dir, "ablation_summary.csv")
    summary_json = os.path.join(args.out_dir, "ablation_summary.json")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case"] + METRIC_KEYS)
        writer.writeheader()
        writer.writerows(rows)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"\nSaved ablation summary CSV:  {summary_csv}")
    print(f"Saved ablation summary JSON: {summary_json}")


if __name__ == "__main__":
    main()