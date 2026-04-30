import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ABLATIONS = [
    {
        "name": "keyframe_only",
        "use_keyframes": True,
        "use_trajectory": False,
        "use_beat": False,
    },
    {
        "name": "trajectory_only",
        "use_keyframes": False,
        "use_trajectory": True,
        "use_beat": False,
    },
    {
        "name": "keyframe_trajectory",
        "use_keyframes": True,
        "use_trajectory": True,
        "use_beat": False,
    },
    {
        "name": "keyframe_trajectory_beat",
        "use_keyframes": True,
        "use_trajectory": True,
        "use_beat": True,
    },
]


def run_command(cmd):
    print("\n" + " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def add_if(cmd, flag, value):
    if value:
        cmd.extend([flag, str(value)])


def main():
    parser = argparse.ArgumentParser("Run four controlled generation ablations.")

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--music", required=True)
    parser.add_argument("--start_pose", default="")
    parser.add_argument("--end_pose", default="")
    parser.add_argument("--mid_poses", default="")
    parser.add_argument("--mid_pose_frames", default="")
    parser.add_argument("--trajectory", default="")
    parser.add_argument("--target_traj", default="")

    parser.add_argument("--out_dir", default="output/ablations")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--model_seq_len", type=int, default=150)
    parser.add_argument("--feature_type", default="hybrid", choices=["hybrid", "baseline", "jukebox"])
    parser.add_argument("--audio_dim", type=int, default=803)

    parser.add_argument("--beat_guidance_weight", type=float, default=0.5)
    parser.add_argument("--postprocess_strength", type=float, default=1.0)
    parser.add_argument("--hard_keyframe_project", action="store_true")
    parser.add_argument("--foot_lock_postprocess", action="store_true")
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for cfg in ABLATIONS:
        name = cfg["name"]
        run_dir = out_root / name
        run_dir.mkdir(parents=True, exist_ok=True)

        infer_cmd = [
            sys.executable,
            "infer_controlled.py",
            "--checkpoint", args.checkpoint,
            "--music", args.music,
            "--feature_type", args.feature_type,
            "--audio_dim", str(args.audio_dim),
            "--frames", str(args.frames),
            "--model_seq_len", str(args.model_seq_len),
            "--out_dir", str(run_dir),
            "--out_name", name,
        ]

        if args.no_render:
            infer_cmd.append("--no_render")

        if cfg["use_keyframes"]:
            add_if(infer_cmd, "--start_pose", args.start_pose)
            add_if(infer_cmd, "--end_pose", args.end_pose)
            add_if(infer_cmd, "--mid_poses", args.mid_poses)
            add_if(infer_cmd, "--mid_pose_frames", args.mid_pose_frames)
            if args.hard_keyframe_project:
                infer_cmd.append("--hard_keyframe_project")

        if cfg["use_trajectory"]:
            add_if(infer_cmd, "--trajectory", args.trajectory)
            add_if(infer_cmd, "--target_traj", args.target_traj)
            infer_cmd.append("--postprocess_trajectory")
            infer_cmd.extend(["--postprocess_strength", str(args.postprocess_strength)])

        if cfg["use_beat"]:
            infer_cmd.extend(["--beat_guidance_weight", str(args.beat_guidance_weight)])

        if args.foot_lock_postprocess:
            infer_cmd.append("--foot_lock_postprocess")

        run_command(infer_cmd)

        raw_motion = run_dir / f"{name}_raw_model.npy"
        final_motion = run_dir / f"{name}_final_system.npy"
        metrics_json = run_dir / f"{name}_metrics.json"
        metrics_csv = run_dir / f"{name}_metrics.csv"

        eval_cmd = [
            sys.executable,
            "eval_quantitative.py",
            "--motion", str(final_motion),
            "--raw_motion", str(raw_motion),
            "--post_motion", str(final_motion),
            "--checkpoint", args.checkpoint,
            "--audio", args.music,
            "--fps", "30",
            "--device", args.device,
            "--out_json", str(metrics_json),
            "--out_csv", str(metrics_csv),
        ]

        if cfg["use_keyframes"]:
            add_if(eval_cmd, "--start_pose", args.start_pose)
            add_if(eval_cmd, "--end_pose", args.end_pose)
            add_if(eval_cmd, "--mid_poses", args.mid_poses)
            add_if(eval_cmd, "--mid_pose_frames", args.mid_pose_frames)

        if cfg["use_trajectory"]:
            add_if(eval_cmd, "--trajectory", args.trajectory)
            add_if(eval_cmd, "--target_traj", args.target_traj)

        run_command(eval_cmd)

        with open(metrics_json, "r", encoding="utf-8") as f:
            payload = json.load(f)

        metrics = payload.get("metrics", {})
        row = {
            "ablation": name,
            "raw_motion": str(raw_motion),
            "final_motion": str(final_motion),
            "metrics_json": str(metrics_json),
            "keyframe_mpjpe_m_mean": metrics.get("keyframe_mpjpe_m_mean"),
            "raw_trajectory_ade_m": metrics.get("raw_trajectory_ade_m"),
            "post_trajectory_ade_m": metrics.get("post_trajectory_ade_m"),
            "foot_slide_rate": metrics.get("foot_slide_rate"),
            "beatalign_symmetric": metrics.get("beatalign_symmetric"),
        }
        summary_rows.append(row)

    summary_path = out_root / "ablation_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "ablation",
            "raw_motion",
            "final_motion",
            "metrics_json",
            "keyframe_mpjpe_m_mean",
            "raw_trajectory_ade_m",
            "post_trajectory_ade_m",
            "foot_slide_rate",
            "beatalign_symmetric",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved ablation summary: {summary_path}")


if __name__ == "__main__":
    main()