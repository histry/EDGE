import argparse
import pickle
import subprocess
from pathlib import Path
import numpy as np

def run(cmd, env_prefix=None):
    if env_prefix:
        cmd = env_prefix + cmd
        subprocess.run(" ".join(cmd), shell=True, check=True)
    else:
        subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--unit_list", required=True)
    ap.add_argument("--data_dir", default="data/dunhuang_stationary_whitelist_v2/processed")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--music", default="test_music_bank/dunhuangwu2.wav")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kf_root = out_dir / "keyframes"
    kf_root.mkdir(parents=True, exist_ok=True)

    units = [x.strip() for x in Path(args.unit_list).read_text().splitlines() if x.strip()]

    for uid in units:
        pkl = Path(args.data_dir) / f"unit_{uid}_gt45.pkl"
        if not pkl.exists():
            print("missing:", pkl)
            continue

        data = pickle.load(open(pkl, "rb"))
        motion = data.get("motion", data.get("motion_151"))
        motion = np.asarray(motion, dtype=np.float32)[:45]

        gt = out_dir / f"unit_{uid}_gt.npy"
        np.save(gt, motion)

        kdir = kf_root / f"unit_{uid}"
        kdir.mkdir(parents=True, exist_ok=True)

        frames = [0, 11, 22, 34, 44]
        for f in frames:
            np.save(kdir / f"frame_{f:03d}.npy", motion[f])

        pred = out_dir / f"unit_{uid}_pred.npy"
        mid_poses = ",".join(str(kdir / f"frame_{f:03d}.npy") for f in [11, 22, 34])
        mid_frames = "11,22,34"

        env_prefix = [
            "EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER=1",
            "EDGE_HARD_KEYFRAME_PROJECT=1",
            "EDGE_INFER_PROJECT_XSTART=1",
            "EDGE_FINAL_KEYFRAME_PROJECT=1",
            "EDGE_ENABLE_TEXT_CONTEXT_RAG=0",
            "EDGE_ENABLE_RAG_SUMMARY_TOKEN=0",
        ]

        run([
            "python", "generate_controlled.py",
            "--checkpoint", args.ckpt,
            "--music", args.music,
            "--start_pose", str(kdir / "frame_000.npy"),
            "--end_pose", str(kdir / "frame_044.npy"),
            "--mid_poses", mid_poses,
            "--mid_pose_frames", mid_frames,
            "--out", str(pred),
            "--seq_len", "45",
            "--num_frames", "45",
            "--pose_space", "physical",
            "--disable_traj_cond",
            "--hard_keyframe_project",
            "--infer_project_xstart",
            "--keyframe_constrain_root_xz",
            "--guidance_weight", "1.0",
            "--sampler", "ddim",
            "--no_tto",
            "--no_ema",
            "--save_normalized_motion",
        ], env_prefix=env_prefix)

        run([
            "python", "scripts/eval_unit45_recon_quality.py",
            "--pred", str(pred),
            "--gt", str(gt),
            "--out", str(out_dir / f"unit_{uid}_eval.json"),
        ])

        run([
            "python", "render_from_npy.py",
            "--motion", str(pred),
            "--audio", args.music,
            "--output", str(out_dir / f"unit_{uid}_fixed.mp4"),
            "--camera_mode", "fixed",
        ])

if __name__ == "__main__":
    main()
