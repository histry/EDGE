"""Lightweight prediction-then-editing diagnostics for ChoreoRAG.

This does not regenerate motion by itself. It reads generated motion + target
trajectory + ChoreoRAG plan metadata, diagnoses bad segments, and writes an
edit recommendation JSON. Use recommendations to re-run generation with higher
contact/entry/exit/text weights or fewer auto-mid frames.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


CONTACT_SLICE = slice(0, 4)
ROOT_X_IDX = 4
ROOT_Z_IDX = 6
ROT_SLICE = slice(7, 151)


def load_motion(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        d = arr.item()
        arr = d.get("motion", d.get("pose", d.get("motion_final", arr)))
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 151:
        raise ValueError(f"Expected [T,151] motion, got {arr.shape}")
    return arr


def transition_jerk(motion: np.ndarray, frame: int, window: int = 8) -> float:
    if len(motion) < 4:
        return 0.0
    x = motion[:, ROT_SLICE]
    acc = x[2:] - 2 * x[1:-1] + x[:-2]
    j = np.linalg.norm(acc, axis=-1)
    center = int(np.clip(frame - 1, 0, len(j) - 1))
    s = max(0, center - window)
    e = min(len(j), center + window + 1)
    return float(j[s:e].mean()) if e > s else 0.0


def contact_phase_break(motion: np.ndarray, frame: int, window: int = 8) -> float:
    c = (motion[:, CONTACT_SLICE] > 0.5).astype(np.float32)
    if len(c) < 2:
        return 0.0
    changes = np.abs(c[1:] - c[:-1]).mean(axis=1)
    center = int(np.clip(frame, 0, len(changes) - 1))
    s = max(0, center - window)
    e = min(len(changes), center + window + 1)
    return float(changes[s:e].mean()) if e > s else 0.0


def freezing_score(motion: np.ndarray, threshold: float = 0.015) -> float:
    if len(motion) < 2:
        return 1.0
    v = np.linalg.norm(motion[1:, ROT_SLICE] - motion[:-1, ROT_SLICE], axis=-1)
    return float((v < threshold).mean())


def path_ade(motion: np.ndarray, target_traj: np.ndarray) -> float:
    traj = np.asarray(target_traj, dtype=np.float32)
    if traj.ndim == 3:
        traj = traj[0]
    if len(traj) != len(motion):
        idx = np.linspace(0, len(traj) - 1, len(motion))
        x = np.interp(idx, np.arange(len(traj)), traj[:, 0])
        z = np.interp(idx, np.arange(len(traj)), traj[:, 1])
        traj = np.stack([x, z], axis=-1)
    root = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    return float(np.linalg.norm(root - traj[:, :2], axis=-1).mean())


def diagnose(motion_path: str, plan_json: str, target_traj_path: str = "", out: str = "") -> Dict:
    motion = load_motion(motion_path)
    with open(plan_json, "r", encoding="utf-8") as f:
        plan = json.load(f)

    keyframes = plan.get("keyframes", [])
    rows: List[Dict] = []
    for kf in keyframes:
        frame = int(kf.get("frame", 0))
        jerk = transition_jerk(motion, frame)
        cpb = contact_phase_break(motion, frame)
        action = "keep"
        reason = []
        if jerk > 0.18:
            reason.append("high_transition_jerk")
        if cpb > 0.35:
            reason.append("contact_phase_break")
        if reason:
            action = "rerun_with_stronger_entry_exit_contact_or_lower_mid_strength"
        rows.append({
            "frame": frame,
            "segment_id": int(kf.get("segment_id", -1)),
            "source": kf.get("source", ""),
            "score": float(kf.get("score", 0.0)),
            "transition_jerk": jerk,
            "contact_phase_break": cpb,
            "action": action,
            "reason": reason,
        })

    result = {
        "motion": motion_path,
        "plan_json": plan_json,
        "freezing_score": freezing_score(motion),
        "segments": rows,
        "recommendation": {
            "if_high_jerk": "export EDGE_UNIT_ENTRY_WEIGHT=0.85; export EDGE_UNIT_EXIT_WEIGHT=0.85; reduce --mid_keyframe_strength to 0.15",
            "if_contact_break": "export EDGE_UNIT_CONTACT_PHASE_WEIGHT=1.20; reduce retrieved_prior_strength if used",
            "if_freezing_high": "try auto_mid_count=1/2 or increase text/energy weights; no-auto-mid likely too static",
        },
    }

    if target_traj_path:
        result["trajectory_ade_m"] = path_ade(motion, np.load(target_traj_path).astype(np.float32))

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ Edit diagnostics saved: {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--plan_json", required=True)
    parser.add_argument("--target_traj", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    diagnose(args.motion, args.plan_json, args.target_traj, args.out)


if __name__ == "__main__":
    main()
