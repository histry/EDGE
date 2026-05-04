"""Diagnostics for ChoreoRAG reward-collapse experiments.

Replacement version that keeps the original transition/freezing/ADE checks and
adds generated expressiveness proxies.  It can be used after every ablation to
verify whether a higher-expressiveness retrieval actually propagates into the
final generated motion without unacceptable jerk/contact break.
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
UPPER_JOINTS = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]


def _rot_indices(joints):
    idx = []
    for j in joints:
        idx.extend(range(7 + 6 * int(j), 7 + 6 * int(j) + 6))
    return np.asarray(idx, dtype=np.int64)


UPPER_IDX = _rot_indices(UPPER_JOINTS)
LOWER_IDX = _rot_indices(LOWER_JOINTS)


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


def motion_activity_stats(motion: np.ndarray) -> Dict[str, float]:
    if len(motion) < 2:
        return {
            "generated_motion_energy": 0.0,
            "generated_upper_activity": 0.0,
            "generated_lower_activity": 0.0,
            "generated_root_speed": 0.0,
            "generated_spatial_range": 0.0,
            "generated_turning": 0.0,
        }
    diff = motion[1:] - motion[:-1]
    root = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    root_vel = root[1:] - root[:-1]
    if len(root_vel) > 2:
        v1, v2 = root_vel[:-1], root_vel[1:]
        n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
        cos = np.sum(v1 * v2, axis=1) / np.clip(n1 * n2, 1e-8, None)
        turning = float(np.mean(1.0 - np.clip(cos, -1.0, 1.0)))
    else:
        turning = 0.0
    return {
        "generated_motion_energy": float(np.sqrt(np.mean(diff[:, ROT_SLICE] ** 2))),
        "generated_upper_activity": float(np.sqrt(np.mean(diff[:, UPPER_IDX] ** 2))),
        "generated_lower_activity": float(np.sqrt(np.mean(diff[:, LOWER_IDX] ** 2))),
        "generated_root_speed": float(np.linalg.norm(root_vel, axis=1).mean()) if len(root_vel) else 0.0,
        "generated_spatial_range": float(np.linalg.norm(root.max(axis=0) - root.min(axis=0))),
        "generated_turning": float(max(0.0, turning)),
    }


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


def _read_plan(plan_json: str) -> Dict:
    with open(plan_json, "r", encoding="utf-8") as f:
        plan = json.load(f)
    if "keyframes" not in plan and "auto_keyframes" in plan:
        plan["keyframes"] = plan["auto_keyframes"]
    return plan


def diagnose(motion_path: str, plan_json: str, target_traj_path: str = "", out: str = "") -> Dict:
    motion = load_motion(motion_path)
    plan = _read_plan(plan_json)

    keyframes = plan.get("keyframes", [])
    rows: List[Dict] = []
    retrieved_expr = []
    retrieved_energy = []
    for kf in keyframes:
        frame = int(kf.get("frame", 0))
        parts = kf.get("score_parts", {}) or {}
        if "expressiveness_score" in parts:
            retrieved_expr.append(float(parts.get("expressiveness_score", 0.0)))
        if "motion_energy_norm" in parts:
            retrieved_energy.append(float(parts.get("motion_energy_norm", 0.0)))
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
            "phase": str(parts.get("phase", "")),
            "source": kf.get("source", ""),
            "score": float(kf.get("score", 0.0)),
            "retrieved_expressiveness": float(parts.get("expressiveness_score", -1.0)),
            "retrieved_energy_norm": float(parts.get("motion_energy_norm", -1.0)),
            "transition_jerk": jerk,
            "contact_phase_break": cpb,
            "action": action,
            "reason": reason,
        })

    result = {
        "motion": motion_path,
        "plan_json": plan_json,
        "freezing_score": freezing_score(motion),
        "retrieved_expressiveness_mean": float(np.mean(retrieved_expr)) if retrieved_expr else None,
        "retrieved_energy_mean": float(np.mean(retrieved_energy)) if retrieved_energy else None,
        **motion_activity_stats(motion),
        "segments": rows,
        "recommendation": {
            "if_high_jerk": "Reduce --mid_keyframe_strength to 0.12-0.15 or enable EDGE_UNIT_SOFT_PRIOR=1 with EDGE_UNIT_PRIOR_STRENGTH=0.04-0.06.",
            "if_contact_break": "Increase EDGE_UNIT_CONTACT_PHASE_WEIGHT or use phase=pose for ending segments; keep unit prior upper-only.",
            "if_freezing_high": "Increase EDGE_UNIT_EXPRESSIVENESS_BONUS or EDGE_UNIT_MIN_EXPRESSIVENESS for attack/flow segments.",
            "if_lower_activity_too_high": "Lower expr_w_root/expr_w_lower when rebuilding DB; prefer upper/spatial/turning expressiveness.",
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
