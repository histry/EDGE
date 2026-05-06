#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

from v10_choreo_planner import (
    default_dual_mid_frames,
    env_int,
    parse_csv,
    parse_frames,
    plan_manual_from_env,
    plan_upperdance_from_rag_db,
)


def _strip_arg(argv: List[str], name: str, takes_value: bool) -> List[str]:
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == name:
            i += 2 if takes_value else 1
            continue
        if takes_value and argv[i].startswith(name + "="):
            i += 1
            continue
        out.append(argv[i])
        i += 1
    return out


def _get_arg(argv: List[str], name: str, default: str = "") -> str:
    for i, item in enumerate(argv):
        if item == name and i + 1 < len(argv):
            return argv[i + 1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return default


def _replace_arg(argv: List[str], name: str, value: str) -> List[str]:
    argv = _strip_arg(argv, name, takes_value=True)
    return argv + [name, value]


def _default_prefix_from_out(argv: List[str]) -> str:
    out = _get_arg(argv, "--out", "output/v10_eval/v10_choreo.npy")
    return str(Path(out).with_suffix(""))


def _prepare_env_defaults() -> None:
    # Unit-prior defaults are intentionally gentle. Override from scripts/env
    # after transition jerk is under control.
    os.environ.setdefault("EDGE_UNIT_SOFT_PRIOR", "1")
    os.environ.setdefault("EDGE_UNIT_PRIOR_DCT", "1")
    os.environ.setdefault("EDGE_UNIT_PRIOR_LOW_FREQ_K", "4")
    os.environ.setdefault("EDGE_UNIT_PRIOR_FEATURES", "upper")
    os.environ.setdefault("EDGE_UNIT_PRIOR_STRENGTH", "0.012")
    os.environ.setdefault("EDGE_UNIT_PRIOR_MAX_LEN", "45")
    os.environ.setdefault("EDGE_UNIT_ENTRY_WEIGHT", "0.85")
    os.environ.setdefault("EDGE_UNIT_EXIT_WEIGHT", "0.85")
    os.environ.setdefault("EDGE_UNIT_CONTACT_PHASE_WEIGHT", "0.80")
    os.environ.setdefault("EDGE_TENSION_AWARE_PLANNER", "1")

    # V9 summary-token inference bridge.
    os.environ.setdefault("EDGE_ENABLE_RAG_SUMMARY_TOKEN", "1")
    os.environ.setdefault("EDGE_RAG_SUMMARY_DIM", "7")
    os.environ.setdefault("EDGE_RAG_SUMMARY_BLEND_RADIUS", "18")


def _safe_frames(frame_text: str, count: int, num_frames: int, mode: str) -> List[int]:
    if frame_text:
        raw = parse_csv(frame_text)
        if len(raw) == count:
            return parse_frames(frame_text, count=count, num_frames=num_frames)
        print(
            f"⚠️ EDGE_V10_MID_FRAMES={frame_text!r} has {len(raw)} values but "
            f"mode={mode} needs {count}; ignoring it and using automatic spacing."
        )

    if count == 2:
        return default_dual_mid_frames(num_frames=num_frames)
    return parse_frames("", count=count, num_frames=num_frames)


def _export_v9_rag_summary_env(plan: dict) -> None:
    unit_paths = [str(p) for p in (plan.get("unit_paths") or []) if p]
    frames = [int(x) for x in (plan.get("mid_pose_frames") or [])]

    if not unit_paths:
        os.environ.pop("EDGE_RAG_SUMMARY_UNIT_PATHS", None)
        os.environ.pop("EDGE_RAG_SUMMARY_MID_FRAMES", None)
        print("ℹ️ V9 RAG summary inference token disabled for this run: no unit_paths in plan.")
        return

    os.environ["EDGE_ENABLE_RAG_SUMMARY_TOKEN"] = "1"
    os.environ["EDGE_RAG_SUMMARY_UNIT_PATHS"] = ",".join(unit_paths)
    os.environ["EDGE_RAG_SUMMARY_MID_FRAMES"] = ",".join(str(x) for x in frames)

    print("✅ V9 RAG summary inference env exported:")
    print(f"  EDGE_RAG_SUMMARY_UNIT_PATHS={os.environ['EDGE_RAG_SUMMARY_UNIT_PATHS']}")
    print(f"  EDGE_RAG_SUMMARY_MID_FRAMES={os.environ['EDGE_RAG_SUMMARY_MID_FRAMES']}")


def build_forward_argv(argv: List[str]) -> List[str]:
    _prepare_env_defaults()

    num_frames = env_int("EDGE_V10_NUM_FRAMES", 150)
    mode = os.environ.get("EDGE_V10_MODE", "dual_auto_mid").strip().lower()
    out_prefix = os.environ.get("EDGE_V10_OUT_PREFIX", _default_prefix_from_out(argv))

    # Bypass old auto-mid clamp-to-1 by injecting explicit mid poses/frames.
    for flag, takes in [
        ("--auto_mid_keyframes", False),
        ("--auto_mid_count", True),
        ("--mid_poses", True),
        ("--mid_pose_frames", True),
    ]:
        argv = _strip_arg(argv, flag, takes)

    manual_plan = plan_manual_from_env(out_prefix=out_prefix, num_frames=num_frames)
    if manual_plan is not None:
        plan = manual_plan
    else:
        rag_db = os.environ.get("EDGE_V10_RAG_DB") or os.environ.get("RAG_DB", "")
        if not rag_db:
            raise RuntimeError("Set EDGE_V10_RAG_DB or RAG_DB, unless EDGE_V10_MANUAL_MID_POSES is set.")

        count = env_int("EDGE_V10_AUTO_MID_COUNT", 3 if mode == "auto_multiunit" else 2)
        if mode in {"dual_auto_mid", "upperdance_rag"}:
            count = 2

        frame_text = os.environ.get("EDGE_V10_MID_FRAMES", "")
        frames = _safe_frames(frame_text, count=count, num_frames=num_frames, mode=mode)

        start_pose = _get_arg(argv, "--start_pose", "")
        end_pose = _get_arg(argv, "--end_pose", "")

        plan = plan_upperdance_from_rag_db(
            rag_db=rag_db,
            out_prefix=out_prefix,
            num_frames=num_frames,
            count=count,
            frames=frames,
            start_pose_path=start_pose,
            end_pose_path=end_pose,
        )

    _export_v9_rag_summary_env(plan)

    mid_poses = ",".join(plan["mid_poses"])
    mid_frames = ",".join(str(x) for x in plan["mid_pose_frames"])
    argv = _replace_arg(argv, "--mid_poses", mid_poses)
    argv = _replace_arg(argv, "--mid_pose_frames", mid_frames)

    if os.environ.get("EDGE_V10_MID_STRENGTH"):
        argv = _replace_arg(argv, "--mid_keyframe_strength", os.environ["EDGE_V10_MID_STRENGTH"])
    if os.environ.get("EDGE_V10_KEYFRAME_WIDTH"):
        argv = _replace_arg(argv, "--infer_keyframe_width", os.environ["EDGE_V10_KEYFRAME_WIDTH"])

    print("🧩 V10 choreo planner enabled")
    print(f"  mode={mode}")
    print(f"  plan={plan.get('plan_path')}")
    print(f"  mid_poses={mid_poses}")
    print(f"  mid_pose_frames={mid_frames}")
    print(f"  forwarded: python generate_controlled_v9.py {' '.join(shlex.quote(x) for x in argv)}")
    return argv


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print("Usage: python generate_v10_choreo.py [same args as generate_controlled.py]")
        return 2

    # Use the reliable wrapper so the V9 patch is installed before
    # generate_controlled.py imports EDGE.
    cmd = [sys.executable, "generate_controlled_v9.py"] + build_forward_argv(argv)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
