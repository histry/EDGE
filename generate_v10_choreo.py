#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


def _repo_bootstrap(repo_root: Path) -> None:
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from edge_experiment_guard import (
        assert_inference_contract,
        configure_inference_feature_flags,
        env_bool,
        infer_checkpoint_path_from_argv,
        install_runtime_patches,
    )

    # V10 wrapper should use the V10 contract unless the caller explicitly asks
    # for another profile.
    os.environ.setdefault("EDGE_EXPERIMENT_PROFILE", "v10")
    checkpoint = infer_checkpoint_path_from_argv(sys.argv[1:])

    configure_inference_feature_flags(
        checkpoint_path=checkpoint,
        profile=os.environ.get("EDGE_EXPERIMENT_PROFILE", "v10"),
        verbose=True,
    )
    install_runtime_patches(
        strict=env_bool("EDGE_STRICT_RUNTIME_PATCHES", False),
        profile=os.environ.get("EDGE_EXPERIMENT_PROFILE", "v10"),
        verbose=True,
    )

    if env_bool("EDGE_STRICT_EXPERIMENT_GUARD", True):
        assert_inference_contract(
            checkpoint_path=checkpoint,
            profile=os.environ.get("EDGE_EXPERIMENT_PROFILE", "v10"),
            strict=True,
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
    return _strip_arg(argv, name, takes_value=True) + [name, value]


def _default_prefix_from_out(argv: List[str]) -> str:
    out = _get_arg(argv, "--out", "output/v10_eval/v10_choreo.npy")
    return str(Path(out).with_suffix(""))


def _prepare_env_defaults() -> None:
    # Path/reproducibility: allow scripts to run outside one fixed server path.
    os.environ.setdefault("EDGE_ROOT", str(Path.cwd()))

    # Unit prior defaults. These are inert unless generate_controlled.py finds
    # corresponding unit specs or EDGE_UNIT_SOFT_PRIOR=1 is used downstream.
    os.environ.setdefault("EDGE_UNIT_SOFT_PRIOR", "1")
    os.environ.setdefault("EDGE_UNIT_PRIOR_DCT", "1")
    os.environ.setdefault("EDGE_UNIT_PRIOR_LOW_FREQ_K", "4")
    os.environ.setdefault("EDGE_UNIT_PRIOR_FEATURES", "upper")
    os.environ.setdefault("EDGE_UNIT_PRIOR_STRENGTH", "0.012")
    os.environ.setdefault("EDGE_UNIT_PRIOR_MAX_LEN", "45")

    os.environ.setdefault("EDGE_TENSION_AWARE_PLANNER", "1")

    # Full RAG DB scan for formal V10 runs. The old default in v10_choreo_planner.py
    # was 20000, which can bias retrieval if the DB has ~77k units. This wrapper
    # overrides it unless the user explicitly sets a smaller cap for smoke tests.
    os.environ.setdefault("EDGE_V10_MAX_RAG_UNITS", "1000000")

    # V9 summary shape defaults. Activation itself is controlled by
    # edge_experiment_guard based on checkpoint keys.
    os.environ.setdefault("EDGE_RAG_SUMMARY_DIM", "7")
    os.environ.setdefault("EDGE_RAG_SUMMARY_BLEND_RADIUS", "18")
    os.environ.setdefault("EDGE_RAG_SUMMARY_MODE", "mean")

    # Text Context shape defaults. Activation itself is controlled by guard.
    if os.environ.get("EDGE_ENABLE_TEXT_CONTEXT_RAG", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
        os.environ.setdefault("EDGE_TEXT_CONTEXT_DIM", os.environ.get("EDGE_TEXT_BRIDGE_FALLBACK_DIM", "512"))
        os.environ.setdefault("EDGE_TEXT_CONTEXT_MAX_POSE_TOKENS", "64")
        os.environ.setdefault("EDGE_RAG_CONTEXT_MAX_LEN", "45")


def _export_v9_rag_summary_env(plan: dict) -> None:
    unit_paths = [str(p) for p in (plan.get("unit_paths") or []) if p]
    frames = [int(x) for x in (plan.get("mid_pose_frames") or [])]
    if not unit_paths:
        os.environ.pop("EDGE_RAG_SUMMARY_UNIT_PATHS", None)
        os.environ.pop("EDGE_RAG_SUMMARY_MID_FRAMES", None)
        print("ℹ️ V9 RAG summary inference token disabled for this run: no unit_paths in plan.")
        return

    if os.environ.get("EDGE_ENABLE_RAG_SUMMARY_TOKEN", "0").strip().lower() in {"1", "true", "yes", "y", "on"}:
        os.environ["EDGE_RAG_SUMMARY_UNIT_PATHS"] = ",".join(unit_paths)
        os.environ["EDGE_RAG_SUMMARY_MID_FRAMES"] = ",".join(str(x) for x in frames)

    if os.environ.get("EDGE_ENABLE_TEXT_CONTEXT_RAG", "0").strip().lower() in {"1", "true", "yes", "y", "on"}:
        os.environ["EDGE_RAG_CONTEXT_UNIT_PATHS"] = ",".join(unit_paths)

    print("✅ V10 planner context env exported:")
    print(f"  unit_paths={len(unit_paths)}")
    print(f"  mid_frames={frames}")
    print(f"  EDGE_ENABLE_RAG_SUMMARY_TOKEN={os.environ.get('EDGE_ENABLE_RAG_SUMMARY_TOKEN', '0')}")
    print(f"  EDGE_ENABLE_TEXT_CONTEXT_RAG={os.environ.get('EDGE_ENABLE_TEXT_CONTEXT_RAG', '0')}")
    print(f"  EDGE_V10_MAX_RAG_UNITS={os.environ.get('EDGE_V10_MAX_RAG_UNITS')}")


def build_forward_argv(argv: List[str]) -> List[str]:
    _prepare_env_defaults()

    # Import after guard/sitecustomize so planner patches and env isolation are active.
    from v10_choreo_planner import build_config_from_env, env_int, plan_choreo_from_rag_db

    num_frames = env_int("EDGE_V10_NUM_FRAMES", 150)
    out_prefix = os.environ.get("EDGE_V10_OUT_PREFIX", _default_prefix_from_out(argv))

    for flag, takes in [
        ("--auto_mid_keyframes", False),
        ("--auto_mid_count", True),
        ("--mid_poses", True),
        ("--mid_pose_frames", True),
    ]:
        argv = _strip_arg(argv, flag, takes)

    config = build_config_from_env(num_frames=num_frames)
    rag_db = os.environ.get("EDGE_V10_RAG_DB") or os.environ.get("RAG_DB", "")
    if not rag_db and not config.manual_mid_poses:
        raise RuntimeError(
            "Set EDGE_V10_RAG_DB/RAG_DB for auto or EDGE_V10_MANUAL_UNITS planning. "
            "Only legacy EDGE_V10_MANUAL_MID_POSES can run without a RAG DB."
        )

    start_pose = _get_arg(argv, "--start_pose", "")
    end_pose = _get_arg(argv, "--end_pose", "")

    plan = plan_choreo_from_rag_db(
        rag_db=rag_db,
        out_prefix=out_prefix,
        config=config,
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

    print("🧩 V10 Unified Choreo Planner enabled")
    print(f"  mode={config.mode}")
    print(f"  search_method={config.search_method}, top_k={config.top_k}, beam_width={config.beam_width}")
    print(f"  mid_frames={config.mid_frames}")
    print(f"  manual_units={config.manual_units}")
    print(f"  plan={plan.get('plan_path')}")
    print(f"  mid_poses={mid_poses}")
    print(f"  forwarded: python generate_controlled_v9.py {' '.join(shlex.quote(x) for x in argv)}")
    return argv


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print("Usage: python generate_v10_choreo.py [same args as generate_controlled.py]")
        return 2

    repo_root = Path(__file__).resolve().parent
    _repo_bootstrap(repo_root)

    cmd = [sys.executable, str(repo_root / "generate_controlled_v9.py")] + build_forward_argv(argv)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
