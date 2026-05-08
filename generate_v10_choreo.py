#!/usr/bin/env python3
"""Formal-safe V10 ChoreoRAG wrapper.

Drop-in replacement for ``generate_v10_choreo.py``.

Adds:
1. EDGE experiment guard before importing EDGE/planner.
2. Formal rejection of legacy auto-mid flags.
3. Explicit unit prior paths/frames exported from V10 plan.
4. Formal defaults for Temporal Unit Prior reports and optional jerk penalty.
5. Optional formal raw-vs-final evaluation after generation.

Recommended formal command prefix:
    EDGE_RUN_MODE=formal \
    EDGE_EXPERIMENT_PROFILE=v10 \
    EDGE_STRICT_EXPERIMENT_GUARD=1 \
    EDGE_STRICT_RUNTIME_PATCHES=1 \
    EDGE_UNIT_PRIOR_REQUIRED=1 \
    python generate_v10_choreo.py ...
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def _env_flag(name: str, default: str = "0") -> bool:
    text = str(os.environ.get(name, default)).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return str(default).strip().lower() in _TRUE


def _formal() -> bool:
    return str(os.environ.get("EDGE_RUN_MODE", "")).strip().lower() == "formal"


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

    os.environ.setdefault("EDGE_EXPERIMENT_PROFILE", "v10")
    if _formal():
        os.environ.setdefault("EDGE_STRICT_EXPERIMENT_GUARD", "1")
        os.environ.setdefault("EDGE_STRICT_RUNTIME_PATCHES", "1")

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
    out: List[str] = []
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


def _has_arg(argv: List[str], name: str) -> bool:
    return any(item == name or item.startswith(name + "=") for item in argv)


def _get_arg(argv: List[str], name: str, default: str = "") -> str:
    for i, item in enumerate(argv):
        if item == name and i + 1 < len(argv):
            return argv[i + 1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return default


def _replace_arg(argv: List[str], name: str, value: str) -> List[str]:
    return _strip_arg(argv, name, takes_value=True) + [name, value]


def _append_arg_if_missing(argv: List[str], name: str, value: str) -> List[str]:
    if value and not _has_arg(argv, name):
        return list(argv) + [name, value]
    return list(argv)


def _normalize_generate_cli_aliases(argv: List[str]) -> List[str]:
    """Normalize user-facing aliases before forwarding to generate_controlled.py.

    The historical lower-level script expects ``--music``.  Several README /
    shell snippets used ``--audio``.  Accept ``--audio`` here and translate it
    to ``--music`` so old commands continue to work.  Also fail early on literal
    ellipsis placeholders from copied examples, because otherwise the V10
    planner may run before argparse reports missing required arguments.
    """
    argv = list(argv)

    bad = [x for x in argv if x == "..." or str(x).startswith("...")]
    if bad:
        raise RuntimeError(
            "Your command still contains placeholder ellipsis tokens: "
            f"{bad}. Replace them with real generate_controlled.py arguments, "
            "for example --checkpoint, --music, --start_pose, --end_pose and --out."
        )

    out: List[str] = []
    i = 0
    has_music = _has_arg(argv, "--music")
    while i < len(argv):
        item = argv[i]
        if item == "--audio":
            if i + 1 >= len(argv):
                raise RuntimeError("--audio was provided without a following file path.")
            if not has_music:
                out += ["--music", argv[i + 1]]
                has_music = True
            i += 2
            continue
        if item.startswith("--audio="):
            if not has_music:
                out.append("--music=" + item.split("=", 1)[1])
                has_music = True
            i += 1
            continue
        out.append(item)
        i += 1

    # Optional env fallbacks make formal batch scripts less brittle.  Explicit
    # CLI args still win and are recommended for reproducibility.
    out = _append_arg_if_missing(out, "--checkpoint", os.environ.get("CHECKPOINT") or os.environ.get("EDGE_CHECKPOINT", ""))
    out = _append_arg_if_missing(out, "--music", os.environ.get("MUSIC") or os.environ.get("EDGE_MUSIC") or os.environ.get("AUDIO") or os.environ.get("EDGE_AUDIO", ""))
    out = _append_arg_if_missing(out, "--start_pose", os.environ.get("START_POSE") or os.environ.get("EDGE_START_POSE", ""))
    out = _append_arg_if_missing(out, "--end_pose", os.environ.get("END_POSE") or os.environ.get("EDGE_END_POSE", ""))
    out = _append_arg_if_missing(out, "--out", os.environ.get("OUT") or os.environ.get("EDGE_OUT", ""))

    return out


def _assert_required_generate_args(argv: List[str]) -> None:
    missing = []
    for flag in ("--checkpoint", "--music", "--start_pose", "--end_pose", "--out"):
        if not _has_arg(argv, flag):
            missing.append(flag)
    if missing:
        raise RuntimeError(
            "generate_v10_choreo.py forwards to generate_controlled.py, whose "
            f"required arguments are missing: {missing}. Note: use --music for "
            "the audio file path; --audio is accepted by this wrapper as an alias "
            "and will be translated to --music."
        )


def _default_prefix_from_out(argv: List[str]) -> str:
    out = _get_arg(argv, "--out", "output/v10_eval/v10_choreo.npy")
    return str(Path(out).with_suffix(""))


def _prepare_env_defaults(out_prefix: str = "") -> None:
    os.environ.setdefault("EDGE_ROOT", str(Path.cwd()))

    # Formal V10 should be explicit about temporal prior behavior.
    os.environ.setdefault("EDGE_UNIT_SOFT_PRIOR", "1")
    os.environ.setdefault("EDGE_UNIT_PRIOR_TEMPORAL", "1")
    os.environ.setdefault("EDGE_UNIT_PRIOR_DCT", "1")
    os.environ.setdefault("EDGE_UNIT_PRIOR_LOW_FREQ_K", "4")
    os.environ.setdefault("EDGE_UNIT_PRIOR_FEATURES", "upper+torso")
    os.environ.setdefault("EDGE_UNIT_PRIOR_STRENGTH", "0.006" if _formal() else "0.012")
    os.environ.setdefault("EDGE_UNIT_PRIOR_MAX_LEN", "45")
    if _formal():
        os.environ.setdefault("EDGE_UNIT_PRIOR_REQUIRED", "1")
        os.environ.setdefault("EDGE_FORMAL_AUTO_EVAL", "1")
        os.environ.setdefault("EDGE_FORMAL_EVAL_REQUIRED", "1")
        if out_prefix and not os.environ.get("EDGE_UNIT_PRIOR_REPORT_JSON"):
            os.environ["EDGE_UNIT_PRIOR_REPORT_JSON"] = out_prefix + "_unit_prior_report.json"
        if out_prefix and not os.environ.get("EDGE_RAG_CONTEXT_REPORT_JSON"):
            os.environ["EDGE_RAG_CONTEXT_REPORT_JSON"] = out_prefix + "_context_report.json"

    os.environ.setdefault("EDGE_TENSION_AWARE_PLANNER", "1")
    os.environ.setdefault("EDGE_V10_MAX_RAG_UNITS", "1000000")

    os.environ.setdefault("EDGE_V10_JERK_PENALTY", "1" if _formal() else "0")
    os.environ.setdefault("EDGE_V10_JERK_THRESHOLD", "0.18")
    os.environ.setdefault("EDGE_V10_JERK_PENALTY_WEIGHT", "0.35")
    os.environ.setdefault("EDGE_V10_JERK_PENALTY_SCALE", "8.0")

    os.environ.setdefault("EDGE_RAG_SUMMARY_DIM", "7")
    os.environ.setdefault("EDGE_RAG_SUMMARY_BLEND_RADIUS", "18")
    os.environ.setdefault("EDGE_RAG_SUMMARY_MODE", "mean")

    if os.environ.get("EDGE_ENABLE_TEXT_CONTEXT_RAG", "").strip().lower() in _TRUE:
        os.environ.setdefault("EDGE_TEXT_CONTEXT_DIM", os.environ.get("EDGE_TEXT_BRIDGE_FALLBACK_DIM", "512"))
        os.environ.setdefault("EDGE_TEXT_CONTEXT_MAX_POSE_TOKENS", "64")
        os.environ.setdefault("EDGE_RAG_CONTEXT_MAX_LEN", "45")


def _install_optional_planner_patch() -> None:
    if not _env_flag("EDGE_V10_JERK_PENALTY", "0"):
        return
    try:
        from v10_choreo_planner_formal_patch import install_v10_choreo_planner_formal_patch
        install_v10_choreo_planner_formal_patch(verbose=True)
    except Exception as exc:
        if _formal():
            raise RuntimeError(f"Formal V10 requested planner jerk patch but it failed: {exc}") from exc
        print(f"⚠️ V10 formal planner patch skipped: {exc}")


def _assert_no_legacy_auto_mid(argv: List[str]) -> None:
    legacy = {
        "--auto_mid_keyframes": False,
        "--auto_mid_count": True,
    }
    if not _formal():
        return
    leaked = [flag for flag in legacy if _has_arg(argv, flag)]
    if leaked:
        raise RuntimeError(
            "Formal V10 runs must not use legacy auto-mid flags because the old planner "
            f"can silently reduce mid count. Found: {leaked}. Use EDGE_V10_MODE and "
            "EDGE_V10_MID_FRAMES instead."
        )


def _export_v9_rag_summary_env(plan: dict) -> None:
    unit_paths = [str(p) for p in (plan.get("unit_paths") or []) if p]
    frames = [int(x) for x in (plan.get("mid_pose_frames") or [])]

    if not unit_paths:
        os.environ.pop("EDGE_RAG_SUMMARY_UNIT_PATHS", None)
        os.environ.pop("EDGE_RAG_SUMMARY_MID_FRAMES", None)
        os.environ.pop("EDGE_RAG_CONTEXT_UNIT_PATHS", None)
        os.environ.pop("EDGE_UNIT_PRIOR_UNIT_PATHS", None)
        os.environ.pop("EDGE_UNIT_PRIOR_MID_FRAMES", None)
        print("ℹ️ V9/V10 RAG context disabled for this run: no unit_paths in plan.")
        if _formal() and _env_flag("EDGE_UNIT_SOFT_PRIOR", "1"):
            raise RuntimeError("Formal V10 run expected unit_paths for temporal prior, but V10 plan has none.")
        return

    if os.environ.get("EDGE_ENABLE_RAG_SUMMARY_TOKEN", "0").strip().lower() in _TRUE:
        os.environ["EDGE_RAG_SUMMARY_UNIT_PATHS"] = ",".join(unit_paths)
        os.environ["EDGE_RAG_SUMMARY_MID_FRAMES"] = ",".join(str(x) for x in frames)

    if os.environ.get("EDGE_ENABLE_TEXT_CONTEXT_RAG", "0").strip().lower() in _TRUE:
        os.environ["EDGE_RAG_CONTEXT_UNIT_PATHS"] = ",".join(unit_paths)

    os.environ["EDGE_UNIT_PRIOR_UNIT_PATHS"] = ",".join(unit_paths)
    os.environ["EDGE_UNIT_PRIOR_MID_FRAMES"] = ",".join(str(x) for x in frames)

    print("✅ V10 planner context env exported:")
    print(f"  unit_paths={len(unit_paths)}")
    print(f"  mid_frames={frames}")
    print(f"  EDGE_ENABLE_RAG_SUMMARY_TOKEN={os.environ.get('EDGE_ENABLE_RAG_SUMMARY_TOKEN', '0')}")
    print(f"  EDGE_ENABLE_TEXT_CONTEXT_RAG={os.environ.get('EDGE_ENABLE_TEXT_CONTEXT_RAG', '0')}")
    print(f"  EDGE_UNIT_SOFT_PRIOR={os.environ.get('EDGE_UNIT_SOFT_PRIOR', '0')}")
    print(f"  EDGE_UNIT_PRIOR_REQUIRED={os.environ.get('EDGE_UNIT_PRIOR_REQUIRED', '0')}")
    print(f"  EDGE_V10_MAX_RAG_UNITS={os.environ.get('EDGE_V10_MAX_RAG_UNITS')}")


def _write_wrapper_contract_json(out_prefix: str, plan: dict) -> None:
    if not (_formal() or os.environ.get("EDGE_V10_WRAPPER_CONTRACT_JSON")):
        return
    path = os.environ.get("EDGE_V10_WRAPPER_CONTRACT_JSON", out_prefix + "_wrapper_contract.json")
    payload = {
        "run_mode": os.environ.get("EDGE_RUN_MODE", "debug"),
        "experiment_profile": os.environ.get("EDGE_EXPERIMENT_PROFILE", ""),
        "mid_pose_frames": plan.get("mid_pose_frames", []),
        "mid_poses": plan.get("mid_poses", []),
        "unit_paths": plan.get("unit_paths", []),
        "env": {k: v for k, v in os.environ.items() if k.startswith("EDGE_")},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ V10 wrapper contract saved: {p}")


def build_forward_argv(argv: List[str]) -> List[str]:
    argv = _normalize_generate_cli_aliases(argv)
    _assert_required_generate_args(argv)
    _assert_no_legacy_auto_mid(argv)

    out_prefix = os.environ.get("EDGE_V10_OUT_PREFIX", _default_prefix_from_out(argv))
    _prepare_env_defaults(out_prefix=out_prefix)
    _install_optional_planner_patch()

    from v10_choreo_planner import build_config_from_env, env_int, plan_choreo_from_rag_db

    num_frames = env_int("EDGE_V10_NUM_FRAMES", 150)

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
    _write_wrapper_contract_json(out_prefix, plan)

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


def _maybe_run_formal_eval(repo_root: Path, argv: List[str]) -> None:
    if not _formal() or not _env_flag("EDGE_FORMAL_AUTO_EVAL", "1"):
        return

    out_path = Path(_get_arg(argv, "--out", ""))
    if not out_path:
        if _env_flag("EDGE_FORMAL_EVAL_REQUIRED", "1"):
            raise RuntimeError("Formal auto-eval requires --out so raw/final asset paths can be inferred.")
        return

    prefix = out_path.with_suffix("")
    raw_motion = Path(str(prefix) + "_raw.npy")
    final_motion = out_path
    target_traj = Path(str(prefix) + "_target_traj.npy")
    meta = Path(str(prefix) + "_meta.json")
    eval_out = Path(str(prefix) + "_eval.json")

    missing = [p for p in [raw_motion, final_motion, target_traj] if not p.exists()]
    if missing:
        msg = "Formal auto-eval missing required assets: " + ", ".join(str(p) for p in missing)
        if _env_flag("EDGE_FORMAL_EVAL_REQUIRED", "1"):
            raise RuntimeError(msg)
        print("⚠️ " + msg)
        return

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "eval_generated_motions.py"),
        "--raw_motion", str(raw_motion),
        "--final_motion", str(final_motion),
        "--target_traj", str(target_traj),
        "--out", str(eval_out),
        "--formal",
    ]
    if meta.exists():
        cmd += ["--meta", str(meta)]

    print("🧪 Running formal raw-vs-final evaluation:")
    print("  " + " ".join(shlex.quote(x) for x in cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        raise RuntimeError(f"Formal raw-vs-final evaluation failed with exit code {ret}")


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print("Usage: python generate_v10_choreo.py [same args as generate_controlled.py]")
        return 2

    repo_root = Path(__file__).resolve().parent
    _repo_bootstrap(repo_root)

    forward_argv = build_forward_argv(argv)
    cmd = [sys.executable, str(repo_root / "generate_controlled_v9.py")] + forward_argv
    ret = subprocess.call(cmd)
    if ret != 0:
        return ret

    _maybe_run_formal_eval(repo_root, forward_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
