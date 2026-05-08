#!/usr/bin/env python3
"""Guarded V9/V10 inference entrypoint.

Run with exactly the same CLI arguments as ``generate_controlled.py``.

Why use this wrapper instead of calling generate_controlled.py directly?
-----------------------------------------------------------------------
``generate_controlled.py`` contains V10-friendly ``os.environ.setdefault``
defaults.  This wrapper runs ``edge_experiment_guard`` first, so those defaults
cannot accidentally turn a clean V9/baseline run into a Text/Pose Context RAG
run.  ``generate_v10_choreo.py`` also calls this wrapper internally.

Recommended clean V9 baseline:
    EDGE_RUN_MODE=formal \
    EDGE_EXPERIMENT_PROFILE=v9_baseline \
    EDGE_STRICT_EXPERIMENT_GUARD=1 \
    EDGE_STRICT_RUNTIME_PATCHES=1 \
    python generate_controlled_v9.py ...
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_TRUE = {"1", "true", "yes", "y", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in _TRUE


def _bootstrap(repo_root: Path) -> str:
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from edge_experiment_guard import (
        assert_inference_contract,
        configure_inference_feature_flags,
        env_bool,
        infer_checkpoint_path_from_argv,
        install_runtime_patches,
        normalize_profile,
    )

    checkpoint = infer_checkpoint_path_from_argv(sys.argv[1:])
    profile = normalize_profile(os.environ.get("EDGE_EXPERIMENT_PROFILE", "auto"))

    # Marker for scripts and logs.  generate_controlled.py itself is not edited
    # here, but this marker records that the guarded path was used.
    os.environ["EDGE_GENERATE_CONTROLLED_GUARDED"] = "1"

    configure_inference_feature_flags(
        checkpoint_path=checkpoint,
        profile=profile,
        verbose=True,
    )

    install_runtime_patches(
        strict=env_bool("EDGE_STRICT_RUNTIME_PATCHES", False),
        profile=profile,
        verbose=True,
    )

    if env_bool("EDGE_STRICT_EXPERIMENT_GUARD", True):
        assert_inference_contract(
            checkpoint_path=checkpoint,
            profile=profile,
            strict=True,
        )

    if profile in {"v9_baseline", "baseline", "pure_v9", "v9"}:
        if _env_bool("EDGE_ENABLE_TEXT_CONTEXT_RAG", False):
            raise RuntimeError(
                "Clean V9/baseline profile still has EDGE_ENABLE_TEXT_CONTEXT_RAG=1 after guard. "
                "This would contaminate the baseline."
            )

    return checkpoint


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    _bootstrap(repo_root)

    target = repo_root / "generate_controlled.py"
    if not target.exists():
        raise FileNotFoundError(f"generate_controlled.py not found at {target}")

    sys.argv = [str(target)] + sys.argv[1:]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
