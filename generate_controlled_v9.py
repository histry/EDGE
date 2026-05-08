#!/usr/bin/env python3
"""Reliable V9/V10 inference entrypoint with experiment-safety guards.

Run this file with exactly the same arguments as generate_controlled.py.

Key fixes compared with the old wrapper:
- Text/Pose Context RAG is NOT enabled just because generate_controlled.py has
  V10-friendly defaults. It is enabled only when the checkpoint contains
  text_context_* weights, unless the user explicitly sets EDGE_ENABLE_TEXT_CONTEXT_RAG=1.
- Clean V9 baselines can set EDGE_EXPERIMENT_PROFILE=v9_baseline and will hard
  fail if Text/Pose Context RAG is accidentally enabled.
- Required runtime patches can be checked hard with EDGE_STRICT_RUNTIME_PATCHES=1.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


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
