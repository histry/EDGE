#!/usr/bin/env python3
"""Reliable V9/V10 inference entrypoint.

Run this file with exactly the same arguments as generate_controlled.py.

Why this wrapper is needed:
- Python does not always import sitecustomize.py from the repository root.
- If runtime patches are not installed before generate_controlled.py imports EDGE,
  V9 RAG Summary Token, Text/Pose Context RAG, and safety patches may not activate.
- This wrapper installs sitecustomize explicitly, then executes generate_controlled.py as __main__.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _install_runtime_patches(repo_root: Path) -> None:
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Defaults required by V9 RAG Summary Token.
    os.environ.setdefault("EDGE_ENABLE_RAG_SUMMARY_TOKEN", "1")
    os.environ.setdefault("EDGE_RAG_SUMMARY_DIM", "7")
    os.environ.setdefault("EDGE_RAG_SUMMARY_BLEND_RADIUS", "18")
    os.environ.setdefault("EDGE_RAG_SUMMARY_MODE", "mean")

    # Force-load all repository runtime patches:
    # - V9 RAG inference patch
    # - full-landing patch
    # - Text/Pose Context RAG model/IO patch
    # - Text Bridge semantic planner patch
    try:
        import sitecustomize  # noqa: F401
    except Exception as exc:
        print(f"⚠️ sitecustomize import failed in generate_controlled_v9.py: {exc}")

    # Hard fallback: at minimum, install V9 RAG bridge directly.
    try:
        from v9_rag_inference_patch import install_v9_rag_inference_patch
        install_v9_rag_inference_patch(verbose=True)
    except Exception as exc:
        print(f"⚠️ direct V9 RAG inference patch install failed: {exc}")

    # Optional direct installs make the wrapper robust even if sitecustomize changes.
    optional_installers = [
        ("edge_full_landing_patch", "install_full_landing_patch"),
        ("text_context_rag_model_patch", "install_text_context_rag_model_patch"),
        ("text_context_rag_io_patch", "install_text_context_rag_io_patch"),
        ("text_bridge_planner_patch", "install_text_bridge_planner_patch"),
    ]
    for module_name, func_name in optional_installers:
        try:
            module = __import__(module_name, fromlist=[func_name])
            func = getattr(module, func_name)
            func(verbose=True)
        except Exception as exc:
            # Do not fail old experiments if optional patches are absent.
            print(f"⚠️ optional runtime patch {module_name}.{func_name} not installed: {exc}")


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    _install_runtime_patches(repo_root)

    target = repo_root / "generate_controlled.py"
    if not target.exists():
        raise FileNotFoundError(f"generate_controlled.py not found at {target}")

    sys.argv = [str(target)] + sys.argv[1:]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
