#!/usr/bin/env python3
"""V11 experimental ChoreoRAG wrapper.

This wrapper reuses generate_v10_choreo.py but installs next-generation runtime
patches after the normal V10 bootstrap and before the model/planner path runs.

Use this for:
- beat-guided sampling
- V11 explicit Cross-Attention RAG
- differentiable contact loss compatibility when running inference utilities

It accepts the same CLI arguments as generate_v10_choreo.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import generate_v10_choreo as v10


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print("Usage: python generate_v11_choreo.py [same args as generate_v10_choreo.py]")
        return 2

    repo_root = Path(__file__).resolve().parent
    v10._repo_bootstrap(repo_root)

    try:
        from edge_nextgen_runtime_patch import install_nextgen_runtime_patches
        install_nextgen_runtime_patches(verbose=True)
    except Exception as exc:
        if v10._formal():
            raise RuntimeError(f"V11 nextgen patch install failed in formal run: {exc}") from exc
        print(f"⚠️ V11 nextgen patches skipped: {exc}")

    forward_argv = v10.build_forward_argv(argv)
    cmd = [sys.executable, str(repo_root / "generate_controlled_v9.py")] + forward_argv
    ret = subprocess.call(cmd)
    if ret != 0:
        return ret

    v10._maybe_run_formal_eval(repo_root, forward_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
