#!/usr/bin/env python3
"""Reliable V9/V10 inference entrypoint.

Run this file with exactly the same arguments as generate_controlled.py.

Why this wrapper is needed:
- Python does not always import sitecustomize.py from the repository root.
- If the V9 patch is not installed before generate_controlled.py imports EDGE,
  the V9 checkpoint's rag_summary_projection weights are ignored at load time.
- This wrapper installs the V9 RAG inference patch first, then executes
  generate_controlled.py as __main__.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    os.environ.setdefault("EDGE_ENABLE_RAG_SUMMARY_TOKEN", "1")
    os.environ.setdefault("EDGE_RAG_SUMMARY_DIM", "7")
    os.environ.setdefault("EDGE_RAG_SUMMARY_BLEND_RADIUS", "18")
    os.environ.setdefault("EDGE_RAG_SUMMARY_MODE", "mean")

    from v9_rag_inference_patch import install_v9_rag_inference_patch
    install_v9_rag_inference_patch(verbose=True)

    target = repo_root / "generate_controlled.py"
    if not target.exists():
        raise FileNotFoundError(f"generate_controlled.py not found at {target}")

    sys.argv = [str(target)] + sys.argv[1:]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
