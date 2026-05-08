#!/usr/bin/env python3
"""Convenience wrapper for clean V9/baseline inference.

This file intentionally does not add new model behavior.  It only presets a
safe experiment profile and delegates to ``generate_controlled_v9.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    os.environ.setdefault("EDGE_RUN_MODE", "formal")
    os.environ.setdefault("EDGE_EXPERIMENT_PROFILE", "v9_baseline")
    os.environ.setdefault("EDGE_STRICT_EXPERIMENT_GUARD", "1")
    os.environ.setdefault("EDGE_STRICT_RUNTIME_PATCHES", "1")
    os.environ.setdefault("EDGE_ENABLE_TEXT_CONTEXT_RAG", "0")
    os.environ.setdefault("EDGE_UNIT_SOFT_PRIOR", "0")

    cmd = [sys.executable, str(repo_root / "generate_controlled_v9.py")] + sys.argv[1:]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
