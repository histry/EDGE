#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


FILES = [
    "tools/v34_warp_aware_retrieval.py",
    "tools/v34_boundary_compatibility.py",
    "tools/v34_gpu_candidate_cache.py",
    "scripts/launch_v34_boundary_compat.sh",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_root", required=True)
    parser.add_argument(
        "--patch_root",
        default=str(Path(__file__).resolve().parent),
    )
    args = parser.parse_args()

    edge = Path(args.edge_root).resolve()
    patch = Path(args.patch_root).resolve()
    if not (edge / "tools").is_dir():
        raise RuntimeError(f"Not an EDGE repository: {edge}")

    backup = edge / ("backup_v34_boundary_compat_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    for relative in FILES:
        source = patch / relative
        destination = edge / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists():
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, saved)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if destination.suffix == ".sh":
            destination.chmod(destination.stat().st_mode | 0o111)
        print(f"[INSTALLED] {relative}")

    print(f"[BACKUP] {backup}")
    print("[DONE] V34 boundary-compatible Event-RAG patch installed")


if __name__ == "__main__":
    main()
