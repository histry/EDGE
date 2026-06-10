#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install the V29 overlay into an EDGE repository with automatic backups."""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge_root",
        required=True,
        help="EDGE repository root, e.g. /home/disk/lsm/storage/EDGE",
    )
    parser.add_argument(
        "--patch_root",
        default=str(Path(__file__).resolve().parent),
    )
    args = parser.parse_args()

    edge_root = Path(args.edge_root).resolve()
    patch_root = Path(args.patch_root).resolve()
    if not (edge_root / "tools").is_dir():
        raise RuntimeError(f"Not an EDGE repository: {edge_root}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = edge_root / f"backup_v29_{timestamp}"
    targets = [
        Path("tools/v29_motion_geometry.py"),
        Path("tools/v27_transition_diffusion.py"),
        Path("tools/build_v27_transition_diffusion_dataset.py"),
        Path("train_v27_transition_diffusion.py"),
        Path("tools/schedule_v29_whole_song.py"),
        Path("tools/evaluate_v26_long_dance.py"),
        Path("tools/evaluate_v27_public_metrics.py"),
        Path("tools/diagnose_v29_jitter.py"),
        Path("render_from_npy.py"),
        Path("scripts/run_v29_whole_song.sh"),
        Path("scripts/run_v29_rebuild_retrain_generate.sh"),
    ]

    for relative in targets:
        source = patch_root / relative
        destination = edge_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists():
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"[INSTALLED] {relative}")

    for script in (
        edge_root / "scripts/run_v29_whole_song.sh",
        edge_root / "scripts/run_v29_rebuild_retrain_generate.sh",
    ):
        script.chmod(script.stat().st_mode | 0o111)

    print(f"[BACKUP] {backup_root}")
    print("[DONE] V29 patch installed")


if __name__ == "__main__":
    main()
