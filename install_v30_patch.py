#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install the EDGE V30 research overlay with automatic backups."""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


TARGETS = [
    "tools/v29_motion_geometry.py",
    "tools/v29_transition_diffusion_legacy.py",
    "tools/v30_continuous_inr.py",
    "tools/v30_geometric_alignment.py",
    "tools/v27_transition_diffusion.py",
    "tools/build_v27_transition_diffusion_dataset.py",
    "tools/build_v30_source_manifest.py",
    "tools/build_v30_pair_manifest_from_schedules.py",
    "tools/build_v30_alignment_dataset.py",
    "tools/enrich_v30_transition_music.py",
    "tools/build_v30_geometric_event_index.py",
    "tools/v30_deep_music_features.py",
    "tools/schedule_v30_whole_song.py",
    "tools/evaluate_v26_long_dance.py",
    "tools/evaluate_v27_public_metrics.py",
    "tools/evaluate_v30_frequency_metrics.py",
    "tools/diagnose_v29_jitter.py",
    "train_v27_transition_diffusion.py",
    "train_v30_geometric_alignment.py",
    "render_from_npy.py",
    "scripts/run_v30_whole_song.sh",
    "scripts/run_v30_full_research.sh",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_root", required=True)
    parser.add_argument(
        "--patch_root", default=str(Path(__file__).resolve().parent)
    )
    args = parser.parse_args()

    edge = Path(args.edge_root).resolve()
    patch = Path(args.patch_root).resolve()
    if not (edge / "tools").is_dir():
        raise RuntimeError(f"Not an EDGE repository: {edge}")

    backup = edge / f"backup_v30_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    for relative in TARGETS:
        source = patch / relative
        destination = edge / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists():
            old = backup / relative
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, old)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"[INSTALLED] {relative}")

    for relative in (
        "scripts/run_v30_whole_song.sh",
        "scripts/run_v30_full_research.sh",
    ):
        path = edge / relative
        path.chmod(path.stat().st_mode | 0o111)

    print(f"[BACKUP] {backup}")
    print("[DONE] EDGE V30 patch installed")


if __name__ == "__main__":
    main()
