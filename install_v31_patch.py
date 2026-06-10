#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install EDGE V31 with automatic backups."""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

TARGETS = [
    "render_from_npy.py",
    "train_v27_transition_diffusion.py",
    "train_v30_geometric_alignment.py",
    "tools/v29_motion_geometry.py",
    "tools/v31_bandlimited_transition.py",
    "tools/v31_transition_quality.py",
    "tools/v27_transition_diffusion.py",
    "tools/build_v27_transition_diffusion_dataset.py",
    "tools/v30_geometric_alignment.py",
    "tools/v30_deep_music_features.py",
    "tools/build_v30_alignment_dataset.py",
    "tools/build_v30_geometric_event_index.py",
    "tools/build_v30_source_manifest.py",
    "tools/build_v30_pair_manifest_from_schedules.py",
    "tools/enrich_v30_transition_music.py",
    "tools/schedule_v31_whole_song.py",
    "tools/evaluate_v26_long_dance.py",
    "tools/evaluate_v27_public_metrics.py",
    "tools/evaluate_v30_frequency_metrics.py",
    "tools/evaluate_v31_retrieval_geometry.py",
    "tools/diagnose_v29_jitter.py",
    "tools/summarize_v31_transition_gate.py",
    "scripts/run_v31_whole_song.sh",
    "scripts/run_v31_full_research.sh",
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
    backup = edge / (
        "backup_v31_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    for relative in TARGETS:
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
        print(f"[INSTALLED] {relative}")
    for relative in (
        "scripts/run_v31_whole_song.sh",
        "scripts/run_v31_full_research.sh",
    ):
        path = edge / relative
        path.chmod(path.stat().st_mode | 0o111)
    print(f"[BACKUP] {backup}")
    print("[DONE] V31 installed")


if __name__ == "__main__":
    main()
