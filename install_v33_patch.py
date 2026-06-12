#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install EDGE V33 event-level contact supervision with automatic backups."""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

TARGETS = [
    "render_from_npy.py",
    "train_v27_transition_diffusion.py",
    "tools/v29_motion_geometry.py",
    "tools/v30_continuous_inr.py",
    "tools/v32_contact_inr.py",
    "tools/v32_contact_losses.py",
    "tools/v32_transition_quality.py",
    "tools/v27_transition_diffusion.py",
    "tools/v33_event_contacts.py",
    "tools/build_v33_event_contact_cache.py",
    "tools/build_v27_transition_diffusion_dataset.py",
    "tools/build_v30_source_manifest.py",
    "tools/build_v30_pair_manifest_from_schedules.py",
    "tools/audit_v32_contact_dataset.py",
    "tools/relabel_v32_contacts.py",
    "tools/schedule_v32_whole_song.py",
    "tools/evaluate_v26_long_dance.py",
    "tools/evaluate_v27_public_metrics.py",
    "tools/evaluate_v30_frequency_metrics.py",
    "tools/evaluate_v32_contact_metrics.py",
    "tools/diagnose_v29_jitter.py",
    "tools/summarize_v32_transition_gate.py",
    "scripts/run_v32_whole_song.sh",
    "scripts/run_v32_contact_inr_full.sh",
    "scripts/launch_v32_weak_overnight.sh",
    "scripts/run_v33_event_contact_inr_full.sh",
    "scripts/launch_v33_event_contact_overnight.sh",
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
        "backup_v33_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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
        "scripts/run_v32_whole_song.sh",
        "scripts/run_v32_contact_inr_full.sh",
        "scripts/launch_v32_weak_overnight.sh",
        "scripts/run_v33_event_contact_inr_full.sh",
        "scripts/launch_v33_event_contact_overnight.sh",
    ):
        target = edge / relative
        target.chmod(target.stat().st_mode | 0o111)

    print(f"[BACKUP] {backup}")
    print("[DONE] V33 event-level contact supervision installed")


if __name__ == "__main__":
    main()
