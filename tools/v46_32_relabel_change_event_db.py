#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Relabel Chang-E all-train events for V46.32 Event-RAG routing.

This fixes the common low-resource failure in which many event windows inherit
pose_hold/pose_motif even when their dance_key is lotus/pipa/meditation.  It is
safe to run after build-db and before training contrastive/refiner/diffusion.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import time
import numpy as np

MAPPING = {
    "thirty_six_postures": ("pose_motif", "pose_hold", "anchor_or_resolution", "in_place_pose", "stable_support"),
    "revelation_meditation": ("calm_flow", "calm_meditative", "anchor_or_resolution", "in_place_pose", "stable_support"),
    "lotus_steps": ("footwork_flow", "footwork_flow", "build_up", "traveling_steps", None),
    "pipa_behind_back": ("instrument_motif", "instrument_phrase", "motif_recall", "upper_body_phrase", None),
    "sogdian_whirl": ("turning_flow", "turning_climax", "climax", "turning_travel", None),
    "ribbon_flow": ("turning_flow", "turning_climax", "climax", "turning_travel", None),
    "lei_gong_drum": ("percussive_accent", "percussive_accent", "climax", "traveling_steps", None),
}


def counts(arr):
    vals, cnt = np.unique(arr.astype(str), return_counts=True)
    return {str(v): int(c) for v, c in zip(vals, cnt)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to events.npz")
    ap.add_argument("--backup", action="store_true", default=True)
    args = ap.parse_args()
    p = Path(args.db)
    if not p.exists():
        raise SystemExit(f"missing db: {p}")
    if args.backup:
        bak = p.with_suffix(p.suffix + f".before_v46_32_relabel_{time.strftime('%Y%m%d_%H%M%S')}.bak")
        shutil.copy2(p, bak)
        print("[BAK]", bak)

    db = np.load(p, allow_pickle=True)
    data = {k: db[k] for k in db.files}
    if "dance_keys" not in data:
        raise SystemExit("events.npz has no dance_keys; cannot relabel")

    dance = data["dance_keys"].astype(str)
    event_family = data.get("event_families", np.array(["unknown"] * len(dance), dtype=object)).astype(object)
    music_align = data.get("music_alignment_labels", np.array(["lyrical_flow"] * len(dance), dtype=object)).astype(object)
    stage_role = data.get("motion_stage_roles", np.array(["normal"] * len(dance), dtype=object)).astype(object)
    locomotion = data.get("locomotion_labels", np.array(["unknown"] * len(dance), dtype=object)).astype(object)
    support = data.get("support_labels", np.array(["stable_support"] * len(dance), dtype=object)).astype(object)

    for i, key in enumerate(dance):
        fam, align, stage, loco, supp = MAPPING.get(str(key), (None, None, None, None, None))
        if fam is not None:
            event_family[i] = fam
        if align is not None:
            music_align[i] = align
        if stage is not None:
            stage_role[i] = stage
        if loco is not None:
            locomotion[i] = loco
        if supp is not None:
            support[i] = supp

    data["event_families"] = event_family.astype(object)
    data["music_alignment_labels"] = music_align.astype(object)
    data["motion_stage_roles"] = stage_role.astype(object)
    data["locomotion_labels"] = locomotion.astype(object)
    data["support_labels"] = support.astype(object)
    np.savez(p, **data)

    print("[OK] relabeled", p)
    for k in ["dance_keys", "event_families", "music_alignment_labels", "motion_stage_roles", "locomotion_labels", "support_labels"]:
        if k in data:
            print("\n==", k, "==")
            for name, c in counts(data[k]).items():
                print(f"{name:32s} {c}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
