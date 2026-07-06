#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Relabel Chang-E all-train Event-RAG DB for V46.33 experiments.

This rewrites only semantic metadata arrays in events.npz. It does not change
motion arrays or event paths. A timestamped backup is created automatically.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np


def counts(arr):
    vals, cnt = np.unique(arr.astype(str), return_counts=True)
    return [[str(v), int(c)] for v, c in zip(vals, cnt)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to events.npz")
    ap.add_argument("--json", default=None, help="Optional relabel report json")
    ap.add_argument("--backup", action="store_true", default=True)
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"[ERROR] missing DB: {db_path}")

    bak = db_path.with_suffix(db_path.suffix + f".before_v46_33_relabel_{time.strftime('%Y%m%d_%H%M%S')}.bak")
    shutil.copy2(db_path, bak)
    print("[BAK]", bak)

    db = np.load(db_path, allow_pickle=True)
    data = {k: db[k] for k in db.files}
    if "dance_keys" not in data:
        raise SystemExit("[ERROR] events.npz has no dance_keys; cannot relabel Chang-E semantics.")

    before = {k: counts(data[k]) for k in [
        "dance_keys", "event_families", "music_alignment_labels", "motion_stage_roles", "locomotion_labels", "support_labels"
    ] if k in data}

    dance = data["dance_keys"].astype(str)
    n = len(dance)

    def ensure(name, default):
        if name not in data or len(data[name]) != n:
            data[name] = np.asarray([default] * n, dtype=object)
        return data[name].astype(object)

    event_family = ensure("event_families", "pose_motif")
    music_align = ensure("music_alignment_labels", "pose_hold")
    stage_role = ensure("motion_stage_roles", "anchor_or_resolution")
    locomotion = ensure("locomotion_labels", "in_place_pose")
    support = ensure("support_labels", "stable_support")
    cultural = ensure("cultural_motifs", "dunhuang_pose")
    prop = ensure("prop_proxy_labels", "none")

    for i, key in enumerate(dance):
        k = str(key)
        if k == "thirty_six_postures":
            event_family[i] = "pose_motif"
            music_align[i] = "pose_hold"
            stage_role[i] = "anchor_or_resolution"
            locomotion[i] = "in_place_pose"
            support[i] = "stable_support"
            cultural[i] = "jiyuetian_pose"
            prop[i] = "none"
        elif k == "revelation_meditation":
            event_family[i] = "calm_flow"
            music_align[i] = "calm_meditative"
            stage_role[i] = "anchor_or_resolution"
            locomotion[i] = "in_place_pose"
            support[i] = "stable_support"
            cultural[i] = "buddhist_meditation"
            prop[i] = "none"
        elif k == "lotus_steps":
            event_family[i] = "footwork_flow"
            music_align[i] = "footwork_flow"
            stage_role[i] = "build_up"
            locomotion[i] = "traveling_steps"
            support[i] = "stable_support"
            cultural[i] = "lotus_step"
            prop[i] = "none"
        elif k == "pipa_behind_back":
            event_family[i] = "instrument_motif"
            music_align[i] = "instrument_phrase"
            stage_role[i] = "motif_recall"
            locomotion[i] = "upper_body_phrase"
            support[i] = "stable_support"
            cultural[i] = "pipa_instrument_pose"
            prop[i] = "pipa_proxy"
        elif k in {"sogdian_whirl", "ribbon_flow"}:
            event_family[i] = "turning_flow"
            music_align[i] = "turning_climax"
            stage_role[i] = "climax"
            locomotion[i] = "turning_travel"
            support[i] = "low_contact_flight_like"
            cultural[i] = "sogdian_whirl"
            prop[i] = "ribbon_sash_proxy"
        elif k == "lei_gong_drum":
            event_family[i] = "percussive_accent"
            music_align[i] = "percussive_accent"
            stage_role[i] = "climax"
            locomotion[i] = "traveling_steps"
            support[i] = "low_contact_flight_like"
            cultural[i] = "thunder_drum"
            prop[i] = "drum_proxy"

    data["event_families"] = event_family.astype(object)
    data["music_alignment_labels"] = music_align.astype(object)
    data["motion_stage_roles"] = stage_role.astype(object)
    data["locomotion_labels"] = locomotion.astype(object)
    data["support_labels"] = support.astype(object)
    data["cultural_motifs"] = cultural.astype(object)
    data["prop_proxy_labels"] = prop.astype(object)

    np.savez(db_path, **data)
    print("[OK] relabeled", db_path)

    db2 = np.load(db_path, allow_pickle=True)
    after = {k: counts(db2[k]) for k in [
        "dance_keys", "event_families", "music_alignment_labels", "motion_stage_roles", "locomotion_labels", "support_labels", "cultural_motifs", "prop_proxy_labels"
    ] if k in db2.files}
    report = {"db": str(db_path), "backup": str(bak), "before": before, "after": after}
    print(json.dumps(report["after"], indent=2, ensure_ascii=False))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
