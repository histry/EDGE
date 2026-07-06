#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create official source-level train/val/test splits for Chang-E/change Event-RAG.

This script is designed for the V46 MotionRAG-Diff pipeline.  It does NOT build
motion events by itself; instead it writes split-specific manifests that are
compatible with tools/v46_motionrag_diff.py build-db.

Scientific contract
-------------------
1) split is performed before event/window slicing;
2) all rows from the same original source_bvh/source_id stay in the same split;
3) train/val/test manifests are written separately;
4) train_db is the only DB used for retrieval/training in official experiments;
5) all-change manifest is optional and marked as qualitative_demo/upper_bound.

Typical usage
-------------
python tools/v46_make_change_splits.py \
  --motion_dir change \
  --manifest change/manifest.csv \
  --out_dir change/splits_official \
  --train_ratio 0.70 --val_ratio 0.15 --test_ratio 0.15 --seed 42

If no manifest is supplied or found, the script scans change/**/*.bvh, .npy,
.npz, .pkl, .pickle and creates one source record per file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MOTION_EXTS = (".bvh", ".npy", ".npz", ".pkl", ".pickle")

CATEGORY_PROFILES: Dict[str, Dict[str, Any]] = {
    "flying_apsaras": {
        "aliases": {"flying", "apsaras", "flying_apsara", "flying_apsaras", "feitian", "fei_tian", "sky_dance"},
        "display": "Flying Apsaras",
        "semantic_role": "aerial_graceful_flow",
        "energy_label": "moderate",
        "rhythm_label": "lyrical",
        "body_focus_label": "upper_body",
        "spatial_label": "aerial_leaning",
        "music_alignment_label": "lyrical_flow",
        "music_alignment_tags": ["lyrical_flow", "turning_climax", "calm_meditative"],
        "preferred_dance_keys": ["flying_apsaras", "sogdian_whirl", "lotus_steps"],
        "cultural_motif": "flying_apsara",
        "prop_proxy_label": "sash_ribbon_proxy",
        "locomotion_label": "floating_leaning",
        "support_label": "low_contact_flight_like",
        "event_family": "aerial_curve",
        "motion_stage_role": "opening_or_climax",
        "natural_duration_range_sec": [2.0, 5.5],
        "energy": 0.52, "onset": 0.28, "travel": 0.32, "turn": 0.38, "lower": 0.38, "upper": 0.72,
        "floorwork": 0.10, "jump": 0.35, "spin": 0.35, "pose_hold": 0.25, "instrument": 0.0, "prop": 0.85,
    },
    "lotus_steps": {
        "aliases": {"lotus", "lotussteps", "lotus_step", "lotus_steps"},
        "display": "Lotus Steps",
        "semantic_role": "flowing_footwork",
        "energy_label": "moderate",
        "rhythm_label": "lyrical",
        "body_focus_label": "lower_body",
        "spatial_label": "traveling",
        "music_alignment_label": "footwork_flow",
        "music_alignment_tags": ["footwork_flow", "lyrical_flow", "calm_meditative"],
        "preferred_dance_keys": ["lotus_steps", "flying_apsaras", "sogdian_whirl"],
        "cultural_motif": "lotus_step",
        "prop_proxy_label": "none",
        "locomotion_label": "traveling_steps",
        "support_label": "alternating_foot_support",
        "event_family": "footwork_flow",
        "motion_stage_role": "development",
        "natural_duration_range_sec": [1.5, 4.0],
        "energy": 0.48, "onset": 0.35, "travel": 0.62, "turn": 0.20, "lower": 0.78, "upper": 0.38,
        "floorwork": 0.05, "jump": 0.12, "spin": 0.10, "pose_hold": 0.20, "instrument": 0.0, "prop": 0.0,
    },
    "thirty_six_postures": {
        "aliases": {"36pose", "36posture", "36postures", "thirtysix", "thirty_six", "thirty_six_postures", "jiyuetian"},
        "display": "Ji Yue Tian Thirty-Six Postures",
        "semantic_role": "iconic_pose_sequence",
        "energy_label": "moderate",
        "rhythm_label": "sustained",
        "body_focus_label": "pose",
        "spatial_label": "in_place",
        "music_alignment_label": "pose_hold",
        "music_alignment_tags": ["pose_hold", "calm_meditative", "lyrical_flow"],
        "preferred_dance_keys": ["thirty_six_postures", "revelation_meditation", "lotus_steps"],
        "cultural_motif": "jiyuetian_pose",
        "prop_proxy_label": "none",
        "locomotion_label": "in_place_pose",
        "support_label": "static_or_low_motion_support",
        "event_family": "pose_motif",
        "motion_stage_role": "anchor_or_resolution",
        "natural_duration_range_sec": [1.2, 3.8],
        "energy": 0.36, "onset": 0.18, "travel": 0.12, "turn": 0.12, "lower": 0.28, "upper": 0.42,
        "floorwork": 0.18, "jump": 0.02, "spin": 0.05, "pose_hold": 0.90, "instrument": 0.0, "prop": 0.0,
    },
    "revelation_meditation": {
        "aliases": {"meditation", "mediation", "revelation", "revelation_meditation", "revelation_mediation"},
        "display": "Revelation Meditation",
        "semantic_role": "calm_meditative_flow",
        "energy_label": "calm",
        "rhythm_label": "sustained",
        "body_focus_label": "full_body",
        "spatial_label": "in_place",
        "music_alignment_label": "calm_meditative",
        "music_alignment_tags": ["calm_meditative", "pose_hold", "lyrical_flow"],
        "preferred_dance_keys": ["revelation_meditation", "thirty_six_postures", "flying_apsaras"],
        "cultural_motif": "buddhist_meditation",
        "prop_proxy_label": "none",
        "locomotion_label": "slow_weight_shift",
        "support_label": "stable_support",
        "event_family": "calm_flow",
        "motion_stage_role": "intro_or_resolution",
        "natural_duration_range_sec": [2.0, 6.0],
        "energy": 0.20, "onset": 0.08, "travel": 0.10, "turn": 0.08, "lower": 0.20, "upper": 0.36,
        "floorwork": 0.38, "jump": 0.0, "spin": 0.03, "pose_hold": 0.78, "instrument": 0.0, "prop": 0.0,
    },
    "sogdian_whirl": {
        "aliases": {"ribbon", "ribbon_flow", "sash", "silk", "whirl", "sogdian", "sogdian_whirl", "turn", "turning"},
        "display": "Sogdian Whirl / Ribbon Flow",
        "semantic_role": "flowing_turning_motif",
        "energy_label": "high",
        "rhythm_label": "lyrical",
        "body_focus_label": "turning_flow",
        "spatial_label": "turning",
        "music_alignment_label": "turning_climax",
        "music_alignment_tags": ["turning_climax", "lyrical_flow", "footwork_flow"],
        "preferred_dance_keys": ["sogdian_whirl", "flying_apsaras", "lotus_steps"],
        "cultural_motif": "sogdian_whirl",
        "prop_proxy_label": "ribbon_sash_proxy",
        "locomotion_label": "turning_travel",
        "support_label": "alternating_or_pivot_support",
        "event_family": "turning_flow",
        "motion_stage_role": "climax",
        "natural_duration_range_sec": [1.6, 4.5],
        "energy": 0.72, "onset": 0.40, "travel": 0.50, "turn": 0.90, "lower": 0.68, "upper": 0.65,
        "floorwork": 0.02, "jump": 0.20, "spin": 0.95, "pose_hold": 0.15, "instrument": 0.0, "prop": 0.75,
    },
    "pipa_behind_back": {
        "aliases": {"pipa", "pipa1", "pipa2", "playing_pipa", "playing_the_pipa", "pipa_behind_back"},
        "display": "Playing the Pipa Behind the Back",
        "semantic_role": "instrument_upper_body_motif",
        "energy_label": "moderate",
        "rhythm_label": "accented",
        "body_focus_label": "upper_body",
        "spatial_label": "in_place",
        "music_alignment_label": "instrument_phrase",
        "music_alignment_tags": ["instrument_phrase", "lyrical_flow", "percussive_accent"],
        "preferred_dance_keys": ["pipa_behind_back", "sogdian_whirl", "lei_gong_drum"],
        "cultural_motif": "pipa_instrument_pose",
        "prop_proxy_label": "pipa_proxy",
        "locomotion_label": "upper_body_phrase",
        "support_label": "stable_support",
        "event_family": "instrument_motif",
        "motion_stage_role": "motif_recall",
        "natural_duration_range_sec": [1.6, 4.5],
        "energy": 0.46, "onset": 0.42, "travel": 0.16, "turn": 0.20, "lower": 0.30, "upper": 0.82,
        "floorwork": 0.06, "jump": 0.05, "spin": 0.10, "pose_hold": 0.45, "instrument": 1.0, "prop": 0.70,
    },
    "lei_gong_drum": {
        "aliases": {"drum", "lei_gong", "leigong", "lei_gong_drum"},
        "display": "Lei Gong Drum",
        "semantic_role": "percussive_high_energy",
        "energy_label": "percussive",
        "rhythm_label": "percussive",
        "body_focus_label": "full_body",
        "spatial_label": "traveling",
        "music_alignment_label": "percussive_accent",
        "music_alignment_tags": ["percussive_accent", "turning_climax", "footwork_flow"],
        "preferred_dance_keys": ["lei_gong_drum", "pipa_behind_back", "sogdian_whirl"],
        "cultural_motif": "thunder_drum",
        "prop_proxy_label": "drum_proxy",
        "locomotion_label": "accented_travel",
        "support_label": "strong_foot_contact",
        "event_family": "percussive_accent",
        "motion_stage_role": "accent_or_climax",
        "natural_duration_range_sec": [1.2, 3.5],
        "energy": 0.82, "onset": 0.88, "travel": 0.52, "turn": 0.35, "lower": 0.75, "upper": 0.76,
        "floorwork": 0.04, "jump": 0.32, "spin": 0.20, "pose_hold": 0.10, "instrument": 0.65, "prop": 0.55,
    },
    "unknown": {
        "aliases": set(),
        "display": "Unknown Chang-E Motion",
        "semantic_role": "unknown_motion",
        "energy_label": "moderate",
        "rhythm_label": "lyrical",
        "body_focus_label": "full_body",
        "spatial_label": "in_place",
        "music_alignment_label": "lyrical_flow",
        "music_alignment_tags": ["lyrical_flow"],
        "preferred_dance_keys": ["lotus_steps", "thirty_six_postures"],
        "cultural_motif": "unknown",
        "prop_proxy_label": "unknown",
        "locomotion_label": "unknown",
        "support_label": "unknown",
        "event_family": "unknown",
        "motion_stage_role": "development",
        "natural_duration_range_sec": [1.5, 4.0],
        "energy": 0.45, "onset": 0.30, "travel": 0.30, "turn": 0.20, "lower": 0.45, "upper": 0.45,
        "floorwork": 0.0, "jump": 0.0, "spin": 0.0, "pose_hold": 0.25, "instrument": 0.0, "prop": 0.0,
    },
}

MUSIC_ALIGNMENT_LABELS = [
    "calm_meditative", "lyrical_flow", "pose_hold", "instrument_phrase",
    "percussive_accent", "turning_climax", "footwork_flow", "aerial_curve",
]


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def clean_text(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip().strip('"').strip("'")
    if not s or s.lower() in {"none", "null", "nan"}:
        return ""
    return s


def clean_stem(path_or_text: object) -> str:
    s = clean_text(path_or_text)
    if not s:
        return "unknown"
    return Path(s).stem.strip().lower().replace("-", "_").replace(" ", "_")


def canonicalize_key(key: object) -> str:
    s = clean_stem(key)
    aliases = {
        "mediation": "revelation_meditation",
        "female_mediation": "revelation_meditation",
        "male_mediation": "revelation_meditation",
        "meditation": "revelation_meditation",
        "revelation_mediation": "revelation_meditation",
        "36pose": "thirty_six_postures",
        "36posture": "thirty_six_postures",
        "36postures": "thirty_six_postures",
        "lotus": "lotus_steps",
        "pipa": "pipa_behind_back",
        "drum": "lei_gong_drum",
        "ribbon": "sogdian_whirl",
        "ribbon_flow": "sogdian_whirl",
        "sogdian": "sogdian_whirl",
        "whirl": "sogdian_whirl",
        "flying": "flying_apsaras",
        "apsaras": "flying_apsaras",
        "feitian": "flying_apsaras",
    }
    if s in aliases:
        return aliases[s]
    for k, prof in CATEGORY_PROFILES.items():
        if s == k or s in prof.get("aliases", set()):
            return k
    return s if s in CATEGORY_PROFILES else "unknown"


def parse_change_semantics(path_or_label: object, row: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    row = row or {}
    explicit_key = clean_text(row.get("dance_key") or row.get("parent_label") or row.get("category"))
    stem = clean_stem(path_or_label)
    tokens = [t for t in stem.split("_") if t]
    gender = clean_text(row.get("gender")) or "unknown"
    rest = tokens[:]
    if gender == "unknown" and rest and rest[0] in {"male", "female"}:
        gender = rest[0]
        rest = rest[1:]

    take_id = clean_text(row.get("take_id") or row.get("source_take"))
    if not take_id and rest and rest[-1].isdigit():
        take_id = rest[-1]
        rest = rest[:-1]

    base = explicit_key or "_".join(rest) or stem
    category_key = canonicalize_key(base)
    if category_key == "unknown":
        for key, prof in CATEGORY_PROFILES.items():
            aliases = set(prof.get("aliases", set()))
            if base in aliases or any(tok in aliases for tok in rest):
                category_key = key
                break
    prof = CATEGORY_PROFILES.get(category_key, CATEGORY_PROFILES["unknown"])
    label = category_key if not take_id else f"{category_key}_take{take_id}"
    return {
        "raw_stem": stem,
        "gender": gender,
        "take_id": take_id if take_id else "-1",
        "dance_key": category_key,
        "dance_category": prof["display"],
        "semantic_role": prof["semantic_role"],
        "energy_label": prof["energy_label"],
        "rhythm_label": prof["rhythm_label"],
        "body_focus_label": prof["body_focus_label"],
        "spatial_label": prof["spatial_label"],
        "music_alignment_label": prof["music_alignment_label"],
        "music_alignment_tags": ";".join(prof["music_alignment_tags"]),
        "preferred_dance_keys": ";".join(prof["preferred_dance_keys"]),
        "cultural_motif": prof.get("cultural_motif", category_key),
        "prop_proxy_label": prof.get("prop_proxy_label", "none"),
        "locomotion_label": prof.get("locomotion_label", "unknown"),
        "support_label": prof.get("support_label", "unknown"),
        "event_family": prof.get("event_family", category_key),
        "motion_stage_role": prof.get("motion_stage_role", "development"),
        "semantic_numeric": ";".join(str(float(prof.get(k, 0.0))) for k in ["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"]),
        "natural_duration_min_sec": str(float(prof.get("natural_duration_range_sec", [1.5, 4.0])[0])),
        "natural_duration_max_sec": str(float(prof.get("natural_duration_range_sec", [1.5, 4.0])[1])),
        "label": clean_text(row.get("label")) or label,
        "parent_label": clean_text(row.get("parent_label")) or category_key,
    }


def strip_fragment_suffix(stem: str) -> str:
    s = clean_stem(stem)
    patterns = [
        r"(_|-)?(clip|frag|fragment|event|window|win|seg|segment|slice)(_)?\d+$",
        r"(_|-)?f\d+(_)?t\d+$",
        r"(_|-)?start\d+(_)?end\d+$",
        r"(_|-)?\d{5,}$",
    ]
    for pat in patterns:
        ns = re.sub(pat, "", s)
        if ns and ns != s:
            return ns
    return s


def source_id_from_row(row: Dict[str, object], fallback_path: str = "") -> str:
    # Strong priority: original BVH/source file.  Fragment paths are used only
    # when no parent source is available.
    for key in [
        "source_bvh", "source_file", "source_path", "bvh", "bvh_file",
        "original_filename", "orig_filename", "video_name", "video_id",
    ]:
        val = clean_text(row.get(key))
        if val:
            return clean_stem(val)
    for key in ["fragment_file", "path", "file", "motion_file"]:
        val = clean_text(row.get(key))
        if val:
            return strip_fragment_suffix(Path(val).stem)
    return strip_fragment_suffix(Path(fallback_path).stem if fallback_path else "unknown")


def resolve_path(value: object, roots: Sequence[Path]) -> str:
    text = clean_text(value)
    if not text:
        return ""
    p = Path(text)
    if p.is_absolute() and p.exists():
        return str(p)
    if p.is_absolute():
        return str(p)
    candidates: List[Path] = []
    for r in roots:
        candidates.append(r / p)
        candidates.append(r / p.name)
    for c in candidates:
        if c.exists():
            return str(c)
    return text


def read_input_manifest(path: Path, motion_dir: Path) -> List[Dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for idx, row in enumerate(reader):
            rr = {str(k): clean_text(v) for k, v in row.items() if k is not None}
            rr["raw_manifest_index"] = str(idx)
            rr["raw_manifest_path"] = str(path)
            rows.append(rr)
    return normalize_records(rows, motion_dir=motion_dir, manifest_path=path)


def scan_motion_dir(motion_dir: Path) -> List[Dict[str, str]]:
    files: List[Path] = []
    for ext in MOTION_EXTS:
        files.extend(motion_dir.rglob(f"*{ext}"))
    files = sorted(set(p for p in files if p.is_file()))
    rows: List[Dict[str, str]] = []
    for idx, p in enumerate(files):
        # Skip previously generated split manifests and generated event DB dirs.
        if "splits" in p.parts or "events" in p.parts:
            continue
        rows.append({
            "source_bvh": str(p),
            "fragment_file": "",
            "fragment_index": "0",
            "raw_manifest_index": str(idx),
            "raw_manifest_path": "",
        })
    return normalize_records(rows, motion_dir=motion_dir, manifest_path=None)


def normalize_records(rows: List[Dict[str, str]], motion_dir: Path, manifest_path: Optional[Path]) -> List[Dict[str, str]]:
    roots = [motion_dir, motion_dir.parent, Path.cwd()]
    if manifest_path is not None:
        roots.insert(0, manifest_path.parent)
    out: List[Dict[str, str]] = []
    for idx, row in enumerate(rows):
        fragment = clean_text(row.get("fragment_file") or row.get("path") or row.get("file") or row.get("motion_file"))
        source = clean_text(row.get("source_bvh") or row.get("source_file") or row.get("bvh") or row.get("bvh_file"))
        load_path = fragment or source
        if not load_path:
            continue
        fragment_resolved = resolve_path(fragment, roots) if fragment else ""
        source_resolved = resolve_path(source, roots) if source else (fragment_resolved or resolve_path(load_path, roots))
        load_resolved = fragment_resolved or source_resolved or resolve_path(load_path, roots)
        sid = source_id_from_row(row, fallback_path=source_resolved or load_resolved)
        sem_input = source_resolved or load_resolved or sid
        sem = parse_change_semantics(sem_input, row)
        uid = clean_stem(source_resolved or load_resolved or sid)
        rec: Dict[str, str] = {}
        # preserve all raw columns but normalized columns take precedence below.
        for k, v in row.items():
            rec[k] = clean_text(v)
        rec.update({
            "source_bvh": source_resolved or load_resolved,
            "fragment_file": fragment_resolved,
            "fragment_index": clean_text(row.get("fragment_index")) or str(idx),
            "source_id": sid,
            "source_uid": uid,
            "source_group": sid,
            "split": "",
            "label": sem["label"],
            "parent_label": sem["parent_label"],
            "gender": sem["gender"],
            "take_id": str(sem["take_id"]),
            "dance_key": sem["dance_key"],
            "dance_category": sem["dance_category"],
            "semantic_role": sem["semantic_role"],
            "energy_label": sem["energy_label"],
            "rhythm_label": sem["rhythm_label"],
            "body_focus_label": sem["body_focus_label"],
            "spatial_label": sem["spatial_label"],
            "music_alignment_label": sem["music_alignment_label"],
            "music_alignment_tags": sem["music_alignment_tags"],
            "preferred_dance_keys": sem["preferred_dance_keys"],
            "cultural_motif": sem.get("cultural_motif", ""),
            "prop_proxy_label": sem.get("prop_proxy_label", ""),
            "locomotion_label": sem.get("locomotion_label", ""),
            "support_label": sem.get("support_label", ""),
            "event_family": sem.get("event_family", ""),
            "motion_stage_role": sem.get("motion_stage_role", ""),
            "semantic_numeric": sem.get("semantic_numeric", ""),
            "natural_duration_min_sec": sem.get("natural_duration_min_sec", ""),
            "natural_duration_max_sec": sem.get("natural_duration_max_sec", ""),
            "raw_stem": sem["raw_stem"],
            "raw_manifest_index": clean_text(row.get("raw_manifest_index")) or str(idx),
            "raw_manifest_path": clean_text(row.get("raw_manifest_path")),
        })
        # Keep timing columns if present; v46_motionrag_diff.py will use them.
        for k in ["start_frame", "end_frame", "start_time", "end_time", "duration_sec", "bvh_fps"]:
            rec[k] = clean_text(row.get(k))
        out.append(rec)
    return out


def mode(values: Iterable[str], default: str = "unknown") -> str:
    vals = [v for v in values if v]
    if not vals:
        return default
    return Counter(vals).most_common(1)[0][0]


def group_records(records: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for r in records:
        sid = r.get("source_id") or r.get("source_group") or stable_hash(r.get("source_bvh", ""))
        if sid not in groups:
            groups[sid] = {"source_id": sid, "records": []}
        groups[sid]["records"].append(r)
    for sid, g in groups.items():
        recs = g["records"]
        g["dance_key"] = mode(r.get("dance_key", "unknown") for r in recs)
        g["gender"] = mode(r.get("gender", "unknown") for r in recs)
        g["source_uid"] = mode(r.get("source_uid", sid) for r in recs)
        g["num_records"] = len(recs)
        g["stratum"] = f"{g['dance_key']}|{g['gender']}"
    return groups


def interleave_by_stratum(groups: Dict[str, Dict[str, Any]], seed: int) -> List[str]:
    rng = random.Random(seed)
    buckets: Dict[str, List[str]] = defaultdict(list)
    for sid, g in groups.items():
        buckets[str(g["stratum"])].append(sid)
    for sids in buckets.values():
        rng.shuffle(sids)
    strata = sorted(buckets.keys())
    rng.shuffle(strata)
    ordered: List[str] = []
    while any(buckets[s] for s in strata):
        for s in list(strata):
            if buckets[s]:
                ordered.append(buckets[s].pop(0))
    return ordered


def target_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, int]:
    """Compute robust non-empty source-level split counts.

    For official experiments, when n >= 3 all three splits must be non-empty.
    This implementation avoids independent rounding failures where test can
    become negative or zero after train/val rounding.
    """
    if n <= 0:
        return {"train": 0, "val": 0, "test": 0}
    ratios = np.asarray([max(0.0, train_ratio), max(0.0, val_ratio), max(0.0, test_ratio)], dtype=np.float64)
    if float(ratios.sum()) <= 0.0:
        ratios = np.asarray([0.70, 0.15, 0.15], dtype=np.float64)
    ratios = ratios / float(ratios.sum())

    if n == 1:
        return {"train": 1, "val": 0, "test": 0}
    if n == 2:
        # Keep one held-out split for leakage checks; val can be empty for tiny smoke tests.
        return {"train": 1, "val": 0, "test": 1}

    # n >= 3: reserve one source per split, then allocate the remaining sources
    # using largest remainder. This guarantees train/val/test are all non-empty.
    base = np.ones(3, dtype=np.int64)
    remaining = int(n - 3)
    if remaining > 0:
        raw = ratios * remaining
        add = np.floor(raw).astype(np.int64)
        leftover = int(remaining - int(add.sum()))
        order = np.argsort(-(raw - add))
        for k in order[:leftover]:
            add[int(k)] += 1
        counts = base + add
    else:
        counts = base

    # Final exact-sum repair; should rarely trigger, but keep it deterministic.
    while int(counts.sum()) < n:
        counts[int(np.argmax(ratios))] += 1
    while int(counts.sum()) > n:
        candidates = [i for i in range(3) if counts[i] > 1]
        j = max(candidates, key=lambda i: counts[i])
        counts[j] -= 1

    return {"train": int(counts[0]), "val": int(counts[1]), "test": int(counts[2])}


def pick_evenly(indices: List[int], k: int) -> List[int]:
    if k <= 0 or not indices:
        return []
    if k >= len(indices):
        return indices[:]
    if k == 1:
        return [indices[len(indices) // 2]]
    positions = [int(round(x)) for x in [i * (len(indices) - 1) / (k - 1) for i in range(k)]]
    picked: List[int] = []
    used = set()
    for pos in positions:
        # ensure uniqueness under rounding
        p = max(0, min(len(indices) - 1, pos))
        while p in used and p + 1 < len(indices):
            p += 1
        while p in used and p - 1 >= 0:
            p -= 1
        if p not in used:
            used.add(p)
            picked.append(indices[p])
    return picked[:k]


def split_groups(groups: Dict[str, Dict[str, Any]], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Dict[str, str]:
    ordered = interleave_by_stratum(groups, seed)
    n = len(ordered)
    counts = target_counts(n, train_ratio, val_ratio, test_ratio)
    all_idx = list(range(n))
    test_idx = set(pick_evenly(all_idx, counts["test"]))
    remain_idx = [i for i in all_idx if i not in test_idx]
    val_idx = set(pick_evenly(remain_idx, counts["val"]))
    assignment: Dict[str, str] = {}
    for i, sid in enumerate(ordered):
        if i in test_idx:
            assignment[sid] = "test"
        elif i in val_idx:
            assignment[sid] = "val"
        else:
            assignment[sid] = "train"
    return assignment


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def counts_by(rows: List[Dict[str, str]], key: str) -> List[Tuple[str, int]]:
    return Counter(str(r.get(key, "unknown")) for r in rows).most_common()


def make_report(records: List[Dict[str, str]], groups: Dict[str, Dict[str, Any]], assignment: Dict[str, str], args: argparse.Namespace) -> Dict[str, Any]:
    split_rows = {s: [r for r in records if assignment.get(r["source_id"]) == s] for s in ["train", "val", "test"]}
    split_sources = {s: sorted({r["source_id"] for r in split_rows[s]}) for s in ["train", "val", "test"]}
    leakage = {
        "train_val": sorted(set(split_sources["train"]) & set(split_sources["val"])),
        "train_test": sorted(set(split_sources["train"]) & set(split_sources["test"])),
        "val_test": sorted(set(split_sources["val"]) & set(split_sources["test"])),
    }
    return {
        "version": "v46_23_official_chang_e_source_split",
        "motion_dir": str(Path(args.motion_dir).resolve()),
        "input_manifest": str(Path(args.manifest).resolve()) if args.manifest else "",
        "out_dir": str(Path(args.out_dir).resolve()),
        "seed": int(args.seed),
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "split_contract": {
            "split_level": "original_source_id_before_event_slicing",
            "rag_memory_policy": "official_generation_uses_train_db_only",
            "val_test_usage": "evaluation_only_not_retrieval_not_training",
            "all_change_db_usage": "qualitative_demo_or_upper_bound_only",
            "internal_dunhuang_usage": "supplement_cross_domain_validation_only",
        },
        "num_records": len(records),
        "num_sources": len(groups),
        "num_sources_by_split": {s: len(split_sources[s]) for s in split_sources},
        "num_records_by_split": {s: len(split_rows[s]) for s in split_rows},
        "source_overlap_leakage": leakage,
        "dance_key_counts_all": counts_by(records, "dance_key"),
        "gender_counts_all": counts_by(records, "gender"),
        "per_split": {
            s: {
                "sources": split_sources[s],
                "dance_key_counts": counts_by(split_rows[s], "dance_key"),
                "gender_counts": counts_by(split_rows[s], "gender"),
                "music_alignment_counts": counts_by(split_rows[s], "music_alignment_label"),
            }
            for s in ["train", "val", "test"]
        },
        "source_table": [
            {
                "source_id": sid,
                "split": assignment[sid],
                "dance_key": str(groups[sid].get("dance_key", "unknown")),
                "gender": str(groups[sid].get("gender", "unknown")),
                "num_records": int(groups[sid].get("num_records", 0)),
                "source_uid": str(groups[sid].get("source_uid", sid)),
            }
            for sid in sorted(groups)
        ],
    }


def write_ontology(out_dir: Path) -> None:
    obj = {
        "version": "v46_23_chang_e_enriched_label_ontology",
        "music_alignment_labels": MUSIC_ALIGNMENT_LABELS,
        "action_categories": {
            k: {kk: vv for kk, vv in v.items() if kk != "aliases"}
            for k, v in CATEGORY_PROFILES.items()
            if k != "unknown"
        },
        "annotation_columns": [
            "source_id", "source_bvh", "fragment_file", "split", "dance_key", "gender",
            "semantic_role", "energy_label", "rhythm_label", "body_focus_label",
            "spatial_label", "music_alignment_label", "music_alignment_tags",
            "preferred_dance_keys", "cultural_motif", "prop_proxy_label",
            "locomotion_label", "support_label", "event_family", "motion_stage_role",
            "semantic_numeric", "natural_duration_min_sec", "natural_duration_max_sec",
        ],
    }
    (out_dir / "chang_e_label_ontology.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_dir", default="change")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--out_dir", default="change/splits_official")
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict", action="store_true", default=True)
    ap.add_argument("--allow_small", action="store_true", help="Allow fewer than 3 source groups; for smoke tests only.")
    args = ap.parse_args()

    motion_dir = Path(args.motion_dir)
    if not motion_dir.exists():
        raise FileNotFoundError(f"motion_dir not found: {motion_dir}")

    manifest = Path(args.manifest) if args.manifest else None
    if manifest and not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if manifest is None:
        for cand in [motion_dir / "manifest.csv", motion_dir / "manifest.tsv", Path("manifest.csv")]:
            if cand.exists():
                manifest = cand
                break

    if manifest is not None:
        records = read_input_manifest(manifest, motion_dir)
    else:
        records = scan_motion_dir(motion_dir)

    if not records:
        raise RuntimeError(f"No Chang-E motion records found under {motion_dir}")

    groups = group_records(records)
    if len(groups) < 3 and args.strict and not args.allow_small:
        raise RuntimeError(
            f"Official source-level split requires at least 3 source groups. Found {len(groups)}. "
            "Use --allow_small only for smoke tests."
        )

    assignment = split_groups(groups, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    for r in records:
        r["split"] = assignment[r["source_id"]]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preferred_fields = [
        "source_bvh", "fragment_file", "fragment_index", "label", "parent_label",
        "start_frame", "end_frame", "start_time", "end_time", "duration_sec", "bvh_fps",
        "split", "source_id", "source_uid", "source_group", "gender", "take_id",
        "dance_key", "dance_category", "semantic_role", "energy_label", "rhythm_label",
        "body_focus_label", "spatial_label", "music_alignment_label", "music_alignment_tags",
        "preferred_dance_keys", "cultural_motif", "prop_proxy_label", "locomotion_label",
        "support_label", "event_family", "motion_stage_role", "semantic_numeric",
        "natural_duration_min_sec", "natural_duration_max_sec",
        "raw_stem", "raw_manifest_index", "raw_manifest_path",
    ]
    extra_fields = sorted(set().union(*(r.keys() for r in records)) - set(preferred_fields))
    fieldnames = preferred_fields + extra_fields

    by_split = {s: [r for r in records if r["split"] == s] for s in ["train", "val", "test"]}
    for s in ["train", "val", "test"]:
        write_csv(out_dir / f"{s}_manifest.csv", by_split[s], fieldnames)
    write_csv(out_dir / "all_manifest.csv", records, fieldnames)

    report = make_report(records, groups, assignment, args)
    if any(report["source_overlap_leakage"].values()):
        raise RuntimeError(f"Source leakage detected: {report['source_overlap_leakage']}")
    if args.strict and len(groups) >= 3:
        empty = [s for s, rows in by_split.items() if not rows]
        if empty:
            raise RuntimeError(f"Empty official split(s): {empty}. Adjust ratios or use more sources.")
    (out_dir / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_ontology(out_dir)

    print(json.dumps({
        "status": "ok",
        "out_dir": str(out_dir),
        "num_records": len(records),
        "num_sources": len(groups),
        "num_records_by_split": report["num_records_by_split"],
        "num_sources_by_split": report["num_sources_by_split"],
        "split_report": str(out_dir / "split_report.json"),
        "train_manifest": str(out_dir / "train_manifest.csv"),
        "val_manifest": str(out_dir / "val_manifest.csv"),
        "test_manifest": str(out_dir / "test_manifest.csv"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
