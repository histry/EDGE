#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Source-aware metadata and balancing utilities for Dunhuang Event-RAG.

The Dunhuang BVH corpus is motion-only and low-resource.  File names encode the
source structure:

    <dancer>_<repeat>_Take_<category>.bvh

Example: ``dyl_002_Take_003.bvh`` means dancer ``dyl``, repeat ``002`` and
category ``Take_003``.  Treating these fields as plain text creates two
problems: over-represented dancers dominate the RAG bank, and repeated takes of
the same posture leak near-duplicates into train/validation/evaluation routes.

This module exposes the source structure as explicit metadata and provides
quality-aware balancing functions used by both database construction and
retrieval.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


SOURCE_RE = re.compile(
    r"(?P<dancer>[A-Za-z]+)_(?P<repeat>\d{3})_Take_(?P<category>\d{3})"
)


@dataclass(frozen=True)
class SourceIdentity:
    dancer_id: str
    repeat_id: str
    category_id: str
    source_uid: str
    dancer_category_group: str
    category_repeat_group: str
    parsed: bool = True


UNKNOWN_IDENTITY = SourceIdentity(
    dancer_id="unknown",
    repeat_id="unknown",
    category_id="Take_unknown",
    source_uid="unknown_unknown_Take_unknown",
    dancer_category_group="unknown_Take_unknown",
    category_repeat_group="Take_unknown_unknown",
    parsed=False,
)


def _source_text(item: Mapping[str, Any]) -> str:
    keys = (
        "source_uid",
        "event_id",
        "source_file",
        "original_filename",
        "source_filename",
        "pkl",
        "path",
        "event_path",
        "motion_path",
    )
    return " ".join(str(item.get(key, "")) for key in keys)


def parse_source_identity(item: Mapping[str, Any]) -> SourceIdentity:
    """Parse dancer/repeat/category metadata from an item or return unknown."""
    dancer = str(item.get("dancer_id", "")).strip().lower()
    repeat = str(item.get("repeat_id", "")).strip()
    category = str(item.get("category_id", "")).strip()
    if dancer and repeat and category:
        category = category if category.startswith("Take_") else f"Take_{category}"
        return SourceIdentity(
            dancer_id=dancer,
            repeat_id=repeat,
            category_id=category,
            source_uid=f"{dancer}_{repeat}_{category}",
            dancer_category_group=f"{dancer}_{category}",
            category_repeat_group=f"{category}_{repeat}",
            parsed=True,
        )

    match = SOURCE_RE.search(_source_text(item))
    if not match:
        return UNKNOWN_IDENTITY

    dancer = match.group("dancer").lower()
    repeat = match.group("repeat")
    category = f"Take_{match.group('category')}"
    return SourceIdentity(
        dancer_id=dancer,
        repeat_id=repeat,
        category_id=category,
        source_uid=f"{dancer}_{repeat}_{category}",
        dancer_category_group=f"{dancer}_{category}",
        category_repeat_group=f"{category}_{repeat}",
        parsed=True,
    )


def enrich_item_with_source_identity(item: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(item)
    ident = parse_source_identity(out)
    out["dancer_id"] = ident.dancer_id
    out["repeat_id"] = ident.repeat_id
    out["category_id"] = ident.category_id
    out["source_uid"] = ident.source_uid
    out["dancer_category_group"] = ident.dancer_category_group
    out["category_repeat_group"] = ident.category_repeat_group
    out["source_identity_parsed"] = bool(ident.parsed)
    return out


def _float_value(item: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    if key in item:
        try:
            return float(item[key])
        except (TypeError, ValueError):
            pass
    desc = item.get("descriptor", {})
    if isinstance(desc, Mapping) and key in desc:
        try:
            return float(desc[key])
        except (TypeError, ValueError):
            pass
    return float(default)


def source_quality_score(item: Mapping[str, Any]) -> float:
    """Quality-aware score used for downsampling within a source group."""
    style = max(
        _float_value(item, "v21_style_score"),
        _float_value(item, "dunhuang_style_score_v20f3"),
        _float_value(item, "dunhuang_style_score"),
        _float_value(item, "dunhuang_style_proxy"),
        _float_value(item, "visual_score"),
        _float_value(item, "quality_score"),
    )
    quality = max(
        _float_value(item, "v21_quality_score"),
        _float_value(item, "original_quality_score"),
        _float_value(item, "segment_quality"),
        _float_value(item, "quality_score"),
    )
    safety = max(_float_value(item, "v21_safety_score"), _float_value(item, "safety_score"), quality)
    density = _float_value(item, "mean_activity", _float_value(item, "motion_density", 0.0))
    boundary = _float_value(item, "boundary_quality", 0.5)
    hold = _float_value(item, "static_hold_score", 0.0)
    first_ratio = _float_value(item, "first1s_energy_ratio", 0.0)
    length = _float_value(item, "event_length", _float_value(item, "length", 0.0))
    length_prior = float(np.exp(-abs(length - 48.0) / 48.0))
    event_type = str(item.get("event_type", "neutral_flow"))
    static_discount = 0.18 if event_type == "pose_hold" else 0.0
    frontload_discount = 0.10 * max(0.0, first_ratio - 0.75)
    return float(
        0.30 * style
        + 0.28 * quality
        + 0.16 * safety
        + 0.10 * density
        + 0.08 * boundary
        + 0.08 * length_prior
        - 0.18 * hold
        - static_discount
        - frontload_discount
    )


def source_distribution(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    enriched = [enrich_item_with_source_identity(item) for item in items]
    source = Counter(str(x["source_uid"]) for x in enriched)
    dancer = Counter(str(x["dancer_id"]) for x in enriched)
    repeat = Counter(str(x["repeat_id"]) for x in enriched)
    category = Counter(str(x["category_id"]) for x in enriched)
    parsed = sum(1 for x in enriched if bool(x.get("source_identity_parsed", False)))
    return {
        "num_events": int(len(enriched)),
        "num_source_uid": int(len(source)),
        "parsed": int(parsed),
        "missing_parse": int(len(enriched) - parsed),
        "source_min": int(min(source.values())) if source else 0,
        "source_max": int(max(source.values())) if source else 0,
        "source_mean": float(sum(source.values()) / max(len(source), 1)),
        "by_dancer": dict(dancer),
        "by_repeat": dict(repeat),
        "by_category": dict(category),
        "top_source_uid": dict(source.most_common(20)),
        "bottom_source_uid": dict(source.most_common()[-20:]),
    }


def _cap_group(rows: Sequence[Dict[str, Any]], key: str, cap: int) -> list[Dict[str, Any]]:
    if cap <= 0:
        return list(rows)
    grouped: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    out: list[Dict[str, Any]] = []
    for group_rows in grouped.values():
        group_rows.sort(key=source_quality_score, reverse=True)
        out.extend(group_rows[:cap])
    return out


def _cap_by_uniform_factor(
    rows: Sequence[Dict[str, Any]],
    key: str,
    factor: float,
) -> list[Dict[str, Any]]:
    if factor <= 0 or not rows:
        return list(rows)
    grouped: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    target = int(math.ceil((len(rows) / max(len(grouped), 1)) * float(factor)))
    target = max(1, target)
    out: list[Dict[str, Any]] = []
    for group_rows in grouped.values():
        group_rows.sort(key=source_quality_score, reverse=True)
        out.extend(group_rows[:target])
    return out


def source_aware_select(
    items: Sequence[Mapping[str, Any]],
    *,
    cap_per_source_uid: int = 72,
    category_cap_factor: float = 1.35,
    repeat_cap_factor: float = 1.55,
    dancer_cap_factor: float = 1.45,
    max_events: int = 0,
) -> Tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Return a quality-weighted, source-aware subset."""
    before = [enrich_item_with_source_identity(item) for item in items]
    for idx, row in enumerate(before):
        row["_source_aware_original_index"] = int(idx)

    rows = _cap_group(before, "source_uid", int(cap_per_source_uid))
    rows = _cap_by_uniform_factor(rows, "category_id", float(category_cap_factor))
    rows = _cap_by_uniform_factor(rows, "repeat_id", float(repeat_cap_factor))
    rows = _cap_by_uniform_factor(rows, "dancer_id", float(dancer_cap_factor))

    if max_events and int(max_events) > 0 and len(rows) > int(max_events):
        rows.sort(key=source_quality_score, reverse=True)
        rows = rows[: int(max_events)]

    rows.sort(
        key=lambda x: (
            str(x.get("category_id", "")),
            str(x.get("dancer_id", "")),
            str(x.get("repeat_id", "")),
            int(x.get("_source_aware_original_index", 0)),
        )
    )
    report = {
        "enabled": True,
        "cap_per_source_uid": int(cap_per_source_uid),
        "category_cap_factor": float(category_cap_factor),
        "repeat_cap_factor": float(repeat_cap_factor),
        "dancer_cap_factor": float(dancer_cap_factor),
        "max_events": int(max_events),
        "before": source_distribution(before),
        "after": source_distribution(rows),
    }
    return rows, report


def identity_lists(items: Sequence[Mapping[str, Any]]) -> Dict[str, list[str]]:
    enriched = [enrich_item_with_source_identity(item) for item in items]
    return {
        "dancer_id": [str(x["dancer_id"]) for x in enriched],
        "repeat_id": [str(x["repeat_id"]) for x in enriched],
        "category_id": [str(x["category_id"]) for x in enriched],
        "source_uid": [str(x["source_uid"]) for x in enriched],
        "dancer_category_group": [str(x["dancer_category_group"]) for x in enriched],
        "category_repeat_group": [str(x["category_repeat_group"]) for x in enriched],
    }
