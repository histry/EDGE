#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Source-aware metadata and balancing utilities for Dunhuang Event-RAG.

The Dunhuang BVH corpus used by this project is motion-only and low-resource.
Each original sequence name follows:

    <dancer>_<repeat>_Take_<category>.bvh

For example ``dyl_002_Take_003.bvh`` means dancer ``dyl``, repeat ``002``,
and category ``Take_003``.  Older Event-RAG code preserved this information in
file names and event ids, but did not expose it as structured metadata.  This
module makes the source structure explicit so database construction and
retrieval can separate dancer style diversity from source/repeat/category bias.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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
    # Prefer already-normalized fields when present.
    dancer = str(item.get("dancer_id", "")).strip()
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


def source_quality_score(item: Mapping[str, Any]) -> float:
    """Quality-aware score used only for downsampling within a source group."""
    def get(key: str, default: float = 0.0) -> float:
        try:
            return float(item.get(key, default))
        except Exception:
            return float(default)

    style = max(
        get("v21_style_score"),
        get("dunhuang_style_score_v20f3"),
        get("dunhuang_style_score"),
        get("dunhuang_style_proxy"),
        get("visual_score"),
        get("quality_score"),
    )
    quality = max(get("v21_quality_score"), get("original_quality_score"), get("quality_score"))
    safety = max(get("v21_safety_score"), get("safety_score"), quality)
    length = get("event_length", get("length", 0.0))
    # Prefer usable snippets; extremely short events often become static glue.
    length_prior = float(np.clip((length - 18.0) / max(72.0 - 18.0, 1.0), 0.0, 1.0))
    event_type = str(item.get("event_type", "neutral_flow"))
    static_discount = 0.10 if event_type == "pose_hold" else 0.0
    return float(0.42 * style + 0.34 * quality + 0.18 * safety + 0.06 * length_prior - static_discount)


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


def _cap_group(
    rows: Sequence[Dict[str, Any]],
    key: str,
    cap: int,
) -> List[Dict[str, Any]]:
    if cap <= 0:
        return list(rows)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    out: List[Dict[str, Any]] = []
    for group_rows in grouped.values():
        group_rows.sort(key=source_quality_score, reverse=True)
        out.extend(group_rows[:cap])
    return out


def _cap_by_uniform_factor(
    rows: Sequence[Dict[str, Any]],
    key: str,
    factor: float,
) -> List[Dict[str, Any]]:
    if factor <= 0 or not rows:
        return list(rows)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(row)
    if not groups:
        return list(rows)
    target = int(math.ceil((len(rows) / max(len(groups), 1)) * float(factor)))
    target = max(1, target)
    out: List[Dict[str, Any]] = []
    for group_rows in groups.values():
        group_rows.sort(key=source_quality_score, reverse=True)
        out.extend(group_rows[:target])
    return out


def source_aware_select(
    items: Sequence[Mapping[str, Any]],
    *,
    cap_per_source_uid: int = 64,
    category_cap_factor: float = 1.35,
    repeat_cap_factor: float = 1.60,
    dancer_cap_factor: float = 1.50,
    max_events: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return a quality-weighted, source-aware subset.

    The selector never fabricates events.  Rare sources keep all valid snippets;
    over-represented sources/categories/repeats are downsampled by quality.  The
    output preserves the selected item dictionaries and appends source metadata.
    """
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


def identity_lists(items: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    enriched = [enrich_item_with_source_identity(item) for item in items]
    return {
        "dancer_id": [str(x["dancer_id"]) for x in enriched],
        "repeat_id": [str(x["repeat_id"]) for x in enriched],
        "category_id": [str(x["category_id"]) for x in enriched],
        "source_uid": [str(x["source_uid"]) for x in enriched],
        "dancer_category_group": [str(x["dancer_category_group"]) for x in enriched],
        "category_repeat_group": [str(x["category_repeat_group"]) for x in enriched],
    }
