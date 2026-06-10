#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V30 whole-song scheduler overlay.

The proven music-boundary lock, V23 duration allocation and graph beam search
remain in the V26 scheduler.  V30 replaces:
  * raw 6D geometry with SO(3)-aware V29 geometry;
  * fixed random CLAP projection with learned Poincare music embeddings;
  * hierarchy node scoring with a blended hierarchy + cross-modal distance.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np

import tools.schedule_v26_whole_song as scheduler
import tools.v26_hierarchical_graph_scheduler as hierarchy_module
from tools.v29_motion_geometry import (
    apply_start_anchor_so3,
    dampen_event_edges_so3,
    endpoint_metrics_np,
    make_so3_transition,
)
from tools.v30_deep_music_features import phrase_geometric_matrix
from tools.v30_geometric_alignment import poincare_distance_pairwise


_original_build_slot_query = hierarchy_module.build_slot_query
_original_node_scores = hierarchy_module.hierarchical_node_scores
_original_sample_transition = scheduler.sample_transition_diffusion
_CURRENT_RULE_QUERIES = np.zeros((0, 12), dtype=np.float32)
_CURRENT_GEOMETRIC_QUERIES = np.zeros((0, 1), dtype=np.float32)


def _boundary_metrics(prev, nxt):
    return endpoint_metrics_np(prev, nxt, fps=30.0)


def _phrase_semantics(audio_path, phrases, *args, **kwargs):
    global _CURRENT_RULE_QUERIES, _CURRENT_GEOMETRIC_QUERIES
    semantic, meta = phrase_geometric_matrix(
        audio_path, phrases, *args, **kwargs
    )
    _CURRENT_RULE_QUERIES = np.stack([
        np.asarray(phrase.query, dtype=np.float32).reshape(-1)[:12]
        for phrase in phrases
    ]).astype(np.float32)
    _CURRENT_GEOMETRIC_QUERIES = np.asarray(semantic, dtype=np.float32)
    return semantic, meta


def _build_slot_query(
    phrase: Any,
    predicted_event: str,
    target_natural: float,
    desired_activity: float,
    music_semantic: np.ndarray | None = None,
    deep_music_weight: float = 0.0,
) -> Dict[str, Any]:
    # Preserve the original rule-level hierarchy query, while carrying the
    # learned Poincare embedding as a separate retrieval signal.
    query = _original_build_slot_query(
        phrase,
        predicted_event,
        target_natural,
        desired_activity,
        music_semantic=None,
        deep_music_weight=0.0,
    )
    if music_semantic is not None:
        query["v30_crossmodal_embed"] = np.asarray(
            music_semantic, dtype=np.float32
        ).reshape(-1)
        query["deep_music_weight"] = float(
            np.clip(deep_music_weight, 0.0, 1.0)
        )
    return query


def _node_scores(
    hierarchy: Dict[str, np.ndarray],
    query: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    base, parts = _original_node_scores(hierarchy, query)
    event_embed = hierarchy.get("v30_crossmodal_embed")
    music_embed = query.get("v30_crossmodal_embed")
    if event_embed is None or music_embed is None:
        return base, parts
    import torch

    event_tensor = torch.from_numpy(np.asarray(event_embed, np.float32))
    music_tensor = torch.from_numpy(
        np.asarray(music_embed, np.float32).reshape(1, -1)
    )
    curvature = float(
        np.asarray(
            hierarchy.get("v30_crossmodal_curvature", [1.0]),
            np.float32,
        ).reshape(-1)[0]
    )
    with torch.no_grad():
        distance = poincare_distance_pairwise(
            music_tensor, event_tensor, curvature
        )[0].cpu().numpy()
    crossmodal = np.exp(-0.65 * distance).astype(np.float32)
    weight = float(np.clip(
        float(os.environ.get("V30_CROSSMODAL_RETRIEVAL_WEIGHT", "0.35")),
        0.0, 0.80,
    ))
    score = ((1.0 - weight) * base + weight * crossmodal).astype(np.float32)
    parts["v30_crossmodal_score"] = crossmodal
    parts["v30_crossmodal_distance"] = distance.astype(np.float32)
    return score, parts



def _transition_with_geometric_music(
    bundle,
    start_frame,
    end_frame,
    length,
    music_query,
    rough=None,
    device="cpu",
    blend=0.85,
    steps=32,
):
    query = np.asarray(music_query, dtype=np.float32).reshape(-1)
    geometric = query
    if len(_CURRENT_RULE_QUERIES) and len(_CURRENT_GEOMETRIC_QUERIES):
        rule = query[: _CURRENT_RULE_QUERIES.shape[1]]
        distance = np.linalg.norm(_CURRENT_RULE_QUERIES - rule[None], axis=1)
        geometric = _CURRENT_GEOMETRIC_QUERIES[int(np.argmin(distance))]
    return _original_sample_transition(
        bundle,
        start_frame,
        end_frame,
        length,
        np.asarray(geometric, dtype=np.float32),
        rough=rough,
        device=device,
        blend=blend,
        steps=steps,
    )

# Patch symbols already imported into schedule_v26_whole_song.
scheduler.make_linear_transition = make_so3_transition
scheduler.dampen_event_edges = dampen_event_edges_so3
scheduler.apply_start_anchor = apply_start_anchor_so3
scheduler.boundary_metrics = _boundary_metrics
scheduler.phrase_semantic_matrix = _phrase_semantics
scheduler.build_slot_query = _build_slot_query
scheduler.hierarchical_node_scores = _node_scores
scheduler.sample_transition_diffusion = _transition_with_geometric_music

# Keep module-level users consistent.
hierarchy_module.build_slot_query = _build_slot_query
hierarchy_module.hierarchical_node_scores = _node_scores


if __name__ == "__main__":
    scheduler.main()
