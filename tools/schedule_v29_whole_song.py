#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V29 research entry point for the existing V26 whole-song scheduler.

The validated music segmentation, boundary lock, hierarchy retrieval, graph
scheduler and V23 duration allocation remain unchanged.  Geometry-sensitive
operations are replaced at runtime by SO(3)-correct V29 implementations.

This design deliberately avoids duplicating the thousand-line V26 scheduler,
so future improvements to the planner remain available while the local motion
layer is scientifically corrected.
"""
from __future__ import annotations

import tools.schedule_v26_whole_song as scheduler
from tools.v29_motion_geometry import (
    apply_start_anchor_so3,
    dampen_event_edges_so3,
    endpoint_metrics_np,
    make_so3_transition,
)


def _boundary_metrics(prev, nxt):
    return endpoint_metrics_np(prev, nxt, fps=30.0)


# Replace module-level symbols imported by schedule_v26_whole_song.
scheduler.make_linear_transition = make_so3_transition
scheduler.dampen_event_edges = dampen_event_edges_so3
scheduler.apply_start_anchor = apply_start_anchor_so3
scheduler.boundary_metrics = _boundary_metrics


if __name__ == "__main__":
    scheduler.main()
