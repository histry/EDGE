#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.49 wrapper around V46.46 closed-loop generation.

Adds:
1. merge of a tiny terminal residual slot into the previous slot;
2. direct-join risk evaluation for an empty explicit transition;
3. final gravity/body-frame hard gate before a result can be reported/rendered.

It imports the repository's latest V46.46 implementation rather than copying it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from tools import v46_46_boundary_closed_loop as base
from tools.v46_49_gravity_contract import (
    GravityThresholds,
    evaluate_gravity_contract,
    gravity_metrics_np,
)


_ORIG_TRANSITION_RISK = base.transition_risk
_ORIG_APPLY_GENERATORS = base.apply_generators


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def env_bool(name: str, default: bool) -> bool:
    return str(os.environ.get(name, "1" if default else "0")).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }


def transition_risk_guard(v46, previous, transition, following, fps):
    tr = np.asarray(transition, dtype=np.float32)
    if tr.shape[0] == 0:
        # An empty bridge means direct joining, not an invalid 1e9 boundary.
        return base.simple_boundary_risk(previous, tr, following, fps)
    risk = _ORIG_TRANSITION_RISK(v46, previous, tr, following, fps)
    vals = []
    for k in (
        "total", "boundary_joint_jerk_max", "exit_fk_jump",
        "exit_rotation_step_rad", "foot_slip", "foot_penetration",
    ):
        try:
            vals.append(float(risk.get(k, 0.0)))
        except Exception:
            vals.append(float("inf"))
    if any((not np.isfinite(v)) or abs(v) >= 1e8 for v in vals):
        return base.simple_boundary_risk(previous, tr, following, fps)
    return risk


def merge_short_terminal_slot(slots, slot_feat, cfg):
    slots = [dict(s) for s in slots]
    feat = np.asarray(slot_feat, dtype=np.float32)
    if len(slots) < 2 or len(feat) != len(slots):
        return slots, feat
    fps = float(getattr(cfg, "fps", 30.0))
    minimum = env_int("V46_49_MIN_TERMINAL_SLOT_FRAMES", int(round(fps)))
    prev_n = base.slot_target_frames(slots[-2], cfg)
    tail_n = base.slot_target_frames(slots[-1], cfg)
    if tail_n >= minimum:
        return slots, feat

    merged = dict(slots[-2])
    tail = slots[-1]
    total = int(prev_n + tail_n)
    merged["target_frames"] = total
    merged["duration"] = merged["duration_sec"] = total / fps
    for key in ("end", "end_sec", "end_time", "end_frame", "audio_end"):
        if key in tail:
            merged[key] = tail[key]
    merged["v46_49_terminal_merge"] = {
        "previous_frames": int(prev_n),
        "tail_frames": int(tail_n),
        "merged_frames": total,
        "threshold_frames": minimum,
    }
    merged_feat = (feat[-2] * prev_n + feat[-1] * tail_n) / max(1, total)
    slots[-2] = merged
    slots.pop()
    feat = feat[:-1].copy()
    feat[-1] = merged_feat
    print(
        f"[V46.49 TAIL MERGE] tail={tail_n} -> previous; "
        f"merged={total}; slots={len(slots)}",
        file=sys.stderr,
    )
    return slots, feat.astype(np.float32)


def load_slots_and_candidates(v46, args, cfg):
    db = v46.load_db(args.db)
    contrastive = v46.load_contrastive(getattr(args, "contrastive", None), cfg)
    slots, slot_feat = v46.audio_slots(
        args.audio,
        cfg,
        args.slot_seconds,
        getattr(args, "slots_json", None),
    )
    slots, slot_feat = merge_short_terminal_slot(slots, slot_feat, cfg)
    path_idx, retrieval_report = v46.retrieve_schedule(
        slots, slot_feat, db, cfg, contrastive
    )
    candidate_lists = base.extract_candidate_lists(
        path_idx, retrieval_report, db, cfg
    )
    return (
        db,
        contrastive,
        list(slots),
        np.asarray(slot_feat, dtype=np.float32),
        list(map(int, path_idx)),
        list(retrieval_report),
        candidate_lists,
    )


def apply_generators_with_gravity(v46, motion_ref, cond, seam_mask, args, cfg):
    motion, stage = _ORIG_APPLY_GENERATORS(
        v46, motion_ref, cond, seam_mask, args, cfg
    )
    metrics = gravity_metrics_np(motion, fps=float(getattr(cfg, "fps", 30.0)))
    ok, reasons = evaluate_gravity_contract(metrics, GravityThresholds())
    stage["v46_49_gravity_contract"] = {
        "ok": bool(ok),
        "reasons": reasons,
        **metrics,
    }
    if not ok and env_bool("V46_49_GRAVITY_HARD_FAIL", True):
        raise RuntimeError(
            "V46.49 gravity contract rejected generated motion: "
            + " | ".join(reasons)
        )
    return motion, stage


def install_runtime_guards() -> None:
    base.transition_risk = transition_risk_guard
    base.load_slots_and_candidates = load_slots_and_candidates
    base.apply_generators = apply_generators_with_gravity


def main(argv: Optional[Sequence[str]] = None) -> int:
    install_runtime_guards()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
