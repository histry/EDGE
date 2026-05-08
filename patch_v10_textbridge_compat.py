#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path

TARGET = Path("v10_choreo_planner.py")

APPEND = """
# ---------------------------------------------------------------------------
# Compatibility helper for text_bridge_planner_patch.py
# ---------------------------------------------------------------------------
def _score_parts_from_score(score):
    # Return a JSON-serializable score dictionary for a UnitScore-like object.
    # Older text_bridge_planner_patch.py expects this private helper to exist.
    # This helper is permissive and preserves dynamically added semantic fields.
    try:
        from dataclasses import asdict, is_dataclass
    except Exception:
        asdict = None
        is_dataclass = lambda _: False  # type: ignore

    out = {}
    if asdict is not None and is_dataclass(score):
        try:
            out.update(asdict(score))
        except Exception:
            pass

    keys = [
        "index", "unit_id", "source_key", "source_index",
        "score", "emission_score", "path_score", "transition_score",
        "transition_cost", "entry_cost", "exit_cost",
        "upper_activity", "lower_activity", "root_speed", "spatial_range",
        "turning", "pose_diversity", "contact_change", "contact_stability",
        "unit_energy", "motion_energy", "expressiveness_score", "text_score",
        "semantic_score", "semantic_score_norm", "semantic_query",
        "motion_text", "text_bridge_weight", "text_bridge_mode", "original_score",
    ]

    for key in keys:
        if not hasattr(score, key):
            continue
        value = getattr(score, key)
        try:
            import numpy as _np
            if isinstance(value, (_np.floating, _np.integer)):
                value = value.item()
            elif isinstance(value, _np.ndarray):
                value = value.tolist()
        except Exception:
            pass
        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
        except Exception:
            pass
        out[key] = value

    if not out:
        for key in dir(score):
            if key.startswith("_"):
                continue
            try:
                value = getattr(score, key)
            except Exception:
                continue
            if callable(value):
                continue
            if isinstance(value, (str, int, float, bool, type(None))):
                out[key] = value

    return out
"""

def main() -> int:
    if not TARGET.exists():
        raise FileNotFoundError(f"{TARGET} not found. Run from the EDGE repo root.")
    text = TARGET.read_text(encoding="utf-8")
    if "def _score_parts_from_score(" in text:
        print("✅ v10_choreo_planner.py already has _score_parts_from_score; no change.")
        return 0
    TARGET.write_text(text.rstrip() + "\n" + APPEND + "\n", encoding="utf-8")
    print("✅ Added _score_parts_from_score compatibility helper to v10_choreo_planner.py")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
