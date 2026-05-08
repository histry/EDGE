#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(payload, key, stage="metrics_raw"):
    try:
        return float(payload[stage][key])
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.eval_dir)
    cases = {}
    for path in root.glob("*_eval.json"):
        name = path.name.replace("_eval.json", "")
        cases[name] = load(path)

    metrics = {}
    for name, p in cases.items():
        metrics[name] = {
            "raw_upper_activity": get_metric(p, "upper_activity"),
            "raw_motion_energy": get_metric(p, "motion_energy"),
            "raw_transition_jerk": get_metric(p, "transition_jerk"),
            "raw_contact_phase_break": get_metric(p, "contact_phase_break"),
            "raw_trajectory_ade_m": get_metric(p, "trajectory_ade_m"),
            "final_trajectory_ade_m": get_metric(p, "trajectory_ade_m", stage="metrics_final"),
            "warnings": p.get("warnings", []),
        }

    def better_context_than(other: str) -> bool | None:
        if "context" not in metrics or other not in metrics:
            return None
        c = metrics["context"]
        o = metrics[other]
        cu = c.get("raw_upper_activity")
        ou = o.get("raw_upper_activity")
        cj = c.get("raw_transition_jerk")
        oj = o.get("raw_transition_jerk")
        if cu is None or ou is None or cj is None or oj is None:
            return None
        # Minimal evidence rule: higher/equal activity with no large jerk increase.
        return (cu >= ou * 1.02) and (cj <= max(oj * 1.20, oj + 0.03))

    summary = {
        "cases_found": sorted(cases),
        "metrics": metrics,
        "claim_checks": {
            "context_beats_no_context": better_context_than("no_context"),
            "context_beats_shuffled_context": better_context_than("shuffled_context"),
            "context_beats_wrong_text": better_context_than("wrong_text"),
        },
        "claim_guidance": (
            "Claim Text/Pose Context RAG semantic effectiveness only if context beats no_context, "
            "shuffled_context and wrong_text under raw metrics, and text_context gate/grad evidence is non-zero."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Context ablation summary saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
