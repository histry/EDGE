#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate V31 transition safety-gate decisions from schedule reports."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_glob", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    rows = []
    total = accepted = fallback = candidates = 0
    for path in sorted(glob.glob(args.report_glob)):
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        for boundary_index, boundary in enumerate(
            report.get("boundary_metrics", [])
        ):
            diffusion = boundary.get("transition_diffusion", {})
            if not diffusion.get("enabled", False):
                continue
            total += 1
            accepted_count = int(diffusion.get("accepted_count", 0))
            candidate_count = int(diffusion.get("candidate_count", 0))
            was_fallback = bool(
                diffusion.get("fallback_to_c2_baseline", False)
            )
            candidates += candidate_count
            accepted += accepted_count
            fallback += int(was_fallback)
            rows.append({
                "report": path,
                "boundary_index": boundary_index,
                "accepted_count": accepted_count,
                "candidate_count": candidate_count,
                "fallback": was_fallback,
                "selected_index": diffusion.get("selected_index", -1),
                "baseline_risk": diffusion.get("baseline_risk", {}),
            })
    result = {
        "version": "v31_transition_gate_summary",
        "num_boundaries": total,
        "fallback_count": fallback,
        "fallback_rate": fallback / max(total, 1),
        "accepted_candidate_count": accepted,
        "sampled_candidate_count": candidates,
        "candidate_acceptance_rate": accepted / max(candidates, 1),
        "rows": rows,
    }
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        key: value for key, value in result.items() if key != "rows"
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
