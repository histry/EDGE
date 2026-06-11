#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate V32 candidate-gate results from schedule reports."""
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
    boundaries = candidates = accepted = fallback = 0
    for path in sorted(glob.glob(args.report_glob)):
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        for index, boundary in enumerate(report.get("boundary_metrics", [])):
            meta = boundary.get("transition_diffusion", {})
            if not meta.get("enabled", False):
                continue
            boundaries += 1
            candidate_count = int(meta.get("candidate_count", 0))
            accepted_count = int(meta.get("accepted_count", 0))
            was_fallback = bool(meta.get("fallback_to_c2_baseline", False))
            candidates += candidate_count
            accepted += accepted_count
            fallback += int(was_fallback)
            rows.append({
                "report": path,
                "boundary_index": index,
                "candidate_count": candidate_count,
                "accepted_count": accepted_count,
                "fallback": was_fallback,
                "selected_index": meta.get("selected_index", -1),
                "baseline_risk": meta.get("baseline_risk", {}),
            })
    result = {
        "version": "v32_contact_inr_gate_summary",
        "num_boundaries": boundaries,
        "sampled_candidates": candidates,
        "accepted_candidates": accepted,
        "candidate_acceptance_rate": accepted / max(candidates, 1),
        "fallback_count": fallback,
        "fallback_rate": fallback / max(boundaries, 1),
        "rows": rows,
    }
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(
        {k: v for k, v in result.items() if k != "rows"},
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
