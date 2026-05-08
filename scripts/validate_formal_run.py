#!/usr/bin/env python3
"""Validate formal V10 ChoreoRAG evidence files.

Use after generation/evaluation:
    python scripts/validate_formal_run.py --prefix output/demo

Checks:
- wrapper contract exists and contains unit_paths
- unit prior report exists, enabled, num_applied > 0, safety invariants true
- context report is present when Text/Pose Context RAG is enabled
- eval report has both metrics_raw and metrics_final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def require_file(path: Path, errors: List[str]) -> bool:
    if not path.exists():
        errors.append(f"missing file: {path}")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="Output prefix without suffix, e.g. output/demo")
    ap.add_argument("--require_context", action="store_true", help="Require context_report attached=true")
    args = ap.parse_args()

    prefix = Path(args.prefix)
    errors: List[str] = []
    warnings: List[str] = []

    wrapper_path = Path(str(prefix) + "_wrapper_contract.json")
    unit_path = Path(str(prefix) + "_unit_prior_report.json")
    context_path = Path(str(prefix) + "_context_report.json")
    eval_path = Path(str(prefix) + "_eval.json")

    if require_file(wrapper_path, errors):
        wrapper = load_json(wrapper_path)
        unit_paths = wrapper.get("unit_paths") or []
        if not unit_paths:
            errors.append("wrapper_contract has empty unit_paths")
        if wrapper.get("run_mode") != "formal":
            warnings.append(f"wrapper_contract run_mode is {wrapper.get('run_mode')!r}, not 'formal'")

    if require_file(unit_path, errors):
        unit = load_json(unit_path)
        if not unit.get("enabled", False):
            errors.append("unit_prior_report enabled=false")
        if int(unit.get("num_applied", 0)) <= 0:
            errors.append(f"unit_prior_report num_applied={unit.get('num_applied')}")
        if not unit.get("aggregate_safe_no_contact", False):
            errors.append("unit prior aggregate_safe_no_contact is false")
        if not unit.get("aggregate_safe_no_root_xz", False):
            errors.append("unit prior aggregate_safe_no_root_xz is false")
        if unit.get("error"):
            errors.append(f"unit_prior_report has error: {unit.get('error')}")

    if context_path.exists():
        context = load_json(context_path)
        if args.require_context and not context.get("attached", False):
            errors.append("context_report attached=false")
        if context.get("attached", False):
            if int(context.get("unit_count", 0)) <= 0:
                errors.append("context_report attached=true but unit_count<=0")
            if not context.get("context_hash"):
                warnings.append("context_report missing context_hash")
    elif args.require_context:
        errors.append(f"missing file: {context_path}")

    if require_file(eval_path, errors):
        ev = load_json(eval_path)
        if "metrics_raw" not in ev or "metrics_final" not in ev:
            errors.append("eval report must contain metrics_raw and metrics_final")
        if ev.get("warnings"):
            warnings.extend(["eval warning: " + str(w) for w in ev.get("warnings", [])])

    result = {
        "prefix": str(prefix),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
