#!/usr/bin/env python3
"""Check inference logs for true Text/Pose Context RAG + trajectory patch contract."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--context_report", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    log_path = Path(args.log)
    text = log_path.read_text(encoding="utf-8", errors="replace")

    checks = {
        "checkpoint_has_rag_true": "checkpoint_has_rag=True" in text,
        "rag_attached": "Text/Pose Context RAG attached to inference cond" in text,
        "rag_appended": "Text/Pose Context RAG appended to decoder memory" in text,
        "trajectory_enhancement_installed": (
            "Installed trajectory enhancement patch" in text
            or "Advanced trajectory branch initialized" in text
        ),
        "gait_patch_installed": (
            "Installed gait phase trajectory patch" in text
            or "Gait phase trajectory wrapper initialized" in text
        ),
        "no_traj_new_init": not bool(re.search(r"trajectory_projection\.[^\\n]*kept newly initialized", text)),
        "no_traj_ignored": not bool(re.search(r"trajectory_projection\.[^\\n]*ignored unexpected", text)),
    }

    report = None
    if args.context_report:
        p = Path(args.context_report)
        if p.exists():
            report = json.loads(p.read_text(encoding="utf-8"))
            checks["context_report_attached"] = bool(report.get("attached", False))
            checks["context_report_unit_count_positive"] = int(report.get("unit_count", 0)) > 0
        else:
            checks["context_report_exists"] = False

    print("=" * 80)
    print("RAG / Trajectory inference contract check")
    print("log:", log_path)
    print("=" * 80)
    for k, v in checks.items():
        print(f"{k}: {'OK' if v else 'FAIL'}")

    if report is not None:
        print("\ncontext_report:")
        print(json.dumps(report, indent=2, ensure_ascii=False)[:4000])

    failed = [k for k, v in checks.items() if not v]
    if failed:
        print("\nFAILED:", failed)
        if args.strict:
            raise SystemExit(1)
    else:
        print("\n✅ All checks passed.")


if __name__ == "__main__":
    main()
