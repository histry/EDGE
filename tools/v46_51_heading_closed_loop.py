#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.51 strict generation entrypoint.

It rejects any schedule that was not rebuilt from the current WAV in the same
generation transaction, then delegates motion planning and generation to the
latest V46.50 heading-aware closed-loop implementation.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v46_51_audio_schedule_contract import (  # noqa: E402
    audit_contract,
    save_json,
)
import tools.v46_50_heading_closed_loop as v4650  # noqa: E402


def _arg_value(argv: Sequence[str], flag: str) -> Optional[str]:
    args = list(argv)
    try:
        idx = args.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(args):
        return None
    return args[idx + 1]


def _patch_report(
    report_path: Path,
    contract: Dict[str, Any],
) -> None:
    if not report_path.is_file():
        return
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return
    report["version"] = "v46_51_fresh_wav_event_heading_closed_loop"
    report["audio_schedule_transaction"] = {
        "schema": contract.get("schema"),
        "ok": contract.get("ok"),
        "audio": contract.get("audio"),
        "schedule_path": contract.get("schedule_path"),
        "schedule_sha256": contract.get("schedule_sha256"),
        "required_run_id": contract.get("required_run_id"),
        "transaction": contract.get("transaction"),
        "num_slots": contract.get("num_slots"),
        "total_target_frames": contract.get("total_target_frames"),
        "expected_audio_target_frames": contract.get(
            "expected_audio_target_frames"
        ),
        "frame_error": contract.get("frame_error"),
        "overlap_count": contract.get("overlap_count"),
        "gap_count": contract.get("gap_count"),
    }
    report["v46_51_env"] = {
        k: v
        for k, v in os.environ.items()
        if k.startswith("V46_51_")
    }
    save_json(report, report_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    audio = _arg_value(args, "--audio")
    schedule = _arg_value(args, "--slots_json")
    report_json = _arg_value(args, "--json")

    if not audio:
        raise RuntimeError("V46.51 requires --audio")
    if not schedule:
        raise RuntimeError(
            "V46.51 requires its run-local freshly generated --slots_json"
        )

    required_run_id = os.environ.get("V46_51_SCHEDULE_RUN_ID")
    if not required_run_id:
        raise RuntimeError(
            "V46_51_SCHEDULE_RUN_ID is required. "
            "Use scripts/run_v46_51_generate_fresh_wav.sh."
        )

    require_fresh = str(
        os.environ.get("V46_51_REQUIRE_FRESH_WAV_SCHEDULE", "1")
    ).strip().lower() not in {"0", "false", "no", "off"}

    contract = audit_contract(
        audio=audio,
        schedule=schedule,
        fps=float(os.environ.get("V46_FPS", "30")),
        required_run_id=required_run_id,
        require_fresh=require_fresh,
        max_frame_error=int(
            float(os.environ.get("V46_51_MAX_FRAME_ERROR", "2"))
        ),
        max_seconds_error=float(
            os.environ.get("V46_51_MAX_SECONDS_ERROR", "0.10")
        ),
        require_raw_report=True,
    )

    contract_out = Path(schedule).with_suffix(
        Path(schedule).suffix + ".pre_generate_contract.json"
    )
    save_json(contract, contract_out)
    if not contract["ok"]:
        raise RuntimeError(
            "V46.51 refused generation because the current-WAV schedule "
            "transaction failed: "
            + "; ".join(contract["reasons"])
        )

    print(
        json.dumps(
            {
                "v46_51_schedule_contract": "PASS",
                "run_id": required_run_id,
                "audio_sha256": contract["audio"]["sha256"],
                "slots": contract["num_slots"],
                "frames": contract["total_target_frames"],
                "schedule": str(Path(schedule).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    rc = int(v4650.main(args))
    if report_json:
        _patch_report(Path(report_json), contract)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
