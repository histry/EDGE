#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V41.2 verifier/hotfix for FK-verified native-floor planner.

This tool verifies two requirements before long-run generation:
1. no raw 151D column foot-Y extraction is used by the injector;
2. the planner has a finite dead-rescue path instead of absolute graph death.

It also patches the rare old manual idiom:
    if is_dead: continue
into a finite rescue penalty guarded by V41_NATIVE_FLOOR_RELAX_ON_EMPTY.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime


REQUIRED_PLANNER_ANCHORS = [
    "_native_floor_measure_arrays",
    "native_floor_penetration_m",
    "native_floor_barrier_dead_rescue_penalty",
    "V41_NATIVE_FLOOR_RELAX_ON_EMPTY",
]

FORBIDDEN_INJECTOR_PATTERNS = [
    "foot_y_indices = [7, 10]",
    "clip[:, foot_y_indices]",
    "REAL_L_FOOT_Y_IDX",
    "REAL_R_FOOT_Y_IDX",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--planner", default="tools/v34_warp_aware_retrieval.py")
    ap.add_argument("--injector", default="tools/v41b_inject_min_foot_y_to_db.py")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    planner = root / args.planner
    injector = root / args.injector

    if not planner.is_file():
        raise FileNotFoundError(planner)
    if not injector.is_file():
        raise FileNotFoundError(injector)

    ptxt = planner.read_text(encoding="utf-8")
    itxt = injector.read_text(encoding="utf-8")

    forbidden = [pat for pat in FORBIDDEN_INJECTOR_PATTERNS if pat in itxt]
    if forbidden:
        raise RuntimeError(
            "Unsafe raw-column foot-Y extraction detected in injector: "
            + ", ".join(forbidden)
        )
    if "rot6d" not in itxt.lower() or "_fk_24" not in itxt or "raw_column_mode" not in itxt:
        raise RuntimeError(
            "Injector does not look FK-verified. It must compute foot height via rot6d -> FK, not raw 151D channels."
        )

    missing = [anchor for anchor in REQUIRED_PLANNER_ANCHORS if anchor not in ptxt]
    if missing:
        raise RuntimeError(
            "V41 planner patch not detected or incomplete; missing anchors: "
            + ", ".join(missing)
        )

    backup = planner.with_suffix(planner.suffix + f".before_v41_2_hotfix_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup.write_text(ptxt, encoding="utf-8")

    changed = False
    old = """if is_dead:\n    continue  # 【Hard Remove】波束搜索根本不把这个子节点压入堆栈！"""
    new = """if is_dead:\n    if not _enabled(\"V41_NATIVE_FLOOR_RELAX_ON_EMPTY\", \"1\"):\n        continue\n    floor_pen = _env_float(\"V41_NATIVE_FLOOR_DEAD_RESCUE_PENALTY\", 25.0)"""
    if old in ptxt:
        ptxt = ptxt.replace(old, new)
        changed = True

    if "V41_2_FK_VERIFIED_NATIVE_FLOOR_HOTFIX" not in ptxt:
        ptxt = ptxt.replace(
            "import os\n",
            "import os\n# V41_2_FK_VERIFIED_NATIVE_FLOOR_HOTFIX: FK-derived metadata + finite dead rescue verified.\n",
            1,
        )
        changed = True

    planner.write_text(ptxt, encoding="utf-8")
    print(f"[OK] V41.2 FK-verified hotfix passed. changed={changed} backup={backup}")


if __name__ == "__main__":
    main()
