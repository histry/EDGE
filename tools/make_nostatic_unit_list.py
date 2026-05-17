#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set


def read_bad_units(path: str) -> Set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    bad: Set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        bad.add(line)
        if line.startswith("unit_"):
            bad.add(line.replace("unit_", ""))
        else:
            bad.add("unit_" + line)
    return bad


def as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def unit_variants(unit: str) -> Set[str]:
    unit = str(unit).strip()
    if unit.startswith("unit_"):
        return {unit, unit.replace("unit_", "")}
    return {unit, "unit_" + unit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_csv", required=True)
    parser.add_argument("--bad_units_txt", default="")
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument("--max_units", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min_active_frame_ratio", type=float, default=0.0)
    parser.add_argument("--max_freeze_tail_ratio", type=float, default=1.0)
    args = parser.parse_args()

    score_path = Path(args.score_csv)
    if not score_path.exists():
        raise FileNotFoundError(f"score_csv not found: {score_path}")

    bad_units = read_bad_units(args.bad_units_txt)

    rows: List[Dict[str, str]] = []
    with score_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unit = str(row.get("unit", "")).strip()
            if not unit:
                continue

            if args.step >= 0:
                try:
                    if int(float(row.get("step", "-999999"))) != args.step:
                        continue
                except Exception:
                    continue

            if unit_variants(unit) & bad_units:
                continue

            if "active_frame_ratio" in row and as_float(row, "active_frame_ratio", 1.0) < args.min_active_frame_ratio:
                continue
            if "freeze_tail_ratio" in row and as_float(row, "freeze_tail_ratio", 0.0) > args.max_freeze_tail_ratio:
                continue

            rows.append(row)

    if not rows:
        raise RuntimeError("No units left after filtering. Check --step, --bad_units_txt, and score CSV.")

    def sort_key(row: Dict[str, str]):
        if "score" in row and row.get("score", "") != "":
            return as_float(row, "score", 1e9)
        return (
            as_float(row, "phys_mse", 0.0)
            + 2.0 * as_float(row, "rootxz_mse", 0.0)
            + 0.15 * as_float(row, "jump_p95", 0.0)
            + 0.15 * as_float(row, "jerk_p95", 0.0)
        )

    rows.sort(key=sort_key)
    selected = rows[: max(1, int(args.max_units))]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(str(row["unit"]).replace("unit_", "") + "\n")

    print(f"✅ selected {len(selected)} units -> {out_path}")
    print("unit,step,score,phys_mse,rootxz_mse,jump_p95,jerk_p95")
    for row in selected:
        print(",".join([
            str(row.get("unit", "")),
            str(row.get("step", "")),
            str(row.get("score", "")),
            str(row.get("phys_mse", "")),
            str(row.get("rootxz_mse", "")),
            str(row.get("jump_p95", "")),
            str(row.get("jerk_p95", "")),
        ]))


if __name__ == "__main__":
    main()
