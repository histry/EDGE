#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V46.44 Chang-E BVH canonicalizer v2
-----------------------------------
Fixes the double-root-scale bug:
  - root POSITION channels are scaled by root_scale, e.g. cm -> m = 0.01
  - all HIERARCHY OFFSET lines are scaled by the same root_scale
  - non-root POSITION channels are removed; non-root JOINTS keep only rotations

Output is a standard rot-only BVH:
  ROOT: 6 channels = X/Y/Zposition + rotations
  JOINT: 3 channels = rotations only

This makes tools/v46_motionrag_diff.py load_bvh_file() see a meter-scale
skeleton and prevents the old auto-scale path from multiplying root by 0.01 again.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _parse_layout(head: str) -> List[Dict[str, object]]:
    layout: List[Dict[str, object]] = []
    current = None
    for line in head.splitlines():
        m = re.match(r"\s*(ROOT|JOINT)\s+([^\s{]+)", line)
        if m:
            current = {"type": m.group(1), "name": m.group(2)}
            continue
        m = re.match(r"(\s*)CHANNELS\s+(\d+)\s+(.+)", line)
        if m and current is not None:
            current["indent"] = m.group(1)
            current["n"] = int(m.group(2))
            current["channels"] = m.group(3).split()
            layout.append(current)
            current = None
    return layout


def _scale_offsets_in_hierarchy(head: str, scale: float) -> Tuple[str, Dict[str, object]]:
    offset_re = re.compile(r"^(\s*OFFSET\s+)([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)(\s*)$", flags=re.M)
    values_before = []
    values_after = []

    def repl(m: re.Match) -> str:
        vals = np.asarray([float(m.group(2)), float(m.group(3)), float(m.group(4))], dtype=np.float64)
        values_before.append(vals.tolist())
        vals2 = vals * float(scale)
        values_after.append(vals2.tolist())
        return f"{m.group(1)}{vals2[0]:.6f} {vals2[1]:.6f} {vals2[2]:.6f}{m.group(5)}"

    new_head = offset_re.sub(repl, head)
    before = np.asarray(values_before, dtype=np.float64) if values_before else np.zeros((0, 3), dtype=np.float64)
    after = np.asarray(values_after, dtype=np.float64) if values_after else np.zeros((0, 3), dtype=np.float64)
    rep = {
        "offset_lines_scaled": int(len(values_before)),
        "offset_scale": float(scale),
        "offset_norm_p90_before": float(np.percentile(np.linalg.norm(before, axis=1), 90)) if len(before) else 0.0,
        "offset_norm_p90_after": float(np.percentile(np.linalg.norm(after, axis=1), 90)) if len(after) else 0.0,
    }
    return new_head, rep


def _split_motion(mot: str) -> Tuple[List[str], List[str]]:
    lines = mot.strip().splitlines()
    header_lines: List[str] = []
    frame_lines: List[str] = []
    started = False
    for line in lines:
        if started:
            if line.strip():
                frame_lines.append(line.strip())
        else:
            header_lines.append(line)
            if line.strip().lower().startswith("frame time"):
                started = True
    return header_lines, frame_lines


def canonicalize_one(src: str | Path, dst: str | Path, root_scale: float = 0.01) -> Dict[str, object]:
    src = Path(src)
    text = src.read_text(encoding="utf-8", errors="ignore")
    if "MOTION" not in text:
        raise RuntimeError(f"No MOTION section: {src}")
    head, mot = text.split("MOTION", 1)

    # Critical: scale offsets too, otherwise a downstream loader may infer cm units
    # from hierarchy offsets and shrink an already-meter root trajectory again.
    head, offset_report = _scale_offsets_in_hierarchy(head, float(root_scale))

    layout = _parse_layout(head)
    if not layout:
        raise RuntimeError(f"No CHANNELS parsed: {src}")

    keep_global: List[int] = []
    root_pos_global: List[int] = []
    cursor = 0

    for idx, item in enumerate(layout):
        chs = list(item["channels"])
        n = int(item["n"])
        if idx == 0:
            keep_local = list(range(n))
            new_chs = chs
            for i, c in enumerate(chs):
                if "position" in c.lower():
                    root_pos_global.append(cursor + i)
        else:
            keep_local = [i for i, c in enumerate(chs) if "rotation" in c.lower()]
            new_chs = [chs[i] for i in keep_local]
            if len(new_chs) != 3:
                raise RuntimeError(f"{src}: abnormal rotation channels for {item['name']}: {chs}")

        keep_global.extend([cursor + i for i in keep_local])
        old = re.compile(
            r"^" + re.escape(str(item["indent"])) +
            r"CHANNELS\s+" + str(n) + r"\s+" +
            re.escape(" ".join(chs)) + r"$",
            flags=re.M,
        )
        new_line = str(item["indent"]) + "CHANNELS " + str(len(new_chs)) + " " + " ".join(new_chs)
        head = old.sub(new_line, head)
        cursor += n

    header_lines, frame_lines = _split_motion(mot)
    if not frame_lines:
        raise RuntimeError(f"No frame data: {src}")

    raw = np.asarray([[float(x) for x in fl.split()] for fl in frame_lines], dtype=np.float64)
    if raw.shape[1] < cursor:
        raise RuntimeError(f"{src}: motion has {raw.shape[1]} channels, expected {cursor}")

    root_before = raw[:, root_pos_global].copy() if root_pos_global else np.zeros((raw.shape[0], 0))
    raw[:, root_pos_global] *= float(root_scale)
    root_after = raw[:, root_pos_global].copy() if root_pos_global else np.zeros((raw.shape[0], 0))
    kept = raw[:, keep_global]

    new_frames = [" ".join(f"{x:.6f}" for x in row.tolist()) for row in kept]
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_text = head.rstrip() + "\nMOTION\n" + "\n".join(header_lines) + "\n" + "\n".join(new_frames) + "\n"
    dst.write_text(out_text, encoding="utf-8")

    def range_l2(a: np.ndarray) -> float:
        if a.size == 0:
            return 0.0
        return float(np.linalg.norm(a.max(axis=0) - a.min(axis=0)))

    return {
        "version": "v46_44_rot_only_meter_offsets_scaled",
        "src": str(src),
        "dst": str(dst),
        "frames": int(raw.shape[0]),
        "nodes": int(len(layout)),
        "old_channels": int(cursor),
        "new_channels": int(len(keep_global)),
        "root_scale": float(root_scale),
        "root_pos_range_before_l2": range_l2(root_before),
        "root_pos_range_after_l2": range_l2(root_after),
        "offset_report": offset_report,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="change")
    ap.add_argument("--out_dir", default="output/change_rot_only_meter_bvh_v46_44")
    ap.add_argument("--root_scale", type=float, default=0.01)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    reports = []
    for p in sorted(in_dir.rglob("*.bvh")):
        rel = p.relative_to(in_dir)
        rep = canonicalize_one(p, out_dir / rel, root_scale=args.root_scale)
        reports.append(rep)
        print(
            "[SAVED]", rep["dst"],
            "old_ch=", rep["old_channels"], "new_ch=", rep["new_channels"],
            "root_range:", round(rep["root_pos_range_before_l2"], 3), "->", round(rep["root_pos_range_after_l2"], 3),
            "offset_p90:", round(rep["offset_report"]["offset_norm_p90_before"], 3), "->", round(rep["offset_report"]["offset_norm_p90_after"], 3),
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = out_dir / "canonicalization_audit_v46_44.json"
    audit.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[AUDIT]", audit)
    print("[DONE] converted:", len(reports))


if __name__ == "__main__":
    main()
