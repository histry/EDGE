#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonicalize Chang-E 6DoF BVH files for the EDGE/V46 single-person pipeline.

The Chang-E BVH files observed in this project use 19 nodes where every node has
six channels: X/Y/Z position + Z/X/Y rotation.  The non-root position channels are
nearly static offsets, while the V46/EDGE pipeline expects a fixed skeleton with
root translation and local joint rotations.  This tool converts the BVH files to a
more standard form:

  ROOT:  X/Y/Z position + Z/X/Y rotation, with root positions scaled to meters.
  JOINT: Z/X/Y rotation only; non-root position channels are removed.

It also writes an audit report and optionally rejects files whose non-root
position channels are not quasi-static.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_layout(head: str) -> List[Dict[str, object]]:
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


def parse_motion(mot: str) -> Tuple[List[str], np.ndarray]:
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
    if not frame_lines:
        raise RuntimeError("No frame data in MOTION section")
    arr = np.asarray([[float(x) for x in fl.split()] for fl in frame_lines], dtype=np.float64)
    return header_lines, arr


def audit_nonroot_positions(layout: List[Dict[str, object]], arr: np.ndarray) -> Dict[str, object]:
    cursor = 0
    root_pos_global: List[int] = []
    nonroot_stats = []
    for idx, item in enumerate(layout):
        chs = list(item["channels"])
        n = len(chs)
        pos_local = [i for i, c in enumerate(chs) if "position" in c.lower()]
        if idx == 0:
            root_pos_global = [cursor + i for i in pos_local]
        elif pos_local:
            pos = arr[:, [cursor + i for i in pos_local]]
            nonroot_stats.append({
                "joint": str(item["name"]),
                "pos_range_l2": float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0))),
                "pos_std": pos.std(axis=0).tolist(),
            })
        cursor += n
    root_range = 0.0
    if root_pos_global:
        root = arr[:, root_pos_global]
        root_range = float(np.linalg.norm(root.max(axis=0) - root.min(axis=0)))
    ranges = [x["pos_range_l2"] for x in nonroot_stats]
    return {
        "root_pos_range_l2": root_range,
        "nonroot_position_nodes": len(nonroot_stats),
        "nonroot_pos_range_max": float(max(ranges)) if ranges else 0.0,
        "nonroot_pos_range_p95": float(np.percentile(ranges, 95)) if ranges else 0.0,
        "nonroot_pos_range_mean": float(np.mean(ranges)) if ranges else 0.0,
        "worst_nonroot_pos": sorted(nonroot_stats, key=lambda x: x["pos_range_l2"], reverse=True)[:8],
    }


def canonicalize_one(src: Path, dst: Path, root_scale: float, max_nonroot_pos_range: float, copy_rejected_dir: Path | None) -> Dict[str, object]:
    text = src.read_text(errors="ignore")
    if "MOTION" not in text:
        raise RuntimeError(f"No MOTION section: {src}")
    head, mot = text.split("MOTION", 1)
    layout = parse_layout(head)
    if not layout:
        raise RuntimeError(f"No CHANNELS parsed: {src}")
    header_lines, raw_arr = parse_motion(mot)
    old_total_channels = int(sum(len(item["channels"]) for item in layout))
    if raw_arr.shape[1] < old_total_channels:
        raise RuntimeError(f"{src}: frame has {raw_arr.shape[1]} channels, expected {old_total_channels}")

    audit = audit_nonroot_positions(layout, raw_arr)
    accepted = bool(audit["nonroot_pos_range_max"] <= max_nonroot_pos_range)
    if not accepted:
        if copy_rejected_dir is not None:
            copy_rejected_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, copy_rejected_dir / src.name)
        return {
            "src": str(src),
            "dst": None,
            "accepted": False,
            "reason": "nonroot_position_not_static",
            "max_nonroot_pos_range": max_nonroot_pos_range,
            **audit,
        }

    keep_global: List[int] = []
    root_pos_global: List[int] = []
    cursor = 0
    for idx, item in enumerate(layout):
        chs = list(item["channels"])
        n = len(chs)
        if idx == 0:
            keep_local = list(range(n))
            new_chs = chs
            root_pos_global = [cursor + i for i, c in enumerate(chs) if "position" in c.lower()]
        else:
            keep_local = [i for i, c in enumerate(chs) if "rotation" in c.lower()]
            new_chs = [chs[i] for i in keep_local]
            if len(new_chs) != 3:
                raise RuntimeError(f"{src}: abnormal rotation channels for {item['name']}: {chs}")
        keep_global.extend([cursor + i for i in keep_local])
        old = re.compile(
            r"^" + re.escape(str(item["indent"])) + r"CHANNELS\s+" + str(n) + r"\s+" + re.escape(" ".join(chs)) + r"$",
            flags=re.M,
        )
        new_line = str(item["indent"]) + "CHANNELS " + str(len(new_chs)) + " " + " ".join(new_chs)
        head = old.sub(new_line, head)
        cursor += n

    before_root = raw_arr[:, root_pos_global].copy() if root_pos_global else np.zeros((raw_arr.shape[0], 3))
    raw_arr[:, root_pos_global] *= float(root_scale)
    after_root = raw_arr[:, root_pos_global].copy() if root_pos_global else np.zeros((raw_arr.shape[0], 3))
    kept = raw_arr[:, keep_global]
    new_frames = [" ".join(f"{x:.6f}" for x in row.tolist()) for row in kept]

    dst.parent.mkdir(parents=True, exist_ok=True)
    out_text = head.rstrip() + "\nMOTION\n" + "\n".join(header_lines) + "\n" + "\n".join(new_frames) + "\n"
    dst.write_text(out_text, encoding="utf-8")

    return {
        "src": str(src),
        "dst": str(dst),
        "accepted": True,
        "nodes": len(layout),
        "old_channels": old_total_channels,
        "new_channels": int(len(keep_global)),
        "root_scale": float(root_scale),
        "root_pos_range_before_l2": float(np.linalg.norm(before_root.max(axis=0) - before_root.min(axis=0))),
        "root_pos_range_after_l2": float(np.linalg.norm(after_root.max(axis=0) - after_root.min(axis=0))),
        "frames": int(raw_arr.shape[0]),
        **audit,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="change")
    ap.add_argument("--out_dir", default="output/change_rot_only_meter_bvh")
    ap.add_argument("--root_scale", type=float, default=0.01)
    ap.add_argument("--max_nonroot_pos_range", type=float, default=1e-3)
    ap.add_argument("--rejected_dir", default="output/change_rejected_bvh")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    rejected_dir = Path(args.rejected_dir) if args.rejected_dir else None
    reports = []
    for p in sorted(in_dir.rglob("*.bvh")):
        rel = p.relative_to(in_dir)
        dst = out_dir / rel
        rep = canonicalize_one(p, dst, args.root_scale, args.max_nonroot_pos_range, rejected_dir)
        reports.append(rep)
        if rep.get("accepted"):
            print("[SAVED]", dst, "old_ch=", rep["old_channels"], "new_ch=", rep["new_channels"], "root_range:", round(rep["root_pos_range_before_l2"], 3), "->", round(rep["root_pos_range_after_l2"], 3))
        else:
            print("[REJECT]", p, rep.get("reason"), "nonroot_max=", rep.get("nonroot_pos_range_max"))
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "canonicalization_audit.json"
    audit_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[AUDIT]", audit_path)
    print("[DONE] accepted:", sum(1 for r in reports if r.get("accepted")), "total:", len(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
