#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import re, json
import numpy as np

def parse_bvh(path):
    text = Path(path).read_text(errors="ignore")
    head, mot = text.split("MOTION", 1)

    names = []
    channel_defs = []
    stack = []

    lines = head.splitlines()
    current = None
    for line in lines:
        m = re.match(r"\s*(ROOT|JOINT)\s+([^\s{]+)", line)
        if m:
            current = m.group(2)
            names.append(current)
            continue
        m = re.match(r"\s*CHANNELS\s+(\d+)\s+(.+)", line)
        if m and current is not None:
            n = int(m.group(1))
            ch = m.group(2).split()
            channel_defs.append((current, ch))
            current = None

    m = re.search(r"Frames:\s*(\d+)", mot)
    frames = int(m.group(1)) if m else None
    frame_lines = []
    start = False
    for line in mot.splitlines():
        if start:
            if line.strip():
                frame_lines.append(line.strip())
        if line.strip().lower().startswith("frame time"):
            start = True

    arr = np.array([[float(x) for x in line.split()] for line in frame_lines], dtype=np.float32)
    return names, channel_defs, arr

reports = []
for p in Path("change").rglob("*.bvh"):
    names, channel_defs, arr = parse_bvh(p)
    cursor = 0
    joint_pos_stats = []
    root_stats = None

    for idx, (name, chs) in enumerate(channel_defs):
        n = len(chs)
        data = arr[:, cursor:cursor+n]
        cursor += n

        pos_idx = [i for i,c in enumerate(chs) if "position" in c.lower()]
        rot_idx = [i for i,c in enumerate(chs) if "rotation" in c.lower()]

        if pos_idx:
            pos = data[:, pos_idx]
            stat = {
                "joint": name,
                "channels": chs,
                "pos_min": pos.min(axis=0).tolist(),
                "pos_max": pos.max(axis=0).tolist(),
                "pos_std": pos.std(axis=0).tolist(),
                "pos_range_l2": float(np.linalg.norm(pos.max(axis=0)-pos.min(axis=0))),
            }
            if idx == 0:
                root_stats = stat
            else:
                joint_pos_stats.append(stat)

    nonroot_pos_ranges = [x["pos_range_l2"] for x in joint_pos_stats]
    reports.append({
        "path": str(p),
        "frames": int(arr.shape[0]),
        "channels_in_file": int(arr.shape[1]),
        "channels_from_hierarchy": int(sum(len(c) for _, c in channel_defs)),
        "num_nodes_with_channels": len(channel_defs),
        "root": names[0] if names else "",
        "root_pos_range_l2": root_stats["pos_range_l2"] if root_stats else None,
        "nonroot_position_nodes": len(joint_pos_stats),
        "nonroot_pos_range_max": float(max(nonroot_pos_ranges)) if nonroot_pos_ranges else 0.0,
        "nonroot_pos_range_p95": float(np.percentile(nonroot_pos_ranges, 95)) if nonroot_pos_ranges else 0.0,
        "nonroot_pos_range_mean": float(np.mean(nonroot_pos_ranges)) if nonroot_pos_ranges else 0.0,
        "worst_nonroot_pos": sorted(joint_pos_stats, key=lambda x: x["pos_range_l2"], reverse=True)[:5],
    })

Path("output").mkdir(exist_ok=True)
out = Path("output/chang_e_bvh_6dof_channel_audit.json")
out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

for r in reports:
    print("\n", r["path"])
    print("frames:", r["frames"], "channels:", r["channels_in_file"])
    print("nonroot_position_nodes:", r["nonroot_position_nodes"])
    print("root_pos_range_l2:", r["root_pos_range_l2"])
    print("nonroot_pos_range_max:", r["nonroot_pos_range_max"])
    print("nonroot_pos_range_p95:", r["nonroot_pos_range_p95"])
    print("worst:", [(x["joint"], round(x["pos_range_l2"], 4)) for x in r["worst_nonroot_pos"]])

print("\nsaved:", out)
