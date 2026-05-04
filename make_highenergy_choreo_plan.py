"""Create a high-energy demo choreography plan from an existing plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", type=int, default=115)
    ap.add_argument("--prompt", default="中高能量敦煌舞展示，飞天手势展开，身体扭转，姿态变化明显，重心稳定")
    ap.add_argument("--music_caption", default="情绪上扬，节奏增强，中高能量，适合姿态展示")
    ap.add_argument("--energy_target", type=float, default=0.80)
    ap.add_argument("--tension_target", type=float, default=0.80)
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        plan = json.load(f)

    segs = plan.get("segments", [])
    if not segs:
        raise RuntimeError("input plan has no segments")

    chosen = None
    for seg in segs:
        if int(seg.get("start", 0)) <= args.frame <= int(seg.get("end", 10**9)):
            chosen = seg
            break
    if chosen is None:
        chosen = min(segs, key=lambda s: abs(int(s.get("center", 0)) - args.frame))

    chosen["music_caption"] = args.music_caption
    chosen["motion_prompt"] = args.prompt
    chosen["query_text"] = f"{args.music_caption}，{args.prompt}"
    chosen["energy_target"] = float(args.energy_target)
    chosen["tension_target"] = float(args.tension_target)
    chosen["phase"] = "attack"
    chosen["min_expressiveness"] = max(float(chosen.get("min_expressiveness", 0.0)), 0.50)

    plan["planner"] = str(plan.get("planner", "music_choreo_planner")) + "+demo_highenergy"
    plan["demo_highenergy_edit"] = {
        "frame": int(args.frame),
        "segment_id": chosen.get("id", None),
        "prompt": args.prompt,
        "energy_target": float(args.energy_target),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    print("✅ saved:", args.out)
    print("edited segment:", chosen.get("id"), chosen.get("start"), chosen.get("end"))


if __name__ == "__main__":
    main()
