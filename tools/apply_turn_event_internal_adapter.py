#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from turn_aware_event_utils import EVENT_DIM, event_feature_matrix, parse_trajectory_string
from turn_event_model_adapter_patch import TurnEventOutputAdapter, make_motion_feature_gate


def load_motion(path):
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", default="output/v13_turn_event_hybrid/dhw4_turn_event_v13.npy")
    parser.add_argument("--ckpt", default="runs/turn_event_internal_adapter/turn_event_internal_adapter.pt")
    parser.add_argument("--trajectory", default="0,0;1.2,0.8;-1.2,1.6;1.2,2.4;0,3.2")
    parser.add_argument("--out", default="output/v13_turn_event_hybrid/dhw4_turn_event_internal_adapter.npy")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    motion_np = load_motion(args.motion)
    T, D = motion_np.shape
    payload = torch.load(args.ckpt, map_location="cpu")
    adapter = TurnEventOutputAdapter(
        nfeats=int(payload.get("nfeats", D)),
        event_dim=int(payload.get("event_dim", EVENT_DIM)),
        hidden=int(payload.get("hidden", 256)),
        max_delta=float(payload.get("max_delta", 0.12)),
    ).to(args.device)
    adapter.load_state_dict(payload.get("state_dict", payload), strict=False)
    adapter.eval()

    traj = parse_trajectory_string(args.trajectory, seq_len=T)
    event_np, report = event_feature_matrix(traj)
    motion = torch.from_numpy(motion_np).to(args.device).unsqueeze(0)
    event = torch.from_numpy(event_np).to(args.device).unsqueeze(0)
    gate = make_motion_feature_gate(D, motion.device, motion.dtype, prefix="EDGE_TURN_EVENT")
    with torch.no_grad():
        out = adapter(motion, event, gate=gate)[0]
        # Always preserve root X/Z for safe diagnostic application.
        out[:, 4] = motion[0, :, 4]
        out[:, 6] = motion[0, :, 6]
    out_np = out.detach().cpu().numpy().astype(np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out_np)
    Path(args.out).with_suffix(".json").write_text(__import__('json').dumps({
        "input": args.motion,
        "ckpt": args.ckpt,
        "out": args.out,
        "event_report": report.to_dict(),
    }, indent=2, ensure_ascii=False))
    print(f"✅ saved internal-adapter motion: {args.out}")


if __name__ == "__main__":
    main()
