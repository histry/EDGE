#!/usr/bin/env python
"""Train the model-internal turn-aware output adapter on pseudo targets.

This trains the same TurnEventOutputAdapter class loaded by
turn_event_model_adapter_patch.py inside DanceDecoder.  It is a low-risk
adapter distillation stage: it does not update the main EDGE checkpoint, but
it produces a checkpoint that can be loaded by the model-internal runtime patch
with EDGE_TURN_EVENT_ADAPTER_CKPT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from turn_aware_event_utils import event_feature_matrix, parse_trajectory_string, EVENT_DIM
from turn_event_model_adapter_patch import (
    TurnEventOutputAdapter,
    LOWER_JOINTS,
    TORSO_JOINTS,
    UPPER_JOINTS,
    ROT_START,
    ROT_DIM,
    ROOT_X_IDX,
    ROOT_Z_IDX,
)


def rot_indices(joints):
    out = []
    for j in joints:
        start = ROT_START + ROT_DIM * int(j)
        out.extend(range(start, start + ROT_DIM))
    return out


def load_motion(path: str, device: str) -> torch.Tensor:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected [T,D] motion, got {arr.shape}: {path}")
    return torch.from_numpy(arr).to(device)


def ensure_len(x: torch.Tensor, T: int) -> torch.Tensor:
    if x.shape[0] == T:
        return x
    return F.interpolate(x.T[None], size=T, mode="linear", align_corners=False)[0].T


def frame_energy(x: torch.Tensor, idxs: List[int]) -> torch.Tensor:
    if not idxs:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    y = x[:, idxs]
    vel = torch.zeros_like(y)
    vel[1:] = y[1:] - y[:-1]
    vel[0] = vel[1]
    return torch.linalg.norm(vel, dim=-1)


def weighted_mse(pred, target, weight):
    while weight.ndim < pred.ndim:
        weight = weight.unsqueeze(-1)
    return ((pred - target) ** 2 * weight).mean()


def corr_loss(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    if weight is not None:
        a = a * weight
        b = b * weight
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a * a).mean() * (b * b).mean()).clamp_min(1e-6)
    return 1.0 - ((a * b).mean() / denom).clamp(-1, 1)


def event_weighted_loss(pred, target, anchor, event, cfg) -> Dict[str, torch.Tensor]:
    # [T,D]
    turn_gate = event[:, 6].clamp(0, 1)
    support_gate = event[:, 7].clamp(0, 1)
    expressive_gate = event[:, 8].clamp(0, 1)
    speed_gate = event[:, 10].clamp(0, 1)
    any_event = torch.clamp(torch.maximum(torch.maximum(turn_gate, support_gate), expressive_gate), 0, 1)

    lower_idx = rot_indices(LOWER_JOINTS)
    torso_idx = rot_indices(TORSO_JOINTS)
    upper_idx = rot_indices(UPPER_JOINTS)
    contact_idx = list(range(0, 4))
    root_idx = [ROOT_X_IDX, ROOT_Z_IDX]

    # General anchor-preserve outside events.
    outside = (1.0 - any_event).clamp(0, 1)
    loss_anchor = weighted_mse(pred, anchor, outside) * cfg.anchor_weight

    # Event-aware body losses.
    loss_lower = weighted_mse(pred[:, lower_idx], target[:, lower_idx], 1.0 + cfg.support_weight * support_gate + 0.4 * speed_gate)
    loss_contact = weighted_mse(pred[:, contact_idx], target[:, contact_idx], 1.0 + cfg.contact_weight * support_gate)
    loss_torso = weighted_mse(pred[:, torso_idx], target[:, torso_idx], 1.0 + cfg.turn_weight * turn_gate + cfg.expressive_weight * expressive_gate)
    loss_upper = weighted_mse(pred[:, upper_idx], target[:, upper_idx], 1.0 + cfg.turn_weight * turn_gate + cfg.expressive_weight * expressive_gate)

    # Hard protect root xz.
    loss_root = F.mse_loss(pred[:, root_idx], anchor[:, root_idx]) * cfg.root_weight

    # Keep event activity from being smoothed away.
    lower_p = frame_energy(pred, lower_idx)
    lower_t = frame_energy(target, lower_idx).detach()
    expr_p = 0.5 * (frame_energy(pred, torso_idx) + frame_energy(pred, upper_idx))
    expr_t = 0.5 * (frame_energy(target, torso_idx) + frame_energy(target, upper_idx)).detach()
    keep_lower = (torch.relu(cfg.keep_ratio * lower_t - lower_p) ** 2 * support_gate).mean()
    keep_expr = (torch.relu(cfg.keep_ratio * expr_t - expr_p) ** 2 * torch.maximum(turn_gate, expressive_gate)).mean()

    # Sync regularizers encourage support/expression to align around event windows.
    sync = corr_loss(lower_p, expr_p, any_event) * cfg.sync_weight
    speed_sync = corr_loss(event[:, 2].detach(), lower_p, speed_gate) * cfg.speed_sync_weight

    loss = (
        loss_anchor
        + loss_lower * cfg.lower_weight
        + loss_contact * cfg.contact_weight
        + loss_torso * cfg.torso_weight
        + loss_upper * cfg.upper_weight
        + loss_root
        + keep_lower * cfg.keep_weight
        + keep_expr * cfg.keep_weight
        + sync
        + speed_sync
    )
    return {
        "loss": loss,
        "anchor": loss_anchor.detach(),
        "lower": loss_lower.detach(),
        "contact": loss_contact.detach(),
        "torso": loss_torso.detach(),
        "upper": loss_upper.detach(),
        "root": loss_root.detach(),
        "keep_lower": keep_lower.detach(),
        "keep_expr": keep_expr.detach(),
        "sync": sync.detach(),
        "speed_sync": speed_sync.detach(),
    }


class LossCfg:
    anchor_weight: float = 0.25
    root_weight: float = 20.0
    lower_weight: float = 1.00
    torso_weight: float = 1.25
    upper_weight: float = 1.25
    contact_weight: float = 1.25
    support_weight: float = 4.0
    turn_weight: float = 4.0
    expressive_weight: float = 4.0
    keep_ratio: float = 0.80
    keep_weight: float = 0.75
    sync_weight: float = 0.15
    speed_sync_weight: float = 0.10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="output/v13_functional_base/dhw4_expr_mobile_base.npy")
    parser.add_argument("--anchor", default="output/v13_turn_event_hybrid/dhw4_turn_event_v13.npy")
    parser.add_argument("--trajectory", default="0,0;1.2,0.8;-1.2,1.6;1.2,2.4;0,3.2")
    parser.add_argument("--targets", nargs="+", default=[
        "output/v13_frame_sweep_hybrid/dhw4_v13_f35_60_85_110_135_mild.npy",
        "output/v13_frame_sweep_hybrid/dhw4_v13_f40_65_90_115_140_mild.npy",
        "output/v13_turn_event_hybrid/dhw4_turn_event_v13.npy",
        "output/v13_functional_hybrid_sweep/dhw4_v13_mild.npy",
    ])
    parser.add_argument("--weights", nargs="+", type=float, default=[1.30, 1.00, 1.20, 0.80])
    parser.add_argument("--out", default="runs/turn_event_internal_adapter/turn_event_internal_adapter.pt")
    parser.add_argument("--epochs", type=int, default=int(float(__import__('os').environ.get("EDGE_TURN_INTERNAL_EPOCHS", 600))))
    parser.add_argument("--lr", type=float, default=float(__import__('os').environ.get("EDGE_TURN_INTERNAL_LR", 7e-4)))
    parser.add_argument("--hidden", type=int, default=int(float(__import__('os').environ.get("EDGE_TURN_INTERNAL_HIDDEN", 256))))
    parser.add_argument("--max_delta", type=float, default=float(__import__('os').environ.get("EDGE_TURN_INTERNAL_MAX_DELTA", 0.14)))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    base = load_motion(args.base, device)
    anchor = load_motion(args.anchor, device)
    T, D = base.shape
    anchor = ensure_len(anchor, T)
    traj = parse_trajectory_string(args.trajectory, seq_len=T)
    event_np, report = event_feature_matrix(traj)
    event = torch.from_numpy(event_np).to(device)

    samples = []
    for i, target_path in enumerate(args.targets):
        path = Path(target_path)
        if not path.exists():
            print(f"skip missing target: {path}")
            continue
        target = ensure_len(load_motion(str(path), device), T)
        w = args.weights[i] if i < len(args.weights) else 1.0
        samples.append({"target": target, "weight": float(w), "path": str(path)})
    if not samples:
        raise SystemExit("No training targets found")

    adapter = TurnEventOutputAdapter(nfeats=D, event_dim=EVENT_DIM, hidden=args.hidden, max_delta=args.max_delta).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    cfg = LossCfg()

    print(f"training internal adapter: samples={len(samples)} T={T} D={D} max_delta={args.max_delta} lr={args.lr}")
    print("event report:", report.to_dict())
    for ep in range(1, args.epochs + 1):
        total = torch.tensor(0.0, device=device)
        logs: Dict[str, float] = {}
        for s in samples:
            pred = adapter(anchor.unsqueeze(0), event.unsqueeze(0))[0]
            parts = event_weighted_loss(pred, s["target"], anchor, event, cfg)
            loss = parts["loss"] * float(s["weight"])
            total = total + loss
            for k, v in parts.items():
                if k != "loss":
                    logs[k] = logs.get(k, 0.0) + float(v.detach().cpu())
        total = total / max(1, len(samples))
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            msg = f"epoch={ep:04d} loss={float(total.detach().cpu()):.6f}"
            if ep % 50 == 0 or ep == args.epochs:
                msg += " " + " ".join(f"{k}={v/len(samples):.4f}" for k, v in sorted(logs.items())[:6])
            print(msg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": adapter.state_dict(),
        "event_dim": EVENT_DIM,
        "hidden": args.hidden,
        "max_delta": args.max_delta,
        "nfeats": D,
        "targets": [s["path"] for s in samples],
        "weights": [s["weight"] for s in samples],
        "event_report": report.to_dict(),
    }, str(out))
    print(f"✅ saved internal adapter checkpoint: {out}")


if __name__ == "__main__":
    main()
