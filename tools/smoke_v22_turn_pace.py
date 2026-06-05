#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal environment smoke test for V22."""
from __future__ import annotations

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_rotation_6d

from model.v22_turn_pace import V22TurnPaceRefiner
from tools.v22_turn_utils import detect_turn_events, project_rot6d_np


def yaw_matrix(angle: np.ndarray) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    out = np.zeros((len(angle), 3, 3), dtype=np.float32)
    out[:, 0, 0] = c
    out[:, 0, 2] = s
    out[:, 1, 1] = 1.0
    out[:, 2, 0] = -s
    out[:, 2, 2] = c
    return out


def main() -> None:
    t = 72
    motion = np.zeros((t, 151), dtype=np.float32)
    identity = np.eye(3, dtype=np.float32)[None, None].repeat(t, axis=0).repeat(24, axis=1)
    rot6d = matrix_to_rotation_6d(torch.from_numpy(identity)).numpy().reshape(t, 144)
    motion[:, 7:] = rot6d
    angle = np.zeros((t,), dtype=np.float32)
    angle[20:35] = np.linspace(0.0, np.pi / 2.0, 15)
    angle[35:] = np.pi / 2.0
    root6d = matrix_to_rotation_6d(torch.from_numpy(yaw_matrix(angle))).numpy()
    motion[:, 7:13] = root6d
    motion = project_rot6d_np(motion)

    events = detect_turn_events(motion, min_peak_dps=20.0)
    print("detected turns:", [event.to_dict() for event in events])
    if not events:
        raise RuntimeError("Synthetic turn was not detected")

    model = V22TurnPaceRefiner(condition_dim=17)
    mask = torch.ones((1, t), dtype=torch.float32)
    cond = torch.zeros((1, 17), dtype=torch.float32)
    output = model(torch.from_numpy(motion[None]), mask, cond)
    print("model output:", tuple(output.shape), "finite=", bool(torch.isfinite(output).all()))
    if output.shape != (1, t, 151) or not torch.isfinite(output).all():
        raise RuntimeError("V22 model smoke test failed")
    print("[PASS] V22 turn-aware pace environment is healthy")


if __name__ == "__main__":
    main()
