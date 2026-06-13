#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small CPU/GPU smoke test for the installed V34 boundary patch."""
from __future__ import annotations

import json

import numpy as np
import torch

from tools.v32_contact_inr import V32ContactINRSystem, V32INRConfig
from tools.v32_transition_quality import transition_risk
from tools.v34_boundary_dynamics import (
    apply_exit_handshake_np,
    boundary_state_from_context_np,
    boundary_state_to_torch,
    make_v34_transition_np,
)


def identity_motion(length: int, phase: float) -> np.ndarray:
    motion = np.zeros((length, 151), np.float32)
    identity = np.tile(
        np.asarray([1, 0, 0, 0, 1, 0], np.float32), 24
    )
    motion[:, 7:] = identity
    motion[:, 5] = 0.02 * np.sin(
        np.linspace(phase, phase + 1.0, length, dtype=np.float32)
    )
    motion[:, :4] = 0.4
    return motion


def main() -> None:
    previous = identity_motion(16, 0.0)
    following = identity_motion(24, 1.2)
    transition = make_v34_transition_np(previous, following, 12)
    if transition.shape != (12, 151) or not np.isfinite(transition).all():
        raise RuntimeError("Invalid septic transition")

    handshaken, handshake = apply_exit_handshake_np(
        transition, following, frames=8
    )
    risk = transition_risk(previous, transition, handshaken)
    required = {
        "boundary_joint_jerk_max",
        "exit_rotation_step_rad",
        "exit_fk_jump",
        "gate_contact_rate",
    }
    missing = required.difference(risk)
    if missing:
        raise RuntimeError(f"Risk keys missing: {sorted(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = V32INRConfig(
        latent_dim=8,
        condition_dim=16,
        encoder_hidden=16,
        inr_hidden=16,
        inr_layers=2,
        fourier_bands=2,
        diffusion_hidden=16,
        diffusion_blocks=2,
        diffusion_steps=8,
        max_len=12,
    )
    system = V32ContactINRSystem(config).to(device)
    start = torch.from_numpy(previous[-1:]).to(device)
    end = torch.from_numpy(following[:1]).to(device)
    start_velocity = torch.from_numpy(previous[-1] - previous[-2]).to(
        device
    ).reshape(1, -1)
    end_velocity = torch.from_numpy(following[1] - following[0]).to(
        device
    ).reshape(1, -1)
    music = torch.zeros((1, 12), device=device)
    length = torch.tensor([[12.0]], device=device)
    condition = system.condition(
        start, end, start_velocity, end_velocity, music, length
    )
    state = boundary_state_to_torch(
        boundary_state_from_context_np(previous, following), device
    )
    coordinate = torch.linspace(
        1.0 / 13.0, 12.0 / 13.0, 12, device=device
    ).reshape(1, 12, 1)
    latent = torch.zeros((1, 8), device=device, requires_grad=True)
    output = system.decode(
        latent,
        start,
        end,
        start_velocity,
        end_velocity,
        condition,
        coordinate,
        length,
        boundary_state=state,
    )
    output.square().mean().backward()
    if latent.grad is None or not torch.isfinite(latent.grad).all():
        raise RuntimeError("V34 decode backward failed")

    print(json.dumps({
        "status": "PASS",
        "device": str(device),
        "transition_shape": list(transition.shape),
        "handshake": handshake,
        "risk": {key: risk[key] for key in sorted(required)},
        "latent_grad_norm": float(torch.linalg.norm(latent.grad).item()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
