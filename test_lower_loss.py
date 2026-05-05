#!/usr/bin/env python3
"""Sanity test for V7 explicit lower velocity loss.

Run from EDGE repo root after replacing trajectory_native_control.py:

    EDGE_EXPLICIT_LOWER_VEL_LOSS_WEIGHT=0.1 python test_lower_loss.py

This test uses a fake diffusion object to avoid loading SMPL/checkpoints.
It verifies:
1) explicit lower loss is > 0 when root moves but feet stay fixed relative to pelvis;
2) the loss has a valid gradient;
3) disabling the env weight returns 0.
"""

from __future__ import annotations

import os
import torch

from trajectory_native_control import _edge_explicit_lower_velocity_loss


class FakeDiffusion:
    normalizer = None
    root_x_idx = 4
    root_y_idx = 5
    root_z_idx = 6
    root_slice = slice(4, 7)

    def _fk_positions(self, motion):
        """Return fake joints with feet fixed relative to pelvis.

        motion: [B,T,151]
        joints: [B,T,24,3]
        """
        B, T, _ = motion.shape
        joints = torch.zeros(B, T, 24, 3, device=motion.device, dtype=motion.dtype)

        root = motion[:, :, self.root_slice]
        joints[:] = root[:, :, None, :]

        offsets = torch.tensor(
            [
                [0.10, -0.90, 0.10],
                [-0.10, -0.90, 0.10],
                [0.10, -0.90, -0.10],
                [-0.10, -0.90, -0.10],
            ],
            device=motion.device,
            dtype=motion.dtype,
        )
        joints[:, :, [7, 8, 10, 11], :] = root[:, :, None, :] + offsets[None, None, :, :]
        return joints


def main():
    os.environ["EDGE_EXPLICIT_LOWER_VEL_LOSS_WEIGHT"] = os.environ.get(
        "EDGE_EXPLICIT_LOWER_VEL_LOSS_WEIGHT",
        "0.1",
    )
    os.environ["EDGE_EXPLICIT_LOWER_RELVEL_RATIO"] = os.environ.get(
        "EDGE_EXPLICIT_LOWER_RELVEL_RATIO",
        "0.3",
    )
    os.environ["EDGE_EXPLICIT_LOWER_MIN_MOTION"] = os.environ.get(
        "EDGE_EXPLICIT_LOWER_MIN_MOTION",
        "0.01",
    )
    os.environ["EDGE_EXPLICIT_LOWER_ROOT_SPEED_THRESHOLD"] = os.environ.get(
        "EDGE_EXPLICIT_LOWER_ROOT_SPEED_THRESHOLD",
        "0.001",
    )
    os.environ["EDGE_EXPLICIT_LOWER_STRICT"] = "1"

    B, T, C = 2, 100, 151
    x0 = torch.zeros(B, T, C, requires_grad=True)

    # Root X moves; feet are fixed relative to pelvis by FakeDiffusion._fk_positions.
    with torch.no_grad():
        x0[:, :, 4] = torch.linspace(0, 5, T)

    diffusion = FakeDiffusion()
    loss = _edge_explicit_lower_velocity_loss(diffusion, x0)

    assert torch.isfinite(loss), "Loss is NaN/Inf"
    assert loss.item() > 0.0, f"Loss should be > 0, got {loss.item()}"

    loss.backward()
    assert x0.grad is not None, "Loss has no gradient"
    assert torch.isfinite(x0.grad).all(), "Gradient has NaN/Inf"

    # Disable env switch and ensure the function gates off.
    os.environ["EDGE_EXPLICIT_LOWER_VEL_LOSS_WEIGHT"] = "0.0"
    off_loss = _edge_explicit_lower_velocity_loss(diffusion, x0.detach())
    assert float(off_loss.item()) == 0.0, f"Disabled loss should be 0, got {off_loss.item()}"

    print(
        "✅ test_lower_loss passed | "
        f"loss={loss.item():.6f}, grad_norm={x0.grad.norm().item():.6f}, "
        f"off_loss={off_loss.item():.6f}"
    )


if __name__ == "__main__":
    main()
