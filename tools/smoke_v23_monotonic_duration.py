#!/usr/bin/env python3
from __future__ import annotations

import torch

from model.v23_monotonic_duration import V23MonotonicDurationNet, warp_motion_so3


def main() -> None:
    batch_size, time_steps = 4, 72
    motion = torch.randn(batch_size, time_steps, 151)
    motion[..., :4] = (torch.rand(batch_size, time_steps, 4) > 0.5).float()
    mask = torch.zeros(batch_size, time_steps)
    mask[:, 22:43] = 1.0
    condition = torch.rand(batch_size, 17)
    model = V23MonotonicDurationNet(
        condition_dim=17,
        hidden_dim=128,
        duration_min_frames=10,
        duration_max_frames=56,
        window_len=time_steps,
    )
    result = model(motion, mask, condition)
    tau = result["tau"]
    assert tau.shape == (batch_size, time_steps)
    assert torch.all(tau[:, 1:] >= tau[:, :-1])
    assert torch.allclose(tau[:, 0], torch.zeros(batch_size), atol=1e-6)
    assert torch.allclose(tau[:, -1], torch.ones(batch_size), atol=1e-6)
    assert torch.all(result["duration_frames"] >= 10.0 - 1e-4)
    assert torch.all(result["duration_frames"] <= 56.0 + 1e-4)
    warped = warp_motion_so3(motion, tau)
    assert warped.shape == motion.shape
    assert torch.isfinite(warped).all()
    print("tau:", tuple(tau.shape))
    print("duration range:", float(result["duration_frames"].min().detach()), float(result["duration_frames"].max().detach()))
    print("edit probability:", result["edit_probability"].detach().tolist())
    print("[PASS] V23-v2 smoke test")


if __name__ == "__main__":
    main()
