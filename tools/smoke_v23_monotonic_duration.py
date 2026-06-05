#!/usr/bin/env python3
from __future__ import annotations
import torch
from model.v23_monotonic_duration import V23MonotonicDurationNet, warp_motion_so3


def main():
    b, t = 2, 72
    motion = torch.randn(b, t, 151)
    motion[..., :4] = (torch.rand(b, t, 4) > 0.5).float()
    mask = torch.zeros(b, t)
    mask[:, 20:50] = 1.0
    condition = torch.rand(b, 17)
    model = V23MonotonicDurationNet(condition_dim=17, hidden_dim=128)
    out = model(motion, mask, condition)
    tau = out["tau"]
    assert tau.shape == (b, t)
    assert torch.all(tau[:, 1:] >= tau[:, :-1])
    assert torch.allclose(tau[:, 0], torch.zeros(b), atol=1e-6)
    assert torch.allclose(tau[:, -1], torch.ones(b), atol=1e-6)
    warped = warp_motion_so3(motion, tau)
    assert warped.shape == motion.shape
    print("tau:", tau.shape, "duration:", out["duration_ratio"].shape)
    print("warped:", warped.shape)
    print("[PASS] V23 monotonic duration smoke test")


if __name__ == "__main__":
    main()
