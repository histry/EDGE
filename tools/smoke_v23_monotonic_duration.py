#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from model.v23_monotonic_duration import V23MonotonicDurationNet, warp_motion_so3


def parse_edges(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_len", type=int, default=120)
    parser.add_argument("--duration_edges", default="12,24,37,50,63,76,89")
    args = parser.parse_args()

    batch_size = 4
    motion = torch.randn(batch_size, args.window_len, 151)
    motion[..., :4] = (torch.rand(batch_size, args.window_len, 4) > 0.5).float()
    mask = torch.zeros(batch_size, args.window_len)
    mask[:, args.window_len // 4 : 3 * args.window_len // 4] = 1.0
    condition = torch.rand(batch_size, 17)
    model = V23MonotonicDurationNet(
        condition_dim=17,
        hidden_dim=64,
        dropout=0.1,
        duration_edges=parse_edges(args.duration_edges),
        window_len=args.window_len,
    )
    model.eval()
    with torch.no_grad():
        duration = model.predict_duration(motion, mask, condition, use_hard_duration=False)
        result = model(motion, mask, condition, use_hard_duration=False)
        warped = warp_motion_so3(motion, result["tau"])
    assert result["tau"].shape == (batch_size, args.window_len)
    assert torch.all(result["tau"][:, 1:] >= result["tau"][:, :-1])
    assert torch.allclose(result["tau"][:, 0], torch.zeros(batch_size), atol=1e-6)
    assert torch.allclose(result["tau"][:, -1], torch.ones(batch_size), atol=1e-6)
    assert warped.shape == motion.shape
    assert torch.isfinite(warped).all()
    print("tau:", tuple(result["tau"].shape))
    print("continuous duration bins:", duration["duration_continuous_bin_index"].tolist())
    print("ordinal argmax bins:", duration["duration_ordinal_bin_index"].tolist())
    print("ordinal thresholds:", model.ordinal_head.thresholds().detach().cpu().tolist())
    print("duration frames:", duration["duration_frames"].tolist())
    print("bin confidence:", duration["duration_bin_confidence"].tolist())
    print("edit probability:", duration["edit_probability"].tolist())
    print("[PASS] V23-v2.5 continuous-calibrated gate smoke test")


if __name__ == "__main__":
    main()
