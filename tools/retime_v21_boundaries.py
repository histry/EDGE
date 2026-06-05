#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


CONTACT = slice(0, 4)
ROOT = slice(4, 7)
ROTATION = slice(7, 151)


def velocity_limited_phase(
    t: torch.Tensor,
    accel_ratio: float = 0.12,
) -> torch.Tensor:
    """低峰值梯形速度时间曲线。

    与 minimum-jerk 相比：
    - 仍然从零速度开始并以零速度结束；
    - 中间采用近似恒定速度；
    - 峰值相位速度更低；
    - 更适合解决动作过渡“赶得太快”的问题。
    """
    a = float(max(0.05, min(0.30, accel_ratio)))
    vmax = 1.0 / (1.0 - a)

    phase = torch.empty_like(t)

    accelerate = t < a
    decelerate = t > (1.0 - a)
    constant = ~(accelerate | decelerate)

    phase[accelerate] = (
        0.5
        * vmax
        / a
        * t[accelerate] ** 2
    )

    phase[constant] = (
        vmax
        * (
            t[constant]
            - 0.5 * a
        )
    )

    remaining = 1.0 - t[decelerate]

    phase[decelerate] = (
        1.0
        - 0.5
        * vmax
        / a
        * remaining ** 2
    )

    return torch.clamp(phase, 0.0, 1.0)


def rotation_path(
    start_rotation: torch.Tensor,
    end_rotation: torch.Tensor,
    progress: torch.Tensor,
) -> torch.Tensor:
    """在 SO(3) 上生成端点姿态之间的测地路径。"""

    relative = torch.matmul(
        start_rotation.transpose(-1, -2),
        end_rotation,
    )

    axis_angle = matrix_to_axis_angle(relative)

    delta = axis_angle_to_matrix(
        progress[:, None, None]
        * axis_angle[None]
    )

    return torch.matmul(
        start_rotation[None],
        delta,
    )


def blend_rotations(
    original: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """将原始姿态沿 SO(3) 插值到目标过渡姿态。"""

    relative = torch.matmul(
        original.transpose(-1, -2),
        target,
    )

    axis_angle = matrix_to_axis_angle(relative)

    delta = axis_angle_to_matrix(
        axis_angle * weight[:, None, None]
    )

    return torch.matmul(original, delta)


def speed_metrics(
    motion: np.ndarray,
    boundary: int,
    radius: int = 6,
) -> dict:
    pose = motion[:, ROTATION]

    velocity = np.mean(
        np.abs(np.diff(pose, axis=0)),
        axis=1,
    )

    acceleration = np.mean(
        np.abs(np.diff(pose, n=2, axis=0)),
        axis=1,
    )

    start = max(0, boundary - radius)
    end = min(len(velocity), boundary + radius)

    velocity_before = pose[boundary - 1] - pose[boundary - 2]
    velocity_after = pose[boundary + 1] - pose[boundary]

    return {
        "local_velocity": float(np.mean(velocity[start:end])),
        "local_p95_velocity": float(
            np.percentile(velocity[start:end], 95)
        ),
        "velocity_jump": float(
            np.mean(np.abs(velocity_after - velocity_before))
        ),
        "local_acceleration": float(
            np.mean(
                acceleration[
                    max(0, start - 1):
                    min(len(acceleration), end)
                ]
            )
        ),
    }


def process_boundary(
    motion: np.ndarray,
    boundary: int,
    window: int,
    strength: float,
    keep_root_xz: bool,
) -> tuple[np.ndarray, dict]:

    output = motion.copy()

    half = window // 2
    left = max(1, boundary - half)
    right = min(len(output) - 2, boundary + half)

    count = right - left + 1

    if count < 12:
        return output, {
            "boundary": boundary,
            "applied": False,
            "reason": "window_too_short",
        }

    before = speed_metrics(output, boundary)

    segment = torch.from_numpy(
        output[left:right + 1]
    ).float()

    t = torch.linspace(0.0, 1.0, count)
    progress = velocity_limited_phase(t, accel_ratio=0.12)

    # 窗口中央处理最强，端点保持原动作
    blend_weight = (
        torch.sin(torch.pi * t) ** 2
    ) * float(strength)

    # ---------------------------------------------------------
    # Root minimum-jerk
    # ---------------------------------------------------------
    original_root = segment[:, ROOT]

    target_root = (
        original_root[0:1]
        + progress[:, None]
        * (
            original_root[-1:]
            - original_root[0:1]
        )
    )

    blended_root = (
        original_root * (1.0 - blend_weight[:, None])
        + target_root * blend_weight[:, None]
    )

    if keep_root_xz:
        blended_root[:, 0] = original_root[:, 0]
        blended_root[:, 2] = original_root[:, 2]

    # ---------------------------------------------------------
    # 24关节旋转 minimum-jerk
    # ---------------------------------------------------------
    original_rot6d = segment[:, ROTATION].reshape(
        count, 24, 6
    )

    original_rotation = rotation_6d_to_matrix(
        original_rot6d
    )

    target_rotation = rotation_path(
        original_rotation[0],
        original_rotation[-1],
        progress,
    )

    blended_rotation = blend_rotations(
        original_rotation,
        target_rotation,
        blend_weight,
    )

    blended_rot6d = matrix_to_rotation_6d(
        blended_rotation
    ).reshape(count, 144)

    processed = segment.clone()

    processed[:, ROOT] = blended_root
    processed[:, ROTATION] = blended_rot6d

    # Contact 保留原始二值标签
    processed[:, CONTACT] = segment[:, CONTACT]

    output[left:right + 1] = (
        processed.numpy().astype(np.float32)
    )

    after = speed_metrics(output, boundary)

    return output, {
        "boundary": int(boundary),
        "applied": True,
        "window": int(count),
        "left": int(left),
        "right": int(right),
        "before": before,
        "after": after,
        "velocity_ratio": float(
            after["local_velocity"]
            / max(before["local_velocity"], 1e-8)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--window", type=int, default=36)
    parser.add_argument("--strength", type=float, default=0.92)
    parser.add_argument("--keep_root_xz", type=int, default=1)
    parser.add_argument("--suffix", default="retimed")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    summary = []

    for report_path in sorted(
        run_dir.glob("*_v21.schedule_report.json")
    ):
        name = report_path.name.replace(
            "_v21.schedule_report.json", ""
        )

        motion_path = run_dir / f"{name}_v21.npy"

        if not motion_path.is_file():
            continue

        raw = np.load(
            motion_path,
            allow_pickle=True,
        ).astype(np.float32)

        had_batch = raw.ndim == 3
        motion = raw[0] if had_batch else raw

        report = json.loads(
            report_path.read_text(encoding="utf-8")
        )

        boundaries = [
            int(value)
            for value in report.get("boundaries", [])
            if 2 <= int(value) <= len(motion) - 2
        ]

        output = motion.copy()
        boundary_results = []

        for boundary in boundaries:
            output, result = process_boundary(
                output,
                boundary=boundary,
                window=args.window,
                strength=args.strength,
                keep_root_xz=bool(args.keep_root_xz),
            )

            boundary_results.append(result)

        output_path = (
            run_dir
            / f"{name}_v21_{args.suffix}.npy"
        )

        np.save(
            output_path,
            output[None] if had_batch else output,
        )

        summary.append({
            "name": name,
            "source": str(motion_path),
            "output": str(output_path),
            "boundaries": boundary_results,
        })

        print("\n", name)

        for result in boundary_results:
            if not result.get("applied"):
                print(result)
                continue

            print(
                "boundary=", result["boundary"],
                "window=", result["window"],
                "velocity=",
                round(
                    result["before"]["local_velocity"], 6
                ),
                "->",
                round(
                    result["after"]["local_velocity"], 6
                ),
                "jump=",
                round(
                    result["before"]["velocity_jump"], 6
                ),
                "->",
                round(
                    result["after"]["velocity_jump"], 6
                ),
            )

    output_report = (
        run_dir
        / "V21_BOUNDARY_RETIMING_REPORT.json"
    )

    output_report.write_text(
        json.dumps(
            {
                "window": args.window,
                "strength": args.strength,
                "results": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nsaved:", output_report)


if __name__ == "__main__":
    main()
