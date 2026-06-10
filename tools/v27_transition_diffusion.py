#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible safe transition entry point for EDGE V31.

The historical module path is retained because the V26 scheduler imports it.
V31 always constructs a deterministic C2 SO(3) baseline, samples several
band-limited coefficient candidates, evaluates each candidate against the
baseline, and falls back whenever learned generation is not demonstrably safe.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from tools.v29_motion_geometry import MOTION_DIM
from tools.v31_bandlimited_transition import (
    V31TransitionModel,
    config_from_dict,
    decode_coefficients,
    linear_beta_schedule,
    selected_timesteps,
)
from tools.v31_transition_quality import (
    accept_candidate,
    transition_risk,
)

_RISK_PREVIOUS: np.ndarray | None = None
_RISK_FOLLOWING: np.ndarray | None = None


def set_transition_risk_context(
    previous: np.ndarray,
    following: np.ndarray,
) -> None:
    """Set exact neighbouring contents for the next sequential transition."""
    global _RISK_PREVIOUS, _RISK_FOLLOWING
    _RISK_PREVIOUS = np.asarray(previous, np.float32).copy()
    _RISK_FOLLOWING = np.asarray(following, np.float32).copy()


def _consume_transition_risk_context(
    start: np.ndarray,
    end: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    global _RISK_PREVIOUS, _RISK_FOLLOWING
    previous = (
        _RISK_PREVIOUS
        if _RISK_PREVIOUS is not None
        else np.stack([start, start], axis=0)
    )
    following = (
        _RISK_FOLLOWING
        if _RISK_FOLLOWING is not None
        else np.stack([end, end], axis=0)
    )
    _RISK_PREVIOUS = None
    _RISK_FOLLOWING = None
    return previous, following


def load_transition_diffusion(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> Dict[str, Any] | None:
    if not path:
        return None
    checkpoint_path = Path(str(path))
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Transition checkpoint not found: {checkpoint_path}")
    device = torch.device(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    config_values = dict(checkpoint.get("config", {}))
    architecture = str(config_values.get("architecture", ""))
    if architecture != "v31_bandlimited_so3_coefficient_diffusion":
        raise RuntimeError(
            f"Checkpoint architecture={architecture!r} is not V31. "
            "Retrain with the V31 train_v27_transition_diffusion.py."
        )
    model_config = config_from_dict(
        dict(config_values.get("model", config_values))
    )
    model = V31TransitionModel(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    ema = checkpoint.get("ema_model")
    if ema is not None:
        model.load_state_dict(ema)
    model.eval()

    def tensor(name: str) -> torch.Tensor:
        if name not in checkpoint:
            raise RuntimeError(f"V31 checkpoint missing {name}")
        return torch.as_tensor(
            checkpoint[name], device=device, dtype=torch.float32
        )

    return {
        "architecture": architecture,
        "model": model,
        "config": config_values,
        "path": str(checkpoint_path),
        "device": device,
        "coefficient_mean": tensor("coefficient_mean").reshape(1, -1),
        "pca_components": tensor("pca_components"),
        "score_mean": tensor("score_mean").reshape(1, -1),
        "score_std": tensor("score_std").reshape(1, -1).clamp_min(1e-5),
        "coefficient_abs_limit": tensor("coefficient_abs_limit").reshape(1, -1),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "epoch": checkpoint.get("epoch"),
    }


def _seed(start: np.ndarray, end: np.ndarray, length: int, candidate: int) -> int:
    base = int(os.getenv("V31_TRANSITION_SEED", "20260610"))
    signature = int(np.round(
        np.sum(np.abs(start[:48])) * 997.0
        + np.sum(np.abs(end[:48])) * 1597.0
    ))
    return int(
        (base + length * 65537 + candidate * 104729 + signature)
        % (2**31 - 1)
    )


def _coordinates(length: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(
        1.0 / (length + 1), length / (length + 1), length,
        device=device,
    ).reshape(1, length, 1)


def _inverse_pca(bundle: Dict[str, Any], normalized: torch.Tensor) -> torch.Tensor:
    score = (
        normalized * bundle["score_std"] + bundle["score_mean"]
    )
    flat = (
        bundle["coefficient_mean"]
        + score @ bundle["pca_components"]
    )
    limit = bundle["coefficient_abs_limit"]
    flat = torch.maximum(torch.minimum(flat, limit), -limit)
    model: V31TransitionModel = bundle["model"]
    return flat.reshape(
        flat.shape[0],
        model.config.basis_count,
        model.config.coefficient_dim,
    )


def _sample_normalized_score(
    bundle: Dict[str, Any],
    condition: torch.Tensor,
    steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    model: V31TransitionModel = bundle["model"]
    device = condition.device
    diffusion_steps = int(
        bundle["config"].get(
            "diffusion_steps", model.config.diffusion_steps
        )
    )
    _, _, alpha_bar = linear_beta_schedule(diffusion_steps, device)
    indices = selected_timesteps(diffusion_steps, steps, device)
    latent = torch.randn(
        (1, model.config.pca_dim),
        device=device,
        generator=generator,
    )
    guidance = float(os.getenv("V31_GUIDANCE", "1.0"))
    with torch.no_grad():
        for position, index in enumerate(indices):
            time = torch.full(
                (1,),
                float(index.item()) / max(diffusion_steps - 1, 1),
                device=device,
            )
            conditional = model.diffusion(latent, time, condition)
            if abs(guidance - 1.0) > 1e-6:
                unconditional = model.diffusion(
                    latent, time, torch.zeros_like(condition)
                )
                epsilon = unconditional + guidance * (
                    conditional - unconditional
                )
            else:
                epsilon = conditional
            ab = alpha_bar[index]
            x0 = (
                latent - torch.sqrt(1.0 - ab) * epsilon
            ) / torch.sqrt(ab).clamp_min(1e-6)
            x0 = x0.clamp(-5.0, 5.0)
            if position + 1 < len(indices):
                previous = indices[position + 1]
                ab_previous = alpha_bar[previous]
                latent = (
                    torch.sqrt(ab_previous) * x0
                    + torch.sqrt(1.0 - ab_previous) * epsilon
                )
            else:
                latent = x0
    return latent


def _decode(
    model: V31TransitionModel,
    coefficients: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    start_velocity: torch.Tensor,
    end_velocity: torch.Tensor,
    length_frames: torch.Tensor,
    length: int,
) -> np.ndarray:
    with torch.no_grad():
        motion = decode_coefficients(
            coefficients,
            start,
            end,
            start_velocity,
            end_velocity,
            _coordinates(length, start.device),
            length_frames,
            model.config,
        )[0]
    return motion.cpu().numpy().astype(np.float32)


def sample_transition_diffusion(
    bundle: Dict[str, Any] | None,
    start_frame: np.ndarray,
    end_frame: np.ndarray,
    length: int,
    music_query: np.ndarray,
    rough: np.ndarray | None = None,
    device: torch.device | str = "cpu",
    blend: float = 0.20,
    steps: int = 32,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    length = int(length)
    if length <= 0:
        return np.zeros((0, MOTION_DIM), np.float32), {
            "enabled": False, "reason": "zero_length"
        }
    start_np = np.asarray(start_frame, np.float32).reshape(-1)
    end_np = np.asarray(end_frame, np.float32).reshape(-1)
    device = torch.device(device)

    if bundle is None:
        raise RuntimeError(
            "V31 transition diffusion is enabled but no checkpoint was loaded"
        )
    model: V31TransitionModel = bundle["model"]
    start = torch.from_numpy(start_np).to(device).reshape(1, -1)
    end = torch.from_numpy(end_np).to(device).reshape(1, -1)

    # Endpoint velocities are estimated from actual neighbouring contents, not
    # from the old learned refiner.  The scheduler passes a rough path; use its
    # first/last step when available, otherwise use zero tangents.
    if rough is not None and len(rough) == length:
        rough_np = np.asarray(rough, np.float32)
        start_velocity_np = rough_np[0] - start_np
        end_velocity_np = end_np - rough_np[-1]
    else:
        start_velocity_np = np.zeros_like(start_np)
        end_velocity_np = np.zeros_like(end_np)
    start_velocity = torch.from_numpy(start_velocity_np).to(device).reshape(1, -1)
    end_velocity = torch.from_numpy(end_velocity_np).to(device).reshape(1, -1)
    music = torch.from_numpy(
        np.asarray(music_query, np.float32).reshape(1, -1)
    ).to(device)
    length_frames = torch.tensor(
        [[float(length)]], device=device, dtype=torch.float32
    )
    condition = model.condition(
        start, end, start_velocity, end_velocity, music, length_frames
    )

    zero = torch.zeros(
        (1, model.config.basis_count, model.config.coefficient_dim),
        device=device,
    )
    baseline = _decode(
        model, zero, start, end, start_velocity, end_velocity,
        length_frames, length,
    )

    # The V31 scheduler supplies exact neighbouring content arrays. Direct
    # standalone calls fall back to duplicated endpoint context.
    previous, following = _consume_transition_risk_context(start_np, end_np)
    baseline_risk = transition_risk(previous, baseline, following)

    candidate_count = max(
        1, int(os.getenv("V31_CANDIDATES", "6"))
    )
    trust = float(np.clip(
        float(os.getenv("V31_RESIDUAL_TRUST", str(blend))),
        0.0, 0.50,
    ))
    max_total_ratio = float(os.getenv("V31_MAX_TOTAL_RISK_RATIO", "1.02"))
    max_boundary_ratio = float(os.getenv("V31_MAX_BOUNDARY_RATIO", "1.04"))
    max_jerk_ratio = float(os.getenv("V31_MAX_JERK_RATIO", "1.03"))
    max_foot_ratio = float(os.getenv("V31_MAX_FOOT_RATIO", "1.05"))
    max_step = float(os.getenv("V31_MAX_ROTATION_STEP_RAD", "0.22"))

    rows: List[Dict[str, Any]] = []
    accepted: List[Tuple[float, np.ndarray, Dict[str, Any]]] = []
    for candidate_index in range(candidate_count):
        generator = torch.Generator(device=device)
        generator.manual_seed(
            _seed(start_np, end_np, length, candidate_index)
        )
        normalized = _sample_normalized_score(
            bundle, condition, int(steps), generator
        )
        coefficients = _inverse_pca(bundle, normalized) * trust
        candidate = _decode(
            model, coefficients, start, end,
            start_velocity, end_velocity, length_frames, length,
        )
        risk = transition_risk(previous, candidate, following)
        is_safe, gate = accept_candidate(
            baseline_risk,
            risk,
            max_total_ratio=max_total_ratio,
            max_boundary_ratio=max_boundary_ratio,
            max_jerk_ratio=max_jerk_ratio,
            max_foot_ratio=max_foot_ratio,
            max_rotation_step_rad=max_step,
        )
        row = {
            "index": candidate_index,
            "risk": risk,
            "gate": gate,
        }
        rows.append(row)
        if is_safe:
            accepted.append((float(risk["total"]), candidate, row))

    if accepted:
        accepted.sort(key=lambda item: item[0])
        selected = accepted[0]
        result = selected[1]
        selected_index = int(selected[2]["index"])
        fallback = False
    else:
        result = baseline
        selected_index = -1
        fallback = True

    return result.astype(np.float32), {
        "enabled": True,
        "architecture": "v31_bandlimited_so3_coefficient_diffusion",
        "checkpoint": str(bundle.get("path", "")),
        "continuous_time": True,
        "basis_count": int(model.config.basis_count),
        "pca_dim": int(model.config.pca_dim),
        "candidate_count": candidate_count,
        "accepted_count": len(accepted),
        "selected_index": selected_index,
        "fallback_to_c2_baseline": fallback,
        "residual_trust": trust,
        "guidance": float(os.getenv("V31_GUIDANCE", "1.0")),
        "baseline_risk": baseline_risk,
        "candidate_audit": rows,
    }
