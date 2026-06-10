#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible transition generation entry point for EDGE V30.

Existing schedulers import ``load_transition_diffusion`` and
``sample_transition_diffusion`` from this historical path.  V30 keeps that
contract but dispatches new checkpoints to continuous INR latent diffusion.
Legacy V29 temporal-sequence checkpoints remain loadable for ablation runs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

from tools.v29_motion_geometry import (
    CONTACT,
    MOTION_DIM,
    NUM_JOINTS,
    ROOT,
    ROOT_X,
    ROOT_Z,
    ROT,
    make_so3_transition,
    project_motion_rotations_np,
    temporal_so3_filter_np,
    transition_blend_envelope,
)
from tools.v30_continuous_inr import (
    V30ContinuousTransitionSystem,
    config_from_dict,
    linear_beta_schedule,
    selected_timesteps,
)


def _load_raw_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    checkpoint_path = Path(str(path))
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Transition checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        str(checkpoint_path), map_location=device, weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Invalid transition checkpoint: {checkpoint_path}")
    return checkpoint


def load_transition_diffusion(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> Dict[str, Any] | None:
    if not path:
        return None
    device = torch.device(device)
    checkpoint = _load_raw_checkpoint(path, device)
    config_values = dict(checkpoint.get("config", {}))
    architecture = str(config_values.get("architecture", "legacy_frame_mlp"))

    if architecture == "v30_continuous_so3_inr_latent_diffusion":
        model_config = config_from_dict(
            dict(config_values.get("model", config_values))
        )
        system = V30ContinuousTransitionSystem(model_config).to(device)
        state = checkpoint.get("system", checkpoint.get("model"))
        if state is None:
            raise RuntimeError("V30 checkpoint has no system/model state")
        system.load_state_dict(state)
        diffusion_state = checkpoint.get("ema_diffusion")
        if diffusion_state is not None:
            system.diffusion.load_state_dict(diffusion_state)
        system.eval()
        latent_mean = torch.as_tensor(
            checkpoint.get("latent_mean", np.zeros((model_config.latent_dim,), np.float32)),
            device=device,
            dtype=torch.float32,
        ).reshape(1, -1)
        latent_std = torch.as_tensor(
            checkpoint.get("latent_std", np.ones((model_config.latent_dim,), np.float32)),
            device=device,
            dtype=torch.float32,
        ).reshape(1, -1).clamp_min(1e-4)
        return {
            "architecture": architecture,
            "system": system,
            "config": config_values,
            "path": str(path),
            "device": device,
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "best_val_loss": checkpoint.get("best_val_loss"),
            "epoch": checkpoint.get("epoch"),
        }

    if architecture == "v29_temporal_dilated_attention":
        from tools.v29_transition_diffusion_legacy import (
            load_transition_diffusion as load_legacy,
        )
        bundle = load_legacy(path, device=device)
        if bundle is not None:
            bundle["architecture"] = architecture
        return bundle

    raise RuntimeError(
        f"Unsupported transition checkpoint architecture={architecture}. "
        "Retrain with the V30 train_v27_transition_diffusion.py script."
    )


def _geodesic_motion_blend(
    base: np.ndarray,
    generated: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    a = np.asarray(base, dtype=np.float32)
    b = np.asarray(generated, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32).reshape(-1, 1)
    if a.shape != b.shape or a.ndim != 2 or a.shape[-1] != MOTION_DIM:
        raise ValueError(f"Blend expects matching [T,151], got {a.shape}, {b.shape}")
    with torch.no_grad():
        ra = rotation_6d_to_matrix(
            torch.from_numpy(a[:, ROT]).reshape(len(a), NUM_JOINTS, 6)
        )
        rb = rotation_6d_to_matrix(
            torch.from_numpy(b[:, ROT]).reshape(len(b), NUM_JOINTS, 6)
        )
        relative = torch.matmul(ra.transpose(-1, -2), rb)
        tangent = matrix_to_axis_angle(relative)
        alpha = torch.from_numpy(w).reshape(len(a), 1, 1)
        rotation = torch.matmul(ra, axis_angle_to_matrix(alpha * tangent))
        rot6d = matrix_to_rotation_6d(rotation).reshape(len(a), -1).cpu().numpy()
    out = a.copy()
    out[:, CONTACT] = (1.0 - w) * a[:, CONTACT] + w * b[:, CONTACT]
    out[:, ROOT] = (1.0 - w) * a[:, ROOT] + w * b[:, ROOT]
    out[:, ROT] = rot6d
    out[:, ROOT_X] = 0.0
    out[:, ROOT_Z] = 0.0
    return project_motion_rotations_np(out)


def _seed_for_transition(
    start: np.ndarray,
    end: np.ndarray,
    length: int,
) -> int:
    base = int(os.getenv("V30_TRANSITION_SEED", "20260610"))
    signature = int(
        np.round(
            np.sum(np.abs(start[: min(48, len(start))])) * 1009.0
            + np.sum(np.abs(end[: min(48, len(end))])) * 1709.0
        )
    )
    return int((base + int(length) * 65537 + signature) % (2**31 - 1))


def _sample_v30(
    bundle: Dict[str, Any],
    start_frame: np.ndarray,
    end_frame: np.ndarray,
    length: int,
    music_query: np.ndarray,
    rough: np.ndarray | None,
    device: torch.device,
    blend: float,
    steps: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    system: V30ContinuousTransitionSystem = bundle["system"]
    config_values = dict(bundle.get("config", {}))
    diffusion_steps = int(config_values.get("diffusion_steps", 100))
    k = int(length)
    start_np = np.asarray(start_frame, dtype=np.float32).reshape(-1)
    end_np = np.asarray(end_frame, dtype=np.float32).reshape(-1)
    if rough is None or len(rough) != k:
        rough_np = make_so3_transition(start_np[None], end_np[None], k)
    else:
        rough_np = np.asarray(rough, dtype=np.float32).copy()

    start = torch.from_numpy(start_np).to(device).reshape(1, -1)
    end = torch.from_numpy(end_np).to(device).reshape(1, -1)
    rough_tensor = torch.from_numpy(rough_np).to(device).reshape(1, k, -1)
    start_velocity = rough_tensor[:, 0] - start
    end_velocity = end - rough_tensor[:, -1]
    music = torch.from_numpy(
        np.asarray(music_query, dtype=np.float32).reshape(1, -1)
    ).to(device)
    length_frames = torch.tensor([[float(k)]], device=device)
    condition = system.condition(
        start,
        end,
        start_velocity,
        end_velocity,
        music,
        length_frames,
    )

    latent_mean = bundle["latent_mean"]
    latent_std = bundle["latent_std"]
    generator = torch.Generator(device=device)
    generator.manual_seed(_seed_for_transition(start_np, end_np, k))
    latent = torch.randn(
        (1, system.config.latent_dim),
        device=device,
        generator=generator,
    )
    _, _, alpha_bar = linear_beta_schedule(diffusion_steps, device)
    indices = selected_timesteps(diffusion_steps, int(steps), device)
    guidance = float(os.getenv("V30_LATENT_GUIDANCE", "1.20"))

    with torch.no_grad():
        for position, index in enumerate(indices):
            time = torch.full(
                (1,),
                float(index.item()) / max(diffusion_steps - 1, 1),
                device=device,
            )
            eps_cond = system.diffusion(latent, time, condition)
            if abs(guidance - 1.0) > 1e-6:
                eps_uncond = system.diffusion(
                    latent, time, torch.zeros_like(condition)
                )
                epsilon = eps_uncond + guidance * (eps_cond - eps_uncond)
            else:
                epsilon = eps_cond
            ab_t = alpha_bar[index]
            x0 = (
                latent - torch.sqrt(1.0 - ab_t) * epsilon
            ) / torch.sqrt(ab_t).clamp_min(1e-6)
            x0 = x0.clamp(-6.0, 6.0)
            if position + 1 < len(indices):
                previous = indices[position + 1]
                ab_previous = alpha_bar[previous]
                latent = (
                    torch.sqrt(ab_previous) * x0
                    + torch.sqrt(1.0 - ab_previous) * epsilon
                )
            else:
                latent = x0

        latent = latent * latent_std + latent_mean
        coordinates = torch.linspace(
            1.0 / (k + 1), k / (k + 1), k,
            device=device,
        ).reshape(1, k, 1)
        generated = system.decode(
            latent,
            start,
            end,
            start_velocity,
            end_velocity,
            condition,
            coordinates,
            length_frames,
        )[0].cpu().numpy().astype(np.float32)

    # The INR is already endpoint-safe; the envelope is retained as a final
    # conservative trust control for early experiments and ablations.
    configured_blend = float(os.getenv("V30_INR_BLEND", str(blend)))
    configured_blend = float(np.clip(configured_blend, 0.0, 1.0))
    envelope_power = float(os.getenv("V30_INR_BLEND_POWER", "1.0"))
    envelope = transition_blend_envelope(k, envelope_power)[:, None]
    result = _geodesic_motion_blend(
        rough_np,
        generated,
        configured_blend * envelope,
    )

    preserve_contacts = os.getenv("V30_PRESERVE_ROUGH_CONTACTS", "1").lower() in {
        "1", "true", "yes", "on",
    }
    if preserve_contacts:
        result[:, CONTACT] = rough_np[:, CONTACT]
    filter_window = int(os.getenv("V30_TRANSITION_FILTER_WINDOW", "3"))
    filter_strength = float(os.getenv("V30_TRANSITION_FILTER_STRENGTH", "0.10"))
    if filter_window > 1 and filter_strength > 0.0:
        result = temporal_so3_filter_np(
            result,
            window=filter_window,
            strength=float(np.clip(filter_strength, 0.0, 1.0)),
            preserve_contacts=preserve_contacts,
        )
    return result.astype(np.float32), {
        "enabled": True,
        "architecture": "v30_continuous_so3_inr_latent_diffusion",
        "checkpoint": str(bundle.get("path", "")),
        "latent_dim": int(system.config.latent_dim),
        "decode_frames": k,
        "continuous_time": True,
        "diffusion_steps_used": int(len(indices)),
        "diffusion_train_steps": int(diffusion_steps),
        "guidance": guidance,
        "inr_blend": configured_blend,
        "filter_window": filter_window,
        "filter_strength": filter_strength,
        "preserve_contacts": preserve_contacts,
    }


def sample_transition_diffusion(
    bundle: Dict[str, Any] | None,
    start_frame: np.ndarray,
    end_frame: np.ndarray,
    length: int,
    music_query: np.ndarray,
    rough: np.ndarray | None = None,
    device: torch.device | str = "cpu",
    blend: float = 0.85,
    steps: int = 32,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    k = int(length)
    if k <= 0:
        return np.zeros((0, MOTION_DIM), dtype=np.float32), {
            "enabled": False,
            "reason": "zero_length",
        }
    start_np = np.asarray(start_frame, dtype=np.float32).reshape(-1)
    end_np = np.asarray(end_frame, dtype=np.float32).reshape(-1)
    if bundle is None:
        transition = (
            np.asarray(rough, dtype=np.float32)
            if rough is not None and len(rough) == k
            else make_so3_transition(start_np[None], end_np[None], k)
        )
        return transition, {"enabled": False, "reason": "no_checkpoint"}

    architecture = str(bundle.get("architecture", bundle.get("config", {}).get(
        "architecture", ""
    )))
    if architecture == "v29_temporal_dilated_attention":
        from tools.v29_transition_diffusion_legacy import (
            sample_transition_diffusion as sample_legacy,
        )
        return sample_legacy(
            bundle,
            start_frame,
            end_frame,
            length,
            music_query,
            rough=rough,
            device=device,
            blend=blend,
            steps=steps,
        )
    if architecture != "v30_continuous_so3_inr_latent_diffusion":
        raise RuntimeError(f"Unsupported transition bundle: {architecture}")
    return _sample_v30(
        bundle,
        start_frame,
        end_frame,
        length,
        music_query,
        rough,
        torch.device(device),
        blend,
        steps,
    )
