#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU batch cache for V34 warp-aware candidate boundary scoring.

The V34 scheduler spends most of its time evaluating the same previous-event
to candidate-event boundary features inside a Python beam loop.  This module
keeps the scientific scoring equations unchanged, but computes the dense
``beam_previous x shortlist`` boundary table once per slot on a CUDA device.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from pytorch3d.transforms import rotation_6d_to_matrix


CONTACT = slice(0, 4)
ROOT_Y = 5
ROT = slice(7, 151)
ROOT_ROT6D = slice(7, 13)
MOTION_DIM = 151
ROT_DIM = 144


def _enabled(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_device(current_device: torch.device | str | None) -> torch.device | None:
    if not _enabled("V34_USE_GPU_RETRIEVAL", "1"):
        return None
    requested = os.getenv("V34_GPU_DEVICE", "").strip()
    if requested:
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        return None
    if device.type == "cuda" and not torch.cuda.is_available():
        return None
    if current_device is not None:
        current = torch.device(current_device)
        if current.type == "cuda" and not requested:
            device = current
    return device


def _as_float_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value, dtype=np.float32), device=device)


def _endpoint_frames(motions: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return padded first/last endpoint frames and source lengths.

    first3[:, 0/1/2] corresponds to motion[0/1/2] when available.
    last3[:, 0/1/2] corresponds to motion[-3/-2/-1] when available.
    Length masks are used later so short motions still match the CPU fallback.
    """
    count = len(motions)
    first3 = np.zeros((count, 3, MOTION_DIM), dtype=np.float32)
    last3 = np.zeros((count, 3, MOTION_DIM), dtype=np.float32)
    lengths = np.zeros((count,), dtype=np.int64)
    for index, motion in enumerate(motions):
        x = np.asarray(motion, dtype=np.float32)
        if x.ndim != 2 or x.shape[-1] != MOTION_DIM or len(x) == 0:
            raise ValueError(f"Invalid EDGE motion at index {index}: {x.shape}")
        lengths[index] = len(x)
        first3[index, 0] = x[0]
        first3[index, 1] = x[1] if len(x) > 1 else x[0]
        first3[index, 2] = x[2] if len(x) > 2 else first3[index, 1]
        last3[index, 2] = x[-1]
        last3[index, 1] = x[-2] if len(x) > 1 else x[-1]
        last3[index, 0] = x[-3] if len(x) > 2 else last3[index, 1]
    return first3, last3, lengths


def _music_transition_frames(phrase: Any, args: Any) -> tuple[int, Dict[str, Any]]:
    base = int(getattr(phrase, "transition_base_frames"))
    profile = str(getattr(phrase, "transition_profile", ""))
    if profile == "accent_cut":
        base = min(base, 24)
    elif profile in {"calm_sustain", "section_sustain"}:
        base = max(base, 24)
    elif profile == "tense_drive":
        base = int(round(0.65 * base + 0.35 * 18))
    base = int(np.clip(base, args.transition_min_frames, args.transition_max_frames))
    return base, {
        "music_transition_frames": int(base),
        "transition_profile": profile,
        "boundary_accent_strength": float(getattr(phrase, "boundary_accent_strength", 0.0)),
        "speed_factor": float(getattr(phrase, "speed_factor", 1.0)),
        "energy": float(getattr(phrase, "energy", 0.0)),
        "onset": float(getattr(phrase, "onset", 0.0)),
        "beat_density": float(getattr(phrase, "beat_density", 0.0)),
        "tension": float(getattr(phrase, "tension", 0.0)),
        "calmness": float(getattr(phrase, "calmness", 0.0)),
    }


@dataclass
class V34SlotBoundaryCache:
    previous_indices: np.ndarray
    candidate_indices: np.ndarray
    previous_lookup: Dict[int, int]
    candidate_lookup: Dict[int, int]
    pose_jump: np.ndarray
    velocity_jump: np.ndarray
    acceleration_jump: np.ndarray
    contact_jump: np.ndarray
    contact_binary_jump: np.ndarray
    support_count_jump: np.ndarray
    aerial_planted_switch: np.ndarray
    stance_flip: np.ndarray
    yaw_gap_deg: np.ndarray
    transition_cost: np.ndarray
    transition_len: np.ndarray
    physical_len: np.ndarray
    yaw_required_frames: np.ndarray
    slot_budget_cap: int
    slot_budget_capped_from: np.ndarray
    music_len: int
    music_meta: Dict[str, Any]
    dominant_reason: np.ndarray

    def get(self, previous: int, candidate: int, args: Any) -> Dict[str, Any] | None:
        row = self.previous_lookup.get(int(previous))
        col = self.candidate_lookup.get(int(candidate))
        if row is None or col is None:
            return None

        boundary = {
            "pose_jump": float(self.pose_jump[row, col]),
            "velocity_jump": float(self.velocity_jump[row, col]),
            "acceleration_jump": float(self.acceleration_jump[row, col]),
            "contact_jump": float(self.contact_jump[row, col]),
            "contact_binary_jump": float(self.contact_binary_jump[row, col]),
            "support_count_jump": float(self.support_count_jump[row, col]),
            "aerial_planted_switch": float(self.aerial_planted_switch[row, col]),
            "stance_flip": float(self.stance_flip[row, col]),
            "yaw_gap_deg": float(self.yaw_gap_deg[row, col]),
        }
        capped_from = int(self.slot_budget_capped_from[row, col])
        transition = int(self.transition_len[row, col])
        physical = int(self.physical_len[row, col])
        reason_code = int(self.dominant_reason[row, col])
        if capped_from >= 0:
            dominant = "slot_budget"
            capped_value: int | None = capped_from
        elif reason_code == 1:
            dominant = "physical"
            capped_value = None
        else:
            dominant = "music"
            capped_value = None
        transition_meta = {
            **self.music_meta,
            "physical_min_frames": physical,
            "pose_jump": boundary["pose_jump"],
            "velocity_jump": boundary["velocity_jump"],
            "acceleration_jump": boundary["acceleration_jump"],
            "contact_jump": boundary["contact_jump"],
            "contact_binary_jump": boundary["contact_binary_jump"],
            "support_count_jump": boundary["support_count_jump"],
            "aerial_planted_switch": boundary["aerial_planted_switch"],
            "stance_flip": boundary["stance_flip"],
            "yaw_gap_deg": boundary["yaw_gap_deg"],
            "yaw_required_frames": int(self.yaw_required_frames[row, col]),
            "chosen_transition_frames": transition,
            "slot_budget_cap": int(self.slot_budget_cap),
            "slot_budget_capped_from": capped_value,
            "dominant_reason": dominant,
            "gpu_boundary_cache": True,
        }
        return {
            "transition_cost": float(self.transition_cost[row, col]),
            "candidate_boundary": boundary,
            "boundary_velocity_penalty": float(
                min(
                    boundary["velocity_jump"] / max(float(args.velocity_jump_reference), 1e-6),
                    float(args.boundary_penalty_cap),
                )
            ),
            "boundary_acceleration_penalty": float(
                min(
                    boundary["acceleration_jump"] / max(float(args.acceleration_jump_reference), 1e-6),
                    float(args.boundary_penalty_cap),
                )
            ),
            "transition_len": transition,
            "transition_meta": transition_meta,
        }


class V34GpuCandidateCache:
    def __init__(
        self,
        arrays: Mapping[str, Any],
        motions: Sequence[np.ndarray],
        device: torch.device,
    ) -> None:
        resolved = _resolve_device(device)
        if resolved is None:
            raise RuntimeError("V34 GPU candidate cache is disabled or CUDA is unavailable")
        self.device = resolved
        first3, last3, lengths = _endpoint_frames(motions)
        self.first3 = _as_float_tensor(first3, self.device)
        self.last3 = _as_float_tensor(last3, self.device)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long, device=self.device)
        self.entry_pose = _as_float_tensor(arrays["entry_pose"], self.device)
        self.exit_pose = _as_float_tensor(arrays["exit_pose"], self.device)
        self.entry_vel = _as_float_tensor(arrays["entry_vel"], self.device)
        self.exit_vel = _as_float_tensor(arrays["exit_vel"], self.device)

    def compute_slot(
        self,
        previous_indices: Sequence[int],
        candidate_indices: Sequence[int],
        phrase: Any,
        args: Any,
    ) -> V34SlotBoundaryCache | None:
        prev_np = np.asarray(sorted({int(x) for x in previous_indices}), dtype=np.int64)
        cand_np = np.asarray([int(x) for x in candidate_indices], dtype=np.int64)
        if len(prev_np) == 0 or len(cand_np) == 0:
            return None

        prev = torch.as_tensor(prev_np, dtype=torch.long, device=self.device)
        cand = torch.as_tensor(cand_np, dtype=torch.long, device=self.device)
        last = self.last3[prev]
        first = self.first3[cand]
        prev_len = self.lengths[prev]
        cand_len = self.lengths[cand]

        prev_exit = last[:, 2, :]
        next_entry = first[:, 0, :]
        pose = torch.linalg.vector_norm(
            prev_exit[:, None, ROT] - next_entry[None, :, ROT],
            dim=-1,
        ) / math.sqrt(float(ROT_DIM))

        pv = last[:, 2, ROT] - last[:, 1, ROT]
        nv = first[:, 1, ROT] - first[:, 0, ROT]
        pv = torch.where((prev_len > 1)[:, None], pv, torch.zeros_like(pv))
        nv = torch.where((cand_len > 1)[:, None], nv, torch.zeros_like(nv))
        velocity = torch.linalg.vector_norm(pv[:, None, :] - nv[None, :, :], dim=-1) / math.sqrt(float(ROT_DIM))

        pa = last[:, 2, ROT] - 2.0 * last[:, 1, ROT] + last[:, 0, ROT]
        na = first[:, 2, ROT] - 2.0 * first[:, 1, ROT] + first[:, 0, ROT]
        pa = torch.where((prev_len > 2)[:, None], pa, torch.zeros_like(pa))
        na = torch.where((cand_len > 2)[:, None], na, torch.zeros_like(na))
        acceleration = torch.linalg.vector_norm(pa[:, None, :] - na[None, :, :], dim=-1) / math.sqrt(float(ROT_DIM))

        contact = torch.mean(
            torch.abs(prev_exit[:, None, CONTACT] - next_entry[None, :, CONTACT]),
            dim=-1,
        )
        contact_threshold = float(os.getenv("V34_COMPAT_CONTACT_BINARY_THRESHOLD", "0.5"))
        prev_contact_hard = (prev_exit[:, CONTACT] >= contact_threshold).to(torch.float32)
        next_contact_hard = (next_entry[:, CONTACT] >= contact_threshold).to(torch.float32)
        contact_binary = torch.mean(
            torch.abs(prev_contact_hard[:, None, :] - next_contact_hard[None, :, :]),
            dim=-1,
        )
        prev_support = torch.sum(prev_contact_hard, dim=-1)
        next_support = torch.sum(next_contact_hard, dim=-1)
        support_count = torch.abs(prev_support[:, None] - next_support[None, :]) / 4.0
        prev_air = prev_support <= 0.5
        next_air = next_support <= 0.5
        prev_planted = prev_support >= 2.0
        next_planted = next_support >= 2.0
        aerial_planted = (
            (prev_air[:, None] & next_planted[None, :])
            | (prev_planted[:, None] & next_air[None, :])
        ).to(torch.float32)
        prev_left = prev_contact_hard[:, 0] + prev_contact_hard[:, 2]
        prev_right = prev_contact_hard[:, 1] + prev_contact_hard[:, 3]
        next_left = next_contact_hard[:, 0] + next_contact_hard[:, 2]
        next_right = next_contact_hard[:, 1] + next_contact_hard[:, 3]
        prev_side = torch.sign(prev_left - prev_right)
        next_side = torch.sign(next_left - next_right)
        stance_flip = (
            (prev_side[:, None] * next_side[None, :] < 0.0)
            & (prev_support[:, None] >= 1.0)
            & (next_support[None, :] >= 1.0)
        ).to(torch.float32)

        prev_root = rotation_6d_to_matrix(prev_exit[:, ROOT_ROT6D])
        next_root = rotation_6d_to_matrix(next_entry[:, ROOT_ROT6D])
        prev_yaw = torch.atan2(prev_root[:, 0, 2], prev_root[:, 2, 2])
        next_yaw = torch.atan2(next_root[:, 0, 2], next_root[:, 2, 2])
        yaw_delta = next_yaw[None, :] - prev_yaw[:, None]
        yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
        yaw_gap = torch.abs(yaw_delta) * (180.0 / math.pi)

        transition_pose = torch.linalg.vector_norm(
            self.exit_pose[prev][:, None, ROT] - self.entry_pose[cand][None, :, ROT],
            dim=-1,
        ) / math.sqrt(float(ROT_DIM))
        transition_vel = torch.linalg.vector_norm(
            self.exit_vel[prev][:, None, ROT] - self.entry_vel[cand][None, :, ROT],
            dim=-1,
        ) / math.sqrt(float(ROT_DIM))
        root_y = torch.abs(self.exit_pose[prev][:, None, ROOT_Y] - self.entry_pose[cand][None, :, ROOT_Y])
        transition_contact = torch.mean(
            torch.abs(self.exit_pose[prev][:, None, CONTACT] - self.entry_pose[cand][None, :, CONTACT]),
            dim=-1,
        )
        transition_cost = transition_pose + 0.35 * transition_vel + 0.50 * root_y + 0.15 * transition_contact

        music_len, music_meta = _music_transition_frames(phrase, args)
        physical_extra = (
            float(args.physical_pose_frames) * torch.minimum(pose / max(float(args.pose_jump_reference), 1e-6), torch.tensor(2.0, device=self.device))
            + float(args.physical_velocity_frames) * torch.minimum(velocity / max(float(args.velocity_jump_reference), 1e-6), torch.tensor(2.0, device=self.device))
            + float(args.physical_acceleration_frames) * torch.minimum(acceleration / max(float(args.acceleration_jump_reference), 1e-6), torch.tensor(2.0, device=self.device))
            + float(args.physical_contact_frames) * contact
        )
        yaw_required = torch.ceil(
            float(args.yaw_transition_safety_factor)
            * yaw_gap
            * float(args.fps)
            / max(float(args.transition_yaw_limit_dps), 1.0)
        ).to(torch.int64)
        physical = torch.round(
            torch.maximum(
                torch.tensor(float(args.transition_min_frames), device=self.device) + physical_extra,
                yaw_required.to(torch.float32),
            )
        ).to(torch.int64)
        physical = torch.clamp(physical, int(args.transition_min_frames), int(args.transition_max_frames))

        chosen = torch.maximum(
            torch.full_like(physical, int(music_len)),
            physical,
        )
        if str(getattr(phrase, "transition_profile", "")) == "accent_cut":
            chosen = torch.where(physical <= int(music_len), torch.minimum(chosen, torch.full_like(chosen, 24)), chosen)
        chosen = torch.clamp(chosen, int(args.transition_min_frames), int(args.transition_max_frames))

        capped_from = torch.full_like(chosen, -1)
        cap = max(0, int(getattr(phrase, "length")) - int(args.min_content_frames))
        if bool(getattr(args, "lock_music_boundaries", False)):
            capped_from = torch.where(chosen > cap, chosen, capped_from)
            chosen = torch.minimum(chosen, torch.full_like(chosen, int(cap)))

        dominant = torch.where(physical > int(music_len), torch.ones_like(chosen), torch.zeros_like(chosen))
        to_cpu = lambda tensor: tensor.detach().cpu().numpy()
        return V34SlotBoundaryCache(
            previous_indices=prev_np,
            candidate_indices=cand_np,
            previous_lookup={int(value): index for index, value in enumerate(prev_np.tolist())},
            candidate_lookup={int(value): index for index, value in enumerate(cand_np.tolist())},
            pose_jump=to_cpu(pose).astype(np.float32),
            velocity_jump=to_cpu(velocity).astype(np.float32),
            acceleration_jump=to_cpu(acceleration).astype(np.float32),
            contact_jump=to_cpu(contact).astype(np.float32),
            contact_binary_jump=to_cpu(contact_binary).astype(np.float32),
            support_count_jump=to_cpu(support_count).astype(np.float32),
            aerial_planted_switch=to_cpu(aerial_planted).astype(np.float32),
            stance_flip=to_cpu(stance_flip).astype(np.float32),
            yaw_gap_deg=to_cpu(yaw_gap).astype(np.float32),
            transition_cost=to_cpu(transition_cost).astype(np.float32),
            transition_len=to_cpu(chosen).astype(np.int32),
            physical_len=to_cpu(physical).astype(np.int32),
            yaw_required_frames=to_cpu(yaw_required).astype(np.int32),
            slot_budget_cap=int(cap),
            slot_budget_capped_from=to_cpu(capped_from).astype(np.int32),
            music_len=int(music_len),
            music_meta=music_meta,
            dominant_reason=to_cpu(dominant).astype(np.int32),
        )


def build_v34_gpu_candidate_cache(
    arrays: Mapping[str, Any],
    motions: Sequence[np.ndarray],
    device: torch.device,
) -> V34GpuCandidateCache | None:
    if not _enabled("V34_USE_GPU_RETRIEVAL", "1"):
        return None
    try:
        cache = V34GpuCandidateCache(arrays, motions, device)
    except Exception as exc:
        if _enabled("V34_GPU_STRICT", "0"):
            raise
        if _enabled("V34_GPU_RETRIEVAL_VERBOSE", "1"):
            print(f"[V34-GPU] disabled: {exc}")
        return None
    if _enabled("V34_GPU_RETRIEVAL_VERBOSE", "1"):
        print(f"[V34-GPU] candidate boundary cache enabled on {cache.device}")
    return cache
