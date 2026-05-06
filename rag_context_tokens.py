"""
RAG Summary Token utilities for EDGE V9.

Summary dim = 7:
  0 unit_energy
  1 upper_activity
  2 lower_activity
  3 root_speed
  4 spatial_range
  5 turning
  6 contact_change_rate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

CONTACT_SLICE = slice(0, 4)
ROOT_XZ_IDX = [4, 6]
ROT_START = 7
N_JOINTS = 24
ROT_DIM = 6

UPPER_JOINTS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]


@dataclass
class RAGMotionSummary:
    unit_energy: float = 0.0
    upper_activity: float = 0.0
    lower_activity: float = 0.0
    root_speed: float = 0.0
    spatial_range: float = 0.0
    turning: float = 0.0
    contact_change_rate: float = 0.0

    @property
    def dim(self) -> int:
        return 7

    def to_array(self, dtype=np.float32) -> np.ndarray:
        return np.asarray(
            [
                self.unit_energy,
                self.upper_activity,
                self.lower_activity,
                self.root_speed,
                self.spatial_range,
                self.turning,
                self.contact_change_rate,
            ],
            dtype=dtype,
        )

    def to_tensor(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.as_tensor(self.to_array(), device=device, dtype=dtype)


def _as_unit_151(unit: Any) -> np.ndarray:
    arr = unit.detach().cpu().numpy() if torch.is_tensor(unit) else np.asarray(unit)
    if arr.ndim != 2:
        raise ValueError(f"Expected motion unit [T,151] or [151,T], got shape {arr.shape}")
    if arr.shape[-1] == 151:
        return arr.astype(np.float32, copy=False)
    if arr.shape[0] == 151:
        return arr.T.astype(np.float32, copy=False)
    raise ValueError(f"Expected one dimension to be 151, got shape {arr.shape}")


def _rot_view(unit_151: np.ndarray) -> np.ndarray:
    rot = unit_151[:, ROT_START : ROT_START + N_JOINTS * ROT_DIM]
    if rot.shape[-1] < N_JOINTS * ROT_DIM:
        pad = np.zeros((rot.shape[0], N_JOINTS * ROT_DIM - rot.shape[-1]), dtype=rot.dtype)
        rot = np.concatenate([rot, pad], axis=-1)
    return rot.reshape(rot.shape[0], N_JOINTS, ROT_DIM)


def _mean_abs_velocity(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(x, axis=0), axis=-1)))


def summarize_151_unit(unit: Any) -> RAGMotionSummary:
    m = _as_unit_151(unit)
    if m.shape[0] < 2:
        return RAGMotionSummary()

    root_xz = m[:, ROOT_XZ_IDX]
    root_vel = np.diff(root_xz, axis=0)
    root_speed = float(np.mean(np.linalg.norm(root_vel, axis=-1)))
    spatial_range = float(np.linalg.norm(root_xz.max(axis=0) - root_xz.min(axis=0)))

    rot = _rot_view(m)
    rot_vel = np.diff(rot, axis=0)

    upper_activity = _mean_abs_velocity(rot[:, UPPER_JOINTS, :])
    lower_activity = _mean_abs_velocity(rot[:, LOWER_JOINTS, :])
    unit_energy = float(np.mean(np.linalg.norm(rot_vel.reshape(rot_vel.shape[0], -1), axis=-1)))

    if root_vel.shape[0] >= 2:
        dirs = root_vel / (np.linalg.norm(root_vel, axis=-1, keepdims=True) + 1e-8)
        dots = np.sum(dirs[1:] * dirs[:-1], axis=-1).clip(-1.0, 1.0)
        turning = float(np.mean(np.arccos(dots)))
    else:
        turning = 0.0

    contact = m[:, CONTACT_SLICE]
    contact_change_rate = float(np.mean(np.abs(np.diff(contact, axis=0)))) if contact.shape[0] >= 2 else 0.0

    return RAGMotionSummary(
        unit_energy=unit_energy,
        upper_activity=upper_activity,
        lower_activity=lower_activity,
        root_speed=root_speed,
        spatial_range=spatial_range,
        turning=turning,
        contact_change_rate=contact_change_rate,
    )


def _fit_dim(vec: torch.Tensor, dim: int, device, dtype) -> torch.Tensor:
    vec = vec.reshape(-1).to(device=device, dtype=dtype)
    if dim > vec.numel():
        vec = torch.cat([vec, torch.zeros(dim - vec.numel(), device=device, dtype=dtype)])
    elif dim < vec.numel():
        vec = vec[:dim]
    return vec


def make_rag_summary_batch_from_motion(
    motion: torch.Tensor,
    dim: int = 7,
    drop_prob: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    """Create [B,T,dim] RAG summaries from target motion.

    Accepts [B,T,151] or [B,151,T].
    """
    if not torch.is_tensor(motion):
        raise TypeError("motion must be a torch.Tensor")
    if motion.ndim != 3:
        raise ValueError(f"motion must be 3D, got {tuple(motion.shape)}")

    if motion.shape[-1] == 151:
        batch = motion.detach()
        seq_len = motion.shape[1]
    elif motion.shape[1] == 151:
        batch = motion.detach().transpose(1, 2)
        seq_len = motion.shape[2]
    else:
        raise ValueError(f"Expected [B,T,151] or [B,151,T], got {tuple(motion.shape)}")

    outs = []
    for i in range(batch.shape[0]):
        unit_np = batch[i].detach().cpu().numpy()
        summary = summarize_151_unit(unit_np)
        vec = summary.to_tensor(device=motion.device, dtype=motion.dtype)
        vec = _fit_dim(vec, dim=dim, device=motion.device, dtype=motion.dtype)
        outs.append(vec[None, :].expand(seq_len, -1))

    rag = torch.stack(outs, dim=0)

    if training and drop_prob > 0:
        keep = (torch.rand((rag.shape[0], 1, 1), device=rag.device) >= float(drop_prob)).to(rag.dtype)
        rag = rag * keep

    return rag


def attach_rag_summary_condition(
    cond: Dict[str, Any],
    unit: Any,
    device=None,
    dtype: torch.dtype = torch.float32,
    dim: int = 7,
) -> Dict[str, Any]:
    """Attach cond["rag_summary"] from a retrieved unit as [1, dim]."""
    if cond is None:
        cond = {}
    summary = summarize_151_unit(unit)
    vec = summary.to_tensor(device=device, dtype=dtype)
    vec = _fit_dim(vec, dim=dim, device=device, dtype=dtype)
    cond["rag_summary"] = vec[None, :]
    return cond
