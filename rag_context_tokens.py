# -*- coding: utf-8 -*-
"""Future RAG context-token stub for TEA-MotionAdapter.

This file intentionally does not patch the model yet.  Full train-time RAG
should only be enabled after trajectory-energy adapter training is stable.
The helper below defines the expected lightweight summary schema so future
experiments do not have to redesign it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch


@dataclass
class RAGMotionSummary:
    unit_energy: float
    upper_activity: float
    lower_activity: float
    root_speed: float
    spatial_range: float
    turning: float = 0.0
    contact_change_rate: float = 0.0

    def to_tensor(self, device=None, dtype=torch.float32) -> torch.Tensor:
        arr = [
            self.unit_energy,
            self.upper_activity,
            self.lower_activity,
            self.root_speed,
            self.spatial_range,
            self.turning,
            self.contact_change_rate,
        ]
        return torch.tensor(arr, device=device, dtype=dtype).reshape(1, -1)


def summarize_151_unit(unit: np.ndarray) -> RAGMotionSummary:
    """Summarize a [T,151] unit into a 7-D context vector.

    This is the recommended first RAG-context representation before full
    45-frame context tokens, because it is cheap, hard to over-copy, and easier
    to ablate.
    """
    x = np.asarray(unit, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 151 or x.shape[0] < 2:
        return RAGMotionSummary(0, 0, 0, 0, 0, 0, 0)

    root = x[:, [4, 6]]
    root_delta = np.linalg.norm(root[1:] - root[:-1], axis=-1)
    root_speed = float(root_delta.mean())
    spatial_range = float(np.linalg.norm(root.max(axis=0) - root.min(axis=0)))

    def idx(joints):
        out = []
        for j in joints:
            out.extend(range(7 + 6*j, 7 + 6*(j+1)))
        return out

    upper = idx([12,13,14,15,16,17,18,19,20,21,22,23])
    lower = idx([1,2,4,5,7,8,10,11])
    upper_activity = float(np.sqrt(np.mean((x[1:, upper] - x[:-1, upper]) ** 2)))
    lower_activity = float(np.sqrt(np.mean((x[1:, lower] - x[:-1, lower]) ** 2)))
    energy = float(np.sqrt(np.mean((x[1:] - x[:-1]) ** 2)))
    contact_change = float(np.abs(np.clip(x[1:, :4], 0, 1) - np.clip(x[:-1, :4], 0, 1)).mean())

    return RAGMotionSummary(
        unit_energy=energy,
        upper_activity=upper_activity,
        lower_activity=lower_activity,
        root_speed=root_speed,
        spatial_range=spatial_range,
        turning=0.0,
        contact_change_rate=contact_change,
    )


def attach_rag_summary_condition(cond: Dict[str, Any], unit: np.ndarray, device=None) -> Dict[str, Any]:
    """Attach cond['rag_summary'] without changing current model behavior.

    Future model patch should read this key only when EDGE_RAG_CONTEXT=summary.
    """
    cond = dict(cond)
    cond["rag_summary"] = summarize_151_unit(unit).to_tensor(device=device)
    return cond
