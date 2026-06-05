#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime patch that injects V21 music-event tokens into DunhuangDataset.

Enable only after creating EDGE_V21_EVENT_MANIFEST:
  export EDGE_ENABLE_V21_EVENT_ADAPTER=1
  export EDGE_V21_EVENT_MANIFEST=data/v21_event_manifest.json

The patch is non-invasive: when disabled, the original dataset behaviour is unchanged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

_TRUE = {"1", "true", "yes", "y", "on"}
_INSTALLED = False


def _enabled() -> bool:
    return str(os.environ.get("EDGE_ENABLE_V21_EVENT_ADAPTER", "0")).strip().lower() in _TRUE


def _resize_feature(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 12:
        raise ValueError(f"V21 event feature must be [T,12], got {x.shape}")
    if len(x) == target_len:
        return x
    old = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.empty((target_len, 12), dtype=np.float32)
    for d in range(12):
        out[:, d] = np.interp(new, old, x[:, d])
    return out


def install_v21_event_adapter_patch(verbose: bool = True) -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    if not _enabled():
        if verbose:
            print("⏭️ V21 event adapter patch disabled")
        return False

    manifest_path = Path(os.environ.get("EDGE_V21_EVENT_MANIFEST", ""))
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"EDGE_V21_EVENT_MANIFEST is missing: {manifest_path}. "
            "Run tools/build_v21_event_manifest.py first."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    window_map: Dict[str, str] = dict(payload.get("windows", {}))

    from dataset.dance_dataset import DunhuangDataset

    original_init = DunhuangDataset.__init__
    original_getitem = DunhuangDataset.__getitem__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._v21_event_window_map = window_map
        self._v21_event_cache = {}

    def patched_getitem(self, idx):
        pose, cond, window_id, audio_source = original_getitem(self, idx)
        feature_path = self._v21_event_window_map.get(str(window_id), "")
        if feature_path:
            if feature_path not in self._v21_event_cache:
                self._v21_event_cache[feature_path] = _resize_feature(
                    np.load(feature_path).astype(np.float32), self.seq_len
                )
            cond = dict(cond)
            cond["rag_summary"] = torch.from_numpy(self._v21_event_cache[feature_path]).float()
            cond["v21_event_paired"] = torch.tensor(1.0, dtype=torch.float32)
        else:
            cond = dict(cond)
            cond["rag_summary"] = torch.zeros((self.seq_len, 12), dtype=torch.float32)
            cond["v21_event_paired"] = torch.tensor(0.0, dtype=torch.float32)
        return pose, cond, window_id, audio_source

    DunhuangDataset.__init__ = patched_init
    DunhuangDataset.__getitem__ = patched_getitem
    _INSTALLED = True
    if verbose:
        print(
            f"✅ V21 event adapter patch installed: manifest={manifest_path}, "
            f"windows={len(window_map)}, rag_summary_dim=12"
        )
    return True
