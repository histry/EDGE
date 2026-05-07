"""Utilities for Text-Bridged Context RAG.

Shared by planner semantic re-ranking, inference context attachment and training
self-context attachment.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

CONTACT_SLICE = slice(0, 4)
ROOT_XZ_IDX = [4, 6]
ROT_START = 7
N_JOINTS = 24
ROT_DIM = 6
UPPER_JOINTS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default)).strip()


def split_csv(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in str(text).replace(";", ",").split(",") if x.strip()]


def l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / max(float(np.linalg.norm(x)), eps)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def as_unit_t151(arr) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] == 151:
        return arr.reshape(1, 151)
    if arr.ndim != 2:
        raise ValueError(f"Expected [T,151] or [151,T], got {arr.shape}")
    if arr.shape[-1] == 151:
        return arr.astype(np.float32, copy=False)
    if arr.shape[0] == 151:
        return arr.T.astype(np.float32, copy=False)
    raise ValueError(f"Expected one dim to be 151, got {arr.shape}")


def rot_view(unit: np.ndarray) -> np.ndarray:
    unit = as_unit_t151(unit)
    rot = unit[:, ROT_START : ROT_START + N_JOINTS * ROT_DIM]
    if rot.shape[-1] < N_JOINTS * ROT_DIM:
        pad = np.zeros((rot.shape[0], N_JOINTS * ROT_DIM - rot.shape[-1]), dtype=rot.dtype)
        rot = np.concatenate([rot, pad], axis=-1)
    return rot.reshape(rot.shape[0], N_JOINTS, ROT_DIM)


def mean_abs_velocity(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(x, axis=0), axis=-1)))


def unit_stats(unit) -> dict:
    m = as_unit_t151(unit)
    if m.shape[0] < 2:
        return dict(unit_energy=0.0, upper_activity=0.0, lower_activity=0.0, root_speed=0.0, spatial_range=0.0, turning=0.0, contact_change_rate=0.0)
    root_xz = m[:, ROOT_XZ_IDX]
    root_vel = np.diff(root_xz, axis=0)
    root_speed = float(np.mean(np.linalg.norm(root_vel, axis=-1)))
    spatial_range = float(np.linalg.norm(root_xz.max(axis=0) - root_xz.min(axis=0)))
    rot = rot_view(m)
    rot_vel = np.diff(rot, axis=0)
    unit_energy = float(np.mean(np.linalg.norm(rot_vel.reshape(rot_vel.shape[0], -1), axis=-1)))
    upper_activity = mean_abs_velocity(rot[:, UPPER_JOINTS, :])
    lower_activity = mean_abs_velocity(rot[:, LOWER_JOINTS, :])
    if root_vel.shape[0] >= 2:
        dirs = root_vel / (np.linalg.norm(root_vel, axis=-1, keepdims=True) + 1e-8)
        dots = np.sum(dirs[1:] * dirs[:-1], axis=-1).clip(-1.0, 1.0)
        turning = float(np.mean(np.arccos(dots)))
    else:
        turning = 0.0
    contact = m[:, CONTACT_SLICE]
    contact_change_rate = float(np.mean(np.abs(np.diff(contact, axis=0)))) if contact.shape[0] >= 2 else 0.0
    return dict(unit_energy=unit_energy, upper_activity=upper_activity, lower_activity=lower_activity, root_speed=root_speed, spatial_range=spatial_range, turning=turning, contact_change_rate=contact_change_rate)


def caption_from_unit(unit) -> str:
    s = unit_stats(unit)
    def level(v, a, b, low, mid, high):
        return low if v < a else (mid if v < b else high)
    energy = level(s["unit_energy"], 0.20, 0.45, "低能量", "中等能量", "高能量")
    upper = level(s["upper_activity"], 0.04, 0.10, "上肢含蓄", "上肢舒展", "上肢大幅展开")
    lower = level(s["lower_activity"], 0.03, 0.08, "下肢稳定", "步伐变化", "下肢活跃")
    root = level(s["root_speed"], 0.015, 0.05, "缓慢移动", "平稳移动", "快速移动")
    space = level(s["spatial_range"], 0.10, 0.50, "小空间", "中等空间", "大空间")
    turn = "带有旋转" if s["turning"] > 0.30 else "方向平稳"
    contact = "重心稳定" if s["contact_change_rate"] < 0.25 else "脚步切换明显"
    return f"{energy}，{root}，{upper}，{lower}，{space}，{turn}，{contact}，敦煌舞，飞天风格"

_TEXT_ENCODER_CACHE = {}


def get_text_encoder(model_name: str | None = None, device: str | None = None, fallback_dim: int | None = None):
    model_name = model_name or env_str("EDGE_TEXT_BRIDGE_MODEL", "BAAI/bge-small-zh-v1.5")
    device = device or env_str("EDGE_TEXT_BRIDGE_DEVICE", "cpu")
    fallback_dim = int(fallback_dim or env_int("EDGE_TEXT_BRIDGE_FALLBACK_DIM", 384))
    key = (model_name, device, fallback_dim)
    if key in _TEXT_ENCODER_CACHE:
        return _TEXT_ENCODER_CACHE[key]
    try:
        from model.text_bridge_encoder import TextBridgeEncoder
    except Exception:
        from text_bridge_encoder import TextBridgeEncoder  # type: ignore
    enc = TextBridgeEncoder(model_name=model_name, device=device, fallback_dim=fallback_dim)
    _TEXT_ENCODER_CACHE[key] = enc
    return enc


def encode_texts(texts: Iterable[str], model_name: str | None = None, device: str | None = None, fallback_dim: int | None = None) -> np.ndarray:
    enc = get_text_encoder(model_name=model_name, device=device, fallback_dim=fallback_dim)
    return enc.encode(list(texts)).astype(np.float32)


def default_query_for_mode(mode: str = "auto_multiunit") -> str:
    custom = env_str("EDGE_TEXT_QUERY", "")
    if custom:
        return custom
    mode = str(mode or "").lower()
    if "upper" in mode:
        return "敦煌舞，飞天风格，上肢大幅舒展，手臂展开，身体线条清晰，中高能量，优雅转身"
    if "manual" in mode:
        return "敦煌舞，人工精选动作单元，上肢舒展，姿态清晰，重心稳定"
    return "敦煌舞，飞天风格，高能量，上肢大幅舒展，流动转身，空间展开，短段落编舞"


def load_unit_paths(paths: Sequence[str], max_len: int | None = None) -> Tuple[np.ndarray, List[str]]:
    units, captions = [], []
    max_len = int(max_len or env_int("EDGE_RAG_CONTEXT_MAX_LEN", 45))
    for p in paths:
        path = Path(str(p))
        if not path.exists():
            continue
        arr = np.load(str(path), allow_pickle=True)
        unit = as_unit_t151(arr)
        if max_len > 0 and unit.shape[0] > max_len:
            c = unit.shape[0] // 2
            start = max(0, min(unit.shape[0] - max_len, c - max_len // 2))
            unit = unit[start:start + max_len]
        units.append(unit.astype(np.float32))
        captions.append(caption_from_unit(unit))
    if not units:
        return np.zeros((0, max_len, 151), dtype=np.float32), []
    T = max(max(u.shape[0] for u in units), max_len)
    padded = np.zeros((len(units), T, 151), dtype=np.float32)
    for i, u in enumerate(units):
        L = min(T, u.shape[0])
        padded[i, :L] = u[:L]
        if L < T:
            padded[i, L:] = u[L - 1:L]
    return padded, captions


def build_context_tensors_from_paths(paths: Sequence[str], batch_size: int, device, dtype=torch.float32, text_model: str | None = None, text_device: str | None = None, fallback_dim: int | None = None):
    units, captions = load_unit_paths(paths)
    if units.shape[0] == 0:
        return None, None, None, captions
    emb = encode_texts(captions, model_name=text_model, device=text_device, fallback_dim=fallback_dim)
    units_t = torch.as_tensor(units, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
    emb_t = torch.as_tensor(emb, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    mask_t = torch.ones((batch_size, units.shape[0]), device=device, dtype=torch.bool)
    return units_t, emb_t, mask_t, captions


def build_self_context_from_motion(motion: torch.Tensor, count: int = 3, length: int = 45) -> torch.Tensor:
    if motion.ndim != 3:
        raise ValueError(f"motion must be [B,T,151] or [B,151,T], got {tuple(motion.shape)}")
    if motion.shape[-1] == 151:
        x = motion
    elif motion.shape[1] == 151:
        x = motion.transpose(1, 2)
    else:
        raise ValueError(f"Expected one dim to be 151, got {tuple(motion.shape)}")
    B, T, C = x.shape
    count = max(1, int(count))
    length = max(4, min(int(length), T))
    frames = [int(round((i + 1) * (T - 1) / (count + 1))) for i in range(count)]
    clips = []
    for f in frames:
        start = max(0, min(T - length, f - length // 2))
        clip = x[:, start:start + length]
        if clip.shape[1] < length:
            clip = torch.cat([clip, clip[:, -1:].expand(-1, length - clip.shape[1], -1)], dim=1)
        clips.append(clip)
    return torch.stack(clips, dim=1).contiguous()


def cosine_scores(query_emb: np.ndarray, item_emb: np.ndarray) -> np.ndarray:
    q = l2_normalize_np(query_emb.reshape(1, -1))[0]
    e = l2_normalize_np(item_emb)
    if e.shape[-1] != q.shape[-1]:
        d = min(e.shape[-1], q.shape[-1])
        q = l2_normalize_np(q[:d])
        e = l2_normalize_np(e[..., :d])
    return np.dot(e, q).astype(np.float32)
