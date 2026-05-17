#!/usr/bin/env python3
"""Utilities for weakly-supervised in-the-wild dance-video RAG ingestion.

This module deliberately does NOT download videos and does NOT store raw video.
It consumes assets that the user has rights to use:

  - motion_path: extracted / cleaned motion in EDGE 151-D format, or a dict/npz
                 containing a compatible [T,151] array
  - audio_path:  wav/mp3 file already extracted from the same video
  - metadata:    source_id, title, rights_tag, optional fps

The output features are weak labels for ChoreoRAG retrieval, not supervised
paired music-to-motion training labels.

Python 3.8-compatible.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import wave
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

CONTACT_SLICE = slice(0, 4)
ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
ROOT_SLICE = slice(4, 7)
ROT_START = 7
ROT_DIM = 6
NFEATS = 151

LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
TORSO_JOINTS = [3, 6, 9]
UPPER_JOINTS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


def as_float32(x):
    return np.asarray(x, dtype=np.float32)


def safe_norm(x, axis=-1, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    return np.sqrt(np.sum(x * x, axis=axis) + eps).astype(np.float32)


def norm01(x, eps=1e-8):
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(x)) if x.size else 0.0
    hi = float(np.max(x)) if x.size else 0.0
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo + eps)).astype(np.float32)


def robust_norm(x, lo_q=5, hi_q=95, eps=1e-8):
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return x.astype(np.float32)
    lo, hi = np.percentile(x, [lo_q, hi_q])
    if float(hi - lo) < eps:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)


def moving_average_1d(x, radius=2):
    x = np.asarray(x, dtype=np.float32)
    if radius <= 0 or x.shape[0] < 3:
        return x.astype(np.float32)
    k = np.ones((2 * radius + 1,), dtype=np.float32)
    k /= k.sum()
    pad = np.pad(x, (radius, radius), mode="edge")
    return np.convolve(pad, k, mode="valid").astype(np.float32)


def moving_average(x, radius=2):
    x = np.asarray(x, dtype=np.float32)
    if radius <= 0 or x.shape[0] < 3:
        return x.astype(np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    for c in range(x.shape[-1]):
        out[:, c] = moving_average_1d(x[:, c], radius=radius)
    return out


def rot6d_indices(joints):
    out = []
    for j in joints:
        s = ROT_START + ROT_DIM * int(j)
        out.extend(range(s, s + ROT_DIM))
    return [i for i in out if 0 <= i < NFEATS]


LOWER_DIMS = rot6d_indices(LOWER_JOINTS)
TORSO_DIMS = rot6d_indices(TORSO_JOINTS)
UPPER_DIMS = rot6d_indices(UPPER_JOINTS)


def renormalize_rot6d(motion):
    """Gram-Schmidt normalize every 6D rotation block.

    This is a lightweight replacement for a full SO(3) projection.  It keeps the
    6D representation on a valid first-two-columns rotation manifold.
    """
    motion = np.asarray(motion, dtype=np.float32).copy()
    if motion.shape[-1] < NFEATS:
        return motion
    rot = motion[:, ROT_START:NFEATS].reshape(motion.shape[0], 24, 6)
    a = rot[..., 0:3]
    b = rot[..., 3:6]
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    rot[..., 0:3] = a
    rot[..., 3:6] = b
    motion[:, ROT_START:NFEATS] = rot.reshape(motion.shape[0], 24 * 6)
    return motion.astype(np.float32)


def clean_motion_151(
    motion,
    smooth_radius=1,
    root_smooth_radius=1,
    freeze_stationary_root=False,
    stationary_root_range_threshold=0.08,
):
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim == 3 and motion.shape[0] == 1:
        motion = motion[0]
    if motion.ndim != 2 or motion.shape[-1] != NFEATS:
        raise ValueError("Expected [T,151] motion, got %s" % (motion.shape,))
    out = motion.copy()
    out[:, CONTACT_SLICE] = np.clip(out[:, CONTACT_SLICE], 0.0, 1.0)
    out[:, CONTACT_SLICE] = (out[:, CONTACT_SLICE] > 0.5).astype(np.float32)

    if root_smooth_radius > 0:
        out[:, ROOT_SLICE] = moving_average(out[:, ROOT_SLICE], radius=root_smooth_radius)
    if smooth_radius > 0:
        out[:, ROT_START:NFEATS] = moving_average(out[:, ROT_START:NFEATS], radius=smooth_radius)

    if freeze_stationary_root:
        root_xz = out[:, [ROOT_X_IDX, ROOT_Z_IDX]]
        rng = np.linalg.norm(root_xz.max(axis=0) - root_xz.min(axis=0))
        if rng <= float(stationary_root_range_threshold):
            out[:, ROOT_X_IDX] = out[0, ROOT_X_IDX]
            out[:, ROOT_Z_IDX] = out[0, ROOT_Z_IDX]

    out = renormalize_rot6d(out)
    return out.astype(np.float32)


def load_motion_151(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() == ".npy":
        obj = np.load(path, allow_pickle=True)
        if obj.ndim == 0 and isinstance(obj.item(), dict):
            obj = obj.item()
            for k in ["motion_151", "motion", "poses", "unit_motion", "x"]:
                if k in obj:
                    return clean_motion_151(obj[k], smooth_radius=0, root_smooth_radius=0)
            raise ValueError("dict npy has no motion_151/motion/poses key: %s" % path)
        return clean_motion_151(obj, smooth_radius=0, root_smooth_radius=0)

    if path.suffix.lower() == ".npz":
        z = np.load(path, allow_pickle=True)
        for k in ["motion_151", "motion", "poses", "unit_motion", "unit_motions_physical", "unit_motions"]:
            if k in z.files:
                arr = z[k]
                if arr.ndim == 3:
                    arr = arr[0]
                return clean_motion_151(arr, smooth_radius=0, root_smooth_radius=0)
        raise ValueError("npz has no compatible motion key: %s keys=%s" % (path, z.files))

    if path.suffix.lower() == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            for k in ["motion_151", "motion", "poses", "unit_motion", "x"]:
                if k in obj:
                    return clean_motion_151(obj[k], smooth_radius=0, root_smooth_radius=0)
        arr = np.asarray(obj, dtype=np.float32)
        return clean_motion_151(arr, smooth_radius=0, root_smooth_radius=0)

    raise ValueError("Unsupported motion file: %s" % path)


def load_audio_mono(path, target_sr=16000):
    """Return mono float32 audio and sample rate.

    Uses librosa when available.  Fallback supports WAV via Python wave.
    """
    path = str(path)
    try:
        import librosa
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return np.asarray(y, dtype=np.float32), int(sr)
    except Exception:
        pass

    # Fallback: wav only.
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(n)
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    if target_sr and sr != target_sr:
        # simple linear resampling fallback
        old_t = np.linspace(0.0, 1.0, len(data), dtype=np.float32)
        new_len = int(round(len(data) * float(target_sr) / float(sr)))
        new_t = np.linspace(0.0, 1.0, max(1, new_len), dtype=np.float32)
        data = np.interp(new_t, old_t, data).astype(np.float32)
        sr = int(target_sr)
    return data.astype(np.float32), int(sr)


def audio_onset_curve(audio_path, motion_T, sr=16000, fps=30.0):
    """Frame-aligned onset/energy curve [T]."""
    y, sr = load_audio_mono(audio_path, target_sr=sr)
    T = int(motion_T)
    if T <= 0:
        return np.zeros((0,), dtype=np.float32)

    try:
        import librosa
        hop = max(1, int(round(sr / float(fps))))
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        if len(onset) < 2:
            raise RuntimeError("onset too short")
        x_old = np.linspace(0, T - 1, len(onset), dtype=np.float32)
        x_new = np.arange(T, dtype=np.float32)
        out = np.interp(x_new, x_old, onset).astype(np.float32)
        return norm01(out)
    except Exception:
        # fallback: frame RMS derivative
        frame_len = max(1, int(round(sr / float(fps))))
        vals = []
        for i in range(T):
            s = i * frame_len
            e = min(len(y), s + frame_len)
            if s >= len(y):
                vals.append(0.0)
            else:
                vals.append(float(np.sqrt(np.mean(y[s:e] ** 2) + 1e-8)))
        vals = np.asarray(vals, dtype=np.float32)
        d = np.zeros_like(vals)
        if len(vals) > 1:
            d[1:] = np.maximum(vals[1:] - vals[:-1], 0.0)
            d[0] = d[1]
        return norm01(d)


def motion_energy_curve(motion):
    motion = np.asarray(motion, dtype=np.float32)
    T = motion.shape[0]
    if T <= 1:
        return np.zeros((T,), dtype=np.float32)
    root = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    rot = motion[:, ROT_START:NFEATS]
    lower = motion[:, LOWER_DIMS] if LOWER_DIMS else rot
    upper = motion[:, UPPER_DIMS] if UPPER_DIMS else rot

    root_v = np.zeros((T,), dtype=np.float32)
    rot_v = np.zeros((T,), dtype=np.float32)
    lower_v = np.zeros((T,), dtype=np.float32)
    upper_v = np.zeros((T,), dtype=np.float32)

    root_v[1:] = safe_norm(root[1:] - root[:-1], axis=-1)
    rot_v[1:] = safe_norm(rot[1:] - rot[:-1], axis=-1)
    lower_v[1:] = safe_norm(lower[1:] - lower[:-1], axis=-1)
    upper_v[1:] = safe_norm(upper[1:] - upper[:-1], axis=-1)

    # Jerk/high-frequency component.
    jerk = np.zeros((T,), dtype=np.float32)
    if T > 3:
        rj = np.diff(rot, n=3, axis=0)
        jerk[3:] = safe_norm(rj, axis=-1)

    energy = 0.20 * norm01(root_v) + 0.35 * norm01(rot_v) + 0.20 * norm01(lower_v) + 0.20 * norm01(upper_v) + 0.05 * norm01(jerk)
    return norm01(moving_average_1d(energy, radius=1))


def xcorr_max(a, b, max_lag=4):
    a = norm01(a).astype(np.float32)
    b = norm01(b).astype(np.float32)
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    best = -1.0
    best_lag = 0
    for lag in range(-int(max_lag), int(max_lag) + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[:len(aa)]
        elif lag > 0:
            aa = a[:-lag]
            bb = b[lag:]
        else:
            aa = a
            bb = b
        if len(aa) < 3:
            continue
        val = float(np.dot(aa, bb) / denom)
        if val > best:
            best = val
            best_lag = lag
    return max(0.0, best), int(best_lag)


def top_peak_mask(x, q=80):
    x = norm01(x)
    if len(x) == 0:
        return x
    th = np.percentile(x, q)
    return (x >= th).astype(np.float32)


def audio_motion_alignment(motion, audio_path, fps=30.0):
    onset = audio_onset_curve(audio_path, motion.shape[0], fps=fps)
    meng = motion_energy_curve(motion)

    onset_p = top_peak_mask(onset, q=80)
    motion_p = top_peak_mask(meng, q=80)

    peak_overlap = float(np.mean(onset_p * motion_p))
    dot = float(np.mean(norm01(onset) * norm01(meng)))
    xcorr, lag = xcorr_max(onset, meng, max_lag=4)
    highfreq = float(np.percentile(meng, 90))

    score = 0.35 * dot + 0.35 * xcorr + 0.20 * min(1.0, peak_overlap * 5.0) + 0.10 * highfreq
    score = float(np.clip(score, 0.0, 1.0))

    return {
        "audio_onset_curve": onset.astype(np.float32),
        "motion_energy_curve": meng.astype(np.float32),
        "video_music_sync_score": score,
        "video_onset_peak_score": float(min(1.0, peak_overlap * 5.0)),
        "audio_motion_dot_score": dot,
        "audio_motion_xcorr_score": float(xcorr),
        "audio_motion_best_lag": int(lag),
        "motion_highfreq_score": highfreq,
    }


def unit_basic_stats(unit):
    unit = np.asarray(unit, dtype=np.float32)
    T = unit.shape[0]
    root = unit[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    rot = unit[:, ROT_START:NFEATS]
    lower = unit[:, LOWER_DIMS] if LOWER_DIMS else rot
    upper = unit[:, UPPER_DIMS] if UPPER_DIMS else rot
    contacts = unit[:, CONTACT_SLICE]

    if T > 1:
        root_delta = root[1:] - root[:-1]
        root_speed = float(np.mean(safe_norm(root_delta, axis=-1)))
        lower_activity = float(np.mean(safe_norm(lower[1:] - lower[:-1], axis=-1)))
        upper_activity = float(np.mean(safe_norm(upper[1:] - upper[:-1], axis=-1)))
        motion_energy = float(np.mean(safe_norm(unit[1:, 4:] - unit[:-1, 4:], axis=-1)))
        contact_switch = float(np.mean(np.abs(contacts[1:] - contacts[:-1])))
    else:
        root_speed = lower_activity = upper_activity = motion_energy = contact_switch = 0.0

    spatial_range = float(np.linalg.norm(root.max(axis=0) - root.min(axis=0)))

    turning = 0.0
    if T > 2:
        v = np.zeros_like(root)
        v[1:] = root[1:] - root[:-1]
        a = np.arctan2(v[:, 1], v[:, 0] + 1e-8)
        d = np.zeros((T,), dtype=np.float32)
        raw = a[1:] - a[:-1]
        d[1:] = np.arctan2(np.sin(raw), np.cos(raw))
        turning = float(np.mean(np.abs(d)))

    contact_stability = float(1.0 - min(1.0, contact_switch))
    alternating_foot_phase = float(contact_switch)
    root_lower_sync = float(min(1.0, root_speed / (lower_activity + 1e-6))) if lower_activity > 1e-6 else 0.0
    expressive_raw = 0.55 * upper_activity + 0.20 * turning + 0.15 * motion_energy + 0.10 * spatial_range

    return {
        "motion_energy": motion_energy,
        "root_speed": root_speed,
        "upper_activity": upper_activity,
        "lower_activity": lower_activity,
        "spatial_range": spatial_range,
        "turning": turning,
        "contact_stability": contact_stability,
        "contact_switch": contact_switch,
        "alternating_foot_phase": alternating_foot_phase,
        "root_lower_sync": root_lower_sync,
        "expressiveness_raw": float(expressive_raw),
    }


def root_dir(unit, start, end):
    start = int(np.clip(start, 0, len(unit) - 1))
    end = int(np.clip(end, 0, len(unit) - 1))
    v = unit[end, [ROOT_X_IDX, ROOT_Z_IDX]] - unit[start, [ROOT_X_IDX, ROOT_Z_IDX]]
    n = float(np.linalg.norm(v))
    if n <= 1e-8:
        return np.zeros((2,), dtype=np.float32)
    return (v / n).astype(np.float32)


def deterministic_text_embedding(texts, dim=384):
    """Fallback embedding if TextBridgeEncoder is unavailable."""
    out = np.zeros((len(texts), int(dim)), dtype=np.float32)
    for i, text in enumerate(texts):
        h = hashlib.sha256(str(text).encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "little", signed=False)
        rng = np.random.RandomState(seed)
        v = rng.normal(size=(int(dim),)).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        out[i] = v
    return out.astype(np.float32)


def encode_texts(texts, model_name="BAAI/bge-small-zh-v1.5", device="cpu", fallback_dim=384):
    try:
        from model.text_bridge_encoder import TextBridgeEncoder
        enc = TextBridgeEncoder(model_name=model_name, device=device, fallback_dim=fallback_dim)
        emb = enc.encode(list(texts)).astype(np.float32)
        return emb, getattr(enc, "backend", "text_bridge")
    except Exception:
        try:
            from text_bridge_encoder import TextBridgeEncoder
            enc = TextBridgeEncoder(model_name=model_name, device=device, fallback_dim=fallback_dim)
            emb = enc.encode(list(texts)).astype(np.float32)
            return emb, getattr(enc, "backend", "text_bridge")
        except Exception:
            return deterministic_text_embedding(texts, dim=fallback_dim), "deterministic_hash_fallback"


def read_manifest(path):
    """Read json/jsonl/csv manifest.

    Required fields:
      motion_path,audio_path

    Optional:
      source_id,title,rights_tag,fps,video_path,notes
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "items" in obj:
            return list(obj["items"])
        if isinstance(obj, list):
            return obj
        raise ValueError("JSON manifest must be list or {'items': list}")
    if path.suffix.lower() == ".jsonl":
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out
    if path.suffix.lower() == ".csv":
        import csv
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError("Unsupported manifest type: %s" % path)


def build_unit_records_from_item(
    item,
    unit_len=45,
    stride=15,
    min_unit_len=30,
    clean_smooth_radius=1,
    clean_root_smooth_radius=1,
    freeze_stationary_root=False,
):
    motion_path = item.get("motion_path") or item.get("motion")
    audio_path = item.get("audio_path") or item.get("audio")
    if not motion_path or not audio_path:
        raise ValueError("manifest item requires motion_path and audio_path: %s" % item)

    fps = float(item.get("fps", 30.0) or 30.0)
    source_id = str(item.get("source_id") or Path(motion_path).stem)
    title = str(item.get("title") or source_id)
    rights_tag = str(item.get("rights_tag") or "unknown")
    video_path = str(item.get("video_path") or "")
    notes = str(item.get("notes") or "")

    motion = load_motion_151(motion_path)
    motion = clean_motion_151(
        motion,
        smooth_radius=clean_smooth_radius,
        root_smooth_radius=clean_root_smooth_radius,
        freeze_stationary_root=freeze_stationary_root,
    )
    align = audio_motion_alignment(motion, audio_path, fps=fps)
    onset = align["audio_onset_curve"]
    meng = align["motion_energy_curve"]

    L = min(int(unit_len), len(motion))
    if L < int(min_unit_len):
        return []

    records = []
    for start in range(0, len(motion) - L + 1, int(stride)):
        end = start + L
        unit = motion[start:end].astype(np.float32)
        c = len(unit) // 2
        stats = unit_basic_stats(unit)
        unit_onset = onset[start:end]
        unit_energy = meng[start:end]

        # Unit-local sync uses the same formula but on sliced curves.
        local_dot = float(np.mean(norm01(unit_onset) * norm01(unit_energy)))
        local_xcorr, local_lag = xcorr_max(unit_onset, unit_energy, max_lag=3)
        local_peaks = float(np.mean(top_peak_mask(unit_onset, 80) * top_peak_mask(unit_energy, 80)))
        local_highfreq = float(np.percentile(unit_energy, 90)) if len(unit_energy) else 0.0
        local_sync = float(np.clip(0.35 * local_dot + 0.35 * local_xcorr + 0.20 * min(1.0, local_peaks * 5.0) + 0.10 * local_highfreq, 0.0, 1.0))

        r = {
            "source": str(motion_path),
            "audio_path": str(audio_path),
            "video_path": video_path,
            "source_id": source_id,
            "title": title,
            "rights_tag": rights_tag,
            "notes": notes,
            "fps": fps,
            "unit_start": int(start),
            "unit_center": int(start + c),
            "unit_end": int(end - 1),
            "unit_motion_physical": unit,
            "entry_pose_physical": unit[0].copy(),
            "center_pose_physical": unit[c].copy(),
            "exit_pose_physical": unit[-1].copy(),
            "contact_entry": (unit[0, CONTACT_SLICE] > 0.5).astype(np.float32),
            "contact_center": (unit[c, CONTACT_SLICE] > 0.5).astype(np.float32),
            "contact_exit": (unit[-1, CONTACT_SLICE] > 0.5).astype(np.float32),
            "root_dir_entry": root_dir(unit, 0, min(5, len(unit) - 1)),
            "root_dir_exit": root_dir(unit, max(0, len(unit) - 6), len(unit) - 1),
            "root_dir_full": root_dir(unit, 0, len(unit) - 1),
            "stats": stats,
            "video_music_sync_score_raw": local_sync,
            "video_onset_peak_score_raw": float(min(1.0, local_peaks * 5.0)),
            "motion_highfreq_score_raw": float(local_highfreq),
            "audio_motion_dot_score_raw": float(local_dot),
            "audio_motion_xcorr_score_raw": float(local_xcorr),
            "audio_motion_best_lag": int(local_lag),
            "global_video_music_sync_score": float(align["video_music_sync_score"]),
        }
        records.append(r)
    return records


def compute_stats_arrays(records):
    keys = [
        "motion_energy", "root_speed", "upper_activity", "lower_activity",
        "spatial_range", "turning", "contact_stability", "contact_switch",
        "alternating_foot_phase", "root_lower_sync", "expressiveness_raw",
    ]
    raw = {}
    for k in keys:
        raw[k] = np.asarray([float(r["stats"].get(k, 0.0)) for r in records], dtype=np.float32)

    out = {}
    for k, v in raw.items():
        out[k] = v.astype(np.float32)
        out[k + "_norm"] = robust_norm(v)

    # Existing DB-compatible score fields.
    out["unit_energy"] = out["motion_energy"].copy()
    out["unit_energy_norm"] = out["motion_energy_norm"].copy()

    out["expressiveness_score"] = robust_norm(raw["expressiveness_raw"])
    out["locomotion_score"] = np.clip(
        0.45 * out["root_speed_norm"]
        + 0.35 * out["lower_activity_norm"]
        + 0.15 * out["spatial_range_norm"]
        + 0.05 * out["turning_norm"],
        0.0,
        1.0,
    ).astype(np.float32)
    out["footstep_score"] = np.clip(
        0.35 * out["contact_switch_norm"]
        + 0.30 * out["alternating_foot_phase_norm"]
        + 0.20 * out["root_lower_sync_norm"]
        + 0.15 * out["contact_stability_norm"],
        0.0,
        1.0,
    ).astype(np.float32)
    out["mobile_score"] = np.clip(0.60 * out["locomotion_score"] + 0.40 * out["footstep_score"], 0.0, 1.0).astype(np.float32)

    # Functional selector-compatible aliases.
    out["speed_lower_sync"] = out["root_lower_sync"].copy()
    out["speed_lower_sync_norm"] = out["root_lower_sync_norm"].copy()
    out["support_context_score"] = np.clip(
        0.35 * out["mobile_score"]
        + 0.30 * out["footstep_score"]
        + 0.20 * out["root_lower_sync_norm"]
        + 0.15 * out["contact_stability_norm"],
        0.0,
        1.0,
    ).astype(np.float32)
    out["turn_expression_response"] = np.clip(
        0.50 * out["turning_norm"] + 0.50 * out["upper_activity_norm"],
        0.0,
        1.0,
    ).astype(np.float32)
    out["turn_expression_response_norm"] = robust_norm(out["turn_expression_response"])
    out["expressive_mobile_score"] = np.clip(
        0.55 * out["expressiveness_score"]
        + 0.25 * out["mobile_score"]
        + 0.20 * out["turn_expression_response_norm"],
        0.0,
        1.0,
    ).astype(np.float32)
    out["mobile_expressive_score"] = np.clip(
        0.50 * out["mobile_score"] + 0.50 * out["expressiveness_score"],
        0.0,
        1.0,
    ).astype(np.float32)
    out["functional_coupling_score"] = np.clip(
        0.40 * out["root_lower_sync_norm"]
        + 0.30 * out["turn_expression_response_norm"]
        + 0.30 * out["footstep_score"],
        0.0,
        1.0,
    ).astype(np.float32)
    out["speed_expression_sync_norm"] = np.clip(
        0.50 * out["root_speed_norm"] + 0.50 * out["upper_activity_norm"],
        0.0,
        1.0,
    ).astype(np.float32)

    # New in-the-wild video sync fields.
    for new_key in [
        "video_music_sync_score",
        "video_onset_peak_score",
        "motion_highfreq_score",
        "audio_motion_dot_score",
        "audio_motion_xcorr_score",
        "global_video_music_sync_score",
    ]:
        raw_name = new_key + "_raw"
        vals = np.asarray([float(r.get(raw_name, r.get(new_key, 0.0))) for r in records], dtype=np.float32)
        out[new_key] = np.clip(vals, 0.0, 1.0).astype(np.float32)
        out[new_key + "_norm"] = robust_norm(vals)

    out["audio_motion_best_lag"] = np.asarray([int(r.get("audio_motion_best_lag", 0)) for r in records], dtype=np.int64)
    out["is_inwild_video"] = np.ones((len(records),), dtype=np.float32)

    # New combined retrieval fields.
    out["video_expressive_sync_score"] = np.clip(
        0.40 * out["video_music_sync_score_norm"]
        + 0.30 * out["motion_highfreq_score_norm"]
        + 0.20 * out["expressiveness_score"]
        + 0.10 * out["turn_expression_response_norm"],
        0.0,
        1.0,
    ).astype(np.float32)
    out["video_support_sync_score"] = np.clip(
        0.35 * out["video_music_sync_score_norm"]
        + 0.25 * out["video_onset_peak_score_norm"]
        + 0.20 * out["footstep_score"]
        + 0.20 * out["root_lower_sync_norm"],
        0.0,
        1.0,
    ).astype(np.float32)
    return out


def captions_from_records(records, stats):
    captions = []
    for i, r in enumerate(records):
        sync = float(stats["video_music_sync_score"][i])
        high = float(stats["motion_highfreq_score"][i])
        expr = float(stats["expressiveness_score"][i])
        mobile = float(stats["mobile_score"][i])
        title = r.get("title", "")
        if sync >= 0.65:
            sync_word = "强音乐卡点"
        elif sync >= 0.40:
            sync_word = "中等音乐卡点"
        else:
            sync_word = "弱音乐卡点"
        high_word = "高频爆发动作" if high >= 0.65 else ("中频动作变化" if high >= 0.35 else "平缓动作")
        expr_word = "高表现力上肢" if expr >= 0.65 else ("中等表现力" if expr >= 0.35 else "含蓄表达")
        mobile_word = "移动/转身明显" if mobile >= 0.55 else "原地造型为主"
        captions.append(
            "%s，%s，%s，%s，网络视频弱监督，敦煌舞，飞天风格，%s"
            % (sync_word, high_word, expr_word, mobile_word, title)
        )
    return captions
