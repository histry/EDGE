#!/usr/bin/env python3
"""Build a lightweight stats cache for V10 ChoreoRAG planners.

This script reads an existing ChoreoRAG motion-unit DB (.npz) and writes a
separate *_stats.npz cache with per-unit statistics used by energy-aware and
expressiveness-aware planning.

It is intentionally format-tolerant.  It accepts DBs containing one of:
- unit_motions / unit_motions_physical: [N,T,151]
- motions / clips / x / units: [N,T,151] or [N,151,T]

Recommended:
python build_choreorag_stats_cache.py \
  --rag_db data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco.npz \
  --out data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_stats.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

REPR_DIM = 151
CONTACT_SLICE = slice(0, 4)
ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
ROOT_XZ_IDX = [ROOT_X_IDX, ROOT_Z_IDX]
ROT_START = 7
ROT_SLICE = slice(7, 151)
N_JOINTS = 24
ROT_DIM = 6
TORSO_JOINTS = [3, 6, 9]
UPPER_JOINTS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]


def _joint_rot_indices(joints: Iterable[int]) -> np.ndarray:
    idx = []
    for j in joints:
        j = int(j)
        idx.extend(range(ROT_START + ROT_DIM * j, ROT_START + ROT_DIM * j + ROT_DIM))
    return np.asarray([i for i in idx if 0 <= i < REPR_DIM], dtype=np.int64)


UPPER_ROT_INDEX = _joint_rot_indices(list(TORSO_JOINTS) + list(UPPER_JOINTS))
LOWER_ROT_INDEX = _joint_rot_indices(LOWER_JOINTS)


def as_unit_t151(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected [T,151] or [151,T], got {arr.shape}")
    if arr.shape[-1] == REPR_DIM:
        return arr
    if arr.shape[0] == REPR_DIM:
        return arr.T
    raise ValueError(f"Expected one dim to be {REPR_DIM}, got {arr.shape}")


def load_units_from_npz(path: str) -> Tuple[np.ndarray, str]:
    npz = np.load(path, allow_pickle=True)
    preferred = [
        "unit_motions",
        "unit_motions_physical",
        "motions",
        "motion_units",
        "clips",
        "units",
        "x",
    ]
    for key in preferred + list(npz.files):
        if key not in npz.files:
            continue
        arr = np.asarray(npz[key])
        if arr.dtype == object:
            continue
        if arr.ndim == 3 and (arr.shape[-1] == REPR_DIM or arr.shape[1] == REPR_DIM):
            if arr.shape[-1] == REPR_DIM:
                return arr.astype(np.float32), key
            return np.transpose(arr, (0, 2, 1)).astype(np.float32), key
    raise ValueError(f"No [N,T,151] unit array found in {path}. keys={npz.files}")


def robust_norm(values: np.ndarray, lo: float | None = None, hi: float | None = None) -> Tuple[np.ndarray, float, float]:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if lo is None or hi is None:
        lo, hi = np.percentile(values, [10, 90]) if values.size else (0.0, 1.0)
    lo = float(lo)
    hi = float(hi)
    if hi - lo <= 1e-8:
        lo = float(values.min()) if values.size else 0.0
        hi = float(values.max() + 1e-6) if values.size else 1.0
    return np.clip((values - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(np.float32), lo, hi


def unit_stats(unit: np.ndarray) -> Dict[str, float]:
    unit = as_unit_t151(unit)
    if len(unit) <= 1:
        return dict(
            unit_energy=0.0,
            motion_energy=0.0,
            upper_activity=0.0,
            lower_activity=0.0,
            pose_diversity=0.0,
            spatial_range=0.0,
            turning=0.0,
            root_speed=0.0,
            contact_change=0.0,
            contact_stability=1.0,
        )

    diff = unit[1:] - unit[:-1]
    rot_diff = diff[:, ROT_SLICE]
    root_xz = unit[:, ROOT_XZ_IDX]
    root_vel = root_xz[1:] - root_xz[:-1]
    rot = unit[:, ROT_SLICE]
    rot_center = rot.mean(axis=0, keepdims=True)

    contacts = (unit[:, CONTACT_SLICE] > 0.5).astype(np.float32)
    contact_change = float(np.mean(np.abs(contacts[1:] - contacts[:-1]))) if len(contacts) > 1 else 0.0
    contact_stability = float(np.clip(1.0 - contact_change, 0.0, 1.0))

    if len(root_vel) >= 2:
        v1, v2 = root_vel[:-1], root_vel[1:]
        n1 = np.linalg.norm(v1, axis=-1)
        n2 = np.linalg.norm(v2, axis=-1)
        cos = np.sum(v1 * v2, axis=-1) / np.clip(n1 * n2, 1e-8, None)
        turning = float(np.mean(1.0 - np.clip(cos, -1.0, 1.0)))
    else:
        turning = 0.0

    return dict(
        unit_energy=float(np.mean(np.linalg.norm(rot_diff, axis=-1))),
        motion_energy=float(np.sqrt(np.mean(rot_diff ** 2))),
        upper_activity=float(np.sqrt(np.mean(diff[:, UPPER_ROT_INDEX] ** 2))) if UPPER_ROT_INDEX.size else 0.0,
        lower_activity=float(np.sqrt(np.mean(diff[:, LOWER_ROT_INDEX] ** 2))) if LOWER_ROT_INDEX.size else 0.0,
        pose_diversity=float(np.mean(np.linalg.norm(rot - rot_center, axis=-1))),
        spatial_range=float(np.linalg.norm(root_xz.max(axis=0) - root_xz.min(axis=0))),
        turning=float(max(0.0, turning)),
        root_speed=float(np.mean(np.linalg.norm(root_vel, axis=-1))) if len(root_vel) else 0.0,
        contact_change=float(np.clip(contact_change, 0.0, 1.0)),
        contact_stability=contact_stability,
    )


def compute_stats(units: np.ndarray) -> Dict[str, np.ndarray]:
    rows = [unit_stats(units[i]) for i in range(units.shape[0])]
    keys = list(rows[0].keys()) if rows else []
    out: Dict[str, np.ndarray] = {k: np.asarray([r[k] for r in rows], dtype=np.float32) for k in keys}

    ranges = {}
    for k in [
        "unit_energy",
        "motion_energy",
        "upper_activity",
        "lower_activity",
        "pose_diversity",
        "spatial_range",
        "turning",
        "root_speed",
        "contact_change",
    ]:
        norm, lo, hi = robust_norm(out[k])
        out[f"{k}_norm"] = norm
        ranges[k] = [lo, hi]

    # Expressiveness is intentionally upper/space/turn/energy-heavy and only weakly root-speed-heavy.
    expressiveness = (
        0.30 * out["unit_energy_norm"]
        + 0.30 * out["upper_activity_norm"]
        + 0.20 * out["spatial_range_norm"]
        + 0.15 * out["turning_norm"]
        + 0.05 * out["root_speed_norm"]
    ).astype(np.float32)
    expressiveness = expressiveness * np.clip(0.50 + 0.50 * out["contact_stability"], 0.0, 1.0)
    out["expressiveness_score"] = np.clip(expressiveness, 0.0, 1.0).astype(np.float32)
    out["_ranges_json"] = np.asarray([json.dumps(ranges, ensure_ascii=False)])
    return out


def maybe_copy(npz: np.lib.npyio.NpzFile, key: str):
    if key in npz.files:
        return np.asarray(npz[key])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag_db", required=True, help="Existing ChoreoRAG DB .npz")
    parser.add_argument("--out", default="", help="Output stats cache .npz. Default: <rag_db stem>_stats.npz")
    parser.add_argument("--copy_text_embeddings", action="store_true", help="Copy motion_text_embedding/motion_embedding into the stats cache")
    args = parser.parse_args()

    rag_db = Path(args.rag_db)
    out = Path(args.out) if args.out else rag_db.with_name(rag_db.stem + "_stats.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    units, unit_key = load_units_from_npz(str(rag_db))
    stats = compute_stats(units)

    npz = np.load(str(rag_db), allow_pickle=True)
    save_dict: Dict[str, np.ndarray] = dict(stats)
    save_dict["entry_pose"] = units[:, 0].astype(np.float32)
    save_dict["exit_pose"] = units[:, -1].astype(np.float32)
    save_dict["center_pose"] = units[:, units.shape[1] // 2].astype(np.float32)
    save_dict["unit_index"] = np.arange(units.shape[0], dtype=np.int64)
    save_dict["source_unit_key"] = np.asarray([unit_key])

    for key in ["source", "source_frame", "unit_start", "unit_center", "unit_end", "motion_text"]:
        value = maybe_copy(npz, key)
        if value is not None and len(value) == units.shape[0]:
            save_dict[key] = value

    if args.copy_text_embeddings:
        for key in ["motion_text_embedding", "motion_embedding", "text_embedding"]:
            value = maybe_copy(npz, key)
            if value is not None and value.shape[0] == units.shape[0]:
                save_dict[key] = value

    np.savez_compressed(out, **save_dict)
    meta = {
        "rag_db": str(rag_db),
        "out": str(out),
        "unit_key": unit_key,
        "count": int(units.shape[0]),
        "unit_len": int(units.shape[1]),
        "fields": sorted(save_dict.keys()),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Stats cache saved: {out}")
    print(f"   units={units.shape[0]} unit_len={units.shape[1]} unit_key={unit_key}")
    print(f"   expressiveness mean={float(stats['expressiveness_score'].mean()):.4f} p90={float(np.percentile(stats['expressiveness_score'], 90)):.4f}")


if __name__ == "__main__":
    main()
