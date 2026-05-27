#!/usr/bin/env python3
"""Build a physics-aware prior pool for EDGE-Dunhuang onset phrase scheduling.

Supported inputs:
  1) .npz RAG DB containing one of:
       unit_motions_physical, unit_motions, motions, motion, poses
  2) a directory recursively containing .npy/.npz/.pkl motion files.

Output:
  .npz with motions [N,T,151], entry/exit state vectors, unit ids and scores.

Example:
  python tools/build_physics_aware_prior_pool.py \
    --input data/dunhuang_choreo_unit_rag/index_v13_functional_u45_s15.npz \
    --out data/dunhuang_choreo_unit_rag/v15_physics_prior_pool.npz \
    --min_activity 0.0005 --max_root_radius 0.30
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from choreorag_physics_state import (
    CONTACT_SLICE,
    ROOT_X_IDX,
    ROOT_Z_IDX,
    ROT_START,
    boundary_state,
    ensure_motion_2d,
    ensure_motion_bank,
    local_jump_metrics,
)

MOTION_KEYS = (
    "unit_motions_physical",
    "unit_motions",
    "motions",
    "motion",
    "poses",
    "arr_0",
)
SCORE_KEYS = (
    "quality_score",
    "support_prior_score",
    "support_score",
    "hf_event_score",
    "scores",
    "score",
    "energy",
    "expressiveness",
)
ID_KEYS = ("unit_ids", "unit_id", "ids", "indices", "source_indices")


def _load_dict_like(path: Path):
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as f:
            return pickle.load(f)
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=True)
        if arr.ndim == 0 and isinstance(arr.item(), dict):
            return arr.item()
        return {"motion": arr}
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    raise ValueError(f"Unsupported file: {path}")


def _extract_motions(record: Dict, path: Path) -> np.ndarray:
    for key in MOTION_KEYS:
        if key in record:
            arr = record[key]
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.ndim == 0:
                arr = arr.item()
            return ensure_motion_bank(arr, name=f"{path}:{key}")
    raise KeyError(f"No motion key found in {path}; tried {MOTION_KEYS}")


def _extract_ids(record: Dict, n: int, stem: str) -> np.ndarray:
    for key in ID_KEYS:
        if key in record:
            ids = np.asarray(record[key]).reshape(-1)
            if len(ids) >= n:
                return np.array([str(x) for x in ids[:n]], dtype=object)
    return np.array([f"{stem}_{i:06d}" for i in range(n)], dtype=object)


def _extract_scores(record: Dict, n: int) -> np.ndarray:
    score = np.zeros((n,), dtype=np.float32)
    count = 0
    for key in SCORE_KEYS:
        if key not in record:
            continue
        try:
            arr = np.asarray(record[key], dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if len(arr) < n:
            continue
        arr = arr[:n]
        # Normalize each score source robustly before averaging.
        lo, hi = float(np.percentile(arr, 5)), float(np.percentile(arr, 95))
        if hi > lo + 1e-8:
            arr = (arr - lo) / (hi - lo)
        arr = np.clip(arr, 0.0, 1.0)
        score += arr.astype(np.float32)
        count += 1
    if count == 0:
        return np.ones((n,), dtype=np.float32) * 0.5
    return (score / float(count)).astype(np.float32)


def _motion_activity(motion: np.ndarray) -> float:
    m = ensure_motion_2d(motion)
    if len(m) < 2:
        return 0.0
    rot = np.linalg.norm(m[1:, ROT_START:] - m[:-1, ROT_START:], axis=-1).mean()
    contacts = np.abs(m[1:, CONTACT_SLICE] - m[:-1, CONTACT_SLICE]).mean()
    return float(rot + 0.05 * contacts)


def _root_radius(motion: np.ndarray) -> float:
    root = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    return float(np.linalg.norm(root - root[:1], axis=-1).max())


def load_pool_inputs(input_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    motions_list: List[np.ndarray] = []
    ids_list: List[str] = []
    scores_list: List[float] = []
    sources: List[str] = []

    if input_path.is_dir():
        files = sorted(
            list(input_path.rglob("*.npy"))
            + list(input_path.rglob("*.npz"))
            + list(input_path.rglob("*.pkl"))
        )
        if not files:
            raise FileNotFoundError(f"No .npy/.npz/.pkl files found under {input_path}")
        for path in files:
            try:
                rec = _load_dict_like(path)
                motions = _extract_motions(rec, path)
                ids = _extract_ids(rec, len(motions), path.stem)
                scores = _extract_scores(rec, len(motions))
            except Exception as exc:
                print(f"⚠️ skip {path}: {exc}")
                continue
            for i, motion in enumerate(motions):
                motions_list.append(ensure_motion_2d(motion))
                ids_list.append(str(ids[i]))
                scores_list.append(float(scores[i]))
                sources.append(str(path))
    else:
        rec = _load_dict_like(input_path)
        motions = _extract_motions(rec, input_path)
        ids = _extract_ids(rec, len(motions), input_path.stem)
        scores = _extract_scores(rec, len(motions))
        for i, motion in enumerate(motions):
            motions_list.append(ensure_motion_2d(motion))
            ids_list.append(str(ids[i]))
            scores_list.append(float(scores[i]))
            sources.append(str(input_path))

    if not motions_list:
        raise RuntimeError("No valid motions loaded")
    # Use minimum length for a rectangular bank. Most project units are 45 frames.
    min_len = min(len(m) for m in motions_list)
    motions = np.stack([m[:min_len].astype(np.float32) for m in motions_list], axis=0)
    return motions, np.asarray(ids_list, dtype=object), np.asarray(scores_list, dtype=np.float32), sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="RAG .npz or directory of motion files")
    ap.add_argument("--out", required=True, help="Output .npz prior pool")
    ap.add_argument("--min_activity", type=float, default=0.0002)
    ap.add_argument("--max_root_radius", type=float, default=0.50)
    ap.add_argument("--max_jump_p95", type=float, default=999.0)
    ap.add_argument("--prefer_inplace", action="store_true", help="Keep only low-root-radius units")
    args = ap.parse_args()

    motions, unit_ids, base_scores, sources = load_pool_inputs(Path(args.input))
    keep = []
    reports = []
    for i, motion in enumerate(motions):
        activity = _motion_activity(motion)
        radius = _root_radius(motion)
        metrics = local_jump_metrics(motion, [])
        jump = metrics.get("global_rot_jump_p95", 0.0)
        ok = activity >= args.min_activity and jump <= args.max_jump_p95
        if args.prefer_inplace:
            ok = ok and radius <= args.max_root_radius
        else:
            ok = ok and radius <= max(args.max_root_radius, 1e-6)
        if ok:
            keep.append(i)
        reports.append({
            "i": int(i),
            "unit_id": str(unit_ids[i]),
            "activity": float(activity),
            "root_radius": float(radius),
            "jump_p95": float(jump),
            "base_score": float(base_scores[i]),
            "keep": bool(ok),
            "source": sources[i] if i < len(sources) else "",
        })

    if not keep:
        print("⚠️ No units passed filters; keeping top-16 by activity as fallback.")
        order = np.argsort([-r["activity"] for r in reports])[: min(16, len(reports))]
        keep = [int(i) for i in order]

    motions_k = motions[keep]
    unit_ids_k = unit_ids[keep]
    scores_k = base_scores[keep]
    entry = np.stack([boundary_state(m, 0).to_vector() for m in motions_k], axis=0)
    exit_ = np.stack([boundary_state(m, len(m) - 1).to_vector() for m in motions_k], axis=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        motions=motions_k.astype(np.float32),
        entry_state=entry.astype(np.float32),
        exit_state=exit_.astype(np.float32),
        unit_ids=unit_ids_k.astype(object),
        base_scores=scores_k.astype(np.float32),
        source_input=str(args.input),
    )
    report_path = out.with_suffix(".report.json")
    report_path.write_text(json.dumps({
        "input": str(args.input),
        "out": str(out),
        "num_loaded": int(len(motions)),
        "num_kept": int(len(keep)),
        "filters": {
            "min_activity": args.min_activity,
            "max_root_radius": args.max_root_radius,
            "max_jump_p95": args.max_jump_p95,
            "prefer_inplace": args.prefer_inplace,
        },
        "kept_head": [reports[i] for i in keep[:20]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ saved physics-aware prior pool: {out} | kept {len(keep)}/{len(motions)}")
    print(f"✅ report: {report_path}")


if __name__ == "__main__":
    main()
