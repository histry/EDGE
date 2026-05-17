#!/usr/bin/env python3
"""Merge multiple ChoreoRAG DB .npz files.

This is intended to merge the original Dunhuang motion-unit DB with the new
in-the-wild video-aligned DB.  Missing per-unit fields are filled with safe
defaults so older selectors remain compatible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCALAR_KEYS = {"db_type", "pose_space", "unit_len", "text_model", "text_backend"}


def db_len(db):
    for k in ["unit_motions", "unit_motions_physical", "poses"]:
        if k in db.files:
            return int(len(db[k]))
    raise KeyError("Cannot infer DB length from %s" % (db.files,))


def is_per_unit(arr, n):
    return hasattr(arr, "shape") and arr.ndim >= 1 and int(arr.shape[0]) == int(n)


def default_like(key, exemplar, n):
    arr = np.asarray(exemplar)
    shape = (int(n),) + tuple(arr.shape[1:])
    if arr.dtype.kind in {"U", "S", "O"}:
        return np.asarray([""] * int(np.prod(shape)), dtype=arr.dtype).reshape(shape)
    if arr.dtype.kind in {"i", "u"}:
        return np.zeros(shape, dtype=arr.dtype)
    return np.zeros(shape, dtype=np.float32)


def choose_exemplar(key, dbs, lengths):
    for db, n in zip(dbs, lengths):
        if key in db.files and is_per_unit(db[key], n):
            return db[key]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--db_type", default="merged_choreo_unit_rag_with_inwild_video")
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    dbs = [np.load(str(p), allow_pickle=True) for p in paths]
    lengths = [db_len(db) for db in dbs]
    total = int(sum(lengths))

    out_data = {}
    all_keys = set()
    for db in dbs:
        all_keys.update(db.files)

    for key in sorted(all_keys):
        if key in SCALAR_KEYS:
            continue
        ex = choose_exemplar(key, dbs, lengths)
        if ex is None:
            continue

        parts = []
        for db, n in zip(dbs, lengths):
            if key in db.files and is_per_unit(db[key], n):
                arr = db[key]
                # Try to cast to exemplar dtype where safe.
                try:
                    if np.asarray(ex).dtype.kind not in {"U", "S", "O"}:
                        arr = np.asarray(arr, dtype=np.asarray(ex).dtype)
                except Exception:
                    pass
                parts.append(arr)
            else:
                parts.append(default_like(key, ex, n))
        try:
            out_data[key] = np.concatenate(parts, axis=0)
        except Exception as exc:
            print("⚠️ skip incompatible key", key, exc)

    # Ensure important new fields exist for old DB entries.
    n = total
    for key in [
        "is_inwild_video",
        "video_music_sync_score",
        "video_music_sync_score_norm",
        "video_onset_peak_score",
        "video_onset_peak_score_norm",
        "motion_highfreq_score",
        "motion_highfreq_score_norm",
        "video_expressive_sync_score",
        "video_support_sync_score",
        "audio_motion_xcorr_score",
        "audio_motion_dot_score",
    ]:
        if key not in out_data:
            out_data[key] = np.zeros((n,), dtype=np.float32)

    for key in ["audio_path", "video_path", "source_id", "title", "rights_tag", "notes"]:
        if key not in out_data:
            out_data[key] = np.asarray([""] * n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out,
        db_type=np.asarray([args.db_type]),
        pose_space=np.asarray(["mixed_or_physical"]),
        unit_len=np.asarray([45], dtype=np.int64),
        **out_data,
    )

    meta = {
        "db_type": args.db_type,
        "inputs": [str(p) for p in paths],
        "input_lengths": lengths,
        "count": total,
        "out": str(out),
        "note": "Merged DB may contain both original curated motion units and in-wild video-aligned units. Use selector env flags to control in-wild contribution.",
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ merged ChoreoRAG DB saved:", out)
    print("   inputs=", len(paths), "total_units=", total)
    print("   meta=", out.with_suffix(".meta.json"))


if __name__ == "__main__":
    main()
