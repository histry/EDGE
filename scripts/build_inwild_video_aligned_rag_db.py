#!/usr/bin/env python3
"""Build in-the-wild video-aligned ChoreoRAG DB.

Input: a manifest of rights-cleared / permitted videos whose audio and 151-D
motion have already been extracted.

Example manifest CSV:

source_id,title,motion_path,audio_path,video_path,rights_tag,fps
silkroad_001,Silk Road clip,output/inwild/silkroad_001_motion151.npy,output/inwild/silkroad_001.wav,/data/videos/silkroad_001.mp4,owned_or_permitted,30

This script does NOT download videos.  It only builds derived motion/audio
alignment features for retrieval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inwild_video_alignment_utils import (
    captions_from_records,
    compute_stats_arrays,
    encode_texts,
    read_manifest,
    build_unit_records_from_item,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="json/jsonl/csv manifest with motion_path,audio_path")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--unit_len", type=int, default=45)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--min_unit_len", type=int, default=30)
    ap.add_argument("--text_model", type=str, default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--text_device", type=str, default="cpu")
    ap.add_argument("--fallback_dim", type=int, default=384)
    ap.add_argument("--max_units", type=int, default=0)
    ap.add_argument("--smooth_radius", type=int, default=1)
    ap.add_argument("--root_smooth_radius", type=int, default=1)
    ap.add_argument("--freeze_stationary_root", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    items = read_manifest(args.manifest)
    if not items:
        raise RuntimeError("empty manifest: %s" % args.manifest)

    records = []
    errors = []
    for item in items:
        try:
            recs = build_unit_records_from_item(
                item,
                unit_len=args.unit_len,
                stride=args.stride,
                min_unit_len=args.min_unit_len,
                clean_smooth_radius=args.smooth_radius,
                clean_root_smooth_radius=args.root_smooth_radius,
                freeze_stationary_root=args.freeze_stationary_root,
            )
            records.extend(recs)
        except Exception as exc:
            errors.append({"item": item, "error": "%s: %s" % (type(exc).__name__, exc)})

    if args.max_units > 0:
        records = records[: int(args.max_units)]

    if not records:
        raise RuntimeError("No valid units built. errors=%s" % errors[:3])

    stats = compute_stats_arrays(records)
    captions = captions_from_records(records, stats)
    emb, backend = encode_texts(
        captions,
        model_name=args.text_model,
        device=args.text_device,
        fallback_dim=args.fallback_dim,
    )

    unit_physical = np.stack([r["unit_motion_physical"] for r in records], axis=0).astype(np.float32)
    poses = np.stack([r["center_pose_physical"] for r in records], axis=0).astype(np.float32)
    entry = np.stack([r["entry_pose_physical"] for r in records], axis=0).astype(np.float32)
    exitp = np.stack([r["exit_pose_physical"] for r in records], axis=0).astype(np.float32)

    np.savez_compressed(
        out,
        db_type=np.asarray(["inwild_video_aligned_choreo_unit_rag"]),
        pose_space=np.asarray(["physical"]),
        unit_len=np.asarray([args.unit_len], dtype=np.int64),
        poses=poses.astype(np.float32),
        entry_poses=entry.astype(np.float32),
        exit_poses=exitp.astype(np.float32),
        unit_motions=unit_physical.astype(np.float32),
        unit_motions_physical=unit_physical.astype(np.float32),
        source=np.asarray([r["source"] for r in records]),
        source_frame=np.asarray([r["unit_center"] for r in records], dtype=np.int64),
        unit_start=np.asarray([r["unit_start"] for r in records], dtype=np.int64),
        unit_center=np.asarray([r["unit_center"] for r in records], dtype=np.int64),
        unit_end=np.asarray([r["unit_end"] for r in records], dtype=np.int64),
        root_vel=np.stack([r["root_dir_full"] for r in records], axis=0).astype(np.float32),
        root_dir_entry=np.stack([r["root_dir_entry"] for r in records], axis=0).astype(np.float32),
        root_dir_exit=np.stack([r["root_dir_exit"] for r in records], axis=0).astype(np.float32),
        contact_entry=np.stack([r["contact_entry"] for r in records], axis=0).astype(np.float32),
        contact_center=np.stack([r["contact_center"] for r in records], axis=0).astype(np.float32),
        contact_exit=np.stack([r["contact_exit"] for r in records], axis=0).astype(np.float32),
        motion_text=np.asarray(captions),
        motion_text_embedding=emb.astype(np.float32),
        motion_embedding=emb.astype(np.float32),
        text_model=np.asarray([args.text_model]),
        text_backend=np.asarray([backend]),
        audio_path=np.asarray([r.get("audio_path", "") for r in records]),
        video_path=np.asarray([r.get("video_path", "") for r in records]),
        source_id=np.asarray([r.get("source_id", "") for r in records]),
        title=np.asarray([r.get("title", "") for r in records]),
        rights_tag=np.asarray([r.get("rights_tag", "unknown") for r in records]),
        fps=np.asarray([float(r.get("fps", 30.0)) for r in records], dtype=np.float32),
        notes=np.asarray([r.get("notes", "") for r in records]),
        **stats,
    )

    meta = {
        "db_type": "inwild_video_aligned_choreo_unit_rag",
        "manifest": args.manifest,
        "out": str(out),
        "count": len(records),
        "unit_len": args.unit_len,
        "stride": args.stride,
        "text_backend": backend,
        "errors": errors[:50],
        "fields_added": [
            "video_music_sync_score",
            "video_onset_peak_score",
            "motion_highfreq_score",
            "audio_motion_dot_score",
            "audio_motion_xcorr_score",
            "video_expressive_sync_score",
            "video_support_sync_score",
            "is_inwild_video",
            "rights_tag",
        ],
        "rights_note": "Only use videos you own, have permission to use, or are licensed for this research use. Do not redistribute raw videos.",
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ In-wild video-aligned ChoreoRAG DB saved:", out)
    print("   units=", len(records), "unit_len=", args.unit_len, "stride=", args.stride)
    print("   video_music_sync mean=%.3f p90=%.3f" % (
        float(np.mean(stats["video_music_sync_score"])),
        float(np.percentile(stats["video_music_sync_score"], 90)),
    ))
    print("   video_expressive_sync mean=%.3f p90=%.3f" % (
        float(np.mean(stats["video_expressive_sync_score"])),
        float(np.percentile(stats["video_expressive_sync_score"], 90)),
    ))
    if errors:
        print("⚠️ skipped items:", len(errors), "see", out.with_suffix(".meta.json"))


if __name__ == "__main__":
    main()
