#!/usr/bin/env python3
"""
Retrieved-clip soft prior postprocess for EDGE 151-D motion.

Why this exists
---------------
Current MMR-RAG auto-mid uses a retrieved clip but only injects its CENTER POSE
as a sparse keyframe. That discards the clip's temporal motion, often causing
"pose pulses" or repeated single-leg jumps. This postprocess uses the retrieved
clip as a SOFT TEMPORAL PRIOR: it blends upper-body rotations from the retrieved
clip segment into the generated motion around each auto-mid frame.

Design
------
- Keep root translation unchanged: trajectory anchor remains valid.
- Keep legs mostly unchanged: Leg IK should handle feet after this step.
- Blend upper-body joints only with a smooth cosine window.
- Protect user-defined start/end frames.
- Use checkpoint normalizer to convert normalized RAG/source clips to physical
  space before blending, if needed.

Recommended pipeline
--------------------
1) generate_controlled.py with --auto_mid_keyframes and MMR-RAG
2) postprocess_retrieved_clip_prior.py
3) postprocess_leg_ik.py
4) eval_quantitative.py and render
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

try:
    # Same converter used by build_mmr_rag_db.py. It supports Dunhuang processed
    # pkl files containing {'pos','q'} and returns EDGE [T,151] motion.
    from mmr_data_utils import load_motion_151 as _edge_load_motion_151
except Exception:
    _edge_load_motion_151 = None


ROOT_X_IDX = 4
ROOT_Z_IDX = 6
ROT_START = 7
NUM_JOINTS = 24
ROT_DIM = 6
MOTION_DIM = 151

# SMPL 24-joint convention used by EDGE.
# We blend only torso/head/arms. Root/hips/legs are left for trajectory + IK.
UPPER_BODY_JOINTS = [
    3,   # spine1
    6,   # spine2
    9,   # spine3
    12,  # neck
    13, 14,  # collars
    15,  # head
    16, 17,  # shoulders
    18, 19,  # elbows
    20, 21,  # wrists
    22, 23,  # hands
]

# More conservative: mostly arms, less torso.
ARMS_ONLY_JOINTS = [13, 14, 16, 17, 18, 19, 20, 21, 22, 23]


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


def _as_array_from_dict(data: Dict[str, Any], source_path: str | Path | None = None) -> np.ndarray:
    """Return EDGE 151-D motion from common processed formats.

    Dunhuang processed pkl files usually contain only:
      - pos: [T,3] root translation
      - q:   [T,24,3] axis-angle rotations

    The v1 script only accepted already-vectorized [T,151] arrays, so raw
    Dunhuang source clips were skipped. v2 reuses mmr_data_utils.load_motion_151,
    the same conversion used by build_mmr_rag_db.py.
    """
    candidate_keys = [
        "motion",
        "motion_151",
        "motions",
        "data",
        "x",
        "arr",
        "pose",
        "poses",
        "pose_seq",
    ]

    for key in candidate_keys:
        if key in data:
            arr = np.asarray(data[key])
            if arr.ndim == 2 and arr.shape[-1] == MOTION_DIM:
                return arr.astype(np.float32)

    if "pos" in data and "q" in data:
        if _edge_load_motion_151 is not None and source_path is not None:
            return _edge_load_motion_151(source_path).astype(np.float32)
        raise ValueError(
            "Found {'pos','q'} but cannot convert to [T,151] because "
            "mmr_data_utils.load_motion_151 is unavailable. Make sure "
            "mmr_data_utils.py exists in the EDGE root."
        )

    # Sometimes processed files are nested. Search one level deep.
    for value in data.values():
        if isinstance(value, dict):
            try:
                return _as_array_from_dict(value, source_path=source_path)
            except Exception:
                pass
        else:
            arr = np.asarray(value)
            if arr.ndim == 2 and arr.shape[-1] == MOTION_DIM:
                return arr.astype(np.float32)

    raise ValueError(
        "Could not find or convert a [T,151] motion array in pkl dict. "
        f"Available keys: {list(data.keys())[:20]}"
    )

def load_motion_any(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=True)
        if arr.ndim == 0 and isinstance(arr.item(), dict):
            arr = _as_array_from_dict(arr.item(), source_path=path)
        else:
            arr = np.asarray(arr)
    elif path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        for key in ["motion", "motions", "data", "poses", "arr"]:
            if key in data:
                arr = np.asarray(data[key])
                break
        else:
            raise ValueError(f"No motion-like key found in {path}: {data.files}")
    elif path.suffix.lower() == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            arr = _as_array_from_dict(data, source_path=path)
        else:
            arr = np.asarray(data)
    else:
        raise ValueError(f"Unsupported motion file suffix: {path.suffix}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[-1] != MOTION_DIM:
        raise ValueError(f"Expected [T,{MOTION_DIM}], got {arr.shape} from {path}")
    return arr


def load_normalizer_from_checkpoint(checkpoint: str | Path):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ["normalizer", "Normalizer", "data_normalizer"]:
        if key in ckpt:
            return ckpt[key]
    # Some checkpoints store args/model only. In that case normalization is skipped.
    return None


def maybe_unnormalize_motion(motion: np.ndarray, normalizer) -> np.ndarray:
    if normalizer is None:
        return motion.astype(np.float32)
    out = normalizer.unnormalize(motion)
    if torch.is_tensor(out):
        out = out.detach().cpu().numpy()
    return np.asarray(out, dtype=np.float32)


# -----------------------------------------------------------------------------
# Prior construction
# -----------------------------------------------------------------------------


def parse_keyframes(text: str, num_frames: int) -> List[int]:
    if not text:
        return []
    out = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(round(float(token)))
        out.append(max(0, min(num_frames - 1, idx)))
    return sorted(set(out))


def smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def cosine_window(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones((length,), dtype=np.float32)
    x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return np.sin(np.pi * smoothstep01(x)).astype(np.float32)


def joint_feature_indices(joints: Sequence[int]) -> List[int]:
    idxs: List[int] = []
    for j in joints:
        start = ROT_START + int(j) * ROT_DIM
        idxs.extend(range(start, start + ROT_DIM))
    return idxs


def resample_clip_to_length(clip: np.ndarray, length: int) -> np.ndarray:
    if len(clip) == length:
        return clip.astype(np.float32)
    old_x = np.linspace(0.0, 1.0, len(clip))
    new_x = np.linspace(0.0, 1.0, length)
    out = np.stack(
        [np.interp(new_x, old_x, clip[:, c]) for c in range(clip.shape[1])],
        axis=1,
    )
    return out.astype(np.float32)


def load_auto_plan(plan_path: str | Path) -> Dict[str, Any]:
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    if "auto_keyframes" not in plan:
        raise ValueError(f"auto_mid_plan has no 'auto_keyframes': {plan_path}")
    return plan


def build_clip_prior_from_plan(
    base_motion: np.ndarray,
    plan: Dict[str, Any],
    edge_checkpoint: str | Path | None,
    source_pose_space: str = "auto",
    window: int = 24,
    feature_indices: Sequence[int] | None = None,
    protect_frames: Sequence[int] | None = None,
    protect_width: int = 3,
    max_segments: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Returns:
      prior_value: [T,151]
      prior_mask:  [T,151] soft weights in [0,1]
      debug_segments
    """
    num_frames = base_motion.shape[0]
    prior_sum = np.zeros_like(base_motion, dtype=np.float32)
    prior_wsum = np.zeros_like(base_motion, dtype=np.float32)

    if feature_indices is None:
        feature_indices = joint_feature_indices(UPPER_BODY_JOINTS)
    feature_indices = list(feature_indices)

    normalizer = None
    if source_pose_space == "normalized":
        if edge_checkpoint is None:
            raise ValueError("source_pose_space=normalized requires --edge_checkpoint")
        normalizer = load_normalizer_from_checkpoint(edge_checkpoint)
    elif source_pose_space == "auto":
        # Raw Dunhuang {'pos','q'} pkl files are converted to physical 151-D and
        # must NOT be unnormalized again. Vectorized npy/npz sources may be
        # normalized, so we load the normalizer when available.
        normalizer = load_normalizer_from_checkpoint(edge_checkpoint) if edge_checkpoint else None

    protected = np.zeros((num_frames,), dtype=bool)
    for f in protect_frames or []:
        s = max(0, int(f) - int(protect_width))
        e = min(num_frames, int(f) + int(protect_width) + 1)
        protected[s:e] = True

    debug_segments: List[Dict[str, Any]] = []
    auto_keyframes = plan.get("auto_keyframes", [])
    if max_segments is not None:
        auto_keyframes = auto_keyframes[: int(max_segments)]

    for kf in auto_keyframes:
        target_frame = int(kf["frame"])
        source = Path(kf["source"])
        source_frame = int(kf.get("source_frame", 0))

        try:
            src_motion = load_motion_any(source)
        except Exception as exc:
            print(f"⚠️ skip source={source}: {exc}")
            continue

        effective_space = source_pose_space
        if source_pose_space == "auto":
            # In the current MMR-RAG plan, sources are raw processed pkl files.
            # load_motion_any converts them to physical 151-D.
            effective_space = "physical" if source.suffix.lower() == ".pkl" else "normalized"

        if effective_space == "normalized":
            if normalizer is None:
                raise ValueError(
                    "source_pose_space=normalized/auto for vectorized sources requires --edge_checkpoint"
                )
            src_motion = maybe_unnormalize_motion(src_motion, normalizer)

        # Extract a local source segment around source_frame.
        half = max(1, int(window))
        src_s = max(0, source_frame - half)
        src_e = min(len(src_motion), source_frame + half + 1)
        src_clip = src_motion[src_s:src_e]

        # Target segment around target_frame; boundary-aware.
        tgt_s = max(0, target_frame - half)
        tgt_e = min(num_frames, target_frame + half + 1)
        tgt_len = max(1, tgt_e - tgt_s)
        src_clip = resample_clip_to_length(src_clip, tgt_len)

        weights = cosine_window(tgt_len)[:, None].astype(np.float32)
        # Do not modify protected user keyframe zones.
        weights[protected[tgt_s:tgt_e]] = 0.0

        if float(weights.max(initial=0.0)) <= 1e-8:
            continue

        # Accumulate only selected feature dimensions.
        prior_sum[tgt_s:tgt_e, feature_indices] += (
            src_clip[:, feature_indices] * weights
        )
        prior_wsum[tgt_s:tgt_e, feature_indices] += weights

        debug_segments.append(
            {
                "target_frame": int(target_frame),
                "target_range": [int(tgt_s), int(tgt_e - 1)],
                "source": str(source),
                "source_frame": int(source_frame),
                "source_range": [int(src_s), int(src_e - 1)],
                "effective_source_space": str(effective_space),
                "score": float(kf.get("score", 0.0)),
                "score_parts": kf.get("score_parts", {}),
            }
        )

    prior_value = base_motion.copy().astype(np.float32)
    prior_mask = np.zeros_like(base_motion, dtype=np.float32)
    valid = prior_wsum > 1e-8
    prior_value[valid] = prior_sum[valid] / np.maximum(prior_wsum[valid], 1e-8)
    prior_mask[valid] = np.clip(prior_wsum[valid], 0.0, 1.0)

    return prior_value, prior_mask, debug_segments


def apply_soft_prior(
    motion: np.ndarray,
    prior_value: np.ndarray,
    prior_mask: np.ndarray,
    strength: float = 0.35,
    temporal_smooth: int = 0,
) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    alpha = np.clip(prior_mask * strength, 0.0, 1.0).astype(np.float32)
    out = motion * (1.0 - alpha) + prior_value * alpha

    # Keep root trajectory exactly unchanged.
    out[:, ROOT_X_IDX] = motion[:, ROOT_X_IDX]
    out[:, ROOT_Z_IDX] = motion[:, ROOT_Z_IDX]

    if temporal_smooth and temporal_smooth > 1:
        out = smooth_motion_features(out, mask=(prior_mask > 0), window=int(temporal_smooth))
        out[:, ROOT_X_IDX] = motion[:, ROOT_X_IDX]
        out[:, ROOT_Z_IDX] = motion[:, ROOT_Z_IDX]

    return out.astype(np.float32)


def smooth_motion_features(motion: np.ndarray, mask: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return motion.astype(np.float32)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    out = motion.copy().astype(np.float32)
    # Only smooth features that were touched by the prior.
    touched_features = np.where(mask.any(axis=0))[0]
    for c in touched_features:
        y = np.convolve(np.pad(out[:, c], (pad, pad), mode="edge"), kernel, mode="valid")
        # Blend smoothing only around touched frames to avoid global over-smoothing.
        touched = mask[:, c]
        out[touched, c] = y[touched]
    return out.astype(np.float32)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply retrieved clip soft prior to EDGE generated 151-D motion."
    )
    parser.add_argument("--motion", required=True, help="Generated final motion .npy")
    parser.add_argument("--auto_plan", required=True, help="*_auto_mid_plan.json")
    parser.add_argument("--out", required=True, help="Output motion .npy")
    parser.add_argument(
        "--edge_checkpoint",
        default="",
        help="EDGE checkpoint containing normalizer; required when source clips are normalized.",
    )
    parser.add_argument(
        "--source_pose_space",
        choices=["auto", "normalized", "physical"],
        default="auto",
        help=(
            "Pose space of retrieved source files. Use auto for MMR-RAG plans: "
            "raw Dunhuang .pkl sources with {'pos','q'} are treated as physical; "
            "already-vectorized .npy/.npz can be normalized/physical."
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=24,
        help="Frames on each side of auto-mid frame to borrow from retrieved clip.",
    )
    parser.add_argument(
        "--blend_strength",
        type=float,
        default=0.35,
        help="Soft blend strength for retrieved clip prior.",
    )
    parser.add_argument(
        "--body_part",
        choices=["upper", "arms"],
        default="upper",
        help="Which rotations to blend from retrieved clips.",
    )
    parser.add_argument(
        "--protect_frames",
        default="0,149",
        help="Comma separated user keyframes to keep unchanged, e.g. 0,149.",
    )
    parser.add_argument("--protect_width", type=int, default=3)
    parser.add_argument("--max_segments", type=int, default=0)
    parser.add_argument("--temporal_smooth", type=int, default=5)
    parser.add_argument("--save_prior_assets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion = load_motion_any(args.motion)
    plan = load_auto_plan(args.auto_plan)

    if args.body_part == "arms":
        feature_indices = joint_feature_indices(ARMS_ONLY_JOINTS)
    else:
        feature_indices = joint_feature_indices(UPPER_BODY_JOINTS)

    protect_frames = parse_keyframes(args.protect_frames, len(motion))

    prior_value, prior_mask, debug_segments = build_clip_prior_from_plan(
        base_motion=motion,
        plan=plan,
        edge_checkpoint=args.edge_checkpoint or None,
        source_pose_space=args.source_pose_space,
        window=args.window,
        feature_indices=feature_indices,
        protect_frames=protect_frames,
        protect_width=args.protect_width,
        max_segments=(args.max_segments if args.max_segments > 0 else None),
    )

    out_motion = apply_soft_prior(
        motion=motion,
        prior_value=prior_value,
        prior_mask=prior_mask,
        strength=args.blend_strength,
        temporal_smooth=args.temporal_smooth,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out_motion.astype(np.float32))

    debug = {
        "motion": str(args.motion),
        "auto_plan": str(args.auto_plan),
        "out": str(out_path),
        "source_pose_space": args.source_pose_space,
        "window": int(args.window),
        "blend_strength": float(args.blend_strength),
        "body_part": args.body_part,
        "protect_frames": protect_frames,
        "protect_width": int(args.protect_width),
        "temporal_smooth": int(args.temporal_smooth),
        "touched_ratio": float((prior_mask > 0).mean()),
        "segments": debug_segments,
    }
    debug_path = out_path.with_suffix(".clip_prior_debug.json")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)

    if args.save_prior_assets:
        np.save(out_path.with_suffix(".prior_value.npy"), prior_value.astype(np.float32))
        np.save(out_path.with_suffix(".prior_mask.npy"), prior_mask.astype(np.float32))

    print(f"✅ saved: {out_path}")
    print(f"✅ debug: {debug_path}")
    print(f"segments: {len(debug_segments)}, touched_ratio={debug['touched_ratio']:.6f}")


if __name__ == "__main__":
    main()
