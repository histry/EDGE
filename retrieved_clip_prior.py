"""Retrieved-clip prior for EDGE diffusion sampling.

This module implements the safer version of RAG-Diffusion prior injection:
- Build a soft motion prior from the continuous clips retrieved by MMR-RAG.
- Install a small monkey patch on GaussianDiffusion.p_mean_variance so the
  prior is blended into predicted x_start inside every DDPM denoising step.

It intentionally does not replace model/diffusion.py.  The patch is installed
at runtime by generate_controlled.py, which is safer for your current EDGE fork
because diffusion.py already contains many project-specific modifications.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from dataset.quaternion import ax_to_6v
    from dataset.preprocess import vectorize_many
    from vis import SMPLSkeleton
except Exception:  # pragma: no cover - keeps import safe in partial envs
    ax_to_6v = None
    vectorize_many = None
    SMPLSkeleton = None

ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
CONTACT_SLICE = slice(0, 4)
ROOT_SLICE = slice(4, 7)
ROT_SLICE = slice(7, 151)

# SMPL-like 24-joint index convention used by EDGE.
JOINT_SETS = {
    "arms": [13, 14, 16, 17, 18, 19, 20, 21, 22, 23],
    "upper": [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "torso": [3, 6, 9, 12, 15],
    "all_rot": list(range(24)),
}


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalizer_normalize(normalizer, motion: np.ndarray) -> np.ndarray:
    if normalizer is None:
        raise ValueError("normalizer is required to convert physical source clips to normalized model space")
    motion = np.asarray(motion, dtype=np.float32)
    mt = torch.from_numpy(motion).float()
    if mt.ndim == 2:
        out = normalizer.normalize(mt[None])
        return _to_numpy(out)[0].astype(np.float32)
    if mt.ndim == 3:
        out = normalizer.normalize(mt)
        return _to_numpy(out).astype(np.float32)
    raise ValueError(f"Expected motion [T,151] or [B,T,151], got {motion.shape}")


def _load_npy_motion_151(path: Path) -> Optional[np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        data = arr.item()
        for key in ("motion", "motion_151", "poses", "pose_seq", "pose"):
            if key in data:
                arr = data[key]
                break
        else:
            return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] == 151:
        arr = arr[None]
    if arr.ndim != 2 or arr.shape[1] != 151:
        return None
    return arr.astype(np.float32)


def _pkl_pos_q_to_motion_151(path: Path) -> Optional[np.ndarray]:
    if SMPLSkeleton is None or ax_to_6v is None or vectorize_many is None:
        raise ImportError(
            "Cannot convert pkl {'pos','q'} source because SMPLSkeleton/ax_to_6v/vectorize_many is unavailable."
        )
    data = pickle.load(open(path, "rb"))
    if not isinstance(data, dict):
        return None
    if "motion" in data:
        arr = np.asarray(data["motion"], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 151:
            return arr
    if "pos" not in data or "q" not in data:
        return None

    pos = torch.tensor(data["pos"], dtype=torch.float32).unsqueeze(0)
    q = torch.tensor(data["q"], dtype=torch.float32).unsqueeze(0)
    q = q.reshape(q.shape[0], q.shape[1], -1, 3)

    smpl = SMPLSkeleton()
    with torch.no_grad():
        joints = smpl.forward(q, pos)
        feet = joints[:, :, [7, 8, 10, 11]]
        feetv = torch.zeros(feet.shape[:3], dtype=feet.dtype)
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).to(q)
        q_6v = ax_to_6v(q)
        motion = vectorize_many([contacts, pos, q_6v])
    return motion[0].detach().cpu().numpy().astype(np.float32)


def load_source_motion_151(path: str) -> Tuple[np.ndarray, str]:
    """Load source motion and return (motion_151, detected_space).

    detected_space:
      - "physical" for pkl {'pos','q'} converted by FK/vectorize_many.
      - "unknown" for npy/npz already containing [T,151].
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix == ".pkl":
        motion = _pkl_pos_q_to_motion_151(p)
        if motion is None:
            raise ValueError(f"Could not read {path} as pkl motion with keys motion or pos/q")
        return motion.astype(np.float32), "physical"

    if p.suffix in [".npy", ".npz"]:
        motion = _load_npy_motion_151(p)
        if motion is None:
            raise ValueError(f"Could not read {path} as [T,151] motion")
        return motion.astype(np.float32), "unknown"

    raise ValueError(f"Unsupported source file type: {path}")


def feature_indices_for_body_part(body_part: str) -> np.ndarray:
    body_part = str(body_part)
    if body_part not in JOINT_SETS:
        raise ValueError(f"body_part must be one of {sorted(JOINT_SETS)}, got {body_part}")
    feats: List[int] = []
    for joint in JOINT_SETS[body_part]:
        start = 7 + int(joint) * 6
        feats.extend(range(start, start + 6))
    return np.asarray(sorted(set(feats)), dtype=np.int64)


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_temporal_weights(length: int, center_index: int) -> np.ndarray:
    """Triangle/smoothstep weights peaking at center, going to 0 at edges."""
    if length <= 1:
        return np.ones((length,), dtype=np.float32)
    idx = np.arange(length, dtype=np.float32)
    dist = np.abs(idx - float(center_index))
    radius = max(float(max(center_index, length - 1 - center_index)), 1.0)
    w = 1.0 - dist / radius
    return smoothstep(w).astype(np.float32)


def parse_frame_list(text_or_list, num_frames: int) -> List[int]:
    if text_or_list is None:
        return []
    if isinstance(text_or_list, (list, tuple, np.ndarray)):
        items = text_or_list
    else:
        text = str(text_or_list).replace(";", ",")
        items = [x for x in text.split(",") if x.strip()]
    frames = []
    for item in items:
        try:
            v = int(round(float(item)))
        except Exception:
            continue
        frames.append(max(0, min(num_frames - 1, v)))
    return sorted(set(frames))


def apply_protection_to_mask(mask: np.ndarray, protect_frames: Sequence[int], protect_width: int):
    if not protect_frames:
        return
    t = mask.shape[0]
    width = max(0, int(protect_width))
    for frame in protect_frames:
        frame = int(frame)
        s = max(0, frame - width)
        e = min(t, frame + width + 1)
        mask[s:e, :] = 0.0


def build_retrieved_clip_prior_from_plan(
    auto_plan_path: str,
    num_frames: int,
    normalizer,
    device,
    source_pose_space: str = "auto",
    window: int = 24,
    body_part: str = "upper",
    protect_frames: Optional[Sequence[int]] = None,
    protect_width: int = 2,
    temporal_smooth: int = 0,
    debug_out_prefix: str = "",
) -> Dict[str, object]:
    """Build normalized-space soft prior tensors from an auto_mid_plan.json.

    Returns a dict ready to be attached to model.diffusion.retrieved_clip_prior.
    """
    plan_path = Path(auto_plan_path)
    plan = json.load(open(plan_path, "r", encoding="utf-8"))
    keyframes = plan.get("auto_keyframes", [])

    value = np.zeros((num_frames, 151), dtype=np.float32)
    mask = np.zeros((num_frames, 151), dtype=np.float32)
    weight_sum = np.zeros((num_frames, 151), dtype=np.float32)

    feature_idx = feature_indices_for_body_part(body_part)
    window = max(1, int(window))

    segment_debug = []
    segments_used = 0

    for item in keyframes:
        source = item.get("source", "")
        target_frame = int(item.get("frame", -1))
        source_frame = int(item.get("source_frame", -1))
        if not source or target_frame < 0 or source_frame < 0:
            segment_debug.append({"source": source, "frame": target_frame, "used": False, "reason": "missing_source_or_frame"})
            continue

        try:
            src_motion, detected_space = load_source_motion_151(source)
        except Exception as exc:
            segment_debug.append({"source": source, "frame": target_frame, "used": False, "reason": str(exc)})
            continue

        effective_space = source_pose_space
        if effective_space == "auto":
            effective_space = "physical" if detected_space == "physical" else "normalized"

        if effective_space == "physical":
            src_motion = _normalizer_normalize(normalizer, src_motion)
        elif effective_space == "normalized":
            src_motion = np.asarray(src_motion, dtype=np.float32)
        else:
            raise ValueError("source_pose_space must be auto, physical, or normalized")

        t0 = max(0, target_frame - window)
        t1 = min(num_frames, target_frame + window + 1)
        if t1 <= t0:
            continue

        # Source clip aligned so source_frame maps to target_frame.
        target_frames = np.arange(t0, t1, dtype=np.int64)
        source_frames = source_frame + (target_frames - target_frame)
        source_frames = np.clip(source_frames, 0, len(src_motion) - 1)

        src_clip = src_motion[source_frames]
        local_center = int(np.where(target_frames == target_frame)[0][0]) if target_frame in target_frames else len(target_frames) // 2
        weights = build_temporal_weights(len(target_frames), local_center)

        for j, tf in enumerate(target_frames):
            w = float(weights[j])
            if w <= 1e-8:
                continue
            value[tf, feature_idx] += src_clip[j, feature_idx] * w
            weight_sum[tf, feature_idx] += w
            mask[tf, feature_idx] = np.maximum(mask[tf, feature_idx], w)

        segments_used += 1
        segment_debug.append({
            "source": source,
            "target_frame": int(target_frame),
            "source_frame": int(source_frame),
            "used": True,
            "target_range": [int(t0), int(t1 - 1)],
            "source_range": [int(source_frames[0]), int(source_frames[-1])],
            "detected_source_space": detected_space,
            "effective_source_space": effective_space,
        })

    nonzero = weight_sum > 1e-8
    value[nonzero] = value[nonzero] / weight_sum[nonzero]

    # For zero-weight features, fill current value as zero; mask remains 0.
    protect_frames = protect_frames or []
    apply_protection_to_mask(mask, protect_frames, protect_width)

    # Optional light smoothing of mask only. We do not smooth 6D rotations by default
    # because the source clip itself is already temporally continuous.
    if temporal_smooth and temporal_smooth > 1:
        k = int(temporal_smooth)
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k,), dtype=np.float32) / float(k)
        for fi in feature_idx:
            m = np.convolve(mask[:, fi], kernel, mode="same")
            mask[:, fi] = np.minimum(1.0, m)
        apply_protection_to_mask(mask, protect_frames, protect_width)

    touched_ratio = float((mask > 1e-6).mean())

    out = {
        "value": torch.from_numpy(value[None]).to(device=device, dtype=torch.float32),
        "mask": torch.from_numpy(mask[None]).to(device=device, dtype=torch.float32),
        "segments": int(segments_used),
        "touched_ratio": touched_ratio,
        "body_part": body_part,
        "window": int(window),
        "protect_frames": [int(x) for x in protect_frames],
        "protect_width": int(protect_width),
        "debug": segment_debug,
    }

    if debug_out_prefix:
        prefix = Path(debug_out_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(prefix) + "_retrieved_prior_value.npy", value.astype(np.float32))
        np.save(str(prefix) + "_retrieved_prior_mask.npy", mask.astype(np.float32))
        with open(str(prefix) + "_retrieved_prior_debug.json", "w", encoding="utf-8") as f:
            json.dump({
                "segments": int(segments_used),
                "touched_ratio": touched_ratio,
                "body_part": body_part,
                "window": int(window),
                "protect_frames": [int(x) for x in protect_frames],
                "protect_width": int(protect_width),
                "segments_detail": segment_debug,
            }, f, ensure_ascii=False, indent=2)

    return out


def _expand_prior_mask(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if mask.shape[-1] == 1:
        return mask.expand_as(value)
    if mask.shape[-1] == value.shape[-1]:
        return mask
    raise ValueError(f"prior mask last dim must be 1 or {value.shape[-1]}, got {mask.shape[-1]}")


def _apply_retrieved_prior_to_xstart(diffusion, x_start, t, constraint=None):
    prior = getattr(diffusion, "retrieved_clip_prior", None)
    if not prior:
        return x_start

    value = prior.get("value", None)
    mask = prior.get("mask", None)
    if value is None or mask is None:
        return x_start

    value = value.to(device=x_start.device, dtype=x_start.dtype)
    mask = mask.to(device=x_start.device, dtype=x_start.dtype)
    if value.shape[1] != x_start.shape[1]:
        # Interpolate if a future user changes num_frames.
        value = F.interpolate(value.transpose(1, 2), size=x_start.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        mask = F.interpolate(mask.transpose(1, 2), size=x_start.shape[1], mode="linear", align_corners=False).transpose(1, 2)
    if value.shape[0] == 1 and x_start.shape[0] > 1:
        value = value.expand(x_start.shape[0], -1, -1)
        mask = mask.expand(x_start.shape[0], -1, -1)

    mask_full = _expand_prior_mask(mask, x_start).clamp(0.0, 1.0)

    strength = float(prior.get("strength", 0.15))
    anneal_power = float(prior.get("anneal_power", 1.0))
    min_t_frac = float(prior.get("min_t_frac", 0.0))
    max_t_frac = float(prior.get("max_t_frac", 1.0))

    # t is high at early noisy steps and low near the end.  We ramp strength up
    # later in denoising so prior guides cleaned predictions without dominating
    # early global structure.
    if torch.is_tensor(t):
        t_frac = t.float() / float(max(int(getattr(diffusion, "n_timestep", 1000)) - 1, 1))
        active = ((t_frac >= min_t_frac) & (t_frac <= max_t_frac)).float()
        late = (1.0 - t_frac).clamp(0.0, 1.0).pow(anneal_power)
        scale = (strength * active * late).view(-1, 1, 1)
    else:
        scale = strength

    blend = (mask_full * scale).clamp(0.0, 1.0)
    x_start = x_start * (1.0 - blend) + value * blend

    # Keyframe/explicit constraints always win over retrieved prior.
    if constraint is not None:
        c_value = constraint.get("value", None)
        c_mask = constraint.get("mask", None)
        if c_value is not None and c_mask is not None:
            c_value = c_value.to(device=x_start.device, dtype=x_start.dtype)
            c_mask = c_mask.to(device=x_start.device, dtype=x_start.dtype)
            c_mask = _expand_prior_mask(c_mask, x_start).clamp(0.0, 1.0)
            x_start = x_start * (1.0 - c_mask) + c_value * c_mask

    return x_start


def install_retrieved_clip_prior_patch():
    """Patch GaussianDiffusion.p_mean_variance once.

    The patch preserves the original public API. Existing p_sample_loop/ddim_sample
    calls continue to work; if `diffusion.retrieved_clip_prior` is unset, behavior is
    identical to the original implementation.
    """
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        raise ImportError(f"Could not import model.diffusion.GaussianDiffusion: {exc}")

    if getattr(GaussianDiffusion, "_retrieved_clip_prior_patch_installed", False):
        return

    def p_mean_variance_with_retrieved_prior(self, x, cond, t, clip_denoised=True, constraint=None):
        _, x_recon = self.model_predictions(
            x,
            cond,
            t,
            clip_x_start=clip_denoised,
            constraint=constraint,
        )

        x_recon = _apply_retrieved_prior_to_xstart(
            self,
            x_recon,
            t,
            constraint=constraint,
        )

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon,
            x_t=x,
            t=t,
        )
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    GaussianDiffusion.p_mean_variance = p_mean_variance_with_retrieved_prior
    GaussianDiffusion._retrieved_clip_prior_patch_installed = True
