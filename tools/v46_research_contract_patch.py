#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V46.20 research-contract patcher for tools/v46_motionrag_diff.py.

Purpose
-------
This patch fixes the Chang-E / change BVH Event-RAG failure mode at code level:
1) BVH fps / scale metadata must never remain in EDGE-151D contact channels.
2) Event motions saved into the RAG DB must satisfy the EDGE-151D contract.
3) Rot6D channels are reset to identity when invalid and re-projected after interpolation / blending.
4) Long-sequence V45/V46 inference uses sliding-window overlap-add instead of hard chunks.
5) V45/V46 training and inference inputs are contract-guarded.
6) Replacement failures are explicit; no silent skip for critical training patches.

Usage
-----
cd /home/disk/lsm/storage/EDGE
python tools/v46_research_contract_patch.py

It creates a timestamped .bak before modifying tools/v46_motionrag_diff.py.
The patcher is designed for the current histry/EDGE V46.12 code line and is
idempotent across repeated runs.
"""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "tools" / "v46_motionrag_diff.py"

HELPERS = r'''

# -----------------------------------------------------------------------------
# V46.20 research contract guards for Chang-E/change RAG DB
# -----------------------------------------------------------------------------
def identity6d_np(shape_prefix: Tuple[int, ...] = ()) -> np.ndarray:
    """Return identity rotation in the repository's 6D convention."""
    base = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    if not shape_prefix:
        return base.copy()
    return np.broadcast_to(base, tuple(shape_prefix) + (6,)).copy().astype(np.float32)


def sanitize_rot6d_np(rot6d: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Replace invalid / degenerate 6D rotations with identity before projection."""
    r = np.asarray(rot6d, dtype=np.float32).copy()
    if r.size == 0:
        return r.astype(np.float32), {"bad_joint_count": 0, "bad_joint_ratio": 0.0}
    r = r.reshape(-1, NUM_JOINTS, 6)
    finite = np.isfinite(r).all(axis=-1)
    a1 = r[..., 0:3]
    a2 = r[..., 3:6]
    n1 = np.linalg.norm(np.nan_to_num(a1, nan=0.0, posinf=0.0, neginf=0.0), axis=-1)
    n2 = np.linalg.norm(np.nan_to_num(a2, nan=0.0, posinf=0.0, neginf=0.0), axis=-1)
    bad = (~finite) | (n1 < 1e-5) | (n2 < 1e-5)
    bad_count = int(np.sum(bad))
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if bad_count:
        r[bad] = identity6d_np((bad_count,))
    report = {
        "bad_joint_count": bad_count,
        "bad_joint_ratio": float(bad_count / max(1, bad.size)),
        "min_a1_norm_before_identity": float(np.nanmin(n1)) if n1.size else 0.0,
        "min_a2_norm_before_identity": float(np.nanmin(n2)) if n2.size else 0.0,
    }
    return r.reshape(np.asarray(rot6d).shape).astype(np.float32), report


def project_edge151_rot6d_np(motion: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Project every EDGE 6D rotation channel back to SO(3)-derived 6D safely."""
    x = np.asarray(motion, dtype=np.float32).copy()
    if x.ndim != 2 or x.shape[1] < EDGE_DIM or x.shape[0] <= 0:
        return x.astype(np.float32), {"projected": False, "reason": "invalid_shape"}
    rot = x[:, ROT6D_START:ROT6D_END].reshape(x.shape[0], NUM_JOINTS, 6)
    rot, sanitize_report = sanitize_rot6d_np(rot)
    x[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(
        rot6d_to_matrix_np(rot.reshape(x.shape[0], NUM_JOINTS, 6))
    ).reshape(x.shape[0], -1)
    sanitize_report["projected"] = True
    return x.astype(np.float32), sanitize_report


def _safe_percentile(arr: np.ndarray, q: float, default: float = 0.0) -> float:
    try:
        a = np.asarray(arr, dtype=np.float32)
        if a.size == 0:
            return float(default)
        return float(np.nanpercentile(a, q))
    except Exception:
        return float(default)


def heuristic_contacts_fallback_np(motion: np.ndarray, cfg: V46Config, source_hint: str = "") -> Tuple[np.ndarray, dict]:
    """Kinematic fallback for contact channels when the main FK contact builder fails.

    V46.19 fix: never replace a time-varying foot contact signal with a static
    scalar such as 0.50 or 0.60.  First try a simple foot-height + foot-velocity
    heuristic from FK joints.  If even FK is unavailable, fall back to a root
    height/speed heuristic so the signal remains temporally varying rather than
    permanently locking or releasing both feet.
    """
    x = np.asarray(motion, dtype=np.float32)[:, :EDGE_DIM]
    T = int(x.shape[0])
    margin = float(getattr(cfg, "ik_height_margin", 0.05))
    speed_gate = float(getattr(cfg, "ik_speed_gate_mpf", 0.035))
    report = {"source_hint": str(source_hint), "mode": "uninitialized"}
    contacts = np.zeros((T, 4), dtype=np.float32)
    if T <= 0:
        report["mode"] = "empty"
        return contacts, report

    try:
        joints = fk_24_np(x)
        foot_ids = list(DEFAULT_FOOT_JOINTS)
        foot = joints[:, foot_ids]
        foot_vxz = np.zeros(foot.shape[:2], dtype=np.float32)
        if T > 1:
            foot_vxz[1:] = np.linalg.norm(foot[1:, :, [0, 2]] - foot[:-1, :, [0, 2]], axis=-1)
        floor_y = float(np.nanpercentile(foot[..., 1].reshape(-1), 5))
        near = foot[..., 1] <= floor_y + max(0.015, margin)
        slow = foot_vxz <= max(0.01, speed_gate)
        contacts = (near & slow).astype(np.float32)
        # Avoid all-zero output caused by overly strict speed thresholds on noisy
        # data.  Use near-floor alone as a second-stage fallback, still per-frame.
        if float(contacts.mean()) < 0.02:
            contacts = near.astype(np.float32)
            report["secondary_mode"] = "near_floor_without_speed_gate"
        report.update({
            "mode": "fk_height_velocity_heuristic",
            "floor_y": floor_y,
            "contact_ratio": float(contacts.mean()),
            "height_margin": float(margin),
            "speed_gate_mpf": float(speed_gate),
        })
        return contacts.astype(np.float32), report
    except Exception as exc:
        report["fk_heuristic_error"] = str(exc)

    # Last-resort fallback when FK is unavailable.  Without foot joints, there is
    # no physically reliable way to decide left/right support.  Therefore V46.19
    # deliberately avoids copying one root-level state to all four foot contacts:
    # that would weld both feet on near-root frames and release both feet otherwise.
    # Instead, produce a weak, non-anchoring, time-varying uncertainty signal that
    # stays below ik_contact_high, so IK will not impose a false strong foot lock.
    root = x[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
    root_speed = np.zeros((T,), dtype=np.float32)
    if T > 1:
        root_speed[1:] = np.linalg.norm(root[1:, [0, 2]] - root[:-1, [0, 2]], axis=-1)
    root_floor = float(np.nanpercentile(root[:, 1], 20)) if T else 0.0
    near_root = root[:, 1] <= root_floor + max(0.02, margin)
    slow_root = root_speed <= max(0.02, speed_gate * 2.0)
    support_like = (near_root & slow_root).astype(np.float32)
    uncertain = min(0.50, max(0.42, float(getattr(cfg, "ik_contact_low", 0.38)) + 0.06))
    release = max(0.05, min(0.25, float(getattr(cfg, "ik_contact_low", 0.38)) - 0.10))
    base = release + (uncertain - release) * support_like
    contacts = np.repeat(base[:, None], 4, axis=1).astype(np.float32)
    report.update({
        "mode": "root_uncertain_nonlocking_no_fk",
        "root_floor_y": root_floor,
        "contact_ratio": float(contacts.mean()),
        "uncertain_contact_value": float(uncertain),
        "release_contact_value": float(release),
        "height_margin": float(margin),
        "speed_gate_mpf": float(speed_gate),
        "warning": "FK unavailable; foot-specific contacts cannot be recovered, so fallback intentionally avoids strong IK anchoring.",
    })
    return contacts.astype(np.float32), report


def enforce_edge151_contract_np(
    motion: np.ndarray,
    cfg: Optional[V46Config] = None,
    source_hint: str = "",
    derive_contact: bool = True,
    project_rot: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Return a valid EDGE-151D motion tensor and an audit report.

    Critical reason:
    direct BVH loading may temporarily use channel 0 to carry native fps before
    resampling.  That is legal only inside loading.  Once an array is saved as an
    Event-RAG clip or passed into V45/V46, EDGE [0:4] must again mean contacts.
    """
    cfg = cfg or V46Config()
    x0 = np.asarray(motion, dtype=np.float32)
    report = {
        "version": "v46_20_edge151_contract_guard",
        "source_hint": str(source_hint),
        "input_shape": list(x0.shape),
    }
    if x0.ndim != 2 or x0.shape[1] < EDGE_DIM:
        raise ValueError(
            f"EDGE151 contract violation: expected [T,151+], got {tuple(x0.shape)} from {source_hint}"
        )

    x = x0[:, :EDGE_DIM].astype(np.float32).copy()
    finite_before = bool(np.isfinite(x).all())
    report["finite_before"] = finite_before

    # Handle root/contact/other scalar channels conservatively, but never let
    # invalid rot6d become all-zero rotations.  Rot6D is sanitized separately.
    scalar_idx = list(range(0, ROT6D_START))
    x[:, scalar_idx] = np.nan_to_num(x[:, scalar_idx], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    rot_flat, rot_sanitize_report = sanitize_rot6d_np(x[:, ROT6D_START:ROT6D_END])
    x[:, ROT6D_START:ROT6D_END] = rot_flat.reshape(x.shape[0], -1)
    report["rot6d_sanitize"] = rot_sanitize_report

    contact_before = x[:, 0:4].copy()
    report["contact_before_min"] = float(np.min(contact_before)) if contact_before.size else 0.0
    report["contact_before_max"] = float(np.max(contact_before)) if contact_before.size else 0.0
    report["contact_before_abs_p95"] = _safe_percentile(np.abs(contact_before), 95)
    contact_polluted = bool(report["contact_before_abs_p95"] > 1.5 or report["contact_before_min"] < -0.05)
    report["contact_metadata_pollution_detected"] = contact_polluted

    if project_rot:
        report["rot6d_abs_p95_before_project"] = _safe_percentile(np.abs(x[:, ROT6D_START:ROT6D_END]), 95)
        x, project_report = project_edge151_rot6d_np(x)
        report["rot6d_project"] = project_report
        report["rot6d_projected"] = True
    else:
        report["rot6d_projected"] = False

    if derive_contact:
        try:
            contacts, conf, floor_y, _ = derive_contacts_np(x, cfg)
            x[:, 0:4] = contacts.astype(np.float32)
            report["contact_rebuilt_from_fk"] = True
            report["contact_ratio"] = float(contacts.mean())
            report["contact_conf_mean"] = float(np.mean(conf))
            report["floor_y"] = float(floor_y)
        except Exception as exc:
            # V46.19: do not replace a dynamic contact signal with a global
            # constant such as 0.50 or 0.60.  Generate a per-frame kinematic
            # fallback so IK neither releases nor welds both feet for the whole clip.
            if contact_polluted:
                contacts_fb, fb_report = heuristic_contacts_fallback_np(
                    x, cfg, source_hint=f"contact_rebuild_failed:{source_hint}"
                )
                x[:, 0:4] = contacts_fb.astype(np.float32)
                report["contact_fallback_mode"] = "time_varying_kinematic_heuristic_due_to_metadata_pollution"
                report["contact_fallback_report"] = fb_report
            else:
                x[:, 0:4] = np.clip(np.nan_to_num(x[:, 0:4], nan=0.0), 0.0, 1.0)
                report["contact_fallback_mode"] = "clipped_existing_contact"
            report["contact_rebuilt_from_fk"] = False
            report["contact_rebuild_error"] = str(exc)
    else:
        if contact_polluted:
            contacts_fb, fb_report = heuristic_contacts_fallback_np(
                x, cfg, source_hint=f"derive_contact_false:{source_hint}"
            )
            x[:, 0:4] = contacts_fb.astype(np.float32)
            report["contact_fallback_mode"] = "derive_contact_false_time_varying_kinematic_heuristic_due_to_metadata_pollution"
            report["contact_fallback_report"] = fb_report
        else:
            x[:, 0:4] = np.clip(np.nan_to_num(x[:, 0:4], nan=0.0), 0.0, 1.0)
            report["contact_fallback_mode"] = "derive_contact_false_clipped_existing_contact"
        report["contact_rebuilt_from_fk"] = False

    root = x[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
    report["root_min"] = [float(v) for v in np.min(root, axis=0)]
    report["root_max"] = [float(v) for v in np.max(root, axis=0)]
    report["root_y_range_m"] = float(np.max(x[:, ROOT_Y_IDX]) - np.min(x[:, ROOT_Y_IDX]))
    report["root_xz_travel_m"] = float(
        np.linalg.norm(x[-1, [ROOT_X_IDX, ROOT_Z_IDX]] - x[0, [ROOT_X_IDX, ROOT_Z_IDX]])
    )
    report["contact_after_abs_p95"] = _safe_percentile(np.abs(x[:, 0:4]), 95)
    report["rot6d_abs_p95_after"] = _safe_percentile(np.abs(x[:, ROT6D_START:ROT6D_END]), 95)
    return x.astype(np.float32), report


def sliding_window_ranges(T: int, window: int, hop: int) -> List[Tuple[int, int]]:
    """Return coverage-complete sliding windows for long-sequence inference."""
    T = int(T)
    window = max(1, int(window))
    hop = max(1, int(hop))
    if T <= window:
        return [(0, T)]
    starts = list(range(0, max(1, T - window + 1), hop))
    last = T - window
    if starts[-1] != last:
        starts.append(last)
    return [(int(s), int(min(T, s + window))) for s in starts]


def overlap_add_weight_np(length: int, start: int, total: int, hop: int, window: int) -> np.ndarray:
    """Raised-cosine weight with global-boundary one-sided protection.

    V46.19 fix:
    a full symmetric Hann window attenuates the very first and very last global
    frames even though no outside window can compensate them.  We therefore keep
    the non-overlapped side of the first/last chunk at weight 1.0 and only use
    cosine weights inside actual cross-window transition regions.
    """
    length = int(length)
    start = int(start)
    total = int(total)
    if length <= 0:
        return np.zeros((0, 1), dtype=np.float32)
    if length == 1 or total <= length:
        return np.ones((length, 1), dtype=np.float32)

    n = np.arange(length, dtype=np.float32)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / float(max(length - 1, 1)))
    w = np.maximum(w, 1e-4).astype(np.float32)

    # The first global chunk has no previous chunk on its left side; do not
    # attenuate the leading half.  The last global chunk has no following chunk
    # on its right side; do not attenuate the trailing half.
    half = max(1, length // 2)
    if start <= 0:
        w[:half] = 1.0
    if start + length >= total:
        w[half:] = 1.0
    return w[:, None].astype(np.float32)

def normalize_quat_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    out = q / np.maximum(norm, 1e-8)
    bad = (~np.isfinite(out).all(axis=-1)) | (norm[..., 0] < 1e-8)
    if np.any(bad):
        out = out.copy()
        out[bad] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return out.astype(np.float32)


def matrix_to_quat_np(R: np.ndarray) -> np.ndarray:
    """Vectorized rotation-matrix to unit quaternion conversion [w,x,y,z].

    V46.19 fix: avoid per-matrix Python loops.  Long whole-song inference calls
    this function many times for [T,24,3,3] arrays, so the branch logic is
    implemented with NumPy masks to keep official evaluation practical.
    """
    arr = np.asarray(R, dtype=np.float32)
    prefix = arr.shape[:-2]
    m = arr.reshape(-1, 3, 3)
    q = np.zeros((m.shape[0], 4), dtype=np.float32)
    if m.shape[0] == 0:
        return q.reshape(prefix + (4,)).astype(np.float32)

    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    tr = m00 + m11 + m22

    mask = tr > 0.0
    if np.any(mask):
        s = np.sqrt(np.maximum(tr[mask] + 1.0, 1e-8)) * 2.0
        q[mask, 0] = 0.25 * s
        q[mask, 1] = (m21[mask] - m12[mask]) / s
        q[mask, 2] = (m02[mask] - m20[mask]) / s
        q[mask, 3] = (m10[mask] - m01[mask]) / s

    rem = ~mask
    mask_x = rem & (m00 > m11) & (m00 > m22)
    if np.any(mask_x):
        s = np.sqrt(np.maximum(1.0 + m00[mask_x] - m11[mask_x] - m22[mask_x], 1e-8)) * 2.0
        q[mask_x, 0] = (m21[mask_x] - m12[mask_x]) / s
        q[mask_x, 1] = 0.25 * s
        q[mask_x, 2] = (m01[mask_x] + m10[mask_x]) / s
        q[mask_x, 3] = (m02[mask_x] + m20[mask_x]) / s

    mask_y = rem & (~mask_x) & (m11 > m22)
    if np.any(mask_y):
        s = np.sqrt(np.maximum(1.0 + m11[mask_y] - m00[mask_y] - m22[mask_y], 1e-8)) * 2.0
        q[mask_y, 0] = (m02[mask_y] - m20[mask_y]) / s
        q[mask_y, 1] = (m01[mask_y] + m10[mask_y]) / s
        q[mask_y, 2] = 0.25 * s
        q[mask_y, 3] = (m12[mask_y] + m21[mask_y]) / s

    mask_z = rem & (~mask_x) & (~mask_y)
    if np.any(mask_z):
        s = np.sqrt(np.maximum(1.0 + m22[mask_z] - m00[mask_z] - m11[mask_z], 1e-8)) * 2.0
        q[mask_z, 0] = (m10[mask_z] - m01[mask_z]) / s
        q[mask_z, 1] = (m02[mask_z] + m20[mask_z]) / s
        q[mask_z, 2] = (m12[mask_z] + m21[mask_z]) / s
        q[mask_z, 3] = 0.25 * s

    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return normalize_quat_np(q.reshape(prefix + (4,)))

def quat_to_matrix_np(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternions [w,x,y,z] to rotation matrices."""
    q = normalize_quat_np(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R.astype(np.float32)


def init_motion_window_accumulators(T: int, D: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    accum = np.zeros((int(T), int(D)), dtype=np.float32)
    weight_sum = np.zeros((int(T), 1), dtype=np.float32)
    rot_quat_accum = np.zeros((int(T), NUM_JOINTS, 4), dtype=np.float32)
    rot_quat_weight = np.zeros((int(T), 1, 1), dtype=np.float32)
    return accum, weight_sum, rot_quat_accum, rot_quat_weight


def accumulate_motion_window_np(
    accum: np.ndarray,
    weight_sum: np.ndarray,
    rot_quat_accum: np.ndarray,
    rot_quat_weight: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    start: int,
    end: int,
) -> None:
    """Accumulate a generated chunk without linearly averaging Rot6D.

    Root/contact/scalar channels use Euclidean overlap-add.  Rotation channels
    are converted to quaternions and accumulated on S^3 with sign alignment
    before a final normalized quaternion-to-rot6d projection.  This avoids the
    near-zero Rot6D cancellation and snap risk caused by direct Rot6D averaging.
    """
    y = np.asarray(y, dtype=np.float32)[: int(end - start), :EDGE_DIM]
    w = np.asarray(w, dtype=np.float32).reshape(-1, 1)[: y.shape[0]]
    if y.shape[0] == 0:
        return
    y_linear = y.copy()
    y_linear[:, ROT6D_START:ROT6D_END] = 0.0
    accum[start:end] += y_linear * w
    weight_sum[start:end] += w

    R = rot6d_to_matrix_np(y[:, ROT6D_START:ROT6D_END].reshape(y.shape[0], NUM_JOINTS, 6))
    q = matrix_to_quat_np(R)
    for li, gi in enumerate(range(int(start), int(end))):
        wi = float(w[li, 0])
        if wi <= 0.0:
            continue
        qi = q[li]
        if float(rot_quat_weight[gi, 0, 0]) > 1e-8:
            ref = normalize_quat_np(rot_quat_accum[gi])
            dots = np.sum(qi * ref, axis=-1, keepdims=True)
            qi = np.where(dots < 0.0, -qi, qi)
        rot_quat_accum[gi] += qi * wi
        rot_quat_weight[gi, 0, 0] += wi


def finalize_motion_window_accum_np(
    accum: np.ndarray,
    weight_sum: np.ndarray,
    rot_quat_accum: np.ndarray,
    rot_quat_weight: np.ndarray,
    cfg: V46Config,
    source_hint: str,
) -> Tuple[np.ndarray, dict]:
    out = accum / np.maximum(weight_sum, 1e-8)
    valid = rot_quat_weight[:, 0, 0] > 1e-8
    q = np.zeros((accum.shape[0], NUM_JOINTS, 4), dtype=np.float32)
    q[:] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if np.any(valid):
        q[valid] = normalize_quat_np(rot_quat_accum[valid])
    R = quat_to_matrix_np(q)
    out[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(R).reshape(accum.shape[0], -1)
    out, report = enforce_edge151_contract_np(out, cfg, source_hint=source_hint, derive_contact=True, project_rot=True)
    report["rotation_overlap_mode"] = "quaternion_sign_aligned_weighted_average"
    report["scalar_overlap_mode"] = "hann_weighted_overlap_add"
    report["weight_sum_min"] = float(np.min(weight_sum)) if weight_sum.size else 0.0
    report["weight_sum_p05"] = float(np.percentile(weight_sum, 5)) if weight_sum.size else 0.0
    return out.astype(np.float32), report


def blend_motion_overlap_np(
    a: np.ndarray,
    b: np.ndarray,
    w_b: np.ndarray,
    cfg: V46Config,
    source_hint: str = "blend_motion_overlap",
) -> Tuple[np.ndarray, dict]:
    """Blend two overlap clips with quaternion rotation fusion, not Rot6D LERP.

    a and b must have the same temporal length. Scalar/root/contact channels
    use Euclidean weights; Rot6D channels are converted to quaternions with
    sign alignment and then mapped back to Rot6D. This is used for RAG event
    boundary blending, where adjacent retrieved clips can have large pose gaps.
    """
    a = np.asarray(a, dtype=np.float32)[:, :EDGE_DIM]
    b = np.asarray(b, dtype=np.float32)[:, :EDGE_DIM]
    L = int(min(len(a), len(b)))
    if L <= 0:
        return np.zeros((0, EDGE_DIM), dtype=np.float32), {"blend_mode": "empty"}
    a = a[:L]
    b = b[:L]
    wb = np.asarray(w_b, dtype=np.float32).reshape(-1, 1)[:L]
    wb = np.clip(wb, 0.0, 1.0)
    wa = 1.0 - wb
    accum, weight_sum, rot_quat_accum, rot_quat_weight = init_motion_window_accumulators(L, EDGE_DIM)
    accumulate_motion_window_np(accum, weight_sum, rot_quat_accum, rot_quat_weight, a, wa, 0, L)
    accumulate_motion_window_np(accum, weight_sum, rot_quat_accum, rot_quat_weight, b, wb, 0, L)
    out, report = finalize_motion_window_accum_np(
        accum, weight_sum, rot_quat_accum, rot_quat_weight, cfg, source_hint=source_hint
    )
    report["blend_mode"] = "scalar_linear_quaternion_rotation"
    report["w_b_min"] = float(np.min(wb)) if wb.size else 0.0
    report["w_b_max"] = float(np.max(wb)) if wb.size else 0.0
    return out.astype(np.float32), report
'''

NEW_RESAMPLE = r'''def resample_motion_to_config_fps(motion: np.ndarray, cfg: V46Config) -> Tuple[np.ndarray, dict]:
    """Resample BVH-derived EDGE-like arrays to cfg.fps before event slicing.

    V46.15 fix:
    channel 0 may temporarily carry BVH native fps from load_bvh_file(), but it
    must never be overwritten with target fps after resampling because EDGE
    channel 0 is a contact channel.  The DB writer rebuilds contacts from FK.
    """
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 2:
        return x.astype(np.float32), {"resampled": False, "reason": "too_short"}

    ch0 = x[:, 0]
    ch0_med = float(np.nanmedian(ch0)) if np.isfinite(ch0).any() else 0.0
    ch0_std = float(np.nanstd(ch0)) if np.isfinite(ch0).any() else 0.0
    ch0_p05 = float(np.nanpercentile(ch0, 5)) if np.isfinite(ch0).any() else 0.0
    ch0_p95 = float(np.nanpercentile(ch0, 95)) if np.isfinite(ch0).any() else 0.0
    looks_like_fps_metadata = bool(
        ch0_med > 2.0 and ch0_p05 > 1.0 and ch0_p95 < 400.0 and ch0_std < max(0.5, ch0_med * 0.05)
    )
    native = ch0_med if looks_like_fps_metadata else float(cfg.fps)
    target = float(cfg.fps)

    if (not bool(getattr(cfg, "bvh_resample_to_config_fps", True))) or abs(native - target) < 1e-3:
        y = x.copy().astype(np.float32)
        return y, {
            "resampled": False,
            "native_fps": float(native),
            "target_fps": float(target),
            "frames": int(len(x)),
            "fps_metadata_detected": looks_like_fps_metadata,
            "note": "channel0_not_overwritten_EDGE_contact_contract",
        }

    new_len = max(2, int(round(x.shape[0] * target / max(native, 1e-6))))
    y = resample_motion_np(x, new_len).astype(np.float32)
    return y, {
        "resampled": True,
        "native_fps": float(native),
        "target_fps": float(target),
        "frames_before": int(len(x)),
        "frames_after": int(new_len),
        "fps_metadata_detected": looks_like_fps_metadata,
        "note": "channel0_preserved_until_event_contract_guard_rebuilds_contacts",
    }


'''

NEW_SAMPLE = r'''def sample_motion_window(paths: np.ndarray, target_len: int, cfg: Optional[V46Config] = None) -> np.ndarray:
    """Sample a training window and keep the EDGE-151D contract after resampling."""
    p = str(random.choice(paths.tolist()))
    m = np.load(p).astype(np.float32)
    if m.shape[0] == target_len:
        out = m
    elif m.shape[0] > target_len:
        st = random.randint(0, m.shape[0] - target_len)
        out = m[st:st + target_len]
    else:
        out = resample_motion_np(m, target_len)
    out, _ = enforce_edge151_contract_np(out, cfg, source_hint=f"sample_motion_window:{p}", derive_contact=True, project_rot=True)
    return out.astype(np.float32)


'''

NEW_DEGRADE = r'''def degrade_for_refiner(clean: np.ndarray, severity: float = 0.06, cfg: Optional[V46Config] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Create seam corruption for V45/V46 without corrupting EDGE contact channels."""
    x = clean.copy().astype(np.float32)
    T, D = x.shape
    seam = np.zeros((T, 1), dtype=np.float32)
    if T > 30:
        c = random.randint(T // 4, 3 * T // 4)
        w = random.randint(6, min(20, T // 5))
        seam[max(0, c - w): min(T, c + w)] = 1.0

        offset = np.zeros(D, dtype=np.float32)
        offset[[ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = np.random.normal(0, severity, size=3)
        offset[ROT6D_START:ROT6D_END] = np.random.normal(0, severity * 0.35, size=ROT6D_END - ROT6D_START)
        x[c:] += offset
        if c + 1 < T:
            drift = np.zeros((1, D), dtype=np.float32)
            drift[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = np.random.normal(0, severity * 0.5, size=(1, 3))
            drift[:, ROT6D_START:ROT6D_END] = np.random.normal(0, severity * 0.18, size=(1, ROT6D_END - ROT6D_START))
            x[c:] += np.linspace(0, 1, T - c, dtype=np.float32)[:, None] * drift

    noise = np.zeros_like(x, dtype=np.float32)
    noise[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = np.random.normal(0, severity * 0.10, size=(T, 3)).astype(np.float32)
    noise[:, ROT6D_START:ROT6D_END] = np.random.normal(0, severity * 0.05, size=(T, ROT6D_END - ROT6D_START)).astype(np.float32)
    x += noise
    x, _ = enforce_edge151_contract_np(x, cfg, source_hint="degrade_for_refiner", derive_contact=True, project_rot=True)
    return x.astype(np.float32), seam


'''
NEW_CONCAT = r'''def concat_events(event_paths: Sequence[str], target_durations: Sequence[float], cfg: V46Config) -> Tuple[np.ndarray, List[dict]]:
    """Concatenate retrieved RAG events under the EDGE-151D contract.

    V46.19 fix:
    RAG event boundary overlap no longer linearly interpolates Rot6D channels.
    Root/contact/scalar channels use linear boundary weights, while rotations use
    the same sign-aligned quaternion fusion used by long-window inference.
    """
    pieces: List[np.ndarray] = []
    rep: List[dict] = []
    for i, (p, dur) in enumerate(zip(event_paths, target_durations)):
        m_raw = np.load(str(p)).astype(np.float32)
        m, pre_report = enforce_edge151_contract_np(
            m_raw, cfg, source_hint=f"concat_load:{p}", derive_contact=True, project_rot=True
        )
        target_len = max(cfg.min_event_frames, int(round(float(dur) * cfg.fps)))
        warp = target_len / max(1, m.shape[0])
        m = resample_motion_np(m, target_len).astype(np.float32)
        m, post_resample_report = enforce_edge151_contract_np(
            m, cfg, source_hint=f"concat_resample:{p}", derive_contact=True, project_rot=True
        )
        used_overlap = 0
        align_report = None
        blend_report = None
        if pieces:
            m = align_next_to_prev(pieces[-1], m)
            m, align_report = enforce_edge151_contract_np(
                m, cfg, source_hint=f"concat_align:{p}", derive_contact=True, project_rot=True
            )
            ov = min(int(cfg.overlap), len(pieces[-1]) // 3, len(m) // 3)
            used_overlap = int(max(0, ov))
            if ov > 0:
                a = pieces[-1][-ov:].copy()
                b = m[:ov].copy()
                w_b = np.linspace(0.0, 1.0, ov, dtype=np.float32)[:, None]
                blend, blend_report = blend_motion_overlap_np(
                    a, b, w_b, cfg, source_hint=f"concat_overlap_quat:{Path(str(p)).name}"
                )
                pieces[-1] = np.concatenate([pieces[-1][:-ov], blend], axis=0)
                pieces[-1], _ = enforce_edge151_contract_np(
                    pieces[-1], cfg, source_hint="concat_piece_after_quat_overlap", derive_contact=True, project_rot=True
                )
                m = m[ov:]
        pieces.append(m.astype(np.float32))
        rep.append({
            "path": str(p),
            "target_frames": int(target_len),
            "source_frames": int(m_raw.shape[0]),
            "warp": float(warp),
            "overlap": int(used_overlap),
            "boundary_blend_mode": "quaternion_rotation" if used_overlap > 0 else "none",
            "contract_pre": pre_report,
            "contract_after_resample": post_resample_report,
            "contract_after_align": align_report,
            "contract_overlap_blend": blend_report,
        })
    final = np.concatenate(pieces, axis=0).astype(np.float32)
    final, final_report = enforce_edge151_contract_np(
        final, cfg, source_hint="concat_final", derive_contact=True, project_rot=True
    )
    if rep:
        rep[-1]["concat_final_contract"] = final_report
    return final.astype(np.float32), rep


'''

NEW_REFINER = r'''def apply_refiner_model(motion: np.ndarray, cond: np.ndarray, seam_mask: np.ndarray, ckpt_path: Optional[str], cfg: V46Config) -> np.ndarray:
    """Apply V45 with sliding-window inference and quaternion rotation fusion."""
    if torch is None or not ckpt_path or not Path(ckpt_path).exists():
        seam_centers = []
        for a, b in contiguous_regions(seam_mask[:, 0] > 0.5):
            seam_centers.append((a + b) // 2)
        refined = analytic_residual_refine(motion, seam_centers)
        refined, _ = enforce_edge151_contract_np(
            refined, cfg, source_hint="apply_refiner_model:analytic", derive_contact=True, project_rot=True
        )
        return refined.astype(np.float32)

    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model = TemporalRefiner(EDGE_DIM, 32).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    T = int(motion.shape[0])
    win = int(cfg.window_len)
    hop = max(1, min(int(getattr(cfg, "hop_len", win)), win))
    accum, weight_sum, rot_quat_accum, rot_quat_weight = init_motion_window_accumulators(T, EDGE_DIM)

    with torch.no_grad():
        for st, ed in sliding_window_ranges(T, win, hop):
            chunk = motion[st:ed]
            mask = seam_mask[st:ed]
            orig_len = len(chunk)
            if orig_len < win:
                chunk_in = resample_motion_np(chunk, win)
                mask_in = resample_motion_np(mask, win)
            else:
                chunk_in = chunk
                mask_in = mask
            chunk_in, _ = enforce_edge151_contract_np(
                chunk_in, cfg, source_hint="apply_refiner_model:input_chunk", derive_contact=True, project_rot=True
            )
            x = torch.from_numpy(chunk_in[None]).float().to(cfg.device)
            c = torch.from_numpy(cond[None].astype(np.float32)).float().to(cfg.device)
            sm = torch.from_numpy(mask_in[None].astype(np.float32)).float().to(cfg.device)
            delta = model(x, c, sm)
            y = x + delta * (0.2 + 0.8 * sm)
            y_np = y[0].detach().cpu().numpy()
            if orig_len < win:
                y_np = resample_motion_np(y_np, orig_len)
            y_np, _ = enforce_edge151_contract_np(
                y_np, cfg, source_hint="apply_refiner_model:output_chunk", derive_contact=True, project_rot=True
            )
            w = overlap_add_weight_np(orig_len, st, T, hop, win)
            accumulate_motion_window_np(accum, weight_sum, rot_quat_accum, rot_quat_weight, y_np, w, st, ed)

    out, _ = finalize_motion_window_accum_np(
        accum, weight_sum, rot_quat_accum, rot_quat_weight, cfg, source_hint="apply_refiner_model:final"
    )
    return out.astype(np.float32)


'''
NEW_DIFFUSION = r'''def apply_diffusion_model(motion: np.ndarray, cond: np.ndarray, seam_mask: np.ndarray, ckpt_path: Optional[str], cfg: V46Config) -> np.ndarray:
    """Apply V46 diffusion with sliding-window inference and quaternion rotation fusion."""
    if torch is None or not ckpt_path or not Path(ckpt_path).exists():
        motion, _ = enforce_edge151_contract_np(
            motion, cfg, source_hint="apply_diffusion_model:disabled", derive_contact=True, project_rot=True
        )
        return motion.astype(np.float32)

    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    Tdiff = int(ckpt.get("diffusion_steps", cfg.diffusion_steps))
    model = DiffusionDenoiser(EDGE_DIM, 32).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    betas, alphas, abar = make_beta_schedule(Tdiff, torch.device(cfg.device))

    T = int(motion.shape[0])
    win = int(cfg.window_len)
    hop = max(1, min(int(getattr(cfg, "hop_len", win)), win))
    accum, weight_sum, rot_quat_accum, rot_quat_weight = init_motion_window_accumulators(T, EDGE_DIM)

    with torch.no_grad():
        for st, ed in sliding_window_ranges(T, win, hop):
            retr_np = motion[st:ed]
            mask_np = seam_mask[st:ed]
            orig_len = len(retr_np)
            if orig_len < win:
                retr_in = resample_motion_np(retr_np, win)
                mask_in = resample_motion_np(mask_np, win)
            else:
                retr_in = retr_np
                mask_in = mask_np
            retr_in, _ = enforce_edge151_contract_np(
                retr_in, cfg, source_hint="apply_diffusion_model:retrieval_chunk", derive_contact=True, project_rot=True
            )
            retr = torch.from_numpy(retr_in[None]).float().to(cfg.device)
            mask = torch.from_numpy(mask_in[None].astype(np.float32)).float().to(cfg.device)
            c = torch.from_numpy(cond[None].astype(np.float32)).float().to(cfg.device)
            x = retr + 0.03 * torch.randn_like(retr) * (0.25 + 0.75 * mask)
            for ti in reversed(range(Tdiff)):
                t = torch.full((1,), ti, device=cfg.device, dtype=torch.long)
                eps = model(x, retr, c, mask, t)
                beta = betas[ti]
                alpha = alphas[ti]
                ab = abar[ti]
                mean = (1 / torch.sqrt(alpha)) * (x - beta / torch.sqrt(1 - ab).clamp_min(1e-6) * eps)
                if ti > 0:
                    x = mean + torch.sqrt(beta) * torch.randn_like(x) * 0.35
                else:
                    x = mean
                x = retr * (1.0 - 0.65 * mask) + x * (0.65 * mask)
            y = x[0].detach().cpu().numpy()
            if orig_len < win:
                y = resample_motion_np(y, orig_len)
            y, _ = enforce_edge151_contract_np(
                y, cfg, source_hint="apply_diffusion_model:output_chunk", derive_contact=True, project_rot=True
            )
            w = overlap_add_weight_np(orig_len, st, T, hop, win)
            accumulate_motion_window_np(accum, weight_sum, rot_quat_accum, rot_quat_weight, y, w, st, ed)

    out, _ = finalize_motion_window_accum_np(
        accum, weight_sum, rot_quat_accum, rot_quat_weight, cfg, source_hint="apply_diffusion_model:final"
    )
    return out.astype(np.float32)


'''


def replace_between(text: str, start_pat: str, end_pat: str, replacement: str, label: str) -> str:
    start = text.find(start_pat)
    if start < 0:
        raise RuntimeError(f"Cannot find start marker for {label}: {start_pat!r}")
    end = text.find(end_pat, start)
    if end < 0:
        raise RuntimeError(f"Cannot find end marker for {label}: {end_pat!r}")
    return text[:start] + replacement + text[end:]


def replace_helper_block(text: str) -> str:
    old_markers = [
        "# V46.20 research contract guards for Chang-E/change RAG DB",
        "# V46.19 research contract guards for Chang-E/change RAG DB",
        "# V46.18 research contract guards for Chang-E/change RAG DB",
        "# V46.17 research contract guards for Chang-E/change RAG DB",
        "# V46.16 research contract guards for Chang-E/change RAG DB",
        "# V46.15 research contract guards for Chang-E/change RAG DB",
        "# V46.14 research contract guards for Chang-E/change RAG DB",
        "# V46.13 research contract guards for Chang-E/change RAG DB",
    ]
    for marker in old_markers:
        if marker in text:
            start = text.rfind("# -----------------------------------------------------------------------------", 0, text.find(marker))
            if start < 0:
                start = text.find(marker)
            end = text.find("def _clean_stem(path: str | Path) -> str:", start)
            if end < 0:
                raise RuntimeError("Cannot locate end of existing V46.13+ helper block.")
            return text[:start] + HELPERS + "\n" + text[end:]
    anchor = "    return entry.astype(np.float32), exit_.astype(np.float32), contact[0], contact[-1]\n\n\n"
    if anchor not in text:
        raise RuntimeError("Cannot locate motion_boundary_state insertion anchor.")
    return text.replace(anchor, anchor + HELPERS, 1)


def ensure_env_lines(text: str) -> str:
    anchor = '            "V46_BEAM_SIZE": ("beam_size", int),\n'
    if anchor not in text:
        raise RuntimeError("Cannot locate V46_BEAM_SIZE env-map anchor.")
    insert_lines = [
        '            "V46_OVERLAP": ("overlap", int),\n',
        '            "V46_WINDOW_LEN": ("window_len", int),\n',
        '            "V46_HOP_LEN": ("hop_len", int),\n',
        '            "V46_MIN_EVENT_FRAMES": ("min_event_frames", int),\n',
        '            "V46_MAX_EVENT_FRAMES": ("max_event_frames", int),\n',
        '            "V46_MANIFEST_SECONDARY_EVENT_SPLIT": ("manifest_secondary_event_split", lambda x: bool(int(x))),\n',
        '            "V46_ENABLE_CONTRASTIVE": ("contrastive_enable", lambda x: bool(int(x))),\n',
        '            "V46_ENABLE_REFINER": ("refiner_enable", lambda x: bool(int(x))),\n',
        '            "V46_ENABLE_DIFFUSION": ("diffusion_enable", lambda x: bool(int(x))),\n',
    ]
    missing = [line for line in insert_lines if line.strip() not in text]
    if missing:
        text = text.replace(anchor, anchor + "".join(missing), 1)
    return text


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)
    text = TARGET.read_text(encoding="utf-8")
    backup = TARGET.with_suffix(TARGET.suffix + f".v46_19_contract_fix_{time.strftime('%Y%m%d_%H%M%S')}.bak")
    shutil.copy2(TARGET, backup)

    text = replace_helper_block(text)
    text = ensure_env_lines(text)

    text = replace_between(
        text,
        "def resample_motion_to_config_fps(motion: np.ndarray, cfg: V46Config) -> Tuple[np.ndarray, dict]:\n",
        "def read_manifest_records(manifest_path: Optional[str], motion_dirs: Sequence[str], cfg: V46Config) -> List[dict]:\n",
        NEW_RESAMPLE,
        "resample_motion_to_config_fps",
    )

    old_save = (
        ") -> None:\n"
        "    np.save(out_path, clip.astype(np.float32))\n"
        "    desc = event_descriptor(clip, cfg.fps)\n"
    )
    new_save = (
        ") -> None:\n"
        "    clip, contract_report = enforce_edge151_contract_np(\n"
        "        clip,\n"
        "        cfg,\n"
        "        source_hint=str(base_meta.get(\"source_file\", out_path)),\n"
        "        derive_contact=True,\n"
        "        project_rot=True,\n"
        "    )\n"
        "    np.save(out_path, clip.astype(np.float32))\n"
        "    desc = event_descriptor(clip, cfg.fps)\n"
    )
    if old_save in text:
        text = text.replace(old_save, new_save, 1)
    elif "contract_report = enforce_edge151_contract_np" not in text:
        raise RuntimeError("Cannot locate add_event_to_db_lists save/descriptor block.")

    old_item = '        "input_mode": base_meta.get("input_mode", "direct_files"),\n    }\n'
    new_item = '        "input_mode": base_meta.get("input_mode", "direct_files"),\n        "edge151_contract_report": contract_report,\n    }\n'
    if '"edge151_contract_report": contract_report' not in text:
        if old_item not in text:
            raise RuntimeError("Cannot locate metadata item block for contract report insertion.")
        text = text.replace(old_item, new_item, 1)

    sample_start = "def sample_motion_window(paths: np.ndarray, target_len: int, cfg: Optional[V46Config] = None) -> np.ndarray:\n" \
        if "def sample_motion_window(paths: np.ndarray, target_len: int, cfg: Optional[V46Config] = None) -> np.ndarray:\n" in text \
        else "def sample_motion_window(paths: np.ndarray, target_len: int) -> np.ndarray:\n"
    degrade_start = "def degrade_for_refiner(clean: np.ndarray, severity: float = 0.06, cfg: Optional[V46Config] = None) -> Tuple[np.ndarray, np.ndarray]:\n" \
        if "def degrade_for_refiner(clean: np.ndarray, severity: float = 0.06, cfg: Optional[V46Config] = None) -> Tuple[np.ndarray, np.ndarray]:\n" in text \
        else "def degrade_for_refiner(clean: np.ndarray, severity: float = 0.06) -> Tuple[np.ndarray, np.ndarray]:\n"

    text = replace_between(
        text,
        sample_start,
        degrade_start,
        NEW_SAMPLE,
        "sample_motion_window",
    )

    degrade_start = "def degrade_for_refiner(clean: np.ndarray, severity: float = 0.06, cfg: Optional[V46Config] = None) -> Tuple[np.ndarray, np.ndarray]:\n" \
        if "def degrade_for_refiner(clean: np.ndarray, severity: float = 0.06, cfg: Optional[V46Config] = None) -> Tuple[np.ndarray, np.ndarray]:\n" in text \
        else "def degrade_for_refiner(clean: np.ndarray, severity: float = 0.06) -> Tuple[np.ndarray, np.ndarray]:\n"
    text = replace_between(
        text,
        degrade_start,
        "def train_refiner(args: argparse.Namespace) -> int:\n",
        NEW_DEGRADE,
        "degrade_for_refiner",
    )

    old_refiner_sample = "            clean = sample_motion_window(paths, cfg.window_len)\n"
    new_refiner_sample = "            clean = sample_motion_window(paths, cfg.window_len, cfg)\n"
    if old_refiner_sample in text:
        text = text.replace(old_refiner_sample, new_refiner_sample, 1)
    elif new_refiner_sample not in text:
        raise RuntimeError("Cannot locate train_refiner sample_motion_window call.")

    old_diff_clean = "            clean = resample_motion_np(clean, cfg.window_len)\n            retr, seam = degrade_for_refiner(clean, severity=0.045)\n"
    new_diff_clean = (
        "            clean = resample_motion_np(clean, cfg.window_len)\n"
        "            clean, _ = enforce_edge151_contract_np(\n"
        "                clean, cfg, source_hint=f\"train_diffusion_clean:{paths[idx]}\", derive_contact=True, project_rot=True\n"
        "            )\n"
        "            retr, seam = degrade_for_refiner(clean, severity=0.045, cfg=cfg)\n"
    )
    if old_diff_clean in text:
        text = text.replace(old_diff_clean, new_diff_clean, 1)
    elif "degrade_for_refiner(clean, severity=0.045)" in text:
        # Defensive fallback for upstream formatting drift: if the exact two-line
        # block no longer matches, still inject the contract guard immediately
        # before degradation instead of only adding cfg=cfg.  Replace the whole
        # assignment line, not only the function call substring.
        pattern = r'(?m)^(\s*)retr,\s*seam\s*=\s*degrade_for_refiner\(clean,\s*severity=0\.045\)\s*$'
        match = re.search(pattern, text)
        if match is None:
            raise RuntimeError("Cannot safely inject train_diffusion clean contract guard around degrade_for_refiner.")
        indent = match.group(1)
        fallback_new = (
            f"{indent}clean, _ = enforce_edge151_contract_np(\n"
            f"{indent}    clean, cfg, source_hint=f\"train_diffusion_clean:{{paths[idx]}}\", derive_contact=True, project_rot=True\n"
            f"{indent})\n"
            f"{indent}retr, seam = degrade_for_refiner(clean, severity=0.045, cfg=cfg)"
        )
        text = text[:match.start()] + fallback_new + text[match.end():]
    elif "degrade_for_refiner(clean, severity=0.045, cfg=cfg)" in text and "train_diffusion_clean:" not in text:
        raise RuntimeError(
            "train_diffusion uses cfg-aware degradation but is missing the clean contract guard "
            "(source_hint=train_diffusion_clean). Please patch from a supported V46 source file."
        )
    elif "degrade_for_refiner(clean, severity=0.045, cfg=cfg)" not in text:
        raise RuntimeError("Cannot locate train_diffusion clean-resample block for contract guard insertion.")

    text = replace_between(
        text,
        "def concat_events(event_paths: Sequence[str], target_durations: Sequence[float], cfg: V46Config) -> Tuple[np.ndarray, List[dict]]:\n",
        "def make_boundary_mask(T: int, seams: Sequence[int], width: int = 18) -> np.ndarray:\n",
        NEW_CONCAT,
        "concat_events",
    )

    text = replace_between(
        text,
        "def apply_refiner_model(motion: np.ndarray, cond: np.ndarray, seam_mask: np.ndarray, ckpt_path: Optional[str], cfg: V46Config) -> np.ndarray:\n",
        "def apply_diffusion_model(motion: np.ndarray, cond: np.ndarray, seam_mask: np.ndarray, ckpt_path: Optional[str], cfg: V46Config) -> np.ndarray:\n",
        NEW_REFINER,
        "apply_refiner_model",
    )

    text = replace_between(
        text,
        "def apply_diffusion_model(motion: np.ndarray, cond: np.ndarray, seam_mask: np.ndarray, ckpt_path: Optional[str], cfg: V46Config) -> np.ndarray:\n",
        "def derive_contacts_np(motion: np.ndarray, cfg: V46Config) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:\n",
        NEW_DIFFUSION,
        "apply_diffusion_model",
    )

    TARGET.write_text(text, encoding="utf-8")
    print(f"[OK] Patched {TARGET}")
    print(f"[BAK] {backup}")


if __name__ == "__main__":
    main()
