#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V46.31 research-contract patcher for tools/v46_motionrag_diff.py.

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
# V46.31 research contract guards for Chang-E/change RAG DB
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
    a1_clean = np.nan_to_num(a1, nan=0.0, posinf=0.0, neginf=0.0)
    a2_clean = np.nan_to_num(a2, nan=0.0, posinf=0.0, neginf=0.0)
    n1 = np.linalg.norm(a1_clean, axis=-1)
    n2 = np.linalg.norm(a2_clean, axis=-1)
    # V46.31: also reject near-collinear 6D vectors.  Gram-Schmidt
    # can collapse when a1 and a2 are parallel/anti-parallel even if both
    # vector norms are valid, which can happen during early diffusion denoising.
    denom = np.maximum(n1 * n2, 1e-8)
    cross_norm = np.linalg.norm(np.cross(a1_clean, a2_clean), axis=-1) / denom
    bad = (~finite) | (n1 < 1e-5) | (n2 < 1e-5) | (cross_norm < 1e-5)
    bad_count = int(np.sum(bad))
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if bad_count:
        r[bad] = identity6d_np((bad_count,))
    report = {
        "bad_joint_count": bad_count,
        "bad_joint_ratio": float(bad_count / max(1, bad.size)),
        "min_a1_norm_before_identity": float(np.nanmin(n1)) if n1.size else 0.0,
        "min_a2_norm_before_identity": float(np.nanmin(n2)) if n2.size else 0.0,
        "min_cross_norm_before_identity": float(np.nanmin(cross_norm)) if cross_norm.size else 0.0,
        "near_collinear_joint_count": int(np.sum(cross_norm < 1e-5)) if cross_norm.size else 0,
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


def rotate_motion_around_y_np(motion: np.ndarray, yaw_delta: float, pivot_xz: Optional[np.ndarray] = None) -> np.ndarray:
    """Rotate a whole EDGE-151D motion around the vertical Y axis.

    This is a world-space rigid yaw transform for event stitching. It rotates
    the root XZ trajectory around ``pivot_xz`` and left-multiplies the root
    joint rotation by R_y(yaw_delta). Child joint local rotations remain valid
    because the root orientation carries the global heading change.
    """
    out = np.asarray(motion, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] < ROT6D_END or out.shape[0] <= 0:
        return out.astype(np.float32)
    yaw = float(yaw_delta)
    if not np.isfinite(yaw) or abs(yaw) < 1e-8:
        return out.astype(np.float32)
    c = float(np.cos(yaw))
    ss = float(np.sin(yaw))
    if pivot_xz is None:
        pivot = out[0, [ROOT_X_IDX, ROOT_Z_IDX]].astype(np.float32)
    else:
        pivot = np.asarray(pivot_xz, dtype=np.float32).reshape(2)
    rel_x = out[:, ROOT_X_IDX].copy() - float(pivot[0])
    rel_z = out[:, ROOT_Z_IDX].copy() - float(pivot[1])
    out[:, ROOT_X_IDX] = c * rel_x + ss * rel_z + float(pivot[0])
    out[:, ROOT_Z_IDX] = -ss * rel_x + c * rel_z + float(pivot[1])

    ry = np.asarray([[c, 0.0, ss], [0.0, 1.0, 0.0], [-ss, 0.0, c]], dtype=np.float32)
    root6 = out[:, ROT6D_START:ROT6D_START + 6].reshape(out.shape[0], 1, 6)
    root_r = rot6d_to_matrix_np(root6)
    root_r = np.matmul(ry[None, None, :, :], root_r).astype(np.float32)
    out[:, ROT6D_START:ROT6D_START + 6] = matrix_to_rot6d_np(root_r).reshape(out.shape[0], 6)
    return out.astype(np.float32)


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
        "version": "v46_24_edge151_contract_guard",
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


NEW_MATRIX_TO_ROT6D = r'''def matrix_to_rot6d_np(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to EDGE/Zhou 6D in column-concatenated form.

    V46.21/V46.31 critical fix:
    The inverse of rot6d_to_matrix_np() must concatenate the first two matrix
    columns as [R[:,0], R[:,1]].  The previous row-major expression
    ``mat[..., :, 0:2].reshape(..., 6)`` interleaves rows as
    [R00, R01, R10, R11, R20, R21], which turns the identity matrix into
    [1, 0, 0, 1, 0, 0] instead of [1, 0, 0, 0, 1, 0].  That silently corrupts
    saved Event-RAG clips and makes strict raw-rot6d audit fail even after
    projection.
    """
    m = np.asarray(mat, dtype=np.float32)
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"matrix_to_rot6d_np expects [...,3,3], got {m.shape}")
    c0 = m[..., :, 0]
    c1 = m[..., :, 1]
    return np.concatenate([c0, c1], axis=-1).astype(np.float32)


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
    finite_ch0 = ch0[np.isfinite(ch0)]
    if finite_ch0.size:
        ch0_med = float(np.nanmedian(finite_ch0))
        ch0_p05 = float(np.nanpercentile(finite_ch0, 5))
        ch0_p95 = float(np.nanpercentile(finite_ch0, 95))
        # V46.31: FPS metadata written by the BVH loader is nearly constant, but
        # a few boundary/cropping frames may contain small jitter.  Use the middle
        # 90% band for the main constant-column test and keep raw std only for
        # diagnostics.  This avoids silently skipping required high-FPS -> 30 FPS
        # resampling because of a few outlier rows.
        trimmed = finite_ch0[(finite_ch0 >= ch0_p05) & (finite_ch0 <= ch0_p95)]
        ch0_std = float(np.nanstd(finite_ch0))
        ch0_trimmed_std = float(np.nanstd(trimmed)) if trimmed.size else ch0_std
    else:
        ch0_med = ch0_p05 = ch0_p95 = ch0_std = ch0_trimmed_std = 0.0
    looks_like_fps_metadata = bool(
        ch0_med > 2.0 and ch0_p05 > 1.0 and ch0_p95 < 400.0
        and ch0_trimmed_std < max(0.75, ch0_med * 0.08)
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
            "channel0_trimmed_std": float(ch0_trimmed_std),
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
        "channel0_trimmed_std": float(ch0_trimmed_std),
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

    V46.31 fix:
    The overlap cross-fade is now compensated locally per segment, not by a
    whole-song global resample.  Every music slot keeps its assigned net frame
    budget after overlap trimming, so local beat/phrase boundaries do not drift.
    We also keep the V46.28/V46.29 yaw-aligned overlap start, no root-Y ramp,
    ov==1 midpoint weighting, and safe one-frame overlap slicing.
    """
    pieces: List[np.ndarray] = []
    rep: List[dict] = []
    target_lens = [max(cfg.min_event_frames, int(round(float(d) * cfg.fps))) for d in target_durations]
    for i, (p, dur) in enumerate(zip(event_paths, target_durations)):
        m_raw = np.load(str(p)).astype(np.float32)
        m, pre_report = enforce_edge151_contract_np(
            m_raw, cfg, source_hint=f"concat_load:{p}", derive_contact=True, project_rot=True
        )
        target_len = int(target_lens[i])
        # V46.31: compensate overlap locally.  Incoming clips lose ov frames
        # when m = m[ov:] removes the overlapped prefix.  Rather than globally
        # resampling the entire final song, pre-extend non-first clips by the
        # maximum plausible overlap and then locally normalize their post-overlap
        # remainder back to target_len.  This preserves per-slot music timing.
        overlap_budget = int(max(0, getattr(cfg, "overlap", 0))) if pieces else 0
        local_resample_len = int(max(cfg.min_event_frames, target_len + overlap_budget))
        warp = local_resample_len / max(1, m.shape[0])
        m = resample_motion_np(m, local_resample_len).astype(np.float32)
        m, post_resample_report = enforce_edge151_contract_np(
            m, cfg, source_hint=f"concat_resample_local_timing:{p}", derive_contact=True, project_rot=True
        )
        used_overlap = 0
        align_report = None
        blend_report = None
        local_timing_report = {
            "expected_net_frames": int(target_len),
            "local_resample_frames_before_overlap": int(local_resample_len),
            "overlap_budget_frames": int(overlap_budget),
            "overlap_trim_frames": 0,
            "post_overlap_frames_before_local_fix": int(local_resample_len),
            "local_timing_fix_applied": False,
            "local_timing_fix_mode": "none",
        }
        if pieces:
            ov = min(int(cfg.overlap), len(pieces[-1]) // 3, len(m) // 3)
            used_overlap = int(max(0, ov))
            if ov > 0:
                # Align incoming m[0] to the previous overlap start in both yaw
                # and XZ position.  This avoids both speed surge and cross-heading
                # tearing inside the quaternion overlap window.
                ref = pieces[-1][-ov].copy()
                try:
                    yaw_ref = float(root_yaw_np(pieces[-1][-ov:][:1])[0])
                    yaw_m = float(root_yaw_np(m[:1])[0])
                    dyaw = float(np.arctan2(np.sin(yaw_ref - yaw_m), np.cos(yaw_ref - yaw_m)))
                except Exception:
                    yaw_ref, yaw_m, dyaw = 0.0, 0.0, 0.0
                m = rotate_motion_around_y_np(m, dyaw, pivot_xz=m[0, [ROOT_X_IDX, ROOT_Z_IDX]])
                delta_xz = ref[[ROOT_X_IDX, ROOT_Z_IDX]] - m[0, [ROOT_X_IDX, ROOT_Z_IDX]]
                m[:, ROOT_X_IDX] += float(delta_xz[0])
                m[:, ROOT_Z_IDX] += float(delta_xz[1])
                # Deliberately do not apply any root-Y ramp.  Height/contact
                # continuity is handled only inside the real overlap blend.
                m, align_report = enforce_edge151_contract_np(
                    m, cfg, source_hint=f"concat_overlap_start_yaw_align:{p}", derive_contact=True, project_rot=True
                )
                if align_report is None:
                    align_report = {}
                align_report.update({
                    "overlap_alignment_mode": "yaw_and_xz_to_overlap_start_no_root_y_ramp",
                    "overlap_ref_frame": "previous_event[-overlap]",
                    "yaw_ref": float(yaw_ref),
                    "yaw_incoming_before": float(yaw_m),
                    "dyaw_applied": float(dyaw),
                    "delta_xz_applied": [float(delta_xz[0]), float(delta_xz[1])],
                    "root_y_ramp_applied": False,
                })
                a = pieces[-1][-ov:].copy()
                b = m[:ov].copy()
                if ov == 1:
                    w_b = np.asarray([[0.5]], dtype=np.float32)
                else:
                    w_b = np.linspace(0.0, 1.0, ov, dtype=np.float32)[:, None]
                blend, blend_report = blend_motion_overlap_np(
                    a, b, w_b, cfg, source_hint=f"concat_overlap_quat:{Path(str(p)).name}"
                )
                pieces[-1] = np.concatenate([pieces[-1][:-ov], blend], axis=0)
                pieces[-1], _ = enforce_edge151_contract_np(
                    pieces[-1], cfg, source_hint="concat_piece_after_quat_overlap", derive_contact=True, project_rot=True
                )
                m = m[ov:]
                local_timing_report["overlap_trim_frames"] = int(ov)
                local_timing_report["post_overlap_frames_before_local_fix"] = int(m.shape[0])
            else:
                m = align_next_to_prev(pieces[-1], m)
                m, align_report = enforce_edge151_contract_np(
                    m, cfg, source_hint=f"concat_align_no_overlap:{p}", derive_contact=True, project_rot=True
                )
                local_timing_report["post_overlap_frames_before_local_fix"] = int(m.shape[0])

            # V46.31: after overlap handling, repair only this incoming segment's
            # net length.  This prevents whole-song interpolation from smearing
            # contact steps and preserves local music slot boundaries.
            if int(m.shape[0]) != int(target_len):
                m = resample_motion_np(m, int(target_len)).astype(np.float32)
                m, local_fix_report = enforce_edge151_contract_np(
                    m, cfg, source_hint=f"concat_local_timing_fix:{p}", derive_contact=True, project_rot=True
                )
                local_timing_report.update({
                    "local_timing_fix_applied": True,
                    "local_timing_fix_mode": "segment_local_resample_after_overlap_trim",
                    "frames_after_local_timing_fix": int(m.shape[0]),
                    "contract_after_local_timing_fix": local_fix_report,
                })
            else:
                local_timing_report["frames_after_local_timing_fix"] = int(m.shape[0])

        pieces.append(m.astype(np.float32))
        rep.append({
            "path": str(p),
            "target_frames": int(target_len),
            "source_frames": int(m_raw.shape[0]),
            "local_resample_frames": int(local_resample_len),
            "warp": float(warp),
            "overlap": int(used_overlap),
            "boundary_blend_mode": "quaternion_rotation" if used_overlap > 0 else "none",
            "contract_pre": pre_report,
            "contract_after_resample": post_resample_report,
            "contract_after_align": align_report,
            "contract_overlap_blend": blend_report,
            "segment_local_timing": local_timing_report,
        })

    final = np.concatenate(pieces, axis=0).astype(np.float32)
    total_target_frames = int(sum(target_lens))
    timing_report = {
        "target_total_frames": int(total_target_frames),
        "frames_before_terminal_guard": int(final.shape[0]),
        "timing_frame_delta_before_terminal_guard": int(total_target_frames - final.shape[0]),
        "timing_compensation_applied": False,
        "timing_compensation_mode": "segment_local_overlap_compensation_no_global_resample",
        "global_resample_applied": False,
    }
    # Terminal guard only.  It should normally be a no-op because each segment is
    # locally length-corrected.  If a pathological one-frame edge case remains,
    # trim or hold the last frame instead of globally resampling thousands of
    # frames, so local beat/contact timing is not redistributed across the song.
    if total_target_frames > 0 and int(final.shape[0]) != int(total_target_frames):
        delta = int(total_target_frames - final.shape[0])
        if delta > 0:
            pad = np.repeat(final[-1:, :], delta, axis=0).astype(np.float32)
            final = np.concatenate([final, pad], axis=0).astype(np.float32)
            mode = "terminal_hold_last_frame_pad_no_global_resample"
        else:
            final = final[:total_target_frames].astype(np.float32)
            mode = "terminal_trim_no_global_resample"
        timing_report.update({
            "timing_compensation_applied": True,
            "timing_compensation_mode": mode,
            "terminal_delta_frames": int(delta),
        })
    timing_report["frames_after_terminal_guard"] = int(final.shape[0])
    final, final_report = enforce_edge151_contract_np(
        final, cfg, source_hint="concat_final", derive_contact=True, project_rot=True
    )
    if rep:
        rep[-1]["concat_timing_compensation"] = timing_report
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


NEW_CHANG_E_SEMANTICS = '\nCHANG_E_CATEGORY_PROFILES = {\n    "flying_apsaras": {"aliases": {"flying", "apsaras", "flying_apsara", "flying_apsaras", "feitian", "fei_tian", "sky_dance"}, "energy": 0.52, "onset": 0.28, "travel": 0.32, "turn": 0.38, "lower": 0.38, "upper": 0.72, "floorwork": 0.10, "jump": 0.35, "spin": 0.35, "pose_hold": 0.25, "instrument": 0.0, "prop": 0.85, "display": "Flying Apsaras", "semantic_role": "aerial_graceful_flow", "energy_label": "moderate", "rhythm_label": "lyrical", "body_focus_label": "upper_body", "spatial_label": "aerial_leaning", "music_alignment_label": "lyrical_flow", "music_alignment_tags": ["lyrical_flow", "turning_climax", "calm_meditative", "aerial_curve"], "preferred_music_roles": ["intro", "build_up", "climax"], "preferred_dance_keys": ["flying_apsaras", "sogdian_whirl", "lotus_steps"], "cultural_motif": "flying_apsara", "prop_proxy_label": "sash_ribbon_proxy", "locomotion_label": "floating_leaning", "support_label": "low_contact_flight_like", "event_family": "aerial_curve", "motion_stage_role": "opening_or_climax", "natural_duration_range_sec": [2.0, 5.5]},\n    "lotus_steps": {"aliases": {"lotus", "lotussteps", "lotus_step", "lotus_steps"}, "energy": 0.48, "onset": 0.35, "travel": 0.62, "turn": 0.20, "lower": 0.78, "upper": 0.38, "floorwork": 0.05, "jump": 0.12, "spin": 0.10, "pose_hold": 0.20, "instrument": 0.0, "prop": 0.0, "display": "Lotus Steps", "semantic_role": "flowing_footwork", "energy_label": "moderate", "rhythm_label": "lyrical", "body_focus_label": "lower_body", "spatial_label": "traveling", "music_alignment_label": "footwork_flow", "music_alignment_tags": ["footwork_flow", "lyrical_flow", "calm_meditative"], "preferred_music_roles": ["normal", "development"], "preferred_dance_keys": ["lotus_steps", "flying_apsaras", "sogdian_whirl"], "cultural_motif": "lotus_step", "prop_proxy_label": "none", "locomotion_label": "traveling_steps", "support_label": "alternating_foot_support", "event_family": "footwork_flow", "motion_stage_role": "development", "natural_duration_range_sec": [1.5, 4.0]},\n    "thirty_six_postures": {"aliases": {"36pose", "36posture", "36postures", "thirtysix", "thirty_six", "thirty_six_postures", "jiyuetian"}, "energy": 0.36, "onset": 0.18, "travel": 0.12, "turn": 0.12, "lower": 0.28, "upper": 0.42, "floorwork": 0.18, "jump": 0.02, "spin": 0.05, "pose_hold": 0.90, "instrument": 0.0, "prop": 0.0, "display": "Ji Yue Tian Thirty-Six Postures", "semantic_role": "iconic_pose_sequence", "energy_label": "moderate", "rhythm_label": "sustained", "body_focus_label": "pose", "spatial_label": "in_place", "music_alignment_label": "pose_hold", "music_alignment_tags": ["pose_hold", "calm_meditative", "lyrical_flow"], "preferred_music_roles": ["intro", "release", "resolution"], "preferred_dance_keys": ["thirty_six_postures", "revelation_meditation", "lotus_steps"], "cultural_motif": "jiyuetian_pose", "prop_proxy_label": "none", "locomotion_label": "in_place_pose", "support_label": "static_or_low_motion_support", "event_family": "pose_motif", "motion_stage_role": "anchor_or_resolution", "natural_duration_range_sec": [1.2, 3.8]},\n    "revelation_meditation": {"aliases": {"meditation", "mediation", "revelation", "revelation_meditation", "revelation_mediation"}, "energy": 0.20, "onset": 0.08, "travel": 0.10, "turn": 0.08, "lower": 0.20, "upper": 0.36, "floorwork": 0.38, "jump": 0.0, "spin": 0.03, "pose_hold": 0.78, "instrument": 0.0, "prop": 0.0, "display": "Revelation Meditation", "semantic_role": "calm_meditative_flow", "energy_label": "calm", "rhythm_label": "sustained", "body_focus_label": "full_body", "spatial_label": "in_place", "music_alignment_label": "calm_meditative", "music_alignment_tags": ["calm_meditative", "pose_hold", "lyrical_flow"], "preferred_music_roles": ["intro", "calm", "release", "resolution"], "preferred_dance_keys": ["revelation_meditation", "thirty_six_postures", "flying_apsaras"], "cultural_motif": "buddhist_meditation", "prop_proxy_label": "none", "locomotion_label": "slow_weight_shift", "support_label": "stable_support", "event_family": "calm_flow", "motion_stage_role": "intro_or_resolution", "natural_duration_range_sec": [2.0, 6.0]},\n    "sogdian_whirl": {"aliases": {"ribbon", "ribbon_flow", "sash", "silk", "whirl", "sogdian", "sogdian_whirl", "turn", "turning"}, "energy": 0.72, "onset": 0.40, "travel": 0.50, "turn": 0.90, "lower": 0.68, "upper": 0.65, "floorwork": 0.02, "jump": 0.20, "spin": 0.95, "pose_hold": 0.15, "instrument": 0.0, "prop": 0.75, "display": "Sogdian Whirl / Ribbon Flow", "semantic_role": "flowing_turning_motif", "energy_label": "high", "rhythm_label": "lyrical", "body_focus_label": "turning_flow", "spatial_label": "turning", "music_alignment_label": "turning_climax", "music_alignment_tags": ["turning_climax", "lyrical_flow", "footwork_flow"], "preferred_music_roles": ["build_up", "climax"], "preferred_dance_keys": ["sogdian_whirl", "flying_apsaras", "lotus_steps"], "cultural_motif": "sogdian_whirl", "prop_proxy_label": "ribbon_sash_proxy", "locomotion_label": "turning_travel", "support_label": "alternating_or_pivot_support", "event_family": "turning_flow", "motion_stage_role": "climax", "natural_duration_range_sec": [1.6, 4.5]},\n    "pipa_behind_back": {"aliases": {"pipa", "pipa1", "pipa2", "playing_pipa", "playing_the_pipa", "pipa_behind_back"}, "energy": 0.46, "onset": 0.42, "travel": 0.16, "turn": 0.20, "lower": 0.30, "upper": 0.82, "floorwork": 0.06, "jump": 0.05, "spin": 0.10, "pose_hold": 0.45, "instrument": 1.0, "prop": 0.70, "display": "Playing the Pipa Behind the Back", "semantic_role": "instrument_upper_body_motif", "energy_label": "moderate", "rhythm_label": "accented", "body_focus_label": "upper_body", "spatial_label": "in_place", "music_alignment_label": "instrument_phrase", "music_alignment_tags": ["instrument_phrase", "lyrical_flow", "percussive_accent"], "preferred_music_roles": ["motif", "normal", "build_up"], "preferred_dance_keys": ["pipa_behind_back", "sogdian_whirl", "lei_gong_drum"], "cultural_motif": "pipa_instrument_pose", "prop_proxy_label": "pipa_proxy", "locomotion_label": "upper_body_phrase", "support_label": "stable_support", "event_family": "instrument_motif", "motion_stage_role": "motif_recall", "natural_duration_range_sec": [1.6, 4.5]},\n    "lei_gong_drum": {"aliases": {"drum", "lei_gong", "leigong", "lei_gong_drum"}, "energy": 0.82, "onset": 0.88, "travel": 0.52, "turn": 0.35, "lower": 0.75, "upper": 0.76, "floorwork": 0.04, "jump": 0.32, "spin": 0.20, "pose_hold": 0.10, "instrument": 0.65, "prop": 0.55, "display": "Lei Gong Drum", "semantic_role": "percussive_high_energy", "energy_label": "percussive", "rhythm_label": "percussive", "body_focus_label": "full_body", "spatial_label": "traveling", "music_alignment_label": "percussive_accent", "music_alignment_tags": ["percussive_accent", "turning_climax", "footwork_flow"], "preferred_music_roles": ["accent", "climax"], "preferred_dance_keys": ["lei_gong_drum", "pipa_behind_back", "sogdian_whirl"], "cultural_motif": "thunder_drum", "prop_proxy_label": "drum_proxy", "locomotion_label": "accented_travel", "support_label": "strong_foot_contact", "event_family": "percussive_accent", "motion_stage_role": "accent_or_climax", "natural_duration_range_sec": [1.2, 3.5]},\n    "unknown": {"aliases": set(), "energy": 0.45, "onset": 0.30, "travel": 0.30, "turn": 0.20, "lower": 0.45, "upper": 0.45, "floorwork": 0.0, "jump": 0.0, "spin": 0.0, "pose_hold": 0.25, "instrument": 0.0, "prop": 0.0, "display": "Unknown Chang-E Motion", "semantic_role": "unknown_motion", "energy_label": "moderate", "rhythm_label": "lyrical", "body_focus_label": "full_body", "spatial_label": "in_place", "music_alignment_label": "lyrical_flow", "music_alignment_tags": ["lyrical_flow"], "preferred_music_roles": ["normal"], "preferred_dance_keys": ["lotus_steps", "thirty_six_postures"], "cultural_motif": "unknown", "prop_proxy_label": "unknown", "locomotion_label": "unknown", "support_label": "unknown", "event_family": "unknown", "motion_stage_role": "development", "natural_duration_range_sec": [1.5, 4.0]},\n}\n\nENERGY_LABELS = ["calm", "moderate", "high", "percussive"]\nRHYTHM_LABELS = ["sustained", "lyrical", "accented", "percussive"]\nBODY_FOCUS_LABELS = ["pose", "lower_body", "upper_body", "full_body", "turning_flow"]\nSPATIAL_LABELS = ["in_place", "traveling", "turning", "aerial_leaning"]\nMUSIC_ALIGNMENT_LABELS = ["calm_meditative", "lyrical_flow", "pose_hold", "instrument_phrase", "percussive_accent", "turning_climax", "footwork_flow", "aerial_curve"]\nEVENT_FAMILY_LABELS = ["calm_flow", "pose_motif", "footwork_flow", "turning_flow", "instrument_motif", "percussive_accent", "aerial_curve", "unknown"]\nSTAGE_ROLE_LABELS = ["intro", "development", "build_up", "motif_recall", "anchor_or_resolution", "intro_or_resolution", "opening_or_climax", "accent_or_climax", "climax", "resolution"]\nCATEGORY_CLASS_OVERRIDES = {}\n\n\ndef canonicalize_chang_e_key(key: object) -> str:\n    key_s = str(key or "unknown").strip().lower().replace("-", "_").replace(" ", "_")\n    try:\n        key_s = re.sub(r"_take\\d+$", "", key_s)\n    except Exception:\n        pass\n    aliases = {"mediation": "revelation_meditation", "female_mediation": "revelation_meditation", "male_mediation": "revelation_meditation", "meditation": "revelation_meditation", "36pose": "thirty_six_postures", "36postures": "thirty_six_postures", "thirtysix": "thirty_six_postures", "lotus": "lotus_steps", "pipa": "pipa_behind_back", "drum": "lei_gong_drum", "leigong": "lei_gong_drum", "ribbon": "sogdian_whirl", "ribbon_flow": "sogdian_whirl", "sogdian": "sogdian_whirl", "whirl": "sogdian_whirl", "flying": "flying_apsaras", "apsaras": "flying_apsaras", "feitian": "flying_apsaras"}\n    if key_s in aliases:\n        return aliases[key_s]\n    for k, prof in CHANG_E_CATEGORY_PROFILES.items():\n        if key_s == k or key_s in set(prof.get("aliases", set())):\n            return k\n    return key_s if key_s in CHANG_E_CATEGORY_PROFILES else "unknown"\n\n\ndef _safe_profile_key(meta: dict) -> str:\n    return canonicalize_chang_e_key(meta.get("dance_key") or meta.get("parent_label") or meta.get("label") or meta.get("source_bvh") or "unknown")\n\n\ndef _label_index(label: str, labels: Sequence[str]) -> int:\n    try:\n        return list(labels).index(str(label))\n    except ValueError:\n        return -1\n\n\ndef _parse_numeric_semantic(meta: dict) -> Dict[str, float]:\n    keys = ["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"]\n    vals = [x for x in re.split(r"[;, ]+", str(meta.get("semantic_numeric", "") or "")) if x]\n    out = {}\n    for k, v in zip(keys, vals):\n        try: out[k] = float(v)\n        except Exception: pass\n    return out\n\n\ndef strong_action_semantics_from_meta(meta: dict, desc: Optional[np.ndarray] = None) -> Dict[str, object]:\n    key = _safe_profile_key(meta)\n    base_prof = dict(CHANG_E_CATEGORY_PROFILES.get(key, CHANG_E_CATEGORY_PROFILES["unknown"]))\n    numeric = {k: float(base_prof.get(k, 0.0)) for k in ["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"]}\n    numeric.update(_parse_numeric_semantic(meta))\n    if desc is not None and len(desc) >= 20:\n        numeric["travel"] = max(numeric["travel"], float(np.clip(desc[1] / 1.2, 0.0, 1.0)))\n        numeric["energy"] = max(numeric["energy"], float(np.clip(desc[5] / 0.14, 0.0, 1.0)))\n        numeric["lower"] = max(numeric["lower"], float(np.clip(desc[7] / 0.10, 0.0, 1.0)))\n        numeric["upper"] = max(numeric["upper"], float(np.clip(desc[8] / 0.10, 0.0, 1.0)))\n        numeric["turn"] = max(numeric["turn"], float(np.clip(abs(desc[17]) / 0.22, 0.0, 1.0)))\n        numeric["jump"] = max(numeric["jump"], float(np.clip(desc[18] / 0.20, 0.0, 1.0)))\n        numeric["pose_hold"] = max(numeric["pose_hold"], float(np.clip(1.0 - desc[5] / 0.12, 0.0, 1.0)))\n    prof = dict(base_prof)\n    for field in ["energy_label", "rhythm_label", "body_focus_label", "spatial_label", "music_alignment_label", "semantic_role", "cultural_motif", "prop_proxy_label", "locomotion_label", "support_label", "event_family", "motion_stage_role"]:\n        if meta.get(field):\n            prof[field] = str(meta.get(field))\n    if meta.get("music_alignment_tags"):\n        prof["music_alignment_tags"] = [x for x in re.split(r"[;|,]", str(meta.get("music_alignment_tags"))) if x]\n    if meta.get("preferred_dance_keys"):\n        prof["preferred_dance_keys"] = [canonicalize_chang_e_key(x) for x in re.split(r"[;|,]", str(meta.get("preferred_dance_keys"))) if x]\n    tags = list(dict.fromkeys([str(prof.get("music_alignment_label"))] + [str(x) for x in prof.get("music_alignment_tags", [])]))\n    prof["music_alignment_tags"] = tags\n    prof.setdefault("preferred_music_roles", base_prof.get("preferred_music_roles", ["normal"]))\n    prof.setdefault("preferred_dance_keys", base_prof.get("preferred_dance_keys", [key]))\n    prof["semantic_numeric"] = ";".join(str(float(numeric[k])) for k in ["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"])\n    prof["classification_text"] = f"action={key}; motif={prof.get(\'cultural_motif\')}; family={prof.get(\'event_family\')}; stage={prof.get(\'motion_stage_role\')}; music_align={prof.get(\'music_alignment_label\')}; numeric={prof[\'semantic_numeric\']}"\n    return refine_chang_e_event_semantics(meta, desc, prof)\n\n\ndef _float_meta(meta: dict, key: str, default: float = 0.0) -> float:\n    try:\n        v = meta.get(key, default)\n        if v is None or str(v).lower() in {"nan", "none", "null", ""}:\n            return float(default)\n        return float(v)\n    except Exception:\n        return float(default)\n\n\ndef _bounded01(x: float) -> float:\n    try:\n        return float(np.clip(float(x), 0.0, 1.0))\n    except Exception:\n        return 0.0\n\n\ndef chang_e_event_quality_from_numbers(nums: Dict[str, float], family: str, duration: float, natural_range: Sequence[float]) -> float:\n    """Quality gate for converting long Chang-E BVH into 72BVH-like RAG events."""\n    energy = _bounded01(nums.get("energy", 0.0)); travel = _bounded01(nums.get("travel", 0.0))\n    turn = _bounded01(nums.get("turn", 0.0)); lower = _bounded01(nums.get("lower", 0.0)); upper = _bounded01(nums.get("upper", 0.0))\n    pose_hold = _bounded01(nums.get("pose_hold", 0.0)); jump = _bounded01(nums.get("jump", 0.0)); onset = _bounded01(nums.get("onset", 0.0))\n    contact_ratio = _bounded01(nums.get("contact_ratio", 0.5))\n    root_y = max(0.0, float(nums.get("root_y_range", 0.0)))\n    lo, hi = 1.5, 4.0\n    try:\n        if natural_range and len(natural_range) >= 2:\n            lo, hi = float(natural_range[0]), float(natural_range[-1])\n    except Exception:\n        pass\n    dur = max(1e-3, float(duration or 0.0))\n    center = max(1e-3, 0.5 * (lo + hi))\n    dur_score = 1.0 if (lo <= dur <= hi) else float(np.exp(-abs(np.log(dur / center))))\n    content = max(energy, travel, turn, lower, upper, onset, jump)\n    if family in {"pose_motif", "calm_flow"}:\n        content = max(content * 0.65, pose_hold)\n    # V46.31: stationary Dunhuang postures / meditation motifs are supposed to\n    # have long stable support. Do not score contact_ratio=1.0 as bad gait.\n    if family in {"pose_motif", "calm_flow"} or pose_hold > 0.70:\n        contact_score = 1.0 if contact_ratio >= 0.70 else float(contact_ratio / 0.70)\n    else:\n        contact_score = 1.0 - min(1.0, abs(contact_ratio - 0.46) / 0.54)\n    root_y_penalty = max(0.0, min(0.25, (root_y - 0.35) * 0.35))\n    dead_penalty = 0.0\n    if family not in {"pose_motif", "calm_flow"} and content < 0.20 and pose_hold < 0.45:\n        dead_penalty = 0.25\n    q = 0.42 * content + 0.22 * pose_hold + 0.20 * dur_score + 0.16 * contact_score - root_y_penalty - dead_penalty\n    return float(np.clip(q, 0.02, 1.0))\n\n\ndef chang_e_semantic_event_starts(seq: np.ndarray, cfg: V46Config) -> List[int]:\n    """Boundary-aware starts for Chang-E long BVH.\n\n    The old 72BVH data behaved well because each file was already a compact\n    action unit. Chang-E files are long performances, so we preserve uniform\n    coverage and add motion-novelty anchors around energy/yaw/contact changes.\n    """\n    x = np.asarray(seq, dtype=np.float32)\n    T = int(x.shape[0])\n    win = max(1, min(int(getattr(cfg, "window_len", 120)), T))\n    hop = max(1, int(getattr(cfg, "hop_len", max(1, win // 2))))\n    minf = max(1, int(getattr(cfg, "min_event_frames", 45)))\n    if T <= max(win, minf):\n        return [0]\n    starts = set([0, max(0, T - win)])\n    for st in range(0, max(1, T - minf + 1), hop):\n        starts.add(int(min(max(0, st), max(0, T - minf))))\n    try:\n        joints = fk_24_np(x)\n        v = np.zeros_like(joints)\n        if T > 1:\n            v[1:] = joints[1:] - joints[:-1]\n        energy = np.linalg.norm(v.reshape(T, -1, 3), axis=-1).mean(axis=1)\n        root = x[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]\n        root_v = np.zeros((T,), dtype=np.float32)\n        if T > 1:\n            root_v[1:] = np.linalg.norm(root[1:, [0, 2]] - root[:-1, [0, 2]], axis=-1)\n        yaw = root_yaw_np(x)\n        yaw_v = np.zeros((T,), dtype=np.float32)\n        if T > 1:\n            yaw_v[1:] = np.abs(angle_diff(yaw[1:], yaw[:-1]))\n        foot = joints[:, list(DEFAULT_FOOT_JOINTS)]\n        floor = np.percentile(foot[..., 1].reshape(-1), 5)\n        near = (foot[..., 1] < floor + 0.05).mean(axis=1)\n        novelty = energy + 0.65 * root_v + 0.55 * yaw_v + 0.20 * np.abs(np.diff(near, prepend=near[:1]))\n        if novelty.size > 7:\n            k = np.ones(5, dtype=np.float32) / 5.0\n            novelty = np.convolve(novelty, k, mode="same")\n        thr = float(np.percentile(novelty, 72)) if novelty.size else 0.0\n        order = np.argsort(-novelty)\n        extra = 0\n        max_extra = int(getattr(cfg, "chang_e_boundary_max_extra_starts", 96))\n        min_sep = max(8, min(hop, win // 3))\n        selected_centers: List[int] = []\n        for c in order.tolist():\n            if novelty[int(c)] < thr or extra >= max_extra:\n                break\n            c = int(c)\n            if any(abs(c - q) < min_sep for q in selected_centers):\n                continue\n            selected_centers.append(c)\n            starts.add(int(np.clip(c - win // 2, 0, max(0, T - minf))))\n            starts.add(int(np.clip(c - win // 3, 0, max(0, T - minf))))\n            extra += 2\n    except Exception:\n        pass\n    out = sorted(starts)\n    merged: List[int] = []\n    min_start_sep = max(6, min(hop // 2, win // 4))\n    tail = max(0, T - win)\n    # V46.31: if the sequence is only slightly longer than one window, [0:win]\n    # and [tail:T] are >95% overlapping twins. Keep one centered representative\n    # rather than polluting the RAG DB with near-identical embeddings.\n    if 0 < tail < min_start_sep:\n        return [int(max(0, tail // 2))]\n    for st in out:\n        st = int(st)\n        # V46.31: keep coverage endpoints without creating near-duplicate twins.\n        # When a novelty start lies too close to the required tail window, replace\n        # the previous start with the exact tail start instead of appending both.\n        if st == 0:\n            if not merged:\n                merged.append(0)\n            continue\n        if st == tail:\n            if merged and abs(st - merged[-1]) < min_start_sep:\n                # V46.31: protect the unique opener for very short sequences.\n                # If T is only slightly larger than win, tail can be only a few\n                # frames after 0. Replacing [0] with [tail] permanently drops\n                # opening frames. Keep the opener and add the tail only in this\n                # unique two-endpoint case; otherwise replace the near-duplicate.\n                if len(merged) == 1 and merged[-1] == 0 and st != 0:\n                    merged.append(st)\n                elif not (len(merged) == 1 and merged[-1] == 0):\n                    merged[-1] = st\n            elif not merged or merged[-1] != st:\n                merged.append(st)\n            continue\n        if not merged or abs(st - merged[-1]) >= min_start_sep:\n            merged.append(st)\n    if merged:\n        if abs(tail - merged[-1]) < min_start_sep:\n            if len(merged) == 1 and merged[-1] == 0 and tail != 0:\n                merged.append(tail)\n            elif not (len(merged) == 1 and merged[-1] == 0):\n                merged[-1] = tail\n        elif merged[-1] != tail:\n            merged.append(tail)\n    else:\n        merged = [0, tail] if tail > 0 else [0]\n    return sorted(set(int(x) for x in merged))\n\n\ndef refine_chang_e_event_semantics(meta: dict, desc: Optional[np.ndarray], prof: Dict[str, object]) -> Dict[str, object]:\n    # V46.31 window-level semantics for Chang-E event slicing.\n    # Chang-E is a long, category-complete MoCap corpus; each local window is\n    # converted into a curated semantic event comparable to the old 72BVH units.\n    out = dict(prof)\n    key = _safe_profile_key(meta)\n    nums = _parse_numeric_semantic(out)\n    if desc is not None and len(desc) >= 20:\n        nums["duration"] = float(desc[0])\n        nums["travel"] = max(float(nums.get("travel", 0.0)), float(np.clip(desc[1] / 1.15, 0.0, 1.0)))\n        nums["energy"] = max(float(nums.get("energy", 0.0)), float(np.clip(desc[5] / 0.135, 0.0, 1.0)))\n        nums["lower"] = max(float(nums.get("lower", 0.0)), float(np.clip(desc[7] / 0.10, 0.0, 1.0)))\n        nums["upper"] = max(float(nums.get("upper", 0.0)), float(np.clip(desc[8] / 0.10, 0.0, 1.0)))\n        nums["turn"] = max(float(nums.get("turn", 0.0)), float(np.clip(abs(desc[17]) / 0.20, 0.0, 1.0)))\n        nums["root_y_range"] = float(desc[18])\n        nums["contact_ratio"] = float(desc[10])\n        local_hold = float(np.clip(1.0 - desc[5] / 0.115, 0.0, 1.0)) * float(np.clip(0.55 + desc[10], 0.0, 1.0))\n        nums["pose_hold"] = max(float(nums.get("pose_hold", 0.0)), local_hold)\n        nums["jump"] = max(float(nums.get("jump", 0.0)), float(np.clip((desc[18] - 0.035) / 0.16, 0.0, 1.0)))\n        nums["spin"] = max(float(nums.get("spin", 0.0)), float(np.clip(abs(desc[17]) / 0.22, 0.0, 1.0)))\n    frac_mid = _float_meta(meta, "event_position_mid", _float_meta(meta, "event_position_fraction", 0.5))\n    duration = float(nums.get("duration", _float_meta(meta, "duration", 0.0)))\n    energy = float(nums.get("energy", 0.0)); onset = float(nums.get("onset", 0.0))\n    travel = float(nums.get("travel", 0.0)); turn = float(nums.get("turn", 0.0))\n    lower = float(nums.get("lower", 0.0)); upper = float(nums.get("upper", 0.0))\n    pose_hold = float(nums.get("pose_hold", 0.0)); jump = float(nums.get("jump", 0.0)); spin = float(nums.get("spin", 0.0))\n    contact_ratio = float(nums.get("contact_ratio", 0.5))\n    if pose_hold > 0.72 and energy < 0.60:\n        family = "pose_motif"\n    elif turn > 0.68 or spin > 0.68:\n        family = "turning_flow"\n    elif onset > 0.72 or (energy > 0.78 and key == "lei_gong_drum"):\n        family = "percussive_accent"\n    elif key == "pipa_behind_back" and upper >= lower * 1.12:\n        family = "instrument_motif"\n    elif travel > 0.50 or lower > upper * 1.18:\n        family = "footwork_flow"\n    elif key == "flying_apsaras" or jump > 0.45:\n        family = "aerial_curve"\n    elif energy < 0.34:\n        family = "calm_flow"\n    else:\n        family = str(out.get("event_family", "footwork_flow"))\n    out["event_family"] = family\n    if frac_mid < 0.12:\n        stage = "intro"\n    elif frac_mid > 0.88:\n        stage = "resolution"\n    elif family in {"percussive_accent", "turning_flow"} and energy > 0.55:\n        stage = "climax"\n    elif family == "instrument_motif":\n        stage = "motif_recall"\n    elif pose_hold > 0.70:\n        stage = "anchor_or_resolution"\n    elif energy > 0.58 or travel > 0.55:\n        stage = "build_up"\n    else:\n        stage = "development"\n    out["motion_stage_role"] = stage\n    if contact_ratio < 0.18 or jump > 0.45:\n        out["support_label"] = "low_contact_flight_like"\n    elif pose_hold > 0.72:\n        out["support_label"] = "stable_support"\n    elif turn > 0.55:\n        out["support_label"] = "alternating_or_pivot_support"\n    elif travel > 0.45 or lower > 0.60:\n        out["support_label"] = "alternating_foot_support"\n    else:\n        out.setdefault("support_label", "stable_support")\n    if turn > 0.60:\n        out["locomotion_label"] = "turning_travel"; out["spatial_label"] = "turning"\n    elif travel > 0.50:\n        out["locomotion_label"] = "traveling_steps"; out["spatial_label"] = "traveling"\n    elif pose_hold > 0.70:\n        out["locomotion_label"] = "in_place_pose"; out["spatial_label"] = "in_place"\n    elif energy < 0.34:\n        out["locomotion_label"] = "slow_weight_shift"; out["spatial_label"] = "in_place"\n    family_to_align = {"calm_flow": "calm_meditative", "pose_motif": "pose_hold", "footwork_flow": "footwork_flow", "turning_flow": "turning_climax", "instrument_motif": "instrument_phrase", "percussive_accent": "percussive_accent", "aerial_curve": "lyrical_flow"}\n    out["music_alignment_label"] = family_to_align.get(family, str(out.get("music_alignment_label", "lyrical_flow")))\n    tags = [out["music_alignment_label"], family, stage, str(out.get("support_label", "")), str(out.get("locomotion_label", ""))]\n    tags += [str(x) for x in out.get("music_alignment_tags", [])]\n    out["music_alignment_tags"] = list(dict.fromkeys([x for x in tags if x]))\n    out["event_position_mid"] = float(frac_mid)\n    natural_range = out.get("natural_duration_range_sec", CHANG_E_CATEGORY_PROFILES.get(key, CHANG_E_CATEGORY_PROFILES["unknown"]).get("natural_duration_range_sec", [1.5, 4.0]))\n    q = chang_e_event_quality_from_numbers(nums, family, duration, natural_range)\n    out["event_quality_score"] = float(q)\n    out["semantic_confidence"] = float(np.clip(0.25 + 0.50 * q + 0.25 * max(energy, pose_hold, turn, travel, upper, lower), 0.10, 1.0))\n    keys = ["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"]\n    out["semantic_numeric"] = ";".join(str(float(nums.get(k, 0.0))) for k in keys)\n    out["classification_text"] = (\n        f"action={key}; motif={out.get(\'cultural_motif\')}; family={out.get(\'event_family\')}; "\n        f"stage={out.get(\'motion_stage_role\')}; support={out.get(\'support_label\')}; "\n        f"locomotion={out.get(\'locomotion_label\')}; music_align={out.get(\'music_alignment_label\')}; "\n        f"event_mid={float(frac_mid):.3f}; quality={q:.3f}; semantic_conf={out[\'semantic_confidence\']:.3f}; numeric={out[\'semantic_numeric\']}"\n    )\n    return out\n\n\ndef class_semantic_vector_from_meta(meta: dict, cfg: Optional[V46Config] = None) -> np.ndarray:\n    key = _safe_profile_key(meta)\n    base = filename_semantic_vector_from_meta(meta, cfg).copy()\n    cls = strong_action_semantics_from_meta(meta)\n    nums = _parse_numeric_semantic(cls)\n    for k in ["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"]:\n        nums.setdefault(k, float(CHANG_E_CATEGORY_PROFILES.get(key, CHANG_E_CATEGORY_PROFILES["unknown"]).get(k, 0.0)))\n    known = [k for k in CHANG_E_CATEGORY_PROFILES.keys() if k != "unknown"]\n    ci = known.index(key) if key in known else -1\n    align_i = _label_index(str(cls.get("music_alignment_label")), MUSIC_ALIGNMENT_LABELS)\n    family_i = _label_index(str(cls.get("event_family")), EVENT_FAMILY_LABELS)\n    stage_i = _label_index(str(cls.get("motion_stage_role")), STAGE_ROLE_LABELS)\n    v = base.astype(np.float32)\n    for i,k in enumerate(["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"]):\n        v[8+i] = nums[k]\n    v[20] = 0.0 if ci < 0 else ci / max(1, len(known) - 1)\n    v[21] = 0.0 if align_i < 0 else align_i / max(1, len(MUSIC_ALIGNMENT_LABELS) - 1)\n    v[22] = 0.0 if family_i < 0 else family_i / max(1, len(EVENT_FAMILY_LABELS) - 1)\n    v[23] = 0.0 if stage_i < 0 else stage_i / max(1, len(STAGE_ROLE_LABELS) - 1)\n    tags = set(str(x) for x in cls.get("music_alignment_tags", []))\n    v[28] = 1.0 if ("calm_meditative" in tags or cls.get("event_family") == "calm_flow") else 0.0\n    v[29] = 1.0 if ("percussive_accent" in tags or nums["onset"] > 0.65) else 0.0\n    v[30] = 1.0 if ("turning_climax" in tags or nums["spin"] > 0.65 or nums["turn"] > 0.65) else 0.0\n    v[31] = 1.0\n    return np.clip(v, 0.0, 1.0).astype(np.float32)\n\n\n'

def replace_between(text: str, start_pat: str, end_pat: str, replacement: str, label: str) -> str:
    start = text.find(start_pat)
    if start < 0:
        raise RuntimeError(f"Cannot find start marker for {label}: {start_pat!r}")
    end = text.find(end_pat, start)
    if end < 0:
        raise RuntimeError(f"Cannot find end marker for {label}: {end_pat!r}")
    return text[:start] + replacement + text[end:]


def replace_helper_block(text: str) -> str:
    """Replace any existing V46 research-contract helper block robustly.

    V46.31 hotfix:
    Do not replace through ``def _clean_stem`` when the Chang-E semantic
    ontology sits between the V46 helper marker and _clean_stem. Otherwise the
    patcher deletes CHANG_E_CATEGORY_PROFILES in memory and the later semantic
    replacement cannot find its anchor.
    """
    match = re.search(r"# V46\.\d+ research contract guards for Chang-E/change RAG DB", text)
    if match:
        start = text.rfind("# -----------------------------------------------------------------------------", 0, match.start())
        if start < 0:
            start = match.start()

        anchors = [
            "CHANG_E_CATEGORY_PROFILES = {",
            "def audio_slot_classification_from_pseudo",
            "def _clean_stem(path: str | Path) -> str:",
        ]
        candidates = []
        for anchor in anchors:
            pos = text.find(anchor, match.start())
            if pos >= 0:
                candidates.append(pos)

        if not candidates:
            raise RuntimeError("Cannot locate end of existing V46 helper block after regex marker.")

        end = min(candidates)
        return text[:start] + HELPERS + "\n" + text[end:]

    anchor = "    return entry.astype(np.float32), exit_.astype(np.float32), contact[0], contact[-1]\n\n\n"
    if anchor not in text:
        raise RuntimeError("Cannot locate motion_boundary_state insertion anchor.")
    return text.replace(anchor, anchor + HELPERS, 1)

def replace_chang_e_semantics(text: str) -> str:
    """Replace V46 filename-only Chang-E semantics with V46.31 enriched ontology."""
    start = text.find("CHANG_E_CATEGORY_PROFILES = {")
    end = text.find("def audio_slot_classification_from_pseudo", start if start >= 0 else 0)

    if start >= 0 and end >= 0:
        return text[:start] + NEW_CHANG_E_SEMANTICS + text[end:]

    # Fallback for partially patched files: if the ontology block is missing but
    # the downstream audio-slot semantic function is still present, insert the
    # full V46.31 ontology immediately before it.
    if start < 0 and end >= 0:
        return text[:end] + NEW_CHANG_E_SEMANTICS + text[end:]

    raise RuntimeError("Cannot locate Chang-E semantic block for V46.31 replacement.")

def ensure_v46_24_config_fields(text: str) -> str:
    anchor = '    classification_report_topk: int = 8\n'
    if anchor not in text:
        raise RuntimeError("Cannot locate classification_report_topk config anchor.")
    fields = (
        '    # V46.31: Chang-E event-level semantic routing.\n'
        '    chang_e_event_semantic_enable: bool = True\n'
        '    semantic_routing_weight: float = 0.72\n'
        '    event_family_bonus: float = 0.58\n'
        '    motion_stage_role_bonus: float = 0.36\n'
        '    preferred_dance_key_bonus: float = 0.28\n'
        '    route_natural_duration_weight: float = 0.20\n'
        '    route_family_balance_penalty: float = 0.18\n'
        '    route_family_recent_window: int = 8\n'
        '    route_family_penalty_cap: float = 0.25\n'
        '    route_dance_key_repeat_penalty: float = 0.16\n'
        '    route_family_repeat_penalty: float = 0.12\n'
        '    route_source_repeat_penalty: float = 0.10\n'
        '    route_motif_recall_bonus: float = 0.12\n'
        '    route_debug_topk: int = 10\n'
        '    # V46.31: convert long Chang-E BVH into a curated 72BVH-like semantic event library.\n'
        '    chang_e_boundary_event_split: bool = True\n'
        '    chang_e_boundary_max_extra_starts: int = 96\n'
        '    chang_e_min_event_quality: float = 0.22\n'
        '    chang_e_keep_pose_anchor_quality: float = 0.16\n'
        '    event_quality_weight: float = 0.22\n'
        '    route_support_bonus: float = 0.12\n'
        '    route_locomotion_bonus: float = 0.14\n'
        '    route_stage_sequence_weight: float = 0.16\n'
        '    route_source_run_hard_penalty: float = 0.30\n'
        '    route_semantic_bonus_scale: float = 1.50\n'
    )
    if 'chang_e_event_semantic_enable: bool' not in text:
        text = text.replace(anchor, anchor + fields, 1)
    elif 'chang_e_boundary_event_split: bool' not in text:
        extra_anchor = '    route_debug_topk: int = 10\n'
        extra_fields = (
            '    # V46.31: convert long Chang-E BVH into a curated 72BVH-like semantic event library.\n'
            '    chang_e_boundary_event_split: bool = True\n'
            '    chang_e_boundary_max_extra_starts: int = 96\n'
            '    chang_e_min_event_quality: float = 0.22\n'
            '    chang_e_keep_pose_anchor_quality: float = 0.16\n'
            '    event_quality_weight: float = 0.22\n'
            '    route_support_bonus: float = 0.12\n'
            '    route_locomotion_bonus: float = 0.14\n'
            '    route_stage_sequence_weight: float = 0.16\n'
            '    route_source_run_hard_penalty: float = 0.30\n'
        )
        if extra_anchor in text:
            text = text.replace(extra_anchor, extra_anchor + extra_fields, 1)
    return text

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
        '            "V46_CHANG_E_EVENT_SEMANTIC_ENABLE": ("chang_e_event_semantic_enable", lambda x: bool(int(x))),\n',
        '            "V46_SEMANTIC_ROUTING_WEIGHT": ("semantic_routing_weight", float),\n',
        '            "V46_EVENT_FAMILY_BONUS": ("event_family_bonus", float),\n',
        '            "V46_MOTION_STAGE_ROLE_BONUS": ("motion_stage_role_bonus", float),\n',
        '            "V46_PREFERRED_DANCE_KEY_BONUS": ("preferred_dance_key_bonus", float),\n',
        '            "V46_ROUTE_NATURAL_DURATION_WEIGHT": ("route_natural_duration_weight", float),\n',
        '            "V46_ROUTE_FAMILY_BALANCE_PENALTY": ("route_family_balance_penalty", float),\n',
        '            "V46_ROUTE_FAMILY_RECENT_WINDOW": ("route_family_recent_window", int),\n',
        '            "V46_ROUTE_FAMILY_PENALTY_CAP": ("route_family_penalty_cap", float),\n',
        '            "V46_ROUTE_DANCE_KEY_REPEAT_PENALTY": ("route_dance_key_repeat_penalty", float),\n',
        '            "V46_ROUTE_FAMILY_REPEAT_PENALTY": ("route_family_repeat_penalty", float),\n',
        '            "V46_ROUTE_SOURCE_REPEAT_PENALTY": ("route_source_repeat_penalty", float),\n',
        '            "V46_ROUTE_MOTIF_RECALL_BONUS": ("route_motif_recall_bonus", float),\n',
        '            "V46_ROUTE_DEBUG_TOPK": ("route_debug_topk", int),\n',
        '            "V46_CHANG_E_BOUNDARY_EVENT_SPLIT": ("chang_e_boundary_event_split", lambda x: bool(int(x))),\n',
        '            "V46_CHANG_E_BOUNDARY_MAX_EXTRA_STARTS": ("chang_e_boundary_max_extra_starts", int),\n',
        '            "V46_CHANG_E_MIN_EVENT_QUALITY": ("chang_e_min_event_quality", float),\n',
        '            "V46_CHANG_E_KEEP_POSE_ANCHOR_QUALITY": ("chang_e_keep_pose_anchor_quality", float),\n',
        '            "V46_EVENT_QUALITY_WEIGHT": ("event_quality_weight", float),\n',
        '            "V46_ROUTE_SUPPORT_BONUS": ("route_support_bonus", float),\n',
        '            "V46_ROUTE_LOCOMOTION_BONUS": ("route_locomotion_bonus", float),\n',
        '            "V46_ROUTE_STAGE_SEQUENCE_WEIGHT": ("route_stage_sequence_weight", float),\n',
        '            "V46_ROUTE_SOURCE_RUN_HARD_PENALTY": ("route_source_run_hard_penalty", float),\n',
        '            "V46_ROUTE_SEMANTIC_BONUS_SCALE": ("route_semantic_bonus_scale", float),\n',
    ]
    missing = [line for line in insert_lines if line.strip() not in text]
    if missing:
        text = text.replace(anchor, anchor + "".join(missing), 1)
    return text


NEW_SEMANTIC_LABEL_MATCH_BONUS = 'def semantic_label_match_bonus(slot: dict, db: dict, cfg: V46Config) -> np.ndarray:\n    """V46.31 interpretable music-router bonus for Chang-E semantic Event-RAG."""\n    n = len(db.get("paths", []))\n    bonus = np.zeros(n, dtype=np.float32)\n    if not bool(getattr(cfg, "classification_semantic_enable", True)) or n == 0:\n        return bonus\n    dance_keys = np.asarray(db.get("dance_keys", np.array(["unknown"] * n, dtype=object)), dtype=object)\n    roles = np.asarray(db.get("semantic_roles", np.array(["unknown"] * n, dtype=object)), dtype=object)\n    energy = np.asarray(db.get("energy_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)\n    rhythm = np.asarray(db.get("rhythm_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)\n    align = np.asarray(db.get("music_alignment_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)\n    families = np.asarray(db.get("event_families", np.array(["unknown"] * n, dtype=object)), dtype=object)\n    stages = np.asarray(db.get("motion_stage_roles", np.array(["unknown"] * n, dtype=object)), dtype=object)\n    motifs = np.asarray(db.get("cultural_motifs", np.array(["unknown"] * n, dtype=object)), dtype=object)\n    locomotion = np.asarray(db.get("locomotion_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)\n    support = np.asarray(db.get("support_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)\n    quality = np.asarray(db.get("event_quality_scores", np.ones(n, dtype=np.float32)), dtype=np.float32)\n    preferred = [canonicalize_chang_e_key(x) for x in slot.get("preferred_dance_keys", [])]\n    preferred_roles = [str(x) for x in slot.get("preferred_semantic_roles", [])]\n    slot_align = str(slot.get("music_alignment_label", slot.get("music_semantic_top_label", "")))\n    slot_energy = str(slot.get("energy_label", "")); slot_rhythm = str(slot.get("rhythm_label", "")); slot_role = str(slot.get("role", "normal"))\n    route_family_map = {"calm_meditative": ["calm_flow", "pose_motif", "aerial_curve"], "pose_hold": ["pose_motif", "calm_flow", "instrument_motif"], "lyrical_flow": ["aerial_curve", "footwork_flow", "instrument_motif", "calm_flow"], "footwork_flow": ["footwork_flow", "turning_flow", "aerial_curve"], "instrument_phrase": ["instrument_motif", "aerial_curve", "pose_motif"], "percussive_accent": ["percussive_accent", "turning_flow", "instrument_motif"], "turning_climax": ["turning_flow", "aerial_curve", "percussive_accent"], "aerial_curve": ["aerial_curve", "turning_flow", "footwork_flow"]}\n    route_stage_map = {"intro": ["intro", "intro_or_resolution", "anchor_or_resolution"], "calm": ["intro", "intro_or_resolution", "resolution", "anchor_or_resolution"], "normal": ["development", "build_up", "motif_recall"], "development": ["development", "build_up"], "build_up": ["build_up", "development", "opening_or_climax"], "motif": ["motif_recall", "development"], "motif_recall": ["motif_recall", "anchor_or_resolution"], "accent": ["accent_or_climax", "climax", "build_up"], "climax": ["climax", "accent_or_climax", "opening_or_climax"], "release": ["resolution", "anchor_or_resolution", "intro_or_resolution"], "resolution": ["resolution", "anchor_or_resolution", "intro_or_resolution"]}\n    route_support_map = {"calm_meditative": ["stable_support", "static_or_low_motion_support"], "pose_hold": ["stable_support", "static_or_low_motion_support"], "footwork_flow": ["alternating_foot_support", "alternating_or_pivot_support"], "turning_climax": ["alternating_or_pivot_support", "low_contact_flight_like"], "percussive_accent": ["strong_foot_contact", "alternating_foot_support"], "lyrical_flow": ["alternating_foot_support", "low_contact_flight_like", "stable_support"], "instrument_phrase": ["stable_support", "alternating_foot_support"]}\n    route_loco_map = {"calm_meditative": ["slow_weight_shift", "in_place_pose", "floating_leaning"], "pose_hold": ["in_place_pose", "slow_weight_shift"], "footwork_flow": ["traveling_steps", "turning_travel"], "turning_climax": ["turning_travel", "floating_leaning"], "percussive_accent": ["accented_travel", "turning_travel", "traveling_steps"], "lyrical_flow": ["floating_leaning", "traveling_steps", "upper_body_phrase"], "instrument_phrase": ["upper_body_phrase", "in_place_pose"]}\n    for k in preferred:\n        if k:\n            bonus += (dance_keys == k).astype(np.float32) * float(getattr(cfg, "preferred_dance_key_bonus", 0.28))\n    for r in preferred_roles:\n        if r:\n            bonus += (roles == r).astype(np.float32) * 0.25\n    if slot_align:\n        bonus += (align == slot_align).astype(np.float32) * 0.45\n        for rank, fam in enumerate(route_family_map.get(slot_align, [])):\n            bonus += (families == fam).astype(np.float32) * float(getattr(cfg, "event_family_bonus", 0.58)) / float(rank + 1)\n        for rank, sup in enumerate(route_support_map.get(slot_align, [])):\n            bonus += (support == sup).astype(np.float32) * float(getattr(cfg, "route_support_bonus", 0.12)) / float(rank + 1)\n        for rank, loc in enumerate(route_loco_map.get(slot_align, [])):\n            bonus += (locomotion == loc).astype(np.float32) * float(getattr(cfg, "route_locomotion_bonus", 0.14)) / float(rank + 1)\n    if slot_role:\n        for rank, st in enumerate(route_stage_map.get(slot_role, [])):\n            bonus += (stages == st).astype(np.float32) * float(getattr(cfg, "motion_stage_role_bonus", 0.36)) / float(rank + 1)\n    if slot_energy:\n        bonus += (energy == slot_energy).astype(np.float32) * 0.14\n    if slot_rhythm:\n        bonus += (rhythm == slot_rhythm).astype(np.float32) * 0.12\n    if slot_align == "instrument_phrase":\n        bonus += np.isin(motifs, ["pipa_instrument_pose", "thunder_drum"]).astype(np.float32) * 0.16\n    conf = np.asarray(db.get("semantic_confidence", np.ones(n, dtype=np.float32)), dtype=np.float32)\n    q_gate = np.clip(0.45 + 0.55 * quality, 0.25, 1.15)\n    bonus *= np.clip(0.65 + 0.35 * conf, 0.5, 1.15) * q_gate\n    # V46.31: never normalize by the current candidate max.  That dynamic\n    # min-max scaling can turn a weak accidental match in a vague slot into a\n    # full-strength routing reward.  Use a fixed saturating scale instead.\n    scale = max(0.25, float(getattr(cfg, "route_semantic_bonus_scale", 1.50)))\n    bonus = 1.0 - np.exp(-np.maximum(bonus, 0.0) / scale)\n    return np.clip(bonus, 0.0, 1.0).astype(np.float32)\n\n\n'
NEW_RETRIEVE_SCHEDULE = 'def retrieve_schedule(slots: List[dict], slot_feat: np.ndarray, db: dict, cfg: V46Config, contrastive=None) -> Tuple[List[int], List[dict]]:\n    """V46.31 retrieval: contrastive similarity + curated Chang-E semantic event router."""\n    desc = np.asarray(db["desc"], dtype=np.float32)\n    desc_z = motion_feature_z_for_alignment(db, cfg, weight=float(getattr(cfg, "classification_retrieval_weight", getattr(cfg, "filename_semantic_retrieval_weight", 0.20))))\n    mean = np.asarray(db["desc_mean"], dtype=np.float32)\n    std = np.asarray(db["desc_std"], dtype=np.float32)\n    if contrastive is not None and hasattr(contrastive, "music_mean") and hasattr(contrastive, "music_std"):\n        music_mean = np.asarray(getattr(contrastive, "music_mean"), dtype=np.float32)\n        music_std = np.asarray(getattr(contrastive, "music_std"), dtype=np.float32)\n        music_z = (slot_feat - music_mean) / np.maximum(music_std, 1e-6)\n    else:\n        music_z = (slot_feat - mean) / std\n    music_z = np.clip(music_z, -8.0, 8.0).astype(np.float32)\n    desc_z = np.clip(desc_z, -8.0, 8.0).astype(np.float32)\n    music_emb, motion_emb = embed_with_contrastive(contrastive, music_z, desc_z, cfg)\n    sources = np.asarray(db["source_groups"], dtype=object)\n    durations = np.asarray(db["durations"], dtype=np.float32)\n    entries = np.asarray(db["entry"], dtype=np.float32); exits = np.asarray(db["exit"], dtype=np.float32)\n    centry = np.asarray(db["contact_entry"], dtype=np.float32); cexit = np.asarray(db["contact_exit"], dtype=np.float32)\n    dance_keys = np.asarray(db.get("dance_keys", np.array(["unknown"] * len(desc), dtype=object)), dtype=object)\n    labels_arr = np.asarray(db.get("labels", np.array(["unknown"] * len(desc), dtype=object)), dtype=object)\n    align_arr = np.asarray(db.get("music_alignment_labels", np.array(["unknown"] * len(desc), dtype=object)), dtype=object)\n    families = np.asarray(db.get("event_families", np.array(["unknown"] * len(desc), dtype=object)), dtype=object)\n    stages = np.asarray(db.get("motion_stage_roles", np.array(["unknown"] * len(desc), dtype=object)), dtype=object)\n    locomotion = np.asarray(db.get("locomotion_labels", np.array(["unknown"] * len(desc), dtype=object)), dtype=object)\n    support = np.asarray(db.get("support_labels", np.array(["unknown"] * len(desc), dtype=object)), dtype=object)\n    sem_conf = np.asarray(db.get("semantic_confidence", np.ones(len(desc), dtype=np.float32)), dtype=np.float32)\n    event_quality = np.asarray(db.get("event_quality_scores", np.ones(len(desc), dtype=np.float32)), dtype=np.float32)\n    nat_min = np.asarray(db.get("natural_duration_min", np.ones(len(desc), dtype=np.float32) * 1.5), dtype=np.float32)\n    nat_max = np.asarray(db.get("natural_duration_max", np.ones(len(desc), dtype=np.float32) * 4.0), dtype=np.float32)\n    beams: List[Tuple[float, List[int], Dict[str, int]]] = [(0.0, [], {})]\n    reports: List[dict] = []\n    for i, slot in enumerate(slots):\n        sim = music_emb[i] @ motion_emb.T\n        slot_dur = max(float(slot.get("duration", durations.mean() if len(durations) else 1.0)), 1e-4)\n        dur_cost = np.abs(np.log(np.maximum(durations, 1e-4) / slot_dur))\n        class_bonus = semantic_label_match_bonus(slot, db, cfg)\n        in_range = ((slot_dur >= nat_min) & (slot_dur <= nat_max)).astype(np.float32)\n        center = np.maximum((nat_min + nat_max) * 0.5, 1e-4)\n        natural_score = in_range + (1.0 - in_range) * np.exp(-np.abs(np.log(slot_dur / center))).astype(np.float32)\n        quality_term = np.clip(event_quality, 0.0, 1.0)\n        low_quality_penalty = np.maximum(0.0, float(getattr(cfg, "chang_e_min_event_quality", 0.22)) - quality_term)\n        base_score = (sim - cfg.retrieval_warp_penalty * dur_cost + float(getattr(cfg, "semantic_routing_weight", 0.72)) * class_bonus + float(getattr(cfg, "route_natural_duration_weight", 0.20)) * natural_score + float(getattr(cfg, "event_quality_weight", 0.22)) * quality_term + 0.04 * np.clip(sem_conf, 0.0, 1.0) - 0.75 * low_quality_penalty)\n        cand = np.argsort(-base_score)[: max(cfg.top_k, cfg.beam_size, int(getattr(cfg, "route_debug_topk", 10)))].tolist()\n        new_beams: List[Tuple[float, List[int], Dict[str, int]]] = []\n        for score, path, usage in beams:\n            prev = path[-1] if path else None\n            for idx in cand:\n                sc = float(base_score[idx])\n                src = str(sources[idx]); dk = str(dance_keys[idx]); fam = str(families[idx]); stg = str(stages[idx])\n                sc -= float(getattr(cfg, "route_source_repeat_penalty", cfg.retrieval_source_penalty)) * usage.get("src::" + src, 0)\n                sc -= float(getattr(cfg, "route_dance_key_repeat_penalty", 0.16)) * usage.get("dance::" + dk, 0)\n                # V46.31: family diversity is local-window based and capped.\n                # For 3-5 minute dances, global family counts inevitably grow;\n                # an unbounded penalty can overpower the music semantic match and\n                # force wrong rare families in the later song.\n                fam_recent_window = max(1, int(getattr(cfg, "route_family_recent_window", 8)))\n                fam_recent_count = sum(1 for p_idx in path[-fam_recent_window:] if str(families[p_idx]) == fam)\n                fam_pen = float(getattr(cfg, "route_family_balance_penalty", 0.18)) * max(0, fam_recent_count - 1)\n                fam_pen = min(float(getattr(cfg, "route_family_penalty_cap", 0.25)), fam_pen)\n                sc -= fam_pen\n                # V46.31: source-run hard penalty is consecutive-run only.\n                # Global source usage above remains a soft diversity prior; do not\n                # blacklist high-quality sources for the entire later song merely\n                # because they were selected twice earlier.\n                run_count = 0\n                for p_idx in reversed(path):\n                    if str(sources[p_idx]) == src:\n                        run_count += 1\n                    else:\n                        break\n                if run_count >= 2:\n                    sc -= float(getattr(cfg, "route_source_run_hard_penalty", 0.30))\n                if str(slot.get("role", "")) in {"motif", "motif_recall"} and usage.get("fam::" + fam, 0) > 0:\n                    sc += float(getattr(cfg, "route_motif_recall_bonus", 0.12))\n                if i == 0 and stg in {"intro", "intro_or_resolution"}:\n                    sc += float(getattr(cfg, "route_stage_sequence_weight", 0.16))\n                elif i >= len(slots) - 2 and stg in {"resolution", "anchor_or_resolution", "intro_or_resolution"}:\n                    sc += float(getattr(cfg, "route_stage_sequence_weight", 0.16))\n                elif str(slot.get("role", "")) in {"build_up", "climax", "accent"} and stg in {"build_up", "climax", "accent_or_climax"}:\n                    sc += float(getattr(cfg, "route_stage_sequence_weight", 0.16)) * 0.8\n                if prev is not None:\n                    sc -= cfg.retrieval_transition_penalty * transition_cost(exits[prev], entries[idx], cexit[prev], centry[idx])\n                    if src == str(sources[prev]):\n                        sc -= cfg.retrieval_repeat_penalty\n                    if fam == str(families[prev]):\n                        sc -= float(getattr(cfg, "route_family_repeat_penalty", 0.12))\n                ns = dict(usage)\n                ns["src::" + src] = ns.get("src::" + src, 0) + 1; ns["dance::" + dk] = ns.get("dance::" + dk, 0) + 1; ns["fam::" + fam] = ns.get("fam::" + fam, 0) + 1\n                new_beams.append((score + sc, path + [int(idx)], ns))\n        new_beams.sort(key=lambda x: x[0], reverse=True); beams = new_beams[: cfg.beam_size]\n        preview_n = max(1, min(int(getattr(cfg, "classification_report_topk", 8)), len(cand)))\n        reports.append({"slot": i, "start": slot.get("start"), "end": slot.get("end"), "duration": slot.get("duration"), "slot_role": slot.get("role"), "slot_music_alignment_label": slot.get("music_alignment_label"), "slot_music_semantic_top_label": slot.get("music_semantic_top_label", slot.get("music_alignment_label")), "slot_preferred_dance_keys": slot.get("preferred_dance_keys", []), "top_candidate": int(cand[0]), "top_candidate_label": str(labels_arr[cand[0]]), "top_candidate_dance_key": str(dance_keys[cand[0]]), "top_candidate_event_family": str(families[cand[0]]), "top_candidate_stage_role": str(stages[cand[0]]), "top_candidate_support_label": str(support[cand[0]]), "top_candidate_locomotion_label": str(locomotion[cand[0]]), "top_candidate_event_quality": float(event_quality[cand[0]]), "top_candidate_music_alignment_label": str(align_arr[cand[0]]), "beam_best_score": float(beams[0][0]), "routing_policy": "V46.31 curated semantic Event-RAG: contrastive/descriptor + family/stage/support/locomotion + quality + diversity", "candidate_preview": [{"event_id": int(j), "score": float(base_score[int(j)]), "semantic_route_bonus": float(class_bonus[int(j)]), "natural_duration_score": float(natural_score[int(j)]), "event_quality": float(event_quality[int(j)]), "source": str(sources[int(j)]), "label": str(labels_arr[int(j)]), "dance_key": str(dance_keys[int(j)]), "event_family": str(families[int(j)]), "motion_stage_role": str(stages[int(j)]), "support_label": str(support[int(j)]), "locomotion_label": str(locomotion[int(j)]), "music_alignment_label": str(align_arr[int(j)])} for j in cand[:preview_n]]})\n    return beams[0][1], reports\n\n\n'

def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)
    text = TARGET.read_text(encoding="utf-8")
    backup = TARGET.with_suffix(TARGET.suffix + f".v46_31_timing_sync_fix_{time.strftime('%Y%m%d_%H%M%S')}.bak")
    shutil.copy2(TARGET, backup)

    text = replace_helper_block(text)
    text = ensure_v46_24_config_fields(text)
    text = ensure_env_lines(text)
    text = replace_chang_e_semantics(text)
    old_arrays = '        classification_texts=np.array([m.get("classification_text", "") for m in meta], dtype=object),\n        take_ids=np.array([int(m.get("take_id", -1) if m.get("take_id", -1) is not None else -1) for m in meta], dtype=np.int32),\n'
    new_arrays = '        classification_texts=np.array([m.get("classification_text", "") for m in meta], dtype=object),\n        event_families=np.array([m.get("event_family", "unknown") for m in meta], dtype=object),\n        motion_stage_roles=np.array([m.get("motion_stage_role", "unknown") for m in meta], dtype=object),\n        cultural_motifs=np.array([m.get("cultural_motif", "unknown") for m in meta], dtype=object),\n        prop_proxy_labels=np.array([m.get("prop_proxy_label", "unknown") for m in meta], dtype=object),\n        locomotion_labels=np.array([m.get("locomotion_label", "unknown") for m in meta], dtype=object),\n        support_labels=np.array([m.get("support_label", "unknown") for m in meta], dtype=object),\n        event_position_mid=np.array([float(m.get("event_position_mid", 0.5)) for m in meta], dtype=np.float32),\n        semantic_confidence=np.array([float(m.get("semantic_confidence", 0.5)) for m in meta], dtype=np.float32),\n        natural_duration_min=np.array([float((m.get("natural_duration_range_sec", [1.5, 4.0]) or [1.5, 4.0])[0]) for m in meta], dtype=np.float32),\n        natural_duration_max=np.array([float((m.get("natural_duration_range_sec", [1.5, 4.0]) or [1.5, 4.0])[-1]) for m in meta], dtype=np.float32),\n        take_ids=np.array([int(m.get("take_id", -1) if m.get("take_id", -1) is not None else -1) for m in meta], dtype=np.int32),\n'
    if old_arrays in text:
        text = text.replace(old_arrays, new_arrays, 1)
    elif 'event_families=np.array' not in text:
        raise RuntimeError("Cannot locate np.savez semantic arrays block for V46.31.")



    # V46.31: add event quality score even when V46.31 arrays already exist.
    if 'event_quality_scores=np.array' not in text:
        q_anchor = '        semantic_confidence=np.array([float(m.get("semantic_confidence", 0.5)) for m in meta], dtype=np.float32),\n'
        if q_anchor in text:
            text = text.replace(q_anchor, q_anchor + '        event_quality_scores=np.array([float(m.get("event_quality_score", 0.5)) for m in meta], dtype=np.float32),\n', 1)
        else:
            raise RuntimeError("Cannot locate semantic_confidence array anchor for V46.31 event_quality_scores.")


    text = replace_between(
        text,
        "def semantic_label_match_bonus(slot: dict, db: dict, cfg: V46Config) -> np.ndarray:\n",
        "def parse_change_bvh_semantics(path: str | Path) -> Dict[str, object]:\n",
        NEW_SEMANTIC_LABEL_MATCH_BONUS,
        "semantic_label_match_bonus",
    )

    text = replace_between(
        text,
        "def retrieve_schedule(slots: List[dict], slot_feat: np.ndarray, db: dict, cfg: V46Config, contrastive=None) -> Tuple[List[int], List[dict]]:\n",
        "def align_next_to_prev(prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:\n",
        NEW_RETRIEVE_SCHEDULE,
        "retrieve_schedule",
    )

    text = replace_between(
        text,
        "def matrix_to_rot6d_np(mat: np.ndarray) -> np.ndarray:\n",
        "def fk_24_np(motion: np.ndarray) -> np.ndarray:\n",
        NEW_MATRIX_TO_ROT6D,
        "matrix_to_rot6d_np",
    )

    text = replace_between(
        text,
        "def resample_motion_to_config_fps(motion: np.ndarray, cfg: V46Config) -> Tuple[np.ndarray, dict]:\n",
        "def read_manifest_records(manifest_path: Optional[str], motion_dirs: Sequence[str], cfg: V46Config) -> List[dict]:\n",
        NEW_RESAMPLE,
        "resample_motion_to_config_fps",
    )

    # V46.31: use boundary-aware event starts inside each source split and pass local source position to semantic annotator.
    old_start_block = """            if rec.get("manifest_id") is not None and not bool(getattr(cfg, "manifest_secondary_event_split", True)):
                starts = [0]
                win = min(T, cfg.max_event_frames)
            else:
                starts = [0] if T <= cfg.max_event_frames else list(range(0, max(1, T - cfg.min_event_frames + 1), cfg.hop_len))
                win = cfg.window_len
"""
    new_start_block = """            if rec.get("manifest_id") is not None and not bool(getattr(cfg, "manifest_secondary_event_split", True)):
                starts = [0]
                win = min(T, cfg.max_event_frames)
            else:
                win = min(int(cfg.window_len), T)
                if bool(getattr(cfg, "chang_e_boundary_event_split", True)):
                    starts = chang_e_semantic_event_starts(seq, cfg)
                else:
                    starts = [0] if T <= cfg.max_event_frames else list(range(0, max(1, T - cfg.min_event_frames + 1), cfg.hop_len))
"""
    if old_start_block in text:
        text = text.replace(old_start_block, new_start_block, 1)
    elif "chang_e_semantic_event_starts(seq, cfg)" not in text:
        raise RuntimeError("Cannot locate build_db start generation block for V46.31 boundary-aware slicing.")

    old_meta_update = """                base_meta = dict(rec)
                base_meta.update({"seq_id": seq_id, "resample_report": res_report, "input_mode": input_report.get("input_mode")})
"""
    new_meta_update = """                base_meta = dict(rec)
                event_mid = (float(st) + 0.5 * float(endf - st)) / max(float(T), 1.0)
                base_meta.update({
                    "seq_id": seq_id,
                    "resample_report": res_report,
                    "input_mode": input_report.get("input_mode"),
                    "event_start": int(st),
                    "event_end": int(endf),
                    "event_source_frames": int(T),
                    "event_position_mid": float(event_mid),
                    "event_position_fraction": float(event_mid),
                })
"""
    if old_meta_update in text:
        text = text.replace(old_meta_update, new_meta_update, 1)
    elif "event_position_mid" not in text[text.find("def build_db"): text.find("if not meta:", text.find("def build_db"))]:
        raise RuntimeError("Cannot locate build_db base_meta update block for V46.31 local event position.")

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
