"""Robust rendering patch for EDGE skeleton_render.

Drop-in replacement for render_contact_fix_patch.py.

Fixes:
    ⚠️ skeleton_render failed: could not broadcast input array from shape (149,) into shape (149,4)

Root cause:
    vis.skeleton_render expects `poses` to be FK joint positions [T,24,3].
    In the training visualization path, some runs pass 151-D motion vectors
    [T,151] directly. Then this line in vis.py fails:

        feet = poses[:, (7, 8, 10, 11)]      # [T,4], not [T,4,3]
        feetv[:-1] = norm(..., axis=-1)      # [T-1] cannot broadcast to [T-1,4]

This patch:
    1) detects [T,151] or [1,T,151] motion inputs;
    2) converts them to SMPL joint positions [T,24,3] before calling the original renderer;
    3) normalizes contact to [T,4];
    4) patches both vis.skeleton_render and model.diffusion.skeleton_render.

It is fail-soft: if FK conversion fails, it falls back to the original call.
"""

from __future__ import annotations

import sys
from typing import Any, Optional, Tuple

import numpy as np


ROOT_SLICE = slice(4, 7)
CONTACT_SLICE = slice(0, 4)
ROT_SLICE = slice(7, 151)


def _to_numpy(x: Any) -> np.ndarray:
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().float().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _as_single_sequence(arr: np.ndarray) -> np.ndarray:
    """Convert [B,T,C] / [1,T,C] to [T,C] for renderer use."""
    arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[-1] == 151:
        # Render the first sequence in a batch.
        return arr[0]

    if arr.ndim == 4 and arr.shape[-1] == 3:
        # [B,T,J,3] -> first sequence.
        return arr[0]

    return arr


def _extract_contact_from_motion151(motion: np.ndarray) -> Optional[np.ndarray]:
    motion = _as_single_sequence(np.asarray(motion))
    if motion.ndim == 2 and motion.shape[-1] == 151:
        return motion[:, CONTACT_SLICE].astype(np.float32, copy=False)
    return None


def _motion151_to_joints(motion: np.ndarray) -> Optional[np.ndarray]:
    """Convert a 151-D EDGE/SMPL motion sequence to FK joints [T,24,3]."""
    motion = _as_single_sequence(np.asarray(motion, dtype=np.float32))

    if motion.ndim != 2 or motion.shape[-1] != 151:
        return None

    try:
        import torch
        from dataset.quaternion import ax_from_6v
        from vis import SMPLSkeleton

        device = torch.device("cpu")
        motion_t = torch.from_numpy(motion).float().unsqueeze(0).to(device)

        pos = motion_t[:, :, ROOT_SLICE]
        rot6d = motion_t[:, :, ROT_SLICE].reshape(1, motion_t.shape[1], 24, 6)

        with torch.no_grad():
            q = ax_from_6v(rot6d)
            smpl = SMPLSkeleton(device)
            joints = smpl.forward(q, pos)

        return joints[0].detach().cpu().numpy().astype(np.float32)

    except Exception as exc:
        print(f"⚠️ render FK conversion from 151D motion failed; fallback to original poses: {exc}")
        return None


def _coerce_poses_for_render(poses: Any) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Return (render_poses, contact_from_motion, mode)."""
    arr = _to_numpy(poses)
    arr = _as_single_sequence(arr)

    # Already FK joints.
    if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[1] >= 24:
        return arr[:, :24, :].astype(np.float32, copy=False), None, "joints"

    # 151-D motion -> FK joints.
    contact = _extract_contact_from_motion151(arr)
    joints = _motion151_to_joints(arr)
    if joints is not None:
        return joints, contact, "motion151_to_joints"

    return arr, contact, "raw"


def _safe_contact(contact: Any, num_steps: int) -> Optional[np.ndarray]:
    """Normalize renderer contact to [T,4]."""
    if contact is None:
        return None

    arr = _to_numpy(contact)

    if arr.ndim == 0:
        arr = np.full((num_steps, 4), bool(arr), dtype=bool)

    elif arr.ndim == 1:
        # [T] or [T-1] -> [T,4]
        if arr.shape[0] == max(0, num_steps - 1) and arr.shape[0] > 0:
            arr = np.concatenate([arr, arr[-1:]], axis=0)
        elif arr.shape[0] < num_steps and arr.shape[0] > 0:
            pad = np.repeat(arr[-1:], num_steps - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > num_steps:
            arr = arr[:num_steps]

        if arr.shape[0] == 0:
            arr = np.zeros((num_steps,), dtype=bool)

        arr = np.repeat(arr[:, None], 4, axis=1)

    elif arr.ndim == 2:
        # [T,1] -> [T,4], [T,C] -> [T,4]
        if arr.shape[0] == max(0, num_steps - 1) and arr.shape[0] > 0:
            arr = np.concatenate([arr, arr[-1:]], axis=0)
        elif arr.shape[0] < num_steps and arr.shape[0] > 0:
            pad = np.repeat(arr[-1:, :], num_steps - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > num_steps:
            arr = arr[:num_steps]

        if arr.shape[0] == 0:
            arr = np.zeros((num_steps, 4), dtype=bool)

        if arr.shape[1] == 1:
            arr = np.repeat(arr, 4, axis=1)
        elif arr.shape[1] < 4:
            pad = np.repeat(arr[:, -1:], 4 - arr.shape[1], axis=1)
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > 4:
            arr = arr[:, :4]

    else:
        # Any higher-dimensional accidental input: flatten all but time.
        arr = arr.reshape(arr.shape[0], -1)
        return _safe_contact(arr, num_steps)

    if arr.shape[0] == 0:
        arr = np.zeros((num_steps, 4), dtype=bool)

    # Final guard.
    if arr.shape[0] != num_steps:
        fixed = np.zeros((num_steps, 4), dtype=arr.dtype)
        n = min(num_steps, arr.shape[0])
        fixed[:n] = arr[:n, :4]
        if n > 0 and n < num_steps:
            fixed[n:] = fixed[n - 1 : n]
        arr = fixed

    return arr[:, :4]


def install_render_contact_fix_patch(verbose: bool = True) -> bool:
    try:
        import vis
    except Exception as exc:
        if verbose:
            print(f"⚠️ render robust fix skipped: cannot import vis: {exc}")
        return False

    if getattr(vis, "_edge_render_contact_fix_v2_installed", False):
        return True

    # If v1 already wrapped skeleton_render, prefer the true original when available.
    original_skeleton_render = getattr(
        vis,
        "_edge_original_skeleton_render",
        vis.skeleton_render,
    )

    def patched_skeleton_render(
        poses,
        epoch=0,
        out="renders",
        name="",
        sound=True,
        stitch=False,
        sound_folder="ood_sliced",
        contact=None,
        render=True,
        camera_mode="follow",
        output_path=None,
        render_smooth_window=9,
    ):
        render_poses = poses
        motion_contact = None
        mode = "raw"

        try:
            render_poses, motion_contact, mode = _coerce_poses_for_render(poses)
        except Exception as exc:
            if verbose:
                print(f"⚠️ render pose normalization failed; using original poses: {exc}")
            render_poses = poses
            motion_contact = None

        try:
            num_steps = int(_to_numpy(render_poses).shape[0])
            if contact is None and motion_contact is not None:
                contact = motion_contact
            contact = _safe_contact(contact, num_steps)
        except Exception as exc:
            if verbose:
                print(f"⚠️ render contact normalization failed; rendering without contact: {exc}")
            contact = None

        if verbose and mode == "motion151_to_joints" and not getattr(patched_skeleton_render, "_logged_151", False):
            print("✅ render robust fix: converted 151D motion to FK joints [T,24,3] before skeleton_render.")
            patched_skeleton_render._logged_151 = True

        try:
            return original_skeleton_render(
                render_poses,
                epoch=epoch,
                out=out,
                name=name,
                sound=sound,
                stitch=stitch,
                sound_folder=sound_folder,
                contact=contact,
                render=render,
                camera_mode=camera_mode,
                output_path=output_path,
                render_smooth_window=render_smooth_window,
            )
        except ValueError as exc:
            # Last-resort fallback for the exact broadcast issue.
            if "could not broadcast input array" not in str(exc):
                raise

            fallback_poses, fallback_contact, fallback_mode = _coerce_poses_for_render(poses)
            if fallback_mode != "motion151_to_joints":
                raise

            fallback_contact = _safe_contact(fallback_contact if contact is None else contact, fallback_poses.shape[0])
            print("✅ render robust fix recovered from broadcast error by FK-converting 151D motion.")
            return original_skeleton_render(
                fallback_poses,
                epoch=epoch,
                out=out,
                name=name,
                sound=sound,
                stitch=stitch,
                sound_folder=sound_folder,
                contact=fallback_contact,
                render=render,
                camera_mode=camera_mode,
                output_path=output_path,
                render_smooth_window=render_smooth_window,
            )

    vis.skeleton_render = patched_skeleton_render
    vis._edge_original_skeleton_render = original_skeleton_render
    vis._edge_render_contact_fix_installed = True
    vis._edge_render_contact_fix_v2_installed = True

    # model.diffusion imports skeleton_render directly, so patch that bound symbol too.
    try:
        import model.diffusion as diffusion_module
        diffusion_module.skeleton_render = patched_skeleton_render
    except Exception as exc:
        if verbose:
            print(f"⚠️ model.diffusion.skeleton_render not patched: {exc}")

    # Patch any already-imported module that cached the old function under this name.
    try:
        for module in list(sys.modules.values()):
            if module is None or module is vis:
                continue
            if getattr(module, "skeleton_render", None) is original_skeleton_render:
                setattr(module, "skeleton_render", patched_skeleton_render)
    except Exception:
        pass

    if verbose:
        print("✅ Installed render robust fix v2: 151D motion -> FK joints + safe [T,4] contact.")
    return True


def install():
    return install_render_contact_fix_patch(verbose=True)
