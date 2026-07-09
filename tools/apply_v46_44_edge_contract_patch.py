#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch V46.43 local code for EDGE-151D representation consistency.

Fixes:
1) matrix_to_rot6d_torch uses the same column-concatenated convention as matrix_to_rot6d_np.
2) load_bvh_file separates root trajectory scaling from OFFSET-based skeleton scaling,
   avoiding double scaling when root motion has already been converted to meters.
3) render_from_npy.py decodes V46 column-concatenated Rot6D with V46's own rot6d_to_matrix_np
   instead of dataset.quaternion.ax_from_6v / PyTorch3D rotation_6d_to_matrix convention.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path.cwd()
V46 = ROOT / "tools" / "v46_motionrag_diff.py"
RENDER = ROOT / "render_from_npy.py"


def backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".v46_44_contract_backup")
    if not bak.exists():
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        print("[BACKUP]", bak)


def patch_v46() -> None:
    if not V46.exists():
        raise FileNotFoundError(V46)
    backup(V46)
    s = V46.read_text(encoding="utf-8")

    old = '''def matrix_to_rot6d_torch(mat):\n    return mat[..., :, 0:2].reshape(*mat.shape[:-2], 6)\n'''
    new = '''def matrix_to_rot6d_torch(mat):\n    """Convert rotation matrices to V46/EDGE column-concatenated 6D.\n\n    Must match matrix_to_rot6d_np(): [R[:,0], R[:,1]].  The old\n    mat[..., :, 0:2].reshape(...) interleaves rows and corrupts identity\n    rotations as [1,0,0,1,0,0] instead of [1,0,0,0,1,0].\n    """\n    c0 = mat[..., :, 0]\n    c1 = mat[..., :, 1]\n    return torch.cat([c0, c1], dim=-1)\n'''
    if old in s:
        s = s.replace(old, new)
        print("[PATCH] matrix_to_rot6d_torch column-concat")
    elif "def matrix_to_rot6d_torch(mat):" in s and "torch.cat([c0, c1]" in s:
        print("[SKIP] matrix_to_rot6d_torch already patched")
    else:
        raise RuntimeError("Could not patch matrix_to_rot6d_torch; unexpected source")

    # Patch the auto-scale block in load_bvh_file.  The repository version derives auto_scale
    # from BVH offsets and applies it to root_xyz.  That double-scales canonicalized BVH whose
    # root channels are already meters but offsets were left in cm.  This replacement uses only
    # root absolute/travel scale for root_scale, and keeps offset_scale diagnostic separate.
    old_block = '''    offsets = np.stack([j["offset"] for j in joints]).astype(np.float32)\n    bone_lens = np.linalg.norm(offsets, axis=1)\n    nonzero = bone_lens[bone_lens > 1e-6]\n    # BVH is often centimeters; meters-scale skeletons have bone lengths < 2.\n    auto_scale = 0.01 if (nonzero.size and float(np.percentile(nonzero, 90)) > 2.0) else 1.0\n    # If root trajectory itself is huge, also treat it as centimeters.\n    root_j = 0\n    root_ch = joints[root_j]["channels"]\n    root_st = int(joints[root_j]["channel_start"])\n    pos_cols = {ch.lower(): root_st + k for k, ch in enumerate(root_ch) if ch.lower().endswith("position")}\n    root_xyz = np.zeros((data.shape[0], 3), dtype=np.float32)\n    for axis, out_i in [("xposition", 0), ("yposition", 1), ("zposition", 2)]:\n        if axis in pos_cols:\n            root_xyz[:, out_i] = data[:, pos_cols[axis]]\n    if np.nanpercentile(np.linalg.norm(root_xyz[:, [0, 2]] - root_xyz[:1, [0, 2]], axis=1), 95) > 20.0:\n        auto_scale = 0.01\n    root_xyz *= float(auto_scale)\n'''
    new_block = '''    offsets = np.stack([j["offset"] for j in joints]).astype(np.float32)\n    bone_lens = np.linalg.norm(offsets, axis=1)\n    nonzero = bone_lens[bone_lens > 1e-6]\n    # V46.44 contract fix:\n    # Offset scale and root trajectory scale must be decoupled.  A canonicalized\n    # BVH may already have meter-scale root channels while legacy hierarchy\n    # offsets remain centimeter-scale; using offsets to scale root again shrinks\n    # the trajectory by 100x and causes moonwalk/static-root artifacts.\n    offset_scale_hint = 0.01 if (nonzero.size and float(np.percentile(nonzero, 90)) > 2.0) else 1.0\n    root_j = 0\n    root_ch = joints[root_j]["channels"]\n    root_st = int(joints[root_j]["channel_start"])\n    pos_cols = {ch.lower(): root_st + k for k, ch in enumerate(root_ch) if ch.lower().endswith("position")}\n    root_xyz = np.zeros((data.shape[0], 3), dtype=np.float32)\n    for axis, out_i in [("xposition", 0), ("yposition", 1), ("zposition", 2)]:\n        if axis in pos_cols:\n            root_xyz[:, out_i] = data[:, pos_cols[axis]]\n    root_abs_p95 = float(np.nanpercentile(np.linalg.norm(root_xyz, axis=1), 95)) if root_xyz.size else 0.0\n    root_xz_travel_p95 = float(np.nanpercentile(np.linalg.norm(root_xyz[:, [0, 2]] - root_xyz[:1, [0, 2]], axis=1), 95)) if root_xyz.size else 0.0\n    scale_mode = str(os.environ.get("V46_BVH_ROOT_SCALE_MODE", "auto")).strip().lower()\n    if scale_mode in {"none", "meter", "meters", "1", "1.0"}:\n        root_scale = 1.0\n    elif scale_mode in {"cm", "centimeter", "centimeters", "0.01"}:\n        root_scale = 0.01\n    else:\n        # Original Chang-E cm data has root_abs/travel in tens/hundreds.\n        # Canonicalized meter data has root_abs/travel around 0.5-2.5.\n        root_scale = 0.01 if (root_abs_p95 > 20.0 or root_xz_travel_p95 > 20.0) else 1.0\n    root_xyz *= float(root_scale)\n'''
    if old_block in s:
        s = s.replace(old_block, new_block)
        print("[PATCH] load_bvh_file decoupled root scale from OFFSET scale")
    elif "V46.44 contract fix" in s:
        print("[SKIP] load_bvh_file already patched")
    else:
        print("[WARN] load_bvh_file scale block not patched; maybe source already diverged")

    # Store root_scale instead of obsolete auto_scale metadata.
    s = s.replace('out[:, 1] = float(auto_scale)', 'out[:, 1] = float(root_scale)')
    s = s.replace('out[:, 1] = float(offset_scale_hint)', 'out[:, 1] = float(root_scale)')

    V46.write_text(s, encoding="utf-8")


def patch_render() -> None:
    if not RENDER.exists():
        raise FileNotFoundError(RENDER)
    backup(RENDER)
    s = RENDER.read_text(encoding="utf-8")

    s = s.replace('from dataset.quaternion import ax_from_6v\n', '')
    if 'from pytorch3d.transforms import matrix_to_axis_angle' not in s:
        s = s.replace('import torch\n', 'import torch\nfrom pytorch3d.transforms import matrix_to_axis_angle\n')
    if 'from tools.v46_motionrag_diff import rot6d_to_matrix_np' not in s:
        s = s.replace('from vis import skeleton_render, SMPLSkeleton\n', 'from vis import skeleton_render, SMPLSkeleton\nfrom tools.v46_motionrag_diff import rot6d_to_matrix_np\n')

    old = '''    q_6d = motion_tensor[:, :, 7:].reshape(1, seq_len, 24, 6)\n    q_ax = ax_from_6v(q_6d)\n    smpl = SMPLSkeleton(device=device)\n    poses_3d = smpl.forward(q_ax, pos).detach().cpu().numpy()[0]\n'''
    new = '''    # V46.44 contract fix: V46 stores Rot6D as column-concatenated\n    # [R[:,0], R[:,1]].  Do not decode with dataset.quaternion.ax_from_6v,\n    # whose backend may assume a different flattening convention.\n    q_6d_np = motion_data[:, 7:].reshape(seq_len, 24, 6).astype(np.float32)\n    rot_m_np = rot6d_to_matrix_np(q_6d_np)\n    rot_m = torch.tensor(rot_m_np, dtype=torch.float32, device=device).unsqueeze(0)\n    q_ax = matrix_to_axis_angle(rot_m)\n    smpl = SMPLSkeleton(device=device)\n    poses_3d = smpl.forward(q_ax, pos).detach().cpu().numpy()[0]\n'''
    if old in s:
        s = s.replace(old, new)
        print("[PATCH] render_from_npy V46 Rot6D decoder")
    elif "V46.44 contract fix" in s:
        print("[SKIP] render_from_npy already patched")
    else:
        raise RuntimeError("Could not patch render_from_npy decode block")
    RENDER.write_text(s, encoding="utf-8")


def main() -> None:
    patch_v46()
    patch_render()
    print("[DONE] V46.44 EDGE contract patch applied")


if __name__ == "__main__":
    main()
