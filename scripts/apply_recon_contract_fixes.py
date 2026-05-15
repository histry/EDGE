from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak_recon_contract")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"✅ backup: {path} -> {bak}")


def write(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    print(f"✅ wrote: {path}")


def patch_edge_experiment_guard():
    p = ROOT / "edge_experiment_guard.py"
    backup(p)
    s = p.read_text(encoding="utf-8")

    if '"nextgen": ("edge_nextgen_runtime_patch", "install_nextgen_runtime_patches")' not in s:
        old = '    "render_contact": ("render_contact_fix_patch", "install_render_contact_fix_patch"),\n}'
        new = (
            '    "render_contact": ("render_contact_fix_patch", "install_render_contact_fix_patch"),\n'
            '    "nextgen": ("edge_nextgen_runtime_patch", "install_nextgen_runtime_patches"),\n'
            '}'
        )
        if old not in s:
            raise RuntimeError("Cannot patch edge_experiment_guard.py PATCH_SPECS.")
        s = s.replace(old, new)

    write(p, s)


def patch_train_py():
    p = ROOT / "train.py"
    backup(p)
    s = p.read_text(encoding="utf-8")

    if '("edge_recon_contract_patch", "install_recon_contract_patch")' not in s:
        anchor = '        ("text_bridge_planner_patch", "install_text_bridge_planner_patch"),\n'
        if anchor not in s:
            raise RuntimeError("Cannot patch train.py patch_specs.")
        s = s.replace(
            anchor,
            anchor + '        ("edge_recon_contract_patch", "install_recon_contract_patch"),\n',
        )

    write(p, s)


def patch_generate_controlled():
    p = ROOT / "generate_controlled.py"
    backup(p)
    s = p.read_text(encoding="utf-8")

    # 1. install nextgen + recon patches before EDGE import
    if '("edge_nextgen_runtime_patch", "install_nextgen_runtime_patches")' not in s:
        anchor = '        ("text_bridge_planner_patch", "install_text_bridge_planner_patch"),\n'
        if anchor not in s:
            raise RuntimeError("Cannot patch generate_controlled.py patch specs.")
        s = s.replace(
            anchor,
            anchor
            + '        ("edge_nextgen_runtime_patch", "install_nextgen_runtime_patches"),\n'
            + '        ("edge_recon_contract_patch", "install_recon_contract_patch"),\n',
        )
    elif '("edge_recon_contract_patch", "install_recon_contract_patch")' not in s:
        anchor = '        ("edge_nextgen_runtime_patch", "install_nextgen_runtime_patches"),\n'
        s = s.replace(
            anchor,
            anchor + '        ("edge_recon_contract_patch", "install_recon_contract_patch"),\n',
        )

    # 2. replace make_keyframe_feature_mask
    new_mask_fn = '''def make_keyframe_feature_mask(
    constrain_contacts: bool,
    constrain_root_xz: bool = False,
) -> np.ndarray:
    """Build feature-wise keyframe mask.

    Default:
      - optional contacts
      - root Y
      - all rotations
      - NOT root X/Z, because trajectory usually owns root path

    For strict stationary reconstruction, use:
      --keyframe_constrain_root_xz
      or EDGE_KEYFRAME_CONSTRAIN_ROOT_XZ=1
    """
    mask_one_frame = np.zeros((151,), dtype=np.float32)

    if constrain_contacts:
        mask_one_frame[CONTACT_SLICE] = 1.0

    if constrain_root_xz or _truthy_env("EDGE_KEYFRAME_CONSTRAIN_ROOT_XZ", "0"):
        mask_one_frame[ROOT_X_IDX] = 1.0
        mask_one_frame[ROOT_Z_IDX] = 1.0

    mask_one_frame[ROOT_Y_IDX] = 1.0
    mask_one_frame[ROT_SLICE] = 1.0
    return mask_one_frame
'''

    pattern = r"def make_keyframe_feature_mask\(constrain_contacts: bool\) -> np\.ndarray:\n.*?\n\n\ndef build_constraint"
    if re.search(pattern, s, flags=re.S):
        s = re.sub(pattern, new_mask_fn + "\n\ndef build_constraint", s, flags=re.S)
    elif "def make_keyframe_feature_mask(" in s and "constrain_root_xz" not in s:
        raise RuntimeError("make_keyframe_feature_mask found but regex did not match.")

    # 3. build_constraint calls new mask helper
    old = '''    frame_mask = make_keyframe_feature_mask(
        constrain_contacts=args.constrain_contacts,
    )
'''
    new = '''    frame_mask = make_keyframe_feature_mask(
        constrain_contacts=args.constrain_contacts,
        constrain_root_xz=bool(getattr(args, "keyframe_constrain_root_xz", False)),
    )
'''
    if old in s:
        s = s.replace(old, new)

    # 4. add CLI args after --constrain_contacts
    if "--hard_keyframe_project" not in s:
        anchor = '    parser.add_argument("--constrain_contacts", action="store_true")\n'
        if anchor not in s:
            raise RuntimeError("Cannot insert parser args after --constrain_contacts.")
        s = s.replace(
            anchor,
            anchor
            + '''    parser.add_argument(
        "--hard_keyframe_project",
        action="store_true",
        help="Strictly project known keyframes during DDPM/DDIM inference.",
    )
    parser.add_argument(
        "--infer_project_xstart",
        action="store_true",
        help="Project predicted clean x_start to known keyframes at every denoising step.",
    )
    parser.add_argument(
        "--keyframe_constrain_root_xz",
        action="store_true",
        help="Also constrain root X/Z from keyframes; use for strict stationary reconstruction.",
    )
    parser.add_argument(
        "--disable_traj_cond",
        action="store_true",
        help="Do not pass trajectory condition to the model during inference.",
    )
    parser.add_argument(
        "--save_normalized_motion",
        action="store_true",
        help="Also save sampled normalized motion as *_norm.npy for diagnostics.",
    )
''',
        )

    # 5. insert final normalized projection helper before unnormalize_motion
    helper = '''def project_clean_motion_numpy(motion_norm: np.ndarray, constraint: dict) -> np.ndarray:
    """Apply final clean-space keyframe projection to normalized [T,151] motion."""
    if constraint is None:
        return motion_norm.astype(np.float32)

    mask = constraint.get("mask", None)
    value = constraint.get("value", None)
    if mask is None or value is None:
        return motion_norm.astype(np.float32)

    mask_np = to_numpy(mask)[0].astype(np.float32)
    value_np = to_numpy(value)[0].astype(np.float32)

    if mask_np.shape[-1] == 1:
        mask_np = np.repeat(mask_np, motion_norm.shape[-1], axis=-1)
    elif mask_np.shape[-1] != motion_norm.shape[-1]:
        raise ValueError(
            f"constraint mask last dim must be 1 or {motion_norm.shape[-1]}, got {mask_np.shape[-1]}"
        )

    return (motion_norm * (1.0 - mask_np) + value_np * mask_np).astype(np.float32)


'''
    if "def project_clean_motion_numpy(" not in s:
        anchor = "def unnormalize_motion(normalizer, motion_norm: np.ndarray) -> np.ndarray:\n"
        if anchor not in s:
            raise RuntimeError("Cannot insert project_clean_motion_numpy.")
        s = s.replace(anchor, helper + anchor)

    # 6. enable hard projection after normalizer
    anchor = "    normalizer = model.normalizer\n"
    block = '''    normalizer = model.normalizer

    if bool(getattr(args, "hard_keyframe_project", False)) or _truthy_env("EDGE_HARD_KEYFRAME_PROJECT", "0"):
        os.environ["EDGE_HARD_KEYFRAME_PROJECT"] = "1"
        os.environ["EDGE_INFER_PROJECT_XSTART"] = "1"
        if hasattr(model, "diffusion"):
            model.diffusion.hard_keyframe_project = True
        print("✅ Strict hard keyframe projection enabled: x_t + clean x_start.")

    if bool(getattr(args, "infer_project_xstart", False)):
        os.environ["EDGE_INFER_PROJECT_XSTART"] = "1"
        print("✅ Clean x_start projection enabled.")
'''
    if "Strict hard keyframe projection enabled" not in s:
        if anchor not in s:
            raise RuntimeError("Cannot patch hard projection flags after normalizer.")
        s = s.replace(anchor, block)

    # 7. disable trajectory condition if requested
    old_cond = '''    cond = {
        "audio": torch.from_numpy(audio_feature[None]).to(device=device, dtype=torch.float32),
        "trajectory": torch.from_numpy(traj_norm[None]).to(device=device, dtype=torch.float32),
    }
'''
    new_cond = '''    cond = {
        "audio": torch.from_numpy(audio_feature[None]).to(device=device, dtype=torch.float32),
    }

    if not bool(getattr(args, "disable_traj_cond", False)) and not _truthy_env("EDGE_DISABLE_TRAJ_COND", "0"):
        cond["trajectory"] = torch.from_numpy(traj_norm[None]).to(device=device, dtype=torch.float32)
    else:
        print("✅ Trajectory condition disabled for inference.")
'''
    if old_cond in s:
        s = s.replace(old_cond, new_cond)

    # 8. final normalized projection and normalized output saving
    old_motion = '''    motion_norm = to_numpy(sample)[0].astype(np.float32)
    motion_raw_physical = unnormalize_motion(normalizer, motion_norm)
'''
    new_motion = '''    motion_norm = to_numpy(sample)[0].astype(np.float32)

    if bool(getattr(args, "hard_keyframe_project", False)) or _truthy_env("EDGE_FINAL_KEYFRAME_PROJECT", "1"):
        motion_norm = project_clean_motion_numpy(motion_norm, constraint)
        print("✅ Final normalized clean-motion keyframe projection applied.")

    if bool(getattr(args, "save_normalized_motion", False)):
        norm_path = Path(args.out).with_suffix("")
        norm_path = norm_path.parent / f"{norm_path.name}_norm.npy"
        np.save(norm_path, motion_norm.astype(np.float32))
        print(f"✅ normalized motion: {norm_path}")

    motion_raw_physical = unnormalize_motion(normalizer, motion_norm)
'''
    if old_motion in s:
        s = s.replace(old_motion, new_motion)

    write(p, s)


def main():
    patch_edge_experiment_guard()
    patch_train_py()
    patch_generate_controlled()
    print("✅ All reconstruction-contract fixes applied.")


if __name__ == "__main__":
    main()
