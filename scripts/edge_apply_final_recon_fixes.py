from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def backup(path: Path, suffix: str):
    bak = path.with_suffix(path.suffix + suffix)
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"✅ backup: {path} -> {bak}")


def write(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    print(f"✅ wrote: {path}")


def patch_train_py():
    p = ROOT / "train.py"
    backup(p, ".bak_final_recon")
    s = p.read_text(encoding="utf-8")

    if '("edge_recon_contract_patch", "install_recon_contract_patch")' not in s:
        anchor = '        ("text_bridge_planner_patch", "install_text_bridge_planner_patch"),\n'
        if anchor not in s:
            raise RuntimeError("Cannot find text_bridge_planner_patch anchor in train.py")
        s = s.replace(
            anchor,
            anchor + '        ("edge_recon_contract_patch", "install_recon_contract_patch"),\n',
        )

    write(p, s)


def patch_generate_controlled_py():
    p = ROOT / "generate_controlled.py"
    backup(p, ".bak_final_recon")
    s = p.read_text(encoding="utf-8")

    # Runtime patch install list.
    if '("edge_recon_contract_patch", "install_recon_contract_patch")' not in s:
        anchor = '        ("text_bridge_planner_patch", "install_text_bridge_planner_patch"),\n'
        if anchor not in s:
            raise RuntimeError("Cannot find text_bridge_planner_patch anchor in generate_controlled.py")
        s = s.replace(
            anchor,
            anchor + '        ("edge_recon_contract_patch", "install_recon_contract_patch"),\n',
        )

    # CLI args.
    if '--hard_keyframe_project' not in s:
        anchor = '    parser.add_argument("--constrain_contacts", action="store_true")\n'
        if anchor not in s:
            raise RuntimeError("Cannot find --constrain_contacts parser anchor")
        s = s.replace(
            anchor,
            '''    parser.add_argument("--constrain_contacts", action="store_true")
    parser.add_argument("--hard_keyframe_project", action="store_true")
    parser.add_argument("--infer_project_xstart", action="store_true")
    parser.add_argument("--keyframe_constrain_root_xz", action="store_true")
    parser.add_argument("--disable_traj_cond", action="store_true")
    parser.add_argument("--save_normalized_motion", action="store_true")
    parser.add_argument("--no_ema", action="store_true")
''',
        )

    if '--no_ema' not in s:
        anchor = '    parser.add_argument("--save_normalized_motion", action="store_true")\n'
        if anchor not in s:
            anchor = '    parser.add_argument("--constrain_contacts", action="store_true")\n'
        s = s.replace(anchor, anchor + '    parser.add_argument("--no_ema", action="store_true")\n')

    # EMA switch.
    old = '''    model = EDGE(
        feature_type=args.feature_type,
        checkpoint_path=args.checkpoint,
        EMA=True,
        audio_dim=args.audio_dim,
        seq_len=num_frames,
        mixed_precision=args.mixed_precision,
    )
'''
    new = '''    model = EDGE(
        feature_type=args.feature_type,
        checkpoint_path=args.checkpoint,
        EMA=not bool(getattr(args, "no_ema", False)),
        audio_dim=args.audio_dim,
        seq_len=num_frames,
        mixed_precision=args.mixed_precision,
    )
'''
    if old in s:
        s = s.replace(old, new)

    # Keyframe mask root X/Z option.
    old = '''def make_keyframe_feature_mask(constrain_contacts: bool) -> np.ndarray:
    mask_one_frame = np.zeros((151,), dtype=np.float32)

    if constrain_contacts:
        mask_one_frame[CONTACT_SLICE] = 1.0

    mask_one_frame[ROOT_Y_IDX] = 1.0
    mask_one_frame[ROT_SLICE] = 1.0
    return mask_one_frame
'''
    new = '''def make_keyframe_feature_mask(constrain_contacts: bool, constrain_root_xz: bool = False) -> np.ndarray:
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
    if old in s:
        s = s.replace(old, new)

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

    # Hard projection flags.
    old = '''    normalizer = model.normalizer

    traj_physical, traj_norm = build_control_trajectory(
'''
    new = '''    normalizer = model.normalizer

    if bool(getattr(args, "hard_keyframe_project", False)) or _truthy_env("EDGE_HARD_KEYFRAME_PROJECT", "0"):
        os.environ["EDGE_HARD_KEYFRAME_PROJECT"] = "1"
        os.environ["EDGE_INFER_PROJECT_XSTART"] = "1"
        if hasattr(model, "diffusion"):
            model.diffusion.hard_keyframe_project = True
        print("✅ Strict hard keyframe projection enabled.")

    if bool(getattr(args, "infer_project_xstart", False)):
        os.environ["EDGE_INFER_PROJECT_XSTART"] = "1"
        print("✅ Clean x_start projection enabled.")

    traj_physical, traj_norm = build_control_trajectory(
'''
    if old in s and "Strict hard keyframe projection enabled" not in s:
        s = s.replace(old, new)

    # Disable trajectory condition.
    old = '''    cond = {
        "audio": torch.from_numpy(audio_feature[None]).to(device=device, dtype=torch.float32),
        "trajectory": torch.from_numpy(traj_norm[None]).to(device=device, dtype=torch.float32),
    }
'''
    new = '''    cond = {
        "audio": torch.from_numpy(audio_feature[None]).to(device=device, dtype=torch.float32),
    }

    if not bool(getattr(args, "disable_traj_cond", False)) and not _truthy_env("EDGE_DISABLE_TRAJ_COND", "0"):
        cond["trajectory"] = torch.from_numpy(traj_norm[None]).to(device=device, dtype=torch.float32)
    else:
        print("✅ Trajectory condition disabled for inference.")
'''
    if old in s:
        s = s.replace(old, new)

    # Helper for final clean projection.
    helper = '''def project_clean_motion_numpy(motion_norm: np.ndarray, constraint: dict) -> np.ndarray:
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
        raise ValueError(f"constraint mask last dim mismatch: {mask_np.shape[-1]}")

    return (motion_norm * (1.0 - mask_np) + value_np * mask_np).astype(np.float32)


'''
    if "def project_clean_motion_numpy(" not in s:
        anchor = 'def unnormalize_motion(normalizer, motion_norm: np.ndarray) -> np.ndarray:\n'
        if anchor not in s:
            raise RuntimeError("Cannot find unnormalize_motion anchor")
        s = s.replace(anchor, helper + anchor)

    # Final projection + save normalized motion.
    old = '''    motion_norm = to_numpy(sample)[0].astype(np.float32)
    motion_raw_physical = unnormalize_motion(normalizer, motion_norm)
'''
    new = '''    motion_norm = to_numpy(sample)[0].astype(np.float32)

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
    if old in s:
        s = s.replace(old, new)

    write(p, s)


def patch_diffusion_py():
    p = ROOT / "model" / "diffusion.py"
    backup(p, ".bak_x0_recon_final")
    s = p.read_text(encoding="utf-8")

    # import os
    first_lines = "\n".join(s.splitlines()[:60])
    if "import os" not in first_lines:
        if "import math" in s:
            s = s.replace("import math", "import os\nimport math", 1)
        else:
            s = "import os\n" + s

    # Insert x0 recon loss.
    if "EDGE_X0_RECON_LOSS" not in s:
        needle = '''        # Main temporal smoothness loss.
        velocity_loss = self._velocity_loss(
            model_motion_x0,
            target_motion_x0,
            t,
        )
'''
        insert = '''        # Optional direct x0 reconstruction loss for strict reconstruction / single-unit overfit.
        # Normal EDGE training optimizes epsilon prediction. For sanity reconstruction we also
        # need the predicted clean motion x0 to match the training motion directly.
        x0_recon_loss = model_motion_x0.new_tensor(0.0)
        if os.environ.get("EDGE_X0_RECON_LOSS", "0").lower() in {"1", "true", "yes", "on"}:
            x0_recon_raw = F.mse_loss(model_motion_x0, target_motion_x0)
            x0_recon_loss = float(os.environ.get("EDGE_X0_RECON_LOSS_WEIGHT", "1.0")) * x0_recon_raw

        # Main temporal smoothness loss.
        velocity_loss = self._velocity_loss(
            model_motion_x0,
            target_motion_x0,
            t,
        )
'''
        if needle not in s:
            raise RuntimeError("Cannot find velocity_loss insertion point in diffusion.py")
        s = s.replace(needle, insert)

    # Add x0 loss into total_loss.
    old = '''        total_loss = sum(losses)
'''
    new = '''        total_loss = sum(losses) + x0_recon_loss
'''
    if old in s and "total_loss = sum(losses) + x0_recon_loss" not in s:
        s = s.replace(old, new)

    write(p, s)


def main():
    patch_train_py()
    patch_generate_controlled_py()
    patch_diffusion_py()
    print("✅ final reconstruction fixes applied.")


if __name__ == "__main__":
    main()
