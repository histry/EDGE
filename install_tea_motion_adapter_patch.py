#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path.cwd()

def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")

def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")

def backup(path: str) -> None:
    p = ROOT / path
    b = p.with_suffix(p.suffix + ".tea_bak")
    if not b.exists():
        b.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"backup: {b}")

def insert_after(text: str, marker: str, insertion: str, tag: str) -> str:
    if tag in text:
        return text
    idx = text.find(marker)
    if idx < 0:
        raise RuntimeError(f"marker not found for insert_after: {marker[:160]}")
    idx += len(marker)
    return text[:idx] + insertion + text[idx:]

def insert_before(text: str, marker: str, insertion: str, tag: str) -> str:
    if tag in text:
        return text
    idx = text.find(marker)
    if idx < 0:
        raise RuntimeError(f"marker not found for insert_before: {marker[:160]}")
    return text[:idx] + insertion + text[idx:]

def replace_once(text: str, old: str, new: str, tag: str | None = None) -> str:
    if tag and tag in text:
        return text
    if old not in text:
        raise RuntimeError(f"old block not found: {old[:180]}")
    return text.replace(old, new, 1)

def regex_replace_once(text: str, pattern: str, repl: str, tag: str | None = None, flags: int = re.S) -> str:
    if tag and tag in text:
        return text
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        raise RuntimeError(f"regex replacement failed: {pattern[:180]}")
    return new

def patch_args() -> None:
    path = "args.py"
    backup(path)
    text = read(path)
    text = text.replace('choices=["full", "stage1", "stage2"]', 'choices=["full", "stage1", "stage2", "adapter"]')
    insertion = '''
    # ===== TEA-MotionAdapter: energy condition + adapter training =====
    parser.add_argument(
        "--energy_condition_prob",
        type=float,
        default=0.7,
        help="Probability of providing normalized motion energy as a conditioning scalar during training.",
    )
    parser.add_argument(
        "--energy_condition_drop_prob",
        type=float,
        default=0.15,
        help="Drop probability for energy condition only; enables energy-conditioned CFG.",
    )
    parser.add_argument(
        "--energy_loss_weight",
        type=float,
        default=0.25,
        help="Weight for matching generated and target motion-energy envelopes.",
    )
    parser.add_argument(
        "--root_lower_coupling_loss_weight",
        type=float,
        default=0.5,
        help="Extra multiplier inside kinematic sync loss for root-speed/lower-body coupling.",
    )
    parser.add_argument(
        "--root_lower_speed_threshold",
        type=float,
        default=0.012,
        help="Normalized root XZ speed threshold above which lower-body response is encouraged.",
    )
    parser.add_argument(
        "--root_lower_min_motion",
        type=float,
        default=0.010,
        help="Minimum lower-body rotational motion expected when root XZ speed is high.",
    )
    parser.add_argument(
        "--adapter_train_decoder",
        action="store_true",
        help="With --train_stage adapter, also unfreeze the main seqTransDecoder/final layer.",
    )
'''
    text = insert_before(text, '    parser.add_argument(\n        "--keyframe_loss_weight",', insertion, "TEA-MotionAdapter: energy condition")
    write(path, text)
    print("patched args.py")

def patch_train() -> None:
    path = "train.py"
    backup(path)
    text = read(path)
    insertion = '''        energy_condition_prob=opt.energy_condition_prob,
        energy_condition_drop_prob=opt.energy_condition_drop_prob,
        energy_loss_weight=opt.energy_loss_weight,
        root_lower_coupling_loss_weight=opt.root_lower_coupling_loss_weight,
        root_lower_speed_threshold=opt.root_lower_speed_threshold,
        root_lower_min_motion=opt.root_lower_min_motion,
        adapter_train_decoder=opt.adapter_train_decoder,
'''
    text = insert_after(text, '        trajectory_velocity_loss_weight=opt.trajectory_velocity_loss_weight,\n', insertion, 'energy_condition_prob=opt.energy_condition_prob')
    write(path, text)
    print("patched train.py")

def patch_edge() -> None:
    path = "EDGE.py"
    backup(path)
    text = read(path)
    text = replace_once(
        text,
        '        trajectory_loss_weight=1.0,\n        trajectory_velocity_loss_weight=0.25,\n        hard_keyframe_project=False,\n        train_stage="full",\n',
        '        trajectory_loss_weight=1.0,\n        trajectory_velocity_loss_weight=0.25,\n        energy_condition_prob=0.7,\n        energy_condition_drop_prob=0.15,\n        energy_loss_weight=0.25,\n        root_lower_coupling_loss_weight=0.5,\n        root_lower_speed_threshold=0.012,\n        root_lower_min_motion=0.010,\n        adapter_train_decoder=False,\n        hard_keyframe_project=False,\n        train_stage="full",\n',
        tag="energy_condition_prob=0.7",
    )
    text = replace_once(
        text,
        '        self._apply_stage_freezing(model, train_stage)\n',
        '        self._apply_stage_freezing(model, train_stage, adapter_train_decoder=adapter_train_decoder)\n',
        tag="adapter_train_decoder=adapter_train_decoder",
    )
    text = insert_after(
        text,
        '            trajectory_velocity_loss_weight=trajectory_velocity_loss_weight,\n',
        '''            energy_condition_prob=energy_condition_prob,
            energy_condition_drop_prob=energy_condition_drop_prob,
            energy_loss_weight=energy_loss_weight,
            root_lower_coupling_loss_weight=root_lower_coupling_loss_weight,
            root_lower_speed_threshold=root_lower_speed_threshold,
            root_lower_min_motion=root_lower_min_motion,
''',
        tag="root_lower_coupling_loss_weight=root_lower_coupling_loss_weight",
    )
    text = replace_once(
        text,
        '    def _apply_stage_freezing(self, model, train_stage):\n',
        '    def _apply_stage_freezing(self, model, train_stage, adapter_train_decoder=False):\n',
        tag="adapter_train_decoder=False",
    )
    adapter_block = '''
        if train_stage == "adapter":
            # TEA-MotionAdapter:
            # Freeze the pretrained motion prior and train only lightweight
            # control branches. This protects physical priors while teaching
            # trajectory speed -> lower-body stepping coupling.
            self._set_requires_grad(model, False)

            train_names = [
                "trajectory_projection",
                "trajectory_encoder",
                "traj_modulate",
                "root_generator",
                "energy_embed",
                "null_energy_embed",
            ]

            for name in train_names:
                module = getattr(model, name, None)
                if module is None:
                    continue
                if isinstance(module, torch.nn.Parameter):
                    module.requires_grad = True
                else:
                    self._set_requires_grad(module, True)

            stack = getattr(getattr(model, "seqTransDecoder", None), "stack", [])
            for layer in stack:
                for adapter_name in [
                    "traj_adapter_self",
                    "traj_adapter_cross",
                    "traj_adapter_ff",
                ]:
                    adapter = getattr(layer, adapter_name, None)
                    if adapter is not None:
                        self._set_requires_grad(adapter, True)

            if adapter_train_decoder:
                self._set_requires_grad(model.seqTransDecoder, True)
                self._set_requires_grad(model.final_layer, True)

            print(
                "🧩 train_stage=adapter: training trajectory adapters + "
                "trajectory encoder/root generator + energy embedding; "
                f"adapter_train_decoder={bool(adapter_train_decoder)}."
            )
            return

'''
    text = insert_before(text, '        raise ValueError(f"Unknown train_stage: {train_stage}")\n', adapter_block, 'train_stage == "adapter"')
    text = insert_before(text, '            "Motion Energy Loss",\n', '            "Root-Lower Coupling Loss",\n', tag='"Root-Lower Coupling Loss"')
    write(path, text)
    print("patched EDGE.py")

def patch_dataset() -> None:
    path = "dataset/dance_dataset.py"
    backup(path)
    text = read(path)
    helper = '''

# ===== TEA-MotionAdapter helper =====
def motion_energy_scalar_from_151(motion) -> torch.Tensor:
    """Return a stable normalized energy scalar in [0,1] for one [T,151] motion."""
    if not torch.is_tensor(motion):
        motion = torch.as_tensor(motion, dtype=torch.float32)
    motion = motion.float()
    if motion.ndim != 2 or motion.shape[0] < 2:
        return torch.tensor([0.0], dtype=torch.float32)

    root_xz = motion[:, [4, 6]]
    lower_joints = [1, 2, 4, 5, 7, 8, 10, 11]
    upper_joints = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    def rot_indices(joints):
        idx = []
        for j in joints:
            idx.extend(range(7 + 6 * j, 7 + 6 * (j + 1)))
        return idx

    root_speed = torch.linalg.norm(root_xz[1:] - root_xz[:-1], dim=-1).mean()
    lower_idx = rot_indices(lower_joints)
    upper_idx = rot_indices(upper_joints)
    lower_energy = torch.sqrt(((motion[1:, lower_idx] - motion[:-1, lower_idx]) ** 2).mean() + 1e-8)
    upper_energy = torch.sqrt(((motion[1:, upper_idx] - motion[:-1, upper_idx]) ** 2).mean() + 1e-8)
    contact = motion[:, :4].clamp(0.0, 1.0)
    contact_change = torch.abs(contact[1:] - contact[:-1]).mean()

    raw = 0.35 * root_speed + 0.30 * lower_energy + 0.25 * upper_energy + 0.10 * contact_change
    energy = torch.sigmoid(8.0 * (raw - 0.04))
    return energy.reshape(1).clamp(0.0, 1.0).float()
'''
    text = insert_before(text, "\nclass AISTPPDataset", helper, "motion_energy_scalar_from_151")
    text = insert_after(
        text,
        '        cond = {\n            "audio": feature,\n            "audio_paired": torch.tensor(1.0, dtype=torch.float32),\n            "onset": onset,\n        }\n',
        '        cond["energy"] = motion_energy_scalar_from_151(pose)\n',
        'cond["energy"] = motion_energy_scalar_from_151(pose)',
    )
    if 'cond["energy"] = motion_energy_scalar_from_151(motion)' not in text:
        text, n = re.subn(
            r'(\n\s+return motion,\s*cond,\s*)',
            '\n        cond["energy"] = motion_energy_scalar_from_151(motion)\\1',
            text,
            count=1,
        )
        if n != 1:
            print("warning: could not auto-insert Dunhuang energy condition; check dataset/dance_dataset.py __getitem__")
    write(path, text)
    print("patched dataset/dance_dataset.py")

def patch_model() -> None:
    path = "model/model.py"
    backup(path)
    text = read(path)
    if "import os" not in text.split("\n")[:30]:
        text = text.replace("from typing import Any, Callable, Optional, Union\n", "from typing import Any, Callable, Optional, Union\nimport os\n", 1)

    energy_modules = '''
        # TEA-MotionAdapter:
        # Scalar motion-energy condition. It is added to timestep/global
        # conditioning rather than cross-attention because it is a global
        # continuous control axis.
        self.energy_embed = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.null_energy_embed = nn.Parameter(torch.zeros(1, latent_dim))
'''
    text = insert_after(text, '        self.traj_type_embed = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)\n', energy_modules, "self.energy_embed = nn.Sequential")

    text = replace_once(
        text,
        '''        if isinstance(cond_embed, dict):
            audio_cond = cond_embed.get("audio", None)
            trajectory_cond = cond_embed.get("trajectory", None)
        else:
            audio_cond = cond_embed
            trajectory_cond = None
''',
        '''        if isinstance(cond_embed, dict):
            audio_cond = cond_embed.get("audio", None)
            trajectory_cond = cond_embed.get("trajectory", None)
            energy_cond = cond_embed.get("energy", None)
        else:
            audio_cond = cond_embed
            trajectory_cond = None
            energy_cond = None
''',
        tag='energy_cond = cond_embed.get("energy", None)',
    )

    text = replace_once(
        text,
        "        return audio_cond, trajectory_cond\n",
        '''        if energy_cond is not None:
            energy_cond = energy_cond.to(device=device, dtype=dtype)
            if energy_cond.ndim == 1:
                energy_cond = energy_cond[:, None]
            elif energy_cond.ndim == 3:
                energy_cond = energy_cond[..., :1].mean(dim=1)
            elif energy_cond.ndim == 2 and energy_cond.shape[-1] != 1:
                energy_cond = energy_cond[:, :1]
            energy_cond = energy_cond.clamp(0.0, 1.0)

        return audio_cond, trajectory_cond, energy_cond
''',
        tag="return audio_cond, trajectory_cond, energy_cond",
    )

    guided_repl = '''    def guided_forward(
        self,
        x,
        cond_embed,
        times,
        guidance_weight,
        force_mask=None,
        force_x_clean=None,
    ):
        """Classifier-free guidance with an optional separate energy axis."""
        b = x.shape[0]
        device = x.device
        drop_all = torch.zeros((b,), dtype=torch.bool, device=device)
        keep_all = torch.ones((b,), dtype=torch.bool, device=device)

        unc = self.forward(
            x,
            cond_embed,
            times,
            cond_drop_prob=1.0,
            force_mask=force_mask,
            force_x_clean=force_x_clean,
            keep_audio_mask=drop_all,
            keep_traj_mask=drop_all,
            keep_energy_mask=drop_all,
        )

        try:
            energy_scale = float(os.environ.get("EDGE_ENERGY_CFG_SCALE", "0"))
        except Exception:
            energy_scale = 0.0

        has_energy = isinstance(cond_embed, dict) and cond_embed.get("energy", None) is not None

        if energy_scale > 0.0 and has_energy:
            base = self.forward(
                x,
                cond_embed,
                times,
                cond_drop_prob=0.0,
                force_mask=force_mask,
                force_x_clean=force_x_clean,
                keep_audio_mask=keep_all,
                keep_traj_mask=keep_all,
                keep_energy_mask=drop_all,
            )
            energy_cond = self.forward(
                x,
                cond_embed,
                times,
                cond_drop_prob=0.0,
                force_mask=force_mask,
                force_x_clean=force_x_clean,
                keep_audio_mask=keep_all,
                keep_traj_mask=keep_all,
                keep_energy_mask=keep_all,
            )
            return unc + (base - unc) * guidance_weight + (energy_cond - base) * energy_scale

        conditioned = self.forward(
            x,
            cond_embed,
            times,
            cond_drop_prob=0.0,
            force_mask=force_mask,
            force_x_clean=force_x_clean,
            keep_audio_mask=keep_all,
            keep_traj_mask=keep_all,
            keep_energy_mask=keep_all,
        )

        return unc + (conditioned - unc) * guidance_weight

    def forward('''
    text = regex_replace_once(text, r'    def guided_forward\([\s\S]*?\n    def forward\(', guided_repl, tag="EDGE_ENERGY_CFG_SCALE")

    text = replace_once(
        text,
        '        keep_audio_mask: Optional[Tensor] = None,\n        keep_traj_mask: Optional[Tensor] = None,\n    ):\n',
        '        keep_audio_mask: Optional[Tensor] = None,\n        keep_traj_mask: Optional[Tensor] = None,\n        keep_energy_mask: Optional[Tensor] = None,\n    ):\n',
        tag="keep_energy_mask: Optional[Tensor] = None",
    )
    text = replace_once(
        text,
        "        audio_cond, trajectory_abs = self._prepare_cond_inputs(\n",
        "        audio_cond, trajectory_abs, energy_cond = self._prepare_cond_inputs(\n",
        tag="audio_cond, trajectory_abs, energy_cond",
    )
    text = insert_after(
        text,
        '        if keep_traj_mask is None:\n            keep_traj_mask = prob_mask_like((batch_size,), keep_prob, device=device)\n',
        '''
        if keep_energy_mask is None:
            energy_drop_prob = cond_drop_prob
            try:
                energy_drop_prob = float(os.environ.get("EDGE_ENERGY_DROP_PROB", energy_drop_prob))
            except Exception:
                pass
            energy_keep_prob = 1.0 - max(0.0, min(1.0, energy_drop_prob))
            keep_energy_mask = prob_mask_like((batch_size,), energy_keep_prob, device=device)
''',
        "keep_energy_mask = prob_mask_like",
    )
    text = insert_after(text, '        keep_traj_mask_root = rearrange(keep_traj_mask, "b -> b 1")\n', '        keep_energy_mask_hidden = rearrange(keep_energy_mask, "b -> b 1")\n', "keep_energy_mask_hidden")

    text = insert_after(
        text,
        "        cond_hidden = torch.where(keep_audio_mask_hidden, cond_hidden, null_cond_hidden)\n",
        '''
        if energy_cond is None:
            energy_hidden = self.null_energy_embed.to(device=t.device, dtype=t.dtype).expand(batch_size, -1)
        else:
            energy_hidden = self.energy_embed(energy_cond.to(device=t.device, dtype=t.dtype))
            null_energy_hidden = self.null_energy_embed.to(device=t.device, dtype=t.dtype).expand_as(energy_hidden)
            energy_hidden = torch.where(keep_energy_mask_hidden, energy_hidden, null_energy_hidden)
''',
        "energy_hidden = self.energy_embed",
    )
    text = text.replace("            t = t + cond_hidden\n", "            t = t + cond_hidden + energy_hidden\n")
    text = text.replace("            t = t + cond_hidden\n", "            t = t + cond_hidden + energy_hidden\n")
    write(path, text)
    print("patched model/model.py")

def patch_diffusion() -> None:
    path = "model/diffusion.py"
    backup(path)
    text = read(path)
    text = replace_once(
        text,
        '        trajectory_loss_weight=1.0,\n        trajectory_velocity_loss_weight=0.25,\n        force_audio_only_drop=False,\n',
        '        trajectory_loss_weight=1.0,\n        trajectory_velocity_loss_weight=0.25,\n        energy_condition_prob=0.7,\n        energy_condition_drop_prob=0.15,\n        energy_loss_weight=0.25,\n        root_lower_coupling_loss_weight=0.5,\n        root_lower_speed_threshold=0.012,\n        root_lower_min_motion=0.010,\n        force_audio_only_drop=False,\n',
        tag="root_lower_coupling_loss_weight=0.5",
    )
    text = insert_after(
        text,
        "        self.trajectory_velocity_loss_weight = float(trajectory_velocity_loss_weight)\n",
        '''
        self.energy_condition_prob = float(energy_condition_prob)
        self.energy_condition_drop_prob = float(energy_condition_drop_prob)
        self.energy_loss_weight = float(energy_loss_weight)
        self.root_lower_coupling_loss_weight = float(root_lower_coupling_loss_weight)
        self.root_lower_speed_threshold = float(root_lower_speed_threshold)
        self.root_lower_min_motion = float(root_lower_min_motion)
''',
        "self.root_lower_coupling_loss_weight = float",
    )
    repl = '''    def _kinematic_sync_loss(self, model_motion_x0):
        """Root-lower coupling loss, logged as Kinematic Sync Loss."""
        if model_motion_x0.shape[-1] != 151 or model_motion_x0.shape[1] < 2:
            return model_motion_x0.new_tensor(0.0)

        if float(getattr(self, "root_lower_coupling_loss_weight", 0.0)) <= 0.0:
            return model_motion_x0.new_tensor(0.0)

        root_xz = model_motion_x0[:, :, [self.root_x_idx, self.root_z_idx]]
        root_speed = safe_norm(root_xz[:, 1:] - root_xz[:, :-1], dim=-1)

        lower_joints = [1, 2, 4, 5, 7, 8, 10, 11]
        lower_indices = []
        for joint in lower_joints:
            lower_indices.extend(range(self.rot_slice.start + 6 * joint, self.rot_slice.start + 6 * (joint + 1)))

        lower = model_motion_x0[:, :, lower_indices]
        lower_motion = safe_norm(lower[:, 1:] - lower[:, :-1], dim=-1)

        contacts = model_motion_x0[:, :, self.contact_slice].clamp(0.0, 1.0)
        contact_change = torch.abs(contacts[:, 1:] - contacts[:, :-1]).mean(dim=-1)

        speed_th = model_motion_x0.new_tensor(float(getattr(self, "root_lower_speed_threshold", 0.012)))
        min_motion = model_motion_x0.new_tensor(float(getattr(self, "root_lower_min_motion", 0.010)))

        fast = F.relu(root_speed - speed_th)
        if not bool((fast > 0).any().item()):
            return model_motion_x0.new_tensor(0.0)

        leg_deficit = F.relu(min_motion - lower_motion)
        contact_deficit = F.relu(0.03 - contact_change)

        denom = fast.detach().mean().clamp_min(1e-6)
        coupling = ((fast * leg_deficit).mean() + 0.25 * (fast * contact_deficit).mean()) / denom
        return coupling * float(getattr(self, "root_lower_coupling_loss_weight", 1.0))

'''
    text = regex_replace_once(text, r'    def _kinematic_sync_loss\(self, model_motion_x0\):[\s\S]*?(?=\n    def _)', repl, tag="Root-lower coupling loss")
    write(path, text)
    print("patched model/diffusion.py")

def patch_generate() -> None:
    path = "generate_controlled.py"
    backup(path)
    text = read(path)
    if "import os" not in text.split("\n")[:40]:
        text = text.replace("import json\n", "import json\nimport os\n", 1)
    helper = '''

# ===== TEA-MotionAdapter inference energy helper =====
def maybe_attach_energy_condition(cond: dict, num_frames: int) -> dict:
    flag = str(os.environ.get("EDGE_ENERGY_COND", "0")).lower() in {"1", "true", "yes", "y", "on"}
    if not flag:
        return cond
    try:
        level = float(os.environ.get("EDGE_ENERGY_LEVEL", "0.5"))
    except Exception:
        level = 0.5
    level = float(np.clip(level, 0.0, 1.0))
    cond["energy"] = torch.full((1, 1), level, dtype=torch.float32)
    print(f"✅ TEA energy condition: level={level:.3f}, cfg_scale={os.environ.get('EDGE_ENERGY_CFG_SCALE', 'default')}")
    return cond
'''
    text = insert_before(text, "\ndef sample_motion(", helper, "maybe_attach_energy_condition")
    text = replace_once(
        text,
        "def sample_motion(model: EDGE, cond: dict, constraint: dict, args, num_frames: int):\n    shape = (1, num_frames, model.repr_dim)\n",
        "def sample_motion(model: EDGE, cond: dict, constraint: dict, args, num_frames: int):\n    cond = maybe_attach_energy_condition(cond, num_frames)\n    cond = {k: (v.to(model.accelerator.device) if torch.is_tensor(v) else v) for k, v in cond.items()} if isinstance(cond, dict) else cond\n    shape = (1, num_frames, model.repr_dim)\n",
        tag="cond = maybe_attach_energy_condition",
    )
    write(path, text)
    print("patched generate_controlled.py")

def main() -> None:
    patch_args()
    patch_train()
    patch_edge()
    patch_dataset()
    patch_model()
    patch_diffusion()
    patch_generate()
    print("\n✅ TEA-MotionAdapter patch installed.")
    print("Next:")
    print("  python -m py_compile args.py train.py EDGE.py dataset/dance_dataset.py model/model.py model/diffusion.py generate_controlled.py")
    print("  bash env_tea_adapter_stageA.sh")

if __name__ == "__main__":
    main()
