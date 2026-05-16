import os

# ===== Strict reconstruction inference defaults: must be set before imports =====
os.environ.setdefault("EDGE_SINGLE_RECON_PATCH", "1")

# Match train_single_recon.py architecture.
os.environ.setdefault("EDGE_ENABLE_TEXT_CONTEXT_RAG", "0")
os.environ.setdefault("EDGE_V11_CROSS_ATTN_RAG", "0")
os.environ.setdefault("EDGE_ENABLE_RAG_SUMMARY_TOKEN", "1")

os.environ.setdefault("EDGE_DYNAMIC_TRAJ_CFG", "0")
os.environ.setdefault("EDGE_GAIT_PHASE_COND", "0")
os.environ.setdefault("EDGE_GAIT_CONTACT_LOSS", "0")
os.environ.setdefault("EDGE_TRAJ_PHYSICS_FEATURES", "0")
os.environ.setdefault("EDGE_TRAJ_FOURIER_FEATURES", "0")
os.environ.setdefault("EDGE_TRAJ_SPARSE_WAYPOINT", "0")
os.environ.setdefault("EDGE_TRAJ_BEV_COND", "0")

os.environ.setdefault("EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER", "1")
os.environ.setdefault("EDGE_CHECKPOINT_COMPAT_CPU_MERGE", "1")
os.environ.setdefault("EDGE_AUDIO_DEVICE", "cpu")

os.environ.setdefault("EDGE_RECON_BRIDGE_COND", "1")
os.environ.setdefault("EDGE_RECON_BRIDGE_FEATURES", "rot+root_y")
os.environ.setdefault("EDGE_RECON_BRIDGE_STRENGTH", "0.35")

os.environ.setdefault("EDGE_HARD_KEYFRAME_PROJECT", "1")
os.environ.setdefault("EDGE_INFER_PROJECT_XSTART", "1")

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _install_runtime_patches():
    # Match train.py / train_single_recon.py, but do NOT install trajectory_gait_phase_patch.
    patch_specs = [
        ("trajectory_native_control", "install_native_trajectory_control_patch"),
        ("edge_safety_patch", "install_edge_safety_patch"),
        ("v9_rag_inference_patch", "install_v9_rag_inference_patch"),
        ("edge_full_landing_patch", "install_full_landing_patch"),
        ("text_context_rag_model_patch", "install_text_context_rag_model_patch"),
        ("text_context_rag_io_patch", "install_text_context_rag_io_patch"),
        ("text_bridge_planner_patch", "install_text_bridge_planner_patch"),
        ("edge_recon_contract_patch", "install_recon_contract_patch"),
        ("gait_phase_dataset_patch", "install_gait_phase_dataset_patch"),
        ("trajectory_enhancement_patch", "install_trajectory_enhancement_patch"),
        ("gait_phase_adapter_patch", "install_gait_phase_adapter_patch"),
    ]

    for module_name, fn_name in patch_specs:
        try:
            module = __import__(module_name, fromlist=[fn_name])
            fn = getattr(module, fn_name)
            try:
                fn(verbose=True)
            except TypeError:
                fn()
        except Exception as exc:
            print(f"⚠️ {module_name}.{fn_name} not installed: {exc}")

    try:
        from edge_text_context_training_fix import install_edge_text_context_training_fix
        # Do not call this here; inference does not need training fix.
    except Exception:
        pass

    try:
        from render_contact_fix_patch import install_render_contact_fix_patch
        install_render_contact_fix_patch(verbose=True)
    except Exception as exc:
        print(f"⚠️ render_contact_fix_patch not installed: {exc}")

    try:
        from edge_nextgen_runtime_patch import install_nextgen_runtime_patches
        install_nextgen_runtime_patches(verbose=True)
    except Exception as exc:
        print(f"⚠️ EDGE nextgen runtime patches not installed: {exc}")

    try:
        from gait_phase_dataset_patch import install_gait_phase_dataset_patch
        from trajectory_enhancement_patch import install_trajectory_enhancement_patch
        from gait_phase_adapter_patch import install_gait_phase_adapter_patch
        install_gait_phase_dataset_patch(verbose=True)
        install_trajectory_enhancement_patch(verbose=True)
        install_gait_phase_adapter_patch(verbose=True)
    except Exception as exc:
        print(f"⚠️ EDGE gait phase patches not installed after EDGE import: {exc}")


_install_runtime_patches()

from EDGE import EDGE
from edge_single_unit_recon_patch import install_single_unit_recon_patch

install_single_unit_recon_patch(verbose=True)

from data.audio_extraction.wav2vec_librosa_features import extract as hybrid_extract

try:
    from data.audio_extraction.baseline_features import extract as baseline_extract
except Exception:
    baseline_extract = None


ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
CONTACT_SLICE = slice(0, 4)
ROT_SLICE = slice(7, 151)


def truthy_env(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def parse_list(text: str):
    if not text:
        return []
    return [x.strip() for x in str(text).replace(";", ",").split(",") if x.strip()]


def parse_mid_frames(text: str, num_mid: int, num_frames: int):
    if num_mid == 0:
        return []
    if not text:
        return [int(round((i + 1) * (num_frames - 1) / (num_mid + 1))) for i in range(num_mid)]
    raw = parse_list(text)
    if len(raw) != num_mid:
        raise ValueError(f"--mid_pose_frames 数量应为 {num_mid}，当前为 {len(raw)}")
    out = []
    for item in raw:
        v = float(item)
        if 0.0 < v < 1.0:
            v = v * (num_frames - 1)
        out.append(max(1, min(num_frames - 2, int(round(v)))))
    return out


def coerce_audio_extract_result(result, feature_type: str) -> np.ndarray:
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, (str, Path)):
                continue
            arr = np.asarray(item, dtype=np.float32)
            if arr.ndim >= 2:
                return arr.astype(np.float32)
        result = result[0]

    if isinstance(result, (str, Path)):
        result = np.load(result)

    arr = np.asarray(result, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"{feature_type} audio feature must be [T,C], got {arr.shape}")
    return arr.astype(np.float32)


def extract_audio_feature(path: str, feature_type: str) -> np.ndarray:
    feature_type = str(feature_type).lower()
    if feature_type == "baseline":
        if baseline_extract is None:
            raise RuntimeError("baseline audio extraction requested, but baseline_features.extract is unavailable")
        return coerce_audio_extract_result(baseline_extract(path), feature_type)
    if feature_type in {"hybrid", "jukebox"}:
        return coerce_audio_extract_result(hybrid_extract(path), feature_type)
    raise ValueError(f"Unsupported feature_type: {feature_type}")


def resample_feature(feature: np.ndarray, target_frames: int) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    if feature.shape[0] == target_frames:
        return feature.astype(np.float32)
    t = torch.from_numpy(feature).float().unsqueeze(0).transpose(1, 2)
    t = F.interpolate(t, size=target_frames, mode="linear", align_corners=False)
    return t.transpose(1, 2).squeeze(0).numpy().astype(np.float32)


def load_151_pose(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        d = arr.item()
        if "motion" in d:
            arr = d["motion"]
        elif "pose" in d:
            arr = d["pose"]
        else:
            raise ValueError(f"{path} is dict npy but has no motion/pose key")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[1] != 151:
            raise ValueError(f"{path} should be [T,151] or [151], got {arr.shape}")
        arr = arr[0]
    arr = arr.reshape(-1)
    if arr.shape[0] != 151:
        raise ValueError(f"{path} must be 151-D pose, got {arr.shape}")
    return arr.astype(np.float32)


def normalize_motion(normalizer, motion_physical: np.ndarray) -> np.ndarray:
    if normalizer is None:
        return np.asarray(motion_physical, dtype=np.float32)
    motion_np = np.asarray(motion_physical, dtype=np.float32)
    motion_t = torch.from_numpy(motion_np[None]).float()
    if hasattr(normalizer, "normalize"):
        motion_norm = normalizer.normalize(motion_t)
    else:
        mean = torch.as_tensor(normalizer.mean, dtype=torch.float32).view(1, 1, -1)
        std = torch.as_tensor(normalizer.std, dtype=torch.float32).view(1, 1, -1)
        motion_norm = (motion_t - mean) / torch.clamp(std, min=1e-8)
    return to_numpy(motion_norm)[0].astype(np.float32)


def unnormalize_motion(normalizer, motion_norm: np.ndarray) -> np.ndarray:
    if normalizer is None:
        return np.asarray(motion_norm, dtype=np.float32)
    motion_t = torch.from_numpy(np.asarray(motion_norm, dtype=np.float32)[None]).float()
    motion_physical = normalizer.unnormalize(motion_t)
    return to_numpy(motion_physical)[0].astype(np.float32)


def normalize_pose_if_needed(pose: np.ndarray, normalizer, pose_space: str) -> np.ndarray:
    if pose_space == "normalized":
        return pose.astype(np.float32)
    if pose_space == "physical":
        return normalize_motion(normalizer, pose[None])[0].astype(np.float32)
    raise ValueError(f"Unknown pose_space: {pose_space}")


def make_keyframe_feature_mask(constrain_contacts: bool, constrain_root_xz: bool = False) -> np.ndarray:
    mask = np.zeros((151,), dtype=np.float32)
    if constrain_contacts:
        mask[CONTACT_SLICE] = 1.0
    if constrain_root_xz:
        mask[ROOT_X_IDX] = 1.0
        mask[ROOT_Z_IDX] = 1.0
    mask[ROOT_Y_IDX] = 1.0
    mask[ROT_SLICE] = 1.0
    return mask


def build_constraint(args, normalizer, num_frames: int, device):
    value = np.zeros((num_frames, 151), dtype=np.float32)
    mask = np.zeros((num_frames, 151), dtype=np.float32)

    frame_mask = make_keyframe_feature_mask(
        constrain_contacts=bool(args.constrain_contacts),
        constrain_root_xz=bool(args.keyframe_constrain_root_xz),
    )

    def add_pose(path: str, frame: int, name: str, strength: float):
        pose = load_151_pose(path)
        pose = normalize_pose_if_needed(pose, normalizer, args.pose_space)
        width = max(0, int(args.infer_keyframe_width))
        start = max(0, frame - width)
        end = min(num_frames, frame + width + 1)
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0:
            return
        for f in range(start, end):
            value[f] = pose
            mask[f] = frame_mask * strength
        print(f"✅ 已添加 {name} 关键帧: frame={frame}, window=[{start},{end - 1}], strength={strength:.3f}, path={path}")

    add_pose(args.start_pose, 0, "start", args.endpoint_keyframe_strength)
    add_pose(args.end_pose, num_frames - 1, "end", args.endpoint_keyframe_strength)

    mid_paths = parse_list(args.mid_poses)
    mid_frames = parse_mid_frames(args.mid_pose_frames, len(mid_paths), num_frames)
    for i, (p, f) in enumerate(zip(mid_paths, mid_frames), start=1):
        add_pose(p, f, f"mid{i}", args.mid_keyframe_strength)

    return {
        "value": torch.from_numpy(value[None]).to(device=device, dtype=torch.float32),
        "mask": torch.from_numpy(mask[None]).to(device=device, dtype=torch.float32),
    }


def add_all_frame_root_xz_constraint(constraint, reference_motion, normalizer, pose_space, num_frames, device):
    ref = np.asarray(reference_motion, dtype=np.float32)
    if ref.ndim != 2 or ref.shape[-1] != 151:
        raise ValueError(f"root_xz_reference must be [T,151], got {ref.shape}")
    if len(ref) < num_frames:
        ref = np.concatenate([ref, np.repeat(ref[-1:], num_frames - len(ref), axis=0)], axis=0)
    ref = ref[:num_frames]

    if str(pose_space).lower() == "physical":
        ref_norm = normalize_motion(normalizer, ref)
    else:
        ref_norm = ref.astype(np.float32)

    value = to_numpy(constraint["value"]).astype(np.float32).copy()
    mask = to_numpy(constraint["mask"]).astype(np.float32).copy()
    if mask.shape[-1] == 1:
        mask = np.repeat(mask, 151, axis=-1)

    value[0, :, ROOT_X_IDX] = ref_norm[:, ROOT_X_IDX]
    value[0, :, ROOT_Z_IDX] = ref_norm[:, ROOT_Z_IDX]
    mask[0, :, ROOT_X_IDX] = 1.0
    mask[0, :, ROOT_Z_IDX] = 1.0

    return {
        "value": torch.from_numpy(value).to(device=device, dtype=torch.float32),
        "mask": torch.from_numpy(mask).to(device=device, dtype=torch.float32),
    }


def project_clean_motion_numpy(motion_norm: np.ndarray, constraint) -> np.ndarray:
    if constraint is None:
        return motion_norm.astype(np.float32)
    mask = to_numpy(constraint["mask"])[0].astype(np.float32)
    value = to_numpy(constraint["value"])[0].astype(np.float32)
    if mask.shape[-1] == 1:
        mask = np.repeat(mask, motion_norm.shape[-1], axis=-1)
    return (motion_norm * (1.0 - mask) + value * mask).astype(np.float32)


def parse_trajectory_points(text: str) -> np.ndarray:
    pts = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        x, z = item.split(",")
        pts.append([float(x), float(z)])
    if not pts:
        pts = [[0.0, 0.0]]
    return np.asarray(pts, dtype=np.float32)


def build_simple_trajectory(args, normalizer, num_frames):
    if args.target_traj:
        traj = np.load(args.target_traj).astype(np.float32)
        if traj.ndim == 3:
            traj = traj[0]
        traj = traj[:, :2]
        if len(traj) != num_frames:
            traj = resample_feature(traj, num_frames)
    else:
        pts = parse_trajectory_points(args.trajectory)
        if len(pts) == 1:
            traj = np.repeat(pts[:1], num_frames, axis=0)
        else:
            u = np.linspace(0.0, 1.0, len(pts), dtype=np.float32)
            q = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
            traj = np.stack([np.interp(q, u, pts[:, 0]), np.interp(q, u, pts[:, 1])], axis=-1).astype(np.float32)

    if not args.keep_absolute_trajectory:
        traj = traj - traj[:1]

    if normalizer is not None and hasattr(normalizer, "mean"):
        out = traj.copy()
        out[:, 0] = (traj[:, 0] - float(normalizer.mean[ROOT_X_IDX])) / (float(normalizer.std[ROOT_X_IDX]) + 1e-8)
        out[:, 1] = (traj[:, 1] - float(normalizer.mean[ROOT_Z_IDX])) / (float(normalizer.std[ROOT_Z_IDX]) + 1e-8)
        return traj.astype(np.float32), out.astype(np.float32)
    return traj.astype(np.float32), traj.astype(np.float32)


def save_eval_assets(out_path, motion_raw_physical, motion_final_physical, traj_physical, args):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.stem
    out_dir = out_path.parent

    raw_path = out_dir / f"{stem}_raw.npy"
    traj_path = out_dir / f"{stem}_target_traj.npy"
    meta_path = out_dir / f"{stem}_meta.json"

    np.save(raw_path, motion_raw_physical.astype(np.float32))
    np.save(out_path, motion_final_physical.astype(np.float32))
    np.save(traj_path, traj_physical.astype(np.float32))

    meta = {
        "checkpoint": args.checkpoint,
        "music": args.music,
        "feature_type": args.feature_type,
        "audio_dim": int(args.audio_dim),
        "start_pose": args.start_pose,
        "end_pose": args.end_pose,
        "mid_poses": args.mid_poses,
        "mid_pose_frames": args.mid_pose_frames,
        "root_xz_reference": args.root_xz_reference,
        "sampler": args.sampler,
        "guidance_weight": float(args.guidance_weight),
        "strict_single_recon_clean_generate": True,
        "text_context_rag": os.environ.get("EDGE_ENABLE_TEXT_CONTEXT_RAG", "0"),
        "v11_cross_attn_rag": os.environ.get("EDGE_V11_CROSS_ATTN_RAG", "0"),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ raw motion: {raw_path}")
    print(f"✅ final motion: {out_path}")
    print(f"✅ target trajectory: {traj_path}")
    print(f"✅ meta: {meta_path}")


def build_arg_parser():
    p = argparse.ArgumentParser("Clean strict single-unit EDGE reconstruction generation")

    p.add_argument("--checkpoint", required=True)
    p.add_argument("--music", required=True)
    p.add_argument("--start_pose", required=True)
    p.add_argument("--end_pose", required=True)
    p.add_argument("--out", required=True)

    p.add_argument("--feature_type", default="hybrid", choices=["hybrid", "baseline", "jukebox"])
    p.add_argument("--audio_dim", type=int, default=803)
    p.add_argument("--seq_len", type=int, default=45)
    p.add_argument("--num_frames", type=int, default=45)
    p.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--sampler", default="ddim", choices=["ddpm", "ddim"])
    p.add_argument("--guidance_weight", type=float, default=1.0)

    p.add_argument("--mid_poses", default="")
    p.add_argument("--mid_pose_frames", default="")
    p.add_argument("--infer_keyframe_width", type=int, default=0)
    p.add_argument("--endpoint_keyframe_strength", type=float, default=1.0)
    p.add_argument("--mid_keyframe_strength", type=float, default=1.0)
    p.add_argument("--pose_space", default="physical", choices=["physical", "normalized"])
    p.add_argument("--constrain_contacts", action="store_true")
    p.add_argument("--hard_keyframe_project", action="store_true")
    p.add_argument("--infer_project_xstart", action="store_true")
    p.add_argument("--keyframe_constrain_root_xz", action="store_true")
    p.add_argument("--root_xz_reference", default="")

    p.add_argument("--trajectory", default="0,0")
    p.add_argument("--target_traj", default="")
    p.add_argument("--keep_absolute_trajectory", action="store_true")
    p.add_argument("--disable_traj_cond", action="store_true")
    p.add_argument("--no_tto", action="store_true")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--save_normalized_motion", action="store_true")

    return p


def main():
    args = build_arg_parser().parse_args()
    ensure_parent(args.out)

    audio_feature = extract_audio_feature(args.music, args.feature_type)
    num_frames = int(args.num_frames or args.seq_len or audio_feature.shape[0])
    audio_feature = resample_feature(audio_feature, num_frames)

    model = EDGE(
        feature_type=args.feature_type,
        checkpoint_path=args.checkpoint,
        EMA=not bool(args.no_ema),
        audio_dim=args.audio_dim,
        seq_len=num_frames,
        mixed_precision=args.mixed_precision,
    )
    model.eval()

    device = model.accelerator.device
    normalizer = model.normalizer

    if hasattr(model, "diffusion") and hasattr(model.diffusion, "guidance_weight"):
        model.diffusion.guidance_weight = float(args.guidance_weight)
        print(f"✅ diffusion guidance_weight set to {model.diffusion.guidance_weight}")

    if args.hard_keyframe_project or truthy_env("EDGE_HARD_KEYFRAME_PROJECT", "0"):
        os.environ["EDGE_HARD_KEYFRAME_PROJECT"] = "1"
        os.environ["EDGE_INFER_PROJECT_XSTART"] = "1"
        if hasattr(model, "diffusion"):
            model.diffusion.hard_keyframe_project = True
        print("✅ Strict hard keyframe projection enabled.")

    if args.infer_project_xstart:
        os.environ["EDGE_INFER_PROJECT_XSTART"] = "1"
        print("✅ Clean x_start projection enabled.")

    traj_physical, traj_norm = build_simple_trajectory(args, normalizer, num_frames)

    cond = {
        "audio": torch.from_numpy(audio_feature[None]).to(device=device, dtype=torch.float32)
    }

    if not args.disable_traj_cond:
        cond["trajectory"] = torch.from_numpy(traj_norm[None]).to(device=device, dtype=torch.float32)
    else:
        print("✅ Trajectory condition disabled for inference.")

    constraint = build_constraint(args, normalizer, num_frames, device)

    if args.root_xz_reference.strip():
        root_ref = np.load(args.root_xz_reference).astype(np.float32)
        constraint = add_all_frame_root_xz_constraint(
            constraint=constraint,
            reference_motion=root_ref,
            normalizer=normalizer,
            pose_space=args.pose_space,
            num_frames=num_frames,
            device=device,
        )
        print(f"✅ Added all-frame root X/Z constraint from: {args.root_xz_reference}")

    shape = (1, num_frames, model.repr_dim)

    with torch.no_grad():
        if args.sampler == "ddim":
            print("🚀 使用 clean DDIM 采样。")
            sample = model.diffusion.ddim_sample(shape, cond, constraint=constraint)
        else:
            print("🚀 使用 clean DDPM 采样。")
            sample = model.diffusion.p_sample_loop(
                shape,
                cond,
                constraint=constraint,
                use_tto=not args.no_tto,
            )

    motion_norm = to_numpy(sample)[0].astype(np.float32)

    if args.hard_keyframe_project or truthy_env("EDGE_FINAL_KEYFRAME_PROJECT", "1"):
        motion_norm = project_clean_motion_numpy(motion_norm, constraint)
        print("✅ Final normalized clean-motion keyframe projection applied.")

    if args.save_normalized_motion:
        norm_path = Path(args.out).with_suffix("")
        norm_path = norm_path.parent / f"{norm_path.name}_norm.npy"
        np.save(norm_path, motion_norm.astype(np.float32))
        print(f"✅ normalized motion: {norm_path}")

    motion_raw_physical = unnormalize_motion(normalizer, motion_norm)
    motion_final_physical = motion_raw_physical.copy()

    save_eval_assets(
        out_path=args.out,
        motion_raw_physical=motion_raw_physical,
        motion_final_physical=motion_final_physical,
        traj_physical=traj_physical,
        args=args,
    )


if __name__ == "__main__":
    main()
