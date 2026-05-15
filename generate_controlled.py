# V10-safe runtime patches MUST be installed before importing EDGE / DanceDecoder.
# This fixes V10 inference where rag_summary_projection and text_context_* keys
# were previously ignored during checkpoint loading.

import os

def _env_default(name: str, value: str):
    os.environ.setdefault(name, value)


# ===== V9/V10 inference feature flags =====
_env_default("EDGE_ENABLE_RAG_SUMMARY_TOKEN", "1")
_env_default("EDGE_ENABLE_TEXT_CONTEXT_RAG", "1")
_env_default("EDGE_TEXT_CONTEXT_DIM", "512")
_env_default("EDGE_TEXT_CONTEXT_MAX_POSE_TOKENS", "64")
_env_default("EDGE_RAG_CONTEXT_MAX_LEN", "45")
_env_default("EDGE_TEXT_CONTEXT_DROP_PROB", "0.0")
_env_default("EDGE_TRAJECTORY_REP", "relative_abs_vel")
_env_default("EDGE_TEXT_CONTEXT_DEBUG", "0")


def _install_runtime_patches():
    patch_specs = [
        ("trajectory_native_control", "install_native_trajectory_control_patch"),
        ("trajectory_enhancement_patch", "install_trajectory_enhancement_patch"),
        ("trajectory_gait_phase_patch", "install_trajectory_gait_phase_patch"),
        ("edge_safety_patch", "install_edge_safety_patch"),
        ("v9_rag_inference_patch", "install_v9_rag_inference_patch"),
        ("edge_full_landing_patch", "install_full_landing_patch"),
        ("text_context_rag_model_patch", "install_text_context_rag_model_patch"),
        ("text_context_rag_io_patch", "install_text_context_rag_io_patch"),
        ("text_bridge_planner_patch", "install_text_bridge_planner_patch"),
        ("edge_recon_contract_patch", "install_recon_contract_patch"),
    ]

    # render patch is useful when this script later renders / saves visual assets.
    try:
        patch_specs.append(("render_contact_fix_patch", "install_render_contact_fix_patch"))
    except Exception:
        pass

    for module_name, fn_name in patch_specs:
        try:
            module = __import__(module_name, fromlist=[fn_name])
            install_fn = getattr(module, fn_name)
            try:
                install_fn(verbose=True)
            except TypeError:
                install_fn()
        except Exception as exc:
            print(f"⚠️ {module_name}.{fn_name} not installed: {exc}")


_install_runtime_patches()

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from choreorag_energy_speed import build_energy_curve_from_audio_traj, energy_curve_summary
except Exception:
    build_energy_curve_from_audio_traj = None
    energy_curve_summary = None

try:
    import scipy.interpolate as spi
except Exception:
    spi = None

# IMPORTANT: EDGE must be imported only after all patches above are installed.
from EDGE import EDGE

try:
    from pace_choreorag_trajectory import (
        apply_pace_choreorag_to_trajectory,
        build_pace_progress,
    )
except Exception:
    apply_pace_choreorag_to_trajectory = None
    build_pace_progress = None

try:
    from choreorag_unit_prior import (
        apply_unit_priors_from_specs,
        infer_unit_specs_from_mid_paths,
    )
except Exception:
    apply_unit_priors_from_specs = None
    infer_unit_specs_from_mid_paths = None

from data.audio_extraction.wav2vec_librosa_features import extract as hybrid_extract

try:
    from data.audio_extraction.baseline_features import extract as baseline_extract
except Exception:
    baseline_extract = None

try:
    from auto_keyframe_planner import (
        append_csv,
        plan_auto_keyframes,
        save_auto_keyframes,
    )
except Exception:
    append_csv = None
    plan_auto_keyframes = None
    save_auto_keyframes = None

try:
    from postprocess_footlock import foot_lock_root_correction, blend_back_to_trajectory
except Exception:
    foot_lock_root_correction = None
    blend_back_to_trajectory = None


ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
CONTACT_SLICE = slice(0, 4)
ROT_SLICE = slice(7, 151)


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _coerce_audio_extract_result(result, feature_type: str) -> np.ndarray:
    """Normalize audio extractor return values to a [T, C] float32 array.

    Some EDGE extractors return only the feature array, while the current
    hybrid/baseline extractors may return (feature_array, save_path).  This
    helper prevents np.asarray((array, path)) from producing an inhomogeneous
    object array and crashing generation.
    """
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, (str, Path)):
                continue
            arr = np.asarray(item, dtype=np.float32)
            if arr.ndim >= 2:
                return arr
        result = result[0]

    if isinstance(result, (str, Path)):
        result_path = Path(result)
        if not result_path.exists():
            raise FileNotFoundError(
                f"{feature_type} audio extractor returned missing file: {result_path}"
            )
        result = np.load(result_path)

    arr = np.asarray(result, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 2:
        raise ValueError(
            f"{feature_type} audio feature must be [T,C], got shape={arr.shape}"
        )

    return arr.astype(np.float32)


def extract_audio_feature(path: str, feature_type: str) -> np.ndarray:
    feature_type = str(feature_type).lower()

    if feature_type == "baseline":
        if baseline_extract is None:
            raise RuntimeError(
                "baseline audio extraction requested, but baseline_features.extract is unavailable"
            )
        return _coerce_audio_extract_result(
            baseline_extract(path),
            feature_type="baseline",
        )

    if feature_type in {"hybrid", "jukebox"}:
        return _coerce_audio_extract_result(
            hybrid_extract(path),
            feature_type=feature_type,
        )

    raise ValueError(f"Unsupported feature_type: {feature_type}")


def load_151_pose(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)

    if arr.ndim == 0 and isinstance(arr.item(), dict):
        data = arr.item()
        if "motion" in data:
            arr = data["motion"]
        elif "pose" in data:
            arr = data["pose"]
        else:
            raise ValueError(f"{path} 是 dict .npy，但没有 motion/pose 键。")

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 2:
        if arr.shape[1] != 151:
            raise ValueError(f"{path} 应该是 [T,151] 或 [151]，当前是 {arr.shape}")
        arr = arr[0]

    arr = arr.reshape(-1)

    if arr.shape[0] != 151:
        raise ValueError(f"{path} 必须是 151 维 pose，当前是 {arr.shape}")

    return arr.astype(np.float32)


def parse_list(text: str):
    if not text:
        return []
    return [x.strip() for x in text.replace(";", ",").split(",") if x.strip()]


def parse_mid_frames(text: str, num_mid: int, num_frames: int):
    if num_mid == 0:
        return []

    if not text:
        return [
            int(round((i + 1) * (num_frames - 1) / (num_mid + 1)))
            for i in range(num_mid)
        ]

    raw = parse_list(text)
    if len(raw) != num_mid:
        raise ValueError(f"--mid_pose_frames 数量应为 {num_mid}，当前为 {len(raw)}")

    frames = []
    for item in raw:
        value = float(item)
        if 0.0 < value < 1.0:
            value = value * (num_frames - 1)
        frame = int(round(value))
        frame = max(1, min(num_frames - 2, frame))
        frames.append(frame)

    return frames



def _truthy_env(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }


def _split_paths_csv(text: str):
    if not text:
        return []
    return [x.strip() for x in str(text).replace(";", ",").split(",") if x.strip()]


def _infer_unit_paths_from_mid_pose_paths(mid_pose_text: str):
    """Infer sibling *_unit.npy files from mid pose paths.

    Example:
        output/foo_mid01_f25.npy -> output/foo_mid01_f25_unit.npy

    This is required because Text/Pose Context RAG inference attaches context
    through EDGE_RAG_CONTEXT_UNIT_PATHS / EDGE_RAG_SUMMARY_UNIT_PATHS, while
    --auto_mid_keyframes / --rag_db only populate mid pose files by default.
    """
    out = []
    for item in _split_paths_csv(mid_pose_text):
        path = Path(item)
        if path.name.endswith("_unit.npy"):
            candidate = path
        elif path.suffix == ".npy":
            candidate = path.with_name(path.stem + "_unit.npy")
        else:
            continue
        if candidate.exists():
            out.append(str(candidate))
    return out


def _sync_text_context_env_from_mid_poses(args, out_path: str = ""):
    """Make retrieved unit clips visible to Text/Pose Context RAG IO patch.

    text_context_rag_io_patch.py reads context unit paths from env variables.
    This helper bridges generate_controlled.py's auto-mid output to that env
    contract, so sampler-time IO patch can attach:

        cond["rag_context"]
        cond["rag_context_text_embedding"]
        cond["rag_context_mask"]
    """
    if not _truthy_env("EDGE_ENABLE_TEXT_CONTEXT_RAG", "1"):
        return []

    existing = os.environ.get("EDGE_RAG_CONTEXT_UNIT_PATHS", "").strip()
    if existing:
        paths = _split_paths_csv(existing)
    else:
        paths = _infer_unit_paths_from_mid_pose_paths(getattr(args, "mid_poses", ""))

        if paths:
            os.environ["EDGE_RAG_CONTEXT_UNIT_PATHS"] = ",".join(paths)
            os.environ.setdefault("EDGE_RAG_SUMMARY_UNIT_PATHS", ",".join(paths))

    if paths:
        os.environ.setdefault("EDGE_RAG_CONTEXT_MODE", "normal")
        if out_path and not os.environ.get("EDGE_RAG_CONTEXT_REPORT_JSON", ""):
            report_path = str(Path(out_path).with_suffix("")) + "_context_report.json"
            os.environ["EDGE_RAG_CONTEXT_REPORT_JSON"] = report_path

        print(
            "✅ Text/Pose Context RAG unit paths exported: "
            f"count={len(paths)}, env=EDGE_RAG_CONTEXT_UNIT_PATHS"
        )
        for i, path in enumerate(paths[:8]):
            print(f"   rag_context_unit[{i}]={path}")
    else:
        if _truthy_env("EDGE_TEXT_CONTEXT_REQUIRED", "0"):
            raise RuntimeError(
                "EDGE_TEXT_CONTEXT_REQUIRED=1 but no retrieved *_unit.npy files "
                "were found from --mid_poses and EDGE_RAG_CONTEXT_UNIT_PATHS is empty."
            )
        print(
            "⚠️ Text/Pose Context RAG enabled, but no unit context paths found. "
            "Set EDGE_RAG_CONTEXT_UNIT_PATHS or use mid poses with sibling *_unit.npy files."
        )

    return paths


def parse_trajectory_points(text: str) -> np.ndarray:
    points = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue

        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"轨迹控制点格式错误: {item}，应为 'x,z'，例如 '0,0;1,1;0,2'"
            )

        try:
            x = float(parts[0].strip())
            z = float(parts[1].strip())
        except ValueError as exc:
            raise ValueError(f"轨迹控制点不是数字: {item}") from exc

        points.append([x, z])

    if len(points) == 0:
        raise ValueError("--trajectory 至少需要一个控制点，例如 '0,0;1,1;0,2'")

    return np.asarray(points, dtype=np.float32)


def remove_duplicate_points(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if len(points) <= 1:
        return points.astype(np.float32)

    kept = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - kept[-1]) > eps:
            kept.append(point)

    return np.asarray(kept, dtype=np.float32)


def resample_feature(feature: np.ndarray, target_frames: int) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)

    if feature.ndim != 2:
        raise ValueError(f"feature 应该是 [T,C]，当前是 {feature.shape}")

    if feature.shape[0] == target_frames:
        return feature.astype(np.float32)

    tensor = torch.from_numpy(feature).float().unsqueeze(0).transpose(1, 2)
    tensor = F.interpolate(
        tensor,
        size=target_frames,
        mode="linear",
        align_corners=False,
    )
    return tensor.transpose(1, 2).squeeze(0).numpy().astype(np.float32)


def _safe_normalize_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)
    max_val = float(x.max()) if x.size else 0.0
    if max_val <= 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x / (max_val + 1e-8)).astype(np.float32)


def extract_onset_strength_from_feature(
    audio_feature: np.ndarray,
    onset_index="auto",
) -> np.ndarray:
    """Return a safe onset/proxy-rhythm curve for any audio feature dimension.

    Hybrid features historically place onset around index 768.  Baseline
    features put onset at index 0.  For unknown dimensions, fall back to the
    frame-to-frame feature-difference norm, which is always valid and prevents
    IndexError when feature_dim <= 768.
    """
    audio_feature = np.asarray(audio_feature, dtype=np.float32)

    if audio_feature.ndim != 2:
        raise ValueError(f"audio_feature must be [T,C], got {audio_feature.shape}")

    num_frames, feature_dim = audio_feature.shape
    if num_frames == 0:
        return np.zeros((0,), dtype=np.float32)

    selected = None

    if isinstance(onset_index, str):
        onset_index = onset_index.strip().lower()
        if onset_index not in {"", "auto", "none"}:
            try:
                onset_index = int(onset_index)
            except ValueError:
                onset_index = "auto"

    if isinstance(onset_index, int):
        if 0 <= onset_index < feature_dim:
            selected = audio_feature[:, onset_index]
        else:
            print(
                f"⚠️ onset_index={onset_index} 超出音频特征维度 {feature_dim}，"
                "回退到自动节奏代理。"
            )

    if selected is None:
        if feature_dim > 768:
            selected = audio_feature[:, 768]
        elif feature_dim >= 35:
            # baseline_features layout: [envelope, 20 mfcc, 12 chroma, peak, beat]
            selected = audio_feature[:, 0] + 0.5 * audio_feature[:, -2] + 0.5 * audio_feature[:, -1]
        elif feature_dim > 1:
            diff = np.zeros((num_frames,), dtype=np.float32)
            if num_frames > 1:
                delta = audio_feature[1:] - audio_feature[:-1]
                diff[1:] = np.linalg.norm(delta, axis=-1)
                diff[0] = diff[1]
            selected = diff
        else:
            selected = audio_feature[:, 0]

    return _safe_normalize_1d(selected)


def build_trajectory_progress(
    audio_feature: np.ndarray,
    use_audio_timing: bool = True,
    onset_index="auto",
    min_speed_bias: float = 0.20,
) -> np.ndarray:
    num_frames = audio_feature.shape[0]

    if (not use_audio_timing) or num_frames <= 1:
        return np.linspace(0.0, 1.0, num_frames, dtype=np.float32)

    onset = extract_onset_strength_from_feature(audio_feature, onset_index=onset_index)

    if onset.shape[0] != num_frames or float(onset.max()) <= 1e-8:
        return np.linspace(0.0, 1.0, num_frames, dtype=np.float32)

    speed = onset + float(min_speed_bias)
    progress = np.cumsum(speed)
    progress = progress - progress[0]
    progress = progress / max(float(progress[-1]), 1e-8)
    return progress.astype(np.float32)


def interpolate_trajectory_smooth(
    points: np.ndarray,
    progress: np.ndarray,
    smooth: bool = True,
) -> np.ndarray:
    points = remove_duplicate_points(points)

    if len(points) == 1:
        return np.repeat(points[None, 0, :], len(progress), axis=0).astype(np.float32)

    seg_len = np.linalg.norm(points[1:] - points[:-1], axis=1)
    u = np.concatenate([[0.0], np.cumsum(seg_len)]).astype(np.float32)

    if float(u[-1]) <= 1e-8:
        return np.repeat(points[None, 0, :], len(progress), axis=0).astype(np.float32)

    u = u / u[-1]

    if smooth and spi is not None and len(points) >= 3:
        k = min(3, len(points) - 1)
        try:
            tck, _ = spi.splprep(
                [points[:, 0], points[:, 1]],
                u=u,
                s=0.0,
                k=k,
            )
            x_new, z_new = spi.splev(progress, tck)
            return np.stack([x_new, z_new], axis=-1).astype(np.float32)
        except Exception as exc:
            print(f"⚠️ Spline 轨迹插值失败，回退到线性插值: {exc}")

    x_new = np.interp(progress, u, points[:, 0])
    z_new = np.interp(progress, u, points[:, 1])
    return np.stack([x_new, z_new], axis=-1).astype(np.float32)


def normalize_trajectory_for_model(
    traj_physical: np.ndarray,
    normalizer,
    root_x_idx: int = ROOT_X_IDX,
    root_z_idx: int = ROOT_Z_IDX,
) -> np.ndarray:
    if normalizer is None or not hasattr(normalizer, "mean") or not hasattr(normalizer, "std"):
        raise ValueError("checkpoint 中没有有效 normalizer，无法归一化 trajectory")

    traj_physical = np.asarray(traj_physical, dtype=np.float32)

    mean_x = float(normalizer.mean[root_x_idx])
    mean_z = float(normalizer.mean[root_z_idx])
    std_x = float(normalizer.std[root_x_idx])
    std_z = float(normalizer.std[root_z_idx])

    traj_norm = traj_physical.copy()
    traj_norm[:, 0] = (traj_physical[:, 0] - mean_x) / (std_x + 1e-8)
    traj_norm[:, 1] = (traj_physical[:, 1] - mean_z) / (std_z + 1e-8)

    return traj_norm.astype(np.float32)


def build_control_trajectory(
    trajectory_text: str,
    audio_feature: np.ndarray,
    normalizer,
    target_traj_path: str = "",
    keep_absolute: bool = False,
    uniform_timing: bool = False,
    smooth: bool = True,
    onset_index="auto",
):
    num_frames = audio_feature.shape[0]

    if target_traj_path:
        traj_physical = np.load(target_traj_path).astype(np.float32)

        if traj_physical.ndim == 3:
            traj_physical = traj_physical[0]

        if traj_physical.ndim != 2 or traj_physical.shape[1] < 2:
            raise ValueError(
                f"--target_traj 应该是 [T,2] 或 [1,T,2]，当前是 {traj_physical.shape}"
            )

        traj_physical = traj_physical[:, :2]

        if len(traj_physical) != num_frames:
            traj_physical = resample_feature(traj_physical, num_frames)
    else:
        points = parse_trajectory_points(trajectory_text)
        progress = build_trajectory_progress(
            audio_feature=audio_feature,
            use_audio_timing=not uniform_timing,
            onset_index=onset_index,
        )

        if build_pace_progress is not None:
            try:
                progress = build_pace_progress(
                    audio_feature=audio_feature,
                    num_frames=num_frames,
                    base_progress=progress,
                )
            except Exception as exc:
                print(f"⚠️ PACE beat-aware progress skipped: {exc}")
        traj_physical = interpolate_trajectory_smooth(
            points=points,
            progress=progress,
            smooth=smooth,
        )

    if not keep_absolute:
        traj_physical = traj_physical - traj_physical[0:1]


    # PACE-ChoreoRAG: root-speed scale cap + optional elastic sparse anchors.
    # Disabled unless EDGE_TRAJ_AUTO_SCALE=1 or EDGE_TRAJ_ELASTIC_ANCHOR=1.
    if apply_pace_choreorag_to_trajectory is not None:
        try:
            traj_physical = apply_pace_choreorag_to_trajectory(
                traj_physical=traj_physical,
                audio_feature=audio_feature,
            )
        except Exception as exc:
            print(f"⚠️ PACE trajectory skipped: {exc}")

    traj_norm = normalize_trajectory_for_model(
        traj_physical=traj_physical,
        normalizer=normalizer,
    )

    return traj_physical.astype(np.float32), traj_norm.astype(np.float32)


def normalize_pose_if_needed(pose: np.ndarray, normalizer, pose_space: str) -> np.ndarray:
    if pose_space == "normalized":
        return pose.astype(np.float32)

    if pose_space == "physical":
        if normalizer is None:
            raise ValueError("--pose_space physical 需要 checkpoint 里有 normalizer")

        pose_t = torch.from_numpy(pose[None, None, :]).float()
        pose_norm = normalizer.normalize(pose_t)
        pose_norm = to_numpy(pose_norm)[0, 0]
        return pose_norm.astype(np.float32)

    raise ValueError(f"Unknown pose_space: {pose_space}")


def make_keyframe_feature_mask(constrain_contacts: bool, constrain_root_xz: bool = False) -> np.ndarray:
    mask_one_frame = np.zeros((151,), dtype=np.float32)

    if constrain_contacts:
        mask_one_frame[CONTACT_SLICE] = 1.0

    if constrain_root_xz or _truthy_env("EDGE_KEYFRAME_CONSTRAIN_ROOT_XZ", "0"):
        mask_one_frame[ROOT_X_IDX] = 1.0
        mask_one_frame[ROOT_Z_IDX] = 1.0

    mask_one_frame[ROOT_Y_IDX] = 1.0
    mask_one_frame[ROT_SLICE] = 1.0
    return mask_one_frame


def build_constraint(args, normalizer, num_frames: int, device) -> dict:
    value = np.zeros((num_frames, 151), dtype=np.float32)
    mask = np.zeros((num_frames, 151), dtype=np.float32)

    frame_mask = make_keyframe_feature_mask(
        constrain_contacts=args.constrain_contacts,
        constrain_root_xz=bool(getattr(args, "keyframe_constrain_root_xz", False)),
    )

    def add_pose(path: str, frame: int, name: str, strength: float = 1.0):
        pose = load_151_pose(path)
        pose = normalize_pose_if_needed(pose, normalizer, args.pose_space)

        width = max(0, int(getattr(args, "infer_keyframe_width", 0)))
        start = max(0, frame - width)
        end = min(num_frames, frame + width + 1)

        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.0:
            print(f"⏭️ 跳过 {name} 关键帧: strength={strength:.3f}, path={path}")
            return

        for f in range(start, end):
            value[f] = pose
            mask[f] = frame_mask * strength

        print(
            f"✅ 已添加 {name} 关键帧: frame={frame}, "
            f"window=[{start},{end - 1}], strength={strength:.3f}, path={path}"
        )

    endpoint_strength = float(getattr(args, "endpoint_keyframe_strength", 1.0))
    mid_strength = float(getattr(args, "mid_keyframe_strength", 0.35))

    add_pose(args.start_pose, 0, "start", strength=endpoint_strength)
    add_pose(args.end_pose, num_frames - 1, "end", strength=endpoint_strength)

    mid_paths = parse_list(args.mid_poses)
    mid_frames = parse_mid_frames(args.mid_pose_frames, len(mid_paths), num_frames)

    for i, (path, frame) in enumerate(zip(mid_paths, mid_frames), start=1):
        add_pose(path, frame, f"mid{i}", strength=mid_strength)
    
    # PACE-ChoreoRAG Phase 4: retrieved 45-frame unit -> weak temporal prior.
    # Disabled unless EDGE_UNIT_SOFT_PRIOR=1.  It infers sibling files
    # like xxx_01_f117.npy -> xxx_01_f117_unit.npy.
    if apply_unit_priors_from_specs is not None and infer_unit_specs_from_mid_paths is not None:
        try:
            all_mid_paths = parse_list(args.mid_poses)
            all_mid_frames = parse_mid_frames(args.mid_pose_frames, len(all_mid_paths), num_frames)
            unit_specs = infer_unit_specs_from_mid_paths(all_mid_paths, all_mid_frames)
            value, mask = apply_unit_priors_from_specs(value, mask, unit_specs)
            if unit_specs:
                print(f"✅ ChoreoRAG retrieved-unit specs found: {unit_specs}")
        except Exception as exc:
            import os as _os
            if str(_os.environ.get("EDGE_UNIT_SOFT_PRIOR", "0")).lower() in {"1", "true", "yes", "y", "on"}:
                print(f"⚠️ failed to apply ChoreoRAG retrieved-unit soft prior: {exc}")

    return {
        "value": torch.from_numpy(value[None]).to(device=device, dtype=torch.float32),
        "mask": torch.from_numpy(mask[None]).to(device=device, dtype=torch.float32),
    }



# ===== TEA-MotionAdapter inference energy helper =====
def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}


def _torch_normalize_01(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.amin(dim=1, keepdim=True)
    denom = x.amax(dim=1, keepdim=True).clamp_min(1e-8)
    return x / denom


def _energy_curve_torch_from_cond(cond: dict, num_frames: int) -> torch.Tensor:
    audio = cond.get("audio", None)
    traj = cond.get("trajectory", None)

    if audio is None:
        B = 1 if traj is None else traj.shape[0]
        device = torch.device("cpu") if traj is None else traj.device
        return torch.full((B, num_frames, 1), float(os.environ.get("EDGE_ENERGY_LEVEL", "0.55")), device=device)

    audio = audio.float()
    B, T = audio.shape[0], audio.shape[1]
    if T != num_frames:
        audio = F.interpolate(audio.transpose(1, 2), size=num_frames, mode="linear", align_corners=False).transpose(1, 2)
        T = num_frames

    if audio.shape[-1] > 768:
        onset = torch.relu(audio[..., 768])
    elif audio.shape[-1] >= 35:
        onset = torch.relu(audio[..., 0] + 0.5 * audio[..., -2] + 0.5 * audio[..., -1])
    else:
        onset = torch.zeros((B, T), device=audio.device, dtype=audio.dtype)
        if T > 1:
            onset[:, 1:] = torch.linalg.norm(audio[:, 1:] - audio[:, :-1], dim=-1)
            onset[:, 0] = onset[:, 1]
    onset = _torch_normalize_01(onset)

    if traj is None:
        speed = torch.zeros_like(onset)
        curv = torch.zeros_like(onset)
    else:
        traj = traj.to(device=audio.device, dtype=audio.dtype)[..., :2]
        if traj.shape[1] != T:
            traj = F.interpolate(traj.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)

        speed = torch.zeros((B, T), device=audio.device, dtype=audio.dtype)
        if T > 1:
            speed[:, 1:] = torch.linalg.norm(traj[:, 1:] - traj[:, :-1], dim=-1)
            speed[:, 0] = speed[:, 1]
        speed = _torch_normalize_01(speed)

        curv = torch.zeros((B, T), device=audio.device, dtype=audio.dtype)
        if T > 2:
            v1 = traj[:, 1:-1] - traj[:, :-2]
            v2 = traj[:, 2:] - traj[:, 1:-1]
            n1 = torch.linalg.norm(v1, dim=-1)
            n2 = torch.linalg.norm(v2, dim=-1)
            cos = (v1 * v2).sum(dim=-1) / (n1 * n2).clamp_min(1e-8)
            curv[:, 1:-1] = 1.0 - cos.clamp(-1.0, 1.0)
        curv = _torch_normalize_01(curv)

    w_speed = float(os.environ.get("EDGE_ENERGY_TRAJ_SPEED_WEIGHT", "0.55"))
    w_audio = float(os.environ.get("EDGE_ENERGY_AUDIO_WEIGHT", "0.30"))
    w_curv = float(os.environ.get("EDGE_ENERGY_CURVATURE_WEIGHT", "0.15"))
    total = max(w_speed + w_audio + w_curv, 1e-8)

    base = float(os.environ.get("EDGE_ENERGY_LEVEL", "0.55"))
    e_min = float(os.environ.get("EDGE_ENERGY_MIN", "0.20"))
    e_max = float(os.environ.get("EDGE_ENERGY_MAX", "0.85"))

    dynamic = _torch_normalize_01((w_speed * speed + w_audio * onset + w_curv * curv) / total)
    curve = (0.35 * base + 0.65 * dynamic).clamp(e_min, e_max)
    return curve.unsqueeze(-1)


def maybe_attach_energy_condition(cond: dict, num_frames: int) -> dict:
    if not _env_flag("EDGE_ENERGY_COND", "0"):
        return cond

    if _env_flag("EDGE_ENERGY_TIME_DEPENDENT", "0"):
        curve = _energy_curve_torch_from_cond(cond, num_frames)
        cond["energy"] = curve
        arr = curve.detach().cpu().numpy()
        print(
            "✅ TEA time-dependent energy condition: "
            f"min={arr.min():.3f}, max={arr.max():.3f}, mean={arr.mean():.3f}, "
            f"cfg_scale={os.environ.get('EDGE_ENERGY_CFG_SCALE', 'default')}"
        )
        return cond

    try:
        level = float(os.environ.get("EDGE_ENERGY_LEVEL", "0.5"))
    except Exception:
        level = 0.5
    level = float(np.clip(level, 0.0, 1.0))
    cond["energy"] = torch.full((1, 1), level, dtype=torch.float32)
    print(f"✅ TEA energy condition: level={level:.3f}, cfg_scale={os.environ.get('EDGE_ENERGY_CFG_SCALE', 'default')}")
    return cond


def sample_motion(model: EDGE, cond: dict, constraint: dict, args, num_frames: int):
    cond = maybe_attach_energy_condition(cond, num_frames)
    cond = {k: (v.to(model.accelerator.device) if torch.is_tensor(v) else v) for k, v in cond.items()} if isinstance(cond, dict) else cond
    shape = (1, num_frames, model.repr_dim)

    if args.sampler == "ddim":
        print("🚀 使用 DDIM 采样。")
        return model.diffusion.ddim_sample(
            shape,
            cond,
            constraint=constraint,
        )

    print("🚀 使用 DDPM 采样。")
    return model.diffusion.p_sample_loop(
        shape,
        cond,
        constraint=constraint,
        use_tto=not args.no_tto,
    )


def apply_trajectory_anchor(
    motion_physical: np.ndarray,
    traj_physical: np.ndarray,
    strength: float,
) -> np.ndarray:
    motion = motion_physical.copy()
    traj = np.asarray(traj_physical, dtype=np.float32)

    if traj.ndim == 3:
        traj = traj[0]

    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError(f"traj_physical 应该是 [T,2] 或 [1,T,2]，当前是 {traj.shape}")

    traj = traj[:, :2]
    if len(traj) != len(motion):
        traj = resample_feature(traj, len(motion))

    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return motion.astype(np.float32)

    motion[:, ROOT_X_IDX] = (
        (1.0 - strength) * motion[:, ROOT_X_IDX] + strength * traj[:, 0]
    )
    motion[:, ROOT_Z_IDX] = (
        (1.0 - strength) * motion[:, ROOT_Z_IDX] + strength * traj[:, 1]
    )
    return motion.astype(np.float32)



def project_clean_motion_numpy(motion_norm: np.ndarray, constraint: dict) -> np.ndarray:
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


def unnormalize_motion(normalizer, motion_norm: np.ndarray) -> np.ndarray:
    if normalizer is None:
        return motion_norm.astype(np.float32)

    motion_t = torch.from_numpy(motion_norm[None]).float()
    motion_physical = normalizer.unnormalize(motion_t)
    motion_physical = to_numpy(motion_physical)[0]
    return motion_physical.astype(np.float32)


def save_eval_assets(
    out_path: str,
    motion_raw_physical: np.ndarray,
    motion_final_physical: np.ndarray,
    traj_physical: np.ndarray,
    args,
    trajectory_control_mode: str = "stage4_stage5_direct",
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stem = out_path.stem
    out_dir = out_path.parent

    raw_motion_path = out_dir / f"{stem}_raw.npy"
    final_motion_path = out_path
    target_traj_path = out_dir / f"{stem}_target_traj.npy"
    meta_path = out_dir / f"{stem}_meta.json"

    np.save(raw_motion_path, motion_raw_physical.astype(np.float32))
    np.save(final_motion_path, motion_final_physical.astype(np.float32))
    np.save(target_traj_path, traj_physical.astype(np.float32))

    meta = {
        "checkpoint": args.checkpoint,
        "music": args.music,
        "feature_type": args.feature_type,
        "audio_dim": int(args.audio_dim),
        "trajectory_onset_index": str(getattr(args, "trajectory_onset_index", "auto")),
        "start_pose": args.start_pose,
        "end_pose": args.end_pose,
        "mid_poses": getattr(args, "mid_poses", ""),
        "mid_pose_frames": getattr(args, "mid_pose_frames", ""),
        "auto_mid_keyframes": bool(getattr(args, "auto_mid_keyframes", False)),
        "auto_mid_count": int(getattr(args, "auto_mid_count", 0)),
        "trajectory": getattr(args, "trajectory", ""),
        "target_traj": getattr(args, "target_traj", ""),
        "motion_raw": str(raw_motion_path),
        "motion_final": str(final_motion_path),
        "target_traj_saved": str(target_traj_path),
        "post_anchor_trajectory": bool(getattr(args, "post_anchor_trajectory", False)),
        "trajectory_anchor_strength": float(getattr(args, "trajectory_anchor_strength", 0.0)),
        "endpoint_keyframe_strength": float(getattr(args, "endpoint_keyframe_strength", 1.0)),
        "mid_keyframe_strength": float(getattr(args, "mid_keyframe_strength", 0.35)),
        "infer_keyframe_width": int(getattr(args, "infer_keyframe_width", 0)),
        "sampler": getattr(args, "sampler", "ddpm"),
        "use_tto": not getattr(args, "no_tto", False),
        "trajectory_control_mode": trajectory_control_mode,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ raw motion: {raw_motion_path}")
    print(f"✅ final motion: {final_motion_path}")
    print(f"✅ target trajectory: {target_traj_path}")
    print(f"✅ meta: {meta_path}")

    return {
        "raw_motion": str(raw_motion_path),
        "final_motion": str(final_motion_path),
        "target_traj": str(target_traj_path),
        "meta": str(meta_path),
    }


def _call_with_supported_kwargs(func, **kwargs):
    try:
        sig = inspect.signature(func)
    except Exception:
        return func(**kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return func(**kwargs)

    supported = {name: value for name, value in kwargs.items() if name in sig.parameters}
    return func(**supported)


def _as_auto_keyframe_list(plan):
    if plan is None:
        return []
    if hasattr(plan, "keyframes"):
        return list(getattr(plan, "keyframes"))
    if isinstance(plan, dict):
        if "keyframes" in plan:
            return list(plan["keyframes"])
        if "poses" in plan and "frames" in plan:
            return [
                {"frame": int(frame), "pose": pose}
                for frame, pose in zip(plan["frames"], plan["poses"])
            ]
    if isinstance(plan, (list, tuple)):
        return list(plan)
    return []


def _get_keyframe_field(item, name, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _save_auto_mid_fallback(keyframes, out_dir: Path, prefix: str):
    saved = []
    for i, item in enumerate(keyframes, start=1):
        pose = _get_keyframe_field(item, "pose", None)
        frame = _get_keyframe_field(item, "frame", None)
        if pose is None or frame is None:
            continue
        path = out_dir / f"{prefix}_auto_mid_{i:02d}.npy"
        np.save(path, np.asarray(pose, dtype=np.float32))
        saved.append({"path": str(path), "frame": int(frame)})
    return saved


def maybe_plan_auto_mid(args, audio_feature, traj_physical, normalizer, num_frames):
    if not getattr(args, "auto_mid_keyframes", False):
        return

    if plan_auto_keyframes is None:
        print("⚠️ auto_keyframe_planner 不可用，跳过 auto mid。")
        return

    auto_count = int(getattr(args, "auto_mid_count", 0))
    if auto_count <= 0:
        return

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(args.out).stem

    manual_paths = parse_list(args.mid_poses)
    manual_frames = parse_mid_frames(args.mid_pose_frames, len(manual_paths), num_frames)

    try:
        start_pose_np = normalize_pose_if_needed(
            load_151_pose(args.start_pose),
            normalizer,
            args.pose_space,
        )
        end_pose_np = normalize_pose_if_needed(
            load_151_pose(args.end_pose),
            normalizer,
            args.pose_space,
        )
        user_mid_poses_np = [
            normalize_pose_if_needed(load_151_pose(path), normalizer, args.pose_space)
            for path in manual_paths
        ]
    except Exception as exc:
        print(f"⚠️ auto mid 读取 start/end/user mid pose 失败，跳过 auto mid: {exc}")
        return

    # auto_keyframe_planner.plan_auto_keyframes in main requires:
    # start_pose, end_pose, user_mid_poses, user_mid_frames, audio_feature,
    # traj_physical, rag_db.  Older versions used different names, so keep
    # aliases; _call_with_supported_kwargs will pass only supported arguments.
    planner_kwargs = dict(
        start_pose=start_pose_np,
        end_pose=end_pose_np,
        user_mid_poses=user_mid_poses_np,
        user_mid_frames=manual_frames,
        audio_feature=audio_feature,
        audio_features=audio_feature,
        traj_physical=traj_physical,
        traj_xz=traj_physical,
        trajectory=traj_physical,
        rag_db=getattr(args, "rag_db", ""),
        normalizer=normalizer,
        num_frames=num_frames,

        max_auto_keyframes=auto_count,
        count=auto_count,
        num_keyframes=auto_count,
        existing_frames=manual_frames,

        min_gap=int(getattr(args, "auto_mid_min_gap", 18)),
        source_gap=int(getattr(args, "auto_mid_source_gap", 150)),
        disallow_same_source=bool(getattr(args, "auto_mid_disallow_same_source", False)),

        rag_pose_space=getattr(args, "auto_mid_pose_space", "normalized"),
        pose_space=getattr(args, "auto_mid_pose_space", "normalized"),
        max_candidates=int(getattr(args, "auto_mid_max_candidates", 5000)),
        sample_stride=int(getattr(args, "auto_mid_sample_stride", 3)),

        music_weight=float(getattr(args, "auto_mid_music_weight", 0.6)),
        trajectory_weight=float(getattr(args, "auto_mid_trajectory_weight", 0.4)),
        mmr_checkpoint=getattr(args, "mmr_checkpoint", ""),
        mmr_weight=float(getattr(args, "auto_mid_mmr_weight", 0.5)),
        pose_weight=float(getattr(args, "auto_mid_pose_weight", 1.0)),
        diversity_weight=float(getattr(args, "auto_mid_diversity_weight", 0.25)),
        energy_weight=float(getattr(args, "auto_mid_energy_weight", 0.45)),
        energy_target=float(getattr(args, "auto_mid_energy_target", 0.55)),
        energy_band=float(getattr(args, "auto_mid_energy_band", 0.25)),
        contact_weight=float(getattr(args, "auto_mid_contact_weight", 0.85)),
        contact_diversity_weight=float(getattr(args, "auto_mid_contact_diversity_weight", 0.60)),
        end_weight=float(getattr(args, "auto_mid_end_weight", 0.30)),

        start_pose_path=args.start_pose,
        end_pose_path=args.end_pose,
    )

    try:
        plan = _call_with_supported_kwargs(plan_auto_keyframes, **planner_kwargs)
        keyframes = _as_auto_keyframe_list(plan)

        if not keyframes:
            print("⚠️ auto_keyframe_planner 没有返回有效 keyframes，跳过 auto mid。")
            return

        saved_records = []
        if save_auto_keyframes is not None:
            try:
                saved = _call_with_supported_kwargs(
                    save_auto_keyframes,
                    plan=plan,
                    keyframes=keyframes,
                    out_dir=out_dir,
                    output_dir=out_dir,
                    prefix=prefix,
                )
                if isinstance(saved, dict) and "keyframes" in saved:
                    saved = saved["keyframes"]
                for item in list(saved or []):
                    if isinstance(item, dict):
                        path = item.get("path") or item.get("pose_path")
                        frame = item.get("frame")
                    else:
                        path = getattr(item, "path", None) or getattr(item, "pose_path", None)
                        frame = getattr(item, "frame", None)
                    if path is not None and frame is not None:
                        saved_records.append({"path": str(path), "frame": int(frame)})
            except Exception as exc:
                print(f"⚠️ save_auto_keyframes 失败，使用 fallback 保存: {exc}")

        if not saved_records:
            saved_records = _save_auto_mid_fallback(keyframes, out_dir, prefix)

        if not saved_records:
            print("⚠️ auto mid 保存失败，未加入中间关键帧。")
            return

        auto_pose_paths = [str(item["path"]) for item in saved_records]
        auto_frames = [int(item["frame"]) for item in saved_records]

        all_paths = manual_paths + auto_pose_paths
        all_frames = manual_frames + auto_frames

        args.mid_poses = ",".join(all_paths)
        args.mid_pose_frames = ",".join(str(x) for x in all_frames)

        print(f"✅ auto mid 已加入: count={len(auto_pose_paths)}, frames={auto_frames}")

    except Exception as exc:
        print(f"⚠️ 自动关键帧规划失败，跳过 auto mid: {exc}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Controlled EDGE generation with direct stage-4/5 trajectory control."
    )

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--music", required=True)
    parser.add_argument("--start_pose", required=True)
    parser.add_argument("--end_pose", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--feature_type", default="hybrid", choices=["hybrid", "baseline", "jukebox"])
    parser.add_argument("--audio_dim", type=int, default=803)
    parser.add_argument("--seq_len", type=int, default=150)
    parser.add_argument("--num_frames", type=int, default=150)
    parser.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--sampler", default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument(
        "--trajectory_onset_index",
        default="auto",
        help=(
            "Feature index used for trajectory timing. "
            "'auto' safely chooses hybrid index 768 when available, baseline onset/beat when not, "
            "or frame-difference fallback for arbitrary feature dimensions."
        ),
    )

    parser.add_argument("--mid_poses", default="")
    parser.add_argument("--mid_pose_frames", default="")
    parser.add_argument("--infer_keyframe_width", type=int, default=0)
    parser.add_argument("--endpoint_keyframe_strength", type=float, default=1.0)
    parser.add_argument("--mid_keyframe_strength", type=float, default=0.35)
    parser.add_argument("--pose_space", default="normalized", choices=["normalized", "physical"])
    parser.add_argument("--constrain_contacts", action="store_true")
    parser.add_argument("--hard_keyframe_project", action="store_true")
    parser.add_argument("--infer_project_xstart", action="store_true")
    parser.add_argument("--keyframe_constrain_root_xz", action="store_true")
    parser.add_argument("--disable_traj_cond", action="store_true")
    parser.add_argument("--save_normalized_motion", action="store_true")
    parser.add_argument("--no_ema", action="store_true")

    parser.add_argument("--trajectory", default="0,0;1,1;0,2")
    parser.add_argument("--target_traj", default="")
    parser.add_argument("--keep_absolute_trajectory", action="store_true")
    parser.add_argument("--uniform_trajectory_timing", action="store_true")
    parser.add_argument("--linear_trajectory", action="store_true")

    parser.add_argument("--post_anchor_trajectory", action="store_true")
    parser.add_argument("--trajectory_anchor_strength", type=float, default=0.0)

    parser.add_argument("--no_tto", action="store_true")
    parser.add_argument("--tto_steps", type=int, default=1)
    parser.add_argument("--tto_interval", type=int, default=50)
    parser.add_argument("--tto_lr", type=float, default=0.03)
    parser.add_argument("--tto_contact_threshold", type=float, default=0.65)

    parser.add_argument("--auto_mid_keyframes", action="store_true")
    parser.add_argument("--rag_db", default="data/dunhuang_rag_db/rag_index.npz")
    parser.add_argument("--auto_mid_count", type=int, default=0)
    parser.add_argument("--auto_mid_min_gap", type=int, default=18)
    parser.add_argument("--auto_mid_source_gap", type=int, default=150)
    parser.add_argument("--auto_mid_disallow_same_source", action="store_true")
    parser.add_argument("--auto_mid_pose_space", default="normalized", choices=["normalized", "physical"])
    parser.add_argument("--auto_mid_max_candidates", type=int, default=5000)
    parser.add_argument("--auto_mid_sample_stride", type=int, default=3)
    parser.add_argument("--auto_mid_music_weight", type=float, default=0.6)
    parser.add_argument("--auto_mid_trajectory_weight", type=float, default=0.4)
    parser.add_argument("--mmr_checkpoint", default="")
    parser.add_argument("--auto_mid_mmr_weight", type=float, default=0.5)
    parser.add_argument("--auto_mid_pose_weight", type=float, default=1.0)
    parser.add_argument("--auto_mid_diversity_weight", type=float, default=0.25)
    parser.add_argument("--auto_mid_energy_weight", type=float, default=0.45)
    parser.add_argument("--auto_mid_energy_target", type=float, default=0.55)
    parser.add_argument("--auto_mid_energy_band", type=float, default=0.25)
    parser.add_argument("--auto_mid_contact_weight", type=float, default=0.85)
    parser.add_argument("--auto_mid_contact_diversity_weight", type=float, default=0.60)
    parser.add_argument("--auto_mid_end_weight", type=float, default=0.30)

    return parser


def main():
    args = build_arg_parser().parse_args()
    ensure_parent(args.out)

    audio_feature = extract_audio_feature(args.music, args.feature_type)

    if audio_feature.ndim != 2:
        raise ValueError(f"audio feature 应该是 [T,C]，当前是 {audio_feature.shape}")

    num_frames = int(args.num_frames or args.seq_len or audio_feature.shape[0])
    audio_feature = resample_feature(audio_feature, num_frames)

    model = EDGE(
        feature_type=args.feature_type,
        checkpoint_path=args.checkpoint,
        EMA=not bool(getattr(args, "no_ema", False)),
        audio_dim=args.audio_dim,
        seq_len=num_frames,
        mixed_precision=args.mixed_precision,
    )
    model.eval()

    device = model.accelerator.device
    normalizer = model.normalizer

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
        trajectory_text=args.trajectory,
        audio_feature=audio_feature,
        normalizer=normalizer,
        target_traj_path=args.target_traj,
        keep_absolute=args.keep_absolute_trajectory,
        uniform_timing=args.uniform_trajectory_timing,
        smooth=not args.linear_trajectory,
        onset_index=args.trajectory_onset_index,
    )

    maybe_plan_auto_mid(
        args=args,
        audio_feature=audio_feature,
        traj_physical=traj_physical,
        normalizer=normalizer,
        num_frames=num_frames,
    )

    cond = {
        "audio": torch.from_numpy(audio_feature[None]).to(device=device, dtype=torch.float32),
    }

    if not bool(getattr(args, "disable_traj_cond", False)) and not _truthy_env("EDGE_DISABLE_TRAJ_COND", "0"):
        cond["trajectory"] = torch.from_numpy(traj_norm[None]).to(device=device, dtype=torch.float32)
    else:
        print("✅ Trajectory condition disabled for inference.")

    constraint = build_constraint(
        args=args,
        normalizer=normalizer,
        num_frames=num_frames,
        device=device,
    )

    if hasattr(model.diffusion, "tto_steps"):
        model.diffusion.tto_steps = int(args.tto_steps)
    if hasattr(model.diffusion, "tto_interval"):
        model.diffusion.tto_interval = int(args.tto_interval)
    if hasattr(model.diffusion, "tto_lr"):
        model.diffusion.tto_lr = float(args.tto_lr)
    if hasattr(model.diffusion, "tto_contact_threshold"):
        model.diffusion.tto_contact_threshold = float(args.tto_contact_threshold)

    with torch.no_grad():
        sample = sample_motion(
            model=model,
            cond=cond,
            constraint=constraint,
            args=args,
            num_frames=num_frames,
        )

    motion_norm = to_numpy(sample)[0].astype(np.float32)

    if bool(getattr(args, "hard_keyframe_project", False)) or _truthy_env("EDGE_FINAL_KEYFRAME_PROJECT", "1"):
        motion_norm = project_clean_motion_numpy(motion_norm, constraint)
        print("✅ Final normalized clean-motion keyframe projection applied.")

    if bool(getattr(args, "save_normalized_motion", False)):
        norm_path = Path(args.out).with_suffix("")
        norm_path = norm_path.parent / f"{norm_path.name}_norm.npy"
        np.save(norm_path, motion_norm.astype(np.float32))
        print(f"✅ normalized motion: {norm_path}")

    motion_raw_physical = unnormalize_motion(normalizer, motion_norm)
    motion_final_physical = motion_raw_physical.copy()

    if args.post_anchor_trajectory or float(args.trajectory_anchor_strength) > 0:
        motion_final_physical = apply_trajectory_anchor(
            motion_final_physical,
            traj_physical,
            strength=float(args.trajectory_anchor_strength),
        )

    save_eval_assets(
        out_path=args.out,
        motion_raw_physical=motion_raw_physical,
        motion_final_physical=motion_final_physical,
        traj_physical=traj_physical,
        args=args,
        trajectory_control_mode="stage4_stage5_direct",
    )


if __name__ == "__main__":
    main()
