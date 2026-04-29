import os
from contextlib import nullcontext
import torch
import gradio as gr
import numpy as np
import scipy.interpolate as spi
import torch.nn.functional as F
import math
import pose_keyframe as pose_keyframe_module
from model.model import DanceDecoder
from model.checkpoint_compat import adapt_checkpoint_state_dict, summarize_adapt_report
from model.diffusion import GaussianDiffusion
from data.audio_extraction.wav2vec_librosa_features import extract
from dataset.quaternion import ax_from_6v
from pose_keyframe import build_pose_151
from vis import skeleton_render, SMPLSkeleton
from trajectory_postprocess import TRAJECTORY_POST_MODES, apply_trajectory_postprocess

DEFAULT_CKPT_CANDIDATES = [
    os.environ.get("EDGE_GRADIO_CKPT", ""),
    "runs/train/exp_dunhuang_stage2B_phys_opt/weights/train-20.pt",
    "runs/train/exp_dunhuang_keyframe_stage2B_stability_from102/weights/train-5.pt",
    "runs/train/exp16/weights/train-300.pt",
]


def resolve_default_checkpoint():
    for candidate in DEFAULT_CKPT_CANDIDATES:
        if candidate and os.path.exists(candidate):
            return candidate
    return DEFAULT_CKPT_CANDIDATES[-1]


class PipelineEngine:
    def __init__(self, ckpt_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.ckpt_path = ckpt_path or resolve_default_checkpoint()
        self.model = None
        self.normalizer = None
        self.is_loaded = False

    def load_models(self):
        if self.is_loaded: return
        print("⏳ 正在初始化全局大模型底座...")
        self.model = DanceDecoder(
            nfeats=151, seq_len=150, cond_feature_dim=803, latent_dim=512, 
            ff_size=1024, num_layers=8, num_heads=8, dropout=0.1
        ).to(self.device)
        
        checkpoint = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        
        # 🌟 修复：反序列化 Normalizer
        norm_data = checkpoint.get("normalizer")
        if isinstance(norm_data, dict) and "mean" in norm_data:
            from dataset.preprocess import Normalizer
            dummy = torch.zeros((1, 1, 151))
            self.normalizer = Normalizer(dummy)
            self.normalizer.mean = np.array(norm_data["mean"])
            self.normalizer.std = np.array(norm_data["std"])
        else:
            self.normalizer = norm_data
            
        state_dict = checkpoint.get('ema_state_dict', checkpoint.get('model_state_dict'))
        if state_dict is None:
            raise ValueError(f"权重文件 {self.ckpt_path} 中没有 ema_state_dict/model_state_dict")
        adapted_state_dict, adapt_report = adapt_checkpoint_state_dict(
            state_dict,
            self.model,
            log_prefix=f"checkpoint:{os.path.basename(self.ckpt_path)}",
        )
        self.model.load_state_dict(adapted_state_dict, strict=True)
        for line in summarize_adapt_report(adapt_report):
            print(f"⚠️ {line}")
        self.model.eval()
        self.is_loaded = True
        print("✅ 模型加载完毕！")

engine = PipelineEngine()

def apply_hybrid_keyframe_mask(mask, value, target_pose_norm, peak_idx, is_start=True, window_size=3):
    device = mask.device
    dtype = mask.dtype
    target_pose = target_pose_norm.to(device).to(dtype)
    seq_len = mask.shape[1]
    
    for i in range(window_size):
        frame_idx = peak_idx + i if is_start else peak_idx - i
        if 0 <= frame_idx < seq_len:
            mask[:, frame_idx, :] = 1.0  # ✅ 修复：采用 1.0 的绝对硬约束
            value[:, frame_idx, :] = target_pose
    return mask, value


def generate_trajectory_tensor(control_points_str, seq_len, device, dtype, audio_feat_np=None):
    control_points = []
    if control_points_str:
        try:
            for p in control_points_str.split(';'):
                x, z = p.split(',')
                control_points.append([float(x.strip()), float(z.strip())])
        except Exception: pass

    if not control_points:
        traj_np = np.zeros((seq_len, 2))
    elif len(control_points) == 1:
        traj_np = np.tile(np.array(control_points[0]), (seq_len, 1))
    else:
        pts = np.array(control_points).T
        k_val = min(3, len(control_points) - 1) 
        tck, _ = spi.splprep(pts, s=0, k=k_val)
        
        if audio_feat_np is not None and audio_feat_np.shape[-1] >= 769:
            onset_strength = audio_feat_np[:, 768]
            speed_curve = onset_strength + (0.2 * np.max(onset_strength) if np.max(onset_strength) > 0 else 1.0)
            cum_progress = np.cumsum(speed_curve)
            u_new = (cum_progress - cum_progress[0]) / (cum_progress[-1] - cum_progress[0])
        else:
            u_new = np.linspace(0, 1, seq_len)
            
        x_new, z_new = spi.splev(u_new, tck)
        traj_np = np.stack([x_new, z_new], axis=1)

    return torch.tensor(traj_np, device=device, dtype=dtype).unsqueeze(0)

# -----------------------------------------------------
# 🌟 核心找回 2：FK 高度预结算 + 专家 PKL 混合通道
# -----------------------------------------------------
def estimate_3d_pose_hybrid(image_path, pkl_file, normalizer, device='cuda', target_trans_xz=None, is_flying=False):
    pose_np = build_pose_151(
        image_path=image_path,
        pkl_path=pkl_file,
        normalizer=normalizer,
        device=device,
        target_trans_xz=target_trans_xz,
        is_flying=is_flying,
        normalize=True,
    )
    return torch.tensor(pose_np, device=device, dtype=torch.float32)

# -----------------------------------------------------
# 🌟 核心保留 3：带 is_mask 的智能切块器
# -----------------------------------------------------
def chunk_tensor(tensor, horizon=150, stride=75, target_frames=150, is_mask=False):
    tensor = tensor.squeeze(0) 
    chunks = []
    
    def safe_pad(seq, target_len, is_mask):
        pad_len = target_len - seq.shape[0]
        if is_mask:
            pad_tensor = torch.zeros_like(seq[-1:]).repeat(pad_len, 1)
            return torch.cat([seq, pad_tensor], dim=0)
        else:
            seq_np = seq.cpu().numpy()
            if seq_np.shape[0] == 1:
                seq_np = np.repeat(seq_np, target_len, axis=0)
            else:
                curr_pad = pad_len
                while curr_pad > 0:
                    step_pad = min(curr_pad, seq_np.shape[0] - 1)
                    seq_np = np.pad(seq_np, ((0, step_pad), (0, 0)), mode='reflect')
                    curr_pad -= step_pad
            return torch.from_numpy(seq_np).to(seq.device).to(seq.dtype)

    if target_frames <= horizon:
        chunk = safe_pad(tensor, horizon, is_mask)
        chunks.append(chunk)
    else:
        for i in range(0, target_frames, stride):
            chunk = tensor[i : i + horizon]
            if chunk.shape[0] < horizon:
                chunk = safe_pad(chunk, horizon, is_mask)
            chunks.append(chunk)
            if i + horizon >= target_frames:
                break
    return torch.stack(chunks, dim=0) 

def run_choreography_pipeline(audio_file, img_start_in, pkl_start_in, img_end_in, pkl_end_in, trajectory_str, trajectory_post_mode, use_tto, hard_keyframe_project, is_flying):
    if not engine.is_loaded: engine.load_models()

    print("🎵 Pipeline Step 1: 提取多模态特征...")
    audio_feat_np, _ = extract(audio_file)
    audio_seq_len = audio_feat_np.shape[0]
    if audio_seq_len % 2 != 0: audio_seq_len -= 1; audio_feat_np = audio_feat_np[:audio_seq_len]
    
    cond_audio = torch.tensor(audio_feat_np, device=engine.device, dtype=engine.dtype).unsqueeze(0)
    has_traj = bool(trajectory_str and trajectory_str.strip())
    physical_target_traj = generate_trajectory_tensor(
        trajectory_str, audio_seq_len, engine.device, engine.dtype, audio_feat_np
    )
    
    # 1. 初始归零：以第一帧为绝对原点
    start_xz_abs = physical_target_traj[:, 0, :].clone()
    physical_target_traj = physical_target_traj - start_xz_abs.unsqueeze(1)
    normalized_cond_traj = physical_target_traj.clone()

    # ==========================================================
    # ✨ 核心改进 1：全局执行量纲对齐 (解决双重归一化问题)
    # ==========================================================
    if engine.normalizer is not None:
        mean_x = torch.tensor(engine.normalizer.mean[4], device=engine.device, dtype=engine.dtype)
        mean_z = torch.tensor(engine.normalizer.mean[6], device=engine.device, dtype=engine.dtype)
        std_x = torch.tensor(engine.normalizer.std[4], device=engine.device, dtype=engine.dtype)
        std_z = torch.tensor(engine.normalizer.std[6], device=engine.device, dtype=engine.dtype)
        
        normalized_cond_traj[:, :, 0] = (normalized_cond_traj[:, :, 0] - mean_x) / (std_x + 1e-6)
        normalized_cond_traj[:, :, 1] = (normalized_cond_traj[:, :, 1] - mean_z) / (std_z + 1e-6)
        
    start_xz_physical = physical_target_traj[0, 0].cpu().float().numpy()
    end_xz_physical = physical_target_traj[0, -1].cpu().float().numpy()

    print("🖼️/📁 Pipeline Step 2: 提取混合起止姿态 (物理贴地校正)...")
    
    start_pose_norm = estimate_3d_pose_hybrid(
        img_start_in,
        pkl_start_in,
        engine.normalizer,
        target_trans_xz=start_xz_physical,
        is_flying=is_flying,
    )
    end_pose_norm = estimate_3d_pose_hybrid(
        img_end_in,
        pkl_end_in,
        engine.normalizer,
        target_trans_xz=end_xz_physical,
        is_flying=is_flying,
    )
    
    start_pose_norm[4] = normalized_cond_traj[0, 0, 0].to(start_pose_norm.dtype)  # Root X
    start_pose_norm[6] = normalized_cond_traj[0, 0, 1].to(start_pose_norm.dtype)  # Root Z
    
    end_pose_norm[4] = normalized_cond_traj[0, -1, 0].to(end_pose_norm.dtype)     # Root X
    end_pose_norm[6] = normalized_cond_traj[0, -1, 1].to(end_pose_norm.dtype)     # Root Z

    mask = torch.zeros(1, audio_seq_len, 151).to(engine.device).to(engine.dtype)
    value = torch.zeros(1, audio_seq_len, 151).to(engine.device).to(engine.dtype)

    mask, value = apply_hybrid_keyframe_mask(mask, value, start_pose_norm, 0, is_start=True, window_size=3)
    mask, value = apply_hybrid_keyframe_mask(mask, value, end_pose_norm, audio_seq_len - 1, is_start=False, window_size=3)
    
    horizon = engine.model.seq_len
    stride = horizon // 2
    
    N_cond_audio = chunk_tensor(cond_audio, horizon, stride, audio_seq_len)
    N_cond_traj  = chunk_tensor(normalized_cond_traj, horizon, stride, audio_seq_len) 
    N_mask       = chunk_tensor(mask, horizon, stride, audio_seq_len, is_mask=True) 
    N_value      = chunk_tensor(value, horizon, stride, audio_seq_len)
    
            
    # 找到 cond_dict 定义的地方，修改为：
    if has_traj:
        cond_dict = {"audio": N_cond_audio, "trajectory": N_cond_traj}
    else:
        # 👇 修复：用户未指定轨迹时，不传入 trajectory，触发模型内部的 null_cond_embed 自由生成
        cond_dict = {"audio": N_cond_audio} 
        print("🕊️ 用户未指定轨迹，已解锁根节点位移限制，允许模型自由发挥！")
        
    constraint_dict = {"mask": N_mask, "value": N_value}
    
    N = N_cond_audio.shape[0]
    
    # ==========================================
    # 🌟 修复：OOM 防御，释放前期辅助网络的显存
    # ==========================================
    print("🧹 正在清理特征提取器显存，为扩散模型极限腾出空间...")
    import gc
    if pose_keyframe_module._ROMP_MODEL is not None:
        del pose_keyframe_module._ROMP_MODEL
        pose_keyframe_module._ROMP_MODEL = None
        
    from data.audio_extraction.wav2vec_librosa_features import _EXTRACTOR
    if _EXTRACTOR is not None and hasattr(_EXTRACTOR, 'model'):
        _EXTRACTOR.model.to('cpu') 
        
    gc.collect()
    torch.cuda.empty_cache()

    print(f"🚀 Pipeline Step 3 & 4: 开始分批次运行扩散模型 (共 {N} 块)...")
    
    active_smpl = SMPLSkeleton(device=engine.device)
    diffusion = GaussianDiffusion(
        model=engine.model, horizon=horizon, repr_dim=151, 
        smpl=active_smpl, n_timestep=1000, 
        predict_epsilon=False,
        guidance_weight=2.5,
        hard_keyframe_project=hard_keyframe_project,
    ).to(engine.device)
    diffusion.normalizer = engine.normalizer 
    
    out_dir = "output/ui_renders"
    os.makedirs(out_dir, exist_ok=True)
    temp_name = "ui_session"
    
    batch_size = 1      
    step = batch_size  
    overlap_frames = 15
    all_outputs = []
    
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=engine.dtype)
        if engine.device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad():
        with autocast_ctx:
            for i in range(0, N, step):
                end_idx = min(i + batch_size, N)
                current_batch_size = end_idx - i
                
                sub_cond_dict = {
                    "audio": cond_dict["audio"][i:end_idx]
                }
                # 安全判断：仅在存在轨迹条件时才进行切片提取
                if "trajectory" in cond_dict:
                    sub_cond_dict["trajectory"] = cond_dict["trajectory"][i:end_idx]
                
                sub_constraint_mask = constraint_dict["mask"][i:end_idx].clone()
                sub_constraint_value = constraint_dict["value"][i:end_idx].clone()
                
                if i > 0 and len(all_outputs) > 0:
                    prev_last_chunk = all_outputs[-1][-1:].to(engine.device).clone() 
                    
                    sub_constraint_mask[0, :overlap_frames, :] = 1.0 
                    sub_constraint_value[0, :overlap_frames, :] = prev_last_chunk[0, 75:75+overlap_frames, :]
                
                sub_output = diffusion.long_inpaint_loop(
                    shape=(current_batch_size, horizon, 151), 
                    cond=sub_cond_dict,
                    constraint={"mask": sub_constraint_mask, "value": sub_constraint_value},
                    use_tto=use_tto 
                )
                
                all_outputs.append(sub_output.detach().cpu())

    output_motion = torch.cat(all_outputs, dim=0)
    output_motion_physical = engine.normalizer.unnormalize(output_motion)
    
    from dataset.quaternion import quat_slerp, ax_from_6v
    from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_matrix, matrix_to_rotation_6d

    current_motion = output_motion_physical[0].clone()
    
    for i in range(1, N):
        next_chunk = output_motion_physical[i].clone()

        # 绝对偏移量校正：保证上一块末尾与下一块开头的根节点物理位置 100% 重合
        delta_x = current_motion[i * stride, 4] - next_chunk[0, 4]
        delta_z = current_motion[i * stride, 6] - next_chunk[0, 6]
        next_chunk[:, 4] += delta_x
        next_chunk[:, 6] += delta_z

        # 动态计算重叠长度，防止切块越界
        blend_len = min(15, stride, next_chunk.shape[0])
        fade_w = torch.linspace(0, 1, blend_len, device=next_chunk.device).unsqueeze(1)
        # 五次多项式缓动函数，保证一阶（速度）和二阶（加速度）导数连续
        fade_w_smooth = 6 * (fade_w ** 5) - 15 * (fade_w ** 4) + 10 * (fade_w ** 3)

        # 平滑融合边界：对根节点位置进行线性插值混合
        next_chunk[:blend_len, 4:7] = (
            current_motion[i * stride : i * stride + blend_len, 4:7] * (1 - fade_w_smooth)
            + next_chunk[:blend_len, 4:7] * fade_w_smooth
        )

        # 平滑融合边界：对 6D 旋转进行四元数球面线性插值 (SLERP)
        q_curr = axis_angle_to_quaternion(
            ax_from_6v(current_motion[i * stride : i * stride + blend_len, 7:].reshape(blend_len, 24, 6))
        )
        q_next = axis_angle_to_quaternion(
            ax_from_6v(next_chunk[:blend_len, 7:].reshape(blend_len, 24, 6))
        )
        q_blended = quat_slerp(q_curr, q_next, fade_w)
        next_chunk[:blend_len, 7:] = matrix_to_rotation_6d(
            quaternion_to_matrix(q_blended)
        ).reshape(blend_len, 144)

        current_motion = torch.cat([current_motion[: i * stride], next_chunk], dim=0)

    output_np = current_motion[:audio_seq_len].cpu().float().numpy()

    print(f"🔧 正在执行轨迹后处理模式: {trajectory_post_mode}")
    target_traj_np = physical_target_traj[0, :audio_seq_len].cpu().float().numpy() if has_traj else None
    output_np = apply_trajectory_postprocess(
        output_np,
        target_traj=target_traj_np,
        mode=trajectory_post_mode,
        device=engine.device,
    )
            
    # ==================== ✨ 正向运动学 (FK) 还原 3D 关节坐标 ====================
    output_tensor = torch.tensor(output_np, device=engine.device, dtype=engine.dtype).unsqueeze(0)
    
    contacts = output_tensor[0, :, 0:4].cpu().float().numpy() 
    pos = output_tensor[:, :, 4:7]                            
    q_6d = output_tensor[:, :, 7:].reshape(1, audio_seq_len, 24, 6) 
    
    q_ax = ax_from_6v(q_6d)
    active_smpl = SMPLSkeleton(device=engine.device)
    poses_3d = active_smpl.forward(q_ax, pos).detach().cpu().float().numpy()[0]

    video_name = f"{temp_name}_{os.path.splitext(os.path.basename(audio_file))[0]}.mp4"
    
    skeleton_render(poses_3d, epoch=temp_name, out=out_dir, name=[audio_file], sound=True, stitch=False, render=True, contact=contacts)
    
    return os.path.join(out_dir, video_name)

with gr.Blocks(title="敦煌数字编舞引擎") as app:
    gr.Markdown("# 🪷 敦煌多模态数字编舞引擎 (Digital Choreography Engine)")
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(type="filepath", label="1. 🎵 上传背景音乐 (.wav)")
            traj_in = gr.Textbox(label="2. ✍️ 绘制空间轨迹坐标 (例: 0,0; 1,2; -1,4; 0,5)", placeholder="0,0; 1.5,1; -1.5,3; 0,5")
                
            with gr.Row():
                with gr.Column():
                    img_start_in = gr.Image(type="filepath", label="3A. 🖼️ 起势姿态图 (ROMP 识别)")
                    pkl_start_in = gr.File(label="3B. 📁 或上传专家起势 (.pkl)")
                with gr.Column():
                    img_end_in = gr.Image(type="filepath", label="4A. 🖼️ 收势姿态图 (ROMP 识别)")
                    pkl_end_in = gr.File(label="4B. 📁 或上传专家收势 (.pkl)")
            with gr.Row():
                trajectory_post_mode_in = gr.Dropdown(
                    choices=list(TRAJECTORY_POST_MODES),
                    value="optimize",
                    label="轨迹后处理",
                )
                use_tto_in = gr.Checkbox(label="🔥 开启 TTO 物理运动学优化 (建议开启，如需极速预览请关闭)", value=False)
                hard_keyframe_project_in = gr.Checkbox(label="🎯 硬投影关键帧 (严格贴合起止姿态)", value=False)
                is_flying_in = gr.Checkbox(label="🕊️ 开启滞空/飞天模式 (关闭则信任 ROMP 强制贴地)", value=False)
            run_btn = gr.Button("🚀 一键生成编舞视频", variant="primary")
            with gr.Column():
                video_out = gr.Video(label="🎬 渲染结果")

    run_btn.click(
        fn=run_choreography_pipeline, 
        inputs=[
            audio_in, 
            img_start_in, 
            pkl_start_in, 
            img_end_in, 
            pkl_end_in, 
            traj_in, 
            trajectory_post_mode_in,
            use_tto_in, 
            hard_keyframe_project_in,
            is_flying_in
        ], 
        outputs=video_out
    )

if __name__ == "__main__":
    engine.load_models() 
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
