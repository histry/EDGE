import copy
import os
import pickle
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from p_tqdm import p_map

from pytorch3d.transforms import (axis_angle_to_quaternion,
                                  quaternion_to_axis_angle,
                                  rotation_6d_to_matrix)
from tqdm import tqdm

from dataset.quaternion import ax_from_6v, quat_slerp
from vis import audio_output_stem, skeleton_render

from .utils import extract, make_beta_schedule, prob_mask_like
from model.mmr_model import CrossModalMMR

def identity(t, *args, **kwargs):
    return t

# ✨ 新增：全局安全的 L2 范数，防止预测为 0 时除以 0 引发 NaN 梯度崩溃
def safe_norm(x, dim=-1, eps=1e-8):
    return torch.sqrt(torch.sum(x**2, dim=dim) + eps)

def rotation_angle_between(rot_mats):
    rel = torch.matmul(rot_mats[:, :-1].transpose(-1, -2), rot_mats[:, 1:])
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    
    # 修复：将精度从 1e-4 提升至 1e-7，彻底消除高达 24度/秒 的抖动死区
    cos_angle = torch.clamp((trace - 1.0) * 0.5, -1.0 + 1e-7, 1.0 - 1e-7)
    
    return torch.acos(cos_angle)

def move_condition_to_device(cond, device):
    if isinstance(cond, dict):
        moved = {}
        for key, value in cond.items():
            moved[key] = value.to(device) if torch.is_tensor(value) else value
        return moved
    return cond.to(device)

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        repr_dim,
        smpl,
        n_timestep=1000,
        schedule="linear",
        loss_type="l1",
        clip_denoised=True,
        predict_epsilon=True,
        guidance_weight=3,
        use_p2=False,
        cond_drop_prob=0.2,
        mmr_model=None,
        mmr_loss_weight=0.0,
        keyframe_condition_prob=0.7,
        keyframe_condition_width=3,
        keyframe_loss_weight=2.0,
        mid_keyframe_condition_prob=0.0,
        mid_keyframe_count=2,
        mid_keyframe_condition_width=1,
        mid_keyframe_selection="motion_peak",
        data_fps=30, 
        contact_loss_weight=0.8,
        foot_loss_weight=2.5,
        sync_loss_weight=1.2,
        force_audio_only_drop=False
    ):
        super().__init__()
        self.dt = 1.0 / data_fps 
        self.force_audio_only_drop = force_audio_only_drop
        self.mmr_loss_weight = mmr_loss_weight
        self.keyframe_condition_prob = keyframe_condition_prob
        self.keyframe_condition_width = keyframe_condition_width
        self.keyframe_loss_weight = keyframe_loss_weight
        self.mid_keyframe_condition_prob = mid_keyframe_condition_prob
        self.mid_keyframe_count = mid_keyframe_count
        self.mid_keyframe_condition_width = mid_keyframe_condition_width
        self.mid_keyframe_selection = mid_keyframe_selection
        self.contact_loss_weight = contact_loss_weight
        self.foot_loss_weight = foot_loss_weight
        self.sync_loss_weight = sync_loss_weight
        self.tto_interval = 50
        self.tto_steps = 1
        self.tto_lr = 0.03
        self.tto_contact_threshold = 0.65
        
        self.mmr_model = mmr_model
        if self.mmr_model is not None:
            self.mmr_model.eval()
            for param in self.mmr_model.parameters():
                param.requires_grad = False
        
        self.horizon = horizon
        self.transition_dim = repr_dim
        self.model = model
        self.ema = EMA(0.9999)
        self.master_model = copy.deepcopy(self.model)
        self.normalizer = None

        self.cond_drop_prob = cond_drop_prob
        self.smpl = smpl

        betas = torch.Tensor(
            make_beta_schedule(schedule=schedule, n_timestep=n_timestep)
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timestep = int(n_timestep)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.guidance_weight = guidance_weight

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))

        self.p2_loss_weight_k = 1
        self.p2_loss_weight_gamma = 0.5 if use_p2 else 0
        self.register_buffer(
            "p2_loss_weight",
            (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -self.p2_loss_weight_gamma,
        )

        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss
        
        # 统一定义物理空间中 Root X 和 Root Z 的特征维度索引，消除 Magic Numbers
        self.root_x_idx = 4
        self.root_z_idx = 6

    def predict_start_from_noise(self, x_t, t, noise):
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, cond, t, weight=None, clip_x_start=False, constraint=None):
        weight = weight if weight is not None else self.guidance_weight

        force_mask, force_x_clean = None, None
        if constraint is not None:
            force_mask = constraint["mask"]
            force_x_clean = constraint["value"]

        cond_input = cond
        if isinstance(cond, dict) and "trajectory" in cond:
            cond_input = cond.copy() 

        model_output = self.model.guided_forward(
            x, cond_input, t, weight, force_mask=force_mask, force_x_clean=force_x_clean
        )

        # 统一预测空间的数学语义
        if self.predict_epsilon:
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
        else:
            x_start = model_output
        
        if clip_x_start:
            if x_start.shape[-1] > 7:
                x_start = torch.cat(
                    [x_start[..., :7], x_start[..., 7:].clamp(-1.0, 1.0)],
                    dim=-1,
                )
            else:
                x_start = x_start.clamp(-1.0, 1.0)
        
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, clip_denoised=True, constraint=None):
        _, x_recon = self.model_predictions(
            x, cond, t, clip_x_start=clip_denoised, constraint=constraint
        )

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    def _should_run_tto(self, use_tto, cond, constraint, t):
        if not use_tto:
            return False

        has_traj = isinstance(cond, dict) and cond.get("trajectory", None) is not None
        has_keyframes = (
            constraint is not None
            and constraint.get("mask", None) is not None
            and constraint.get("value", None) is not None
        )
        if not has_traj and not has_keyframes:
            return False

        time_value = int(t[0].item())
        if time_value > int(self.n_timestep * 0.75) or time_value < int(self.n_timestep * 0.05):
            return False
        return time_value % max(1, int(self.tto_interval)) == 0

    def _tto_loss(self, pred_xstart, cond, constraint=None):
        loss = pred_xstart.new_tensor(0.0)
        
        if getattr(self, "normalizer", None) is not None:
            physical_xstart = self.normalizer.unnormalize(pred_xstart)
        else:
            physical_xstart = pred_xstart
            
        root_xz = physical_xstart[:, :, [self.root_x_idx, self.root_z_idx]] if physical_xstart.shape[-1] == 151 else physical_xstart[:, :, [0, 2]]

        if isinstance(cond, dict) and cond.get("trajectory", None) is not None:
            target_traj_norm = cond["trajectory"].to(device=pred_xstart.device, dtype=pred_xstart.dtype)
            if target_traj_norm.shape[1] != root_xz.shape[1]:
                target_traj_norm = F.interpolate(
                    target_traj_norm.transpose(1, 2),
                    size=root_xz.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            target_traj_norm = target_traj_norm[..., :2]
            
            normalizer = getattr(self, "normalizer", None)
            if normalizer is not None and hasattr(normalizer, "mean"):
                mean_x = target_traj_norm.new_tensor(normalizer.mean[self.root_x_idx])
                mean_z = target_traj_norm.new_tensor(normalizer.mean[self.root_z_idx])
                std_x = target_traj_norm.new_tensor(normalizer.std[self.root_x_idx])
                std_z = target_traj_norm.new_tensor(normalizer.std[self.root_z_idx])
                
                target_traj = target_traj_norm.clone()
                target_traj[..., 0] = target_traj_norm[..., 0] * std_x + mean_x
                target_traj[..., 1] = target_traj_norm[..., 1] * std_z + mean_z
            else:
                target_traj = target_traj_norm

            traj_loss = F.mse_loss(root_xz, target_traj)
            traj_velocity_loss = F.mse_loss(root_xz[:, 1:] - root_xz[:, :-1], target_traj[:, 1:] - target_traj[:, :-1])
            loss = loss + 4.0 * traj_loss + 0.5 * traj_velocity_loss

        if constraint is not None and constraint.get("mask", None) is not None and constraint.get("value", None) is not None:
            mask = constraint["mask"].to(device=pred_xstart.device, dtype=pred_xstart.dtype)
            value = constraint["value"].to(device=pred_xstart.device, dtype=pred_xstart.dtype)
            key_loss = ((pred_xstart - value) ** 2 * mask).sum() / (mask.sum() * pred_xstart.shape[-1] + 1e-6)
            loss = loss + 2.0 * key_loss

        if root_xz.shape[1] > 2:
            root_acc = root_xz[:, 2:] - 2.0 * root_xz[:, 1:-1] + root_xz[:, :-2]
            loss = loss + 0.05 * root_acc.pow(2).mean()

        if pred_xstart.shape[-1] == 151 and getattr(self, "normalizer", None) is not None:
            contacts = physical_xstart[:, :, 0:4] > self.tto_contact_threshold
            contact_pairs = contacts[:, 1:] & contacts[:, :-1]
            if bool(contact_pairs.any().item()):
                pos = physical_xstart[:, :, 4:7]
                q = ax_from_6v(physical_xstart[:, :, 7:].reshape(physical_xstart.shape[0], physical_xstart.shape[1], 24, 6))
                feet = self.smpl.forward(q, pos)[:, :, [7, 8, 10, 11], :]
                feet_delta = feet[:, 1:] - feet[:, :-1]
                foot_error = feet_delta[..., [0, 2]].pow(2).sum(dim=-1)
                loss = loss + 0.25 * foot_error[contact_pairs].mean()

        return loss

    def _apply_tto(self, x, cond, t, constraint=None):
        x_opt = x.detach()
        for _ in range(max(1, int(self.tto_steps))):
            with torch.enable_grad():
                x_opt = x_opt.detach().requires_grad_(True)
                _, pred_xstart = self.model_predictions(
                    x_opt,
                    cond,
                    t,
                    # ✨ 修复：强行关闭 TTO 阶段的截断，保证极限误差下的梯度全量回传
                    clip_x_start=False, 
                    constraint=constraint,
                )
                tto_loss = self._tto_loss(pred_xstart, cond, constraint)
                grad = torch.autograd.grad(tto_loss, x_opt, allow_unused=True)[0]
                if grad is None:
                    break
                grad = torch.nan_to_num(grad)
                grad_norm = safe_norm(grad.flatten(1), dim=1).clamp_min(1e-6)
                grad = grad / grad_norm.view(-1, *([1] * (grad.ndim - 1)))
                x_opt = x_opt - self.tto_lr * grad
        return x_opt.detach()

    def p_sample(self, x, cond, t, constraint=None, use_tto=True):
        b, *_, device = *x.shape, x.device

        if self._should_run_tto(use_tto, cond, constraint, t):
            x = self._apply_tto(x, cond, t, constraint=constraint)

        with torch.no_grad():
            model_mean, posterior_variance, model_log_variance, pred_xstart = self.p_mean_variance(
                x, cond, t, constraint=constraint
            )

            noise = torch.randn_like(model_mean)
            nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(noise.shape) - 1)))
            x_out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_out, pred_xstart

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
        use_tto=True, 
    ):
        device = self.betas.device
        start_point = self.n_timestep if start_point is None else start_point
        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = move_condition_to_device(cond, device)

        if return_diffusion:
            diffusion = [x]

        for i in tqdm(reversed(range(0, start_point))):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            x, _ = self.p_sample(x, cond, timesteps, constraint=constraint, use_tto=use_tto)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def ddim_sample(self, shape, cond, constraint=None, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 0
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x = torch.randn(shape, device=device)
        cond = move_condition_to_device(cond, device)

        for time, time_next in tqdm(time_pairs, desc='ddim sampling'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                x, cond, time_cond, clip_x_start=self.clip_denoised, constraint=constraint
            )

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)
            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
        return x

    @torch.no_grad()
    def long_ddim_sample(self, shape, cond, constraint=None, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 0

        if batch == 1:
            return self.ddim_sample(shape, cond, constraint=constraint)

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        weights = np.clip(np.linspace(0, self.guidance_weight * 2, sampling_timesteps), None, self.guidance_weight)
        time_pairs = list(zip(times[:-1], times[1:], weights))

        x = torch.randn(shape, device=device)
        cond = move_condition_to_device(cond, device)

        assert batch > 1
        assert x.shape[1] % 2 == 0
        half = x.shape[1] // 2

        x_start = None

        for time, time_next, weight in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                x, cond, time_cond, weight=weight, clip_x_start=self.clip_denoised, constraint=constraint
            )

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            if time > 0:
                x[1:, :half] = x[:-1, half:]
        return x

    @torch.no_grad()
    def inpaint_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
        use_tto=True, 
    ):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = move_condition_to_device(cond, device)
        if return_diffusion:
            diffusion = [x]

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            x, _ = self.p_sample(x, cond, timesteps, constraint=constraint, use_tto=use_tto)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def long_inpaint_loop(
        self, shape, cond, noise=None, constraint=None, return_diffusion=False, start_point=None, use_tto=True, 
    ):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = move_condition_to_device(cond, device)
        if return_diffusion:
            diffusion = [x]

        assert x.shape[1] % 2 == 0
        if batch_size == 1:
            return self.inpaint_loop(
                shape, cond, noise=noise, constraint=constraint, return_diffusion=return_diffusion, start_point=start_point,
                use_tto=use_tto,
            )
        assert batch_size > 1
        half = x.shape[1] // 2

        start_point = self.n_timestep if start_point is None else start_point

        for i in tqdm(reversed(range(0, start_point))):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            x, _ = self.p_sample(x, cond, timesteps, constraint=constraint, use_tto=use_tto)

            if i > 0:
                x[1:, :half] = x[:-1, half:].clone()

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, shape, cond, constraint=None, *args, horizon=None, **kwargs):
        device = self.betas.device
        horizon = horizon or self.horizon
        return self.p_sample_loop(shape, cond, constraint=constraint, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def _normalize_keyframe_scores(self, scores):
        scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
        scores = scores - scores.amin(dim=1, keepdim=True)
        denom = scores.amax(dim=1, keepdim=True).clamp_min(1e-6)
        return scores / denom

    def _middle_keyframe_scores(self, x_start, cond):
        b, s, c = x_start.shape
        device = x_start.device
        mode = self.mid_keyframe_selection
        score_parts = []

        if mode in ["motion_peak", "mixed"]:
            motion_feat = x_start[..., 4:] if c == 151 else x_start
            motion_delta = torch.zeros((b, s), device=device, dtype=torch.float32)
            if s > 1:
                motion_diff = motion_feat[:, 1:].float() - motion_feat[:, :-1].float()
                motion_delta[:, 1:] = safe_norm(motion_diff, dim=-1)
                motion_delta[:, 0] = motion_delta[:, 1]

            kernel = min(5, s)
            if kernel % 2 == 0:
                kernel -= 1
            if kernel > 1:
                motion_delta = F.avg_pool1d(
                    motion_delta.unsqueeze(1),
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2,
                ).squeeze(1)
            score_parts.append(self._normalize_keyframe_scores(motion_delta))

        if mode in ["audio_onset", "mixed"]:
            audio_feat = cond.get("audio", None) if isinstance(cond, dict) else None
            if audio_feat is not None:
                audio_feat = audio_feat.to(device=device).float()
                if audio_feat.shape[1] != s:
                    audio_feat = F.interpolate(
                        audio_feat.transpose(1, 2),
                        size=s,
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)

                if audio_feat.shape[-1] > 768:
                    audio_score = audio_feat[..., 768]
                else:
                    audio_score = torch.zeros((b, s), device=device, dtype=torch.float32)
                    if s > 1:
                        audio_diff = audio_feat[:, 1:] - audio_feat[:, :-1]
                        audio_score[:, 1:] = safe_norm(audio_diff, dim=-1)
                        audio_score[:, 0] = audio_score[:, 1]
                score_parts.append(self._normalize_keyframe_scores(audio_score))

        if mode == "random" or not score_parts:
            return torch.rand((b, s), device=device, dtype=torch.float32)

        return torch.stack(score_parts, dim=0).mean(dim=0)

    def _build_keyframe_condition(self, x_start, cond=None):
        b, s, _ = x_start.shape
        device = x_start.device
        dtype = x_start.dtype

        force_mask = torch.zeros((b, s, 1), device=device, dtype=dtype)

        if self.keyframe_condition_prob > 0:
            use_endpoint = torch.rand((b,), device=device) < self.keyframe_condition_prob
            if bool(use_endpoint.any().item()):
                width = max(1, int(self.keyframe_condition_width))
                width = min(width, s)
                force_mask[use_endpoint, :width, :] = 1.0
                force_mask[use_endpoint, -width:, :] = 1.0

        if self.mid_keyframe_condition_prob > 0 and self.mid_keyframe_count > 0 and s > 2:
            use_middle = torch.rand((b,), device=device) < self.mid_keyframe_condition_prob
            if bool(use_middle.any().item()):
                scores = self._middle_keyframe_scores(x_start, cond)
                middle_width = max(1, int(self.mid_keyframe_condition_width))
                middle_width = min(middle_width, s)
                max_middle = max(1, int(self.mid_keyframe_count))
                margin = max(4, int(self.keyframe_condition_width) + middle_width)
                if s <= 2 * margin:
                    margin = max(1, s // 4)
                min_distance = max(middle_width * 2 + 1, s // 10)

                for batch_idx in range(b):
                    if not bool(use_middle[batch_idx].item()):
                        continue

                    num_middle = int(
                        torch.randint(1, max_middle + 1, (1,), device=device).item()
                    )
                    candidate_scores = scores[batch_idx].clone()
                    candidate_scores[:margin] = float("-inf")
                    candidate_scores[s - margin :] = float("-inf")
                    candidate_scores[force_mask[batch_idx, :, 0] > 0.5] = float("-inf")

                    for _ in range(num_middle):
                        finite_mask = torch.isfinite(candidate_scores)
                        if not bool(finite_mask.any().item()):
                            break

                        valid_count = int(finite_mask.sum().item())
                        pool_size = min(max(4, max_middle * 4), valid_count)
                        top_candidates = torch.topk(candidate_scores, k=pool_size).indices
                        random_offset = int(
                            torch.randint(pool_size, (1,), device=device).item()
                        )
                        center = int(top_candidates[random_offset].item())

                        start = max(0, center - middle_width // 2)
                        end = min(s, start + middle_width)
                        force_mask[batch_idx, start:end, 0] = 1.0

                        suppress_start = max(0, center - min_distance)
                        suppress_end = min(s, center + min_distance + 1)
                        candidate_scores[suppress_start:suppress_end] = float("-inf")

        if not bool((force_mask > 0).any().item()):
            return None, None

        return force_mask, x_start.detach()

    def p_losses(self, x_start, cond, t, current_epoch=None):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        cond_input = cond 

        b, s, c = x_start.shape
        
        rand_probs = torch.rand((b,), device=x_start.device)
        
        keep_audio_mask = torch.ones((b,), dtype=torch.bool, device=x_start.device)
        keep_traj_mask = torch.ones((b,), dtype=torch.bool, device=x_start.device)
        
        if self.cond_drop_prob > 0:
            p_uncond = self.cond_drop_prob * 0.4
            p_drop_audio = self.cond_drop_prob * 0.7
            p_drop_traj = self.cond_drop_prob
            
            uncond_mask = rand_probs < p_uncond
            drop_audio_mask = (rand_probs >= p_uncond) & (rand_probs < p_drop_audio)
            drop_traj_mask = (rand_probs >= p_drop_audio) & (rand_probs < p_drop_traj)
            
            keep_audio_mask[uncond_mask | drop_audio_mask] = False
            keep_traj_mask[uncond_mask | drop_traj_mask] = False

        if getattr(self, "force_audio_only_drop", False):
            keep_audio_mask[:] = False
            keep_traj_mask[:] = True

        force_mask, force_x_clean = self._build_keyframe_condition(x_start, cond_input)

        x_recon = self.model(
            x_noisy,
            cond_input,
            t,
            cond_drop_prob=0.0, 
            force_mask=force_mask,
            force_x_clean=force_x_clean,
            keep_audio_mask=keep_audio_mask,
            keep_traj_mask=keep_traj_mask  
        )

        assert noise.shape == x_recon.shape

        model_out = x_recon
        if self.predict_epsilon:
            target = noise
        else:
            target = x_start

        loss = self.loss_fn(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        keyframe_loss = torch.tensor(0.0, device=x_start.device)
        if force_mask is not None:
            pred_x0_for_key = self.predict_start_from_noise(x_noisy, t, model_out) if self.predict_epsilon else model_out
            keyframe_error = self.loss_fn(pred_x0_for_key, x_start, reduction="none") * force_mask
            keyframe_loss = keyframe_error.sum() / (
                force_mask.sum() * model_out.shape[-1] + 1e-6
            )

        model_motion_x0 = self.predict_start_from_noise(x_noisy, t, model_out) if self.predict_epsilon else model_out
        target_motion_x0 = x_start

        if model_out.shape[2] == 381:
            target_v = target_motion_x0[:, 1:] - target_motion_x0[:, :-1]
            model_out_v = model_motion_x0[:, 1:] - model_motion_x0[:, :-1]
            v_loss = self.loss_fn(model_out_v, target_v, reduction="none")
            v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
            v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

            zero = torch.tensor(0.0, device=x_start.device)
            losses = (
                1.0 * loss.mean(),
                3.0 * v_loss.mean(),
                zero,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                self.keyframe_loss_weight * keyframe_loss,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
            )
            return sum(losses), losses
        else:
            _, model_out_main_x0 = torch.split(model_motion_x0, (4, model_motion_x0.shape[2] - 4), dim=2)
            _, target_main_x0 = torch.split(target_motion_x0, (4, target_motion_x0.shape[2] - 4), dim=2)

            target_v = target_main_x0[:, 1:] - target_main_x0[:, :-1]
            model_out_v = model_out_main_x0[:, 1:] - model_out_main_x0[:, :-1]
            v_loss = self.loss_fn(model_out_v, target_v, reduction="none")
            v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
            v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

            normalizer = getattr(self, "normalizer", None)
            if normalizer is not None and hasattr(normalizer, "mean"):
                model_motion_phys = normalizer.unnormalize(model_motion_x0)
                target_motion_phys = normalizer.unnormalize(target_motion_x0)
            else:
                model_motion_phys = model_motion_x0
                target_motion_phys = target_motion_x0

            model_contact, model_out_main = torch.split(model_motion_phys, (4, model_motion_phys.shape[2] - 4), dim=2)
            target_contact, target_main = torch.split(target_motion_phys, (4, target_motion_phys.shape[2] - 4), dim=2)

            model_x = model_out_main[:, :, :3]
            model_q = ax_from_6v(model_out_main[:, :, 3:].reshape(b, s, -1, 6))
            target_x = target_main[:, :, :3]
            target_q = ax_from_6v(target_main[:, :, 3:].reshape(b, s, -1, 6))

            model_contact_phys = model_contact
            target_contact_phys = target_contact.clamp(0.0, 1.0)

            contact_loss = self.loss_fn(model_contact_phys, target_contact_phys, reduction="none")
            contact_loss = contact_loss.mean(dim=(1, 2))
            contact_loss = contact_loss * extract(self.p2_loss_weight, t, contact_loss.shape)

            model_root_rot = rotation_6d_to_matrix(model_out_main[:, :, 3:9])
            target_root_rot = rotation_6d_to_matrix(target_main[:, :, 3:9])
            model_ang_v = rotation_angle_between(model_root_rot) / self.dt
            target_ang_v = rotation_angle_between(target_root_rot) / self.dt

            turn_velocity_loss = self.loss_fn(model_ang_v, target_ang_v, reduction="none")
            turn_velocity_loss = turn_velocity_loss.mean(dim=1)
            turn_velocity_loss = turn_velocity_loss * extract(self.p2_loss_weight, t, turn_velocity_loss.shape)

            if s > 2:
                model_ang_acc = model_ang_v[:, 1:] - model_ang_v[:, :-1]
                target_ang_acc = target_ang_v[:, 1:] - target_ang_v[:, :-1]
                turn_acc_loss = self.loss_fn(model_ang_acc, target_ang_acc, reduction="none")
                turn_acc_loss = turn_acc_loss.mean(dim=1)
                turn_acc_loss = turn_acc_loss * extract(self.p2_loss_weight, t, turn_acc_loss.shape)
            else:
                turn_acc_loss = turn_velocity_loss.new_zeros(turn_velocity_loss.shape)
            turn_smooth_loss = turn_velocity_loss + 0.35 * turn_acc_loss

            chunk_len = min(30, s) 
            start_idx = torch.randint(0, s - chunk_len + 1, (1,), device=x_start.device).item()
            sub_s_idx = torch.arange(start_idx, start_idx + chunk_len, device=x_start.device)
            
            sparse_step = (s + 29) // 30 
            global_s_idx = torch.arange(0, s, sparse_step, device=x_start.device)

            model_q_sub = model_q[:, sub_s_idx]
            model_x_sub = model_x[:, sub_s_idx]
            target_q_sub = target_q[:, sub_s_idx]
            target_x_sub = target_x[:, sub_s_idx]
            t_sub = t

            model_xp = self.smpl.forward(model_q_sub, model_x_sub)
            target_xp = self.smpl.forward(target_q_sub, target_x_sub)
            
            model_xp_global = self.smpl.forward(model_q[:, global_s_idx], model_x[:, global_s_idx])
            target_xp_global = self.smpl.forward(target_q[:, global_s_idx], target_x[:, global_s_idx])

            model_feet = model_xp[:, :, [7, 8, 10, 11], :]
            
            pelvis_global = model_xp_global[:, :, 0, :]
            spine_global = model_xp_global[:, :, 6, :]
            neck_global = model_xp_global[:, :, 12, :]
            target_pelvis_global = target_xp_global[:, :, 0, :]
            target_neck_global = target_xp_global[:, :, 12, :]
                
            vec_lower_global = spine_global - pelvis_global
            vec_upper_global = neck_global - spine_global

            model_torso = F.normalize(neck_global - pelvis_global, dim=-1, eps=1e-6)
            target_torso = F.normalize(target_neck_global - target_pelvis_global, dim=-1, eps=1e-6)
            torso_dir_loss_raw = 1.0 - F.cosine_similarity(model_torso, target_torso, dim=-1)

            model_height = model_xp_global[..., 1].amax(dim=2) - model_xp_global[..., 1].amin(dim=2)
            target_height = target_xp_global[..., 1].amax(dim=2) - target_xp_global[..., 1].amin(dim=2)
            height_loss_raw = self.loss_fn(model_height, target_height, reduction="none")

            model_span = safe_norm(model_xp_global.amax(dim=2) - model_xp_global.amin(dim=2), dim=-1)
            target_span = safe_norm(target_xp_global.amax(dim=2) - target_xp_global.amin(dim=2), dim=-1)
            span_loss_raw = self.loss_fn(model_span, target_span, reduction="none")

            body_stability_raw = (
                torso_dir_loss_raw
                + 0.35 * height_loss_raw
                + 0.25 * span_loss_raw
            )
            body_stability_loss = body_stability_raw.mean(dim=1)
            
            if chunk_len > 2:
                model_root_y_acc = model_x_sub[:, 2:, 1] - 2.0 * model_x_sub[:, 1:-1, 1] + model_x_sub[:, :-2, 1]
                target_root_y_acc = target_x_sub[:, 2:, 1] - 2.0 * target_x_sub[:, 1:-1, 1] + target_x_sub[:, :-2, 1]
                root_y_acc_loss = self.loss_fn(model_root_y_acc, target_root_y_acc, reduction="none")
                root_y_acc_loss = root_y_acc_loss.mean(dim=1)
                body_stability_loss = body_stability_loss + 0.15 * root_y_acc_loss
            body_stability_loss = body_stability_loss * extract(self.p2_loss_weight, t_sub, body_stability_loss.shape)

            energy_joints = [4, 5, 7, 8, 10, 11, 18, 19, 20, 21]
            model_energy_v = (model_xp[:, 1:, energy_joints] - model_xp[:, :-1, energy_joints]) / self.dt
            target_energy_v = (target_xp[:, 1:, energy_joints] - target_xp[:, :-1, energy_joints]) / self.dt
            model_energy = safe_norm(model_energy_v, dim=-1)
            target_energy = safe_norm(target_energy_v, dim=-1)
            motion_energy_loss = self.loss_fn(model_energy, target_energy, reduction="none")
            motion_energy_loss = motion_energy_loss.mean(dim=(1, 2))
            motion_energy_loss = motion_energy_loss * extract(self.p2_loss_weight, t_sub, motion_energy_loss.shape)

            biomech_weight = 0.0
            if current_epoch is not None:
                biomech_start = 100.0
                biomech_duration = 100.0
                if current_epoch > biomech_start:
                    biomech_weight = min(1.0, (current_epoch - biomech_start) / biomech_duration)
            else:
                biomech_weight = 1.0
            
            scurve_loss = torch.tensor(0.0, device=x_start.device)
            hunchback_loss = torch.tensor(0.0, device=x_start.device)
            asymmetry_loss = torch.tensor(0.0, device=x_start.device)

            if biomech_weight > 0.0:
                vec_lower_yz = vec_lower_global[..., [1, 2]]
                vec_upper_yz = vec_upper_global[..., [1, 2]]
                cos_sim_yz = torch.nn.functional.cosine_similarity(vec_lower_yz, vec_upper_yz, dim=-1)
                
                loss_too_straight = F.relu(cos_sim_yz - 0.985)
                loss_too_bent = F.relu(0.866 - cos_sim_yz)
                scurve_loss_raw = loss_too_straight + loss_too_bent
                
                hunchback_penalty = F.relu(-vec_lower_global[..., 2] - 0.05) + F.relu(-vec_upper_global[..., 2] - 0.05)
                scoliosis_penalty = torch.abs(vec_lower_global[..., 0]) + torch.abs(vec_upper_global[..., 0])
                hunchback_loss_raw = hunchback_penalty + scoliosis_penalty
                
                l_wrist = model_xp_global[:, :, 20, :]
                r_wrist = model_xp_global[:, :, 21, :]
                dist_wrists = safe_norm(l_wrist - r_wrist, dim=-1)
                asymmetry_loss_raw = F.relu(0.02 - dist_wrists)
                
                if force_mask is not None:
                    mask_inv = (1.0 - force_mask[:, global_s_idx, 0])
                else:
                    mask_inv = torch.ones_like(scurve_loss_raw)
                    
                scurve_loss_raw = scurve_loss_raw * mask_inv
                hunchback_loss_raw = hunchback_loss_raw * mask_inv
                asymmetry_loss_raw = asymmetry_loss_raw * mask_inv
                    
                scurve_loss = reduce(scurve_loss_raw, "b ... -> b (...)", "mean")
                scurve_loss = scurve_loss * extract(self.p2_loss_weight, t_sub, scurve_loss.shape) * biomech_weight
                
                hunchback_loss = reduce(hunchback_loss_raw, "b ... -> b (...)", "mean")
                hunchback_loss = hunchback_loss * extract(self.p2_loss_weight, t_sub, hunchback_loss.shape) * biomech_weight
                
                asymmetry_loss = reduce(asymmetry_loss_raw, "b ... -> b (...)", "mean")
                asymmetry_loss = asymmetry_loss * extract(self.p2_loss_weight, t_sub, asymmetry_loss.shape) * biomech_weight

            fk_loss = self.loss_fn(model_xp, target_xp, reduction="none")
            fk_loss = reduce(fk_loss, "b ... -> b (...)", "mean")
            fk_loss = fk_loss * extract(self.p2_loss_weight, t_sub, fk_loss.shape)

            model_contact_sub = model_contact_phys[:, sub_s_idx].clamp(0.0, 1.0)
            target_contact_sub = target_contact_phys[:, sub_s_idx].clamp(0.0, 1.0)
            static_idx = ((model_contact_sub > 0.5) | (target_contact_sub > 0.5)).detach()
            
            static_continuous = static_idx[:, :-1, :] & static_idx[:, 1:, :]
            
            foot_dist_vec = model_feet[:, 1:] - model_feet[:, :-1]
            foot_dist_norm = safe_norm(foot_dist_vec, dim=-1)
            slide_penalty = F.relu(foot_dist_norm - 0.01)
            
            static_weight = static_continuous.float()
            foot_loss_raw = slide_penalty * static_weight
            
            static_weight_sum = static_weight.sum(dim=(1, 2))
            valid_mask = (static_weight_sum > 0).float() 
            
            foot_loss = foot_loss_raw.sum(dim=(1, 2)) / static_weight_sum.clamp_min(1.0)
            foot_loss = foot_loss * valid_mask           
            
            foot_loss = foot_loss * extract(self.p2_loss_weight, t_sub, foot_loss.shape)

            model_extremities = model_xp[:, :, [20, 21], :] 
            model_extremities_v = (model_extremities[:, 1:] - model_extremities[:, :-1]) / self.dt
            model_v_norm = safe_norm(model_extremities_v, dim=-1).mean(dim=(1, 2))
            
            base_threshold = 0.005
            dynamic_threshold = torch.full_like(model_v_norm, base_threshold)
            
            audio_feat = cond.get("audio", None) if isinstance(cond, dict) else None
            if audio_feat is not None:
                audio_feat = audio_feat.to(device=x_start.device).float()
                if audio_feat.shape[1] != s:
                    audio_feat = F.interpolate(
                        audio_feat.transpose(1, 2), size=s, mode="linear", align_corners=False
                    ).transpose(1, 2)
            
            if audio_feat is not None and audio_feat.shape[-1] > 768:
                onset = audio_feat[:, sub_s_idx[:-1], 768]
                onset_mean = onset.mean(dim=-1, keepdim=True)
                active_ratio = (onset > (0.5 * onset_mean)).float().mean(dim=-1)
                
                raw_dynamic_threshold = base_threshold * torch.clamp(active_ratio, min=0.25)
                dynamic_threshold = torch.where(
                    keep_audio_mask, 
                    raw_dynamic_threshold, 
                    torch.full_like(raw_dynamic_threshold, base_threshold)
                )

            anti_freeze_loss = F.relu(dynamic_threshold - model_v_norm)
            anti_freeze_loss = anti_freeze_loss * extract(self.p2_loss_weight, t_sub, anti_freeze_loss.shape)
            
            root_xz = model_xp[:, :, 0, [0, 2]]
            root_v = safe_norm(root_xz[:, 1:] - root_xz[:, :-1], dim=-1) / self.dt
            
            feet_xz = model_feet[:, :, :, [0, 2]]
            feet_rel = feet_xz - root_xz.unsqueeze(2)
            feet_rel_v = safe_norm(feet_rel[:, 1:] - feet_rel[:, :-1], dim=-1) / self.dt
            max_leg_swing = torch.max(feet_rel_v, dim=2)[0]
            
            grounded_mask = (static_idx[:, :-1, :].sum(dim=-1) >= 2).float()
            sync_loss_raw = F.relu(root_v - max_leg_swing - 0.5) * grounded_mask
            
            sync_loss = reduce(sync_loss_raw, "b ... -> b (...)", "mean")
            sync_loss = sync_loss * extract(self.p2_loss_weight, t_sub, sync_loss.shape)

            model_ang_v_sub = model_ang_v[:, sub_s_idx[:-1]]
            target_ang_v_sub = target_ang_v[:, sub_s_idx[:-1]]
            contact_turn_raw = self.loss_fn(model_ang_v_sub, target_ang_v_sub, reduction="none") * grounded_mask
            contact_turn_loss = contact_turn_raw.sum(dim=-1) / (grounded_mask.sum(dim=-1) + 1e-6)
            contact_turn_loss = contact_turn_loss * extract(self.p2_loss_weight, t_sub, contact_turn_loss.shape)

            mmr_loss = torch.tensor(0.0, device=x_start.device)
            if audio_feat is not None and self.mmr_model is not None and self.mmr_loss_weight > 0:
                valid_audio_mask = keep_audio_mask.float()
                if valid_audio_mask.bool().any():
                    physical_out = model_motion_phys
                    
                    with torch.no_grad():
                        audio_latent = self.mmr_model.encode_audio(audio_feat)
                    motion_latent = self.mmr_model.encode_motion(physical_out)
                    raw_mmr_loss = 1.0 - F.cosine_similarity(motion_latent, audio_latent, dim=-1) 
                    
                    progress = 1.0 - (t.float() / self.n_timestep)
                    dynamic_mmr_weight = 4.0 * progress * (1.0 - progress)
                    
                    weighted_mmr_loss = (
                        raw_mmr_loss 
                        * extract(self.p2_loss_weight, t, raw_mmr_loss.shape).view(-1) 
                        * dynamic_mmr_weight
                    )
                    mmr_loss = (weighted_mmr_loss * valid_audio_mask).sum() / (valid_audio_mask.sum() + 1e-6)

            traj_loss = torch.tensor(0.0, device=x_start.device)
            raw_target_traj = cond.get("trajectory", None) if isinstance(cond, dict) else None
            
            if raw_target_traj is not None:
                target_traj_norm = raw_target_traj.to(device=model_out_main.device, dtype=model_out_main.dtype)
                
                if target_traj_norm.shape[1] != s:
                    target_traj_norm = F.interpolate(
                        target_traj_norm.transpose(1, 2), size=s, mode="linear", align_corners=False
                    ).transpose(1, 2)
                    
                normalizer = getattr(self, "normalizer", None)
                if normalizer is not None and hasattr(normalizer, "mean"):
                    mean_x = target_traj_norm.new_tensor(normalizer.mean[self.root_x_idx])
                    mean_z = target_traj_norm.new_tensor(normalizer.mean[self.root_z_idx])
                    std_x = target_traj_norm.new_tensor(normalizer.std[self.root_x_idx])
                    std_z = target_traj_norm.new_tensor(normalizer.std[self.root_z_idx])
                    
                    target_traj = target_traj_norm.clone()
                    target_traj[..., 0] = target_traj_norm[..., 0] * std_x + mean_x
                    target_traj[..., 1] = target_traj_norm[..., 1] * std_z + mean_z
                else:
                    target_traj = target_traj_norm
                
                pred_traj = model_out_main[:, :, [0, 2]]
                
                pos_traj_loss = self.loss_fn(pred_traj, target_traj, reduction="none").mean(dim=(1, 2))
                vel_traj_loss = self.loss_fn(
                    pred_traj[:, 1:] - pred_traj[:, :-1],
                    target_traj[:, 1:] - target_traj[:, :-1],
                    reduction="none",
                ).mean(dim=(1, 2))

                if s > 2:
                    pred_acc = pred_traj[:, 2:] - 2.0 * pred_traj[:, 1:-1] + pred_traj[:, :-2]
                    target_acc = target_traj[:, 2:] - 2.0 * target_traj[:, 1:-1] + target_traj[:, :-2]
                    acc_traj_loss = self.loss_fn(pred_acc, target_acc, reduction="none").mean(dim=(1, 2))
                else:
                    acc_traj_loss = vel_traj_loss.new_zeros(vel_traj_loss.shape)
                
                anchor_indices = [0, s // 2, s - 1] 
                pred_anchors = pred_traj[:, anchor_indices, :]
                target_anchors = target_traj[:, anchor_indices, :]
                anchor_loss = F.l1_loss(pred_anchors, target_anchors, reduction="none").mean(dim=(1, 2))
                
                raw_traj_loss = (
                    pos_traj_loss
                    + 0.5 * vel_traj_loss
                    + 0.2 * acc_traj_loss
                    + 0.5 * anchor_loss
                )
                p2_weight = extract(self.p2_loss_weight, t, raw_traj_loss.shape).view(-1)
                weighted_traj_loss = raw_traj_loss * p2_weight
                
                valid_traj_mask = keep_traj_mask.float()
                if valid_traj_mask.bool().any():
                    traj_loss = (weighted_traj_loss * valid_traj_mask).sum() / (valid_traj_mask.sum() + 1e-6)

        warmup_epochs = 10.0
        if current_epoch is not None:
            physics_weight = min(1.0, current_epoch / warmup_epochs)
        else:
            physics_weight = 1.0 

        mmr_target_weight = self.mmr_loss_weight
        mmr_warmup_epochs = 20.0
        if current_epoch is not None:
            mmr_weight = mmr_target_weight * min(1.0, current_epoch / mmr_warmup_epochs)
        else:
            mmr_weight = mmr_target_weight

        turn_loss = turn_smooth_loss.mean()
        contact_turn = contact_turn_loss.mean()
        body_stability = body_stability_loss.mean()
        motion_energy = motion_energy_loss.mean()

        losses = (
            5.0 * loss.mean(),        
            0.5 * v_loss.mean(),      
            self.contact_loss_weight * contact_loss.mean(),
            1.0 * fk_loss.mean(),     
            self.foot_loss_weight * physics_weight * foot_loss.mean(),  
            0.01 * physics_weight * anti_freeze_loss.mean(),  
            mmr_weight * mmr_loss,   
            1.0 * traj_loss,
            self.keyframe_loss_weight * keyframe_loss,
            self.sync_loss_weight * physics_weight * sync_loss.mean(),
            0.1 * physics_weight * scurve_loss.mean() + 0.1 * physics_weight * hunchback_loss.mean() + 0.1 * physics_weight * asymmetry_loss.mean(),
            0.08 * turn_loss,
            0.04 * physics_weight * contact_turn,
            0.05 * physics_weight * body_stability,
            0.03 * motion_energy,
        )
        return sum(losses), losses

    def loss(self, x, cond, t_override=None, current_epoch=None):
        batch_size = len(x)
        if t_override is None:
            t = torch.randint(0, self.n_timestep, (batch_size,), device=x.device).long()
        else:
            t = torch.full((batch_size,), t_override, device=x.device).long()
        return self.p_losses(x, cond, t, current_epoch)

    def forward(self, x, cond, t_override=None, current_epoch=None):
        return self.loss(x, cond, t_override, current_epoch)

    def partial_denoise(self, x, cond, t):
        x_noisy = self.noise_to_t(x, t)
        return self.p_sample_loop(x.shape, cond, noise=x_noisy, start_point=t)

    def noise_to_t(self, x, timestep):
        batch_size = len(x)
        t = torch.full((batch_size,), timestep, device=x.device).long()
        return self.q_sample(x, t) if timestep > 0 else x

    def render_sample(
        self,
        shape,
        cond,
        normalizer,
        epoch,
        render_out,
        fk_out=None,
        name=None,
        sound=True,
        mode="normal",
        noise=None,
        constraint=None,
        sound_folder="ood_sliced",
        start_point=None,
        render=True,
        target_frames=None,
        use_tto=True, 
    ):
        if isinstance(shape, tuple):
            if mode == "inpaint":
                func_class = self.inpaint_loop
            elif mode == "normal":
                func_class = self.ddim_sample
            elif mode == "long":
                func_class = self.long_ddim_sample
            elif mode == "long_inpaint":
                func_class = self.long_inpaint_loop
            else:
                assert False, "Unrecognized inference mode"
            samples = (
                func_class(
                    shape,
                    cond,
                    noise=noise,
                    constraint=constraint,
                    start_point=start_point,
                    use_tto=use_tto  
                )
                .detach()
                .cpu()
            )
        else:
            samples = shape

        samples = normalizer.unnormalize(samples)

        if samples.shape[2] == 151:
            sample_contact, samples = torch.split(
                samples, (4, samples.shape[2] - 4), dim=2
            )
        else:
            sample_contact = None

        target_device = self.betas.device
        if isinstance(cond, dict):
            for value in cond.values():
                if torch.is_tensor(value):
                    target_device = value.device
                    break
        elif torch.is_tensor(cond):
            target_device = cond.device

        b, s, c = samples.shape
        pos = samples[:, :, :3].to(target_device)
        q = samples[:, :, 3:].reshape(b, s, 24, 6)
        q = ax_from_6v(q).to(target_device)

        if mode in ["long", "long_inpaint"]:
            b, s, c1, c2 = q.shape
            assert s % 2 == 0
            half = s // 2
            if b > 1:
                for i in range(1, b):
                    delta_x = pos[i-1, half, 0] - pos[i, 0, 0]
                    delta_z = pos[i-1, half, 2] - pos[i, 0, 2]
                    pos[i, :, 0] += delta_x
                    pos[i, :, 2] += delta_z

                fade_out = torch.ones((1, s, 1)).to(pos.device)
                fade_in = torch.ones((1, s, 1)).to(pos.device)
                fade_out[:, half:, :] = torch.linspace(1, 0, half)[None, :, None].to(
                    pos.device
                )
                fade_in[:, :half, :] = torch.linspace(0, 1, half)[None, :, None].to(
                    pos.device
                )

                pos[:-1] *= fade_out
                pos[1:] *= fade_in

                full_pos = torch.zeros((s + half * (b - 1), 3)).to(pos.device)
                idx = 0
                for pos_slice in pos:
                    full_pos[idx : idx + s] += pos_slice
                    idx += half

                slerp_weight = torch.linspace(0, 1, half)[None, :, None].to(pos.device)

                left, right = q[:-1, half:], q[1:, :half]
                left, right = (
                    axis_angle_to_quaternion(left),
                    axis_angle_to_quaternion(right),
                )
                merged = quat_slerp(left, right, slerp_weight)
                merged = quaternion_to_axis_angle(merged)

                full_q = torch.zeros((s + half * (b - 1), c1, c2)).to(pos.device)
                full_q[:half] += q[0, :half]
                idx = half
                for q_slice in merged:
                    full_q[idx : idx + half] += q_slice
                    idx += half
                full_q[idx : idx + half] += q[-1, half:]

                full_pos = full_pos.unsqueeze(0)
                full_q = full_q.unsqueeze(0)
            else:
                full_pos = pos
                full_q = q
                
            if target_frames is not None:
                full_pos = full_pos[:, :target_frames, :]
                full_q = full_q[:, :target_frames, :, :]

            full_pose = (
                self.smpl.forward(full_q, full_pos).detach().cpu().numpy()
            )

            try:
                pelvis = full_pose[0][:, 0, :]
                spine = full_pose[0][:, 6, :]
                neck = full_pose[0][:, 12, :]
                
                vec_lower = spine - pelvis
                vec_upper = neck - spine
                cos_sim = np.sum(vec_lower * vec_upper, axis=-1) / (np.linalg.norm(vec_lower, axis=-1) * np.linalg.norm(vec_upper, axis=-1) + 1e-6)
                curve_score = np.mean(1.0 - cos_sim) * 1000 
                
                l_wrist = full_pose[0][:, 20, :]
                r_wrist = full_pose[0][:, 21, :]
                asymmetry_score = np.mean(np.linalg.norm(l_wrist - r_wrist, axis=-1)) * 50
                
                dss_score = np.clip((curve_score * 0.6) + (asymmetry_score * 0.4), 0, 100)
                
                print(f"\n✨ [评估报告] {name[0] if name else '当前生成'} 视频渲染完成！")
                print(f"   ▶ 脊柱柔韧度 (S-Curve): {curve_score:.2f}")
                print(f"   ▶ 肢体不对称律 (Asymmetry): {asymmetry_score:.2f}")
                print(f"   🏆 综合敦煌舞姿得分 (DSS): {dss_score:.2f} / 100.00\n")
            except Exception as e:
                print(f"打分系统跳过: {e}")
            
            skeleton_render(
                full_pose[0],
                epoch=f"{epoch}",
                out=render_out,
                name=name,
                sound=sound,
                stitch=True,
                sound_folder=sound_folder,
                render=render
            )
            if fk_out is not None:
                outname = f"{epoch}_{audio_output_stem(name)}.pkl"
                Path(fk_out).mkdir(parents=True, exist_ok=True)
                
                target_traj_np = None
                if isinstance(cond, dict) and "trajectory" in cond:
                    target_traj_np = cond["trajectory"][0].detach().cpu().numpy()
                
                pickle.dump(
                    {
                        "smpl_poses": full_q.squeeze(0).reshape((-1, 72)).cpu().numpy(),
                        "smpl_trans": full_pos.squeeze(0).cpu().numpy(),
                        "full_pose": full_pose[0],
                        "target_trajectory": target_traj_np, 
                    },
                    open(os.path.join(fk_out, outname), "wb"),
                )
            return

        poses = self.smpl.forward(q, pos).detach().cpu().numpy()

        sample_contact = (
            sample_contact.detach().cpu().numpy()
            if sample_contact is not None
            else None
        )

        def inner(xx):
            num, pose = xx
            filename = name[num] if name is not None else None
            contact = sample_contact[num] if sample_contact is not None else None
            skeleton_render(
                pose,
                epoch=f"e{epoch}_b{num}",
                out=render_out,
                name=filename,
                sound=sound,
                contact=contact,
            )

        p_map(inner, enumerate(poses))

        if fk_out is not None and mode != "long":
            Path(fk_out).mkdir(parents=True, exist_ok=True)
            for num, (qq, pos_, filename, pose) in enumerate(zip(q, pos, name, poses)):
                path = os.path.normpath(filename)
                pathparts = path.split(os.sep)
                pathparts[-1] = pathparts[-1].replace("npy", "wav")
                pathparts[2] = "wav_sliced"
                audioname = os.path.join(*pathparts)
                outname = f"{epoch}_{num}_{pathparts[-1][:-4]}.pkl"
                pickle.dump(
                    {
                        "smpl_poses": qq.reshape((-1, 72)).cpu().numpy(),
                        "smpl_trans": pos_.cpu().numpy(),
                        "full_pose": pose,
                    },
                    open(f"{fk_out}/{outname}", "wb"),
                )