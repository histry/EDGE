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
                                  quaternion_to_axis_angle)
from tqdm import tqdm

from dataset.quaternion import ax_from_6v, quat_slerp
from vis import skeleton_render

from .utils import extract, make_beta_schedule


def identity(t, *args, **kwargs):
    return t


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
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = repr_dim
        self.model = model
        self.ema = EMA(0.9999)
        self.master_model = copy.deepcopy(self.model)

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
            force_mask = constraint["mask"][:, :, :1]
            force_x_clean = constraint["value"]

        model_output = self.model.guided_forward(
            x, cond, t, weight, force_mask=force_mask, force_x_clean=force_x_clean
        )
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        x_start = model_output
        x_start = maybe_clip(x_start)
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

    def p_mean_variance(self, x, cond, t, constraint=None):
        if t[0] > 1.0 * self.n_timestep:
            weight = min(self.guidance_weight, 0)
        elif t[0] < 0.1 * self.n_timestep:
            weight = min(self.guidance_weight, 1)
        else:
            weight = self.guidance_weight

        force_mask, force_x_clean = None, None
        if constraint is not None:
            force_mask = constraint["mask"][:, :, :1]
            force_x_clean = constraint["value"]

        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.model.guided_forward(
                x, cond, t, weight, force_mask=force_mask, force_x_clean=force_x_clean
            )
        )

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, cond, t, constraint=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, cond=cond, t=t, constraint=constraint
        )
        noise = torch.randn_like(model_mean)
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(noise.shape) - 1))
        )
        x_out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_out, x_start

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
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
            x, _ = self.p_sample(x, cond, timesteps, constraint=constraint)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def ddim_sample(self, shape, cond, constraint=None, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x = torch.randn(shape, device=device)
        cond = move_condition_to_device(cond, device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
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
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

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
    ):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = move_condition_to_device(cond, device)
        if return_diffusion:
            diffusion = [x]

        mask = constraint["mask"].to(device)
        value = constraint["value"].to(device)

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            x, _ = self.p_sample(x, cond, timesteps, constraint=constraint)

            value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
            x = value_ * mask + (1.0 - mask) * x

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def long_inpaint_loop(
        self, shape, cond, noise=None, constraint=None, return_diffusion=False, start_point=None,
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
            )
        assert batch_size > 1
        half = x.shape[1] // 2

        start_point = self.n_timestep if start_point is None else start_point
        mask = constraint["mask"].to(device) if constraint else None
        value = constraint["value"].to(device) if constraint else None

        for i in tqdm(reversed(range(0, start_point))):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            x, _ = self.p_sample(x, cond, timesteps, constraint=constraint)

            if mask is not None and value is not None:
                value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
                x = value_ * mask + (1.0 - mask) * x

            if i > 0:
                x[1:, :half] = x[:-1, half:]

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

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        b, s, c = x_start.shape
        force_mask = torch.zeros((b, s, 1), device=x_start.device, dtype=x_start.dtype)

        for i in range(b):
            rand = torch.rand(1).item()
            if rand < 0.3:
                pass
            elif rand < 0.6:
                keep_len = torch.randint(1, max(2, s // 5), (1,)).item()
                force_mask[i, :keep_len, 0] = 1.0
                force_mask[i, -keep_len:, 0] = 1.0
            else:
                interval = torch.randint(10, 30, (1,)).item()
                force_mask[i, ::interval, 0] = 1.0

        x_recon = self.model(
            x_noisy,
            cond,
            t,
            cond_drop_prob=self.cond_drop_prob,
            force_mask=force_mask,
            force_x_clean=x_start
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

        if model_out.shape[2] == 381:
            target_v = target[:, 1:] - target[:, :-1]
            model_out_v = model_out[:, 1:] - model_out[:, :-1]
            v_loss = self.loss_fn(model_out_v, target_v, reduction="none")
            v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
            v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

            losses = (
                1.0 * loss.mean(),
                3.0 * v_loss.mean(),
                0.0,
                0.0
            )
            return sum(losses), losses
        else:
            model_contact, model_out = torch.split(model_out, (4, model_out.shape[2] - 4), dim=2)
            target_contact, target = torch.split(target, (4, target.shape[2] - 4), dim=2)

            target_v = target[:, 1:] - target[:, :-1]
            model_out_v = model_out[:, 1:] - model_out[:, :-1]
            v_loss = self.loss_fn(model_out_v, target_v, reduction="none")
            v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
            v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

            model_x = model_out[:, :, :3]
            model_q = ax_from_6v(model_out[:, :, 3:].reshape(b, s, -1, 6))
            target_x = target[:, :, :3]
            target_q = ax_from_6v(target[:, :, 3:].reshape(b, s, -1, 6))

            model_xp = self.smpl.forward(model_q, model_x)
            target_xp = self.smpl.forward(target_q, target_x)

            fk_loss = self.loss_fn(model_xp, target_xp, reduction="none")
            fk_loss = reduce(fk_loss, "b ... -> b (...)", "mean")
            fk_loss = fk_loss * extract(self.p2_loss_weight, t, fk_loss.shape)

            foot_idx = [7, 8, 10, 11]
            static_idx = model_contact > 0.95
            model_feet = model_xp[:, :, foot_idx]
            model_foot_v = torch.zeros_like(model_feet)
            model_foot_v[:, :-1] = (
                model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
            )
            model_foot_v[~static_idx] = 0
            foot_loss = self.loss_fn(
                model_foot_v, torch.zeros_like(model_foot_v), reduction="none"
            )
            foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")
            foot_loss = foot_loss * extract(self.p2_loss_weight, t, foot_loss.shape)

            losses = (
                0.636 * loss.mean(),
                2.964 * v_loss.mean(),
                0.646 * fk_loss.mean(),
                10.942 * foot_loss.mean(),
            )
            return sum(losses), losses

    def loss(self, x, cond, t_override=None):
        batch_size = len(x)
        if t_override is None:
            t = torch.randint(0, self.n_timestep, (batch_size,), device=x.device).long()
        else:
            t = torch.full((batch_size,), t_override, device=x.device).long()
        return self.p_losses(x, cond, t)

    def forward(self, x, cond, t_override=None):
        return self.loss(x, cond, t_override)

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
        render=True
    ):
        if isinstance(shape, tuple):
            if mode == "inpaint":
                func_class = self.inpaint_loop
            elif mode == "normal":
                func_class = self.ddim_sample
            elif mode == "long":
                func_class = self.long_ddim_sample
            else:
                assert False, "Unrecognized inference mode"
            samples = (
                func_class(
                    shape,
                    cond,
                    noise=noise,
                    constraint=constraint,
                    start_point=start_point,
                )
                .detach()
                .cpu()
            )
        else:
            samples = shape

        samples = normalizer.unnormalize(samples)

        # ==============================================================================
        # 👇 保留上一次的修改：强制锁死全局位移 (Root Translation)，钉在画面中央
        # ==============================================================================
        if samples.shape[2] == 151:
            samples[..., 4:7] = 0.0
        # ==============================================================================

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

        if mode == "long":
            b, s, c1, c2 = q.shape
            assert s % 2 == 0
            half = s // 2
            if b > 1:
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
            full_pose = (
                self.smpl.forward(full_q, full_pos).detach().cpu().numpy()
            )
            
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
                outname = f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.pkl'
                Path(fk_out).mkdir(parents=True, exist_ok=True)
                pickle.dump(
                    {
                        "smpl_poses": full_q.squeeze(0).reshape((-1, 72)).cpu().numpy(),
                        "smpl_trans": full_pos.squeeze(0).cpu().numpy(),
                        "full_pose": full_pose[0],
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