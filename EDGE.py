import multiprocessing
import os
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dance_dataset import AISTPPDataset, DunhuangDataset
from dataset.preprocess import increment_path
from model.adan import Adan
from model.checkpoint_compat import adapt_checkpoint_state_dict, summarize_adapt_report
from model.diffusion import GaussianDiffusion
from model.model import DanceDecoder
from model.mmr_model import CrossModalMMR
from vis import SMPLSkeleton

def _print_trainable_summary(self, model):
    if not self.accelerator.is_main_process:
        return

    total = 0
    trainable = 0
    module_rows = []

    for name, module in model.named_children():
        module_total = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total += module_total
        trainable += module_trainable
        module_rows.append((name, module_trainable, module_total))

    print("📊 Trainable parameter summary:")
    for name, module_trainable, module_total in module_rows:
        ratio = 0.0 if module_total == 0 else module_trainable / module_total * 100.0
        print(f"  - {name}: {module_trainable}/{module_total} ({ratio:.2f}%)")

    print(f"  Total trainable: {trainable}/{total} ({trainable / max(total, 1) * 100.0:.2f}%)")

def move_condition_to_device(cond, device):
    if isinstance(cond, dict):
        moved = {}
        for key, value in cond.items():
            moved[key] = value.to(device) if torch.is_tensor(value) else value
        return moved
    return cond.to(device)

def slice_condition(cond, count):
    if isinstance(cond, dict):
        sliced = {}
        for key, value in cond.items():
            sliced[key] = value[:count] if torch.is_tensor(value) else value
        return sliced
    return cond[:count]

def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}

def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)

class EDGE:
    @staticmethod
    def _extract_state_dict_for_load(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value, key
            if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
                return checkpoint, "plain_state_dict"
        raise ValueError("checkpoint 既不是 plain state_dict，也不是包含 model_state_dict/state_dict 的 wrapped checkpoint。")

    @staticmethod
    def _normalize_state_dict_prefix(state_dict, reference_state_dict):
        if set(state_dict.keys()) == set(reference_state_dict.keys()):
            return state_dict, "unchanged"

        if state_dict and all(key.startswith("module.") for key in state_dict.keys()):
            stripped = {key[len("module."):]: value for key, value in state_dict.items()}
            if set(stripped.keys()) == set(reference_state_dict.keys()):
                return stripped, "stripped_module_prefix"

        return state_dict, "unchanged"

    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        normalizer=None,
        EMA=True,
        learning_rate=4e-4,
        weight_decay=0.02,
        audio_dim=803,
        seq_len=150,
        mixed_precision="bf16",
        gradient_checkpointing=False,
        use_sparse_attn=False,
        sparse_attn_window=24,
        cond_drop_prob=0.25,
        audio_pairing_mode="proxy",
        mmr_loss_weight=0.0,
        keyframe_condition_prob=0.7,
        keyframe_condition_width=3,
        keyframe_loss_weight=2.0,
        contact_loss_weight=0.8,
        foot_loss_weight=2.5,
        sync_loss_weight=1.2,
        mid_keyframe_condition_prob=-1.0,
        mid_keyframe_count=2,
        mid_keyframe_condition_width=1,
        mid_keyframe_selection="motion_peak",
        beat_guidance_weight=0.0,
        trajectory_loss_weight=1.0,
        trajectory_velocity_loss_weight=0.25,
        energy_condition_prob=0.7,
        energy_condition_drop_prob=0.15,
        energy_loss_weight=0.25,
        root_lower_coupling_loss_weight=0.5,
        root_lower_speed_threshold=0.012,
        root_lower_min_motion=0.010,
        adapter_train_decoder=False,
        hard_keyframe_project=False,
        train_stage="full",
        strict_audio_checkpoint=False,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            mixed_precision=mixed_precision,
        )
        state = AcceleratorState()
        num_processes = state.num_processes
        self.feature_type = feature_type

        self.repr_dim = repr_dim = 151
        self.audio_dim = audio_dim
        self.audio_pairing_mode = audio_pairing_mode

        if audio_pairing_mode != "paired" and mmr_loss_weight > 0:
            print(
                f"⚠️ audio_pairing_mode={audio_pairing_mode}，当前没有真实音乐-动作配对监督；"
                f"将 mmr_loss_weight 从 {mmr_loss_weight} 强制设为 0，避免把 proxy music 当作真标签。"
            )
            mmr_loss_weight = 0.0
        self.horizon = horizon = seq_len
        self.train_stage = train_stage

        if mid_keyframe_condition_prob < 0:
            mid_keyframe_condition_prob = 0.7 if train_stage == "stage2" else 0.0

        self.accelerator.wait_for_everyone()

        self.normalizer = normalizer
        checkpoint = None
        if checkpoint_path != "":
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.accelerator.device,
                weights_only=False
            )
            if "normalizer" in checkpoint:
                norm_data = checkpoint["normalizer"]
                if isinstance(norm_data, dict) and "mean" in norm_data:
                    from dataset.preprocess import Normalizer
                    import numpy as np
                    dummy = torch.zeros((1, 1, 151))
                    self.normalizer = Normalizer(dummy)
                    self.normalizer.mean = np.array(norm_data["mean"])
                    self.normalizer.std = np.array(norm_data["std"])
                else:
                    self.normalizer = norm_data
            else:
                self.normalizer = None

        model = DanceDecoder(
            nfeats=repr_dim,
            seq_len=horizon,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=audio_dim,
            activation=F.gelu,
            use_gradient_checkpointing=gradient_checkpointing,
            use_sparse_attn=use_sparse_attn,
            sparse_attn_window=sparse_attn_window,
        )

        self._apply_stage_freezing(model, train_stage, adapter_train_decoder=adapter_train_decoder)
        _print_trainable_summary(self, model)

        smpl = SMPLSkeleton(self.accelerator.device)
        
        self.mmr_model = None
        mmr_ckpt_path = "weights/mmr_pretrained.pt"
        if mmr_loss_weight <= 0:
            print("⏭️ MMR 跨模态损失权重为 0：已关闭音频-动作对齐监督，适合未配对音乐/动作数据。")
        else:
            self.mmr_model = CrossModalMMR(
                motion_dim=151,
                audio_dim=self.audio_dim,
                latent_dim=256,
            ).to(self.accelerator.device)
            if os.path.exists(mmr_ckpt_path):
                try:
                    mmr_checkpoint = torch.load(mmr_ckpt_path, map_location=self.accelerator.device)
                    mmr_state_dict, mmr_format = self._extract_state_dict_for_load(mmr_checkpoint)
                    mmr_state_dict, prefix_action = self._normalize_state_dict_prefix(
                        mmr_state_dict, self.mmr_model.state_dict()
                    )
                    self.mmr_model.load_state_dict(mmr_state_dict, strict=True)

                    prefix_msg = "" if prefix_action == "unchanged" else "，并已自动去除 module. 前缀"
                    print(
                        f"✅ 成功激活 MMR 跨模态引导：已加载预训练权重 {mmr_ckpt_path} "
                        f"(格式: {mmr_format}{prefix_msg})"
                    )
                except Exception as exc:
                    print(
                        f"⚠️ 警告：检测到 MMR 权重文件 {mmr_ckpt_path}，"
                        f"但加载失败（{exc}）。本次训练将临时禁用 MMR 引导。"
                    )
                    self.mmr_model = None
            else:
                print("⚠️ 警告：未找到 MMR 预训练权重！为防止随机梯度导致扩散模型坍缩，本次训练临时禁用 MMR 引导。")
                self.mmr_model = None
        
        diffusion = GaussianDiffusion(
            model,
            horizon,
            repr_dim,
            smpl,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=cond_drop_prob,
            guidance_weight=2,
            clip_denoised=False,
            mmr_model=self.mmr_model,
            mmr_loss_weight=mmr_loss_weight,
            keyframe_condition_prob=keyframe_condition_prob,
            keyframe_condition_width=keyframe_condition_width,
            keyframe_loss_weight=keyframe_loss_weight,
            contact_loss_weight=contact_loss_weight,
            foot_loss_weight=foot_loss_weight,
            sync_loss_weight=sync_loss_weight,
            mid_keyframe_condition_prob=mid_keyframe_condition_prob,
            mid_keyframe_count=mid_keyframe_count,
            mid_keyframe_condition_width=mid_keyframe_condition_width,
            mid_keyframe_selection=mid_keyframe_selection,
            data_fps=30,
            beat_guidance_weight=beat_guidance_weight,
            trajectory_loss_weight=trajectory_loss_weight,
            trajectory_velocity_loss_weight=trajectory_velocity_loss_weight,
            energy_condition_prob=energy_condition_prob,
            energy_condition_drop_prob=energy_condition_drop_prob,
            energy_loss_weight=energy_loss_weight,
            root_lower_coupling_loss_weight=root_lower_coupling_loss_weight,
            root_lower_speed_threshold=root_lower_speed_threshold,
            root_lower_min_motion=root_lower_min_motion,
            hard_keyframe_project=hard_keyframe_project,
        )
        
        diffusion.normalizer = self.normalizer
        
        if train_stage in ["stage1", "stage2"]:
            diffusion.cond_drop_prob = 0.0
            diffusion.force_audio_only_drop = True
        else:
            diffusion.force_audio_only_drop = False

        if train_stage == "stage2":
            print(
                "🎯 第二阶段多关键帧训练已配置: "
                f"middle_prob={mid_keyframe_condition_prob}, "
                f"max_middle={mid_keyframe_count}, "
                f"width={mid_keyframe_condition_width}, "
                f"selection={mid_keyframe_selection}"
            )

        print(f"Model has {sum(y.numel() for y in model.parameters())} parameters")

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        # ✨ 新增这行：将 diffusion 内部的模型引用强制替换为被 Accelerator 包装后的 DDP 模型
        self.diffusion.model = self.model
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters left after stage freezing.")
        optim = Adan(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            if EMA and "ema_state_dict" in checkpoint:
                state_dict = checkpoint["ema_state_dict"]
            else:
                state_dict = checkpoint.get("model_state_dict")
            if state_dict is None:
                raise ValueError(f"❌ 权重文件 {checkpoint_path} 中既没有 ema_state_dict 也没有 model_state_dict！")

            adapted_state_dict, adapt_report = adapt_checkpoint_state_dict(
                state_dict,
                self.model,
                log_prefix=f"checkpoint:{os.path.basename(checkpoint_path)}",
            )
            self.model.load_state_dict(adapted_state_dict, strict=True)
            self._check_audio_checkpoint_compatibility(
                adapt_report=adapt_report,
                strict_audio_checkpoint=strict_audio_checkpoint,
                train_stage=train_stage,
            )
            # 同步 EMA master_model，避免续训时 EMA 从随机初始化开始
            unwrapped_loaded_model = self.accelerator.unwrap_model(self.model)
            if EMA and isinstance(checkpoint, dict) and "ema_state_dict" in checkpoint:
                ema_adapted_state_dict, ema_adapt_report = adapt_checkpoint_state_dict(
                    checkpoint["ema_state_dict"],
                    self.diffusion.master_model,
                    log_prefix=f"ema_checkpoint:{os.path.basename(checkpoint_path)}",
                )
                self.diffusion.master_model.load_state_dict(
                    ema_adapted_state_dict,
                    strict=True,
                )

                self._check_audio_checkpoint_compatibility(
                    adapt_report=ema_adapt_report,
                    strict_audio_checkpoint=False,
                    train_stage=train_stage,
                )

                if self.accelerator.is_main_process:
                    for line in summarize_adapt_report(ema_adapt_report):
                        print(f"⚠️ EMA {line}")
            else:
                self.diffusion.master_model.load_state_dict(
                    unwrapped_loaded_model.state_dict(),
                    strict=True,
                )

            if self.accelerator.is_main_process:
                for line in summarize_adapt_report(adapt_report):
                    print(f"⚠️ {line}")

    @staticmethod
    def _set_requires_grad(module, enabled):
        for parameter in module.parameters():
            parameter.requires_grad = enabled
    
    @staticmethod
    def _check_audio_checkpoint_compatibility(
        adapt_report,
        strict_audio_checkpoint=False,
        train_stage="full",
    ):
        """
        Check whether audio-related modules were skipped or newly initialized
        during checkpoint loading.

        Why this matters:
        - If current audio_dim=803 but checkpoint cond_projection.weight expects
          another input dimension, adapt_checkpoint_state_dict will skip that
          tensor and keep the newly initialized layer.
        - In that case, we must not claim that the audio encoder/projection was
          fully inherited from the pretrained checkpoint.
        - This is especially dangerous in stage1/stage2, because those stages
          may freeze audio-related modules. Freezing newly initialized audio
          weights would preserve a random audio mapping.
        """
        audio_markers = (
            "cond_projection",
            "cond_encoder",
            "non_attn_cond_projection",
            "null_cond_embed",
            "null_cond_hidden",
        )

        def normalize_key(key):
            key = str(key)
            if key.startswith("module."):
                key = key[len("module."):]
            return key

        def is_audio_key(key):
            key = normalize_key(key)
            return any(marker in key for marker in audio_markers)

        skipped_audio = []
        for item in adapt_report.get("skipped_shape", []):
            # skipped_shape item format:
            # (key, checkpoint_shape, model_shape)
            key = item[0] if isinstance(item, (list, tuple)) and item else str(item)
            if is_audio_key(key):
                skipped_audio.append(item)

        kept_audio = [
            key for key in adapt_report.get("kept_other", [])
            if is_audio_key(key)
        ]

        if not skipped_audio and not kept_audio:
            return

        message = (
            "\n⚠️ Audio checkpoint compatibility warning:\n"
            f"  train_stage={train_stage}\n"
            f"  skipped_audio_shape_keys={skipped_audio[:8]}\n"
            f"  newly_initialized_audio_keys={kept_audio[:8]}\n"
            "  说明：部分音频条件分支没有从 checkpoint 成功加载，"
            "不能声称完整继承预训练音乐编码能力。\n"
        )

        print(message)

        if train_stage in ["stage1", "stage2"]:
            print(
                "⚠️ 当前 train_stage 会冻结或部分冻结音频分支。"
                "如果这些音频层是新初始化的，冻结它们会保留随机音频映射。"
                "建议：1) 使用匹配 audio_dim 的 checkpoint；"
                "2) 或改用 train_stage=full；"
                "3) 或明确汇报音乐部分只是弱节奏引导。"
            )

        if strict_audio_checkpoint:
            raise RuntimeError(
                message
                + "\n由于启用了 --strict_audio_checkpoint，训练已停止。"
            )
    def _apply_stage_freezing(self, model, train_stage, adapter_train_decoder=False):
        """
        Training stages:
        - full:
            Train all modules.
        - stage1:
            Control-adaptation stage. Keep audio encoder frozen to preserve the
            pretrained audio prior, and train motion/control-side modules.
        - stage2:
            Multi-keyframe refinement stage. Train decoder and control-related
            modules for stronger inpainting/keyframe adaptation.
        """
        if train_stage == "full":
            print("🧩 train_stage=full: all modules are trainable.")
            return

        if train_stage == "stage1":
            # Preserve audio prior.
            self._set_requires_grad(model.cond_projection, False)
            self._set_requires_grad(model.cond_encoder, False)
            self._set_requires_grad(model.non_attn_cond_projection, False)

            # Adapt motion/control path.
            self._set_requires_grad(model.input_projection, True)
            self._set_requires_grad(model.seqTransDecoder, True)
            self._set_requires_grad(model.final_layer, True)

            if hasattr(model, "trajectory_projection"):
                self._set_requires_grad(model.trajectory_projection, True)
            if hasattr(model, "traj_modulate"):
                self._set_requires_grad(model.traj_modulate, True)

            print(
                "🧩 train_stage=stage1: frozen audio condition encoder; "
                "training decoder + input projection + trajectory/control branches."
            )
            return

        if train_stage == "stage2":
            # Start from a clean frozen state.
            self._set_requires_grad(model, False)

            # Refine inpainting/control path.
            self._set_requires_grad(model.input_projection, True)
            self._set_requires_grad(model.seqTransDecoder, True)
            self._set_requires_grad(model.final_layer, True)

            if hasattr(model, "trajectory_projection"):
                self._set_requires_grad(model.trajectory_projection, True)
            if hasattr(model, "traj_modulate"):
                self._set_requires_grad(model.traj_modulate, True)

            print(
                "🧩 train_stage=stage2: frozen audio encoder and non-control modules; "
                "training decoder + keyframe/trajectory control path."
            )
            return


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

        raise ValueError(f"Unknown train_stage: {train_stage}")

    @staticmethod
    def _loss_keys():
        return [
            "Recon Loss",
            "Velocity Loss",
            "Contact Loss",
            "FK Loss",
            "Foot Loss",
            "Anti-Freeze Loss",
            "MMR Loss",
            "Trajectory Loss",
            "Keyframe Loss",
            "Kinematic Sync Loss",
            "Biomech Loss",
            "Root Turn Loss",
            "Contact Turn Loss",
            "Body Stability Loss",
            "Root-Lower Coupling Loss",
            "Motion Energy Loss",
        ]

    def _run_validation(self, data_loader, epoch, max_batches=10):
        if max_batches <= 0:
            return None

        self.eval()
        totals = None
        counted_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break

                x, cond, _, _ = batch
                x = x.to(self.accelerator.device)
                cond = move_condition_to_device(cond, self.accelerator.device)

                # Make validation more reproducible across epochs/runs.
                # This does not remove stochastic diffusion training, but fixes validation-time masks/timesteps.
                with torch.random.fork_rng(devices=[self.accelerator.device] if self.accelerator.device.type == "cuda" else []):
                    torch.manual_seed(12345 + int(epoch) * 1000 + int(batch_idx))
                    if self.accelerator.device.type == "cuda":
                        torch.cuda.manual_seed_all(12345 + int(epoch) * 1000 + int(batch_idx))

                    val_loss, val_losses = self.diffusion(x, cond, current_epoch=epoch)

                metric_tensors = [val_loss.detach()]
                for item in val_losses:
                    if torch.is_tensor(item):
                        metric_tensors.append(item.detach())
                    else:
                        metric_tensors.append(
                            torch.tensor(float(item), device=self.accelerator.device)
                        )

                stacked = torch.stack(
                    [metric.float().mean() for metric in metric_tensors], dim=0
                )
                gathered = self.accelerator.gather_for_metrics(stacked.unsqueeze(0))
                batch_metrics = gathered.mean(dim=0)

                totals = batch_metrics if totals is None else totals + batch_metrics
                counted_batches += 1

        if counted_batches == 0 or totals is None:
            return None

        averages = totals / counted_batches
        metrics = {"Val Loss": averages[0].item()}
        for key, value in zip(self._loss_keys(), averages[1:]):
            metrics[f"Val {key}"] = value.item()
        return metrics
        
    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        data_path = opt.data_path

        # 统一定义 trajectory condition 开关，避免只在敦煌分支定义导致 AIST++ 分支报错。
        use_traj_cond = not getattr(opt, "disable_traj_cond", False)

        is_dunhuang = "dunhuang" in data_path.lower()
        if is_dunhuang:
            print(f"\n🪷 检测到敦煌数据集路径 ({data_path})，启动中华古典舞纯视觉微调模式！")
            if opt.audio_pairing_mode == "none":
                train_audio_sample_mode = "zero"
                val_audio_sample_mode = "zero"
            elif opt.audio_pairing_mode == "paired":
                train_audio_sample_mode = "best"
                val_audio_sample_mode = "best"
            else:
                train_audio_sample_mode = "random"
                val_audio_sample_mode = getattr(opt, "dunhuang_val_audio_mode", "best")
            if self.accelerator.is_main_process:
                print(f"🧭 trajectory condition enabled: {use_traj_cond}")
            train_dataset = DunhuangDataset(
                data_path=data_path,
                train=True,
                seq_len=opt.seq_len,
                audio_dim=self.audio_dim,
                normalizer=self.normalizer,
                return_traj=use_traj_cond,
                split_ratio=getattr(opt, "dunhuang_split_ratio", 0.9),
                split_seed=getattr(opt, "dunhuang_split_seed", 42),
                audio_sample_mode=train_audio_sample_mode,
                audio_pairing_mode=opt.audio_pairing_mode,
                paired_audio_missing_policy=getattr(opt, "paired_audio_missing_policy", "error"),
                weak_pairs_path=getattr(opt, "weak_pairs_path", "data/proxy_weak_pairs/weak_pairs.csv"),
                traj_aug_prob=getattr(opt, "traj_aug_prob", 0.3),
                traj_aug_scale_range=(
                    getattr(opt, "traj_aug_scale_min", 0.8),
                    getattr(opt, "traj_aug_scale_max", 1.25),
                ),
                traj_aug_rot_deg=getattr(opt, "traj_aug_rot_deg", 30.0),
            )
            self.normalizer = train_dataset.normalizer
            self.diffusion.normalizer = self.normalizer

            test_dataset = DunhuangDataset(
                data_path=data_path,
                train=False,
                seq_len=opt.seq_len,
                audio_dim=self.audio_dim,
                normalizer=self.normalizer,
                return_traj=use_traj_cond,
                split_ratio=getattr(opt, "dunhuang_split_ratio", 0.9),
                split_seed=getattr(opt, "dunhuang_split_seed", 42),
                audio_sample_mode=val_audio_sample_mode,
                audio_pairing_mode=opt.audio_pairing_mode,
                paired_audio_missing_policy=getattr(opt, "paired_audio_missing_policy", "error"),
                weak_pairs_path=getattr(opt, "weak_pairs_path", "data/proxy_weak_pairs/weak_pairs.csv"),
                traj_aug_prob=0.0,
            )

            self.normalizer = train_dataset.normalizer
            self.diffusion.normalizer = self.normalizer

            print(
                f"✅ 敦煌数据已按源文件划分: "
                f"train_windows={len(train_dataset)}, val_windows={len(test_dataset)}, "
                f"val_audio_mode={getattr(opt, 'dunhuang_val_audio_mode', 'best')}"
            )
            if len(test_dataset) == 0:
                print("⚠️ 敦煌验证集没有足够长的切片，将临时复用训练集做运行期 sanity check。")
                test_dataset = train_dataset
        else:
            print("🎶 Loading Official AIST++ dataset (with FK & 6D rotations)...")
            actual_data_path = data_path if os.path.exists(os.path.join(data_path, "train")) else "data"
            train_dataset = AISTPPDataset(
                data_path=actual_data_path,
                train=True,
                backup_path=os.path.join(actual_data_path, "backup"),
                feature_type=self.feature_type,
                seq_len=opt.seq_len,
                return_traj=use_traj_cond,
            )
            test_dataset = AISTPPDataset(
                data_path=actual_data_path,
                train=False,
                backup_path=os.path.join(actual_data_path, "backup"),
                feature_type=self.feature_type,
                seq_len=opt.seq_len,
                normalizer=train_dataset.normalizer,
                return_traj=use_traj_cond,
            )
            self.normalizer = train_dataset.normalizer
            self.diffusion.normalizer = self.normalizer
        
        if self.accelerator.is_main_process:
            print("\n🔎 Training configuration sanity check:")
            print(f"  data_path={data_path}")
            print(f"  audio_pairing_mode={getattr(opt, 'audio_pairing_mode', 'unknown')}")
            print(f"  mmr_loss_weight={getattr(opt, 'mmr_loss_weight', 0.0)}")
            print(f"  beat_guidance_weight={getattr(opt, 'beat_guidance_weight', 0.0)}")
            print(f"  trajectory_condition_enabled={use_traj_cond}")
            print(f"  keyframe_condition_prob={getattr(opt, 'keyframe_condition_prob', 0.0)}")
            print(f"  keyframe_loss_weight={getattr(opt, 'keyframe_loss_weight', 0.0)}")

            if getattr(opt, "audio_pairing_mode", "proxy") != "paired" and getattr(opt, "mmr_loss_weight", 0.0) > 0:
                print(
                    "⚠️ MMR loss was requested without paired audio-motion data. "
                    "EDGE.__init__ will disable it to avoid false cross-modal supervision."
                )

            if is_dunhuang and getattr(opt, "audio_pairing_mode", "proxy") != "paired":
                print(
                    "📌 Dunhuang music claim: no true paired audio-motion supervision. "
                    "Report music as weak/proxy rhythm guidance, not strict paired learning."
                )
        if len(train_dataset) == 0:
            raise RuntimeError(f"在 {data_path} 目录下没有找到足够长的训练切片！")
            
        num_cpus = multiprocessing.cpu_count()
        train_workers = min(int(num_cpus * 0.75), 16)
        val_workers = 2

        if getattr(opt, "train_num_workers", -1) >= 0:
            train_workers = int(opt.train_num_workers)
        if getattr(opt, "val_num_workers", -1) >= 0:
            val_workers = int(opt.val_num_workers)

        if self.accelerator.is_main_process:
            print(
                f"🧪 DataLoader workers: train={train_workers}, val={val_workers}; "
                f"max_train_batches={getattr(opt, 'max_train_batches', 0)}"
            )

        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=train_workers,
            pin_memory=True,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=True,
            drop_last=False,
        )

        train_data_loader, test_data_loader = self.accelerator.prepare(
            train_data_loader, test_data_loader
        )

        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = Path(save_dir).name
            os.makedirs(save_dir, exist_ok=True)
            opt.render_dir = os.path.join(save_dir, "renders")
            os.makedirs(opt.render_dir, exist_ok=True)
            print(f"Directory created at {save_dir}")
            wandb.init(project="EDGE", name=opt.exp_name)

        step = 0
        for epoch in range(1, opt.epochs + 1):
            train_loss = 0.0
            self.train()
            for batch_idx, batch in enumerate(tqdm(train_data_loader, leave=False)):
                max_train_batches = int(getattr(opt, "max_train_batches", 0) or 0)
                if max_train_batches > 0 and batch_idx >= max_train_batches:
                    if self.accelerator.is_main_process:
                        print(f"🧪 max_train_batches reached: {max_train_batches}; ending epoch early.")
                    break

                x, cond, name, wav = batch
                
                x = x.to(self.accelerator.device)
                cond = move_condition_to_device(cond, self.accelerator.device)
                
                with self.accelerator.accumulate(self.model):
                    loss, losses = self.diffusion(x, cond, current_epoch=epoch)
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                    self.optim.step()
                    self.optim.zero_grad()
                    
                    # ✨ 核心修复 3：应用与更新 EMA 模型权重，锁定生成下限质量
                    if self.accelerator.is_main_process and step % opt.ema_interval == 0:
                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                        self.diffusion.ema.update_model_average(self.diffusion.master_model, unwrapped_model)
                        
                train_loss += loss.item()
                step += 1

                if self.accelerator.is_main_process and step % 10 == 0:
                    log_dict = {"Train Loss": loss.item(), "Epoch": epoch, "Step": step}

                    for key, val in zip(self._loss_keys(), losses):
                        log_dict[key] = val.item() if isinstance(val, torch.Tensor) else val
                    wandb.log(log_dict)

            should_validate = (epoch % opt.save_interval == 0) or (epoch == opt.epochs)
            val_metrics = None
            if should_validate:
                val_metrics = self._run_validation(
                    test_data_loader,
                    epoch,
                    max_batches=getattr(opt, "val_batches", 10),
                )
                self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                train_loss /= len(train_data_loader)
                print(f"Epoch {epoch} | Train Loss: {train_loss:.6f}")

                if val_metrics is not None:
                    val_log = {"Epoch": epoch, "Step": step, **val_metrics}
                    wandb.log(val_log)
                    print(
                        "Validation | "
                        + " | ".join(
                            [
                                f"{key}: {value:.6f}"
                                for key, value in val_metrics.items()
                                if isinstance(value, (float, int))
                            ]
                        )
                    )

                if should_validate:
                    weight_dir = os.path.join(save_dir, "weights")
                    os.makedirs(weight_dir, exist_ok=True)
                    save_path = os.path.join(weight_dir, f"train-{epoch}.pt")
                    
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    
                    # ✨ 核心修复 3 延续：同步保存 ema_state_dict 以供推理时加载
                    torch.save(
                        {
                            "model_state_dict": unwrapped_model.state_dict(),
                            "ema_state_dict": self.diffusion.master_model.state_dict(),
                            "normalizer": {
                                "mean": self.normalizer.mean.tolist(),
                                "std": self.normalizer.std.tolist(),
                            },

                            # 保存完整训练配置，方便复现实验和回答老师关于 loss / 数据设置的问题。
                            "train_config": vars(opt),

                            # 明确保存音乐监督边界，避免把 proxy music 误报为真实配对监督。
                            "claim_notes": {
                                "audio_pairing_mode": getattr(opt, "audio_pairing_mode", "unknown"),
                                "mmr_loss_weight": float(getattr(opt, "mmr_loss_weight", 0.0)),
                                "beat_guidance_weight": float(getattr(opt, "beat_guidance_weight", 0.0)),
                                "music_supervision_claim": (
                                    "paired_audio_motion_supervision"
                                    if getattr(opt, "audio_pairing_mode", "proxy") == "paired"
                                    else "weak_or_proxy_rhythm_guidance_only"
                                ),
                                "trajectory_condition_enabled": bool(use_traj_cond),
                                "keyframe_condition_prob": float(getattr(opt, "keyframe_condition_prob", 0.0)),
                                "keyframe_loss_weight": float(getattr(opt, "keyframe_loss_weight", 0.0)),
                                "contact_loss_weight": float(getattr(opt, "contact_loss_weight", 0.0)),
                                "foot_loss_weight": float(getattr(opt, "foot_loss_weight", 0.0)),
                                "sync_loss_weight": float(getattr(opt, "sync_loss_weight", 0.0)),
                            },
                        },
                        save_path,
                    )
                    print(f"✅ [MODEL SAVED at Epoch {epoch}] - 权重已保存: {save_path}")

                    self.eval()

                    # 数据集内 batch 抽查
                    try:
                        print("Generating Sample for Visualization...")
                        batch = next(iter(test_data_loader))
                        render_count = min(opt.batch_size, 2)
                        
                        self.render_sample(
                            data_tuple=batch,
                            label=epoch,
                            render_dir=opt.render_dir,
                            render_count=render_count,
                            render=True
                        )
                    except Exception as e:
                        print(f"⚠️ 渲染测试样本时出错 (但权重已安全保存): {e}")

                    # OOD 未见音乐抽查默认关闭，避免长音频推理把训练节奏拖得过慢。
                    if getattr(opt, "enable_ood_eval", False):
                        try:
                            print(f"\n🎵 暂停训练，开始执行 OOD 测试曲目自动抽查 (Epoch {epoch})...")
                            test_music_dir = getattr(opt, "ood_music_dir", "test_music_bank")
                            ood_out_dir = os.path.join(opt.render_dir, "ood_eval")
                            os.makedirs(ood_out_dir, exist_ok=True)
                            
                            if os.path.exists(test_music_dir):
                                import glob
                                import librosa
                                import torch.nn.functional as F
                                from data.audio_extraction.wav2vec_librosa_features import extract as hybrid_extract
                                
                                test_wavs = sorted(glob.glob(os.path.join(test_music_dir, "*.wav")))
                                max_ood_files = getattr(opt, "ood_max_files", 0)
                                if max_ood_files > 0:
                                    test_wavs = test_wavs[:max_ood_files]

                                if not test_wavs:
                                    print(f"⚠️ {test_music_dir} 中没有 wav 文件，跳过抽查。")
                                
                                for wav_path in test_wavs:
                                    song_name = os.path.basename(wav_path).replace(".wav", "")
                                    print(f"   ▶️ 正在推理并渲染测试曲: [{song_name}]")
                                    
                                    audio_feat_np, _ = hybrid_extract(wav_path)
                                    raw_feat_t = torch.from_numpy(audio_feat_np).float().unsqueeze(0).transpose(1, 2)

                                    y, sr = librosa.load(wav_path, sr=None)
                                    duration = librosa.get_duration(y=y, sr=sr)
                                    
                                    print(f"   🎬 正在按完整时长进行推理，总计帧数: {target_frames} (约 {duration:.1f} 秒)")
                                    aligned_feat = F.interpolate(raw_feat_t, size=target_frames, mode='linear', align_corners=False).transpose(1, 2).squeeze(0)
                                    
                                    horizon = self.horizon  
                                    stride = horizon // 2   
                                    
                                    cond_list = []
                                    if target_frames <= horizon:
                                        pad_len = horizon - target_frames
                                        last_frame = aligned_feat[-1:]
                                        padded_feat = torch.cat([aligned_feat, last_frame.repeat(pad_len, 1)], dim=0)
                                        cond_list.append(padded_feat)
                                    else:
                                        for i in range(0, target_frames, stride):
                                            chunk = aligned_feat[i : i + horizon]
                                            if chunk.shape[0] < horizon:
                                                pad_len = horizon - chunk.shape[0]
                                                last_frame = chunk[-1:]
                                                chunk = torch.cat([chunk, last_frame.repeat(pad_len, 1)], dim=0)
                                            cond_list.append(chunk)
                                            if i + horizon >= target_frames:
                                                break
                                                
                                    cond_audio = torch.stack(cond_list, dim=0).to(self.accelerator.device)
                                    
                                    cond_traj = torch.zeros((cond_audio.shape[0], cond_audio.shape[1], 2), device=self.accelerator.device, dtype=cond_audio.dtype)
                                    
                                    if self.normalizer is not None and hasattr(self.normalizer, 'mean'):
                                        mean_x = self.normalizer.mean[4]
                                        mean_z = self.normalizer.mean[6]
                                        std_x = self.normalizer.std[4]
                                        std_z = self.normalizer.std[6]
                                        
                                        cond_traj[:, :, 0] = (cond_traj[:, :, 0] - mean_x) / std_x
                                        cond_traj[:, :, 1] = (cond_traj[:, :, 1] - mean_z) / std_z
                                    cond_dict = {"audio": cond_audio, "trajectory": cond_traj}
                                    
                                    self.render_sample(
                                        data_tuple=(None, cond_dict, [song_name], [wav_path]),
                                        label=f"ood_{epoch}",
                                        render_dir=ood_out_dir,
                                        render_count=-1,
                                        render=True
                                    )
                                    
                                print(f"✅ Epoch {epoch} 的 OOD 抽查视频全部渲染完毕，准备恢复训练！\n")
                                torch.cuda.empty_cache() 
                            else:
                                print(f"⚠️ 未找到 {test_music_dir} 文件夹，请创建并放入测试音乐。")
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            print(f"⚠️ OOD 抽查执行失败，但不影响训练继续: {e}")
                    else:
                        print("⏭️ 已跳过 OOD 未见音乐抽查（默认关闭，可通过 --enable_ood_eval 开启）。")

        if self.accelerator.is_main_process:
            wandb.finish()

    def render_sample(
        self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True, use_tto=True
    ):
        x, cond, name, wav = data_tuple
        
        cond_for_len = cond["audio"] if isinstance(cond, dict) else cond
        assert len(cond_for_len.shape) == 3
        if render_count < 0:
            render_count = len(cond_for_len)
            
        seq_len = cond_for_len.shape[1]
        shape = (render_count, seq_len, self.repr_dim)
        
        cond = move_condition_to_device(cond, self.accelerator.device)
        is_dummy_audio = torch.all(cond_for_len == 0).item()
        has_real_audio = wav is not None and isinstance(wav, (list, tuple)) and len(wav) > 0 and os.path.exists(wav[0])

        if has_real_audio:
            sound_dir = os.path.dirname(wav[0])
            pure_names = [os.path.splitext(w)[0] for w in wav] 
            import librosa
            try:
                y, sr = librosa.load(wav[0], sr=None)
            except Exception as e:
                print(f"获取时长失败，回退到默认帧数: {e}")
        else:
            sound_dir = "ood_sliced"
            pure_names = name

        self.diffusion.render_sample(
            shape, 
            slice_condition(cond, render_count), 
            self.normalizer,
            label,
            render_dir,
            name=pure_names[:render_count] if not has_real_audio else pure_names, 
            sound=has_real_audio, 
            mode="long" if has_real_audio else "normal",
            fk_out=fk_out,
            render=render,
            use_tto=use_tto
        )
