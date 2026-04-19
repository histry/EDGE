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

from dataset.dance_dataset import AISTPPDataset  # 改用我们修复好参数映射和维度的 AISTPP

from dataset.preprocess import increment_path
from model.adan import Adan
from model.diffusion import GaussianDiffusion
from model.model import DanceDecoder
from vis import SMPLSkeleton


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
        train_stage="full",
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
        self.horizon = horizon = seq_len
        self.train_stage = train_stage

        self.accelerator.wait_for_everyone()

        self.normalizer = normalizer
        checkpoint = None
        if checkpoint_path != "":
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.accelerator.device,
                weights_only=False
            )
            self.normalizer = checkpoint["normalizer"]

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

        self._apply_stage_freezing(model, train_stage)

        smpl = SMPLSkeleton(self.accelerator.device)
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
        )
        if train_stage == "stage1":
            diffusion.cond_drop_prob = 1.0

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters left after stage freezing.")
        optim = Adan(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    num_processes,
                )
            )

    @staticmethod
    def _set_requires_grad(module, enabled):
        for parameter in module.parameters():
            parameter.requires_grad = enabled

    def _apply_stage_freezing(self, model, train_stage):
        if train_stage == "full":
            return
        if train_stage == "stage1":
            self._set_requires_grad(model.cond_projection, False)
            self._set_requires_grad(model.cond_encoder, False)
            self._set_requires_grad(model.non_attn_cond_projection, False)
            if hasattr(model, "trajectory_projection"):
                self._set_requires_grad(model.trajectory_projection, False)
            if hasattr(model, "trajectory_root_head"):
                self._set_requires_grad(model.trajectory_root_head, False)
            return
        if train_stage == "stage2":
            self._set_requires_grad(model, False)
            self._set_requires_grad(model.cond_projection, True)
            self._set_requires_grad(model.cond_encoder, True)
            self._set_requires_grad(model.non_attn_cond_projection, True)
            if hasattr(model, "trajectory_projection"):
                self._set_requires_grad(model.trajectory_projection, True)
            if hasattr(model, "trajectory_root_head"):
                self._set_requires_grad(model.trajectory_root_head, True)
            return
        raise ValueError(f"Unknown train_stage: {train_stage}")

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        print("Loading Official AIST++ dataset (with FK & 6D rotations)...")
        
        data_path = opt.data_path if os.path.exists(os.path.join(opt.data_path, "train")) else "data"

        # 初始化训练集
        train_dataset = AISTPPDataset(
            data_path=data_path,
            train=True,
            backup_path=os.path.join(data_path, "backup"),
            feature_type=self.feature_type,
            seq_len=opt.seq_len, # 将命令行的截断要求传递给数据集底层
        )
        
        # 初始化测试集（添加回代码，用于渲染预览）
        test_dataset = AISTPPDataset(
            data_path=data_path,
            train=False,
            backup_path=os.path.join(data_path, "backup"),
            feature_type=self.feature_type,
            seq_len=opt.seq_len,
            normalizer=train_dataset.normalizer, # 复用训练集的 normalizer
        )
        
        if len(train_dataset) == 0:
            raise RuntimeError(
                f"在 {data_path}/train 目录下没有找到足够长的训练切片！"
            )
        self.normalizer = train_dataset.normalizer

        num_cpus = multiprocessing.cpu_count()
        
        # 训练数据加载器
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=min(int(num_cpus * 0.75), 32),
            pin_memory=True,
            drop_last=False,
        )

        # 测试数据加载器（添加回代码）
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)
        if len(train_data_loader) == 0:
            raise RuntimeError("DataLoader has zero batches. Reduce --batch_size or add more training data.")
            
        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )
        
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()
        for epoch in range(1, opt.epochs + 1):
            avg_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_footloss = 0
            self.train()
            
            for step, (x, cond, filename, wavnames) in enumerate(
                load_loop(train_data_loader)
            ):
                cond = move_condition_to_device(cond, x.device)
                with self.accelerator.autocast():
                    total_loss, (loss, v_loss, fk_loss, foot_loss) = self.diffusion(
                        x, cond, t_override=None
                    )
                self.optim.zero_grad(set_to_none=True)
                self.accelerator.backward(total_loss)
                self.optim.step()

                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    if isinstance(fk_loss, torch.Tensor):
                        avg_fkloss += fk_loss.detach().cpu().numpy()
                    if isinstance(foot_loss, torch.Tensor):
                        avg_footloss += foot_loss.detach().cpu().numpy()
                    
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )
            
            # Save model
            if (epoch % opt.save_interval) == 0:
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.eval()
                    denom = max(len(train_data_loader), 1)
                    avg_loss /= denom
                    avg_vloss /= denom
                    avg_fkloss /= denom
                    avg_footloss /= denom
                    log_dict = {
                        "Train Loss": avg_loss,
                        "V Loss": avg_vloss,
                        "FK Loss": avg_fkloss,
                        "Foot Loss": avg_footloss,
                    }
                    wandb.log(log_dict)

                    print(f"\n📊 Epoch {epoch} | 总损失(Train Loss): {avg_loss:.4f} | 运动学损失(FK): {avg_fkloss:.4f} | 滑步损失(Foot): {avg_footloss:.4f}")

                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                    print(f"[MODEL SAVED at Epoch {epoch}] - 权重已保存: {wdir}/train-{epoch}.pt")
                    
                    # ============ 👇 增加验证集渲染测试代码 👇 ============
                    print("Generating Sample for Visualization...")
                    try:
                        # 从测试集抽取一个 batch 的验证数据
                        (x_test, cond_test, filename_test, wavnames_test) = next(iter(test_data_loader))
                        cond_test = move_condition_to_device(cond_test, self.accelerator.device)
                        
                        # 为了可视化快点，我们每次只抽 2 个样本渲染，如果 batch_size 太小就取实际长度
                        render_count = min(2, len(x_test)) 
                        
                        self.diffusion.render_sample(
                            (render_count, self.horizon, self.repr_dim),
                            slice_condition(cond_test, render_count),
                            self.normalizer,
                            epoch,
                            os.path.join(opt.render_dir, "train_" + opt.exp_name),
                            name=wavnames_test[:render_count],
                            sound=True,
                        )
                        print(f"✅ 渲染完成! 预览文件保存在: {opt.render_dir}")
                    except Exception as e:
                        print(f"⚠️ 渲染测试样本时出错 (但权重已安全保存): {e}")
                    # ==================================================

        if self.accelerator.is_main_process:
            wandb.run.finish()

    def render_sample(
        self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True
    ):
        _, cond, wavname = data_tuple
        cond_for_len = cond["audio"] if isinstance(cond, dict) else cond
        assert len(cond_for_len.shape) == 3
        if render_count < 0:
            render_count = len(cond_for_len)
        shape = (render_count, self.horizon, self.repr_dim)
        cond = move_condition_to_device(cond, self.accelerator.device)
        self.diffusion.render_sample(
            shape,
            slice_condition(cond, render_count),
            self.normalizer,
            label,
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode="long",
            fk_out=fk_out,
            render=render
        )