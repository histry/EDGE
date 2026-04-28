import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dance_dataset import AISTPPDataset
from dataset.preprocess import increment_path
from model.mmr_model import CrossModalMMR


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    motion, cond, _, _ = batch
    audio = cond["audio"].to(device, non_blocking=True)
    motion = motion.to(device, non_blocking=True)
    return motion, audio


def build_datasets(opt):
    actual_data_path = (
        opt.data_path if os.path.exists(os.path.join(opt.data_path, "train")) else "data"
    )
    backup_path = os.path.join(actual_data_path, "backup")

    train_dataset = AISTPPDataset(
        data_path=actual_data_path,
        train=True,
        backup_path=backup_path,
        feature_type=opt.feature_type,
        seq_len=opt.seq_len,
        force_reload=opt.force_reload,
    )
    val_dataset = AISTPPDataset(
        data_path=actual_data_path,
        train=False,
        backup_path=backup_path,
        feature_type=opt.feature_type,
        seq_len=opt.seq_len,
        normalizer=train_dataset.normalizer,
        force_reload=opt.force_reload,
    )
    return train_dataset, val_dataset


def build_dataloaders(train_dataset, val_dataset, opt):
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=opt.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=max(0, min(4, opt.num_workers)),
        pin_memory=True,
        drop_last=False,
        persistent_workers=opt.num_workers > 0,
    )
    return train_loader, val_loader


def get_autocast_context(device, precision):
    if device.type != "cuda" or precision == "no":
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def symmetric_contrastive_loss(audio_latent, motion_latent, temperature):
    audio_latent = F.normalize(audio_latent, dim=-1)
    motion_latent = F.normalize(motion_latent, dim=-1)

    logits = audio_latent @ motion_latent.t()
    logits = logits / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_a2m = F.cross_entropy(logits, labels)
    loss_m2a = F.cross_entropy(logits.t(), labels)
    loss = 0.5 * (loss_a2m + loss_m2a)

    with torch.no_grad():
        top1_a2m = (logits.argmax(dim=1) == labels).float().mean()
        top1_m2a = (logits.t().argmax(dim=1) == labels).float().mean()
        pos_cos = F.cosine_similarity(audio_latent, motion_latent, dim=-1).mean()

    metrics = {
        "loss": loss.detach(),
        "top1_a2m": top1_a2m,
        "top1_m2a": top1_m2a,
        "pos_cos": pos_cos,
    }
    return loss, metrics


def run_epoch(model, loader, normalizer, optimizer, scaler, device, opt, train=True):
    model.train(train)
    totals = {
        "loss": 0.0,
        "top1_a2m": 0.0,
        "top1_m2a": 0.0,
        "pos_cos": 0.0,
    }
    steps = 0

    autocast_ctx = get_autocast_context(device, opt.mixed_precision)
    iterator = tqdm(loader, leave=False, desc="train" if train else "val")

    for batch in iterator:
        motion_norm, audio = move_batch_to_device(batch, device)
        motion_phys = normalizer.unnormalize(motion_norm)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with autocast_ctx:
                audio_latent = model.encode_audio(audio)
                motion_latent = model.encode_motion(motion_phys)
                loss, metrics = symmetric_contrastive_loss(
                    audio_latent, motion_latent, opt.temperature
                )

            if train:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if opt.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if opt.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                    optimizer.step()

        steps += 1
        for key in totals:
            totals[key] += metrics[key].item()

        iterator.set_postfix(
            loss=f"{metrics['loss'].item():.4f}",
            a2m=f"{metrics['top1_a2m'].item():.3f}",
            m2a=f"{metrics['top1_m2a'].item():.3f}",
        )

    if steps == 0:
        return {key: 0.0 for key in totals}
    return {key: value / steps for key, value in totals.items()}


def save_weights_only(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def save_checkpoint(model, optimizer, epoch, metrics, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Pretrain CrossModalMMR with AIST++ pairs")
    parser.add_argument("--data_path", type=str, default="data", help="AIST++ root path")
    parser.add_argument("--feature_type", type=str, default="hybrid", choices=["hybrid", "baseline", "jukebox"])
    parser.add_argument("--seq_len", type=int, default=150)
    parser.add_argument("--audio_dim", type=int, default=803)
    parser.add_argument("--motion_dim", type=int, default=151)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_bins", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--force_reload", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=str, default="runs/mmr")
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--output", type=str, default="weights/mmr_pretrained.pt")
    opt = parser.parse_args()

    set_seed(opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = build_datasets(opt)
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, opt)

    model = CrossModalMMR(
        motion_dim=opt.motion_dim,
        audio_dim=opt.audio_dim,
        latent_dim=opt.latent_dim,
        num_bins=opt.num_bins,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay
    )
    scaler = GradScaler(enabled=(device.type == "cuda" and opt.mixed_precision == "fp16"))

    save_dir = Path(increment_path(Path(opt.project) / opt.exp_name))
    weight_dir = save_dir / "weights"
    best_path = Path(opt.output)
    best_checkpoint_path = weight_dir / "mmr-best-checkpoint.pt"

    best_val_top1 = -1.0
    best_val_loss = float("inf")

    print(f"Training MMR on device: {device}")
    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")
    print(f"Outputs will be saved to: {save_dir}")

    for epoch in range(1, opt.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            normalizer=train_dataset.normalizer,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            opt=opt,
            train=True,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            normalizer=train_dataset.normalizer,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            opt=opt,
            train=False,
        )

        val_top1 = 0.5 * (val_metrics["top1_a2m"] + val_metrics["top1_m2a"])
        improved = (val_top1 > best_val_top1) or (
            abs(val_top1 - best_val_top1) < 1e-8 and val_metrics["loss"] < best_val_loss
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_a2m={val_metrics['top1_a2m']:.3f} | "
            f"val_m2a={val_metrics['top1_m2a']:.3f} | "
            f"val_cos={val_metrics['pos_cos']:.3f}"
        )

        if improved:
            best_val_top1 = val_top1
            best_val_loss = val_metrics["loss"]
            save_weights_only(model, best_path)
            save_checkpoint(model, optimizer, epoch, val_metrics, best_checkpoint_path)
            print(f"✅ Saved best MMR weights (plain state_dict) to {best_path}")
            print(f"🧾 Saved best MMR training checkpoint to {best_checkpoint_path}")

        if epoch % opt.save_every == 0 or epoch == opt.epochs:
            epoch_path = weight_dir / f"mmr-epoch-{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, val_metrics, epoch_path)
            print(f"💾 Saved epoch checkpoint to {epoch_path}")

    print(
        f"🏁 Finished. Best val top1={(best_val_top1 * 100):.2f}% | best val loss={best_val_loss:.4f}"
    )


if __name__ == "__main__":
    main()
