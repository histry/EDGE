"""Fine-tune EDGE with retrieved segment prior imitation.

This is a training-level RAG-Diffusion stage.  It does not replace the stable
inference scripts.  It fine-tunes from your current best EDGE checkpoint so the
model learns to use retrieved continuous clips during denoising, instead of only
receiving sparse center-pose keyframes at inference.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from EDGE import EDGE, move_condition_to_device
from dataset.retrieved_segment_dataset import RetrievedSegmentDataset
from rag_segment_losses import rag_segment_training_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pairs_jsonl", required=True)
    p.add_argument("--val_pairs_jsonl", default="")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--exp_name", default="exp_rag_segment")
    p.add_argument("--feature_type", default="hybrid", choices=["hybrid", "baseline", "jukebox"])
    p.add_argument("--audio_dim", type=int, default=803)
    p.add_argument("--seq_len", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--save_interval", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.02)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--train_stage", default="stage2", choices=["full", "stage1", "stage2"])

    p.add_argument("--segment_min_len", type=int, default=24)
    p.add_argument("--segment_max_len", type=int, default=60)
    p.add_argument("--prior_feature_mode", default="upper", choices=["arms", "upper", "upper_safe_plus", "all_rot", "full"])
    p.add_argument("--segment_keyframe_width", type=int, default=2)
    p.add_argument("--include_contact_prior", action="store_true")

    p.add_argument("--base_loss_weight", type=float, default=1.0)
    p.add_argument("--segment_imitation_weight", type=float, default=1.0)
    p.add_argument("--segment_velocity_weight", type=float, default=0.5)
    p.add_argument("--transition_smooth_weight", type=float, default=0.2)
    p.add_argument("--rag_loss_start_epoch", type=int, default=1)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def _make_edge(args) -> EDGE:
    return EDGE(
        feature_type=args.feature_type,
        checkpoint_path=args.checkpoint,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        audio_dim=args.audio_dim,
        seq_len=args.seq_len,
        mixed_precision=args.mixed_precision,
        cond_drop_prob=0.0,
        audio_pairing_mode="none",
        mmr_loss_weight=0.0,
        keyframe_condition_prob=0.7,
        keyframe_condition_width=args.segment_keyframe_width,
        keyframe_loss_weight=2.0,
        contact_loss_weight=0.8,
        foot_loss_weight=2.5,
        sync_loss_weight=1.2,
        mid_keyframe_condition_prob=0.7,
        mid_keyframe_count=2,
        mid_keyframe_condition_width=1,
        mid_keyframe_selection="motion_peak",
        beat_guidance_weight=0.0,
        trajectory_loss_weight=1.0,
        trajectory_velocity_loss_weight=0.25,
        hard_keyframe_project=False,
        train_stage=args.train_stage,
        strict_audio_checkpoint=False,
    )


def _make_loader(args, normalizer, pairs_jsonl: str, shuffle: bool):
    ds = RetrievedSegmentDataset(
        pairs_jsonl=pairs_jsonl,
        normalizer=normalizer,
        seq_len=args.seq_len,
        audio_dim=args.audio_dim,
        segment_min_len=args.segment_min_len,
        segment_max_len=args.segment_max_len,
        prior_feature_mode=args.prior_feature_mode,
        protect_width=args.segment_keyframe_width,
        seed=args.seed,
        include_contact_prior=args.include_contact_prior,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    edge = _make_edge(args)
    accelerator = edge.accelerator
    normalizer = edge.normalizer
    edge.diffusion.normalizer = normalizer

    train_loader = _make_loader(args, normalizer, args.pairs_jsonl, shuffle=True)
    val_loader = None
    if args.val_pairs_jsonl:
        val_loader = _make_loader(args, normalizer, args.val_pairs_jsonl, shuffle=False)

    if val_loader is None:
        train_loader = accelerator.prepare(train_loader)
    else:
        train_loader, val_loader = accelerator.prepare(train_loader, val_loader)

    if accelerator.is_main_process:
        save_dir = Path(args.project) / args.exp_name
        weight_dir = save_dir / "weights"
        weight_dir.mkdir(parents=True, exist_ok=True)
        print("📁 save_dir:", save_dir)
        print("🧩 RAG segment training:", vars(args))
    else:
        save_dir = Path(args.project) / args.exp_name
        weight_dir = save_dir / "weights"

    step = 0
    for epoch in range(1, args.epochs + 1):
        edge.train()
        totals = {"loss": 0.0, "base": 0.0, "rag": 0.0, "imit": 0.0, "vel": 0.0, "trans": 0.0}
        count = 0
        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"rag-seg epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break
            x, cond, _, _ = batch
            x = x.to(accelerator.device)
            cond = move_condition_to_device(cond, accelerator.device)

            with accelerator.accumulate(edge.model):
                base_loss, _ = edge.diffusion(x, cond, current_epoch=epoch)
                if epoch >= args.rag_loss_start_epoch:
                    rag_loss, rag_logs = rag_segment_training_loss(
                        edge.diffusion,
                        x,
                        cond,
                        keyframe_width=args.segment_keyframe_width,
                        imitation_weight=args.segment_imitation_weight,
                        velocity_weight=args.segment_velocity_weight,
                        transition_weight=args.transition_smooth_weight,
                    )
                else:
                    rag_loss = base_loss.new_tensor(0.0)
                    rag_logs = {
                        "rag_segment_imitation": rag_loss,
                        "rag_segment_velocity": rag_loss,
                        "rag_transition_smooth": rag_loss,
                    }

                loss = float(args.base_loss_weight) * base_loss + rag_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(edge.model.parameters(), 1.0)
                edge.optim.step()
                edge.optim.zero_grad()

                if accelerator.is_main_process and step % 1 == 0:
                    unwrapped_model = accelerator.unwrap_model(edge.model)
                    edge.diffusion.ema.update_model_average(edge.diffusion.master_model, unwrapped_model)

            totals["loss"] += float(loss.detach().item())
            totals["base"] += float(base_loss.detach().item())
            totals["rag"] += float(rag_loss.detach().item())
            totals["imit"] += float(rag_logs["rag_segment_imitation"].item())
            totals["vel"] += float(rag_logs["rag_segment_velocity"].item())
            totals["trans"] += float(rag_logs["rag_transition_smooth"].item())
            count += 1
            step += 1
            if accelerator.is_main_process:
                pbar.set_postfix(loss=totals["loss"] / max(count, 1), rag=totals["rag"] / max(count, 1))

        if accelerator.is_main_process:
            msg = " | ".join(f"{k}={v / max(count, 1):.6f}" for k, v in totals.items())
            print(f"Epoch {epoch} | {msg}")

        should_save = epoch % args.save_interval == 0 or epoch == args.epochs
        if should_save and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(edge.model)
            ckpt = {
                "model_state_dict": unwrapped.state_dict(),
                "ema_state_dict": edge.diffusion.master_model.state_dict(),
                "normalizer": {
                    "mean": normalizer.mean.tolist() if hasattr(normalizer, "mean") else [],
                    "std": normalizer.std.tolist() if hasattr(normalizer, "std") else [],
                },
                "args": vars(args),
                "rag_segment_training": True,
            }
            out = weight_dir / f"train-{epoch}.pt"
            torch.save(ckpt, out)
            print("✅ saved:", out)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
