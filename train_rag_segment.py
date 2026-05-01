"""Safe retrieval-consistency fine-tuning for EDGE.

Drop-in replacement for the old train_rag_segment.py.

The old script defaulted to strong segment imitation. This replacement keeps the
same command-line interface but changes the default route to weak retrieval
consistency regularization, adds partial freezing, and stops early when the base
diffusion loss degrades.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from EDGE import EDGE, move_condition_to_device
from dataset.retrieved_segment_dataset import RetrievedSegmentDataset
from model.adan import Adan
from rag_segment_losses import rag_segment_training_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pairs_jsonl", required=True)
    p.add_argument("--val_pairs_jsonl", default="")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--exp_name", default="exp_rag_consistency")
    p.add_argument("--feature_type", default="hybrid", choices=["hybrid", "baseline", "jukebox"])
    p.add_argument("--audio_dim", type=int, default=803)
    p.add_argument("--seq_len", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--save_interval", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.02)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--train_stage", default="stage2", choices=["full", "stage1", "stage2"])

    p.add_argument("--segment_min_len", type=int, default=24)
    p.add_argument("--segment_max_len", type=int, default=60)
    p.add_argument("--prior_feature_mode", default="upper", choices=["arms", "upper", "upper_safe_plus", "all_rot", "full"])
    p.add_argument("--segment_keyframe_width", type=int, default=2)
    p.add_argument("--include_contact_prior", action="store_true")

    # Safer defaults: retrieval consistency, not strong imitation.
    p.add_argument("--base_loss_weight", type=float, default=2.0)
    p.add_argument("--segment_imitation_weight", type=float, default=0.15)
    p.add_argument("--segment_velocity_weight", type=float, default=0.08)
    p.add_argument("--transition_smooth_weight", type=float, default=0.03)
    p.add_argument("--rag_loss_start_epoch", type=int, default=1)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--max_val_batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)

    # New safety controls.
    p.add_argument("--partial_freeze", action="store_true", default=True)
    p.add_argument("--no_partial_freeze", dest="partial_freeze", action="store_false")
    p.add_argument("--unfreeze_last_n_decoder_layers", type=int, default=2)
    p.add_argument("--early_stop_patience", type=int, default=2)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    p.add_argument("--early_stop_metric", default="base", choices=["base", "loss", "val_base", "val_loss"])
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


def _num_decoder_layers(model) -> int:
    seq = getattr(model, "seqTransDecoder", None)
    stack = getattr(seq, "stack", None)
    try:
        return len(stack)
    except Exception:
        return 0


def _should_train_param(name: str, num_layers: int, last_n: int) -> bool:
    train_markers = (
        "input_projection",
        "trajectory_projection",
        "traj_modulate",
        "final_layer",
    )
    if any(marker in name for marker in train_markers):
        return True

    # Last N decoder layers only. Parameter names are usually like:
    # seqTransDecoder.stack.6.self_attn.in_proj_weight
    marker = "seqTransDecoder.stack."
    if marker in name and num_layers > 0:
        try:
            after = name.split(marker, 1)[1]
            layer_idx = int(after.split(".", 1)[0])
            return layer_idx >= max(0, num_layers - int(last_n))
        except Exception:
            return False
    return False


def configure_partial_freeze(edge: EDGE, args):
    accelerator = edge.accelerator
    model = accelerator.unwrap_model(edge.model)
    if not args.partial_freeze:
        for p in model.parameters():
            p.requires_grad = True
        return

    num_layers = _num_decoder_layers(model)
    for name, param in model.named_parameters():
        param.requires_grad = _should_train_param(name, num_layers, args.unfreeze_last_n_decoder_layers)

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("partial_freeze left no trainable parameters")

    if accelerator.is_main_process:
        total = sum(p.numel() for _, p in model.named_parameters())
        trainable_count = sum(p.numel() for _, p in trainable)
        print("🧊 Partial freeze enabled")
        print(f"  decoder_layers={num_layers}, unfreeze_last_n={args.unfreeze_last_n_decoder_layers}")
        print(f"  trainable={trainable_count}/{total} ({trainable_count / max(total, 1) * 100:.2f}%)")
        print("  trainable modules: input_projection, trajectory_projection, traj_modulate, last decoder layers, final_layer")

    # Recreate optimizer after changing requires_grad. EDGE constructed one
    # earlier, but it may include too many parameters for this RAG-v2 stage.
    optimizer = Adan([p for _, p in trainable], lr=args.learning_rate, weight_decay=args.weight_decay)
    edge.optim = accelerator.prepare(optimizer)


def _move_batch(batch, device):
    x, cond, _, _ = batch
    x = x.to(device)
    cond = move_condition_to_device(cond, device)
    return x, cond


def _compute_losses(edge: EDGE, x, cond, epoch: int, args) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
            "rag_segment_imitation": rag_loss.detach(),
            "rag_segment_velocity": rag_loss.detach(),
            "rag_transition_smooth": rag_loss.detach(),
        }
    loss = float(args.base_loss_weight) * base_loss + rag_loss
    logs = {
        "loss": loss.detach(),
        "base": base_loss.detach(),
        "rag": rag_loss.detach(),
        "imit": rag_logs["rag_segment_imitation"].detach(),
        "vel": rag_logs["rag_segment_velocity"].detach(),
        "trans": rag_logs["rag_transition_smooth"].detach(),
    }
    return loss, logs


def _average_logs(totals: Dict[str, float], count: int) -> Dict[str, float]:
    return {k: float(v) / max(int(count), 1) for k, v in totals.items()}


def evaluate(edge: EDGE, loader, epoch: int, args) -> Dict[str, float]:
    if loader is None:
        return {}
    edge.eval()
    totals = {"val_loss": 0.0, "val_base": 0.0, "val_rag": 0.0, "val_imit": 0.0, "val_vel": 0.0, "val_trans": 0.0}
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                break
            x, cond = _move_batch(batch, edge.accelerator.device)
            loss, logs = _compute_losses(edge, x, cond, epoch, args)
            totals["val_loss"] += float(loss.detach().item())
            totals["val_base"] += float(logs["base"].item())
            totals["val_rag"] += float(logs["rag"].item())
            totals["val_imit"] += float(logs["imit"].item())
            totals["val_vel"] += float(logs["vel"].item())
            totals["val_trans"] += float(logs["trans"].item())
            count += 1
    return _average_logs(totals, count) if count else {}


def save_checkpoint(edge: EDGE, normalizer, args, weight_dir: Path, epoch: int, tag: str = "") -> Path:
    weight_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = edge.accelerator.unwrap_model(edge.model)
    name = f"train-{epoch}{('-' + tag) if tag else ''}.pt"
    out = weight_dir / name
    torch.save(
        {
            "model_state_dict": unwrapped.state_dict(),
            "ema_state_dict": edge.diffusion.master_model.state_dict(),
            "normalizer": {
                "mean": normalizer.mean.tolist() if hasattr(normalizer, "mean") else [],
                "std": normalizer.std.tolist() if hasattr(normalizer, "std") else [],
            },
            "args": vars(args),
            "rag_segment_training": True,
            "rag_training_version": "v2_weak_retrieval_consistency",
            "note": "Weak retrieval consistency regularization; not strong segment imitation.",
        },
        out,
    )
    return out


def metric_for_early_stop(train_logs: Dict[str, float], val_logs: Dict[str, float], metric_name: str) -> float:
    if metric_name.startswith("val_"):
        if metric_name in val_logs:
            return float(val_logs[metric_name])
        # Fall back to train metric when no validation loader is supplied.
        fallback = metric_name[len("val_") :]
        return float(train_logs.get(fallback, train_logs.get("base", train_logs.get("loss", 0.0))))
    return float(train_logs.get(metric_name, train_logs.get("base", train_logs.get("loss", 0.0))))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    edge = _make_edge(args)
    accelerator = edge.accelerator
    normalizer = edge.normalizer
    edge.diffusion.normalizer = normalizer

    configure_partial_freeze(edge, args)

    train_loader = _make_loader(args, normalizer, args.pairs_jsonl, shuffle=True)
    val_loader = _make_loader(args, normalizer, args.val_pairs_jsonl, shuffle=False) if args.val_pairs_jsonl else None
    if val_loader is None:
        train_loader = accelerator.prepare(train_loader)
    else:
        train_loader, val_loader = accelerator.prepare(train_loader, val_loader)

    save_dir = Path(args.project) / args.exp_name
    weight_dir = save_dir / "weights"
    if accelerator.is_main_process:
        weight_dir.mkdir(parents=True, exist_ok=True)
        print("📁 save_dir:", save_dir)
        print("🧩 RAG consistency training:", vars(args))

    step = 0
    best_metric = None
    bad_epochs = 0
    best_path = None

    for epoch in range(1, args.epochs + 1):
        edge.train()
        totals = {"loss": 0.0, "base": 0.0, "rag": 0.0, "imit": 0.0, "vel": 0.0, "trans": 0.0}
        count = 0
        pbar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"rag-consistency epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break
            x, cond = _move_batch(batch, accelerator.device)
            with accelerator.accumulate(edge.model):
                loss, logs = _compute_losses(edge, x, cond, epoch, args)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_([p for p in edge.model.parameters() if p.requires_grad], 1.0)
                edge.optim.step()
                edge.optim.zero_grad()

                if accelerator.is_main_process and step % 1 == 0:
                    unwrapped_model = accelerator.unwrap_model(edge.model)
                    edge.diffusion.ema.update_model_average(edge.diffusion.master_model, unwrapped_model)

            for k in totals:
                totals[k] += float(logs[k].item())
            count += 1
            step += 1
            if accelerator.is_main_process:
                pbar.set_postfix(loss=totals["loss"] / max(count, 1), base=totals["base"] / max(count, 1), rag=totals["rag"] / max(count, 1))

        train_logs = _average_logs(totals, count)
        val_logs = evaluate(edge, val_loader, epoch, args)
        current_metric = metric_for_early_stop(train_logs, val_logs, args.early_stop_metric)

        improved = best_metric is None or current_metric < best_metric - float(args.early_stop_min_delta)
        if improved:
            best_metric = current_metric
            bad_epochs = 0
        else:
            bad_epochs += 1

        if accelerator.is_main_process:
            msg = " | ".join(f"{k}={v:.6f}" for k, v in {**train_logs, **val_logs}.items())
            print(f"Epoch {epoch} | {msg} | early_stop_metric={args.early_stop_metric}:{current_metric:.6f} | bad_epochs={bad_epochs}")

        should_save = epoch % args.save_interval == 0 or epoch == args.epochs or improved
        if should_save and accelerator.is_main_process:
            out = save_checkpoint(edge, normalizer, args, weight_dir, epoch)
            print("✅ saved:", out)
            if improved:
                best_path = save_checkpoint(edge, normalizer, args, weight_dir, epoch, tag="best")
                print("🌟 saved best:", best_path)

        accelerator.wait_for_everyone()
        if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
            if accelerator.is_main_process:
                print(f"🛑 Early stopping: {args.early_stop_metric} did not improve for {bad_epochs} epoch(s). best={best_metric:.6f}")
                if best_path is not None:
                    print("best checkpoint:", best_path)
            break

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
