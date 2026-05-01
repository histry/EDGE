"""Train true Music-Motion Retrieval (MMR) on paired AIST++ clips.

Input index JSON format:
[
  {"audio_feature": ".../clip_audio.npy", "motion": ".../clip_motion.npy"},
  ...
]
Each item must be a paired music-motion clip. This is the supervised step that
turns MMR from a proxy score into a real shared latent-space retrieval model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.mmr_encoder import MMRConfig, MusicMotionRetrievalModel, save_mmr_checkpoint
from losses.mmr_loss import symmetric_infonce_loss, retrieval_metrics
from mmr_data_utils import load_audio_feature, load_motion_151, resample_sequence


class PairedMusicMotionDataset(Dataset):
    def __init__(self, index_json: str, seq_len: int = 150):
        self.items = json.load(open(index_json, "r", encoding="utf-8"))
        self.seq_len = int(seq_len)
        if not self.items:
            raise ValueError(f"Empty index: {index_json}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio = load_audio_feature(item["audio_feature"], key=item.get("audio_key"))
        motion = load_motion_151(item["motion"], key=item.get("motion_key"))
        audio = resample_sequence(audio, self.seq_len)
        motion = resample_sequence(motion, self.seq_len)
        return {
            "audio": torch.from_numpy(audio).float(),
            "motion": torch.from_numpy(motion).float(),
        }


def run_eval(model, loader, device, max_batches=20):
    model.eval()
    losses = []
    metrics_acc = {}
    n = 0
    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            if bidx >= max_batches:
                break
            audio = batch["audio"].to(device)
            motion = batch["motion"].to(device)
            out = model(audio, motion)
            loss = symmetric_infonce_loss(out["logits"])
            metrics = retrieval_metrics(out["logits"], topk=(1, 5, 10))
            losses.append(float(loss.detach().cpu()))
            for k, v in metrics.items():
                metrics_acc[k] = metrics_acc.get(k, 0.0) + v
            n += 1
    if n == 0:
        return {"loss": 0.0}
    result = {"loss": float(np.mean(losses))}
    result.update({k: v / n for k, v in metrics_acc.items()})
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index_json", required=True)
    p.add_argument("--val_index_json", default="")
    p.add_argument("--out", default="runs/mmr/mmr_aist_pretrain.pt")
    p.add_argument("--seq_len", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--audio_dim", type=int, default=803)
    p.add_argument("--motion_dim", type=int, default=151)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_every", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_ds = PairedMusicMotionDataset(args.index_json, seq_len=args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = None
    if args.val_index_json:
        val_ds = PairedMusicMotionDataset(args.val_index_json, seq_len=args.seq_len)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)

    config = MMRConfig(
        audio_dim=args.audio_dim,
        motion_dim=args.motion_dim,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_len=max(512, args.seq_len),
    )
    model = MusicMotionRetrievalModel(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_val = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_r1 = 0.0
        count = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            audio = batch["audio"].to(device)
            motion = batch["motion"].to(device)
            out = model(audio, motion)
            loss = symmetric_infonce_loss(out["logits"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            metrics = retrieval_metrics(out["logits"], topk=(1,))
            total_loss += float(loss.detach().cpu())
            total_r1 += metrics["r_at_1"]
            count += 1
            pbar.set_postfix(loss=total_loss / count, r1=total_r1 / count)

        log = {"train_loss": total_loss / max(count, 1), "train_r1": total_r1 / max(count, 1)}
        if val_loader is not None:
            val = run_eval(model, val_loader, device)
            log.update({f"val_{k}": v for k, v in val.items()})
            current = val.get("r_at_1", 0.0)
            if best_val is None or current > best_val:
                best_val = current
                save_mmr_checkpoint(args.out.replace(".pt", "_best.pt"), model, {"args": vars(args), "epoch": epoch, "log": log})
        print("Epoch", epoch, log)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_mmr_checkpoint(args.out, model, {"args": vars(args), "epoch": epoch, "log": log})
            print("saved:", args.out)


if __name__ == "__main__":
    main()
