import argparse
import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dance_dataset import AISTPPDataset
from model.mmr_model import CrossModalMMR


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


def build_loader(dataset, opt):
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=opt.num_workers > 0,
    )


def get_autocast_context(device, precision):
    if device.type != "cuda" or precision == "no":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def load_model(opt, device):
    ckpt = torch.load(opt.ckpt, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    model = CrossModalMMR(
        motion_dim=opt.motion_dim,
        audio_dim=opt.audio_dim,
        latent_dim=opt.latent_dim,
        num_bins=opt.num_bins,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, ckpt


def collect_embeddings(model, loader, normalizer, device, opt):
    audio_chunks = []
    motion_chunks = []
    autocast_ctx = get_autocast_context(device, opt.mixed_precision)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="encoding", leave=False)):
            motion_norm, cond, _, _ = batch
            audio = cond["audio"].to(device, non_blocking=True)
            motion_norm = motion_norm.to(device, non_blocking=True)
            motion_phys = normalizer.unnormalize(motion_norm)

            with autocast_ctx:
                audio_latent = model.encode_audio(audio)
                motion_latent = model.encode_motion(motion_phys)

            audio_chunks.append(F.normalize(audio_latent.float(), dim=-1).cpu())
            motion_chunks.append(F.normalize(motion_latent.float(), dim=-1).cpu())

            if opt.max_batches > 0 and (batch_idx + 1) >= opt.max_batches:
                break

    return torch.cat(audio_chunks, dim=0), torch.cat(motion_chunks, dim=0)


def topk_hits(sim, k):
    k = min(k, sim.shape[1])
    topk = sim.topk(k=k, dim=1).indices
    labels = torch.arange(sim.shape[0]).unsqueeze(1)
    return (topk == labels).any(dim=1).float().mean().item()


def compute_global_metrics(audio_emb, motion_emb, temperature, chunk_size):
    n = audio_emb.shape[0]
    labels = torch.arange(n)

    pos_cos = (audio_emb * motion_emb).sum(dim=-1)
    loss_sum = 0.0
    top1_a2m_hits = 0
    top5_a2m_hits = 0
    hardest_neg_sum_a2m = 0.0
    neg_mean_sum_a2m = 0.0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim = audio_emb[start:end] @ motion_emb.t()
        logits = sim / temperature
        chunk_labels = labels[start:end]

        loss_sum += F.cross_entropy(logits, chunk_labels, reduction="sum").item()
        top1_a2m_hits += (sim.argmax(dim=1) == chunk_labels).sum().item()
        top5 = sim.topk(k=min(5, n), dim=1).indices
        top5_a2m_hits += (top5 == chunk_labels.unsqueeze(1)).any(dim=1).sum().item()

        row_idx = torch.arange(end - start)
        pos_chunk = sim[row_idx, chunk_labels]
        sim_wo_diag = sim.clone()
        sim_wo_diag[row_idx, chunk_labels] = -1e9
        hardest_neg_sum_a2m += sim_wo_diag.max(dim=1).values.sum().item()
        neg_mean_sum_a2m += ((sim.sum(dim=1) - pos_chunk) / max(1, n - 1)).sum().item()

    top1_m2a_hits = 0
    top5_m2a_hits = 0
    hardest_neg_sum_m2a = 0.0
    neg_mean_sum_m2a = 0.0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim = motion_emb[start:end] @ audio_emb.t()
        chunk_labels = labels[start:end]

        top1_m2a_hits += (sim.argmax(dim=1) == chunk_labels).sum().item()
        top5 = sim.topk(k=min(5, n), dim=1).indices
        top5_m2a_hits += (top5 == chunk_labels.unsqueeze(1)).any(dim=1).sum().item()

        row_idx = torch.arange(end - start)
        pos_chunk = sim[row_idx, chunk_labels]
        sim_wo_diag = sim.clone()
        sim_wo_diag[row_idx, chunk_labels] = -1e9
        hardest_neg_sum_m2a += sim_wo_diag.max(dim=1).values.sum().item()
        neg_mean_sum_m2a += ((sim.sum(dim=1) - pos_chunk) / max(1, n - 1)).sum().item()

    metrics = {
        "num_samples": n,
        "loss": loss_sum / n,
        "top1_a2m": top1_a2m_hits / n,
        "top5_a2m": top5_a2m_hits / n,
        "top1_m2a": top1_m2a_hits / n,
        "top5_m2a": top5_m2a_hits / n,
        "pos_cos": pos_cos.mean().item(),
        "hardest_neg_cos_a2m": hardest_neg_sum_a2m / n,
        "hardest_neg_cos_m2a": hardest_neg_sum_m2a / n,
        "mean_neg_cos_a2m": neg_mean_sum_a2m / n,
        "mean_neg_cos_m2a": neg_mean_sum_m2a / n,
    }
    metrics["margin_a2m"] = metrics["pos_cos"] - metrics["hardest_neg_cos_a2m"]
    metrics["margin_m2a"] = metrics["pos_cos"] - metrics["hardest_neg_cos_m2a"]
    metrics["random_top1"] = 1.0 / n
    metrics["random_top5"] = min(5, n) / n
    metrics["top1_gain_a2m"] = metrics["top1_a2m"] / max(metrics["random_top1"], 1e-12)
    metrics["top1_gain_m2a"] = metrics["top1_m2a"] / max(metrics["random_top1"], 1e-12)
    metrics["balance_gap"] = abs(metrics["top1_a2m"] - metrics["top1_m2a"])
    return metrics


def verdict(metrics):
    passed = []
    warned = []

    if metrics["top1_gain_a2m"] >= 10 and metrics["top1_gain_m2a"] >= 10:
        passed.append("top1 已明显高于随机基线 10x")
    else:
        warned.append("top1 还没有稳定高出随机基线 10x")

    if metrics["pos_cos"] >= 0.45:
        passed.append("正样本余弦相似度达到可用区间")
    else:
        warned.append("正样本余弦相似度偏低，隐空间还不够稳")

    if metrics["balance_gap"] <= 0.08:
        passed.append("audio->motion 与 motion->audio 较平衡")
    else:
        warned.append("双向检索不平衡，可能影响主训练稳定性")

    if metrics["margin_a2m"] > 0 and metrics["margin_m2a"] > 0:
        passed.append("正样本和最难负样本已经拉开")
    else:
        warned.append("正负样本分离不够，存在塌缩风险")

    if metrics["top1_a2m"] >= 0.20 and metrics["top1_m2a"] >= 0.20 and metrics["pos_cos"] >= 0.55:
        overall = "较好：可以比较放心地接回主扩散训练"
    elif metrics["top1_gain_a2m"] >= 10 and metrics["top1_gain_m2a"] >= 10 and metrics["pos_cos"] >= 0.45:
        overall = "通过：已经可用，建议先小权重接入主训练"
    else:
        overall = "待提升：建议继续训练或调参后再接回主训练"

    return overall, passed, warned


def print_metrics(metrics, ckpt):
    print("\n========== MMR Check ==========")
    if isinstance(ckpt, dict):
        if "epoch" in ckpt:
            print(f"checkpoint epoch      : {ckpt['epoch']}")
        if "metrics" in ckpt:
            print(f"saved checkpoint stats: {ckpt['metrics']}")

    print(f"samples               : {metrics['num_samples']}")
    print(f"global val loss        : {metrics['loss']:.4f}")
    print(f"top1 a2m               : {metrics['top1_a2m']:.4%}")
    print(f"top5 a2m               : {metrics['top5_a2m']:.4%}")
    print(f"top1 m2a               : {metrics['top1_m2a']:.4%}")
    print(f"top5 m2a               : {metrics['top5_m2a']:.4%}")
    print(f"random top1 baseline   : {metrics['random_top1']:.4%}")
    print(f"random top5 baseline   : {metrics['random_top5']:.4%}")
    print(f"top1 gain a2m          : {metrics['top1_gain_a2m']:.2f}x")
    print(f"top1 gain m2a          : {metrics['top1_gain_m2a']:.2f}x")
    print(f"positive cosine        : {metrics['pos_cos']:.4f}")
    print(f"hardest neg cosine a2m : {metrics['hardest_neg_cos_a2m']:.4f}")
    print(f"hardest neg cosine m2a : {metrics['hardest_neg_cos_m2a']:.4f}")
    print(f"mean neg cosine a2m    : {metrics['mean_neg_cos_a2m']:.4f}")
    print(f"mean neg cosine m2a    : {metrics['mean_neg_cos_m2a']:.4f}")
    print(f"margin a2m             : {metrics['margin_a2m']:.4f}")
    print(f"margin m2a             : {metrics['margin_m2a']:.4f}")
    print(f"balance gap            : {metrics['balance_gap']:.4f}")

    overall, passed, warned = verdict(metrics)
    print("\n结论:", overall)
    for item in passed:
        print(f"[PASS] {item}")
    for item in warned:
        print(f"[WARN] {item}")
    print("================================\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate mmr_pretrained.pt on AIST++ retrieval")
    parser.add_argument("--ckpt", type=str, default="weights/mmr_pretrained.pt")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"])
    parser.add_argument("--feature_type", type=str, default="hybrid", choices=["hybrid", "baseline", "jukebox"])
    parser.add_argument("--seq_len", type=int, default=150)
    parser.add_argument("--audio_dim", type=int, default=803)
    parser.add_argument("--motion_dim", type=int, default=151)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_bins", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--force_reload", action="store_true")
    parser.add_argument("--max_batches", type=int, default=-1, help="debug only; -1 means full split")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(opt, device)
    train_dataset, val_dataset = build_datasets(opt)
    dataset = val_dataset if opt.split == "val" else train_dataset
    loader = build_loader(dataset, opt)

    print(f"Checking MMR on device: {device}")
    print(f"Split: {opt.split} | Samples: {len(dataset)} | CKPT: {opt.ckpt}")

    audio_emb, motion_emb = collect_embeddings(
        model=model,
        loader=loader,
        normalizer=train_dataset.normalizer,
        device=device,
        opt=opt,
    )
    metrics = compute_global_metrics(
        audio_emb=audio_emb,
        motion_emb=motion_emb,
        temperature=opt.temperature,
        chunk_size=opt.chunk_size,
    )
    print_metrics(metrics, ckpt)


if __name__ == "__main__":
    main()
