#!/usr/bin/env python3
import argparse, csv, json, pickle, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_hf_event_contrastive import read_wav_mono, audio_segment_to_frame_features
from hf_event_contrastive import (
    load_hf_event_encoder,
    motion_event_features_torch,
    audio_event_features_torch,
)

def load_motion_pkl(path):
    obj = pickle.load(open(path, "rb"))
    m = obj.get("motion_151", obj.get("motion"))
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 3:
        m = m[0]
    return m[:45].astype(np.float32), obj

def norm01(x):
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    return np.clip((x - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", required=True)
    ap.add_argument("--pool", default="data/dunhuang_bvh/support_quality_prior_pool_v15_u45")
    ap.add_argument("--encoder", default="checkpoints/hf_event_contrastive/hf_event_day01/hf_event_encoder.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=45)
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = Path(args.pool)
    files = sorted(pool.glob("*.pkl"))
    if not files:
        raise RuntimeError(f"No pkl prior files found: {pool}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_hf_event_encoder(args.encoder, device=device, freeze=True)
    encoder.eval()

    wav, sr = read_wav_mono(args.music, sr_target=args.sr)
    audio_np = audio_segment_to_frame_features(wav, sr, seq_len=args.seq_len, seconds=args.seq_len / 30.0)
    audio = torch.from_numpy(audio_np).float().to(device).unsqueeze(0)

    with torch.no_grad():
        a_feat = audio_event_features_torch(audio, target_len=args.seq_len)
        a_emb = encoder.encode_audio_features(a_feat)
        a_emb = torch.nn.functional.normalize(a_emb, dim=-1)

    rows = []
    motion_batch = []
    meta = []

    for p in files:
        m, obj = load_motion_pkl(p)
        motion_batch.append(m)
        meta.append((p, obj))

    motion = torch.from_numpy(np.stack(motion_batch, axis=0)).float().to(device)

    with torch.no_grad():
        m_feat = motion_event_features_torch(motion)
        m_emb = encoder.encode_motion_features(m_feat)
        m_emb = torch.nn.functional.normalize(m_emb, dim=-1)
        sim = (m_emb * a_emb).sum(dim=-1).detach().cpu().numpy().astype(np.float32)

    support_q = np.asarray([float(obj.get("support_prior_quality", 0.0)) for _, obj in meta], dtype=np.float32)
    hf_score = np.asarray([float(obj.get("hf_event_score", 0.0)) for _, obj in meta], dtype=np.float32)
    tail = np.asarray([float(obj.get("tail_activity_ratio", 0.0)) for _, obj in meta], dtype=np.float32)
    jump = np.asarray([float(obj.get("jump_p95", 0.0)) for _, obj in meta], dtype=np.float32)
    jerk = np.asarray([float(obj.get("jerk_p95", 0.0)) for _, obj in meta], dtype=np.float32)

    sim_n = norm01(sim)
    hf_n = norm01(hf_score)
    jump_n = norm01(jump)
    jerk_n = norm01(jerk)

    final = (
        0.45 * support_q
        + 0.30 * sim_n
        + 0.15 * hf_n
        + 0.10 * np.clip(tail, 0.0, 1.0)
        - 0.08 * jump_n
        - 0.08 * jerk_n
    )

    order = np.argsort(-final)[: args.top_k]

    out_list = out_dir / f"{Path(args.music).stem}_top{args.top_k}_prior_list.txt"
    out_csv = out_dir / f"{Path(args.music).stem}_top{args.top_k}_prior_scores.csv"

    with out_list.open("w", encoding="utf-8") as fp:
        for i in order:
            fp.write(str(meta[int(i)][0]) + "\n")

    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=[
            "rank", "file", "unit_index", "final_score", "audio_motion_sim",
            "support_prior_quality", "hf_event_score", "tail_activity_ratio", "jump_p95", "jerk_p95"
        ])
        w.writeheader()
        for rank, i in enumerate(order):
            p, obj = meta[int(i)]
            row = {
                "rank": rank,
                "file": str(p),
                "unit_index": obj.get("unit_index", ""),
                "final_score": float(final[int(i)]),
                "audio_motion_sim": float(sim[int(i)]),
                "support_prior_quality": float(support_q[int(i)]),
                "hf_event_score": float(hf_score[int(i)]),
                "tail_activity_ratio": float(tail[int(i)]),
                "jump_p95": float(jump[int(i)]),
                "jerk_p95": float(jerk[int(i)]),
            }
            w.writerow(row)

    summary = {
        "music": args.music,
        "pool": str(pool),
        "encoder": args.encoder,
        "top_k": args.top_k,
        "prior_list": str(out_list),
        "score_csv": str(out_csv),
    }
    (out_dir / f"{Path(args.music).stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("✅ selected music-aware V15 priors")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Top priors:")
    for rank, i in enumerate(order[: min(8, len(order))]):
        print(rank, meta[int(i)][0], "score", float(final[int(i)]), "sim", float(sim[int(i)]), "q", float(support_q[int(i)]))

if __name__ == "__main__":
    main()
