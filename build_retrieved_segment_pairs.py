"""Build retrieved segment training pairs for EDGE RAG-Diffusion stage.

This converts the clip-level MMR-RAG index into JSONL training pairs.  Each row
contains a target motion clip and a retrieved prior clip.  The training dataset
will crop both clips to seq_len and randomly choose a middle segment where the
retrieved clip is exposed as prior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm


def _as_str_array(x):
    return [str(v) for v in list(x)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rag_db", required=True, help="data/dunhuang_rag_db/mmr_rag_index.npz")
    p.add_argument("--out", default="data/rag_segment_pairs/train_pairs.jsonl")
    p.add_argument("--max_pairs", type=int, default=20000)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--source_gap", type=int, default=150)
    p.add_argument("--disallow_same_source", action="store_true")
    p.add_argument("--energy_target", type=float, default=0.55)
    p.add_argument("--energy_band", type=float, default=0.25)
    p.add_argument("--contact_weight", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    db = np.load(args.rag_db, allow_pickle=True)
    emb = np.asarray(db["motion_embeddings"], dtype=np.float32)
    sources = _as_str_array(db["source"])
    frames = np.asarray(db["source_frame"], dtype=np.int32)
    energy = np.asarray(db.get("motion_energy", np.zeros(len(emb))), dtype=np.float32)
    contact = np.asarray(db.get("contact_stability", np.zeros(len(emb))), dtype=np.float32)

    if len(emb) == 0:
        raise RuntimeError(f"empty rag_db: {args.rag_db}")

    # Normalize embeddings for cosine distance.
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    e_min, e_max = float(energy.min()), float(energy.max())
    energy_norm = (energy - e_min) / max(e_max - e_min, 1e-8)

    order = np.arange(len(emb))
    rng.shuffle(order)
    if args.max_pairs > 0:
        order = order[: min(args.max_pairs, len(order))]

    rows: List[dict] = []
    for idx in tqdm(order, desc="build rag segment pairs"):
        sim = emb @ emb[idx]
        nn = np.argsort(-sim)[: max(args.top_k * 4, args.top_k + 1)]
        candidates = []
        for j in nn:
            if int(j) == int(idx):
                continue
            same_source = sources[j] == sources[idx]
            if same_source and args.disallow_same_source:
                continue
            if same_source and abs(int(frames[j]) - int(frames[idx])) < args.source_gap:
                continue
            e_cost = abs(float(energy_norm[j]) - float(args.energy_target)) / max(float(args.energy_band), 1e-6)
            c_cost = 1.0 - float(contact[j])
            score = (1.0 - float(sim[j])) + e_cost + float(args.contact_weight) * c_cost
            candidates.append((score, int(j), e_cost, c_cost, float(sim[j])))
            if len(candidates) >= args.top_k:
                break
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        _, j, e_cost, c_cost, sim_j = candidates[0]
        rows.append({
            "target_source": sources[idx],
            "target_center": int(frames[idx]),
            "prior_source": sources[j],
            "prior_center": int(frames[j]),
            "score": float(candidates[0][0]),
            "cosine_similarity": float(sim_j),
            "target_energy_norm": float(energy_norm[idx]),
            "prior_energy_norm": float(energy_norm[j]),
            "prior_energy_target_cost": float(e_cost),
            "prior_contact_stability": float(contact[j]),
            "prior_contact_cost": float(c_cost),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "rag_db": args.rag_db,
        "pair_count": len(rows),
        "top_k": args.top_k,
        "source_gap": args.source_gap,
        "disallow_same_source": bool(args.disallow_same_source),
        "energy_target": args.energy_target,
        "energy_band": args.energy_band,
    }
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("✅ saved pairs:", out)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
