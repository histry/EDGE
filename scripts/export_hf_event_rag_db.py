#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hf_event_contrastive import (
    HFEventContrastiveEncoder,
    intensity_labels_from_features,
    load_hf_event_encoder,
    motion_event_features_torch,
)


def choose_unit_key(db):
    if "unit_motions_physical" in db.files:
        return "unit_motions_physical"
    if "unit_motions" in db.files:
        return "unit_motions"
    if "motions" in db.files:
        return "motions"
    raise KeyError(f"No unit_motions_physical/unit_motions/motions in DB. keys={db.files}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_db", required=True)
    ap.add_argument("--out_db", required=True)
    ap.add_argument("--encoder_ckpt", default="")
    ap.add_argument("--seq_len", type=int, default=45)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--copy_arrays", type=int, default=1)
    args = ap.parse_args()

    in_db = Path(args.in_db)
    out_db = Path(args.out_db)
    out_db.parent.mkdir(parents=True, exist_ok=True)

    db = np.load(in_db, allow_pickle=True)
    unit_key = choose_unit_key(db)
    units = np.asarray(db[unit_key], dtype=np.float32)

    if units.ndim != 3 or units.shape[-1] != 151:
        raise ValueError(f"{unit_key} expected [N,T,151], got {units.shape}")

    units = units[:, : args.seq_len].astype(np.float32)
    n = units.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_hf_event_encoder(args.encoder_ckpt, device=device, freeze=True) if args.encoder_ckpt else None

    feats = []
    embs = []
    labels = []
    scores = []

    print("============================================================")
    print("Export HF Event RAG DB")
    print(f"in_db={in_db}")
    print(f"unit_key={unit_key}")
    print(f"units={n}")
    print(f"encoder={args.encoder_ckpt or 'handcrafted-only'}")
    print("============================================================")

    with torch.no_grad():
        for s in range(0, n, args.batch_size):
            e = min(n, s + args.batch_size)
            motion = torch.from_numpy(units[s:e]).to(device)
            f = motion_event_features_torch(motion)

            if encoder is not None:
                z = encoder.encode_motion_features(f)
            else:
                z = torch.nn.functional.normalize(f, dim=-1)

            lab = intensity_labels_from_features(f)
            score = f[:, :12].abs().mean(dim=-1)

            feats.append(f.cpu().numpy().astype(np.float32))
            embs.append(z.cpu().numpy().astype(np.float32))
            labels.append(lab.cpu().numpy().astype(np.int64))
            scores.append(score.cpu().numpy().astype(np.float32))

            if (s // args.batch_size) % 20 == 0:
                print(f"processed {e}/{n}", flush=True)

    hf_event_features = np.concatenate(feats, axis=0)
    hf_event_embedding = np.concatenate(embs, axis=0)
    hf_event_label = np.concatenate(labels, axis=0)
    hf_event_score = np.concatenate(scores, axis=0)

    payload = {}
    if args.copy_arrays:
        for k in db.files:
            payload[k] = db[k]

    payload.update({
        "hf_event_features": hf_event_features,
        "hf_event_embedding": hf_event_embedding,
        "hf_event_label": hf_event_label,
        "hf_event_score": hf_event_score,
        "hf_event_source_db": np.asarray(str(in_db)),
        "hf_event_encoder_ckpt": np.asarray(args.encoder_ckpt),
        "hf_event_note": np.asarray("High-frequency motion event embeddings for RAG reranking."),
    })

    np.savez_compressed(out_db, **payload)

    summary = {
        "in_db": str(in_db),
        "out_db": str(out_db),
        "unit_key": unit_key,
        "units": int(n),
        "embedding_shape": list(hf_event_embedding.shape),
        "feature_shape": list(hf_event_features.shape),
        "label_counts": {str(i): int((hf_event_label == i).sum()) for i in sorted(set(hf_event_label.tolist()))},
        "encoder_ckpt": args.encoder_ckpt,
    }
    out_json = out_db.with_suffix(".summary.json")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ export done")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
