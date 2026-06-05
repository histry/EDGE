#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model.v21_style_ranker import load_style_ranker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_json", required=True)
    ap.add_argument("--index_npz", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--blend", type=float, default=0.70, help="Learned ranker weight")
    args = ap.parse_args()

    json_path = Path(args.index_json)
    npz_path = Path(args.index_npz)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    z = np.load(npz_path)
    arrays = {k: z[k] for k in z.files}
    embed = np.asarray(arrays["mmr_embed"], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_style_ranker(args.checkpoint, device)
    scores = []
    with torch.no_grad():
        for start in range(0, len(embed), 1024):
            raw = model(torch.from_numpy(embed[start : start + 1024]).to(device))
            scores.append(torch.sigmoid(raw).cpu().numpy())
    learned = np.concatenate(scores).astype(np.float32)
    old = np.asarray(arrays["style_score"], dtype=np.float32)
    old_lo, old_hi = np.percentile(old, [10, 90])
    old_norm = np.clip((old - old_lo) / (old_hi - old_lo + 1e-8), 0.0, 1.0)
    combined = args.blend * learned + (1.0 - args.blend) * old_norm
    arrays["style_score"] = combined.astype(np.float32)
    np.savez_compressed(npz_path, **arrays)
    for item, learned_score, combined_score in zip(meta["items"], learned, combined):
        item["v21_learned_style_score"] = float(learned_score)
        item["v21_style_score"] = float(combined_score)
    meta["style_ranker_checkpoint"] = str(args.checkpoint)
    meta["style_ranker_blend"] = float(args.blend)
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("rescored:", json_path, npz_path)
    print("combined p10/p50/p90:", np.percentile(combined, [10, 50, 90]))


if __name__ == "__main__":
    main()
