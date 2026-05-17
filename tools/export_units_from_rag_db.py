#!/usr/bin/env python3
"""Export selected 45-frame RAG DB units into a DunhuangDataset-compatible folder.

The generated .pkl files contain a direct 151D key:

  {"motion": [T,151], "unit_id": ..., "original_filename": ...}

Current dataset/dance_dataset.py can read this via the 151D fallback path.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


MOTION_KEYS = ["unit_motions_physical", "motions_physical", "unit_motions", "motions", "motion"]
ID_KEYS = ["unit_ids", "ids", "indices", "unit_indices", "motion_ids"]


def load_unit_list(path: str) -> List[str]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        out.append(text.replace("unit_", ""))
    return out


def find_array(data: np.lib.npyio.NpzFile, keys: List[str]) -> Tuple[str, np.ndarray]:
    for key in keys:
        if key in data:
            return key, np.asarray(data[key])
    raise KeyError(f"None of keys exist in npz: {keys}. Available={list(data.keys())}")


def build_id_map(data: np.lib.npyio.NpzFile, n: int) -> Dict[str, int]:
    for key in ID_KEYS:
        if key in data:
            arr = np.asarray(data[key]).reshape(-1)
            if len(arr) == n:
                return {str(x).replace("unit_", ""): i for i, x in enumerate(arr)}
    # Fallback: row index is the id.
    return {str(i): i for i in range(n)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag_db", required=True)
    ap.add_argument("--unit_list", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--root_center", action="store_true", default=True, help="Subtract first-frame root X/Z from each unit")
    ap.add_argument("--no_root_center", dest="root_center", action="store_false")
    args = ap.parse_args()

    rag_db = Path(args.rag_db)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(rag_db, allow_pickle=True)
    motion_key, motions = find_array(data, MOTION_KEYS)
    if motions.ndim != 3 or motions.shape[-1] != 151:
        raise ValueError(f"{motion_key} must be [N,T,151], got {motions.shape}")

    id_map = build_id_map(data, motions.shape[0])
    units = load_unit_list(args.unit_list)

    missing = []
    written = []
    for unit in units:
        if unit not in id_map:
            missing.append(unit)
            continue
        idx = id_map[unit]
        motion = np.asarray(motions[idx], dtype=np.float32).copy()
        if args.root_center:
            motion[:, 4] -= float(motion[0, 4])
            motion[:, 6] -= float(motion[0, 6])
        out_path = out_dir / f"unit_{unit}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(
                {
                    "motion": motion.astype(np.float32),
                    "unit_id": unit,
                    "rag_row_index": int(idx),
                    "original_filename": f"unit_{unit}",
                    "source_file": str(rag_db),
                    "motion_key": motion_key,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        written.append(str(out_path))

    print(f"motion_key={motion_key}, motions={motions.shape}")
    print(f"written={len(written)} to {out_dir}")
    for p in written:
        print(p)
    if missing:
        print(f"missing={missing}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
