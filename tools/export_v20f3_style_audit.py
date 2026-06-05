#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_motion(path):
    obj = pickle.load(open(path, "rb"))

    for key in [
        "motion",
        "motion_151",
        "canonical_motion",
    ]:
        if key in obj:
            x = np.asarray(
                obj[key],
                dtype=np.float32,
            )
            break
    else:
        raise ValueError(path)

    if x.ndim == 3:
        x = x[0]

    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_k_per_type", type=int, default=3)
    args = ap.parse_args()

    data = json.loads(
        Path(args.db).read_text(
            encoding="utf-8"
        )
    )

    items = (
        data["items"]
        if isinstance(data, dict)
        else data
    )

    groups = defaultdict(list)

    for item in items:
        groups[
            str(
                item.get(
                    "event_type",
                    "unknown",
                )
            )
        ].append(item)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = []

    for event_type, group in sorted(
        groups.items()
    ):
        group.sort(
            key=lambda x: (
                float(
                    x.get(
                        "dunhuang_style_score_v20f3",
                        0.0,
                    )
                ),
                float(
                    x.get("quality_score", 0.0)
                ),
            ),
            reverse=True,
        )

        for rank, item in enumerate(
            group[:args.top_k_per_type],
            1,
        ):
            pkl_path = Path(item["pkl"])
            motion = load_motion(pkl_path)

            name = (
                f"{event_type}_rank{rank:02d}_"
                f"{item.get('event_id', pkl_path.stem)}"
            )

            npy_path = out_dir / f"{name}.npy"
            np.save(
                npy_path,
                motion[None],
            )

            report.append(
                {
                    "name": name,
                    "npy": str(npy_path),
                    "pkl": str(pkl_path),
                    "event_type": event_type,
                    "style_score": item.get(
                        "dunhuang_style_score_v20f3"
                    ),
                    "prototype_similarity": item.get(
                        "prototype_similarity"
                    ),
                    "quality_score": item.get(
                        "quality_score"
                    ),
                }
            )

    (
        out_dir / "audit_report.json"
    ).write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("exported:", len(report))
    print("out:", out_dir)


if __name__ == "__main__":
    main()
