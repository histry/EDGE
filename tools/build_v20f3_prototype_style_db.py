#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


QUOTAS = {
    "arm_flourish": 700,
    "high_tension": 1000,
    "support_shift": 1000,
    "build_up": 800,
    "release": 500,
    "calm_flow": 1000,
    "neutral_flow": 1600,
    "pose_hold": 150,
}


def get(item, key, default=0.0):
    if key in item:
        try:
            return float(item[key])
        except (TypeError, ValueError):
            pass

    desc = item.get("descriptor", {})
    if isinstance(desc, dict) and key in desc:
        try:
            return float(desc[key])
        except (TypeError, ValueError):
            pass

    return float(default)


def load_motion(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        x = np.load(path, allow_pickle=True)
    else:
        with path.open("rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, dict):
            x = obj
        else:
            x = None
            for key in ["motion", "motion_151", "canonical_motion"]:
                if key in obj:
                    x = obj[key]
                    break

            if x is None:
                raise ValueError(
                    f"No motion field in {path}"
                )

    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 3:
        x = x[0]

    if x.ndim != 2 or x.shape[-1] != 151:
        raise ValueError(
            f"{path}: expected [T,151], got {x.shape}"
        )

    return x


def resample_motion(x: np.ndarray, target_len=48) -> np.ndarray:
    if len(x) == target_len:
        return x.copy()

    old_t = np.linspace(0.0, 1.0, len(x))
    new_t = np.linspace(0.0, 1.0, target_len)

    y = np.empty(
        (target_len, x.shape[1]),
        dtype=np.float32,
    )

    for d in range(x.shape[1]):
        y[:, d] = np.interp(
            new_t,
            old_t,
            x[:, d],
        )

    return y


def unit_norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))

    if n < 1e-8:
        return np.zeros_like(v)

    return v / n


def motion_feature(x: np.ndarray) -> np.ndarray:
    x = resample_motion(x, 48)

    # 原地中心化，但保留 root_y 起伏
    x = x.copy()
    x[:, 4] -= x[0, 4]
    x[:, 6] -= x[0, 6]
    x[:, 5] -= x[0, 5]

    contacts = x[:, 0:4]
    root = x[:, 4:7]
    rotations = x[:, 7:151]

    rot_vel = np.diff(
        rotations,
        axis=0,
        prepend=rotations[:1],
    )

    rot_acc = np.diff(
        rot_vel,
        axis=0,
        prepend=rot_vel[:1],
    )

    # 低频动态特征，保留动作轮廓和身韵
    centered = rotations - rotations.mean(
        axis=0,
        keepdims=True,
    )

    spectrum = np.abs(
        np.fft.rfft(centered, axis=0)
    )

    low_freq = spectrum[1:5].T.reshape(-1)

    contact_switch = np.abs(
        np.diff(
            contacts,
            axis=0,
            prepend=contacts[:1],
        )
    ).sum(axis=0)

    root_feature = np.array(
        [
            np.ptp(root[:, 0]),
            np.ptp(root[:, 1]),
            np.ptp(root[:, 2]),
            np.abs(np.diff(root[:, 1])).mean(),
            np.abs(np.diff(root[:, 1], n=2)).mean(),
        ],
        dtype=np.float32,
    )

    blocks = [
        1.40 * unit_norm(rotations.mean(axis=0)),
        1.00 * unit_norm(rotations.std(axis=0)),
        0.90 * unit_norm(np.abs(rot_vel).mean(axis=0)),
        0.65 * unit_norm(np.abs(rot_acc).mean(axis=0)),
        0.80 * unit_norm(low_freq),
        0.25 * unit_norm(contacts.mean(axis=0)),
        0.25 * unit_norm(contact_switch),
        0.30 * unit_norm(root_feature),
    ]

    feature = np.concatenate(blocks).astype(
        np.float32
    )

    return unit_norm(feature)


def positive_windows(
    motion: np.ndarray,
    window=48,
    stride=24,
):
    if len(motion) <= window:
        return [motion]

    windows = []

    for start in range(
        0,
        max(len(motion) - window + 1, 1),
        stride,
    ):
        windows.append(
            motion[start:start + window]
        )

    if windows[-1].shape[0] < window:
        windows[-1] = motion[-window:]

    if (
        len(motion) >= window
        and not np.array_equal(
            windows[-1],
            motion[-window:],
        )
    ):
        windows.append(motion[-window:])

    return windows


def robust_normalize(values):
    values = np.asarray(values, dtype=np.float32)
    lo, hi = np.percentile(values, [10, 90])

    return np.clip(
        (values - lo) / (hi - lo + 1e-8),
        0.0,
        1.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--positive_dir", required=True)
    ap.add_argument(
        "--positive_pattern",
        action="append",
        required=True,
    )
    ap.add_argument("--max_total", type=int, default=6000)
    ap.add_argument(
        "--min_similarity_percentile",
        type=float,
        default=35.0,
    )
    args = ap.parse_args()

    src = Path(args.input)
    data = json.loads(
        src.read_text(encoding="utf-8")
    )

    items = (
        data["items"]
        if isinstance(data, dict) and "items" in data
        else data
    )

    positive_dir = Path(args.positive_dir)
    positive_files = []

    for pattern in args.positive_pattern:
        positive_files.extend(
            positive_dir.glob(pattern)
        )

    positive_files = sorted(
        set(positive_files)
    )

    if not positive_files:
        raise RuntimeError(
            "No positive prototype motions found"
        )

    prototype_features = []

    for path in positive_files:
        motion = load_motion(path)

        for window in positive_windows(motion):
            prototype_features.append(
                motion_feature(window)
            )

        print(
            "[POS]",
            path,
            "frames=",
            len(motion),
        )

    prototypes = np.stack(
        prototype_features,
        axis=0,
    )

    print(
        "positive files:",
        len(positive_files),
    )
    print(
        "prototype windows:",
        len(prototypes),
    )
    print(
        "feature dim:",
        prototypes.shape[1],
    )

    valid_items = []
    similarities = []

    for index, item in enumerate(items):
        pkl_path = Path(
            item.get(
                "pkl",
                item.get("path", ""),
            )
        )

        if not pkl_path.is_file():
            continue

        try:
            motion = load_motion(pkl_path)
            feat = motion_feature(motion)

            # 所有特征已 L2 normalize，因此点积即 cosine
            similarity = float(
                np.max(prototypes @ feat)
            )

            valid_items.append(
                (index, item)
            )
            similarities.append(similarity)

        except Exception as exc:
            print(
                "[SKIP]",
                pkl_path,
                exc,
            )

    similarities = np.asarray(
        similarities,
        dtype=np.float32,
    )

    sim_norm = robust_normalize(
        similarities
    )

    sim_threshold = float(
        np.percentile(
            similarities,
            args.min_similarity_percentile,
        )
    )

    groups = defaultdict(list)
    rejected = Counter()

    for local_index, (original_index, item) in enumerate(
        valid_items
    ):
        similarity = float(
            similarities[local_index]
        )

        if similarity < sim_threshold:
            rejected["prototype_similarity"] += 1
            continue

        similarity_norm = float(
            sim_norm[local_index]
        )

        old_proxy = float(
            item.get(
                "dunhuang_style_proxy",
                0.5,
            )
        )

        quality = get(
            item,
            "original_quality_score",
            get(item, "quality_score", 0.0),
        )

        safety = get(
            item,
            "safety_score",
            0.0,
        )

        # 原型相似度成为主要敦煌风格依据
        style_score = (
            0.72 * similarity_norm
            + 0.28 * old_proxy
        )

        combined = (
            0.42 * quality
            + 0.18 * safety
            + 0.40 * style_score
        )

        out = dict(item)

        out["prototype_similarity"] = similarity
        out["prototype_similarity_norm"] = (
            similarity_norm
        )
        out["dunhuang_style_score_v20f3"] = (
            float(style_score)
        )

        out["quality_score"] = float(combined)
        out["visual_score"] = float(combined)

        event_type = str(
            out.get(
                "event_type",
                "neutral_flow",
            )
        )

        groups[event_type].append(out)

    selected = []

    for event_type, group in groups.items():
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

        quota = QUOTAS.get(
            event_type,
            600,
        )

        selected.extend(
            group[:quota]
        )

    selected.sort(
        key=lambda x: (
            float(
                x.get(
                    "dunhuang_style_score_v20f3",
                    0.0,
                )
            ),
            float(x.get("quality_score", 0.0)),
        ),
        reverse=True,
    )

    selected = selected[:args.max_total]

    result = {
        "version": "v20f3_prototype_style_gate",
        "source": str(src),
        "positive_files": [
            str(x) for x in positive_files
        ],
        "prototype_window_count": len(prototypes),
        "similarity_threshold": sim_threshold,
        "num_input": len(items),
        "num_valid": len(valid_items),
        "num_selected": len(selected),
        "event_type_counts": dict(
            Counter(
                str(
                    x.get(
                        "event_type",
                        "unknown",
                    )
                )
                for x in selected
            )
        ),
        "rejected": dict(rejected),
        "items": selected,
    }

    dst = Path(args.output)
    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dst.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 60)
    print("selected:", len(selected))
    print(
        "event types:",
        result["event_type_counts"],
    )
    print(
        "similarity threshold:",
        sim_threshold,
    )
    print("saved:", dst)


if __name__ == "__main__":
    main()
