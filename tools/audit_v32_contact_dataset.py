#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit V32 transition/contact training data before expensive training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--min_contact_rate", type=float, default=0.005)
    parser.add_argument("--max_contact_rate", type=float, default=0.95)
    parser.add_argument("--require_real_samples", type=int, default=1000)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    target = np.asarray(data["target"], np.float32)
    mask = np.asarray(data["mask"], np.float32)
    contact = target[..., :4].clip(0.0, 1.0)
    valid = mask[..., None]
    denominator = valid.sum(axis=(0, 1)).clip(min=1.0)
    channel_rate = (contact * valid).sum(axis=(0, 1)) / denominator
    overall = float((contact * valid).sum() / (valid.sum() * 4.0))
    transitions = np.abs(np.diff(contact, axis=1))
    pair_mask = mask[:, 1:] * mask[:, :-1]
    transition_rate = (
        (transitions * pair_mask[..., None]).sum(axis=(0, 1))
        / (pair_mask.sum() + 1e-6)
    )

    kinds = np.asarray(
        data["sample_kind"]
        if "sample_kind" in data.files
        else ["unknown"] * len(target),
        dtype=object,
    )
    names, counts = np.unique(kinds, return_counts=True)
    real = np.asarray(
        data["real_target"]
        if "real_target" in data.files
        else np.ones((len(target),), np.bool_),
        np.bool_,
    )
    meta = (
        json.loads(str(data["meta"].item()))
        if "meta" in data.files else {}
    )
    report = {
        "version": "v32_contact_dataset_audit",
        "num_samples": int(len(target)),
        "real_samples": int(real.sum()),
        "synthetic_samples": int((~real).sum()),
        "sample_kind_counts": {
            str(k): int(v) for k, v in zip(names, counts)
        },
        "contact_rate_overall": overall,
        "contact_rate_per_channel": channel_rate.astype(float).tolist(),
        "contact_switch_rate_per_channel":
            transition_rate.astype(float).tolist(),
        "length_min": int(np.asarray(data["length"]).min()),
        "length_max": int(np.asarray(data["length"]).max()),
        "length_mean": float(np.asarray(data["length"]).mean()),
        "source_meta": meta,
    }
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if real.sum() < int(args.require_real_samples):
        raise RuntimeError(
            f"Only {int(real.sum())} real samples; "
            f"required={args.require_real_samples}"
        )
    if not (
        float(args.min_contact_rate)
        <= overall
        <= float(args.max_contact_rate)
    ):
        raise RuntimeError(
            f"Contact rate={overall:.6f} is outside "
            f"[{args.min_contact_rate},{args.max_contact_rate}]. "
            "Do not train until contact channels are verified."
        )


if __name__ == "__main__":
    main()
