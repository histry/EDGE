#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 Emotion-conditioned V17 Refiner 训练集。

输入：
- V16C/V17 scheduler 输出的 .npy motion 文件，形状 [1,T,151] 或 [T,151]
- 音乐文件或预提取音乐情感 .npy

输出：
- 每个样本一个 .npz：
  noisy      [T,151]  人工扰动后的输入，用于学习 seam correction
  target     [T,151]  V16C 稳定短语作为 teacher
  music      [T,8]    音乐情感语义条件
  seam_mask  [T,1]    接缝附近权重
  meta       JSON 字符串

训练思想：
- V16C 是老师，Refiner 是学生；
- 通过给接缝附近添加小扰动，训练模型学会“局部修复 + 保持风格”；
- 不让模型重写整段动作，避免风格被平均化。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np

CONTACT = slice(0, 4)
ROOT_X = 4
ROOT_Y = 5
ROOT_Z = 6
ROT = slice(7, 151)


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).replace(";", ",").split(",") if x.strip()]


def parse_starts(text: str) -> List[int]:
    return [int(float(x.strip())) for x in str(text).replace(";", ",").split(",") if x.strip()]


def load_motion(path: Path) -> np.ndarray:
    x = np.load(path)
    if x.ndim == 3:
        x = x[0]
    if x.ndim != 2 or x.shape[-1] != 151:
        raise ValueError(f"{path} should be [T,151] or [1,T,151], got {x.shape}")
    return x.astype(np.float32)


def resize_feature(feat: np.ndarray, length: int) -> np.ndarray:
    feat = np.asarray(feat, dtype=np.float32)
    if feat.ndim == 1:
        feat = feat[:, None]
    if feat.shape[0] == length:
        return feat.astype(np.float32)
    src = np.linspace(0.0, 1.0, feat.shape[0], dtype=np.float32)
    dst = np.linspace(0.0, 1.0, length, dtype=np.float32)
    out = np.stack([np.interp(dst, src, feat[:, i]) for i in range(feat.shape[1])], axis=-1)
    return out.astype(np.float32)


def build_seam_mask(T: int, starts: List[int], radius: int = 8) -> np.ndarray:
    m = np.zeros((T, 1), dtype=np.float32)
    for s in starts:
        if s <= 0 or s >= T:
            continue
        lo = max(0, int(s) - radius)
        hi = min(T, int(s) + radius + 1)
        for t in range(lo, hi):
            d = abs(t - s) / max(radius, 1)
            m[t, 0] = max(m[t, 0], 1.0 - d)
    return m


def corrupt_motion(target: np.ndarray, seam_mask: np.ndarray, noise_std: float, smooth_prob: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = target.copy()
    T = len(target)
    mask = seam_mask.reshape(T, 1).astype(np.float32)

    # 接缝附近加轻微旋转扰动，模拟边界突变 / 局部噪声。
    noise = rng.normal(0.0, noise_std, size=target.shape).astype(np.float32)
    channel = np.zeros((1, 151), dtype=np.float32)
    channel[:, ROOT_Y] = 0.25
    channel[:, ROT] = 1.0
    noisy = noisy + noise * mask * channel

    # 随机把接缝附近局部做过度平滑，训练模型恢复动作张力。
    if rng.random() < smooth_prob:
        rot = noisy[:, ROT].copy()
        k = 5
        pad = k // 2
        padded = np.pad(rot, ((pad, pad), (0, 0)), mode="edge")
        sm = np.stack([padded[i:i+k].mean(axis=0) for i in range(T)], axis=0).astype(np.float32)
        noisy[:, ROT] = (1.0 - mask) * noisy[:, ROT] + mask * sm

    noisy[:, 0:4] = target[:, 0:4]
    noisy[:, ROOT_X] = target[:, ROOT_X]
    noisy[:, ROOT_Z] = target[:, ROOT_Z]
    return noisy.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_dir", required=True, help="包含 V16C/V17 .npy 输出的目录")
    ap.add_argument("--audio", default="", help="单个或多个 wav，逗号分隔")
    ap.add_argument("--music_npy", default="", help="已提取的音乐情感特征 npy，优先级高于 audio")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--starts", default="0,32,64,96")
    ap.add_argument("--seam_radius", type=int, default=8)
    ap.add_argument("--noise_std", type=float, default=0.015)
    ap.add_argument("--smooth_prob", type=float, default=0.50)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    motions = sorted(Path(args.motion_dir).glob("*.npy"))
    if args.limit > 0:
        motions = motions[: args.limit]
    if not motions:
        raise RuntimeError(f"No .npy motion files found in {args.motion_dir}")

    starts = parse_starts(args.starts)

    music_features = []
    if args.music_npy:
        for p in parse_list(args.music_npy):
            music_features.append((Path(p).stem, np.load(p).astype(np.float32)))
    elif args.audio:
        from tools.extract_music_emotion_features import extract_music_emotion_features
        for a in parse_list(args.audio):
            # 先用 150 帧提取，后面按 motion T resize。
            feat, summary = extract_music_emotion_features(a, num_frames=150, fps=args.fps)
            music_features.append((Path(a).stem, feat))
    else:
        music_features.append(("zero_music", np.zeros((150, 8), dtype=np.float32)))

    count = 0
    index = []
    for mi, mp in enumerate(motions):
        target = load_motion(mp)
        T = len(target)
        seam_mask = build_seam_mask(T, starts, radius=args.seam_radius)
        for ai, (audio_stem, mf) in enumerate(music_features):
            music = resize_feature(mf, T)
            seed = 100000 + mi * 1000 + ai
            noisy = corrupt_motion(target, seam_mask, args.noise_std, args.smooth_prob, seed=seed)

            meta = {
                "motion": str(mp),
                "audio_stem": audio_stem,
                "starts": starts,
                "seam_radius": int(args.seam_radius),
                "noise_std": float(args.noise_std),
                "smooth_prob": float(args.smooth_prob),
            }
            out = out_dir / f"{mp.stem}__{audio_stem}__sample{count:05d}.npz"
            np.savez_compressed(
                out,
                noisy=noisy.astype(np.float32),
                target=target.astype(np.float32),
                music=music.astype(np.float32),
                seam_mask=seam_mask.astype(np.float32),
                meta=json.dumps(meta, ensure_ascii=False),
            )
            index.append({"path": str(out), **meta})
            count += 1

    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved dataset: {out_dir}")
    print(f"samples: {count}")


if __name__ == "__main__":
    main()
