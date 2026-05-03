"""Heuristic / JSON choreography planner for ChoreoRAG.

This file implements the DanceChat/TM2D-inspired layer:
music feature -> time-ranged choreography script.

It does NOT call online LLMs. For reproducible experiments you can:
1) use heuristic mode, or
2) provide a manually/LLM-written JSON plan with the same schema.

Example:
python music_choreo_planner.py \
  --audio_feature output/example_audio_feat.npy \
  --num_frames 150 \
  --out output/choreo_plan/example_plan.json \
  --style_hint "敦煌舞，飞天感，上肢舒展，重心稳定"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _safe_norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - float(x.min()) if x.size else x
    m = float(x.max()) if x.size else 0.0
    return np.zeros_like(x) if m <= 1e-8 else (x / m).astype(np.float32)


def onset_curve(audio_feature: np.ndarray) -> np.ndarray:
    audio_feature = np.asarray(audio_feature, dtype=np.float32)
    if audio_feature.ndim != 2 or len(audio_feature) == 0:
        return np.zeros((0,), dtype=np.float32)
    if audio_feature.shape[1] > 768:
        x = np.maximum(audio_feature[:, 768], 0.0)
    elif audio_feature.shape[1] >= 35:
        x = np.maximum(audio_feature[:, 0], 0.0) + 0.5 * np.maximum(audio_feature[:, -1], 0.0)
    elif len(audio_feature) > 1:
        x = np.zeros((len(audio_feature),), dtype=np.float32)
        x[1:] = np.linalg.norm(audio_feature[1:] - audio_feature[:-1], axis=-1)
        x[0] = x[1]
    else:
        x = np.zeros((len(audio_feature),), dtype=np.float32)
    return _safe_norm01(x)


def frame_energy(audio_feature: np.ndarray) -> np.ndarray:
    audio_feature = np.asarray(audio_feature, dtype=np.float32)
    if audio_feature.ndim != 2 or len(audio_feature) == 0:
        return np.zeros((0,), dtype=np.float32)
    return _safe_norm01(np.sqrt(np.mean(audio_feature ** 2, axis=-1)))


def _label_energy(v: float) -> str:
    if v < 0.33:
        return "低能量"
    if v < 0.66:
        return "中等能量"
    return "高能量"


def _label_rhythm(v: float) -> str:
    if v < 0.25:
        return "节奏舒缓"
    if v < 0.55:
        return "节奏平稳"
    return "节奏密集"


def _motion_prompt(energy: float, rhythm: float, seg_id: int, num_segments: int, style_hint: str) -> str:
    e = _label_energy(energy)
    if seg_id == 0:
        base = "起势，上肢含蓄展开，重心稳定"
    elif seg_id == num_segments - 1:
        base = "收束，姿态回落，动作平稳结束"
    else:
        if energy >= 0.66 or rhythm >= 0.55:
            base = "上肢大幅舒展，带有旋转或方向变化，保持重心控制"
        elif energy >= 0.33:
            base = "中等幅度转身，上肢舒展，下肢步伐平稳"
        else:
            base = "缓慢过渡，上肢柔和展开，下肢稳定"
    return f"{e}，{base}，{style_hint}"


def build_heuristic_choreo_plan(
    audio_feature: np.ndarray,
    num_frames: int,
    style_hint: str = "敦煌舞，飞天感，上肢舒展，重心稳定",
    max_segments: int = 3,
) -> Dict:
    num_frames = int(num_frames)
    max_segments = max(1, int(max_segments))
    if num_frames <= 150:
        nseg = min(3, max_segments)
    elif num_frames <= 240:
        nseg = min(4, max_segments)
    else:
        nseg = min(5, max_segments)

    onset = onset_curve(audio_feature)
    energy = frame_energy(audio_feature)
    if len(onset) != num_frames and len(onset) > 0:
        onset = np.interp(np.linspace(0, 1, num_frames), np.linspace(0, 1, len(onset)), onset).astype(np.float32)
    if len(energy) != num_frames and len(energy) > 0:
        energy = np.interp(np.linspace(0, 1, num_frames), np.linspace(0, 1, len(energy)), energy).astype(np.float32)
    if len(onset) == 0:
        onset = np.zeros((num_frames,), dtype=np.float32)
    if len(energy) == 0:
        energy = np.zeros((num_frames,), dtype=np.float32)

    boundaries = [int(round(i * (num_frames - 1) / nseg)) for i in range(nseg + 1)]
    boundaries[0] = 0
    boundaries[-1] = num_frames - 1

    segments: List[Dict] = []
    for i in range(nseg):
        s = int(boundaries[i])
        e = int(boundaries[i + 1])
        if i > 0:
            s += 1
        sl = slice(s, e + 1)
        e_mean = float(np.clip(energy[sl].mean() if e >= s else 0.0, 0.0, 1.0))
        o_mean = float(np.clip(onset[sl].mean() if e >= s else 0.0, 0.0, 1.0))
        music_caption = f"{_label_rhythm(o_mean)}，{_label_energy(e_mean)}，{'重音明显' if o_mean > 0.5 else '重音稀疏'}"
        motion_prompt = _motion_prompt(e_mean, o_mean, i, nseg, style_hint)
        segments.append({
            "id": i,
            "start": s,
            "end": e,
            "center": int(round((s + e) / 2)),
            "music_caption": music_caption,
            "motion_prompt": motion_prompt,
            "query_text": f"{music_caption}，{motion_prompt}",
            "energy_target": e_mean,
            "rhythm_target": o_mean,
            "style_tags": ["敦煌舞", "飞天", "上肢舒展", "重心稳定"],
        })

    return {
        "planner": "choreo_unit_rag_v5",
        "mode": "heuristic",
        "num_frames": num_frames,
        "global_style": style_hint,
        "segments": segments,
    }


def load_or_build_choreo_plan(
    plan_json: str = "",
    audio_feature: Optional[np.ndarray] = None,
    num_frames: int = 150,
    style_hint: str = "敦煌舞，飞天感，上肢舒展，重心稳定",
    max_segments: int = 3,
) -> Dict:
    if plan_json:
        path = Path(plan_json)
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    if audio_feature is None:
        audio_feature = np.zeros((num_frames, 1), dtype=np.float32)
    return build_heuristic_choreo_plan(
        audio_feature=audio_feature,
        num_frames=num_frames,
        style_hint=style_hint,
        max_segments=max_segments,
    )


def save_choreo_plan(plan: Dict, out: str):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    print(f"✅ Choreography plan saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_feature", type=str, required=True, help="Path to [T,C] .npy audio feature")
    parser.add_argument("--num_frames", type=int, default=150)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--style_hint", type=str, default="敦煌舞，飞天感，上肢舒展，重心稳定")
    parser.add_argument("--max_segments", type=int, default=3)
    args = parser.parse_args()

    feat = np.load(args.audio_feature).astype(np.float32)
    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = feat[0]
    plan = build_heuristic_choreo_plan(
        feat,
        num_frames=args.num_frames,
        style_hint=args.style_hint,
        max_segments=args.max_segments,
    )
    save_choreo_plan(plan, args.out)


if __name__ == "__main__":
    main()
