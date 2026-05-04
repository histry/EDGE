"""Heuristic / JSON choreography planner for ChoreoRAG.

Replacement version for tension-aware ChoreoRAG.

The file still supports the previous reproducible heuristic mode and manual JSON
mode.  It now adds a per-segment tension/phase control layer:

- tension_target in [0, 1]
- phase in {attack, flow, pose}
- min_expressiveness / min_energy suggestions
- stability_weight_scale / expressiveness_weight_scale suggestions

These fields are inert unless auto_keyframe_planner.py is run with
EDGE_TENSION_AWARE_PLANNER=1.
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
    if x.size == 0:
        return x.astype(np.float32)
    x = x - float(x.min())
    m = float(x.max())
    return np.zeros_like(x, dtype=np.float32) if m <= 1e-8 else (x / m).astype(np.float32)


def _resample_1d(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == n:
        return x.astype(np.float32)
    if len(x) == 0:
        return np.zeros((n,), dtype=np.float32)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x).astype(np.float32)


def onset_curve(audio_feature: np.ndarray) -> np.ndarray:
    audio_feature = np.asarray(audio_feature, dtype=np.float32)
    if audio_feature.ndim != 2 or len(audio_feature) == 0:
        return np.zeros((0,), dtype=np.float32)
    if audio_feature.shape[1] > 768:
        x = np.maximum(audio_feature[:, 768], 0.0)
    elif audio_feature.shape[1] >= 35:
        # baseline_features layout is usually envelope + mfcc + chroma + peak + beat.
        x = np.maximum(audio_feature[:, 0], 0.0) + 0.5 * np.maximum(audio_feature[:, -2], 0.0) + 0.5 * np.maximum(audio_feature[:, -1], 0.0)
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


def tension_curve(audio_feature: np.ndarray, num_frames: int) -> np.ndarray:
    """Music tension proxy: energy + onset density + rising energy.

    This is intentionally deterministic and does not call an online LLM.  Manual
    LLM plans can override it by writing tension_target/phase into JSON.
    """
    e = _resample_1d(frame_energy(audio_feature), num_frames)
    o = _resample_1d(onset_curve(audio_feature), num_frames)
    de = np.zeros_like(e, dtype=np.float32)
    if len(e) > 1:
        de[1:] = np.maximum(e[1:] - e[:-1], 0.0)
    t = 0.50 * e + 0.35 * o + 0.15 * _safe_norm01(de)
    return _safe_norm01(t)


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


def _phase_from_segment(seg_id: int, num_segments: int, tension: float, rhythm: float) -> str:
    # Ends are usually presentation / settling moments in the default heuristic.
    # A manual JSON plan can override this field for special choreographies.
    if seg_id == num_segments - 1 and tension < 0.70:
        return "pose"
    if tension >= 0.62 or rhythm >= 0.62:
        return "attack"
    if tension <= 0.28 and (seg_id == 0 or seg_id == num_segments - 1):
        return "pose"
    return "flow"


def _motion_prompt(energy: float, rhythm: float, tension: float, phase: str, seg_id: int, num_segments: int, style_hint: str) -> str:
    e = _label_energy(energy)
    if phase == "attack":
        base = "发力，动作释放，上肢大幅舒展，带有转身或方向变化"
    elif phase == "pose":
        if seg_id == 0:
            base = "起势，姿态端住，上肢含蓄展开，重心稳定"
        elif seg_id == num_segments - 1:
            base = "亮相收束，姿态回落，动作平稳结束，重心控制"
        else:
            base = "短暂停顿或亮相，身体线条清晰，重心稳定"
    else:
        base = "流动过渡，中等幅度转身，上肢舒展，下肢步伐平稳"
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

    onset = _resample_1d(onset_curve(audio_feature), num_frames)
    energy = _resample_1d(frame_energy(audio_feature), num_frames)
    tension = tension_curve(audio_feature, num_frames)

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
        t_mean = float(np.clip(tension[sl].mean() if e >= s else 0.0, 0.0, 1.0))
        phase = _phase_from_segment(i, nseg, t_mean, o_mean)

        music_caption = f"{_label_rhythm(o_mean)}，{_label_energy(e_mean)}，{'音乐张力高' if t_mean > 0.62 else '音乐张力中低'}，{'重音明显' if o_mean > 0.5 else '重音稀疏'}"
        motion_prompt = _motion_prompt(e_mean, o_mean, t_mean, phase, i, nseg, style_hint)

        # Planner suggestions.  They are not hard until EDGE_TENSION_AWARE_PLANNER=1.
        if phase == "attack":
            min_expr = 0.48 + 0.20 * t_mean
            min_energy = 0.40 + 0.20 * e_mean
            stability_scale = 0.70
            expr_scale = 1.35
        elif phase == "flow":
            min_expr = 0.30 + 0.18 * t_mean
            min_energy = 0.25 + 0.18 * e_mean
            stability_scale = 1.00
            expr_scale = 1.10
        else:
            min_expr = -1.0
            min_energy = -1.0
            stability_scale = 1.35
            expr_scale = 0.65

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
            "tension_target": t_mean,
            "phase": phase,
            "min_expressiveness": float(np.clip(min_expr, -1.0, 1.0)),
            "min_energy": float(np.clip(min_energy, -1.0, 1.0)),
            "stability_weight_scale": float(stability_scale),
            "expressiveness_weight_scale": float(expr_scale),
            "style_tags": ["敦煌舞", "飞天", "上肢舒展", "重心稳定"],
        })

    return {
        "planner": "choreo_unit_rag_tension_v1",
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
                plan = json.load(f)
            # Backfill fields for old plans, without overriding manual values.
            if audio_feature is not None and "segments" in plan:
                fallback = build_heuristic_choreo_plan(audio_feature, num_frames, style_hint, max_segments)
                for seg, fb in zip(plan.get("segments", []), fallback.get("segments", [])):
                    for key in ["tension_target", "phase", "min_expressiveness", "min_energy", "stability_weight_scale", "expressiveness_weight_scale"]:
                        seg.setdefault(key, fb.get(key))
            return plan
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
