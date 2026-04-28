import argparse
import csv
import json
import math
import os
import pickle
from pathlib import Path

import librosa
import numpy as np
import scipy.signal
from scipy.signal import find_peaks

if not hasattr(scipy.signal, "hann") and hasattr(scipy.signal, "windows"):
    scipy.signal.hann = scipy.signal.windows.hann


def smooth_1d(x, window):
    window = max(1, int(window))
    if window <= 1 or len(x) < 3:
        return x.astype(np.float32)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), kernel, mode="valid").astype(np.float32)


def safe_mean(values, default=0.0):
    values = np.asarray(values, dtype=np.float32)
    return float(values.mean()) if values.size else float(default)


def estimate_bpm_from_frames(beat_frames, fps, default_bpm=0.0):
    beat_frames = np.asarray(beat_frames, dtype=np.float32)
    if len(beat_frames) < 2:
        return float(default_bpm)
    intervals = np.diff(beat_frames)
    intervals = intervals[intervals > 1e-6]
    if len(intervals) == 0:
        return float(default_bpm)
    return float(60.0 * fps / np.median(intervals))


def normalize_curve(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo = float(np.percentile(x, 5))
    hi = float(np.percentile(x, 95))
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def extract_proxy_music_features(audio_path, fps):
    y, sr = librosa.load(audio_path, sr=None)
    if len(y) == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

    hop_length = max(1, int(round(sr / fps)))
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length).astype(np.float32)
    duration = float(librosa.get_duration(y=y, sr=sr))

    try:
        tempo, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=sr,
            onset_envelope=onset,
            hop_length=hop_length,
            units="frames",
        )
        bpm = float(np.asarray(tempo).reshape(-1)[0])
    except Exception:
        beat_frames = np.asarray([], dtype=np.int64)
        bpm = 0.0

    peak_distance = max(1, int(round(fps * 0.20)))
    peak_height = float(np.mean(onset) + 0.5 * np.std(onset))
    onset_peaks, _ = find_peaks(onset, distance=peak_distance, height=peak_height)

    if bpm <= 0:
        bpm = estimate_bpm_from_frames(beat_frames, fps)
    if bpm <= 0:
        bpm = estimate_bpm_from_frames(onset_peaks, fps)

    beat_density = float(len(beat_frames) / max(duration, 1e-6))
    onset_density = float(len(onset_peaks) / max(duration, 1e-6))
    onset_energy = float(np.mean(normalize_curve(onset)))

    return {
        "audio_path": audio_path,
        "duration_sec": duration,
        "bpm": bpm,
        "beat_count": int(len(beat_frames)),
        "beat_density": beat_density,
        "onset_count": int(len(onset_peaks)),
        "onset_density": onset_density,
        "onset_energy": onset_energy,
    }


def motion_accent_curve(pos, q, fps):
    root_v = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
    root_v = np.pad(root_v, (1, 0), mode="edge")
    root_a = np.abs(np.diff(root_v, prepend=root_v[:1])) * fps

    q = q.reshape(q.shape[0], -1, 3)
    rot_v = np.linalg.norm(q[1:] - q[:-1], axis=-1).mean(axis=-1) * fps
    rot_v = np.pad(rot_v, (1, 0), mode="edge")
    rot_a = np.abs(np.diff(rot_v, prepend=rot_v[:1])) * fps

    root_v = root_v[: len(pos)]
    root_a = root_a[: len(pos)]
    curve = 0.45 * normalize_curve(root_v) + 0.35 * normalize_curve(root_a) + 0.20 * normalize_curve(rot_a)
    return smooth_1d(curve.astype(np.float32), window=max(3, int(round(fps * 0.15))))


def extract_motion_window_features(pos, q, source_path, start, seq_len, fps):
    end = start + seq_len
    pos_w = pos[start:end]
    q_w = q[start:end]
    duration = float(seq_len / fps)
    accent = motion_accent_curve(pos_w, q_w, fps)

    min_distance = max(2, int(round(fps * 0.25)))
    prominence = max(float(np.std(accent) * 0.35), 1e-5)
    peaks, props = find_peaks(accent, distance=min_distance, prominence=prominence)
    bpm = estimate_bpm_from_frames(peaks, fps)

    if bpm <= 0:
        root_speed = np.linalg.norm(np.diff(pos_w, axis=0), axis=1) * fps
        valleys, _ = find_peaks(-smooth_1d(root_speed, 5), distance=min_distance)
        bpm = estimate_bpm_from_frames(valleys, fps)

    return {
        "motion_path": source_path,
        "window_id": f"{Path(source_path).stem}_{start:06d}_{end:06d}",
        "start_frame": int(start),
        "end_frame": int(end - 1),
        "duration_sec": duration,
        "bpm": float(bpm),
        "accent_count": int(len(peaks)),
        "accent_density": float(len(peaks) / max(duration, 1e-6)),
        "accent_energy": safe_mean(accent),
        "root_path_len_m": float(np.linalg.norm(np.diff(pos_w[:, [0, 2]], axis=0), axis=1).sum()),
        "root_speed_mean_mps": float(np.linalg.norm(np.diff(pos_w[:, [0, 2]], axis=0), axis=1).mean() * fps),
    }


def iter_motion_windows(motion_dir, seq_len, stride, fps, max_windows_per_motion):
    motion_paths = sorted(Path(motion_dir).glob("*.pkl"))
    for motion_path in motion_paths:
        data = pickle.load(open(motion_path, "rb"))
        pos = np.asarray(data["pos"], dtype=np.float32)
        q = np.asarray(data["q"], dtype=np.float32)
        if len(pos) < seq_len:
            continue

        starts = list(range(0, len(pos) - seq_len + 1, stride))
        if max_windows_per_motion > 0 and len(starts) > max_windows_per_motion:
            idx = np.linspace(0, len(starts) - 1, max_windows_per_motion).round().astype(int)
            starts = [starts[i] for i in idx]

        for start in starts:
            yield extract_motion_window_features(
                pos=pos,
                q=q,
                source_path=str(motion_path),
                start=start,
                seq_len=seq_len,
                fps=fps,
            )


def gaussian_score(delta, sigma):
    sigma = max(float(sigma), 1e-6)
    return float(math.exp(-0.5 * (float(delta) / sigma) ** 2))


def relative_bpm_delta(motion_bpm, audio_bpm):
    if motion_bpm <= 1e-6 or audio_bpm <= 1e-6:
        return 1.0
    candidates = [audio_bpm, audio_bpm * 0.5, audio_bpm * 2.0]
    return min(abs(motion_bpm - c) / max(motion_bpm, 1e-6) for c in candidates)


def pair_score(motion, audio, args):
    bpm_delta = relative_bpm_delta(motion["bpm"], audio["bpm"])
    density_delta = abs(motion["accent_density"] - audio["beat_density"]) / max(motion["accent_density"], 1e-6)
    onset_delta = abs(motion["accent_density"] - audio["onset_density"]) / max(motion["accent_density"], 1e-6)
    duration_delta = abs(motion["duration_sec"] - audio["duration_sec"]) / max(motion["duration_sec"], 1e-6)
    energy_delta = abs(motion["accent_energy"] - audio["onset_energy"])

    bpm_score = gaussian_score(bpm_delta, args.bpm_sigma)
    density_score = gaussian_score(density_delta, args.density_sigma)
    onset_score = gaussian_score(onset_delta, args.onset_sigma)
    duration_score = gaussian_score(duration_delta, args.duration_sigma)
    energy_score = gaussian_score(energy_delta, args.energy_sigma)

    total = (
        args.w_bpm * bpm_score
        + args.w_density * density_score
        + args.w_onset * onset_score
        + args.w_duration * duration_score
        + args.w_energy * energy_score
    ) / max(args.w_bpm + args.w_density + args.w_onset + args.w_duration + args.w_energy, 1e-6)

    return {
        "score": float(total),
        "bpm_score": bpm_score,
        "density_score": density_score,
        "onset_score": onset_score,
        "duration_score": duration_score,
        "energy_score": energy_score,
        "bpm_relative_delta": float(bpm_delta),
    }


def build_pairs(motion_features, audio_features, args):
    pairs = []
    for motion in motion_features:
        scored = []
        for audio in audio_features:
            scores = pair_score(motion, audio, args)
            if scores["score"] >= args.min_score:
                row = {
                    **scores,
                    "window_id": motion["window_id"],
                    "motion_path": motion["motion_path"],
                    "start_frame": motion["start_frame"],
                    "end_frame": motion["end_frame"],
                    "motion_bpm": motion["bpm"],
                    "motion_accent_density": motion["accent_density"],
                    "motion_accent_count": motion["accent_count"],
                    "audio_path": audio["audio_path"],
                    "audio_bpm": audio["bpm"],
                    "audio_beat_density": audio["beat_density"],
                    "audio_onset_density": audio["onset_density"],
                    "weak_pair_claim": "proxy rhythmic candidate, not a real music-motion label",
                }
                scored.append(row)
        scored = sorted(scored, key=lambda item: item["score"], reverse=True)[: args.top_k]
        pairs.extend(scored)
    return pairs


def write_csv(path, rows):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build BPM/beat based weak proxy-music pair candidates.")
    parser.add_argument("--motion_dir", default="data/dunhuang_bvh/processed")
    parser.add_argument("--proxy_dir", default="proxy_music")
    parser.add_argument("--out_dir", default="data/proxy_weak_pairs")
    parser.add_argument("--seq_len", type=int, default=150)
    parser.add_argument("--stride", type=int, default=75)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--max_windows_per_motion", type=int, default=40)
    parser.add_argument("--bpm_sigma", type=float, default=0.12)
    parser.add_argument("--density_sigma", type=float, default=0.35)
    parser.add_argument("--onset_sigma", type=float, default=0.60)
    parser.add_argument("--duration_sigma", type=float, default=0.80)
    parser.add_argument("--energy_sigma", type=float, default=0.30)
    parser.add_argument("--w_bpm", type=float, default=0.45)
    parser.add_argument("--w_density", type=float, default=0.25)
    parser.add_argument("--w_onset", type=float, default=0.15)
    parser.add_argument("--w_duration", type=float, default=0.0)
    parser.add_argument("--w_energy", type=float, default=0.10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = sorted(Path(args.proxy_dir).glob("*.wav"))
    if not audio_paths:
        raise FileNotFoundError(f"No .wav files found in {args.proxy_dir}")

    print(f"Extracting proxy music features from {len(audio_paths)} wav files...")
    audio_features = [extract_proxy_music_features(str(path), args.fps) for path in audio_paths]

    print("Extracting motion rhythm features...")
    motion_features = list(
        iter_motion_windows(
            motion_dir=args.motion_dir,
            seq_len=args.seq_len,
            stride=args.stride,
            fps=args.fps,
            max_windows_per_motion=args.max_windows_per_motion,
        )
    )
    if not motion_features:
        raise RuntimeError(f"No valid motion windows found in {args.motion_dir}")

    print(f"Building weak pair candidates: {len(motion_features)} motion windows x {len(audio_features)} proxy tracks")
    pairs = build_pairs(motion_features, audio_features, args)

    manifest = {
        "note": "These are BPM/beat based proxy candidates, not real music-motion labels.",
        "config": vars(args),
        "audio_features": audio_features,
        "motion_features": motion_features,
        "pairs": pairs,
    }
    with open(out_dir / "weak_pairs.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    write_csv(out_dir / "weak_pairs.csv", pairs)
    write_csv(out_dir / "proxy_music_features.csv", audio_features)
    write_csv(out_dir / "motion_rhythm_features.csv", motion_features)

    print(f"Saved weak pair manifest: {out_dir / 'weak_pairs.json'}")
    print(f"Saved weak pair table:    {out_dir / 'weak_pairs.csv'}")
    print("Top examples:")
    for row in sorted(pairs, key=lambda item: item["score"], reverse=True)[:10]:
        print(
            f"  score={row['score']:.3f} motion={row['window_id']} "
            f"proxy={Path(row['audio_path']).name} "
            f"motion_bpm={row['motion_bpm']:.1f} audio_bpm={row['audio_bpm']:.1f}"
        )


if __name__ == "__main__":
    main()
