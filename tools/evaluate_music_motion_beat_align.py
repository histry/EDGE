import argparse
import json
from pathlib import Path

import numpy as np

try:
    import librosa
except Exception as e:
    raise RuntimeError("Need librosa for audio onset analysis") from e

try:
    from scipy.signal import find_peaks
except Exception as e:
    raise RuntimeError("Need scipy for peak detection") from e

ROT = slice(7, 151)

def load_motion(path):
    x = np.load(path, allow_pickle=True)
    if x.ndim == 0 and isinstance(x.item(), dict):
        obj = x.item()
        x = obj.get("motion", obj.get("motion_151", obj.get("pose", x)))
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[1] != 151:
        raise ValueError(f"expected [T,151] or [1,T,151], got {x.shape}")
    return x

def norm01(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x)
    x = x - x.min()
    mx = x.max()
    if mx < 1e-8:
        return np.zeros_like(x)
    return x / (mx + 1e-8)

def resample_curve(curve, target_len):
    if len(curve) == target_len:
        return curve.astype(np.float32)
    old = np.linspace(0, 1, len(curve), dtype=np.float32)
    new = np.linspace(0, 1, target_len, dtype=np.float32)
    return np.interp(new, old, curve).astype(np.float32)

def motion_energy(m):
    rot = m[:, ROT]
    e = np.zeros((len(m),), dtype=np.float32)
    if len(m) > 1:
        e[1:] = np.linalg.norm(rot[1:] - rot[:-1], axis=1)
        e[0] = e[1]
    return norm01(e)

def onset_curve(audio_path, target_len, fps):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    hop = max(1, int(sr / fps))
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    env = norm01(env)
    return resample_curve(env, target_len)

def peak_frames(curve, min_gap, threshold):
    if len(curve) == 0:
        return np.array([], dtype=np.int64)
    height = float(threshold)
    peaks, _ = find_peaks(curve, height=height, distance=max(1, int(min_gap)))
    return peaks.astype(np.int64)

def align_metrics(onset_peaks, motion_peaks, fps):
    out = {}
    for tol in [3, 5, 8, 12]:
        hits = 0
        offsets = []
        for o in onset_peaks:
            if len(motion_peaks) == 0:
                continue
            d = motion_peaks - o
            j = int(np.argmin(np.abs(d)))
            if abs(int(d[j])) <= tol:
                hits += 1
                offsets.append(int(d[j]))
        out[f"hit_rate_{tol}f"] = float(hits / max(len(onset_peaks), 1))
        if offsets:
            out[f"mean_abs_offset_{tol}f"] = float(np.mean(np.abs(offsets)))
            out[f"mean_abs_offset_{tol}s"] = float(np.mean(np.abs(offsets)) / fps)
        else:
            out[f"mean_abs_offset_{tol}f"] = None
            out[f"mean_abs_offset_{tol}s"] = None
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--onset_threshold", type=float, default=0.35)
    ap.add_argument("--motion_threshold", type=float, default=0.35)
    ap.add_argument("--min_gap", type=int, default=6)
    args = ap.parse_args()

    m = load_motion(args.motion)
    T = len(m)

    onset = onset_curve(args.audio, T, args.fps)
    energy = motion_energy(m)

    onset_peaks = peak_frames(onset, args.min_gap, args.onset_threshold)
    motion_peaks = peak_frames(energy, args.min_gap, args.motion_threshold)

    corr = float(np.corrcoef(onset, energy)[0, 1]) if np.std(onset) > 1e-8 and np.std(energy) > 1e-8 else 0.0

    report = {
        "motion": args.motion,
        "audio": args.audio,
        "frames": int(T),
        "fps": float(args.fps),
        "duration_sec": float(T / args.fps),
        "onset_peak_count": int(len(onset_peaks)),
        "motion_peak_count": int(len(motion_peaks)),
        "onset_peaks": onset_peaks.tolist(),
        "motion_peaks": motion_peaks.tolist(),
        "onset_motion_corr": corr,
        **align_metrics(onset_peaks, motion_peaks, args.fps),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
