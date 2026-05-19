#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import random
import sys
import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hf_event_contrastive import (
    AUDIO_EVENT_DIM,
    MOTION_EVENT_DIM,
    HFEventContrastiveEncoder,
    SupConLoss,
    audio_event_features_torch,
    intensity_labels_from_features,
    motion_event_features_torch,
)


def read_wav_mono(path: str, sr_target: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Minimal PCM wav reader. Avoids hard dependency on librosa/soundfile.
    """
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"Unsupported wav sample width={sampwidth}: {path}")

    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)

    if sr != sr_target and len(x) > 1:
        # Simple linear resampling.
        dur = len(x) / float(sr)
        new_len = max(1, int(round(dur * sr_target)))
        old_t = np.linspace(0.0, dur, num=len(x), endpoint=False)
        new_t = np.linspace(0.0, dur, num=new_len, endpoint=False)
        x = np.interp(new_t, old_t, x).astype(np.float32)
        sr = sr_target

    return x.astype(np.float32), sr


def audio_segment_to_frame_features(wav: np.ndarray, sr: int, seq_len: int = 45, seconds: float = 1.5) -> np.ndarray:
    """
    Convert raw waveform segment to [seq_len, 8] event descriptors.
    """
    need = int(sr * seconds)
    if len(wav) < need:
        pad = np.zeros((need - len(wav),), dtype=np.float32)
        wav = np.concatenate([wav, pad], axis=0)
    if len(wav) > need:
        start = random.randint(0, len(wav) - need)
        wav = wav[start:start + need]

    frames = np.array_split(wav, seq_len)
    feats = []
    for fr in frames:
        if len(fr) < 8:
            fr = np.pad(fr, (0, 8 - len(fr)))
        rms = float(np.sqrt(np.mean(fr ** 2) + 1e-8))
        mean_abs = float(np.mean(np.abs(fr)))
        diff = np.diff(fr)
        onset = float(np.mean(np.maximum(diff, 0.0))) if len(diff) else 0.0
        zcr = float(np.mean(np.abs(np.diff(np.sign(fr))) > 0)) if len(fr) > 1 else 0.0

        spec = np.abs(np.fft.rfft(fr * np.hanning(len(fr))))
        if spec.size < 6:
            spec = np.pad(spec, (0, 6 - spec.size))
        n = spec.size
        low = float(np.mean(spec[: max(1, n // 3)]))
        mid = float(np.mean(spec[max(1, n // 3): max(2, 2 * n // 3)]))
        high = float(np.mean(spec[max(2, 2 * n // 3):]))
        centroid = float((np.arange(n) * spec).sum() / (spec.sum() + 1e-8) / max(1, n))

        feats.append([rms, mean_abs, onset, zcr, low, mid, high, centroid])
    return np.asarray(feats, dtype=np.float32)


def load_motion_from_pkl(path: str, seq_len: int = 45) -> np.ndarray:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        for key in ["motion_151", "motion", "unit_motion", "unit_motions_physical"]:
            if key in obj:
                arr = np.asarray(obj[key], dtype=np.float32)
                break
        else:
            raise KeyError(f"No motion key in {path}")
    else:
        arr = np.asarray(obj, dtype=np.float32)

    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape[-1] != 151:
        raise ValueError(f"Expected 151D motion, got {arr.shape} in {path}")
    if arr.shape[0] < seq_len:
        pad = np.repeat(arr[-1:], seq_len - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    return arr[:seq_len].astype(np.float32)


class HFEventDataset(Dataset):
    def __init__(self, motion_files: List[str], wav_files: List[str], seq_len: int = 45, sr: int = 16000):
        self.motion_files = motion_files
        self.wav_files = wav_files
        self.seq_len = int(seq_len)
        self.sr = int(sr)

        if len(self.motion_files) == 0:
            raise RuntimeError("No motion pkl files found.")
        if len(self.wav_files) == 0:
            raise RuntimeError("No wav files found. Pass --music_glob test_music_bank/*.wav")

    def __len__(self):
        return max(len(self.motion_files), len(self.wav_files)) * 4

    def __getitem__(self, idx):
        mpath = self.motion_files[idx % len(self.motion_files)]
        wpath = self.wav_files[random.randint(0, len(self.wav_files) - 1)]

        motion = load_motion_from_pkl(mpath, seq_len=self.seq_len)
        wav, sr = read_wav_mono(wpath, sr_target=self.sr)
        audio = audio_segment_to_frame_features(wav, sr, seq_len=self.seq_len, seconds=self.seq_len / 30.0)

        return {
            "motion": torch.from_numpy(motion),
            "audio": torch.from_numpy(audio),
            "motion_file": mpath,
            "wav_file": wpath,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/dunhuang_bvh/footwork_v3h_u45")
    ap.add_argument("--music_glob", default="test_music_bank/*.wav")
    ap.add_argument("--out_dir", default="checkpoints/hf_event_contrastive/day01")
    ap.add_argument("--seq_len", type=int, default=45)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    motion_files = sorted(str(p) for p in data_dir.glob("*.pkl"))
    wav_files = sorted(glob.glob(args.music_glob))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = HFEventDataset(motion_files, wav_files, seq_len=args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HFEventContrastiveEncoder(
        audio_dim=AUDIO_EVENT_DIM,
        motion_dim=MOTION_EVENT_DIM,
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        dropout=0.05,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    supcon = SupConLoss(temperature=args.temperature)

    print("============================================================")
    print("HF Event Contrastive Encoder Training")
    print(f"motion_files={len(motion_files)}")
    print(f"wav_files={len(wav_files)}")
    print(f"out_dir={out_dir}")
    print("============================================================")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        cos_losses = []
        sup_losses = []

        for batch in dl:
            motion = batch["motion"].to(device).float()
            audio = batch["audio"].to(device).float()

            m_feat = motion_event_features_torch(motion)
            a_feat = audio_event_features_torch(audio, target_len=args.seq_len)

            m_emb = model.encode_motion_features(m_feat)
            a_emb = model.encode_audio_features(a_feat)

            m_label = intensity_labels_from_features(m_feat)
            a_label = intensity_labels_from_features(a_feat)

            emb = torch.cat([m_emb, a_emb], dim=0)
            labels = torch.cat([m_label, a_label], dim=0)

            loss_sup = supcon(emb, labels)
            loss_cos = 1.0 - (m_emb * a_emb).sum(dim=-1).mean()
            loss = loss_cos + 0.25 * loss_sup

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))
            cos_losses.append(float(loss_cos.detach().cpu()))
            sup_losses.append(float(loss_sup.detach().cpu()))

        print(
            f"Epoch {epoch:04d} | "
            f"loss={np.mean(losses):.6f} "
            f"cos={np.mean(cos_losses):.6f} "
            f"supcon={np.mean(sup_losses):.6f}",
            flush=True,
        )

        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt = {
                "model": model.state_dict(),
                "config": {
                    "audio_dim": AUDIO_EVENT_DIM,
                    "motion_dim": MOTION_EVENT_DIM,
                    "hidden_dim": args.hidden_dim,
                    "emb_dim": args.emb_dim,
                    "dropout": 0.05,
                    "seq_len": args.seq_len,
                    "temperature": args.temperature,
                },
                "epoch": epoch,
            }
            torch.save(ckpt, out_dir / f"hf_event_encoder_e{epoch:04d}.pt")
            torch.save(ckpt, out_dir / "hf_event_encoder.pt")

    meta = {
        "data_dir": str(data_dir),
        "music_glob": args.music_glob,
        "motion_files": len(motion_files),
        "wav_files": len(wav_files),
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "note": "Weak/pseudo-supervised HF audio-motion event contrastive encoder.",
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ saved: {out_dir / 'hf_event_encoder.pt'}")


if __name__ == "__main__":
    main()
