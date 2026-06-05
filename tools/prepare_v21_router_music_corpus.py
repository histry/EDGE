#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


AUDIO_SUFFIXES = {
    ".wav", ".au", ".mp3", ".flac", ".ogg",
    ".m4a", ".aac", ".aif", ".aiff",
}


def valid_source(path: Path) -> bool:
    lower_parts = [x.lower() for x in path.parts]

    if "__macosx" in lower_parts:
        return False

    if path.name.startswith("._"):
        return False

    if path.name in {".DS_Store", "Thumbs.db"}:
        return False

    if path.suffix.lower() not in AUDIO_SUFFIXES:
        return False

    # AppleDouble 往往只有几百字节
    if path.stat().st_size < 50_000:
        return False

    return True


def safe_name(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    return text.strip("_")[:80] or "music"


def probe_audio(path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return result.returncode == 0 and bool(result.stdout.strip())


def convert_one(
    index: int,
    source: Path,
    source_root: Path,
    output_root: Path,
):
    relative = str(source.relative_to(source_root))
    short_hash = hashlib.sha1(
        relative.encode("utf-8")
    ).hexdigest()[:8]

    target = output_root / (
        f"{index:04d}_"
        f"{safe_name(source.stem)}_"
        f"{short_hash}.wav"
    )

    if not probe_audio(source):
        return {
            "ok": False,
            "source": str(source),
            "reason": "ffprobe_failed",
        }

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(source),
        "-vn",
        "-ac", "1",
        "-ar", "22050",
        "-c:a", "pcm_s16le",
        str(target),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        target.unlink(missing_ok=True)
        return {
            "ok": False,
            "source": str(source),
            "reason": result.stderr[-500:],
        }

    try:
        with wave.open(str(target), "rb") as wf:
            channels = wf.getnchannels()
            rate = wf.getframerate()
            width = wf.getsampwidth()
            frames = wf.getnframes()
            duration = frames / max(rate, 1)

        if channels != 1 or rate != 22050 or width != 2:
            raise ValueError(
                f"invalid PCM contract: "
                f"channels={channels}, rate={rate}, width={width}"
            )

        if duration < 15.0:
            raise ValueError(
                f"audio too short: {duration:.2f}s"
            )

    except Exception as exc:
        target.unlink(missing_ok=True)
        return {
            "ok": False,
            "source": str(source),
            "reason": str(exc),
        }

    return {
        "ok": True,
        "source": str(source),
        "target": str(target),
        "duration": float(duration),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def hardlink_or_copy(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    canonical_dir = output_dir / "canonical"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    canonical_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and valid_source(p)
    )

    print("candidate sources:", len(sources))

    results = []

    with ThreadPoolExecutor(
        max_workers=max(1, args.workers)
    ) as executor:
        futures = {
            executor.submit(
                convert_one,
                index,
                source,
                input_dir,
                canonical_dir,
            ): source
            for index, source in enumerate(sources, 1)
        }

        for done, future in enumerate(
            as_completed(futures),
            1,
        ):
            result = future.result()
            results.append(result)

            status = "OK" if result["ok"] else "FAIL"
            print(
                f"[{done}/{len(futures)}] "
                f"{status}: {result['source']}",
                flush=True,
            )

    valid = [x for x in results if x["ok"]]
    failed = [x for x in results if not x["ok"]]

    # 精确去重
    unique = []
    hash_owner = {}
    duplicates = []

    for record in sorted(
        valid,
        key=lambda x: x["target"],
    ):
        path = Path(record["target"])
        digest = sha256(path)

        if digest in hash_owner:
            duplicates.append({
                "duplicate": str(path),
                "original": hash_owner[digest],
                "sha256": digest,
            })
            path.unlink()
            continue

        hash_owner[digest] = str(path)
        record["sha256"] = digest
        unique.append(record)

    rng = random.Random(args.seed)
    rng.shuffle(unique)

    total = len(unique)
    train_count = int(total * args.train_ratio)
    val_count = int(total * args.val_ratio)

    splits = {
        "train": unique[:train_count],
        "val": unique[
            train_count:train_count + val_count
        ],
        "test": unique[
            train_count + val_count:
        ],
    }

    for split, records in splits.items():
        split_dir = output_dir / "splits" / split

        for record in records:
            source = Path(record["target"])
            target = split_dir / source.name
            hardlink_or_copy(source, target)

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "candidate_sources": len(sources),
        "converted_valid": len(valid),
        "failed_count": len(failed),
        "duplicate_count": len(duplicates),
        "unique_count": len(unique),
        "split_counts": {
            key: len(value)
            for key, value in splits.items()
        },
        "failed": failed,
        "duplicates": duplicates,
        "splits": splits,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 70)
    print("converted valid:", len(valid))
    print("failed:", len(failed))
    print("duplicates:", len(duplicates))
    print("unique:", len(unique))
    print(
        "split counts:",
        manifest["split_counts"],
    )
    print("manifest:", manifest_path)

    if len(unique) < 900:
        raise RuntimeError(
            f"Only {len(unique)} unique valid audio files remain; "
            "expected at least 900."
        )


if __name__ == "__main__":
    main()
