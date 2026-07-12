#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v46_51_audio_schedule_contract import (  # noqa: E402
    audit_contract,
    audio_info,
    save_json,
    stamp_descriptor,
)
from tools.v46_51_split_event_db import assign_sources  # noqa: E402


def write_wav(path: Path, seconds: float, sample_rate: int = 16000) -> None:
    frames = int(round(seconds * sample_rate))
    t = np.arange(frames, dtype=np.float32) / sample_rate
    signal = 0.1 * np.sin(2.0 * np.pi * 220.0 * t)
    pcm = np.clip(signal * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def make_transaction(tmp: Path, seconds: float = 4.0):
    audio = tmp / "current.wav"
    write_wav(audio, seconds)
    info = audio_info(audio, fps=30.0)
    total = int(info["target_frames"])
    split = total // 2
    slots = [
        {
            "slot_id": 0,
            "start_frame": 0,
            "end_frame": split,
            "target_frames": split,
            "start": 0.0,
            "end": split / 30.0,
            "duration": split / 30.0,
        },
        {
            "slot_id": 1,
            "start_frame": split,
            "end_frame": total,
            "target_frames": total - split,
            "start": split / 30.0,
            "end": total / 30.0,
            "duration": (total - split) / 30.0,
        },
    ]

    raw_motion = tmp / "current_v26.npy"
    np.save(raw_motion, np.zeros((1, total, 151), dtype=np.float32))
    raw_report = tmp / "current_v26.schedule_report.json"
    save_json(
        {
            "version": "test_v26",
            "audio": str(audio.resolve()),
            "out_npy": str(raw_motion.resolve()),
            "schedule": slots,
        },
        raw_report,
    )

    base = {
        "descriptor_type": "music_semantic_slot_descriptor",
        "descriptor_schema_version": "v46_38_mssd_aesd_routing_descriptor",
        "usage": "generate_schedule",
        "is_final_schedule": True,
        "slot_source": "v21_router_v26_planner",
        "audio": str(audio.resolve()),
        "fps": 30.0,
        "num_slots": len(slots),
        "total_target_frames": total,
        "slots": slots,
        "segments": slots,
        "provenance": {},
    }
    run_id = "unit_test_run"
    stamped = stamp_descriptor(
        base,
        audio=audio,
        fps=30.0,
        run_id=run_id,
        run_dir=tmp,
        raw_schedule_json=raw_report,
        scheduler_command=["python", "schedule_v26_whole_song.py"],
        assets={},
        hash_assets=False,
    )
    schedule = tmp / "current.final.mssd.json"
    save_json(stamped, schedule)
    return audio, schedule, run_id, total


def test_fresh_transaction_passes():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        audio, schedule, run_id, total = make_transaction(tmp)
        report = audit_contract(
            audio=audio,
            schedule=schedule,
            fps=30.0,
            required_run_id=run_id,
            require_fresh=True,
        )
        assert report["ok"], report["reasons"]
        assert report["total_target_frames"] == total
        assert report["gap_count"] == 0
        assert report["overlap_count"] == 0


def test_changed_audio_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        audio, schedule, run_id, _ = make_transaction(tmp)
        write_wav(audio, 5.0)
        report = audit_contract(
            audio=audio,
            schedule=schedule,
            fps=30.0,
            required_run_id=run_id,
            require_fresh=True,
        )
        assert not report["ok"]
        assert "audio_sha256_mismatch" in report["reasons"]


def test_wrong_run_id_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        audio, schedule, _, _ = make_transaction(tmp)
        report = audit_contract(
            audio=audio,
            schedule=schedule,
            fps=30.0,
            required_run_id="another_run",
            require_fresh=True,
        )
        assert not report["ok"]
        assert any(
            x.startswith("schedule_run_id_mismatch")
            for x in report["reasons"]
        )


def test_gap_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        audio, schedule, run_id, _ = make_transaction(tmp)
        obj = json.loads(schedule.read_text(encoding="utf-8"))
        obj["slots"][1]["start_frame"] += 3
        obj["segments"] = obj["slots"]
        save_json(obj, schedule)
        report = audit_contract(
            audio=audio,
            schedule=schedule,
            fps=30.0,
            required_run_id=run_id,
            require_fresh=True,
        )
        assert not report["ok"]
        assert report["gap_count"] == 1


def test_source_assignment_is_deterministic_and_disjoint():
    source_to_label = {
        f"pose_{i}": "thirty_six_postures" for i in range(6)
    }
    source_to_label.update(
        {f"turn_{i}": "sogdian_whirl" for i in range(6)}
    )
    a = assign_sources(
        source_to_label,
        seed=20260712,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    b = assign_sources(
        source_to_label,
        seed=20260712,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    assert a == b
    assert set(a) == set(source_to_label)
    assert set(a.values()) == {"train", "val", "test"}
    splits = {
        split: {s for s, x in a.items() if x == split}
        for split in ("train", "val", "test")
    }
    assert not (splits["train"] & splits["val"])
    assert not (splits["train"] & splits["test"])
    assert not (splits["val"] & splits["test"])


if __name__ == "__main__":
    tests = [
        test_fresh_transaction_passes,
        test_changed_audio_is_rejected,
        test_wrong_run_id_is_rejected,
        test_gap_is_rejected,
        test_source_assignment_is_deterministic_and_disjoint,
    ]
    for fn in tests:
        fn()
        print("[PASS]", fn.__name__)
    print(f"[PASS] {len(tests)} V46.51 tests")
