#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V46.34 router-slot loader patch for tools/v46_motionrag_diff.py.

This patch is deliberately small and safe:
  - keeps a timestamped backup;
  - renames the current audio_slots() to audio_slots_v46_default();
  - installs a robust audio_slots() that understands V46.34 slot-plan JSON;
  - optionally enforces strict pretrained-router slot usage with
        V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=1.

It does not replace the rest of the V46.33 reference-transition code.  If you
already applied V46.33 transition-budget / masked-diffusion patches, this patch
keeps them intact and only changes the music slot entry point used by generate().
"""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import time

TARGET = Path("tools/v46_motionrag_diff.py")
MARK = "# ===== V46.34 PRETRAINED ROUTER SLOT PATCH START ====="
END_MARK = "# ===== V46.34 PRETRAINED ROUTER SLOT PATCH END ====="


def find_func_span(text: str, name: str):
    m = re.search(rf"^def\s+{re.escape(name)}\s*\(", text, flags=re.M)
    if not m:
        return None
    start = m.start()
    m2 = re.search(r"^(def\s+|class\s+)", text[m.end():], flags=re.M)
    end = m.end() + m2.start() if m2 else len(text)
    return start, end


def backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + f".v46_34_router_slot_{time.strftime('%Y%m%d_%H%M%S')}.bak")
    shutil.copy2(path, bak)
    print("[BAK]", bak)
    return bak


PATCH_BLOCK = r'''
# ===== V46.34 PRETRAINED ROUTER SLOT PATCH START =====
def _v46_34_env_bool(name: str, default: bool = False) -> bool:
    try:
        return bool(int(os.environ.get(name, "1" if default else "0")))
    except Exception:
        return bool(default)


_V46_34_SEMANTIC_LABELS = [
    "calm_meditative",
    "lyrical_flow",
    "pose_hold",
    "instrument_phrase",
    "percussive_accent",
    "turning_climax",
    "footwork_flow",
]


def _v46_34_normalize_probs(obj, top_label=None):
    if isinstance(obj, dict):
        raw = {str(k): float(v) for k, v in obj.items() if str(k) in _V46_34_SEMANTIC_LABELS}
    else:
        raw = {}
    if not raw and top_label in _V46_34_SEMANTIC_LABELS:
        raw = {k: 0.02 for k in _V46_34_SEMANTIC_LABELS}
        raw[str(top_label)] = 0.88
    if not raw:
        raw = {k: 1.0 / len(_V46_34_SEMANTIC_LABELS) for k in _V46_34_SEMANTIC_LABELS}
    s = sum(max(0.0, float(v)) for v in raw.values())
    if s <= 1e-8:
        return {k: 1.0 / len(_V46_34_SEMANTIC_LABELS) for k in _V46_34_SEMANTIC_LABELS}
    return {k: float(max(0.0, raw.get(k, 0.0)) / s) for k in _V46_34_SEMANTIC_LABELS}


def _v46_34_top_label(probs):
    p = _v46_34_normalize_probs(probs)
    return max(p.items(), key=lambda kv: kv[1])[0]


def _v46_34_feature_from_slot(slot: dict) -> np.ndarray:
    dur = float(slot.get("duration", slot.get("duration_sec", 4.0)))
    probs = _v46_34_normalize_probs(slot.get("music_semantic_probs", {}), slot.get("music_semantic_top_label", slot.get("music_alignment_label")))
    energy = float(slot.get("energy", slot.get("music_energy", slot.get("slot_energy", 0.06))))
    onset = float(slot.get("onset", slot.get("accent", slot.get("music_accent_score", 0.02))))
    dyn = float(slot.get("dynamic", slot.get("beat_density", 0.04)))
    x = np.zeros(32, dtype=np.float32)
    x[0] = dur
    x[1] = 2.0 * energy + 0.25 * probs["footwork_flow"] + 0.25 * probs["percussive_accent"]
    x[2] = energy
    x[3] = float(slot.get("energy_p90", energy))
    x[4] = dyn
    x[5] = energy + onset + 0.35 * probs["percussive_accent"]
    x[6] = float(slot.get("onset_p90", onset))
    x[7] = energy + 0.5 * onset + 0.25 * probs["footwork_flow"]
    x[8] = energy
    x[9] = 1.0 + onset + 0.35 * probs["turning_climax"]
    x[10] = 0.65 * probs["calm_meditative"] + 0.45 * probs["pose_hold"]
    x[11] = probs["pose_hold"]
    x[12] = probs["calm_meditative"]
    x[13] = max(0.02, onset + 0.2 * probs["percussive_accent"])
    x[15] = 0.25 * probs["turning_climax"]
    x[16] = probs["turning_climax"]
    x[17] = probs["footwork_flow"]
    x[18] = probs["instrument_phrase"]
    x[19] = probs["lyrical_flow"]
    x[20] = probs["percussive_accent"]
    x[21] = probs["calm_meditative"]
    x[22] = probs["pose_hold"]
    for i, lab in enumerate(_V46_34_SEMANTIC_LABELS):
        if 23 + i < 32:
            x[23 + i] = probs[lab]
    x[31] = 1.0
    return x.astype(np.float32)


def _v46_34_find_slots_in_json(data):
    if isinstance(data, dict):
        if isinstance(data.get("slots"), list):
            return data.get("slots"), data
        sr = data.get("stage_reports")
        if isinstance(sr, dict) and isinstance(sr.get("retrieval"), list):
            slots = []
            for r in sr.get("retrieval", []):
                if not isinstance(r, dict):
                    continue
                slots.append({
                    "slot_id": r.get("slot", r.get("slot_id", len(slots))),
                    "duration": r.get("duration", 4.0),
                    "music_alignment_label": r.get("slot_music_alignment_label", r.get("music_alignment_label", "calm_meditative")),
                    "music_semantic_top_label": r.get("slot_music_semantic_top_label", r.get("slot_music_alignment_label", "calm_meditative")),
                    "music_semantic_probs": r.get("slot_music_semantic_probs", {}),
                    "preferred_dance_keys": r.get("slot_preferred_dance_keys", []),
                })
            if slots:
                return slots, data
        for v in data.values():
            got, meta = _v46_34_find_slots_in_json(v)
            if got is not None:
                return got, meta
    elif isinstance(data, list):
        if data and all(isinstance(x, dict) for x in data):
            keys = set()
            for x in data[: min(8, len(data))]:
                keys.update(x.keys())
            if "duration" in keys or {"start", "end"}.issubset(keys):
                return data, {"version": "list_slots"}
        for v in data:
            got, meta = _v46_34_find_slots_in_json(v)
            if got is not None:
                return got, meta
    return None, {}


def _v46_34_load_slots_json(slots_json: str | Path, cfg: V46Config) -> Tuple[List[dict], np.ndarray, dict]:
    data = load_json(slots_json)
    slots, meta = _v46_34_find_slots_in_json(data)
    if not slots:
        raise RuntimeError(f"V46.34 slots_json has no slots: {slots_json}")
    fps = float(getattr(cfg, "fps", 30.0))
    out_slots: List[dict] = []
    feats: List[np.ndarray] = []
    cursor = 0.0
    for i, s0 in enumerate(slots):
        s = dict(s0)
        dur = s.get("duration", s.get("duration_sec", None))
        st = s.get("start", s.get("start_sec", s.get("music_start", None)))
        ed = s.get("end", s.get("end_sec", s.get("music_end", None)))
        if dur is None and st is not None and ed is not None:
            dur = float(ed) - float(st)
        if dur is None:
            dur = 4.0
        dur = max(0.10, float(dur))
        if st is None:
            st = cursor
        st = float(st)
        if ed is None:
            ed = st + dur
        ed = float(ed)
        dur = max(0.10, ed - st)
        cursor = ed
        probs = _v46_34_normalize_probs(s.get("music_semantic_probs", {}), s.get("music_semantic_top_label", s.get("music_alignment_label")))
        top = s.get("music_semantic_top_label", s.get("music_alignment_label", _v46_34_top_label(probs)))
        if top not in _V46_34_SEMANTIC_LABELS:
            top = _v46_34_top_label(probs)
        base = np.asarray(s.get("feature", []), dtype=np.float32)
        if base.size < 32 or float(np.max(np.abs(base))) == 0.0:
            base = _v46_34_feature_from_slot({**s, "duration": dur, "music_semantic_probs": probs, "music_semantic_top_label": top})
        if base.size < 32:
            base = np.pad(base, (0, 32 - base.size))
        s.update({
            "slot_id": int(s.get("slot_id", s.get("slot", i))),
            "start": float(st),
            "end": float(ed),
            "duration": float(dur),
            "target_frames": int(s.get("target_frames", round(float(dur) * fps))),
            "music_alignment_label": str(s.get("music_alignment_label", top)),
            "music_semantic_top_label": str(top),
            "music_semantic_probs": probs,
            "slot_plan_source": str(s.get("slot_plan_source", meta.get("slot_source", "v46_34_slots_json"))),
            "feature": base[:32].astype(float).tolist(),
        })
        out_slots.append(s)
        feats.append(base[:32].astype(np.float32))
    return out_slots, np.stack(feats).astype(np.float32), meta if isinstance(meta, dict) else {}


def audio_slots(path: str | Path, cfg: V46Config, slot_seconds: float = 4.0, slots_json: Optional[str] = None) -> Tuple[List[dict], np.ndarray]:
    """V46.34 router-aware slot loader.

    Scientific mode: set V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=1 and pass
    --slots_json generated by tools/v46_34_pretrained_music_slot_plan.py.  This
    prevents accidental fallback to regular fixed-window audio slots.
    """
    strict = _v46_34_env_bool("V46_REQUIRE_PRETRAINED_ROUTER_SLOTS", False)
    if slots_json and Path(slots_json).exists():
        slots, feats, meta = _v46_34_load_slots_json(slots_json, cfg)
        allowed = not strict
        src = str(meta.get("slot_source", ""))
        raw = str(meta.get("router_ckpt", "")) + " " + str(meta.get("planner_ckpt", "")) + " " + src
        if ("v21" in raw.lower()) or ("v26" in raw.lower()) or ("pretrained" in raw.lower()) or ("router" in raw.lower()):
            allowed = True
        if not allowed:
            raise RuntimeError(
                "V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=1 but slots_json is not marked as pretrained V21/V26 router output. "
                f"slots_json={slots_json}, slot_source={src}"
            )
        print(f"[V46.34] loaded pretrained router slot plan: {slots_json} slots={len(slots)} source={src}")
        return slots, feats
    if strict:
        raise RuntimeError(
            "V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=1 but --slots_json was not provided or does not exist. "
            "Generate it with tools/v46_34_pretrained_music_slot_plan.py."
        )
    return audio_slots_v46_default(path, cfg, slot_seconds, slots_json)
# ===== V46.34 PRETRAINED ROUTER SLOT PATCH END =====
'''


def main() -> int:
    if not TARGET.exists():
        raise SystemExit(f"[ERROR] Missing {TARGET}. Run from EDGE root.")
    text = TARGET.read_text(encoding="utf-8")
    backup(TARGET)

    if MARK in text and END_MARK in text:
        text = re.sub(re.escape(MARK) + r".*?" + re.escape(END_MARK) + r"\n*", "", text, flags=re.S)

    # If previous patch already preserved default, remove any duplicate router-aware audio_slots block.
    if "def audio_slots_v46_default(" not in text:
        span = find_func_span(text, "audio_slots")
        if span is None:
            raise SystemExit("[ERROR] Cannot find def audio_slots in tools/v46_motionrag_diff.py")
        original = text[span[0]:span[1]]
        original_renamed = original.replace("def audio_slots(", "def audio_slots_v46_default(", 1)
        text = text[:span[0]] + original_renamed + "\n\n" + PATCH_BLOCK + "\n" + text[span[1]:]
    else:
        # Insert/refresh patched wrapper after default audio_slots function.
        span = find_func_span(text, "audio_slots_v46_default")
        if span is None:
            raise SystemExit("[ERROR] Cannot locate audio_slots_v46_default")
        text = text[:span[1]] + "\n\n" + PATCH_BLOCK + "\n" + text[span[1]:]

    TARGET.write_text(text, encoding="utf-8")
    print(f"[OK] Patched {TARGET} with V46.34 pretrained-router slot loader")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
