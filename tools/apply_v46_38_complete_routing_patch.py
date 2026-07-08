#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply V46.38 complete MSSD-AESD routing patch to tools/v46_motionrag_diff.py.

The patch is reversible and injects runtime overrides before the module's
`if __name__ == "__main__"` block.  That placement is important: it overrides
functions used by `train-contrastive` and `generate` after all original V46
functions have been defined, but before `main()` is executed.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

TARGET = Path("tools/v46_motionrag_diff.py")
START = "# ===== V46.38 COMPLETE MSSD-AESD ROUTING PATCH START ====="
END = "# ===== V46.38 COMPLETE MSSD-AESD ROUTING PATCH END ====="

PATCH = r'''
# ===== V46.38 COMPLETE MSSD-AESD ROUTING PATCH START =====
# Complete chain: MSSD music slots -> AESD action descriptors -> routing-aware
# Semantic OT for V44 -> routing-aware beam search for Event-RAG.
try:
    from tools.v46_38_music_action_descriptor import (
        MUSIC_SEMANTIC_LABELS as _V46_38_LABELS,
        parse_descriptor_file as _v46_38_parse_descriptor_file,
        load_descriptor_for_audio as _v46_38_load_descriptor_for_audio,
        env_bool as _v46_38_env_bool,
        get_aesd_prob_matrix as _v46_38_get_aesd_prob_matrix,
        slot_prob_vector as _v46_38_slot_prob_vector,
        dot_compat as _v46_38_dot_compat,
        normalize_vector as _v46_38_normalize_vector,
    )
except Exception as _v46_38_import_exc:  # pragma: no cover
    _V46_38_LABELS = ["calm_meditative", "pose_hold", "lyrical_flow", "instrument_phrase", "percussive_accent", "turning_climax", "footwork_flow", "aerial_curve"]
    _v46_38_parse_descriptor_file = None
    _v46_38_load_descriptor_for_audio = None
    _v46_38_env_bool = None
    _v46_38_get_aesd_prob_matrix = None
    _v46_38_slot_prob_vector = None
    _v46_38_dot_compat = None
    _v46_38_normalize_vector = None
    print(f"[V46.38 WARN] import failed: {_v46_38_import_exc}", file=sys.stderr)


def _v46_38_bool(name: str, default: bool = False) -> bool:
    try:
        return bool(int(os.environ.get(name, "1" if default else "0")))
    except Exception:
        return bool(default)


def _v46_38_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _v46_38_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _v46_38_descriptor_dirs_from_cfg(cfg: V46Config) -> str:
    parts = []
    for attr in ["music_descriptor_dirs", "external_music_semantic_dirs"]:
        val = getattr(cfg, attr, "")
        if val:
            if isinstance(val, (list, tuple)):
                parts.extend([str(x) for x in val])
            else:
                parts.extend(str(val).replace(";", os.pathsep).split(os.pathsep))
    extra = os.environ.get("V46_MSSD_DESCRIPTOR_DIRS", "") or os.environ.get("V46_38_DESCRIPTOR_DIRS", "")
    if extra:
        parts.extend(extra.replace(";", os.pathsep).split(os.pathsep))
    return os.pathsep.join([p for p in parts if str(p).strip()])


def _v46_38_mark_slots_with_meta(slots: List[dict], meta: dict) -> List[dict]:
    out = []
    for s0 in slots:
        s = dict(s0)
        s.setdefault("mssd_usage", meta.get("usage"))
        s.setdefault("mssd_is_final_schedule", bool(meta.get("is_final_schedule", False)))
        s.setdefault("mssd_slot_source", meta.get("slot_source", s.get("slot_source", "")))
        for k in ["router_ckpt", "planner_ckpt", "v23_ckpt", "raw_schedule_json", "schedule_summary_json", "descriptor_schema_version"]:
            if meta.get(k) is not None:
                s.setdefault("mssd_" + k, meta.get(k))
        out.append(s)
    return out


def _v46_38_load_slots_json(slots_json: str | Path, cfg: V46Config, require_final: bool = False) -> Tuple[List[dict], np.ndarray, dict]:
    if _v46_38_parse_descriptor_file is None:
        slots, feats, meta = _v46_34_load_slots_json(slots_json, cfg)
        return slots, feats, meta
    slots, feats, meta = _v46_38_parse_descriptor_file(
        slots_json,
        require_final_schedule=bool(require_final),
        fps=float(getattr(cfg, "fps", 30.0)),
        temperature=float(getattr(cfg, "external_music_semantic_temperature", 0.65)),
        usage="generate_schedule" if require_final else "auto",
    )
    slots = _v46_38_mark_slots_with_meta(slots, meta)
    return slots, feats, meta


def _v46_38_audio_slots(path: str | Path, cfg: V46Config, slot_seconds: float = 4.0, slots_json: Optional[str] = None) -> Tuple[List[dict], np.ndarray]:
    strict = _v46_34_env_bool("V46_REQUIRE_PRETRAINED_ROUTER_SLOTS", False)
    mssd_enabled = _v46_38_bool("V46_MSSD_ENABLE", True)
    require_final = strict or _v46_38_bool("V46_MSSD_REQUIRE_FINAL_SCHEDULE_FOR_GENERATE", False)
    if slots_json and Path(slots_json).exists():
        if mssd_enabled and _v46_38_parse_descriptor_file is not None:
            slots, feats, meta = _v46_38_load_slots_json(slots_json, cfg, require_final=require_final)
            print(f"[V46.38 MSSD] loaded descriptor: {slots_json} slots={len(slots)} usage={meta.get('usage')} final={meta.get('is_final_schedule')} source={meta.get('slot_source')}")
            return slots, feats
        slots, feats, meta = _v46_34_load_slots_json(slots_json, cfg)
        allowed = not strict
        raw = " ".join(str(meta.get(k, "")) for k in ["slot_source", "router_ckpt", "planner_ckpt", "v23_ckpt"])
        if any(k in raw.lower() for k in ["v21", "v23", "v26", "router", "planner", "pretrained"]):
            allowed = True
        if not allowed:
            raise RuntimeError("V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=1 but slots_json is not final trained-router/planner MSSD")
        return _v46_38_mark_slots_with_meta(slots, meta), feats
    if mssd_enabled and _v46_38_load_descriptor_for_audio is not None:
        loaded = _v46_38_load_descriptor_for_audio(
            path,
            descriptor_dirs=_v46_38_descriptor_dirs_from_cfg(cfg),
            require_final_schedule=require_final,
            fps=float(getattr(cfg, "fps", 30.0)),
            temperature=float(getattr(cfg, "external_music_semantic_temperature", 0.65)),
            usage="generate_schedule" if require_final else "auto",
        )
        if loaded is not None:
            slots, feats, meta = loaded
            slots = _v46_38_mark_slots_with_meta(slots, meta)
            print(f"[V46.38 MSSD] loaded sidecar descriptor: audio={path} slots={len(slots)} usage={meta.get('usage')} final={meta.get('is_final_schedule')} source={meta.get('slot_source')}")
            return slots, feats
    if strict:
        raise RuntimeError("V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=1 but no final MSSD/slots_json was provided. Build it with tools/v46_38_build_music_semantic_slot_descriptor.py")
    return audio_slots_v46_default(path, cfg, slot_seconds, slots_json)


# Override old loaders.  Generate uses strict final MSSD; V44 training does not.
audio_slots = _v46_38_audio_slots


def parse_external_music_semantic_file(path: str | Path, cfg: V46Config) -> Optional[Tuple[List[dict], np.ndarray]]:
    if _v46_38_parse_descriptor_file is None:
        return None
    try:
        slots, feats, meta = _v46_38_parse_descriptor_file(
            path,
            require_final_schedule=False,
            fps=float(getattr(cfg, "fps", 30.0)),
            temperature=float(getattr(cfg, "external_music_semantic_temperature", 0.65)),
            usage="train_semantic",
        )
        return _v46_38_mark_slots_with_meta(slots, meta), feats
    except Exception as exc:
        print(f"[V46.38 MSSD WARN] failed parsing weak descriptor {path}: {exc}", file=sys.stderr)
        return None


def load_external_music_semantic_slots(audio_path: str | Path, cfg: V46Config, slot_seconds: float) -> Optional[Tuple[List[dict], np.ndarray]]:
    if not bool(getattr(cfg, "external_music_semantic_enable", True)):
        return None
    if _v46_38_load_descriptor_for_audio is not None:
        loaded = _v46_38_load_descriptor_for_audio(
            audio_path,
            descriptor_dirs=_v46_38_descriptor_dirs_from_cfg(cfg),
            require_final_schedule=False,
            fps=float(getattr(cfg, "fps", 30.0)),
            temperature=float(getattr(cfg, "external_music_semantic_temperature", 0.65)),
            usage="train_semantic",
        )
        if loaded is not None:
            slots, feats, meta = loaded
            return _v46_38_mark_slots_with_meta(slots, meta), feats
    cmd_out = run_external_music_semantic_cmd(audio_path, cfg)
    if cmd_out is not None:
        parsed = parse_external_music_semantic_file(cmd_out, cfg)
        if parsed is not None:
            return parsed
    if bool(getattr(cfg, "external_music_semantic_proxy_enable", True)):
        prox = filename_proxy_music_semantic(audio_path, cfg, slot_seconds)
        if prox is not None:
            return prox
    if bool(getattr(cfg, "external_music_semantic_required", False)):
        raise RuntimeError(f"External/MSSD music semantic is required but none was found for {audio_path}")
    return None


def load_unpaired_audio_feature_pool(audio_dirs: Optional[Sequence[str]], cfg: V46Config) -> Tuple[np.ndarray, List[dict]]:
    files = collect_audio_files(audio_dirs)
    feats: List[np.ndarray] = []
    meta: List[dict] = []
    old_strict = os.environ.get("V46_REQUIRE_PRETRAINED_ROUTER_SLOTS")
    try:
        os.environ["V46_REQUIRE_PRETRAINED_ROUTER_SLOTS"] = "0"
        for f in files:
            try:
                parsed = load_external_music_semantic_slots(f, cfg, slot_seconds=float(cfg.unpaired_audio_slot_seconds))
                if parsed is not None:
                    slots, sf = parsed
                else:
                    slots, sf = audio_slots_v46_default(f, cfg, slot_seconds=float(cfg.unpaired_audio_slot_seconds), slots_json=None)
            except Exception as exc:
                print(f"[V46.38 WARN] failed unpaired audio feature extraction {f}: {exc}", file=sys.stderr)
                continue
            for slot, feat in zip(slots, sf):
                feats.append(feat.astype(np.float32))
                meta.append({"audio": str(f), "slot": dict(slot)})
    finally:
        if old_strict is None:
            os.environ.pop("V46_REQUIRE_PRETRAINED_ROUTER_SLOTS", None)
        else:
            os.environ["V46_REQUIRE_PRETRAINED_ROUTER_SLOTS"] = old_strict
    if not feats:
        return np.zeros((0, 32), dtype=np.float32), []
    return np.stack(feats).astype(np.float32), meta


def _v46_38_slot_prob_matrix(audio_meta: List[dict]) -> np.ndarray:
    rows = []
    for m in audio_meta:
        slot = m.get("slot", {})
        if _v46_38_slot_prob_vector is not None:
            rows.append(_v46_38_slot_prob_vector(slot))
        else:
            rows.append(np.ones((len(_V46_38_LABELS),), dtype=np.float32) / max(1, len(_V46_38_LABELS)))
    return np.stack(rows).astype(np.float32) if rows else np.zeros((0, len(_V46_38_LABELS)), dtype=np.float32)


def build_unpaired_audio_motion_pairs(db: dict, audio_dirs: Optional[Sequence[str]], cfg: V46Config) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]]:
    """V46.38 MSSD-AESD Semantic OT.

    It still uses real unpaired music features and motion descriptors, but the OT
    cost now includes a soft MSSD-to-AESD probability distance.  Thus V44 is no
    longer trained from only low-level energy/duration hints.
    """
    audio_raw, audio_meta = load_unpaired_audio_feature_pool(audio_dirs, cfg)
    if audio_raw.shape[0] < int(cfg.unpaired_min_audio_slots):
        return None
    motion_z = motion_feature_z_for_alignment(db, cfg, weight=float(getattr(cfg, "classification_ot_weight", getattr(cfg, "filename_semantic_ot_weight", 0.35))))
    desc_mean = np.asarray(db["desc_mean"], dtype=np.float32)
    desc_std = np.asarray(db["desc_std"], dtype=np.float32)
    music_z_all = ((audio_raw - desc_mean) / np.maximum(desc_std, 1e-6)).astype(np.float32)
    music_z_all = np.clip(music_z_all, -8.0, 8.0).astype(np.float32)
    motion_z = np.clip(motion_z, -8.0, 8.0).astype(np.float32)
    dims, weights = semantic_dims_and_weights()
    diff = music_z_all[:, None, dims] - motion_z[None, :, dims]
    cost_num = np.sum((diff * weights[None, None, :]) ** 2, axis=-1)
    if _v46_38_get_aesd_prob_matrix is not None:
        slot_probs = _v46_38_slot_prob_matrix(audio_meta)
        aesd_probs = _v46_38_get_aesd_prob_matrix(db, motion_z.shape[0])
        sem_compat = np.clip(slot_probs @ aesd_probs.T, 0.0, 1.0)
        cost_sem = 1.0 - sem_compat
    else:
        sem_compat = np.zeros((music_z_all.shape[0], motion_z.shape[0]), dtype=np.float32)
        cost_sem = 0.0
    lam_sem = _v46_38_float("V46_38_OT_SEMANTIC_WEIGHT", 1.25)
    rng = np.random.default_rng(int(cfg.seed) + 4638)
    cost = cost_num + lam_sem * cost_sem + rng.normal(0.0, 1e-5, size=cost_num.shape).astype(np.float32)
    topk = max(1, min(int(cfg.unpaired_positive_topk), motion_z.shape[0]))
    pairs_per = max(1, min(int(cfg.unpaired_pairs_per_audio_slot), topk))
    music_pairs: List[np.ndarray] = []
    motion_pairs: List[np.ndarray] = []
    pair_preview: List[dict] = []
    for ai in range(cost.shape[0]):
        top = np.argpartition(cost[ai], topk - 1)[:topk]
        top = top[np.argsort(cost[ai, top])]
        chosen = top[:pairs_per]
        for mi in chosen:
            music_pairs.append(music_z_all[ai])
            motion_pairs.append(motion_z[int(mi)])
        if len(pair_preview) < 20:
            pair_preview.append({
                "audio": audio_meta[ai].get("audio", ""),
                "slot_id": int(audio_meta[ai].get("slot", {}).get("slot_id", ai)),
                "slot_music_semantic_label": str(audio_meta[ai].get("slot", {}).get("music_semantic_top_label", audio_meta[ai].get("slot", {}).get("music_alignment_label", ""))),
                "slot_descriptor_usage": str(audio_meta[ai].get("slot", {}).get("usage", audio_meta[ai].get("slot", {}).get("mssd_usage", ""))),
                "top_motion_ids": [int(x) for x in top[: min(5, len(top))].tolist()],
                "top_costs": [float(cost[ai, int(x)]) for x in top[: min(5, len(top))].tolist()],
                "top_mssd_aesd_compat": [float(sem_compat[ai, int(x)]) for x in top[: min(5, len(top))].tolist()] if isinstance(sem_compat, np.ndarray) and sem_compat.size else [],
            })
    if len(music_pairs) < 2:
        return None
    music = np.stack(music_pairs).astype(np.float32)
    motion = np.stack(motion_pairs).astype(np.float32)
    report = {
        "mode": "v46_38_mssd_aesd_semantic_ot",
        "audio_files": sorted(set(m["audio"] for m in audio_meta)),
        "num_audio_slots": int(audio_raw.shape[0]),
        "num_motion_events": int(motion_z.shape[0]),
        "num_training_pairs": int(music.shape[0]),
        "positive_topk": int(topk),
        "pairs_per_audio_slot": int(pairs_per),
        "semantic_dims": [int(x) for x in dims.tolist()],
        "semantic_weights": [float(x) for x in weights.tolist()],
        "mssd_aesd_semantic_weight": float(lam_sem),
        "aesd_arrays_used": bool("aesd_music_alignment_probs" in db),
        "pair_preview": pair_preview,
    }
    return music, motion, desc_mean.astype(np.float32), desc_std.astype(np.float32), report


def _v46_38_stage_score(slot: dict, stages: np.ndarray) -> np.ndarray:
    role = str(slot.get("role", slot.get("slot_role", "normal")))
    out = np.zeros((len(stages),), dtype=np.float32)
    good = {
        "intro": {"intro", "intro_or_resolution", "anchor_or_resolution"},
        "calm": {"intro", "intro_or_resolution", "resolution", "anchor_or_resolution"},
        "release": {"resolution", "anchor_or_resolution", "intro_or_resolution"},
        "resolution": {"resolution", "anchor_or_resolution", "intro_or_resolution"},
        "normal": {"development", "build_up", "motif_recall"},
        "development": {"development", "build_up"},
        "build_up": {"build_up", "development", "opening_or_climax"},
        "accent": {"accent_or_climax", "climax", "build_up"},
        "climax": {"climax", "accent_or_climax", "opening_or_climax"},
        "motif": {"motif_recall", "development"},
        "motif_recall": {"motif_recall", "anchor_or_resolution"},
    }.get(role, {"development", "build_up"})
    for i, st in enumerate(stages):
        if str(st) in good:
            out[i] = 1.0
    return out


def retrieve_schedule(slots: List[dict], slot_feat: np.ndarray, db: dict, cfg: V46Config, contrastive=None) -> Tuple[List[int], List[dict]]:
    """V46.38 global MSSD-AESD routing-aware Event-RAG."""
    desc = np.asarray(db["desc"], dtype=np.float32)
    desc_z = motion_feature_z_for_alignment(db, cfg, weight=float(getattr(cfg, "classification_retrieval_weight", getattr(cfg, "filename_semantic_retrieval_weight", 0.20))))
    mean = np.asarray(db["desc_mean"], dtype=np.float32)
    std = np.asarray(db["desc_std"], dtype=np.float32)
    if contrastive is not None and hasattr(contrastive, "music_mean") and hasattr(contrastive, "music_std"):
        music_mean = np.asarray(getattr(contrastive, "music_mean"), dtype=np.float32)
        music_std = np.asarray(getattr(contrastive, "music_std"), dtype=np.float32)
        music_z = (slot_feat - music_mean) / np.maximum(music_std, 1e-6)
    else:
        music_z = (slot_feat - mean) / np.maximum(std, 1e-6)
    music_z = np.clip(music_z, -8.0, 8.0).astype(np.float32)
    desc_z = np.clip(desc_z, -8.0, 8.0).astype(np.float32)
    music_emb, motion_emb = embed_with_contrastive(contrastive, music_z, desc_z, cfg)
    n = int(len(desc))
    sources = np.asarray(db["source_groups"], dtype=object)
    source_uids = np.asarray(db.get("source_uids", sources), dtype=object)
    durations = np.asarray(db["durations"], dtype=np.float32)
    entries = np.asarray(db["entry"], dtype=np.float32); exits = np.asarray(db["exit"], dtype=np.float32)
    centry = np.asarray(db["contact_entry"], dtype=np.float32); cexit = np.asarray(db["contact_exit"], dtype=np.float32)
    dance_keys = np.asarray(db.get("dance_keys", np.array(["unknown"] * n, dtype=object)), dtype=object)
    labels_arr = np.asarray(db.get("labels", np.array(["unknown"] * n, dtype=object)), dtype=object)
    align_arr = np.asarray(db.get("music_alignment_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)
    families = np.asarray(db.get("event_families", np.array(["unknown"] * n, dtype=object)), dtype=object)
    stages = np.asarray(db.get("motion_stage_roles", np.array(["unknown"] * n, dtype=object)), dtype=object)
    locomotion = np.asarray(db.get("locomotion_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)
    support = np.asarray(db.get("support_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)
    sem_conf = np.asarray(db.get("semantic_confidence", np.ones(n, dtype=np.float32)), dtype=np.float32)
    event_quality = np.asarray(db.get("event_quality_scores", np.ones(n, dtype=np.float32)), dtype=np.float32)
    nat_min = np.asarray(db.get("natural_duration_min", np.ones(n, dtype=np.float32) * 1.5), dtype=np.float32)
    nat_max = np.asarray(db.get("natural_duration_max", np.ones(n, dtype=np.float32) * 4.0), dtype=np.float32)
    aesd_probs = _v46_38_get_aesd_prob_matrix(db, n) if _v46_38_get_aesd_prob_matrix is not None else np.zeros((n, len(_V46_38_LABELS)), dtype=np.float32)
    aesd_risk = np.asarray(db.get("aesd_boundary_risk", np.zeros(n, dtype=np.float32)), dtype=np.float32)
    aesd_semantics = np.asarray(db.get("aesd_event_semantics", align_arr), dtype=object)

    w_contrastive = _v46_38_float("V46_38_ROUTE_CONTRASTIVE_WEIGHT", 1.00)
    w_mssd_aesd = _v46_38_float("V46_38_ROUTE_MSSD_AESD_WEIGHT", 1.15)
    w_legacy_sem = _v46_38_float("V46_38_ROUTE_LEGACY_SEM_WEIGHT", float(getattr(cfg, "semantic_routing_weight", 0.72)))
    w_duration = _v46_38_float("V46_38_ROUTE_DURATION_WEIGHT", float(getattr(cfg, "route_natural_duration_weight", 0.20)))
    w_quality = _v46_38_float("V46_38_ROUTE_QUALITY_WEIGHT", float(getattr(cfg, "event_quality_weight", 0.22)))
    w_stage = _v46_38_float("V46_38_ROUTE_STAGE_WEIGHT", float(getattr(cfg, "route_stage_sequence_weight", 0.16)))
    w_boundary_risk = _v46_38_float("V46_38_ROUTE_BOUNDARY_RISK_WEIGHT", 0.35)
    top_debug = max(1, int(getattr(cfg, "classification_report_topk", 8)))
    candidate_k = max(int(getattr(cfg, "top_k", 32)), int(getattr(cfg, "beam_size", 8)), _v46_38_int("V46_38_ROUTE_CANDIDATE_TOPK", 96), top_debug)

    beams: List[Tuple[float, List[int], Dict[str, int]]] = [(0.0, [], {})]
    reports: List[dict] = []
    for i, slot in enumerate(slots):
        sim = music_emb[i] @ motion_emb.T
        slot_dur = max(float(slot.get("duration", durations.mean() if len(durations) else 1.0)), 1e-4)
        dur_cost = np.abs(np.log(np.maximum(durations, 1e-4) / slot_dur))
        in_range = ((slot_dur >= nat_min) & (slot_dur <= nat_max)).astype(np.float32)
        center = np.maximum((nat_min + nat_max) * 0.5, 1e-4)
        natural_score = in_range + (1.0 - in_range) * np.exp(-np.abs(np.log(slot_dur / center))).astype(np.float32)
        legacy_sem = semantic_label_match_bonus(slot, db, cfg)
        if _v46_38_slot_prob_vector is not None and _v46_38_dot_compat is not None:
            slot_prob = _v46_38_slot_prob_vector(slot)
            mssd_aesd_score = _v46_38_dot_compat(slot_prob, aesd_probs)
        else:
            mssd_aesd_score = legacy_sem.astype(np.float32)
        stage_score = _v46_38_stage_score(slot, stages)
        quality_term = np.clip(event_quality, 0.0, 1.0)
        low_quality_penalty = np.maximum(0.0, float(getattr(cfg, "chang_e_min_event_quality", 0.22)) - quality_term)
        base_score = (
            w_contrastive * sim
            + w_mssd_aesd * mssd_aesd_score
            + w_legacy_sem * legacy_sem
            + w_duration * natural_score
            + w_quality * quality_term
            + w_stage * stage_score
            + 0.04 * np.clip(sem_conf, 0.0, 1.0)
            - float(getattr(cfg, "retrieval_warp_penalty", 0.18)) * dur_cost
            - w_boundary_risk * np.clip(aesd_risk, 0.0, 1.0)
            - 0.75 * low_quality_penalty
        )
        cand = np.argsort(-base_score)[: min(candidate_k, n)].tolist()
        new_beams: List[Tuple[float, List[int], Dict[str, int]]] = []
        for score, path, usage in beams:
            prev = path[-1] if path else None
            for idx in cand:
                sc = float(base_score[idx])
                src = str(sources[idx]); suid = str(source_uids[idx]); dk = str(dance_keys[idx]); fam = str(families[idx]); stg = str(stages[idx])
                source_key = "src::" + src
                uid_key = "suid::" + suid
                dance_key = "dance::" + dk
                fam_key = "fam::" + fam
                sc -= float(getattr(cfg, "route_source_repeat_penalty", cfg.retrieval_source_penalty)) * usage.get(source_key, 0)
                sc -= 0.08 * usage.get(uid_key, 0)
                sc -= float(getattr(cfg, "route_dance_key_repeat_penalty", 0.16)) * usage.get(dance_key, 0)
                fam_recent_window = max(1, int(getattr(cfg, "route_family_recent_window", 8)))
                fam_recent_count = sum(1 for p_idx in path[-fam_recent_window:] if str(families[p_idx]) == fam)
                fam_pen = float(getattr(cfg, "route_family_balance_penalty", 0.18)) * max(0, fam_recent_count - 1)
                sc -= min(float(getattr(cfg, "route_family_penalty_cap", 0.25)), fam_pen)
                run_count = 0
                for p_idx in reversed(path):
                    if str(sources[p_idx]) == src:
                        run_count += 1
                    else:
                        break
                if run_count >= 2:
                    sc -= float(getattr(cfg, "route_source_run_hard_penalty", 0.30))
                if str(slot.get("role", "")) in {"motif", "motif_recall"} and usage.get(fam_key, 0) > 0:
                    sc += float(getattr(cfg, "route_motif_recall_bonus", 0.12))
                if i == 0 and stg in {"intro", "intro_or_resolution"}:
                    sc += w_stage
                elif i >= len(slots) - 2 and stg in {"resolution", "anchor_or_resolution", "intro_or_resolution"}:
                    sc += w_stage
                if prev is not None:
                    raw_tc = transition_cost(exits[prev], entries[idx], cexit[prev], centry[idx])
                    transition_pen = float(getattr(cfg, "retrieval_transition_penalty", 0.65)) * raw_tc
                    # Risk-aware local transition penalty: difficult incoming events need cleaner previous exits.
                    transition_pen += 0.18 * float(aesd_risk[idx])
                    if str(support[prev]) != str(support[idx]):
                        transition_pen += 0.04
                    sc -= transition_pen
                    if src == str(sources[prev]):
                        sc -= float(getattr(cfg, "retrieval_repeat_penalty", 0.15))
                    if fam == str(families[prev]):
                        sc -= float(getattr(cfg, "route_family_repeat_penalty", 0.12))
                ns = dict(usage)
                ns[source_key] = ns.get(source_key, 0) + 1
                ns[uid_key] = ns.get(uid_key, 0) + 1
                ns[dance_key] = ns.get(dance_key, 0) + 1
                ns[fam_key] = ns.get(fam_key, 0) + 1
                new_beams.append((score + sc, path + [int(idx)], ns))
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[: max(1, int(getattr(cfg, "beam_size", 8)))]
        preview = []
        for j in cand[: min(top_debug, len(cand))]:
            j = int(j)
            preview.append({
                "event_id": j,
                "final_local_base_score": float(base_score[j]),
                "contrastive_similarity": float(sim[j]),
                "mssd_aesd_semantic_score": float(mssd_aesd_score[j]),
                "legacy_semantic_bonus": float(legacy_sem[j]),
                "natural_duration_score": float(natural_score[j]),
                "duration_log_cost": float(dur_cost[j]),
                "stage_score": float(stage_score[j]),
                "event_quality": float(event_quality[j]),
                "aesd_boundary_risk": float(aesd_risk[j]),
                "source": str(sources[j]),
                "source_uid": str(source_uids[j]),
                "label": str(labels_arr[j]),
                "dance_key": str(dance_keys[j]),
                "event_family": str(families[j]),
                "aesd_event_semantic": str(aesd_semantics[j]),
                "motion_stage_role": str(stages[j]),
                "support_label": str(support[j]),
                "locomotion_label": str(locomotion[j]),
                "music_alignment_label": str(align_arr[j]),
            })
        reports.append({
            "slot": i,
            "start": slot.get("start"),
            "end": slot.get("end"),
            "duration": slot.get("duration"),
            "target_frames": slot.get("target_frames"),
            "slot_role": slot.get("role", slot.get("slot_role")),
            "slot_music_alignment_label": slot.get("music_alignment_label"),
            "slot_music_semantic_top_label": slot.get("music_semantic_top_label", slot.get("music_alignment_label")),
            "slot_music_semantic_probs": slot.get("music_semantic_probs", {}),
            "slot_preferred_dance_keys": slot.get("preferred_dance_keys", []),
            "mssd_audit": {k: slot.get(k) for k in ["usage", "is_final_schedule", "slot_source", "mssd_usage", "mssd_is_final_schedule", "mssd_slot_source", "mssd_router_ckpt", "mssd_planner_ckpt", "mssd_v23_ckpt", "mssd_raw_schedule_json"] if k in slot},
            "top_candidate": int(cand[0]) if cand else -1,
            "beam_best_score": float(beams[0][0]) if beams else float("nan"),
            "routing_policy": "V46.38 MSSD-AESD global Event-RAG: contrastive + MSSD-AESD semantic + natural duration + stage + quality + boundary/source/family costs",
            "routing_weights": {"contrastive": w_contrastive, "mssd_aesd": w_mssd_aesd, "legacy_semantic": w_legacy_sem, "duration": w_duration, "quality": w_quality, "stage": w_stage, "boundary_risk": w_boundary_risk},
            "candidate_preview": preview,
        })
    return beams[0][1], reports
# ===== V46.38 COMPLETE MSSD-AESD ROUTING PATCH END =====
'''


def main() -> int:
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)
    text = TARGET.read_text(encoding="utf-8")
    if START in text and END in text:
        pre = text.split(START, 1)[0]
        post = text.split(END, 1)[1]
        new_text = pre + PATCH + post
    else:
        marker = 'if __name__ == "__main__":'
        idx = text.rfind(marker)
        if idx < 0:
            raise RuntimeError('Could not find final if __name__ == "__main__" marker')
        new_text = text[:idx] + PATCH + "\n\n" + text[idx:]
    if new_text == text:
        print("[OK] V46.38 complete routing patch already up to date")
        return 0
    backup = TARGET.with_suffix(TARGET.suffix + f".v46_38_complete_routing_{time.strftime('%Y%m%d_%H%M%S')}.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(new_text, encoding="utf-8")
    print(f"[BAK] {backup}")
    print(f"[OK] patched {TARGET} with V46.38 complete MSSD-AESD routing overrides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
