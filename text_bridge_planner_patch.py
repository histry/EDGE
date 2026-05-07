"""Text Bridge candidate filtering for V10 UnifiedChoreoPlanner.

This patch upgrades Text Bridge from a weak score bonus to a real retrieval
control mechanism.

Previous behavior:
    final_score = motion_score + weight * semantic_score
This often did not change Beam Search results because start/end and transition
compatibility dominated the global path score.

New behavior:
    EDGE_TEXT_BRIDGE_MODE=rerank     -> old compatible behavior
    EDGE_TEXT_BRIDGE_MODE=hybrid     -> union(original top-K, semantic top-K)
    EDGE_TEXT_BRIDGE_MODE=filter     -> restrict candidate pool to semantic top-K
    EDGE_TEXT_BRIDGE_MODE=force_topk -> semantic top-K is the candidate pool and
                                       semantic score is the primary emission score

Recommended verification:
    EDGE_TEXT_BRIDGE_MODE=filter EDGE_TEXT_BRIDGE_TOP_K=256 EDGE_TEXT_BRIDGE_WEIGHT=1.0
    EDGE_TEXT_BRIDGE_MODE=force_topk EDGE_TEXT_BRIDGE_TOP_K=128 EDGE_TEXT_BRIDGE_WEIGHT=1.0

Environment variables:
    EDGE_TEXT_BRIDGE_WEIGHT        default 0.0
    EDGE_TEXT_BRIDGE_MODE          rerank|hybrid|filter|force_topk
    EDGE_TEXT_BRIDGE_TOP_K         default 256
    EDGE_TEXT_BRIDGE_SCORE_NORM    1/0, default 1
    EDGE_TEXT_BRIDGE_ORIG_MIX      default 0.15 in force_topk
    EDGE_TEXT_QUERY                custom Chinese/English query
    EDGE_TEXT_BRIDGE_MODEL         default BAAI/bge-small-zh-v1.5
    EDGE_TEXT_BRIDGE_DEVICE        cpu|cuda
"""

from __future__ import annotations

import os
from functools import wraps
from typing import Dict, Iterable, List, Tuple

import numpy as np

from text_context_rag_utils import cosine_scores, default_query_for_mode, encode_texts, env_float, env_int, env_str


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _semantic_weight() -> float:
    return env_float("EDGE_TEXT_BRIDGE_WEIGHT", 0.0)


def _semantic_mode() -> str:
    mode = env_str("EDGE_TEXT_BRIDGE_MODE", "rerank").strip().lower()
    aliases = {
        "score": "rerank",
        "reranking": "rerank",
        "semantic_filter": "filter",
        "topk": "force_topk",
        "force": "force_topk",
        "force_top": "force_topk",
    }
    return aliases.get(mode, mode)


def _top_k() -> int:
    return max(1, env_int("EDGE_TEXT_BRIDGE_TOP_K", 256))


def _normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo + eps)).astype(np.float32)


def _clone_score(score, UnitScore):
    try:
        from dataclasses import asdict
        return UnitScore(**asdict(score))
    except Exception:
        return score


def _safe_unit_id(score) -> str:
    return str(getattr(score, "unit_id", f"unit_{getattr(score, 'index', -1)}"))


def _load_text_fields_from_npz(rag_db: str):
    try:
        npz = np.load(rag_db, allow_pickle=True)
        motion_text = np.asarray(npz["motion_text"]) if "motion_text" in npz.files else None
        if "motion_text_embedding" in npz.files:
            emb = np.asarray(npz["motion_text_embedding"], dtype=np.float32)
        elif "motion_embedding" in npz.files:
            emb = np.asarray(npz["motion_embedding"], dtype=np.float32)
        else:
            emb = None
        return motion_text, emb
    except Exception as exc:
        print(f"⚠️ Could not load text bridge fields from RAG DB: {exc}")
        return None, None


def install_text_bridge_planner_patch(verbose: bool = True) -> bool:
    try:
        import v10_choreo_planner as planner
    except Exception as exc:
        if verbose:
            print(f"⚠️ Text Bridge candidate-filter patch skipped: {exc}")
        return False

    if getattr(planner, "_edge_text_bridge_candidate_filter_patch_installed", False):
        return True

    RAGUnitDB = getattr(planner, "RAGUnitDB", None)
    UnifiedChoreoPlanner = getattr(planner, "UnifiedChoreoPlanner", None)
    UnitScore = getattr(planner, "UnitScore", None)
    if RAGUnitDB is None or UnifiedChoreoPlanner is None or UnitScore is None:
        if verbose:
            print("⚠️ Text Bridge candidate-filter patch skipped: missing planner classes.")
        return False

    original_db_init = RAGUnitDB.__init__
    original_score_all = UnifiedChoreoPlanner._score_all
    original_score_parts = planner._score_parts_from_score

    @wraps(original_db_init)
    def patched_db_init(self, rag_db: str, max_units=None):
        original_db_init(self, rag_db, max_units=max_units)
        motion_text, emb = _load_text_fields_from_npz(rag_db)
        self.motion_text = motion_text
        self.motion_text_embedding = emb
        if verbose and emb is not None:
            print(f"✅ Text Bridge fields loaded from RAG DB: embeddings={emb.shape}")

    def _semantic_vectors(self, scores):
        db = getattr(self, "db", None)
        emb = getattr(db, "motion_text_embedding", None)
        if emb is None:
            return None, None, None, None

        query = default_query_for_mode(getattr(self.config, "mode", "auto_multiunit"))
        try:
            q = encode_texts([query])[0]
            all_sims = cosine_scores(q, emb)
        except Exception as exc:
            if verbose:
                print(f"⚠️ Text Bridge semantic scoring failed: {exc}")
            return None, None, None, None

        idxs = np.asarray([int(getattr(s, "index", -1)) for s in scores], dtype=np.int64)
        valid = (idxs >= 0) & (idxs < len(all_sims))
        sims = np.full((len(scores),), -1e9, dtype=np.float32)
        sims[valid] = all_sims[idxs[valid]]

        if _env_bool("EDGE_TEXT_BRIDGE_SCORE_NORM", True):
            finite = sims > -1e8
            if np.any(finite):
                sims_norm = sims.copy()
                sims_norm[finite] = _normalize01(sims[finite])
                sims_norm[~finite] = 0.0
            else:
                sims_norm = np.zeros_like(sims)
        else:
            sims_norm = sims.copy()
            sims_norm[sims_norm < -1e8] = 0.0

        return query, sims, sims_norm, all_sims

    def _annotate_scores(self, scores, query, sims, sims_norm, weight, mode):
        db = getattr(self, "db", None)
        motion_text = getattr(db, "motion_text", None)

        out = []
        for pos, base in enumerate(scores):
            s = _clone_score(base, UnitScore)
            raw = float(sims[pos])
            norm = float(sims_norm[pos])
            setattr(s, "semantic_score", raw)
            setattr(s, "semantic_score_norm", norm)
            setattr(s, "semantic_query", str(query))
            setattr(s, "text_bridge_weight", float(weight))
            setattr(s, "text_bridge_mode", str(mode))

            idx = int(getattr(s, "index", -1))
            if motion_text is not None and 0 <= idx < len(motion_text):
                try:
                    setattr(s, "motion_text", str(motion_text[idx]))
                except Exception:
                    pass

            original_score = float(getattr(s, "score", 0.0))
            setattr(s, "original_score", original_score)

            if mode == "force_topk":
                # Make semantic similarity the primary emission objective.
                # A small original-score mix preserves basic motion quality.
                orig_mix = env_float("EDGE_TEXT_BRIDGE_ORIG_MIX", 0.15)
                s.score = float(weight * norm + orig_mix * original_score)
            else:
                s.score = float(original_score + weight * norm)

            s.emission_score = float(s.score)
            out.append(s)
        return out

    @wraps(original_score_all)
    def patched_score_all(self):
        base_scores = list(original_score_all(self))
        weight = _semantic_weight()
        mode = _semantic_mode()

        if weight <= 0.0 or mode in {"off", "none", "disable", "disabled"}:
            return base_scores

        query, sims_raw, sims_norm, _all_sims = _semantic_vectors(self, base_scores)
        if query is None:
            if verbose and not getattr(self, "_edge_text_bridge_no_emb_warned", False):
                print("⚠️ EDGE_TEXT_BRIDGE_WEIGHT>0 but no usable motion_text_embedding; semantic filtering skipped.")
                self._edge_text_bridge_no_emb_warned = True
            return base_scores

        top_k = min(_top_k(), len(base_scores))
        semantic_order = np.argsort(-sims_norm)[:top_k].tolist()

        if mode == "rerank":
            selected_positions = list(range(len(base_scores)))
            annotated = _annotate_scores(self, base_scores, query, sims_raw, sims_norm, weight, mode)
            result = sorted(annotated, key=lambda x: float(getattr(x, "score", 0.0)), reverse=True)

        elif mode == "hybrid":
            # Keep original strong motion/transition candidates and add semantic candidates.
            original_keep = list(range(min(len(base_scores), max(getattr(self.config, "top_k", 64), top_k))))
            pos_set = []
            seen = set()
            for p in original_keep + semantic_order:
                idx = int(getattr(base_scores[p], "index", -1))
                if idx not in seen:
                    seen.add(idx)
                    pos_set.append(p)
            annotated = _annotate_scores(
                self,
                [base_scores[p] for p in pos_set],
                query,
                sims_raw[pos_set],
                sims_norm[pos_set],
                weight,
                mode,
            )
            result = sorted(annotated, key=lambda x: float(getattr(x, "score", 0.0)), reverse=True)

        elif mode == "filter":
            # Candidate pool is semantic top-K, then original motion score + semantic bonus ranks inside it.
            pos_set = semantic_order
            annotated = _annotate_scores(
                self,
                [base_scores[p] for p in pos_set],
                query,
                sims_raw[pos_set],
                sims_norm[pos_set],
                weight,
                mode,
            )
            result = sorted(annotated, key=lambda x: float(getattr(x, "score", 0.0)), reverse=True)

        elif mode == "force_topk":
            # Candidate pool is semantic top-K and semantic score is primary.
            pos_set = semantic_order
            annotated = _annotate_scores(
                self,
                [base_scores[p] for p in pos_set],
                query,
                sims_raw[pos_set],
                sims_norm[pos_set],
                weight,
                mode,
            )
            result = sorted(
                annotated,
                key=lambda x: (
                    float(getattr(x, "semantic_score_norm", 0.0)),
                    float(getattr(x, "score", 0.0)),
                ),
                reverse=True,
            )

        else:
            if verbose and not getattr(self, "_edge_text_bridge_bad_mode_warned", False):
                print(f"⚠️ Unknown EDGE_TEXT_BRIDGE_MODE={mode!r}; falling back to rerank.")
                self._edge_text_bridge_bad_mode_warned = True
            annotated = _annotate_scores(self, base_scores, query, sims_raw, sims_norm, weight, "rerank")
            result = sorted(annotated, key=lambda x: float(getattr(x, "score", 0.0)), reverse=True)

        if verbose and not getattr(self, "_edge_text_bridge_logged", False):
            top_semantic = sorted(result, key=lambda x: float(getattr(x, "semantic_score_norm", 0.0)), reverse=True)[:5]
            top_final = result[:5]
            print(
                "✅ Text Bridge candidate filtering active: "
                f"mode={mode}, weight={weight}, top_k={top_k}, query={query!r}"
            )
            print(
                "   top_semantic="
                + ", ".join(
                    f"{_safe_unit_id(x)}:raw={float(getattr(x,'semantic_score',0.0)):.3f},norm={float(getattr(x,'semantic_score_norm',0.0)):.3f}"
                    for x in top_semantic
                )
            )
            print(
                "   top_final="
                + ", ".join(
                    f"{_safe_unit_id(x)}:score={float(getattr(x,'score',0.0)):.3f},sem={float(getattr(x,'semantic_score_norm',0.0)):.3f}"
                    for x in top_final
                )
            )
            self._edge_text_bridge_logged = True

        return result

    def patched_score_parts(score):
        d = original_score_parts(score)
        for key in [
            "semantic_score",
            "semantic_score_norm",
            "semantic_query",
            "motion_text",
            "text_bridge_weight",
            "text_bridge_mode",
            "original_score",
        ]:
            if hasattr(score, key):
                value = getattr(score, key)
                try:
                    if isinstance(value, (float, int, np.floating, np.integer)):
                        value = float(value)
                except Exception:
                    pass
                d[key] = value
        return d

    RAGUnitDB.__init__ = patched_db_init
    UnifiedChoreoPlanner._score_all = patched_score_all
    planner._score_parts_from_score = patched_score_parts

    planner._edge_text_bridge_candidate_filter_patch_installed = True
    planner._edge_text_bridge_planner_patch_installed = True

    if verbose:
        print("✅ Installed Text Bridge candidate-filter planner patch.")
    return True


def install():
    return install_text_bridge_planner_patch(verbose=True)
