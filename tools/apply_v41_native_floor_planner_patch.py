#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply V41 Native-Floor-Aware Planner patch to tools/v34_warp_aware_retrieval.py.

This patch is intentionally source-level and idempotent because the local EDGE
repo already contains V34/V40 research edits that may not exist in upstream.
It injects a piecewise exponential native-floor barrier into choose_events_v34,
without touching V40 motion post-processing or any V41 beat-support code.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

MARKER = "V41_NATIVE_FLOOR_AWARE_PLANNER_PATCH"

FUNCTIONS = r'''

# === V41_NATIVE_FLOOR_AWARE_PLANNER_PATCH: native-floor barrier operators ===
def compute_native_floor_barrier_potential(
    native_min_y: float,
    *,
    native_floor_y: float | None = None,
    native_penetration_m: float | None = None,
    tau_safe_m: float | None = None,
    tau_dead_m: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    cap: float | None = None,
) -> Tuple[float, bool, Dict[str, float]]:
    """Piecewise exponential barrier for native foot-floor pathology.

    The paper-level formula can be written with ``y_min``.  In this EDGE codebase
    the absolute root/foot Y coordinate is not globally zero-centred; logs show
    floor_y around -0.97m.  Therefore the operational variable is the *relative*
    native penetration depth:

        p = max(0, native_floor_y + margin - native_min_y)

    or the precomputed ``native_floor_penetration_m`` from V40/V40B audit.
    ``tau_safe_m`` and ``tau_dead_m`` are relative penetration thresholds.
    """
    safe = _env_float("V41_NATIVE_FLOOR_TAU_SAFE_M", 0.012) if tau_safe_m is None else float(tau_safe_m)
    dead = _env_float("V41_NATIVE_FLOOR_TAU_DEAD_M", 0.052) if tau_dead_m is None else float(tau_dead_m)
    a = _env_float("V41_NATIVE_FLOOR_ALPHA", 9.0) if alpha is None else float(alpha)
    b = _env_float("V41_NATIVE_FLOOR_BETA", 2.5) if beta is None else float(beta)
    penalty_cap = _env_float("V41_NATIVE_FLOOR_PENALTY_CAP", 18.0) if cap is None else float(cap)
    margin = _env_float("V41_NATIVE_FLOOR_MARGIN", 0.006)

    if native_penetration_m is None:
        if native_floor_y is None:
            penetration = max(0.0, -float(native_min_y))
        else:
            penetration = max(0.0, float(native_floor_y) + margin - float(native_min_y))
    else:
        penetration = max(0.0, float(native_penetration_m))

    if penetration <= safe:
        return 0.0, False, {
            "native_penetration_m": float(penetration),
            "tau_safe_m": float(safe),
            "tau_dead_m": float(dead),
            "violation_ratio": 0.0,
        }

    hard = bool(penetration >= dead)
    denom = max(dead - safe, 1e-8)
    violation_ratio = max(0.0, (penetration - safe) / denom)
    penalty = float(a * (violation_ratio ** b))
    if penalty_cap > 0:
        penalty = min(penalty, float(penalty_cap))
    return penalty, hard, {
        "native_penetration_m": float(penetration),
        "tau_safe_m": float(safe),
        "tau_dead_m": float(dead),
        "violation_ratio": float(violation_ratio),
    }


def _native_floor_measure_arrays(
    items: Sequence[Mapping[str, Any]],
    motions: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return native_min_y, native_floor_y and relative native penetration."""
    n = len(motions)
    native_min_y = np.zeros((n,), dtype=np.float32)
    native_floor_y = np.zeros((n,), dtype=np.float32)
    native_pen = np.zeros((n,), dtype=np.float32)
    source: List[str] = []
    q = float(np.clip(_env_float("V41_NATIVE_FLOOR_QUANTILE", _env_float("V40_NATIVE_FLOOR_QUANTILE", 0.05)), 0.005, 0.45))
    margin = _env_float("V41_NATIVE_FLOOR_MARGIN", _env_float("V40_NATIVE_FLOOR_MARGIN", 0.006))

    for i in range(n):
        item = items[i] if i < len(items) else {}
        try:
            if isinstance(item, Mapping) and "native_floor_penetration_m" in item:
                native_min_y[i] = float(item.get("native_min_foot_y", 0.0) or 0.0)
                native_floor_y[i] = float(item.get("native_floor_y", 0.0) or 0.0)
                native_pen[i] = max(0.0, float(item.get("native_floor_penetration_m", 0.0) or 0.0))
                source.append("json")
                continue
            x = np.asarray(motions[i], dtype=np.float32)
            if x.ndim == 3:
                x = x[0]
            if x.ndim != 2 or x.shape[1] < 151 or len(x) < 2:
                source.append("missing")
                continue
            t = x.shape[0]
            root = x[:, [4, 5, 6]]
            local_r = _rot6d_to_matrix_np(x[:, 7:151].reshape(t, 24, 6))
            joints = np.zeros((t, 24, 3), dtype=np.float32)
            global_r = np.zeros((t, 24, 3, 3), dtype=np.float32)
            joints[:, 0] = root
            global_r[:, 0] = local_r[:, 0]
            for j in range(1, 24):
                p = int(_FK_PARENTS[j])
                global_r[:, j] = np.matmul(global_r[:, p], local_r[:, j])
                joints[:, j] = joints[:, p] + np.matmul(
                    global_r[:, p], _FK_OFFSETS[j][None, :, None]
                )[..., 0]
            foot_y = joints[:, [7, 8, 10, 11], 1].reshape(-1)
            fy = float(np.quantile(foot_y, q))
            my = float(np.min(foot_y))
            native_min_y[i] = my
            native_floor_y[i] = fy
            native_pen[i] = max(0.0, fy + margin - my)
            source.append("computed")
        except Exception:
            source.append("failed")

    meta = {
        "quantile": float(q),
        "margin": float(margin),
        "source_counts": {str(k): int(source.count(k)) for k in sorted(set(source))},
        "max_native_penetration_m": float(np.max(native_pen)) if n else 0.0,
        "mean_native_penetration_m": float(np.mean(native_pen)) if n else 0.0,
        "p95_native_penetration_m": float(np.percentile(native_pen, 95)) if n else 0.0,
    }
    return native_min_y, native_floor_y, native_pen, meta


def _native_floor_barrier_arrays(
    items: Sequence[Mapping[str, Any]],
    motions: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """V41 non-convex native-floor barrier for retrieval-time beam expansion."""
    n = len(motions)
    zeros = np.zeros((n,), dtype=np.float32)
    hard = np.zeros((n,), dtype=bool)
    if not _enabled("V41_NATIVE_FLOOR_PLANNER", "0") or n == 0:
        return zeros, hard, zeros, zeros, zeros, {"enabled": False}

    min_y, floor_y, pen, measure_meta = _native_floor_measure_arrays(items, motions)
    penalty = np.zeros((n,), dtype=np.float32)
    abs_dead_raw = os.getenv("V41_NATIVE_FLOOR_ABSOLUTE_Y_DEAD", "")
    abs_dead = None
    try:
        abs_dead = float(abs_dead_raw) if abs_dead_raw.strip() else None
    except Exception:
        abs_dead = None

    for i in range(n):
        p, h, _ = compute_native_floor_barrier_potential(
            float(min_y[i]),
            native_floor_y=float(floor_y[i]),
            native_penetration_m=float(pen[i]),
        )
        penalty[i] = float(p)
        hard[i] = bool(h)
        if abs_dead is not None and float(min_y[i]) <= abs_dead:
            hard[i] = True

    rescue_penalty = np.zeros((n,), dtype=np.float32)
    dead_rescue_weight = _env_float("V41_NATIVE_FLOOR_DEAD_RESCUE_PENALTY", 25.0)
    rescue_penalty[hard] = float(dead_rescue_weight)

    meta = {
        "enabled": True,
        "version": "v41_native_floor_aware_planner_barrier",
        "tau_safe_m": _env_float("V41_NATIVE_FLOOR_TAU_SAFE_M", 0.012),
        "tau_dead_m": _env_float("V41_NATIVE_FLOOR_TAU_DEAD_M", 0.052),
        "alpha": _env_float("V41_NATIVE_FLOOR_ALPHA", 9.0),
        "beta": _env_float("V41_NATIVE_FLOOR_BETA", 2.5),
        "penalty_cap": _env_float("V41_NATIVE_FLOOR_PENALTY_CAP", 18.0),
        "hard_mask_enabled": _enabled("V41_NATIVE_FLOOR_HARD_MASK", "1"),
        "relax_on_empty": _enabled("V41_NATIVE_FLOOR_RELAX_ON_EMPTY", "1"),
        "dead_rescue_penalty": float(dead_rescue_weight),
        "absolute_y_dead": abs_dead,
        "hard_mask_count": int(np.sum(hard)),
        "soft_penalty_count": int(np.sum(penalty > 0.0)),
        "penalty_max": float(np.max(penalty)) if n else 0.0,
        "penalty_mean": float(np.mean(penalty)) if n else 0.0,
        **measure_meta,
    }
    return penalty.astype(np.float32), hard, rescue_penalty, min_y, pen, meta
# === end V41_NATIVE_FLOOR_AWARE_PLANNER_PATCH ===
'''

AFTER_NATIVE_LINE = '''    native_floor_penalty, native_floor_meta = _native_floor_penalty_arrays(items, motions)'''
AFTER_NATIVE_INSERT = '''    native_floor_penalty, native_floor_meta = _native_floor_penalty_arrays(items, motions)
    (
        native_floor_barrier_penalty,
        native_floor_barrier_hard_mask,
        native_floor_barrier_dead_rescue_penalty,
        native_floor_barrier_min_y,
        native_floor_barrier_penetration,
        native_floor_barrier_meta,
    ) = _native_floor_barrier_arrays(items, motions)'''

AFTER_BASE_NATIVE = '''        if len(native_floor_penalty) == len(base):
            base = base - native_floor_penalty'''
AFTER_BASE_BARRIER = '''        if len(native_floor_penalty) == len(base):
            base = base - native_floor_penalty
        if len(native_floor_barrier_penalty) == len(base):
            base = base - native_floor_barrier_penalty'''

BEFORE_NO_FEASIBLE = '''        if len(feasible_indices) == 0:
            natural_min = float(np.min(natural)) if len(natural) else float("nan")'''
FLOOR_MASK_BLOCK = '''        floor_barrier_relaxed = False
        if (
            _enabled("V41_NATIVE_FLOOR_PLANNER", "0")
            and _enabled("V41_NATIVE_FLOOR_HARD_MASK", "1")
            and len(native_floor_barrier_hard_mask) == len(broad_feasible)
        ):
            strict_floor_feasible = broad_feasible & (~native_floor_barrier_hard_mask)
            if np.any(strict_floor_feasible):
                broad_feasible = strict_floor_feasible
                feasible_indices = np.flatnonzero(broad_feasible)
            elif _enabled("V41_NATIVE_FLOOR_RELAX_ON_EMPTY", "1"):
                # Avoid V40B-style topological disconnection: if the only
                # warp-feasible graph branch is native-floor-dead, keep it as a
                # last-resort rescue but add a very large thermodynamic barrier.
                floor_barrier_relaxed = True
                if len(native_floor_barrier_dead_rescue_penalty) == len(base):
                    base = base - native_floor_barrier_dead_rescue_penalty
                feasible_indices = np.flatnonzero(broad_feasible)
                if _enabled("V41_NATIVE_FLOOR_VERBOSE", "1"):
                    print(
                        "[V41-FLOOR-RELAX] "
                        f"slot={slot} all warp-feasible candidates are native-floor-hard; "
                        f"using dead-rescue penalty="
                        f"{_env_float('V41_NATIVE_FLOOR_DEAD_RESCUE_PENALTY', 25.0):.3f}."
                    )
            else:
                broad_feasible = strict_floor_feasible
                feasible_indices = np.flatnonzero(broad_feasible)

        if len(feasible_indices) == 0:
            natural_min = float(np.min(natural)) if len(natural) else float("nan")'''

PART_OLD = '''                    "v40_native_floor_prior_meta": native_floor_meta,
                    "v40_native_floor_penalty": float(native_floor_penalty[idx]) if len(native_floor_penalty) == len(base) else 0.0,'''
PART_NEW = '''                    "v40_native_floor_prior_meta": native_floor_meta,
                    "v40_native_floor_penalty": float(native_floor_penalty[idx]) if len(native_floor_penalty) == len(base) else 0.0,
                    "v41_native_floor_barrier_meta": native_floor_barrier_meta,
                    "v41_native_floor_barrier_penalty": float(native_floor_barrier_penalty[idx]) if len(native_floor_barrier_penalty) == len(base) else 0.0,
                    "v41_native_floor_barrier_hard_mask": bool(native_floor_barrier_hard_mask[idx]) if len(native_floor_barrier_hard_mask) == len(base) else False,
                    "v41_native_floor_dead_rescue_penalty": float(native_floor_barrier_dead_rescue_penalty[idx]) if len(native_floor_barrier_dead_rescue_penalty) == len(base) else 0.0,
                    "v41_native_floor_min_y": float(native_floor_barrier_min_y[idx]) if len(native_floor_barrier_min_y) == len(base) else 0.0,
                    "v41_native_floor_penetration_m": float(native_floor_barrier_penetration[idx]) if len(native_floor_barrier_penetration) == len(base) else 0.0,
                    "v41_native_floor_relaxed_on_empty": bool(floor_barrier_relaxed),'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Cannot locate patch anchor: {label}")
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--target", default="tools/v34_warp_aware_retrieval.py")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    target = root / args.target
    if not target.is_file():
        raise FileNotFoundError(target)

    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"[SKIP] {target} already contains {MARKER}")
        return

    backup = target.with_suffix(target.suffix + f".before_v41_native_floor_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup.write_text(text, encoding="utf-8")
    print(f"[BACKUP] {backup}")

    text = text.replace("\ndef _build_rhythm_feature_arrays", FUNCTIONS + "\n\ndef _build_rhythm_feature_arrays", 1)
    text = replace_once(text, AFTER_NATIVE_LINE, AFTER_NATIVE_INSERT, "native floor arrays")
    text = replace_once(text, AFTER_BASE_NATIVE, AFTER_BASE_BARRIER, "base barrier penalty")
    text = replace_once(text, BEFORE_NO_FEASIBLE, FLOOR_MASK_BLOCK, "floor hard mask before no-feasible check")
    text = replace_once(text, PART_OLD, PART_NEW, "part metadata fields")

    target.write_text(text, encoding="utf-8")
    print(f"[OK] patched {target}")


if __name__ == "__main__":
    main()
