#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the V46.52 anatomy-gated retarget cache.

Preferred source order per stem:
1. official fitted SMPL .npz/.pkl parameters;
2. Chang-E .bvh through anatomy-constrained optimization.

Invalid sources are excluded from the cache when --allow_partial is used.  The
pipeline still enforces a minimum number of accepted sources.
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.chang_e_edge_retarget as legacy
from tools.v46_52_anatomy_contract import env_bool, env_int
from tools.v46_52_anatomy_retarget import load_official_smpl_motion, retarget_bvh_anatomy


def _discover(in_dir: Path) -> List[Path]:
    bvh = sorted(in_dir.rglob("*.bvh"))
    smpl = sorted([*in_dir.rglob("*.npz"), *in_dir.rglob("*.pkl"), *in_dir.rglob("*.pickle")])
    # Avoid selecting generated event/index files accidentally.
    smpl = [p for p in smpl if not any(t in p.name.lower() for t in ("event", "index", "feature", "cache", "split"))]
    prefer_smpl = env_bool("V46_52_PREFER_OFFICIAL_SMPL", True)
    grouped: Dict[str, List[Path]] = {}
    for p in [*bvh, *smpl]:
        grouped.setdefault(str(p.relative_to(in_dir).with_suffix("")), []).append(p)
    selected: List[Path] = []
    for _, paths in sorted(grouped.items()):
        paths.sort(key=lambda p: (0 if (prefer_smpl and p.suffix.lower() in {".npz", ".pkl", ".pickle"}) else 1, str(p)))
        selected.append(paths[0])
    return selected


def _report_valid(rep: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if str(rep.get("version", "")).startswith("v46_52") is False:
        reasons.append("not_v46_52")
    if not bool(rep.get("ok", False)):
        reasons.append("not_ok")
    if not bool(rep.get("anatomy_ok", False)):
        reasons.append("anatomy_not_ok")
    if not bool(rep.get("gravity_ok", False)):
        reasons.append("gravity_not_ok")
    if not bool(rep.get("fit_ok", False)):
        reasons.append("fit_not_ok")
    return not reasons, reasons


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="change")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _discover(in_dir)
    if not files:
        raise RuntimeError(f"No BVH/SMPL source files found under {in_dir}")

    cfg = legacy.RetargetConfig.from_env()
    if args.device:
        cfg.device = args.device
    allow_partial = bool(args.allow_partial or env_bool("V46_52_ALLOW_PARTIAL_RETARGET", True))
    min_ok = env_int("V46_52_MIN_OK_SOURCES", min(8, len(files)))

    reports: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    stale: List[Dict[str, Any]] = []

    for idx, src in enumerate(files, 1):
        rel = src.relative_to(in_dir)
        dst = (out_dir / rel).with_suffix(".npy")
        rep_path = dst.with_suffix(".retarget.json")
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[V46.52 RETARGET {idx}/{len(files)}] {src} -> {dst}", flush=True)
        try:
            if dst.exists() and rep_path.exists() and not args.overwrite:
                old = json.loads(rep_path.read_text(encoding="utf-8"))
                valid, reasons = _report_valid(old)
                if valid:
                    print("[SKIP] existing formal V46.52 anatomy cache", flush=True)
                    reports.append(old)
                    continue
                stale.append({"source": str(src), "reasons": reasons})
                print(f"[REBUILD STALE] {reasons}", flush=True)

            # Prefer official fitted SMPL parameters, but never discard a source
            # merely because a sidecar file uses an unsupported schema.  When an
            # exact-stem BVH exists, it is the scientifically safer fallback.
            candidates = [src]
            if src.suffix.lower() != ".bvh":
                bvh_fallback = src.with_suffix(".bvh")
                if bvh_fallback.is_file():
                    candidates.append(bvh_fallback)

            candidate_errors = []
            motion = None
            rep = None
            source_used = None
            for candidate in candidates:
                try:
                    if candidate.suffix.lower() == ".bvh":
                        motion, rep = retarget_bvh_anatomy(candidate, cfg)
                    else:
                        motion, rep = load_official_smpl_motion(
                            candidate,
                            target_fps=float(cfg.target_fps),
                        )
                    source_used = candidate
                    break
                except Exception as candidate_exc:
                    candidate_errors.append({
                        "source": str(candidate),
                        "error": str(candidate_exc),
                    })
                    motion = None
                    rep = None

            if motion is None or rep is None or source_used is None:
                raise RuntimeError(
                    "All V46.52 source representations failed: "
                    + json.dumps(candidate_errors, ensure_ascii=False)
                )

            rep = dict(rep)
            rep["output"] = str(dst)
            rep["source_relative"] = str(rel.with_suffix(source_used.suffix))
            rep["preferred_source"] = str(src)
            rep["source_used"] = str(source_used)
            rep["representation_fallbacks"] = candidate_errors
            rep["v46_52_cache_contract"] = {
                "schema": "v46_52_anatomy_retarget_cache",
                "requires_anatomy_ok": True,
                "requires_gravity_ok": True,
                "requires_fit_ok": True,
                "official_smpl_preferred": env_bool("V46_52_PREFER_OFFICIAL_SMPL", True),
            }
            valid, reasons = _report_valid(rep)
            if not valid:
                raise RuntimeError(f"Non-formal V46.52 report: {reasons}")
            np.save(dst, np.asarray(motion, dtype=np.float32))
            rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
            reports.append(rep)
        except Exception as exc:
            fail = {
                "source": str(src),
                "output": str(dst),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(fail)
            print(f"[REJECTED SOURCE] {src}: {exc}", flush=True)
            # Do not leave stale invalid files discoverable by the split builder.
            for p in (dst, rep_path):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            if not allow_partial:
                break

    all_ok = len(reports) >= min_ok and (allow_partial or not failures)
    summary = {
        "schema": "v46_52_anatomy_retarget_cache",
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "num_inputs": len(files),
        "num_ok": len(reports),
        "num_failed": len(failures),
        "minimum_ok_sources": int(min_ok),
        "allow_partial": bool(allow_partial),
        "all_ok": bool(all_ok),
        "source_formats": {
            "official_smpl": sum(str(r.get("source_format", "")).startswith("official_smpl") for r in reports),
            "bvh_optimized": sum("bvh" in str(r.get("version", "")) for r in reports),
        },
        "anatomy_quality_min": min(
            [float(r.get("anatomy", {}).get("anatomy_quality", 0.0)) for r in reports],
            default=0.0,
        ),
        "fit_rmse_p95_max_m": max(
            [float(r.get("fit", {}).get("fit_rmse_p95_m", 0.0)) for r in reports],
            default=0.0,
        ),
        "stale_rebuilt": stale,
        "reports": reports,
        "failures": failures,
    }
    # Keep the legacy filename because the V46.51 split script checks it.
    for name in ("v46_50_retarget_cache_report.json", "v46_52_retarget_cache_report.json"):
        (out_dir / name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("num_inputs", "num_ok", "num_failed", "minimum_ok_sources", "all_ok")}, indent=2))
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
