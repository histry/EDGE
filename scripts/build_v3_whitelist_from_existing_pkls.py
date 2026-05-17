#!/usr/bin/env python3
import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np


def load_motion(path: Path):
    try:
        data = pickle.load(open(path, "rb"))
    except Exception:
        return None, "bad_pkl"

    if not isinstance(data, dict):
        return None, "not_dict"

    for key in ["motion", "motion_151", "poses", "unit_motions_physical"]:
        if key in data:
            m = np.asarray(data[key], dtype=np.float32)
            if m.ndim == 2 and m.shape[-1] == 151:
                return m, key
            if m.ndim == 3 and m.shape[-1] == 151 and m.shape[0] == 1:
                return m[0], key

    return None, "no_151d"


def rot_indices(joints):
    idx = []
    for j in joints:
        idx.extend(range(7 + 6 * j, 7 + 6 * (j + 1)))
    return idx


UPPER = rot_indices([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
LOWER = rot_indices([1, 2, 4, 5, 7, 8, 10, 11])
BODY = list(range(7, 151))


def metrics(m, seq_len=45):
    m = m[:seq_len].astype(np.float32)
    if m.shape[0] < seq_len:
        return None

    root_xz = m[:, [4, 6]]
    root_range = float(np.linalg.norm(root_xz.max(axis=0) - root_xz.min(axis=0)))

    d = m[1:] - m[:-1]
    dd = d[1:] - d[:-1]

    jump = np.linalg.norm(d[:, BODY], axis=-1)
    jerk = np.linalg.norm(dd[:, BODY], axis=-1)

    upper_energy = float(np.sqrt(np.mean((m[1:, UPPER] - m[:-1, UPPER]) ** 2) + 1e-8))
    lower_energy = float(np.sqrt(np.mean((m[1:, LOWER] - m[:-1, LOWER]) ** 2) + 1e-8))
    body_energy = float(np.sqrt(np.mean((m[1:, BODY] - m[:-1, BODY]) ** 2) + 1e-8))

    jump_p95 = float(np.percentile(jump, 95))
    jerk_p95 = float(np.percentile(jerk, 95))
    top4_share = float(np.sort(jump)[-4:].sum() / max(jump.sum(), 1e-8))

    return {
        "root_range": root_range,
        "upper_energy": upper_energy,
        "lower_energy": lower_energy,
        "body_energy": body_energy,
        "jump_p95": jump_p95,
        "jerk_p95": jerk_p95,
        "top4_share": top4_share,
    }


def score(mm):
    return (
        2.0 * mm["root_range"]
        + 0.8 * mm["jump_p95"]
        + 0.5 * mm["jerk_p95"]
        + 3.0 * mm["top4_share"]
        - 8.0 * mm["upper_energy"]
        - 2.0 * mm["body_energy"]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/dunhuang_bvh/stationary_whitelist_v3_27units")
    ap.add_argument("--max_units", type=int, default=27)
    ap.add_argument("--min_units", type=int, default=20)
    ap.add_argument("--seq_len", type=int, default=45)
    ap.add_argument("--max_root_range", type=float, default=0.60)
    ap.add_argument("--min_upper_energy", type=float, default=0.003)
    ap.add_argument("--min_body_energy", type=float, default=0.002)
    ap.add_argument("--max_top4_share", type=float, default=0.55)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.is_dir():
        raise SystemExit(f"src not found: {src}")

    rows = []
    for p in sorted(src.glob("*.pkl")):
        m, key = load_motion(p)
        if m is None:
            print(f"skip {p.name}: {key}")
            continue
        mm = metrics(m, seq_len=args.seq_len)
        if mm is None:
            print(f"skip {p.name}: too short")
            continue

        keep = (
            mm["root_range"] <= args.max_root_range
            and mm["upper_energy"] >= args.min_upper_energy
            and mm["body_energy"] >= args.min_body_energy
            and mm["top4_share"] <= args.max_top4_share
        )
        rows.append({
            "path": str(p),
            "key": key,
            "keep": keep,
            "score": score(mm),
            **mm,
        })

    if not rows:
        raise SystemExit(f"No valid 151D pkl found under {src}")

    kept = sorted([r for r in rows if r["keep"]], key=lambda r: r["score"])

    if len(kept) < args.min_units:
        print(f"⚠️ strict filter kept only {len(kept)} units; relaxing by score from all {len(rows)} candidates.")
        kept = sorted(rows, key=lambda r: r["score"])

    selected = kept[: args.max_units]

    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.pkl"):
        old.unlink()

    manifest = []
    for i, r in enumerate(selected):
        src_path = Path(r["path"])
        dst = out / f"v3_unit_{i:03d}_{src_path.stem}.pkl"
        shutil.copy2(src_path, dst)
        r2 = dict(r)
        r2["dst"] = str(dst)
        manifest.append(r2)

    report = {
        "src": str(src),
        "out": str(out),
        "num_candidates": len(rows),
        "num_selected": len(selected),
        "selected": manifest,
    }
    (out / "v3_whitelist_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"✅ selected {len(selected)} units -> {out}")
    print(f"✅ report: {out / 'v3_whitelist_report.json'}")
    for r in manifest:
        print(
            f"{Path(r['dst']).name} | score={r['score']:.4f} "
            f"root={r['root_range']:.4f} upper={r['upper_energy']:.5f} "
            f"body={r['body_energy']:.5f} jump95={r['jump_p95']:.4f} "
            f"jerk95={r['jerk_p95']:.4f} top4={r['top4_share']:.3f}"
        )


if __name__ == "__main__":
    main()
