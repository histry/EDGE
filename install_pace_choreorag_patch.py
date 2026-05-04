#!/usr/bin/env python3
"""Install PACE-ChoreoRAG hooks into an existing EDGE checkout.

Usage from EDGE repo root:
    python install_pace_choreorag_patch.py

What it does:
  1. Copies/keeps pace_choreorag_trajectory.py and choreorag_unit_prior.py.
  2. Patches generate_controlled.py to:
       - import PACE trajectory helpers;
       - apply beat-aware progress inside build_control_trajectory();
       - apply scale-cap / elastic anchors before trajectory normalization;
       - apply retrieved-unit priors inside build_constraint().
  3. Optionally patches auto_keyframe_planner.py with root/spatial hard-lock envs
     when a compatible insertion point is found.

The patch is idempotent and writes *.pace_bak once before modifying files.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


GENERATE_IMPORT = '''\ntry:\n    from pace_choreorag_trajectory import (\n        apply_pace_choreorag_to_trajectory,\n        build_pace_progress,\n    )\nexcept Exception:\n    apply_pace_choreorag_to_trajectory = None\n    build_pace_progress = None\n\ntry:\n    from choreorag_unit_prior import (\n        apply_unit_priors_from_specs,\n        infer_unit_specs_from_mid_paths,\n    )\nexcept Exception:\n    apply_unit_priors_from_specs = None\n    infer_unit_specs_from_mid_paths = None\n'''

PACE_PROGRESS_SNIPPET = '''\n        if build_pace_progress is not None:\n            try:\n                progress = build_pace_progress(\n                    audio_feature=audio_feature,\n                    num_frames=num_frames,\n                    base_progress=progress,\n                )\n            except Exception as exc:\n                print(f"⚠️ PACE beat-aware progress skipped: {exc}")\n'''

PACE_APPLY_SNIPPET = '''\n\n    # PACE-ChoreoRAG: root-speed scale cap + optional elastic sparse anchors.\n    # Disabled unless EDGE_TRAJ_AUTO_SCALE=1 or EDGE_TRAJ_ELASTIC_ANCHOR=1.\n    if apply_pace_choreorag_to_trajectory is not None:\n        try:\n            traj_physical = apply_pace_choreorag_to_trajectory(\n                traj_physical=traj_physical,\n                audio_feature=audio_feature,\n            )\n        except Exception as exc:\n            print(f"⚠️ PACE trajectory skipped: {exc}")\n'''

UNIT_PRIOR_SNIPPET = '''\n\n    # PACE-ChoreoRAG Phase 4: retrieved 45-frame unit -> weak temporal prior.\n    # Disabled unless EDGE_UNIT_SOFT_PRIOR=1.  It infers sibling files\n    # like xxx_01_f117.npy -> xxx_01_f117_unit.npy.\n    if apply_unit_priors_from_specs is not None and infer_unit_specs_from_mid_paths is not None:\n        try:\n            all_mid_paths = parse_list(args.mid_poses)\n            all_mid_frames = parse_mid_frames(args.mid_pose_frames, len(all_mid_paths), num_frames)\n            unit_specs = infer_unit_specs_from_mid_paths(all_mid_paths, all_mid_frames)\n            value, mask = apply_unit_priors_from_specs(value, mask, unit_specs)\n            if unit_specs:\n                print(f"✅ ChoreoRAG retrieved-unit specs found: {unit_specs}")\n        except Exception as exc:\n            import os as _os\n            if str(_os.environ.get("EDGE_UNIT_SOFT_PRIOR", "0")).lower() in {"1", "true", "yes", "y", "on"}:\n                print(f"⚠️ failed to apply ChoreoRAG retrieved-unit soft prior: {exc}")\n'''

PLANNER_LOCK_SNIPPET = '''\n        # PACE-ChoreoRAG optional trajectory-span hard lock.\n        # Use only for large-span segments after validating trajectory scale.\n        min_root_speed = _env_float("EDGE_UNIT_MIN_ROOT_SPEED_NORM", -1.0)\n        min_spatial_range = _env_float("EDGE_UNIT_MIN_SPATIAL_RANGE_NORM", -1.0)\n        if min_root_speed >= 0.0 and float(cand.get("root_speed_norm", 0.0) or 0.0) < min_root_speed:\n            rejected += 1\n            continue\n        if min_spatial_range >= 0.0 and float(cand.get("spatial_range_norm", 0.0) or 0.0) < min_spatial_range:\n            rejected += 1\n            continue\n'''


def backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".pace_bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"backup: {bak}")


def patch_generate(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    original = text
    backup(path)

    if "pace_choreorag_trajectory" not in text:
        marker = "from EDGE import EDGE\n"
        if marker in text:
            text = text.replace(marker, marker + GENERATE_IMPORT + "\n", 1)
        else:
            text = GENERATE_IMPORT + "\n" + text
        print("patched generate_controlled.py imports")

    if "build_pace_progress(" not in text:
        pattern = re.compile(
            r"(\s+progress\s*=\s*build_trajectory_progress\(\n"
            r"\s+audio_feature=audio_feature,\n"
            r"\s+use_audio_timing=not uniform_timing,\n"
            r"\s+onset_index=onset_index,\n"
            r"\s+\)\n)"
        )
        text, n = pattern.subn(r"\1" + PACE_PROGRESS_SNIPPET, text, count=1)
        if n:
            print("patched beat-aware progress hook")
        else:
            print("warning: could not find build_trajectory_progress block; progress hook not inserted")

    if "PACE-ChoreoRAG: root-speed scale cap" not in text:
        marker = "    if not keep_absolute:\n        traj_physical = traj_physical - traj_physical[0:1]\n"
        if marker in text:
            text = text.replace(marker, marker + PACE_APPLY_SNIPPET, 1)
            print("patched PACE trajectory apply hook")
        else:
            print("warning: could not find keep_absolute block; trajectory apply hook not inserted")

    if "PACE-ChoreoRAG Phase 4" not in text:
        marker = "    return {\n        \"value\": torch.from_numpy(value[None]).to(device=device, dtype=torch.float32),\n        \"mask\": torch.from_numpy(mask[None]).to(device=device, dtype=torch.float32),\n    }\n"
        if marker in text:
            text = text.replace(marker, UNIT_PRIOR_SNIPPET + "\n" + marker, 1)
            print("patched retrieved-unit prior hook")
        else:
            print("warning: could not find build_constraint return block; unit-prior hook not inserted")

    if text != original:
        path.write_text(text, encoding="utf-8")
    else:
        print("generate_controlled.py already patched; no changes")


def patch_planner(path: Path) -> None:
    if not path.exists():
        print("auto_keyframe_planner.py not found; skip")
        return
    text = path.read_text(encoding="utf-8")
    original = text
    backup(path)
    if "EDGE_UNIT_MIN_ROOT_SPEED_NORM" in text:
        print("planner root/spatial hard-lock already present; skip")
        return

    # Insert after min-expressiveness gate if present.
    patterns = [
        r"(\n\s+if\s+min_expr\s*>=\s*0\.0\s+and\s+expr\s*<\s*min_expr:\n\s+rejected\s*\+=\s*1\n\s+continue\n)",
        r"(\n\s+if\s+min_expr\s*>=\s*0\.0.*?continue\n)",
    ]
    for pat in patterns:
        text2, n = re.subn(pat, r"\1" + PLANNER_LOCK_SNIPPET, text, count=1, flags=re.DOTALL)
        if n:
            path.write_text(text2, encoding="utf-8")
            print("patched planner optional root_speed/spatial_range hard-lock")
            return
    print("warning: could not find min-expressiveness gate; planner hard-lock not inserted")
    if text != original:
        path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".", help="EDGE repo root")
    parser.add_argument("--skip_planner", action="store_true")
    args = parser.parse_args()
    root = Path(args.repo).resolve()
    patch_generate(root / "generate_controlled.py")
    if not args.skip_planner:
        patch_planner(root / "auto_keyframe_planner.py")
    print("done. Now run: python -m py_compile generate_controlled.py choreorag_unit_prior.py pace_choreorag_trajectory.py")


if __name__ == "__main__":
    main()
