#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.49.4 absolute root-orientation contract patch."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "tools" / "chang_e_edge_retarget.py"
MARKER = "# ===== V46.49.4 ABSOLUTE ROOT ORIENTATION CONTRACT ====="

FIELD_OLD = """    root_velocity_weight: float = 1.5
    gradient_clip: float = 2.0
"""
FIELD_NEW = """    root_velocity_weight: float = 1.5
    # ===== V46.49.4 ABSOLUTE ROOT ORIENTATION CONTRACT =====
    root_orientation_lock: bool = True
    # ===== V46.49.4 ABSOLUTE ROOT ORIENTATION CONTRACT END =====
    gradient_clip: float = 2.0
"""

ENV_OLD = """            root_velocity_weight=f("V46_49_RETARGET_ROOT_VEL_W", 1.5),
            gradient_clip=f("V46_49_RETARGET_GRAD_CLIP", 2.0),
"""
ENV_NEW = """            root_velocity_weight=f("V46_49_RETARGET_ROOT_VEL_W", 1.5),
            root_orientation_lock=b("V46_49_ROOT_ORIENTATION_LOCK", True),
            gradient_clip=f("V46_49_RETARGET_GRAD_CLIP", 2.0),
"""

INIT_OLD = """    init_rot = torch.tensor(init_rot6d, dtype=torch.float32, device=device)
    source_root = target[:, 0].detach()
"""
INIT_NEW = """    init_rot = torch.tensor(init_rot6d, dtype=torch.float32, device=device)
    reference_root_rot6d = _project6d_torch(init_rot[:, 0]).detach()
    source_root = target[:, 0].detach()
"""

LOOP_OLD = """        rp = _project6d_torch(rot)
        joints = _fk_target_torch(root, rp)
"""
LOOP_NEW = """        rp = _project6d_torch(rot)
        if cfg.root_orientation_lock:
            rp = torch.cat(
                [reference_root_rot6d[:, None, :], rp[:, 1:]],
                dim=1,
            )
        joints = _fk_target_torch(root, rp)
"""

FINAL_OLD = """    with torch.no_grad():
        final_rot = _project6d_torch(rot).cpu().numpy().astype(np.float32)
        final_root = root.cpu().numpy().astype(np.float32)
"""
FINAL_NEW = """    with torch.no_grad():
        final_rot_t = _project6d_torch(rot)
        if cfg.root_orientation_lock:
            final_rot_t = torch.cat(
                [reference_root_rot6d[:, None, :], final_rot_t[:, 1:]],
                dim=1,
            )
        final_rot = final_rot_t.cpu().numpy().astype(np.float32)
        final_root = root.cpu().numpy().astype(np.float32)
"""

REPORT_OLD = """        "fit_rmse_p95_m": float(np.percentile(fit_arr, 95)) if fit_arr.size else 0.0,
        "chunk_reports": chunk_reports,
"""
REPORT_NEW = """        "fit_rmse_p95_m": float(np.percentile(fit_arr, 95)) if fit_arr.size else 0.0,
        "root_orientation_contract": {
            "version": "v46_49_4_absolute_root_orientation_contract",
            "mode": "absolute_reference_lock"
            if cfg.root_orientation_lock
            else "unconstrained_ablation",
            "root_translation": "optimized",
            "root_orientation": "fixed_to_corrected_source_body_frame"
            if cfg.root_orientation_lock
            else "optimized",
            "local_joints_1_to_23": "optimized",
        },
        "chunk_reports": chunk_reports,
"""


def replace_once(text: str, old: str, new: str, name: str) -> str:
    if old not in text:
        raise RuntimeError(f"Cannot locate patch anchor: {name}")
    return text.replace(old, new, 1)


def main() -> int:
    if not TARGET.is_file():
        raise FileNotFoundError(TARGET)

    text = TARGET.read_text(encoding="utf-8")
    if MARKER in text:
        print("[SKIP] V46.49.4 root-orientation patch already applied")
        return 0

    backup = TARGET.with_suffix(
        TARGET.suffix
        + f".v46_49_4_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(TARGET, backup)
    print("[BACKUP]", backup)

    text = replace_once(text, FIELD_OLD, FIELD_NEW, "RetargetConfig field")
    text = replace_once(text, ENV_OLD, ENV_NEW, "from_env")
    text = replace_once(text, INIT_OLD, INIT_NEW, "_fit_chunk init")
    text = replace_once(text, LOOP_OLD, LOOP_NEW, "_fit_chunk loop")
    text = replace_once(text, FINAL_OLD, FINAL_NEW, "_fit_chunk final")
    text = replace_once(text, REPORT_OLD, REPORT_NEW, "fit report")

    TARGET.write_text(text, encoding="utf-8")
    print("[DONE] V46.49.4 absolute root-orientation contract patched")
    print("[FORMAL] export V46_49_ROOT_ORIENTATION_LOCK=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
