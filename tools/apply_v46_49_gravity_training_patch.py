#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Idempotently add V46.49 gravity losses to the latest V46.47 trainer.

This patch intentionally changes only two scientifically relevant loss sites:
V45 refiner reconstruction and V46 x0 reconstruction.  It does not replace the
large v46_motionrag_diff.py and therefore preserves later local fixes.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "tools" / "v46_motionrag_diff.py"
MARKER = "# ===== V46.49 GRAVITY TRAINING CONTRACT ====="


def main() -> int:
    if not TARGET.is_file():
        raise FileNotFoundError(TARGET)
    text = TARGET.read_text(encoding="utf-8")
    if MARKER in text:
        print("[SKIP] V46.49 gravity training patch already applied")
        return 0

    backup = TARGET.with_suffix(
        TARGET.suffix + f".v46_49_gravity_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(TARGET, backup)
    print("[BACKUP]", backup)

    import_anchor = "import numpy as np\n"
    import_code = (
        "import numpy as np\n"
        f"{MARKER}\n"
        "from tools.v46_49_gravity_contract import gravity_loss_torch\n"
    )
    if import_anchor not in text:
        raise RuntimeError("Cannot locate numpy import")
    text = text.replace(import_anchor, import_code, 1)

    old_v45 = (
        "        rec = F.smooth_l1_loss(pred, clean_t)\n"
        "        smooth = F.smooth_l1_loss(pred[:, 1:] - pred[:, :-1], clean_t[:, 1:] - clean_t[:, :-1])\n"
        "        loss = rec + 0.25 * smooth\n"
    )
    new_v45 = (
        "        rec = F.smooth_l1_loss(pred, clean_t)\n"
        "        smooth = F.smooth_l1_loss(pred[:, 1:] - pred[:, :-1], clean_t[:, 1:] - clean_t[:, :-1])\n"
        "        gravity_terms = gravity_loss_torch(pred, reference=clean_t)\n"
        "        gravity_w = float(os.environ.get(\"V46_49_GRAVITY_LOSS_W\", \"2.0\"))\n"
        "        loss = rec + 0.25 * smooth + gravity_w * gravity_terms[\"total\"]\n"
    )
    if old_v45 not in text:
        raise RuntimeError("Cannot locate V45 loss block; source version differs")
    text = text.replace(old_v45, new_v45, 1)

    old_v46 = (
        "        loss_vel = F.smooth_l1_loss(x0_hat[:, 1:] - x0_hat[:, :-1], x0[:, 1:] - x0[:, :-1])\n"
        "        loss = loss_noise + 0.10 * loss_vel\n"
    )
    new_v46 = (
        "        loss_vel = F.smooth_l1_loss(x0_hat[:, 1:] - x0_hat[:, :-1], x0[:, 1:] - x0[:, :-1])\n"
        "        gravity_terms = gravity_loss_torch(x0_hat, reference=x0)\n"
        "        gravity_w = float(os.environ.get(\"V46_49_DIFFUSION_GRAVITY_LOSS_W\", "
        "os.environ.get(\"V46_49_GRAVITY_LOSS_W\", \"2.0\")))\n"
        "        loss = loss_noise + 0.10 * loss_vel + gravity_w * gravity_terms[\"total\"]\n"
    )
    if old_v46 not in text:
        raise RuntimeError("Cannot locate V46 loss block; source version differs")
    text = text.replace(old_v46, new_v46, 1)

    old_print_v45 = (
        '            print(f"[V45 refiner] step={step} loss={loss.item():.6f} rec={rec.item():.6f}")'
    )
    new_print_v45 = (
        '            print(f"[V45 refiner] step={step} loss={loss.item():.6f} '
        'rec={rec.item():.6f} gravity={gravity_terms[\'total\'].item():.6f}")'
    )
    if old_print_v45 in text:
        text = text.replace(old_print_v45, new_print_v45, 1)

    old_print_v46 = (
        '            print(f"[V46 diffusion] step={step} loss={loss.item():.6f} '
        'noise={loss_noise.item():.6f}")'
    )
    new_print_v46 = (
        '            print(f"[V46 diffusion] step={step} loss={loss.item():.6f} '
        'noise={loss_noise.item():.6f} gravity={gravity_terms[\'total\'].item():.6f}")'
    )
    if old_print_v46 in text:
        text = text.replace(old_print_v46, new_print_v46, 1)

    TARGET.write_text(text, encoding="utf-8")
    print("[DONE] patched", TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
