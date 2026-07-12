#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.49.2 non-root position contract patch.

Chang-E BVH exposes 6 channels on every articulated joint. In the current
dataset, non-root XYZ position channels are approximately static calibration
values. Adding them again on top of hierarchy OFFSET distorts every bone chain.

Formal mode:
  V46_49_NONROOT_POSITION_MODE=ignore

Ablations:
  delta : preserve only position-channel change relative to the first frame
  raw   : old behaviour; add full non-root position values to OFFSET

Root XYZ position is always retained.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "tools" / "chang_e_edge_retarget.py"
MARKER = "# ===== V46.49.2 NONROOT POSITION CONTRACT ====="


OLD_BLOCK = '''    for j_idx, joint in enumerate(bvh.joints):
        local_t[:, j_idx] = joint.offset[None]
        if use_motion and joint.channels:
            for k, ch in enumerate(joint.channels):
                col = joint.channel_start + k
                low = ch.lower()
                if low.endswith("position"):
                    axis = "xyz".index(low[0])
                    local_t[:, j_idx, axis] += bvh.values[:, col]
            for k, ch in enumerate(joint.channels):
                if ch.lower().endswith("rotation"):
                    local_r[:, j_idx] = local_r[:, j_idx] @ _axis_matrix_batch(
                        ch[0], bvh.values[:, joint.channel_start + k]
                    )
'''

NEW_BLOCK = '''    # ===== V46.49.2 NONROOT POSITION CONTRACT =====
    # Chang-E files expose XYZ position channels on every articulated joint.
    # Dataset inspection shows that non-root position channels are approximately
    # static calibration values. They must not be added on top of hierarchy
    # OFFSET in the formal retargeting path, otherwise each bone chain is
    # translated twice and global keypoint fitting becomes physically impossible.
    #
    # Modes:
    #   ignore (formal/default): keep root XYZ only; non-root uses hierarchy OFFSET
    #   delta: preserve only non-root displacement relative to frame 0
    #   raw: old diagnostic behaviour; add full non-root position channels
    _position_mode = str(
        os.environ.get("V46_49_NONROOT_POSITION_MODE", "ignore")
    ).strip().lower()
    if _position_mode not in {"ignore", "delta", "raw"}:
        raise ValueError(
            "V46_49_NONROOT_POSITION_MODE must be ignore/delta/raw, "
            f"got {_position_mode!r}"
        )

    for j_idx, joint in enumerate(bvh.joints):
        local_t[:, j_idx] = joint.offset[None]
        if use_motion and joint.channels:
            _pos_values = {}
            for k, ch in enumerate(joint.channels):
                col = joint.channel_start + k
                low = ch.lower()
                if low.endswith("position"):
                    axis = "xyz".index(low[0])
                    _pos_values[axis] = bvh.values[:, col].astype(np.float32)

            # Root translation is the only absolute translation retained.
            if j_idx == 0:
                for axis, values in _pos_values.items():
                    local_t[:, j_idx, axis] += values
            elif _position_mode == "raw":
                for axis, values in _pos_values.items():
                    local_t[:, j_idx, axis] += values
            elif _position_mode == "delta":
                for axis, values in _pos_values.items():
                    local_t[:, j_idx, axis] += values - values[:1]
            # ignore: hierarchy OFFSET alone defines non-root translation.

            for k, ch in enumerate(joint.channels):
                if ch.lower().endswith("rotation"):
                    local_r[:, j_idx] = local_r[:, j_idx] @ _axis_matrix_batch(
                        ch[0], bvh.values[:, joint.channel_start + k]
                    )
    # ===== V46.49.2 NONROOT POSITION CONTRACT END =====
'''

REPORT_ANCHOR = '''        "source_joint_count": int(len(bvh.joints)),
'''

REPORT_REPLACEMENT = '''        "source_joint_count": int(len(bvh.joints)),
        "source_position_contract": {
            "version": "v46_49_2_nonroot_position_contract",
            "root_position": "retained",
            "nonroot_position_mode": str(
                os.environ.get("V46_49_NONROOT_POSITION_MODE", "ignore")
            ).strip().lower(),
            "hierarchy_offsets": "retained",
        },
'''


def main() -> int:
    if not TARGET.is_file():
        raise FileNotFoundError(TARGET)

    text = TARGET.read_text(encoding="utf-8")
    if MARKER in text:
        print("[SKIP] V46.49.2 non-root position patch already applied")
        return 0

    if OLD_BLOCK not in text:
        raise RuntimeError(
            "Cannot locate the V46.49.1 source_fk translation block. "
            "Verify tools/chang_e_edge_retarget.py is the current V46.49.1 file."
        )

    backup = TARGET.with_suffix(
        TARGET.suffix + f".v46_49_2_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(TARGET, backup)
    print("[BACKUP]", backup)

    text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)

    if REPORT_ANCHOR not in text:
        raise RuntimeError("Cannot locate retarget report insertion anchor")
    text = text.replace(REPORT_ANCHOR, REPORT_REPLACEMENT, 1)

    TARGET.write_text(text, encoding="utf-8")
    print("[DONE] V46.49.2 non-root position contract patched:", TARGET)
    print("[FORMAL] export V46_49_NONROOT_POSITION_MODE=ignore")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
