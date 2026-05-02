"""Safety patch for EDGE training/reporting risks.

This file does not replace the large EDGE.py implementation.  It patches EDGE
at import time to make important experiment assumptions explicit:

1) requested vs effective MMR loss is recorded and printed;
2) stage1/stage2 cannot silently freeze randomly initialized audio layers when
   checkpoint audio_dim is mismatched, unless explicitly allowed;
3) training logs expose the effective MMR state for reporting.

Install via sitecustomize.py:
    from edge_safety_patch import install_edge_safety_patch
    install_edge_safety_patch()
"""
from __future__ import annotations

import os
from typing import Any


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_arg(args, index, kwargs, name, default=None):
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return default


def _has_audio_checkpoint_mismatch(adapt_report: dict) -> bool:
    audio_markers = (
        "cond_projection",
        "cond_encoder",
        "non_attn_cond_projection",
        "null_cond_embed",
        "null_cond_hidden",
    )

    def is_audio_key(key):
        key = str(key)
        if key.startswith("module."):
            key = key[len("module."):]
        return any(marker in key for marker in audio_markers)

    for item in adapt_report.get("skipped_shape", []):
        key = item[0] if isinstance(item, (list, tuple)) and item else str(item)
        if is_audio_key(key):
            return True

    for key in adapt_report.get("kept_other", []):
        if is_audio_key(key):
            return True

    return False


def install_edge_safety_patch(verbose: bool = True):
    try:
        import EDGE as edge_module
        EDGEClass = edge_module.EDGE
    except Exception as exc:
        if verbose:
            print(f"⚠️ EDGE safety patch skipped: {exc}")
        return False

    if getattr(EDGEClass, "_edge_safety_patch_installed", False):
        return True

    original_init = EDGEClass.__init__

    def patched_init(self, *args, **kwargs):
        audio_pairing_mode = _get_arg(args, 17, kwargs, "audio_pairing_mode", "proxy")
        requested_mmr_loss_weight = float(_get_arg(args, 18, kwargs, "mmr_loss_weight", 0.0) or 0.0)
        effective_mmr_loss_weight = (
            requested_mmr_loss_weight
            if str(audio_pairing_mode) == "paired"
            else 0.0
        )

        self.requested_mmr_loss_weight = requested_mmr_loss_weight
        self.effective_mmr_loss_weight = effective_mmr_loss_weight
        self.audio_pairing_mode_for_reporting = str(audio_pairing_mode)

        original_init(self, *args, **kwargs)

        # Restore explicit report fields after original __init__.
        self.requested_mmr_loss_weight = requested_mmr_loss_weight
        self.effective_mmr_loss_weight = effective_mmr_loss_weight
        self.audio_pairing_mode_for_reporting = str(audio_pairing_mode)

        if getattr(self, "accelerator", None) is None or self.accelerator.is_main_process:
            if requested_mmr_loss_weight > 0 and effective_mmr_loss_weight == 0:
                print(
                    "📌 Effective MMR Loss: requested="
                    f"{requested_mmr_loss_weight}, effective=0.0 because "
                    f"audio_pairing_mode={audio_pairing_mode}. "
                    "Do NOT report this run as MMR-loss supervised."
                )
            else:
                print(
                    "📌 Effective MMR Loss: requested="
                    f"{requested_mmr_loss_weight}, effective={effective_mmr_loss_weight}, "
                    f"audio_pairing_mode={audio_pairing_mode}."
                )

    EDGEClass.__init__ = patched_init

    if hasattr(EDGEClass, "_check_audio_checkpoint_compatibility"):
        original_check_audio = EDGEClass._check_audio_checkpoint_compatibility

        def patched_check_audio_checkpoint_compatibility(
            adapt_report,
            strict_audio_checkpoint=False,
            train_stage="full",
        ):
            original_check_audio(
                adapt_report,
                strict_audio_checkpoint=strict_audio_checkpoint,
                train_stage=train_stage,
            )

            mismatch = _has_audio_checkpoint_mismatch(adapt_report)
            if not mismatch:
                return

            if str(train_stage) in {"stage1", "stage2"}:
                allow = _env_bool("EDGE_ALLOW_FROZEN_RANDOM_AUDIO", False)
                if not allow:
                    raise RuntimeError(
                        "\n❌ Audio checkpoint mismatch detected while using "
                        f"train_stage={train_stage}.\n"
                        "This stage freezes audio condition modules, so mismatched "
                        "audio weights would remain randomly initialized.\n"
                        "Use one of:\n"
                        "  1) matching audio_dim checkpoint;\n"
                        "  2) --train_stage full;\n"
                        "  3) --strict_audio_checkpoint for explicit failure;\n"
                        "  4) EDGE_ALLOW_FROZEN_RANDOM_AUDIO=1 only for ablation, "
                        "and report that audio is not a learned prior.\n"
                    )

        EDGEClass._check_audio_checkpoint_compatibility = staticmethod(
            patched_check_audio_checkpoint_compatibility
        )

    EDGEClass._edge_safety_patch_installed = True
    if verbose:
        print(
            "✅ Installed EDGE safety patch: effective MMR reporting + "
            "frozen random audio guard."
        )
    return True


def install():
    return install_edge_safety_patch()
