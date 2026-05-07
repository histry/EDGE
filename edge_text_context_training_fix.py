"""Training-stage fix for Text/Pose Context RAG.

Drop this file into the EDGE project root.

Problem fixed:
- text_context_rag_model_patch.py adds:
    text_context_pose_projection
    text_context_pose_encoder
    text_context_text_projection
    text_context_type_embed
    text_context_gate
- But train_stage=adapter in EDGE._apply_stage_freezing freezes the whole model
  first and only unfreezes trajectory/energy/RAG-summary modules.
- Therefore Text/Pose Context RAG can be instantiated but remain frozen.

This patch keeps the original freezing policy and additionally unfreezes the
Text/Pose Context RAG modules when train_stage="adapter".

It is intentionally idempotent and does not change checkpoints/state_dicts.
"""

from __future__ import annotations

import os
import torch


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _set_trainable(obj, enabled: bool) -> int:
    """Set requires_grad for a module/parameter and return number of params touched."""
    if obj is None:
        return 0

    if isinstance(obj, torch.nn.Parameter):
        obj.requires_grad = bool(enabled)
        return int(obj.numel())

    if hasattr(obj, "parameters"):
        count = 0
        for p in obj.parameters():
            p.requires_grad = bool(enabled)
            count += int(p.numel())
        return count

    return 0


def _unfreeze_text_context_modules(model) -> int:
    names = [
        "text_context_pose_projection",
        "text_context_pose_encoder",
        "text_context_text_projection",
        "text_context_type_embed",
        "text_context_gate",
    ]

    touched = 0
    for name in names:
        touched += _set_trainable(getattr(model, name, None), True)

    return touched


def _text_context_param_summary(model) -> str:
    names = [
        "text_context_pose_projection",
        "text_context_pose_encoder",
        "text_context_text_projection",
        "text_context_type_embed",
        "text_context_gate",
    ]

    rows = []
    total = 0
    trainable = 0

    for name in names:
        obj = getattr(model, name, None)
        if obj is None:
            continue

        if isinstance(obj, torch.nn.Parameter):
            n_total = int(obj.numel())
            n_train = int(obj.numel()) if obj.requires_grad else 0
        else:
            params = list(obj.parameters()) if hasattr(obj, "parameters") else []
            n_total = sum(int(p.numel()) for p in params)
            n_train = sum(int(p.numel()) for p in params if p.requires_grad)

        total += n_total
        trainable += n_train
        rows.append(f"{name}: {n_train}/{n_total}")

    if not rows:
        return "no text_context_* modules found"

    return "; ".join(rows) + f"; total={trainable}/{total}"


def install_edge_text_context_training_fix(EDGEClass=None, verbose: bool = True) -> bool:
    """Patch EDGE._apply_stage_freezing so adapter stage trains text_context_* modules."""
    if EDGEClass is None:
        try:
            from EDGE import EDGE as EDGEClass  # type: ignore
        except Exception as exc:
            if verbose:
                print(f"⚠️ Text Context training fix skipped: cannot import EDGE: {exc}")
            return False

    if getattr(EDGEClass, "_edge_text_context_training_fix_installed", False):
        return True

    original_apply_stage_freezing = EDGEClass._apply_stage_freezing

    def patched_apply_stage_freezing(self, model, train_stage, adapter_train_decoder=False):
        out = original_apply_stage_freezing(
            self,
            model,
            train_stage,
            adapter_train_decoder=adapter_train_decoder,
        )

        enabled = bool(getattr(model, "enable_text_context_rag", False)) or _env_bool(
            "EDGE_ENABLE_TEXT_CONTEXT_RAG",
            False,
        )

        if str(train_stage) == "adapter" and enabled:
            touched = _unfreeze_text_context_modules(model)
            if verbose:
                print(
                    "🧩 train_stage=adapter: Text/Pose Context RAG trainable fix applied; "
                    f"params_touched={touched}; {_text_context_param_summary(model)}"
                )

        return out

    EDGEClass._apply_stage_freezing = patched_apply_stage_freezing
    EDGEClass._edge_original_apply_stage_freezing = original_apply_stage_freezing
    EDGEClass._edge_text_context_training_fix_installed = True

    if verbose:
        print("✅ Installed Text/Pose Context RAG training fix for adapter-stage freezing.")
    return True


def install():
    return install_edge_text_context_training_fix(verbose=True)
