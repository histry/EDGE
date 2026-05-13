"""Repository-wide runtime patches for EDGE advanced trajectory / ChoreoRAG work.

This version preserves the existing patch order and adds the optional
turn-aware event model adapter patch.  All behavior is environment-gated, so
clean baselines remain clean unless the corresponding EDGE_* flags are enabled.
"""
from __future__ import annotations

try:
    from edge_experiment_guard import env_bool, install_runtime_patches

    install_runtime_patches(
        strict=env_bool("EDGE_STRICT_RUNTIME_PATCHES", False),
        profile=None,
        verbose=True,
    )
except Exception as exc:
    print(f"⚠️ EDGE sitecustomize base patch installer failed: {exc}")

# Advanced patches are idempotent and fail-soft here.  Formal train.py also
# installs them after EDGE imports, so a startup-time miss is not fatal.
for module_name, fn_name in [
    ("gait_phase_dataset_patch", "install_gait_phase_dataset_patch"),
    ("trajectory_enhancement_patch", "install_trajectory_enhancement_patch"),
    ("gait_phase_adapter_patch", "install_gait_phase_adapter_patch"),
    ("turn_event_model_adapter_patch", "install_turn_event_model_adapter_patch"),
]:
    try:
        module = __import__(module_name, fromlist=[fn_name])
        fn = getattr(module, fn_name)
        try:
            fn(verbose=True)
        except TypeError:
            fn()
    except Exception as exc:
        print(f"⚠️ EDGE advanced sitecustomize patch {module_name}.{fn_name} skipped: {exc}")
