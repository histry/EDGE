"""Repository-wide runtime patches for EDGE advanced trajectory / ChoreoRAG work.

This version preserves the existing patch order and adds:

1. trajectory_event_condition_patch
   Native trajectory-event condition branch:
     X/Z + speed + heading + curvature + support/turn/expression gates
     -> trajectory_projection residual adapter

2. trajectory_weak_energy_guidance_patch
   Optional weak tolerance-band energy guidance during sampling.
   This is OFF by default and should not be treated as the main method.
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
    ("trajectory_event_condition_patch", "install_trajectory_event_condition_patch"),
    ("gait_phase_adapter_patch", "install_gait_phase_adapter_patch"),
    ("turn_event_model_adapter_patch", "install_turn_event_model_adapter_patch"),
    ("trajectory_weak_energy_guidance_patch", "install_weak_trajectory_energy_guidance_patch"),
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
