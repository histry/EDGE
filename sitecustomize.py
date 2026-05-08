"""Repository-wide runtime patches for EDGE.

This replacement centralizes patch installation through edge_experiment_guard.
It remains fail-soft by default for interactive development, but formal scripts
can set EDGE_STRICT_RUNTIME_PATCHES=1 to turn missing required patches into hard
errors.
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
    # Keep sitecustomize fail-soft; formal entrypoints call the same installer
    # with strict checks and will raise when needed.
    print(f"⚠️ EDGE sitecustomize runtime patch installer failed: {exc}")
