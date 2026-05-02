"""Repository-wide runtime patches.

Python imports sitecustomize automatically when this repository root is on
PYTHONPATH / current working directory.  Keep this file tiny and robust.
"""

try:
    from trajectory_native_control import install_native_trajectory_control_patch
    install_native_trajectory_control_patch(verbose=True)
except Exception as exc:
    print(f"⚠️ native trajectory runtime patch not installed: {exc}")

try:
    from edge_safety_patch import install_edge_safety_patch
    install_edge_safety_patch(verbose=True)
except Exception as exc:
    print(f"⚠️ EDGE safety patch not installed: {exc}")
