"""Auto-install EDGE native trajectory-control patch.

Direct replacement for: sitecustomize.py
"""

try:
    from trajectory_native_control import install_native_trajectory_control_patch
    install_native_trajectory_control_patch()
except Exception as exc:
    print(f"⚠️ native trajectory patch not installed: {exc}")
