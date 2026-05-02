"""Auto-install EDGE native trajectory-control patch.

Place this file in the EDGE project root. Python imports ``sitecustomize``
automatically when running scripts from that directory, so no existing training
or generation entrypoint needs to be edited.
"""
try:
    from trajectory_native_control import install_native_trajectory_control_patch
    install_native_trajectory_control_patch()
except Exception as exc:  # keep normal scripts runnable even if patch fails
    print(f"⚠️ native trajectory patch not installed: {exc}")
