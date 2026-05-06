"""Repository-wide runtime patches."""

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

try:
    from v9_rag_inference_patch import install_v9_rag_inference_patch
    install_v9_rag_inference_patch(verbose=True)
except Exception as exc:
    print(f"⚠️ V9 RAG inference patch not installed: {exc}")
