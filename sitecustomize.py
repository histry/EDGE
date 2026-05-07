"""Repository-wide runtime patches. All patches are idempotent and fail-soft."""
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
try:
    from edge_full_landing_patch import install_full_landing_patch
    install_full_landing_patch(verbose=True)
except Exception as exc:
    print(f"⚠️ EDGE full-landing patch not installed: {exc}")
try:
    from text_context_rag_model_patch import install_text_context_rag_model_patch
    install_text_context_rag_model_patch(verbose=True)
except Exception as exc:
    print(f"⚠️ Text/Pose Context RAG model patch not installed: {exc}")
try:
    from text_context_rag_io_patch import install_text_context_rag_io_patch
    install_text_context_rag_io_patch(verbose=True)
except Exception as exc:
    print(f"⚠️ Text/Pose Context RAG IO patch not installed: {exc}")
try:
    from text_bridge_planner_patch import install_text_bridge_planner_patch
    install_text_bridge_planner_patch(verbose=True)
except Exception as exc:
    print(f"⚠️ Text Bridge planner patch not installed: {exc}")
