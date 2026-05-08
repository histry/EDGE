"""Install next-generation EDGE experimental patches.

This keeps the main training/generation scripts small and makes new objectives
env-gated.  It is safe to import even when all features are disabled.

Installed patches:
- Differentiable Contact Loss
- Beat-guided Sampling
- V11 explicit Cross-Attention RAG
"""
from __future__ import annotations


def install_nextgen_runtime_patches(verbose: bool = True) -> bool:
    ok = True
    specs = [
        ("differentiable_contact_loss_patch", "install_differentiable_contact_loss_patch"),
        ("beat_guided_sampling_patch", "install_beat_guided_sampling_patch"),
        ("v11_cross_attention_rag_patch", "install_v11_cross_attention_rag_patch"),
    ]
    for module_name, fn_name in specs:
        try:
            module = __import__(module_name, fromlist=[fn_name])
            fn = getattr(module, fn_name)
            try:
                result = fn(verbose=verbose)
            except TypeError:
                result = fn()
            ok = bool(result) and ok
        except Exception as exc:
            ok = False
            if verbose:
                print(f"⚠️ nextgen patch not installed: {module_name}.{fn_name}: {exc}")
    return ok


def install():
    return install_nextgen_runtime_patches(verbose=True)
