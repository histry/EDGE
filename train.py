# Runtime patches MUST be installed before constructing EDGE / DanceDecoder.
#
# Replacement train.py with explicit experiment profiles.
#
# Default behavior:
#   EDGE_TRAIN_PROFILE=legacy       -> preserves the previous patch-heavy training path.
#
# V3 clean unit reconstruction:
#   EDGE_TRAIN_PROFILE=v3_unit_recon
#   EDGE_V3_UNIT_RECON=1
# This profile intentionally avoids V2E/V2F freeze/progress/burst patches,
# trajectory-event wrappers, weak energy guidance, Text/Pose RAG, gait adapters,
# and beat guidance. It installs only safe/common guards plus
# unit_reconstruction_patch.py.

import os


_TRUE = {"1", "true", "yes", "y", "on"}


def _truthy(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in _TRUE


def _profile():
    p = os.environ.get("EDGE_TRAIN_PROFILE", "").strip().lower()
    if not p and _truthy("EDGE_V3_UNIT_RECON", False):
        p = "v3_unit_recon"
    return p or "legacy"


def _call_install(module_name, fn_name, verbose=True):
    try:
        module = __import__(module_name, fromlist=[fn_name])
        install_fn = getattr(module, fn_name)
        try:
            install_fn(verbose=verbose)
        except TypeError:
            install_fn()
        return True
    except Exception as exc:
        print(f"⚠️ {module_name}.{fn_name} not installed: {exc}")
        return False


def _install_runtime_patches():
    profile = _profile()
    print(f"🧭 EDGE train profile: {profile}")

    if profile == "v3_unit_recon":
        # Keep this list deliberately short. The goal is to learn the temporal
        # unit distribution, not to satisfy sparse control constraints.
        patch_specs = [
            ("edge_safety_patch", "install_edge_safety_patch"),
            ("edge_recon_contract_patch", "install_recon_contract_patch"),
            ("unit_reconstruction_patch", "install_v3_unit_reconstruction_patch"),
            ("v3c_visible_fk_patch", "install_v3c_visible_fk_patch"),
        ]
    else:
        # Previous patch-heavy path retained for backward compatibility.
        patch_specs = [
            ("trajectory_native_control", "install_native_trajectory_control_patch"),
            ("edge_safety_patch", "install_edge_safety_patch"),
            ("v9_rag_inference_patch", "install_v9_rag_inference_patch"),
            ("edge_full_landing_patch", "install_full_landing_patch"),
            ("text_context_rag_model_patch", "install_text_context_rag_model_patch"),
            ("text_context_rag_io_patch", "install_text_context_rag_io_patch"),
            ("text_bridge_planner_patch", "install_text_bridge_planner_patch"),
            ("edge_recon_contract_patch", "install_recon_contract_patch"),
            ("gait_phase_dataset_patch", "install_gait_phase_dataset_patch"),
            ("trajectory_enhancement_patch", "install_trajectory_enhancement_patch"),
            ("trajectory_event_condition_patch", "install_trajectory_event_condition_patch"),
            ("gait_phase_adapter_patch", "install_gait_phase_adapter_patch"),
            ("trajectory_weak_energy_guidance_patch", "install_weak_trajectory_energy_guidance_patch"),
        ]

    for module_name, fn_name in patch_specs:
        _call_install(module_name, fn_name, verbose=True)


_install_runtime_patches()

from args import parse_train_opt
from EDGE import EDGE


if _profile() == "v3_unit_recon":
    # Install once more after model.diffusion is definitely importable.
    _call_install("unit_reconstruction_patch", "install_v3_unit_reconstruction_patch", verbose=True)
    _call_install("v3c_visible_fk_patch", "install_v3c_visible_fk_patch", verbose=True)
else:
    try:
        from edge_text_context_training_fix import install_edge_text_context_training_fix
        install_edge_text_context_training_fix(EDGE, verbose=True)
    except Exception as exc:
        print(f"⚠️ edge_text_context_training_fix not installed: {exc}")

    try:
        from render_contact_fix_patch import install_render_contact_fix_patch
        install_render_contact_fix_patch(verbose=True)
    except Exception as exc:
        print(f"⚠️ render_contact_fix_patch not installed: {exc}")

    try:
        from edge_nextgen_runtime_patch import install_nextgen_runtime_patches
        install_nextgen_runtime_patches(verbose=True)
    except Exception as exc:
        print(f"⚠️ EDGE nextgen runtime patches not installed: {exc}")

    try:
        from gait_phase_dataset_patch import install_gait_phase_dataset_patch
        from trajectory_enhancement_patch import install_trajectory_enhancement_patch
        from trajectory_event_condition_patch import install_trajectory_event_condition_patch
        from gait_phase_adapter_patch import install_gait_phase_adapter_patch
        from trajectory_weak_energy_guidance_patch import install_weak_trajectory_energy_guidance_patch

        install_gait_phase_dataset_patch(verbose=True)
        install_trajectory_enhancement_patch(verbose=True)
        install_trajectory_event_condition_patch(verbose=True)
        install_gait_phase_adapter_patch(verbose=True)
        install_weak_trajectory_energy_guidance_patch(verbose=True)
    except Exception as exc:
        print(f"⚠️ EDGE native trajectory/event patches not installed after EDGE import: {exc}")

    # V2E/V2F route. Do NOT install this in V3 unit reconstruction.
    try:
        from freeze_aware_motion_patch import install_freeze_aware_motion_patch
        install_freeze_aware_motion_patch(verbose=True)
    except Exception as exc:
        print(f"⚠️ V2E/V2F temporal-progress motion patch not installed: {exc}")


def train(opt):
    if _profile() == "v3_unit_recon":
        # Fail-safe option overrides. These mirror the shell scripts, but they
        # also protect against stale tmux/env copy-paste.
        opt.audio_pairing_mode = "none"
        opt.mmr_loss_weight = 0.0
        opt.disable_traj_cond = True
        opt.trajectory_loss_weight = 0.0
        opt.trajectory_velocity_loss_weight = 0.0
        opt.keyframe_condition_prob = 0.0
        opt.keyframe_loss_weight = 0.0
        opt.mid_keyframe_condition_prob = 0.0
        opt.mid_keyframe_count = 0
        opt.sync_loss_weight = 0.0
        opt.energy_loss_weight = 0.0
        opt.root_lower_coupling_loss_weight = 0.0
        opt.beat_guidance_weight = 0.0
        # Dynamic decoder-memory branches are not needed for V3.
        if getattr(opt, "gradient_checkpointing", False):
            print("⚠️ V3 profile: disabling gradient checkpointing for clean dynamic-condition behavior.")
            opt.gradient_checkpointing = False

    model = EDGE(
        feature_type=opt.feature_type,
        checkpoint_path=opt.checkpoint,
        learning_rate=opt.learning_rate,
        weight_decay=opt.weight_decay,
        audio_dim=opt.audio_dim,
        seq_len=opt.seq_len,
        mixed_precision=opt.mixed_precision,
        gradient_checkpointing=opt.gradient_checkpointing,
        use_sparse_attn=opt.use_sparse_attn,
        sparse_attn_window=opt.sparse_attn_window,
        cond_drop_prob=opt.cond_drop_prob,
        audio_pairing_mode=opt.audio_pairing_mode,
        mmr_loss_weight=opt.mmr_loss_weight,
        keyframe_condition_prob=opt.keyframe_condition_prob,
        keyframe_condition_width=opt.keyframe_condition_width,
        keyframe_loss_weight=opt.keyframe_loss_weight,
        contact_loss_weight=opt.contact_loss_weight,
        foot_loss_weight=opt.foot_loss_weight,
        sync_loss_weight=opt.sync_loss_weight,
        mid_keyframe_condition_prob=opt.mid_keyframe_condition_prob,
        mid_keyframe_count=opt.mid_keyframe_count,
        mid_keyframe_condition_width=opt.mid_keyframe_condition_width,
        mid_keyframe_selection=opt.mid_keyframe_selection,
        beat_guidance_weight=opt.beat_guidance_weight,
        hard_keyframe_project=opt.hard_keyframe_project,
        train_stage=opt.train_stage,
        strict_audio_checkpoint=opt.strict_audio_checkpoint,
        trajectory_loss_weight=opt.trajectory_loss_weight,
        trajectory_velocity_loss_weight=opt.trajectory_velocity_loss_weight,
        energy_condition_prob=opt.energy_condition_prob,
        energy_condition_drop_prob=opt.energy_condition_drop_prob,
        energy_loss_weight=opt.energy_loss_weight,
        root_lower_coupling_loss_weight=opt.root_lower_coupling_loss_weight,
        root_lower_speed_threshold=opt.root_lower_speed_threshold,
        root_lower_min_motion=opt.root_lower_min_motion,
        adapter_train_decoder=opt.adapter_train_decoder,
        enable_rag_summary_token=getattr(opt, "enable_rag_summary_token", False),
        rag_summary_dim=getattr(opt, "rag_summary_dim", 7),
        rag_summary_drop_prob=getattr(opt, "rag_summary_drop_prob", 0.15),
    )
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
