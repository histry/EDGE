import os

# ===== Strict single-unit reconstruction defaults =====
os.environ.setdefault("EDGE_SINGLE_RECON_PATCH", "1")

# Single-unit dataset has only one source group. This is intentional for strict
# reconstruction sanity check, so allow train/val reuse for this script only.
os.environ.setdefault("EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT", "1")

# Match clean inference architecture: no text-context / V11 branch unless
# explicitly enabled by caller.
os.environ.setdefault("EDGE_ENABLE_TEXT_CONTEXT_RAG", "0")
os.environ.setdefault("EDGE_V11_CROSS_ATTN_RAG", "0")
os.environ.setdefault("EDGE_ENABLE_RAG_SUMMARY_TOKEN", "1")

# Disable unrelated generation-control modules.
os.environ.setdefault("EDGE_DYNAMIC_TRAJ_CFG", "0")
os.environ.setdefault("EDGE_GAIT_PHASE_COND", "0")
os.environ.setdefault("EDGE_GAIT_CONTACT_LOSS", "0")
os.environ.setdefault("EDGE_TRAJ_PHYSICS_FEATURES", "0")
os.environ.setdefault("EDGE_TRAJ_FOURIER_FEATURES", "0")
os.environ.setdefault("EDGE_TRAJ_SPARSE_WAYPOINT", "0")
os.environ.setdefault("EDGE_TRAJ_BEV_COND", "0")

# Keep trajectory/gait wrappers off for clean architecture.
os.environ.setdefault("EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER", "1")
os.environ.setdefault("EDGE_CHECKPOINT_COMPAT_CPU_MERGE", "1")
os.environ.setdefault("EDGE_AUDIO_DEVICE", "cpu")

# Dense teacher constraints for single-unit reconstruction.
os.environ.setdefault("EDGE_RECON_TRAIN_DENSE_KEYFRAMES", "1")
os.environ.setdefault("EDGE_RECON_TRAIN_DENSE_STRIDE", "4")
os.environ.setdefault("EDGE_RECON_TRAIN_ROOT_XZ_ALL", "1")
os.environ.setdefault("EDGE_RECON_TRAIN_HARD_FEATURES", "rot+root_y+contacts")

# Bridge condition: soft model input, not hard projection.
os.environ.setdefault("EDGE_RECON_BRIDGE_COND", "1")
os.environ.setdefault("EDGE_RECON_BRIDGE_FEATURES", "rot+root_y")
os.environ.setdefault("EDGE_RECON_BRIDGE_STRENGTH", "0.35")

# Extra reconstruction losses.
os.environ.setdefault("EDGE_RECON_EXTRA_LOSS", "1")
os.environ.setdefault("EDGE_RECON_EXTRA_X0_W", "50.0")
os.environ.setdefault("EDGE_RECON_EXTRA_VEL_W", "25.0")
os.environ.setdefault("EDGE_RECON_EXTRA_ACC_W", "5.0")
os.environ.setdefault("EDGE_RECON_EXTRA_KEY_NEIGHBOR_W", "10.0")
os.environ.setdefault("EDGE_RECON_KEY_NEIGHBOR_RADIUS", "1")

# Body-part weights.
os.environ.setdefault("EDGE_RECON_LOSS_ROOT_XZ_W", "0.0")
os.environ.setdefault("EDGE_RECON_LOSS_ROOT_Y_W", "1.0")
os.environ.setdefault("EDGE_RECON_LOSS_CONTACT_W", "0.25")
os.environ.setdefault("EDGE_RECON_LOSS_PELVIS_W", "4.0")
os.environ.setdefault("EDGE_RECON_LOSS_LOWER_W", "2.0")
os.environ.setdefault("EDGE_RECON_LOSS_TORSO_W", "4.0")
os.environ.setdefault("EDGE_RECON_LOSS_UPPER_W", "8.0")

# Useful debug.
os.environ.setdefault("EDGE_RECON_EXTRA_DEBUG", "1")

import train as edge_train
from args import parse_train_opt

from edge_single_unit_recon_patch import install_single_unit_recon_patch

# Re-install after train.py / nextgen / gait patches, so this patch is last.
install_single_unit_recon_patch(verbose=True)


if __name__ == "__main__":
    opt = parse_train_opt()
    edge_train.train(opt)
