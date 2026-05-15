import os

# ===== Strict single-unit reconstruction defaults =====
os.environ.setdefault("EDGE_SINGLE_RECON_PATCH", "1")

# Disable unrelated generation-control modules.
os.environ.setdefault("EDGE_DYNAMIC_TRAJ_CFG", "0")
os.environ.setdefault("EDGE_GAIT_PHASE_COND", "0")
os.environ.setdefault("EDGE_GAIT_CONTACT_LOSS", "0")
os.environ.setdefault("EDGE_TRAJ_PHYSICS_FEATURES", "0")
os.environ.setdefault("EDGE_TRAJ_FOURIER_FEATURES", "0")
os.environ.setdefault("EDGE_TRAJ_SPARSE_WAYPOINT", "0")
os.environ.setdefault("EDGE_TRAJ_BEV_COND", "0")

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

# Body-part weights: emphasize the parts that failed in v14 diagnostics.
os.environ.setdefault("EDGE_RECON_LOSS_ROOT_XZ_W", "0.0")
os.environ.setdefault("EDGE_RECON_LOSS_ROOT_Y_W", "1.0")
os.environ.setdefault("EDGE_RECON_LOSS_CONTACT_W", "0.25")
os.environ.setdefault("EDGE_RECON_LOSS_PELVIS_W", "4.0")
os.environ.setdefault("EDGE_RECON_LOSS_LOWER_W", "2.0")
os.environ.setdefault("EDGE_RECON_LOSS_TORSO_W", "4.0")
os.environ.setdefault("EDGE_RECON_LOSS_UPPER_W", "8.0")

# Useful debug.
os.environ.setdefault("EDGE_RECON_EXTRA_DEBUG", "1")

# Import original training entry. This installs the normal project patches first.
import train as edge_train
from args import parse_train_opt

from edge_single_unit_recon_patch import install_single_unit_recon_patch

# Re-install after train.py / nextgen / gait patches, so this patch is last.
install_single_unit_recon_patch(verbose=True)


if __name__ == "__main__":
    opt = parse_train_opt()
    edge_train.train(opt)
