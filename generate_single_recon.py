import os

# ===== Strict reconstruction inference defaults =====
os.environ.setdefault("EDGE_SINGLE_RECON_PATCH", "1")

# Disable unrelated control modules.
os.environ.setdefault("EDGE_DYNAMIC_TRAJ_CFG", "0")
os.environ.setdefault("EDGE_GAIT_PHASE_COND", "0")
os.environ.setdefault("EDGE_GAIT_CONTACT_LOSS", "0")
os.environ.setdefault("EDGE_TRAJ_PHYSICS_FEATURES", "0")
os.environ.setdefault("EDGE_TRAJ_FOURIER_FEATURES", "0")
os.environ.setdefault("EDGE_TRAJ_SPARSE_WAYPOINT", "0")
os.environ.setdefault("EDGE_TRAJ_BEV_COND", "0")

# Keep RAG branches only when checkpoint has them, but they are not the main fix.
os.environ.setdefault("EDGE_ENABLE_TEXT_CONTEXT_RAG", "1")
os.environ.setdefault("EDGE_ENABLE_RAG_SUMMARY_TOKEN", "1")
os.environ.setdefault("EDGE_V11_CROSS_ATTN_RAG", "1")

# Bridge condition from sparse/dense keyframes.
os.environ.setdefault("EDGE_RECON_BRIDGE_COND", "1")
os.environ.setdefault("EDGE_RECON_BRIDGE_FEATURES", "rot+root_y")
os.environ.setdefault("EDGE_RECON_BRIDGE_STRENGTH", "0.35")

# Hard projection remains only for true hard keyframes / root reference.
os.environ.setdefault("EDGE_HARD_KEYFRAME_PROJECT", "1")
os.environ.setdefault("EDGE_INFER_PROJECT_XSTART", "1")

# Import original generation script. It installs normal runtime patches first.
import generate_controlled as gen

from edge_single_unit_recon_patch import install_single_unit_recon_patch

# Re-install after generate_controlled.py patches so bridge model_predictions wins.
install_single_unit_recon_patch(verbose=True)


if __name__ == "__main__":
    gen.main()
