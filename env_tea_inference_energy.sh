#!/usr/bin/env bash
# TEA energy-conditioned inference flags. Source before generate_controlled.py.

export EDGE_ENERGY_COND=1
export EDGE_ENERGY_LEVEL=0.75
export EDGE_ENERGY_CFG_SCALE=1.5

# Keep PACE safety layer as inference guard.
export EDGE_TRAJ_BEAT_PACING=1
export EDGE_TRAJ_AUTO_SCALE=1
export EDGE_TRAJ_TARGET_ROOT_SPEED=0.010
export EDGE_TRAJ_MAX_ROOT_SPEED=0.014
export EDGE_TRAJ_MIN_SCALE=0.25
export EDGE_TRAJ_MAX_SCALE=0.38
export EDGE_TRAJ_ELASTIC_ANCHOR=1
export EDGE_TRAJ_ANCHOR_STRIDE=20
export EDGE_TRAJ_ANCHOR_BLEND=0.55
