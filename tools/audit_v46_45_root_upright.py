#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import numpy as np

from tools.v46_motionrag_diff import load_bvh_file, rot6d_to_matrix_np, ROT6D_START


def root_tilt_stats(m):
    R = rot6d_to_matrix_np(m[:, ROT6D_START:ROT6D_START+6].reshape(-1,1,6))[:,0]
    up = R[:,:,1]
    cos = np.clip(up[:,1] / np.maximum(np.linalg.norm(up,axis=1), 1e-8), -1, 1)
    tilt = np.degrees(np.arccos(cos))
    root = m[:, [4,5,6]]
    xz_range = float(np.linalg.norm(root[:,[0,2]].max(axis=0)-root[:,[0,2]].min(axis=0)))
    yr = float(root[:,1].max()-root[:,1].min())
    return {
        "tilt_p50": float(np.percentile(tilt,50)),
        "tilt_p95": float(np.percentile(tilt,95)),
        "tilt_max": float(np.max(tilt)),
        "root_xz_range": xz_range,
        "root_y_range": yr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh_dir", default="output/change_rot_only_meter_bvh_v46_44")
    args = ap.parse_args()
    for p in sorted(Path(args.bvh_dir).glob("*.bvh")):
        m = load_bvh_file(p)[0]
        st = root_tilt_stats(m)
        print(p.name, "tilt_p95=", round(st["tilt_p95"],3), "tilt_max=", round(st["tilt_max"],3), "root_xz=", round(st["root_xz_range"],3), "root_y=", round(st["root_y_range"],3))

if __name__ == "__main__":
    main()
