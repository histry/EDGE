#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deprecated window-level contact relabel entry point.

V33 forbids relabeling an already-windowed transition NPZ because overlapping
windows can assign inconsistent labels to the same source frame.  Build the
event contact cache first and rebuild the transition dataset synchronously.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Window-level contact relabeling is disabled in V33."
    )
    parser.add_argument("--input_npz", default="")
    parser.add_argument("--out_npz", default="")
    parser.parse_known_args()
    raise RuntimeError(
        "V33 forbids window-level contact relabeling. Run:\n"
        "  python tools/build_v33_event_contact_cache.py ...\n"
        "then rebuild the transition dataset with:\n"
        "  python tools/build_v27_transition_diffusion_dataset.py "
        "--event_contact_cache <cache.npz> ..."
    )


if __name__ == "__main__":
    main()
