#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional EDGE event-adapter training entrypoint.

This deliberately reuses the project's existing train.py / EDGE.py and only
forces the V21 RAG-summary settings. Use after the retrieval/scheduler path is
stable; it is not required for V21 query-time routing.
"""
from __future__ import annotations

import os

os.environ.setdefault("EDGE_ENABLE_V21_EVENT_ADAPTER", "1")
os.environ.setdefault("EDGE_TRAIN_PROFILE", "legacy")

from v21_event_adapter_patch import install_v21_event_adapter_patch

install_v21_event_adapter_patch(verbose=True)

from args import parse_train_opt
from train import train


if __name__ == "__main__":
    opt = parse_train_opt()
    opt.enable_rag_summary_token = True
    opt.rag_summary_dim = 12
    opt.rag_summary_drop_prob = float(os.environ.get("EDGE_V21_EVENT_DROP_PROB", "0.15"))
    opt.train_stage = os.environ.get("EDGE_V21_TRAIN_STAGE", "adapter")
    opt.adapter_train_decoder = str(os.environ.get("EDGE_V21_ADAPTER_TRAIN_DECODER", "0")).lower() in {
        "1", "true", "yes", "on"
    }
    train(opt)
