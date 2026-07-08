#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HN-DPO-style safety preference fine-tuning for V46 DiffusionDenoiser.

This is a lightweight preference calibration pass, not a full RLHF stack.  It
uses hard-negative triples produced by V46.41 TGT/KBO:
  snapshot:   local reference window
  preferred: KBO-safe output or rollback snapshot
  rejected:  candidate that triggered KBO

Loss = diffusion reconstruction on preferred + margin ranking that penalizes the
model when rejected is easier to explain than preferred.  It is a DPO-style
surrogate adapted to the existing residual diffusion code.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v46_motionrag_diff_config.json")
    ap.add_argument("--base_diffusion", required=True)
    ap.add_argument("--pairs_jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--margin", type=float, default=0.05)
    args = ap.parse_args()

    import tools.v46_motionrag_diff as v46
    if v46.torch is None:
        raise RuntimeError("PyTorch is required")
    torch = v46.torch
    F = v46.F

    cfg = v46.V46Config.from_json(args.config).apply_env()
    pairs = []
    with open(args.pairs_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    if not pairs:
        raise RuntimeError("No HN-DPO pairs found")

    def load_ckpt(path):
        try:
            return torch.load(path, map_location=cfg.device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=cfg.device)

    ckpt = load_ckpt(args.base_diffusion)
    Tdiff = int(ckpt.get("diffusion_steps", cfg.diffusion_steps))
    model = v46.DiffusionDenoiser(v46.EDGE_DIM, 32).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-5)

    betas, alphas, abar = v46.make_beta_schedule(Tdiff, torch.device(cfg.device))
    cond = torch.zeros((1, 32), device=cfg.device)

    def load_motion(path):
        x = np.load(path).astype(np.float32)
        if x.ndim == 3:
            x = x[0]
        x, _ = v46.enforce_edge151_contract_np(x, cfg, source_hint="v46_41_hn_dpo_load", derive_contact=True, project_rot=True)
        return torch.from_numpy(x[None]).float().to(cfg.device)

    for step in range(int(args.steps)):
        rec = random.choice(pairs)
        ref = load_motion(rec["snapshot"])
        pos = load_motion(rec["preferred"])
        neg = load_motion(rec["rejected"])
        # Align lengths defensively.
        L = min(ref.shape[1], pos.shape[1], neg.shape[1])
        ref = ref[:, :L]; pos = pos[:, :L]; neg = neg[:, :L]
        mask = torch.ones((1, L, 1), device=cfg.device)
        t = torch.randint(0, Tdiff, (1,), device=cfg.device, dtype=torch.long)
        ab = abar[t].view(1, 1, 1)
        noise = torch.randn_like(pos)
        pos_t = torch.sqrt(ab) * pos + torch.sqrt(1 - ab) * noise
        neg_t = torch.sqrt(ab) * neg + torch.sqrt(1 - ab) * noise
        eps_pos = model(pos_t, ref, cond, mask, t)
        eps_neg = model(neg_t, ref, cond, mask, t)
        mse_pos = F.mse_loss(eps_pos, noise)
        mse_neg = F.mse_loss(eps_neg, noise)
        rank = torch.relu(mse_pos - mse_neg + float(args.margin))
        reg = F.smooth_l1_loss(pos, ref)
        loss = mse_pos + 0.35 * rank + 0.02 * reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == int(args.steps) - 1:
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu()), "mse_pos": float(mse_pos.detach().cpu()), "mse_neg": float(mse_neg.detach().cpu()), "rank": float(rank.detach().cpu())}, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt["state_dict"] = model.state_dict()
    ckpt["v46_41_hn_dpo_style_finetune"] = {"steps": int(args.steps), "pairs": len(pairs), "margin": float(args.margin), "lr": float(args.lr)}
    torch.save(ckpt, out)
    print(json.dumps({"out": str(out), "pairs": len(pairs)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
