#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kinetic-preserving HN-DPO-style safety preference fine-tuning for V46 diffusion.

Fixes the V46.41 static-mode-collapse risk.  Hard-negative pairs are useful only
if the preferred candidate remains dynamically expressive.  Therefore, besides
DPO-style ranking, this tool adds a kinetic-energy floor:

    KE(x0_pred) >= kinetic_floor_ratio * KE(reference)

where KE is mean squared root/rotation velocity.  This discourages reward hacking
where the model avoids KBO by producing frozen/static transitions.
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
    ap.add_argument("--rank_weight", type=float, default=0.35)
    ap.add_argument("--anchor_reg_weight", type=float, default=0.02)
    ap.add_argument("--kinetic_weight", type=float, default=0.18)
    ap.add_argument("--kinetic_floor_ratio", type=float, default=0.80)
    ap.add_argument("--kinetic_min_ref", type=float, default=1e-5)
    ap.add_argument("--rot_kinetic_weight", type=float, default=0.35)
    ap.add_argument("--root_kinetic_weight", type=float, default=1.00)
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
                rec = json.loads(line)
                # Accept both old/new key names.
                if "preferred" not in rec and "accepted" in rec:
                    rec["preferred"] = rec["accepted"]
                if all(k in rec for k in ("snapshot", "preferred", "rejected")):
                    pairs.append(rec)
    if not pairs:
        raise RuntimeError("No valid HN-DPO pairs found")

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
        x, _ = v46.enforce_edge151_contract_np(x, cfg, source_hint="v46_42_hn_dpo_load", derive_contact=True, project_rot=True)
        return torch.from_numpy(x[None]).float().to(cfg.device)

    def kinetic_energy(x):
        # x: [1,T,151].  Use root trajectory and rot6d temporal changes.
        if x.shape[1] < 2:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        root_v = x[:, 1:, [v46.ROOT_X_IDX, v46.ROOT_Y_IDX, v46.ROOT_Z_IDX]] - x[:, :-1, [v46.ROOT_X_IDX, v46.ROOT_Y_IDX, v46.ROOT_Z_IDX]]
        rot_v = x[:, 1:, v46.ROT6D_START:v46.ROT6D_END] - x[:, :-1, v46.ROT6D_START:v46.ROT6D_END]
        root_ke = (root_v ** 2).mean()
        rot_ke = (rot_v ** 2).mean()
        return float(args.root_kinetic_weight) * root_ke + float(args.rot_kinetic_weight) * rot_ke

    def x0_from_noisy(x_t, eps, ab):
        return (x_t - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab).clamp_min(1e-6)

    for step in range(int(args.steps)):
        rec = random.choice(pairs)
        ref = load_motion(rec["snapshot"])
        pos = load_motion(rec["preferred"])
        neg = load_motion(rec["rejected"])
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

        x0_pred = x0_from_noisy(pos_t, eps_pos, ab)
        ke_pred = kinetic_energy(x0_pred)
        ke_ref = kinetic_energy(ref).detach()
        if float(ke_ref.detach().cpu()) > float(args.kinetic_min_ref):
            kinetic_floor = float(args.kinetic_floor_ratio) * ke_ref
            kinetic_loss = torch.relu(kinetic_floor - ke_pred) / (ke_ref + 1e-6)
        else:
            kinetic_loss = torch.zeros((), device=cfg.device)

        loss = mse_pos + float(args.rank_weight) * rank + float(args.anchor_reg_weight) * reg + float(args.kinetic_weight) * kinetic_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == int(args.steps) - 1:
            print(json.dumps({
                "step": step,
                "loss": float(loss.detach().cpu()),
                "mse_pos": float(mse_pos.detach().cpu()),
                "mse_neg": float(mse_neg.detach().cpu()),
                "rank": float(rank.detach().cpu()),
                "ke_ref": float(ke_ref.detach().cpu()),
                "ke_pred": float(ke_pred.detach().cpu()),
                "kinetic_loss": float(kinetic_loss.detach().cpu()),
            }, ensure_ascii=False), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt["state_dict"] = model.state_dict()
    ckpt["v46_42_kinetic_hn_dpo_finetune"] = {
        "steps": int(args.steps),
        "pairs": len(pairs),
        "margin": float(args.margin),
        "lr": float(args.lr),
        "kinetic_floor_ratio": float(args.kinetic_floor_ratio),
        "kinetic_weight": float(args.kinetic_weight),
        "purpose": "avoid static mode collapse while suppressing KBO hard negatives",
    }
    torch.save(ckpt, out)
    print(json.dumps({"out": str(out), "pairs": len(pairs), "kinetic_floor_ratio": float(args.kinetic_floor_ratio)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
