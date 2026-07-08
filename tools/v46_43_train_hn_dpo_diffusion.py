#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Motion-density preserving HN-DPO-style fine-tuning for V46 diffusion.

V46.43 fixes static-mode collapse more strictly than V46.42.  The model is not
allowed to win KBO by becoming a lazy/static dancer:

  KE(x0_pred)        >= kinetic_floor_ratio * max(KE(snapshot), KE(preferred))
  MD(x0_pred)        >= motion_density_floor_ratio * MD(preferred)
  VelShape(x0_pred)  stays close to the preferred/reference transition velocity

This is intentionally lightweight and compatible with the existing V46 denoiser.
It is a safety-preference fine-tune, not a full RLHF stack.
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
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--lr", type=float, default=8e-6)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--rank_weight", type=float, default=0.35)
    ap.add_argument("--anchor_reg_weight", type=float, default=0.02)
    ap.add_argument("--kinetic_weight", type=float, default=0.22)
    ap.add_argument("--kinetic_floor_ratio", type=float, default=0.80)
    ap.add_argument("--motion_density_weight", type=float, default=0.18)
    ap.add_argument("--motion_density_floor_ratio", type=float, default=0.75)
    ap.add_argument("--velocity_shape_weight", type=float, default=0.08)
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
            if not line:
                continue
            rec = json.loads(line)
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
        x, _ = v46.enforce_edge151_contract_np(x, cfg, source_hint="v46_43_hn_dpo_load", derive_contact=True, project_rot=True)
        return torch.from_numpy(x[None]).float().to(cfg.device)

    def velocity_components(x):
        root_v = x[:, 1:, [v46.ROOT_X_IDX, v46.ROOT_Y_IDX, v46.ROOT_Z_IDX]] - x[:, :-1, [v46.ROOT_X_IDX, v46.ROOT_Y_IDX, v46.ROOT_Z_IDX]]
        rot_v = x[:, 1:, v46.ROT6D_START:v46.ROT6D_END] - x[:, :-1, v46.ROT6D_START:v46.ROT6D_END]
        return root_v, rot_v

    def kinetic_energy(x):
        if x.shape[1] < 2:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        root_v, rot_v = velocity_components(x)
        root_ke = (root_v ** 2).mean()
        rot_ke = (rot_v ** 2).mean()
        return float(args.root_kinetic_weight) * root_ke + float(args.rot_kinetic_weight) * rot_ke

    def motion_density(x):
        if x.shape[1] < 2:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        root_v, rot_v = velocity_components(x)
        # Mean absolute speed is less dominated by rare spikes than KE.
        return root_v.norm(dim=-1).mean() + 0.10 * rot_v.abs().mean()

    def velocity_shape_loss(x, target):
        if x.shape[1] < 2 or target.shape[1] < 2:
            return torch.zeros((), device=x.device)
        xr, xrot = velocity_components(x)
        tr, trot = velocity_components(target)
        return F.smooth_l1_loss(xr, tr) + 0.20 * F.smooth_l1_loss(xrot, trot)

    def x0_from_noisy(x_t, eps, ab):
        return (x_t - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab).clamp_min(1e-6)

    for step in range(int(args.steps)):
        rec = random.choice(pairs)
        snap = load_motion(rec["snapshot"])
        pos = load_motion(rec["preferred"])
        neg = load_motion(rec["rejected"])
        L = min(snap.shape[1], pos.shape[1], neg.shape[1])
        snap = snap[:, :L]; pos = pos[:, :L]; neg = neg[:, :L]
        mask = torch.ones((1, L, 1), device=cfg.device)
        t = torch.randint(0, Tdiff, (1,), device=cfg.device, dtype=torch.long)
        ab = abar[t].view(1, 1, 1)
        noise = torch.randn_like(pos)
        pos_t = torch.sqrt(ab) * pos + torch.sqrt(1 - ab) * noise
        neg_t = torch.sqrt(ab) * neg + torch.sqrt(1 - ab) * noise
        eps_pos = model(pos_t, snap, cond, mask, t)
        eps_neg = model(neg_t, snap, cond, mask, t)
        mse_pos = F.mse_loss(eps_pos, noise)
        mse_neg = F.mse_loss(eps_neg, noise)
        rank = torch.relu(mse_pos - mse_neg + float(args.margin))
        # Keep the preferred transition close, not the static snapshot alone.
        pref_reg = F.smooth_l1_loss(pos, snap)

        x0_pred = x0_from_noisy(pos_t, eps_pos, ab)
        ke_pred = kinetic_energy(x0_pred)
        ke_snap = kinetic_energy(snap).detach()
        ke_pos = kinetic_energy(pos).detach()
        ke_ref = torch.maximum(ke_snap, ke_pos)
        if float(ke_ref.detach().cpu()) > float(args.kinetic_min_ref):
            kinetic_floor = float(args.kinetic_floor_ratio) * ke_ref
            kinetic_loss = torch.relu(kinetic_floor - ke_pred) / (ke_ref + 1e-6)
        else:
            kinetic_loss = torch.zeros((), device=cfg.device)

        md_pred = motion_density(x0_pred)
        md_pos = motion_density(pos).detach()
        if float(md_pos.detach().cpu()) > float(args.kinetic_min_ref):
            md_floor = float(args.motion_density_floor_ratio) * md_pos
            md_loss = torch.relu(md_floor - md_pred) / (md_pos + 1e-6)
        else:
            md_loss = torch.zeros((), device=cfg.device)

        vshape = velocity_shape_loss(x0_pred, pos)
        loss = (
            mse_pos
            + float(args.rank_weight) * rank
            + float(args.anchor_reg_weight) * pref_reg
            + float(args.kinetic_weight) * kinetic_loss
            + float(args.motion_density_weight) * md_loss
            + float(args.velocity_shape_weight) * vshape
        )
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
                "ke_snap": float(ke_snap.detach().cpu()),
                "ke_pos": float(ke_pos.detach().cpu()),
                "ke_pred": float(ke_pred.detach().cpu()),
                "kinetic_loss": float(kinetic_loss.detach().cpu()),
                "md_pos": float(md_pos.detach().cpu()),
                "md_pred": float(md_pred.detach().cpu()),
                "motion_density_loss": float(md_loss.detach().cpu()),
                "velocity_shape_loss": float(vshape.detach().cpu()),
            }, ensure_ascii=False), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt["state_dict"] = model.state_dict()
    ckpt["v46_43_motion_density_preserving_hn_dpo"] = {
        "steps": int(args.steps),
        "pairs": len(pairs),
        "margin": float(args.margin),
        "lr": float(args.lr),
        "kinetic_floor_ratio": float(args.kinetic_floor_ratio),
        "kinetic_weight": float(args.kinetic_weight),
        "motion_density_floor_ratio": float(args.motion_density_floor_ratio),
        "motion_density_weight": float(args.motion_density_weight),
        "velocity_shape_weight": float(args.velocity_shape_weight),
        "purpose": "avoid static/lazy-dancer mode while suppressing KBO hard negatives",
    }
    torch.save(ckpt, out)
    print(json.dumps({"out": str(out), "pairs": len(pairs), "kinetic_floor_ratio": float(args.kinetic_floor_ratio), "motion_density_floor_ratio": float(args.motion_density_floor_ratio)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
