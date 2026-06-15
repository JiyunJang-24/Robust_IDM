#!/usr/bin/env python
"""Diagnose predicted vs ground-truth action magnitude for diffusion_idm checkpoints.

Tests the hypothesis "the robot barely moves because the policy outputs near-zero actions".
At matched (o_t, o_{t+H}) inputs drawn from seen-eval episodes, it compares — in ENV
(un-normalized) action space — the policy's predicted action chunk against the dataset GT
action, for baseline and rewind. If predicted |eef delta| << GT, the policy is under-acting
(e.g. ReWiND's zero-mean action target could bias toward ~0 motion).

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python diag_action_scale.py --n 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion_idm.modeling_diffusion_idm import OBS_GOAL_IMAGES, DiffusionIDMPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


class Normalizer:
    def __init__(self, ds, image_keys):
        s = ds.meta.stats
        self.img = {
            k: (
                torch.tensor(np.asarray(s[k]["mean"], dtype=np.float32)).reshape(3, 1, 1),
                torch.tensor(np.asarray(s[k]["std"], dtype=np.float32)).reshape(3, 1, 1),
            )
            for k in image_keys
        }
        self.s_lo = torch.tensor(np.asarray(s[OBS_STATE]["min"], dtype=np.float32))
        self.s_hi = torch.tensor(np.asarray(s[OBS_STATE]["max"], dtype=np.float32))
        self.a_lo = torch.tensor(np.asarray(s[ACTION]["min"], dtype=np.float32))
        self.a_hi = torch.tensor(np.asarray(s[ACTION]["max"], dtype=np.float32))

    def image(self, x, key):
        m, sd = self.img[key]
        return (x - m) / (sd + 1e-8)

    def state(self, x):
        return 2 * (x - self.s_lo) / (self.s_hi - self.s_lo + 1e-8) - 1

    def unnorm_action(self, a):  # normalized [-1,1] -> env scale
        return (a + 1) / 2 * (self.a_hi - self.a_lo) + self.a_lo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="/workspace/data/libero_goal_image")
    p.add_argument("--repo-id", default="lerobot/libero_goal_image")
    p.add_argument("--split", default="/workspace/data/libero_goal_split.json")
    p.add_argument("--n", type=int, default=30, help="seen-eval episodes to sample")
    p.add_argument("--H", type=int, default=20)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    split = json.load(open(args.split))
    seen_eval = split["seen_eval_episodes"]
    ds = LeRobotDataset(args.repo_id, root=args.dataset_root)
    ep_from = {e: int(ds.meta.episodes["dataset_from_index"][e]) for e in range(ds.meta.total_episodes)}
    ep_to = {e: int(ds.meta.episodes["dataset_to_index"][e]) for e in range(ds.meta.total_episodes)}
    rng = np.random.default_rng(0)
    eps = rng.permutation(seen_eval)[: args.n].tolist()
    positions = [0.0, 0.3, 0.5, 0.7]

    variants = {
        "baseline": "/workspace/lerobot/outputs/idm_baseline/checkpoints/last/pretrained_model",
        "rewind": "/workspace/lerobot/outputs/idm_rewind/checkpoints/last/pretrained_model",
    }

    print(f"sampling {len(eps)} seen-eval episodes x {len(positions)} positions, H={args.H}\n")
    # GT computed once (variant-independent), but image_keys come from policy; assume agentview
    for variant, ckpt in variants.items():
        policy = DiffusionIDMPolicy.from_pretrained(ckpt).to(args.device).eval()
        cfg = policy.config
        n_obs, na = cfg.n_obs_steps, cfg.n_action_steps
        image_keys = list(cfg.image_features)
        nz = Normalizer(ds, image_keys)

        pred_eef, gt_eef, pred_grip, gt_grip, pred_dx = [], [], [], [], []
        for ep in eps:
            f, to = ep_from[ep], ep_to[ep]
            L = to - f
            for pos in positions:
                t = f + int(round(pos * (L - 1)))
                idxs = [max(f, t - (n_obs - 1) + k) for k in range(n_obs)]
                states, imgs = [], []
                for i in idxs:
                    it = ds[i]
                    states.append(nz.state(it[OBS_STATE]))
                    imgs.append(torch.stack([nz.image(it[k], k) for k in image_keys], dim=0))
                gi = min(t + args.H, to - 1)
                gf = ds[gi]
                goal = torch.stack([nz.image(gf[k], k) for k in image_keys], dim=0)
                batch = {
                    OBS_STATE: torch.stack(states).unsqueeze(0).to(args.device),
                    OBS_IMAGES: torch.stack(imgs).unsqueeze(0).to(args.device),
                    OBS_GOAL_IMAGES: goal.unsqueeze(0).to(args.device),
                }
                with torch.no_grad():
                    a = policy.diffusion.generate_actions(batch)[0].cpu()  # (na, adim) normalized
                a_env = nz.unnorm_action(a)
                gt = torch.stack([ds[min(t + k, to - 1)][ACTION] for k in range(na)])  # env scale
                pred_eef.append(a_env[:, :6].norm(dim=-1).mean().item())
                gt_eef.append(gt[:, :6].norm(dim=-1).mean().item())
                pred_grip.append(a_env[:, 6].mean().item())
                gt_grip.append(gt[:, 6].mean().item())
                pred_dx.append(a_env[:, :3].abs().mean().item())

        print(f"=== {variant} ===")
        print(f"  pred |eef delta| (per-step L2):  mean={np.mean(pred_eef):.4f}  median={np.median(pred_eef):.4f}")
        print(f"  GT   |eef delta| (per-step L2):  mean={np.mean(gt_eef):.4f}  median={np.median(gt_eef):.4f}")
        print(f"  ratio pred/GT eef:               {np.mean(pred_eef)/ (np.mean(gt_eef)+1e-9):.3f}")
        print(f"  pred |xyz delta| mean:           {np.mean(pred_dx):.4f}   (GT xyz would be ~similar order)")
        print(f"  pred gripper mean:               {np.mean(pred_grip):+.3f}   GT gripper mean: {np.mean(gt_grip):+.3f}")
        print()


if __name__ == "__main__":
    main()
