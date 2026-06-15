#!/usr/bin/env python
"""Diagnose gripper learning for diffusion_idm: is the gripper open/close transition under-learned?

(1) Dataset stats: how rare are gripper transitions (the action is a binary absolute open/close,
    held for long stretches with ~2 switches per episode).
(2) Policy prediction: at seen-eval frames (o_t in-distribution), does the policy's predicted
    gripper match GT — overall vs specifically NEAR transitions (where it matters)?

Usage: CUDA_VISIBLE_DEVICES=0 uv run python diag_gripper.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion_idm.modeling_diffusion_idm import OBS_GOAL_IMAGES, DiffusionIDMPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

ROOT = "/workspace/data/libero_goal_image"
GRIP = 6  # gripper action dim


class Nz:
    def __init__(self, ds, image_keys):
        s = ds.meta.stats
        self.img = {k: (torch.tensor(np.asarray(s[k]["mean"], np.float32)).reshape(3, 1, 1),
                        torch.tensor(np.asarray(s[k]["std"], np.float32)).reshape(3, 1, 1)) for k in image_keys}
        self.s_lo = torch.tensor(np.asarray(s[OBS_STATE]["min"], np.float32))
        self.s_hi = torch.tensor(np.asarray(s[OBS_STATE]["max"], np.float32))
        self.a_lo = torch.tensor(np.asarray(s[ACTION]["min"], np.float32))
        self.a_hi = torch.tensor(np.asarray(s[ACTION]["max"], np.float32))

    def image(self, x, k):
        m, sd = self.img[k]; return (x - m) / (sd + 1e-8)

    def state(self, x):
        return 2 * (x - self.s_lo) / (self.s_hi - self.s_lo + 1e-8) - 1

    def unnorm_a(self, a):
        return (a + 1) / 2 * (self.a_hi - self.a_lo) + self.a_lo


def main():
    ds = LeRobotDataset("lerobot/libero_goal_image", root=ROOT)
    ep = ds.meta.episodes
    n_ep = ds.meta.total_episodes

    # (1) dataset gripper stats
    grip = []
    for p in sorted(glob.glob(f"{ROOT}/data/*/*.parquet")):
        a = np.asarray(pq.read_table(p, columns=["action"]).column("action").to_pylist(), np.float32)
        grip.append(a[:, GRIP])
    grip = np.concatenate(grip)
    print("=== dataset gripper ===")
    print("  unique values:", np.unique(np.round(grip, 2)))
    print(f"  %open(<0): {(grip < 0).mean() * 100:.1f}   %close(>0): {(grip > 0).mean() * 100:.1f}")
    trans, lengths = [], []
    for e in range(n_ep):
        fr, to = int(ep["dataset_from_index"][e]), int(ep["dataset_to_index"][e])
        g = np.sign(grip[fr:to])
        trans.append(int((g[1:] != g[:-1]).sum()))
        lengths.append(to - fr)
    trans = np.array(trans)
    frac = trans.sum() / sum(lengths)
    print(f"  transitions/episode: mean={trans.mean():.2f} (min {trans.min()}, max {trans.max()})")
    print(f"  => only {frac * 100:.2f}% of all frames are a gripper transition step")

    # (2) policy gripper prediction at seen-eval frames
    split = json.load(open("/workspace/data/libero_goal_split.json"))
    seen_eval = split["seen_eval_episodes"]
    image_keys = None
    rng = np.random.default_rng(0)
    H = 16
    for variant in ["baseline", "rewind"]:
        ckpt = f"/workspace/lerobot/outputs/idm_{variant}/checkpoints/last/pretrained_model"
        policy = DiffusionIDMPolicy.from_pretrained(ckpt).cuda().eval()
        cfg = policy.config
        n_obs = cfg.n_obs_steps
        image_keys = list(cfg.image_features)
        nz = Nz(ds, image_keys)
        ef = {e: (int(ep["dataset_from_index"][e]), int(ep["dataset_to_index"][e])) for e in range(n_ep)}

        all_ok, near_ok, near_n, hold_ok, hold_n = [], 0, 0, 0, 0
        eps = rng.permutation(seen_eval)[:20].tolist()
        for e in eps:
            fr, to = ef[e]
            gsign = np.sign(grip[fr:to])
            trans_idx = set(np.nonzero(gsign[1:] != gsign[:-1])[0] + 1)  # local transition frames
            # sample frames: all transition frames + some random
            local = sorted(set(list(trans_idx) + rng.integers(0, to - fr, size=8).tolist()))
            for li in local:
                t = fr + li
                idxs = [max(fr, t - (n_obs - 1) + k) for k in range(n_obs)]
                states = torch.stack([nz.state(ds[i][OBS_STATE]) for i in idxs])
                imgs = torch.stack([torch.stack([nz.image(ds[i][k], k) for k in image_keys]) for i in idxs])
                gi = min(t + H, to - 1)
                goal = torch.stack([nz.image(ds[gi][k], k) for k in image_keys])
                batch = {OBS_STATE: states[None].cuda(), OBS_IMAGES: imgs[None].cuda(),
                         OBS_GOAL_IMAGES: goal[None].cuda()}
                with torch.no_grad():
                    a = policy.diffusion.generate_actions(batch)[0].cpu()
                pred_g = nz.unnorm_a(a)[0, GRIP].item()  # first executed step
                gt_g = grip[t]
                ok = (np.sign(pred_g) == np.sign(gt_g))
                all_ok.append(ok)
                if li in trans_idx or (li - 1) in trans_idx or (li + 1) in trans_idx:
                    near_ok += ok; near_n += 1
                else:
                    hold_ok += ok; hold_n += 1
        print(f"=== {variant} gripper sign accuracy (seen-eval, in-dist o_t) ===")
        print(f"  overall: {np.mean(all_ok) * 100:.1f}%  ({len(all_ok)} frames)")
        print(f"  HOLD frames:       {hold_ok / max(1, hold_n) * 100:.1f}%  ({hold_n})")
        print(f"  NEAR-transition:   {near_ok / max(1, near_n) * 100:.1f}%  ({near_n})  <- the hard case")


if __name__ == "__main__":
    main()
