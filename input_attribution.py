#!/usr/bin/env python
"""How much does the policy rely on the GOAL image o_{t+H} vs the CURRENT image o_t?

The two images go through separate ResNet encoders whose features are concatenated into the
diffusion global conditioning. To measure each stream's contribution we ABLATE one image at a
time (replace it with the dataset-mean image = normalized zeros, i.e. an information-free input)
and measure how much the predicted action chunk changes. Diffusion noise is FIXED so the only
variable is which image was ablated.

  obs_sens  = || a(o_t, goal) - a(MEAN, goal) ||      (reliance on the current image o_t)
  goal_sens = || a(o_t, goal) - a(o_t, MEAN)  ||      (reliance on the goal image o_{t+H})
  goal_share = goal_sens / (obs_sens + goal_sens)     (fraction of attention on the goal)

Our method hypothesis: ReWiND makes (o_t, a) non-unique, so the policy must read the goal ->
higher goal_share than baseline, growing through the trajectory (mid) where o_t alone would
otherwise determine the action.

Compares two models (h8weighted baseline vs rewind) across trajectory positions.

Usage:
  uv run python input_attribution.py \
    --baseline outputs/idm_baseline/checkpoints/last/pretrained_model \
    --rewind   outputs/idm_rewind/checkpoints/last/pretrained_model \
    --goal-offset 8 --n-episodes 30 --out /workspace/input_attribution.png
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/lerobot/src")
sys.path.insert(0, "/workspace")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion_idm.modeling_diffusion_idm import DiffusionIDMPolicy, OBS_GOAL_IMAGES
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from goal_influence_probe import Normalizer, _episode_task_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--rewind", required=True)
    ap.add_argument("--dataset-root", default="/workspace/data/libero_goal_image")
    ap.add_argument("--repo-id", default="lerobot/libero_goal_image")
    ap.add_argument("--split", default="/workspace/data/libero_goal_split.json")
    ap.add_argument("--goal-offset", type=int, default=8)
    ap.add_argument("--positions", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7])
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/workspace/input_attribution.png")
    ap.add_argument("--out-json", default="/workspace/input_attribution.json")
    args = ap.parse_args()

    dev = args.device
    split = json.load(open(args.split))
    seen_eval = split["seen_eval_episodes"]

    models = {
        "baseline": DiffusionIDMPolicy.from_pretrained(args.baseline).to(dev).eval(),
        "rewind": DiffusionIDMPolicy.from_pretrained(args.rewind).to(dev).eval(),
    }
    cfg = models["baseline"].config
    n_obs, H = cfg.n_obs_steps, args.goal_offset
    image_keys = list(cfg.image_features)
    adim = None

    ds = LeRobotDataset(args.repo_id, root=str(args.dataset_root))
    nz = Normalizer(ds, image_keys)
    ep_task = _episode_task_map(ds)
    ep_from = {e: int(ds.meta.episodes["dataset_from_index"][e]) for e in range(ds.meta.total_episodes)}
    ep_to = {e: int(ds.meta.episodes["dataset_to_index"][e]) for e in range(ds.meta.total_episodes)}

    def obs_hist(ep, t):
        idxs = [max(ep_from[ep], t - (n_obs - 1) + k) for k in range(n_obs)]
        states, imgs = [], []
        for i in idxs:
            it = ds[i]
            states.append(nz.state(it[OBS_STATE]))
            imgs.append(torch.stack([nz.image(it[k], k) for k in image_keys], dim=0))
        return torch.stack(states), torch.stack(imgs)

    def norm_goal(ep, idx):
        idx = max(ep_from[ep], min(ep_to[ep] - 1, idx))
        it = ds[idx]
        return torch.stack([nz.image(it[k], k) for k in image_keys], dim=0)

    @torch.no_grad()
    def gen(model, state_hist, img_hist, goal, noise):
        batch = {
            OBS_STATE: state_hist.unsqueeze(0).to(dev),
            OBS_IMAGES: img_hist.unsqueeze(0).to(dev),
            OBS_GOAL_IMAGES: goal.unsqueeze(0).to(dev),
        }
        return model.diffusion.generate_actions(batch, noise=noise)[0]

    @torch.no_grad()
    def attribution(model, state_hist, img_hist, goal):
        nonlocal adim
        if adim is None:
            adim = nz.a_lo.numel()
        g = torch.Generator(device=dev).manual_seed(args.seed)
        noise = torch.randn(1, cfg.horizon, adim, generator=g, device=dev)
        mean_img = torch.zeros_like(img_hist)  # normalized mean image = info-free o_t
        mean_goal = torch.zeros_like(goal)
        a_full = gen(model, state_hist, img_hist, goal, noise)
        a_obs_off = gen(model, state_hist, mean_img, goal, noise)  # ablate current image
        a_goal_off = gen(model, state_hist, img_hist, mean_goal, noise)  # ablate goal image
        obs_sens = float(torch.norm(a_full - a_obs_off))
        goal_sens = float(torch.norm(a_full - a_goal_off))
        return obs_sens, goal_sens

    rng = np.random.default_rng(args.seed)
    eps = list(seen_eval)
    rng.shuffle(eps)
    eps = eps[: args.n_episodes]

    # acc[model][pos] = {obs:[], goal:[]}
    acc = {m: {f"{p:.2f}": {"obs": [], "goal": []} for p in args.positions} for m in models}
    for ep in eps:
        L = ep_to[ep] - ep_from[ep]
        for pos in args.positions:
            t = ep_from[ep] + int(round(pos * (L - 1)))
            sh, ih = obs_hist(ep, t)
            goal = norm_goal(ep, t + H)
            for mname, model in models.items():
                o, gl = attribution(model, sh, ih, goal)
                acc[mname][f"{pos:.2f}"]["obs"].append(o)
                acc[mname][f"{pos:.2f}"]["goal"].append(gl)

    # summarize
    summary = {}
    print(f"\n{'model':>9} {'pos':>5} | {'obs_sens':>9} {'goal_sens':>9} {'goal_share':>11}")
    for mname in models:
        summary[mname] = {}
        for pos in args.positions:
            d = acc[mname][f"{pos:.2f}"]
            o, gl = float(np.mean(d["obs"])), float(np.mean(d["goal"]))
            share = gl / (o + gl + 1e-8)
            summary[mname][f"{pos:.2f}"] = {"obs_sens": o, "goal_sens": gl, "goal_share": share}
            print(f"{mname:>9} {pos:>5.2f} | {o:>9.4f} {gl:>9.4f} {share:>10.1%}")
    json.dump(summary, open(args.out_json, "w"), indent=2)

    # plot: goal_share vs position (line) + obs/goal sens stacked bars
    positions = args.positions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for mname, color in [("baseline", "tab:blue"), ("rewind", "tab:orange")]:
        shares = [summary[mname][f"{p:.2f}"]["goal_share"] * 100 for p in positions]
        ax1.plot(positions, shares, "o-", color=color, label=mname, lw=2)
    ax1.axhline(50, ls="--", c="gray", lw=1)
    ax1.set_xlabel("trajectory position (o_t)")
    ax1.set_ylabel("goal_share  =  goal_sens / (obs_sens+goal_sens)  [%]")
    ax1.set_title("Reliance on GOAL image vs current image o_t")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # stacked sensitivity bars at each position
    x = np.arange(len(positions))
    w = 0.35
    for i, mname in enumerate(["baseline", "rewind"]):
        obs = [summary[mname][f"{p:.2f}"]["obs_sens"] for p in positions]
        gl = [summary[mname][f"{p:.2f}"]["goal_sens"] for p in positions]
        ax2.bar(x + (i - 0.5) * w, obs, w, color="tab:gray", alpha=0.6,
                label="obs_sens" if i == 0 else None)
        ax2.bar(x + (i - 0.5) * w, gl, w, bottom=obs,
                color="tab:orange" if mname == "rewind" else "tab:blue",
                label=f"goal_sens ({mname})")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{p:.1f}" for p in positions])
    ax2.set_xlabel("trajectory position")
    ax2.set_ylabel("sensitivity ||Δaction||")
    ax2.set_title("obs vs goal sensitivity (bars: baseline left, rewind right)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nwrote {args.out} and {args.out_json}")


if __name__ == "__main__":
    main()
