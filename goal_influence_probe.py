#!/usr/bin/env python
"""Goal-influence probe for the goal-conditioned IDM (open-loop, sim-free).

Tests whether the policy actually attends to the goal image o_{t+H}, with o_t held
IN-DISTRIBUTION, by measuring how the predicted action chunk changes when the goal is
swapped — at different trajectory positions (init vs mid) and for different goal types
(correct / seen-task swap / unseen-task swap).

Probe states o_t come from the SEEN-task HELD-OUT (seen-eval) episodes (in-distribution
o_t, not memorized). For each state:
  - correct goal       : the same episode's frame at t+H
  - seen-task swap goal : a frame from a different SEEN task's episode
  - unseen-task swap    : a frame from an UNSEEN task's episode
Metrics per (position, goal-type):
  - influence = || a(o_t, swapped_goal) - a(o_t, correct_goal) ||   (attention; baseline≈0 if it ignores the goal)
  - correct_err = || a(o_t, correct_goal) - GT_action ||            (sanity: does it work with the right goal)
All in normalized action space (consistent across conditions).

Hypothesis: baseline influence collapses at MID-traj (where o_t already determines the task)
and especially for UNSEEN goals; ReWiND keeps influence high. The mid × unseen cell shows
the largest baseline↔ReWiND gap.

Usage:
    uv run python goal_influence_probe.py \
        --policy-path /workspace/lerobot/outputs/idm_baseline/checkpoints/last/pretrained_model \
        --dataset-root /workspace/data/libero_goal_image \
        --split /workspace/data/libero_goal_split.json \
        --n-episodes 30 --output /workspace/probe_baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion_idm.modeling_diffusion_idm import OBS_GOAL_IMAGES, DiffusionIDMPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def _episode_task_map(ds: LeRobotDataset) -> dict[int, int]:
    """episode_index -> task_index."""
    ep = ds.meta.episodes
    tt = pq.read_table(Path(ds.root) / "meta" / "tasks.parquet")
    desc2i = dict(zip(tt.column("task").to_pylist(), tt.column("task_index").to_pylist()))
    out = {}
    for e in range(ds.meta.total_episodes):
        tasks = ep["tasks"][e]
        desc = tasks[0] if hasattr(tasks, "__len__") and not isinstance(tasks, str) else tasks
        out[e] = desc2i[desc]
    return out


class Normalizer:
    def __init__(self, ds: LeRobotDataset, image_keys: list[str]):
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

    def image(self, x: torch.Tensor, key: str) -> torch.Tensor:
        m, sd = self.img[key]
        return (x - m) / (sd + 1e-8)

    def state(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * (x - self.s_lo) / (self.s_hi - self.s_lo + 1e-8) - 1

    def action(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * (x - self.a_lo) / (self.a_hi - self.a_lo + 1e-8) - 1


def main() -> None:
    p = argparse.ArgumentParser(description="Goal-influence probe for diffusion_idm.")
    p.add_argument("--policy-path", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--repo-id", type=str, default="lerobot/libero_goal_image")
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--positions", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7])
    p.add_argument("--n-episodes", type=int, default=30, help="seen-eval episodes to probe")
    p.add_argument("--goal-offset", type=int, default=20, help="H (frames) for the subgoal")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    split = json.load(open(args.split))
    seen_eval = split["seen_eval_episodes"]
    unseen_eval = split["unseen_eval_episodes"]

    policy = DiffusionIDMPolicy.from_pretrained(args.policy_path).to(args.device).eval()
    cfg = policy.config
    n_obs, H = cfg.n_obs_steps, args.goal_offset
    image_keys = list(cfg.image_features)

    ds = LeRobotDataset(args.repo_id, root=str(args.dataset_root))
    nz = Normalizer(ds, image_keys)
    ep_task = _episode_task_map(ds)
    ep_from = {e: int(ds.meta.episodes["dataset_from_index"][e]) for e in range(ds.meta.total_episodes)}
    ep_to = {e: int(ds.meta.episodes["dataset_to_index"][e]) for e in range(ds.meta.total_episodes)}

    # pools of swap episodes by split
    seen_eval_by_task: dict[int, list[int]] = {}
    for e in seen_eval:
        seen_eval_by_task.setdefault(ep_task[e], []).append(e)

    def frames_at(ep: int, idx: int) -> dict[str, torch.Tensor]:
        """Raw CHW float frames for both cameras at absolute frame idx (clamped to episode)."""
        idx = max(ep_from[ep], min(ep_to[ep] - 1, idx))
        item = ds[idx]
        return {k: item[k] for k in image_keys}

    def obs_history(ep: int, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized (state_hist (n_obs,sdim), img_hist (n_obs,num_cams,3,H,W))."""
        idxs = [max(ep_from[ep], t - (n_obs - 1) + k) for k in range(n_obs)]
        states, imgs = [], []
        for i in idxs:
            item = ds[i]
            states.append(nz.state(item[OBS_STATE]))
            imgs.append(torch.stack([nz.image(item[k], k) for k in image_keys], dim=0))
        return torch.stack(states), torch.stack(imgs)

    def norm_goal(ep: int, idx: int) -> torch.Tensor:
        f = frames_at(ep, idx)
        return torch.stack([nz.image(f[k], k) for k in image_keys], dim=0)  # (num_cams,3,H,W)

    @torch.no_grad()
    def predict(state_hist, img_hist, goal_imgs) -> torch.Tensor:
        batch = {
            OBS_STATE: state_hist.unsqueeze(0).to(args.device),
            OBS_IMAGES: img_hist.unsqueeze(0).to(args.device),
            OBS_GOAL_IMAGES: goal_imgs.unsqueeze(0).to(args.device),
        }
        return policy.diffusion.generate_actions(batch)[0].cpu()  # (n_action_steps, adim)

    probe_eps = list(seen_eval)
    rng.shuffle(probe_eps)
    probe_eps = probe_eps[: args.n_episodes]

    # accumulate metrics: results[pos][goal_type] = list of values
    results: dict = {f"{pos:.2f}": {"influence_seen": [], "influence_unseen": [], "correct_err": []} for pos in args.positions}

    for ep in probe_eps:
        task = ep_task[ep]
        ep_len = ep_to[ep] - ep_from[ep]
        for pos in args.positions:
            t = ep_from[ep] + int(round(pos * (ep_len - 1)))
            state_hist, img_hist = obs_history(ep, t)
            # GT action at o_t (normalized)
            gt = nz.action(ds[t][ACTION])
            # goals
            correct_goal = norm_goal(ep, t + H)
            # seen swap: a different seen task
            other_tasks = [tk for tk in seen_eval_by_task if tk != task]
            sw_task = int(rng.choice(other_tasks))
            sw_ep = int(rng.choice(seen_eval_by_task[sw_task]))
            sw_len = ep_to[sw_ep] - ep_from[sw_ep]
            seen_goal = norm_goal(sw_ep, ep_from[sw_ep] + int(round(pos * (sw_len - 1))) + H)
            # unseen swap
            us_ep = int(rng.choice(unseen_eval))
            us_len = ep_to[us_ep] - ep_from[us_ep]
            unseen_goal = norm_goal(us_ep, ep_from[us_ep] + int(round(pos * (us_len - 1))) + H)

            a_correct = predict(state_hist, img_hist, correct_goal)
            a_seen = predict(state_hist, img_hist, seen_goal)
            a_unseen = predict(state_hist, img_hist, unseen_goal)

            r = results[f"{pos:.2f}"]
            r["influence_seen"].append(float((a_seen - a_correct).norm(dim=-1).mean()))
            r["influence_unseen"].append(float((a_unseen - a_correct).norm(dim=-1).mean()))
            r["correct_err"].append(float((a_correct - gt).norm(dim=-1).mean()))

    summary = {
        "policy_path": str(args.policy_path),
        "n_episodes": len(probe_eps),
        "goal_offset_H": H,
        "positions": {},
    }
    print(f"\npolicy: {args.policy_path}")
    print(f"{'pos':>5} | {'infl_seen':>10} | {'infl_unseen':>12} | {'correct_err':>11}")
    for pos in args.positions:
        r = results[f"{pos:.2f}"]
        m = {k: float(np.mean(v)) for k, v in r.items()}
        summary["positions"][f"{pos:.2f}"] = m
        tag = " (init)" if pos == 0.0 else " (mid)"
        print(f"{pos:>5.2f} | {m['influence_seen']:>10.4f} | {m['influence_unseen']:>12.4f} | {m['correct_err']:>11.4f}{tag}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
