#!/usr/bin/env python
"""Build a reproducible seen/unseen task split of a LeRobot LIBERO dataset.

Splits the suite's tasks into SEEN (used for training) and UNSEEN (held out entirely),
and within each seen task holds out a fraction of episodes for evaluation (so seen-task
eval does not reuse the exact trajectories the policy trained on).

Outputs a JSON split file with:
  - seen_task_ids / unseen_task_ids
  - train_episodes            : episodes to train on  (--dataset.episodes)
  - seen_eval_episodes        : held-out seen-task episodes (eval subgoal source)
  - unseen_eval_episodes      : all unseen-task episodes  (eval subgoal source)
  - per_task                  : breakdown for transparency

Example:
    uv run python make_libero_split.py \
        --dataset-root /workspace/data/libero_goal_image \
        --unseen-task-ids 7 9 --holdout-frac 0.2 --seed 42 \
        --output /workspace/data/libero_goal_split.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _task_to_episodes(dataset_dir: Path) -> tuple[dict[int, list[int]], dict[int, str]]:
    ep = pq.read_table(sorted((dataset_dir / "meta" / "episodes").glob("*/*.parquet"))).to_pandas()
    tt = pq.read_table(dataset_dir / "meta" / "tasks.parquet")
    desc2idx = dict(zip(tt.column("task").to_pylist(), tt.column("task_index").to_pylist()))
    idx2desc = {v: k for k, v in desc2idx.items()}
    ep["ti"] = ep["tasks"].apply(lambda x: desc2idx[x[0]])
    task_eps = {
        int(ti): sorted(int(e) for e in ep[ep["ti"] == ti]["episode_index"].tolist())
        for ti in sorted(ep["ti"].unique())
    }
    return task_eps, idx2desc


def make_split(
    dataset_dir: Path,
    unseen_task_ids: list[int],
    holdout_frac: float,
    seed: int,
) -> dict:
    task_eps, idx2desc = _task_to_episodes(dataset_dir)
    all_tasks = sorted(task_eps)
    unseen = sorted(set(unseen_task_ids))
    seen = [t for t in all_tasks if t not in unseen]
    rng = np.random.default_rng(seed)

    train_eps: list[int] = []
    seen_eval_eps: list[int] = []
    per_task = {}
    for t in seen:
        eps = task_eps[t][:]
        rng.shuffle(eps)
        n_hold = max(1, round(len(eps) * holdout_frac))
        hold = sorted(eps[:n_hold])
        tr = sorted(eps[n_hold:])
        train_eps += tr
        seen_eval_eps += hold
        per_task[t] = {"desc": idx2desc[t], "split": "seen", "n_train": len(tr), "n_eval": len(hold)}

    unseen_eval_eps: list[int] = []
    for t in unseen:
        unseen_eval_eps += task_eps[t]
        per_task[t] = {"desc": idx2desc[t], "split": "unseen", "n_train": 0, "n_eval": len(task_eps[t])}

    # In the COMBINED dataset (original + time-reversed), reversed episode e lives at index
    # n_original + e. The combined train set is the original train episodes plus their
    # reversed copies (forward + reverse goals from the same o_t). Eval stays forward-only.
    n_original = sum(len(v) for v in task_eps.values())

    return {
        "dataset": str(dataset_dir),
        "seed": seed,
        "holdout_frac": holdout_frac,
        "n_original_episodes": n_original,
        "seen_task_ids": seen,
        "unseen_task_ids": unseen,
        "train_episodes": sorted(train_eps),
        "combined_train_episodes": sorted(train_eps + [n_original + e for e in train_eps]),
        "seen_eval_episodes": sorted(seen_eval_eps),
        "unseen_eval_episodes": sorted(unseen_eval_eps),
        "per_task": {str(k): v for k, v in sorted(per_task.items())},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build a seen/unseen task split of a LeRobot LIBERO dataset.")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--unseen-task-ids", type=int, nargs="+", default=[7, 9])
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("/workspace/data/libero_goal_split.json"))
    args = p.parse_args()

    split = make_split(args.dataset_root, args.unseen_task_ids, args.holdout_frac, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(split, f, indent=2)

    print(f"seen tasks   ({len(split['seen_task_ids'])}): {split['seen_task_ids']}")
    print(f"unseen tasks ({len(split['unseen_task_ids'])}): {split['unseen_task_ids']}")
    print(f"train episodes:        {len(split['train_episodes'])}")
    print(f"seen-eval episodes:    {len(split['seen_eval_episodes'])}")
    print(f"unseen-eval episodes:  {len(split['unseen_eval_episodes'])}")
    print("\nper-task:")
    for k, v in split["per_task"].items():
        print(f"  task {k} [{v['split']:6s}] train={v['n_train']:3d} eval={v['n_eval']:3d}  {v['desc']!r}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
