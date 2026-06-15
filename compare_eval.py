#!/usr/bin/env python
"""Aggregate rollout_eval results into a baseline-vs-rewind comparison table.

Reads results_<variant>.json (a list of per-demo dicts written by rollout_eval.py) and
reports, per (split, task) cell and overall, the two headline stats the user cares about:
  - success rate (mean of `success`)
  - eef reach (mean of `mean_reach_pos` = ||robot eef pos - goal-frame eef pos|| over the
    rollout; lower = better goal-following). reach_rot / reach_grip shown too.

Usage:
  python compare_eval.py [--dir /workspace/rollout_eval_out_fixedmap] [--variants baseline rewind]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(d: Path, variant: str):
    f = d / f"results_{variant}.json"
    if not f.exists():
        return None
    return json.load(open(f))


def agg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/rollout_eval_out_fixedmap")
    ap.add_argument("--variants", nargs="+", default=["baseline", "rewind"])
    args = ap.parse_args()
    d = Path(args.dir)

    data = {v: load(d, v) for v in args.variants}
    present = [v for v in args.variants if data[v]]
    if not present:
        print(f"no results found in {d} for {args.variants}")
        return

    # cell -> variant -> rows
    cells: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for v in present:
        for r in data[v]:
            cells[(r["split"], r["task_id"])][v].append(r)

    name = {}
    for v in present:
        for r in data[v]:
            name[r["task_id"]] = Path(r["video"]).stem.split("_task")[-1]  # e.g. "0_put_the_bowl..."

    cols = present
    w = 24
    print(f"\n{'cell':>16} | {'n':>2} | " + " | ".join(f"{c+' succ':>12} {c+' reachP':>13}" for c in cols))
    print("-" * (24 + len(cols) * 28))

    def row(label, getter_rows):
        parts = []
        n = 0
        for c in cols:
            rows = getter_rows(c)
            n = max(n, len(rows))
            sr = 100 * agg(rows, "success")
            rp = agg(rows, "mean_reach_pos")
            parts.append(f"{sr:>11.0f}% {rp:>13.3f}")
        print(f"{label:>16} | {n:>2} | " + " | ".join(parts))

    order = sorted(cells, key=lambda c: (c[0] != "seen", c[1]))
    last_split = None
    for (split, tid) in order:
        if split != last_split:
            last_split = split
        row(f"{split}/task{tid:02d}", lambda c, s=split, t=tid: cells[(s, t)][c])

    print("-" * (24 + len(cols) * 28))
    # split-level + overall means (averaged over demos, not cells)
    for split in ["seen", "unseen"]:
        row(f"** {split} ALL", lambda c, s=split: [r for r in data[c] if r["split"] == s])
    row("** OVERALL", lambda c: list(data[c]))

    # secondary table: reach rot / grip (overall + per split)
    print(f"\n{'reach detail':>16} | " + " | ".join(f"{c+' rot':>10} {c+' grip':>11}" for c in cols))
    print("-" * (18 + len(cols) * 24))
    for split in ["seen", "unseen"]:
        parts = []
        for c in cols:
            rows = [r for r in data[c] if r["split"] == split]
            parts.append(f"{agg(rows,'mean_reach_rot'):>10.3f} {agg(rows,'mean_reach_grip'):>11.3f}")
        print(f"{'** '+split:>16} | " + " | ".join(parts))


if __name__ == "__main__":
    main()
