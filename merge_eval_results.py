#!/usr/bin/env python
"""Merge per-task results_<variant>.json (from rollout_eval_isolated.sh) into one file.

Each isolated task run writes <dir>/task<t>/results_<variant>.json (a list of per-demo dicts).
This concatenates them into <dir>/results_<variant>.json so compare_eval.py / downstream tooling
see the same single-file layout a one-shot rollout_eval.py run would have produced.
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--variant", default="rewind")
    args = ap.parse_args()

    parts = sorted(glob.glob(os.path.join(args.dir, "task*", f"results_{args.variant}.json")))
    merged = []
    for p in parts:
        rows = json.load(open(p))
        merged.extend(rows)
        print(f"  + {p}: {len(rows)} rows")

    out = os.path.join(args.dir, f"results_{args.variant}.json")
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"merged {len(merged)} rows from {len(parts)} task files -> {out}")

    # quick summary so the merge is self-checking
    by = defaultdict(lambda: {"n": 0, "succ": 0})
    for r in merged:
        c = by[f"{r['split']}/task{r['task_id']:02d}"]
        c["n"] += 1
        c["succ"] += int(r["success"])
    for k in sorted(by):
        c = by[k]
        print(f"  {k:>16} | n={c['n']:>2} | success={100*c['succ']/c['n']:5.1f}%")


if __name__ == "__main__":
    main()
