#!/usr/bin/env python
"""Single source of truth for dataset task_index <-> LIBERO benchmark task_id.

WHY THIS EXISTS
---------------
A LeRobot dataset numbers its tasks (`observation` `task_index`) in whatever order
they were discovered during HDF5->LeRobot conversion. The LIBERO *benchmark*
(`benchmark.get_benchmark_dict()[suite]().tasks`) numbers them in its own fixed
order. THESE TWO ORDERS ARE NOT THE SAME (for libero_goal they are almost reversed).

So a numeric task id is NOT a portable identifier between the dataset and the env.
If you feed a dataset `task_index` straight into `LiberoEnv(task_ids=[...])` you build
the WRONG task's env (wrong BDDL goal + wrong init states) and success can never fire.

The only stable identifier across the dataset and the benchmark is the task's
**language string**. This module maps between the two id spaces by matching language,
for ANY suite, with a 1:1 sanity check so a mismatch fails loudly instead of silently
evaluating the wrong task. Use it everywhere an env is built from a dataset episode.

USAGE
-----
    from libero_task_map import dataset_to_benchmark_map
    ds2bench = dataset_to_benchmark_map(ds.meta, "libero_goal")
    env_cfg = LiberoEnv(task="libero_goal", task_ids=[ds2bench[dataset_task_index]])
"""

from __future__ import annotations

import re


def _norm(text: str) -> str:
    """Normalize a task language string for robust matching (case/space-insensitive)."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def benchmark_languages(suite_name: str) -> dict[str, int]:
    """{normalized language -> benchmark task_id} for a LIBERO suite."""
    from libero.libero import benchmark  # lazy: heavy import, needs the lerobot venv

    suite = benchmark.get_benchmark_dict()[suite_name]()
    out: dict[str, int] = {}
    for i, task in enumerate(suite.tasks):
        out[_norm(task.language)] = i
    return out


def dataset_languages(ds_meta) -> dict[int, str]:
    """{dataset task_index -> normalized language} from a LeRobotDatasetMetadata."""
    out: dict[int, str] = {}
    for lang, row in ds_meta.tasks.iterrows():  # index is the language string
        out[int(row["task_index"])] = _norm(lang)
    return out


def dataset_to_benchmark_map(ds_meta, suite_name: str) -> dict[int, int]:
    """Map every dataset task_index to its LIBERO benchmark task_id by language.

    Raises ValueError if any dataset task has no language match in the suite, or if
    the resulting map is not 1:1 (which would mean two dataset tasks collide onto one
    benchmark task — a sign the dataset/suite pairing is wrong).
    """
    bench_by_lang = benchmark_languages(suite_name)
    ds_lang = dataset_languages(ds_meta)

    mapping: dict[int, int] = {}
    missing: list[tuple[int, str]] = []
    for ds_idx, lang in ds_lang.items():
        if lang in bench_by_lang:
            mapping[ds_idx] = bench_by_lang[lang]
        else:
            missing.append((ds_idx, lang))

    if missing:
        raise ValueError(
            f"{len(missing)} dataset task(s) have no language match in benchmark suite "
            f"'{suite_name}': {missing}. Available benchmark languages: {sorted(bench_by_lang)}"
        )
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(
            f"dataset->benchmark map for '{suite_name}' is not 1:1: {mapping}. "
            "Two dataset tasks matched the same benchmark task — wrong suite for this dataset?"
        )
    return mapping


if __name__ == "__main__":
    # Self-test / inspection: print the map for a dataset.
    import argparse
    import sys

    sys.path.insert(0, "/workspace/lerobot/src")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="libero_goal_image")
    ap.add_argument("--dataset-root", default="/workspace/data/libero_goal_image")
    ap.add_argument("--suite", default="libero_goal")
    args = ap.parse_args()

    ds = LeRobotDataset(args.repo_id, root=args.dataset_root)
    m = dataset_to_benchmark_map(ds.meta, args.suite)
    print(f"dataset '{args.repo_id}' x suite '{args.suite}':")
    ds_lang = dataset_languages(ds.meta)
    for k in sorted(m):
        print(f"  dataset {k:>2} -> benchmark {m[k]:>2}  | {ds_lang[k]}")
