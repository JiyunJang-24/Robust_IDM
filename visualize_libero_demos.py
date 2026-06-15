#!/usr/bin/env python
"""Visualize one demonstration episode per LIBERO task as an mp4.

For each task in the dataset it renders one demo episode — agentview + wrist cameras side by
side with an action-bar overlay — to ``<out>/task<id>_<slug>.mp4``. Useful for inspecting the
demonstrated behavior of each task (e.g. comparing an unseen-task policy rollout against the
behaviors present in the seen training tasks).

Usage:
    uv run python visualize_libero_demos.py \
        --dataset-root /workspace/data/libero_goal_image \
        --out /workspace/libero_goal_visualize --which 0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import write_video

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit("opencv (cv2) is required.") from e

PANEL = 256
LABEL_H = 28
ACT_H = 96
ACT_NAMES = ["dx", "dy", "dz", "dRx", "dRy", "dRz", "grip"]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def to_uint8_hwc(img_chw_float: torch.Tensor) -> np.ndarray:
    return (img_chw_float.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()


def task_map(ds: LeRobotDataset):
    tt = pq.read_table(Path(ds.root) / "meta" / "tasks.parquet")
    desc2i = dict(zip(tt.column("task").to_pylist(), tt.column("task_index").to_pylist()))
    i2desc = {v: k for k, v in desc2i.items()}
    ep = ds.meta.episodes
    ep_task = {}
    for e in range(ds.meta.total_episodes):
        tasks = ep["tasks"][e]
        desc = tasks[0] if hasattr(tasks, "__len__") and not isinstance(tasks, str) else tasks
        ep_task[e] = int(desc2i[desc])
    return ep_task, i2desc


def _action_bars(action: np.ndarray, width: int) -> np.ndarray:
    panel = np.full((ACT_H, width, 3), 22, dtype=np.uint8)
    n = len(action)
    row_h = ACT_H // n
    cx = width // 2
    cv2.line(panel, (cx, 0), (cx, ACT_H), (70, 70, 70), 1)
    for i, v in enumerate(action):
        y = i * row_h + row_h // 2
        val = float(np.clip(v, -1.0, 1.0))
        length = int(abs(val) * (width // 2 - 60))
        x2 = cx + length if val >= 0 else cx - length
        color = (90, 200, 90) if val >= 0 else (90, 90, 220)
        cv2.line(panel, (cx, y), (x2, y), color, max(2, row_h // 3))
        cv2.putText(panel, f"{ACT_NAMES[i]}:{v:+.2f}", (6, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (210, 210, 210), 1, cv2.LINE_AA)
    return panel


def compose(agent: np.ndarray, wrist: np.ndarray, action: np.ndarray, *, title: str, t: int, T: int) -> np.ndarray:
    if wrist.shape[:2] != (PANEL, PANEL):
        wrist = cv2.resize(wrist, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
    div = np.full((PANEL, 6, 3), 15, dtype=np.uint8)
    cams = np.concatenate([agent, div, wrist], axis=1)  # (256, 518, 3)
    width = cams.shape[1]
    header = np.full((LABEL_H, width, 3), 30, dtype=np.uint8)
    cv2.putText(header, f"{title}   frame {t}/{T - 1}", (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(cams, "agentview", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(cams, "wrist", (PANEL + 14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    bars = _action_bars(action, width)
    frame = np.concatenate([header, cams, bars], axis=0)
    h, w = frame.shape[:2]
    return frame[: h - (h % 2), : w - (w % 2)]


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize one demo episode per LIBERO task.")
    p.add_argument("--dataset-root", default="/workspace/data/libero_goal_image")
    p.add_argument("--repo-id", default="lerobot/libero_goal_image")
    p.add_argument("--out", type=Path, default=Path("/workspace/libero_goal_visualize"))
    p.add_argument("--which", type=int, default=0, help="which episode (sorted) of each task to render")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--tasks", type=int, nargs="*", default=None, help="restrict to these task ids")
    args = p.parse_args()

    ds = LeRobotDataset(args.repo_id, root=args.dataset_root)
    ep_task, i2desc = task_map(ds)
    task_eps: dict[int, list[int]] = {}
    for e, t in ep_task.items():
        task_eps.setdefault(t, []).append(e)

    args.out.mkdir(parents=True, exist_ok=True)
    img_key, wrist_key = "observation.images.image", "observation.images.wrist_image"
    targets = args.tasks if args.tasks is not None else sorted(task_eps)
    for t in targets:
        eps = sorted(task_eps[t])
        ep = eps[min(args.which, len(eps) - 1)]
        desc = i2desc[t]
        fr = int(ds.meta.episodes["dataset_from_index"][ep])
        to = int(ds.meta.episodes["dataset_to_index"][ep])
        frames = []
        for k in range(fr, to):
            item = ds[k]
            agent = to_uint8_hwc(item[img_key])
            wrist = to_uint8_hwc(item[wrist_key])
            frames.append(compose(agent, wrist, item[ACTION].numpy(),
                                  title=f"task{t}: {desc}", t=k - fr, T=to - fr))
        out_path = args.out / f"task{t}_{slug(desc)}.mp4"
        write_video(str(out_path), frames, args.fps)
        print(f"task {t} ep{ep} ({to - fr} frames) -> {out_path.name}  [{desc}]")
    print(f"\nwrote {len(targets)} videos to {args.out}")


if __name__ == "__main__":
    main()
