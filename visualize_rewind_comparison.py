#!/usr/bin/env python
"""Side-by-side visual check of a LeRobot episode vs its ReWiND (time-reversed) copy.

Renders, for the same episode index, the ORIGINAL episode (left, playing forward) next to
the REWIND episode (right, playing forward = original frames in reverse). Each side shows
the front + wrist camera and a bar panel of the action vector. Use it to eyeball that:
  - the right side is the left side played backwards (REWIND frame i == ORIGINAL frame L-1-i),
  - the end-effector action bars on the right are negated/reversed,
  - the gripper bar (last action dim) is time-reversed but NOT flipped open<->close.

Example:
    uv run python visualize_rewind_comparison.py \
        --orig-root /workspace/data/libero_goal_image \
        --rewind-root /workspace/data/libero_goal_image_rewind \
        --episode-index 0 \
        --output /workspace/rewind_check/episode_0_compare.mp4
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def _resolve_dataset_dir(root: Path) -> Path:
    if (root / "meta" / "info.json").exists():
        return root
    candidates = sorted(p for p in root.iterdir() if (p / "meta" / "info.json").exists())
    if not candidates:
        raise FileNotFoundError(f"No LeRobot dataset found under {root}")
    return candidates[0]


def _any_data_file(dataset_dir: Path) -> Path:
    files = sorted((dataset_dir / "data").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir / 'data'}")
    return files[0]


def _load_episode_rows(dataset_dir: Path, episodes, episode_index: int) -> list[dict]:
    """Load only the rows of one episode from its data file (avoids whole-dataset concat)."""
    if episode_index not in set(episodes["episode_index"].tolist()):
        raise ValueError(f"Episode {episode_index} not found in {dataset_dir}.")
    ep = episodes.loc[episodes["episode_index"] == episode_index].iloc[0]
    chunk, file_idx = int(ep["data/chunk_index"]), int(ep["data/file_index"])
    path = dataset_dir / "data" / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.parquet"
    table = pq.read_table(path)
    ep_col = np.asarray(table.column("episode_index").to_pylist())
    rows = np.nonzero(ep_col == episode_index)[0]
    return table.slice(int(rows[0]), len(rows)).to_pylist()


def _detect_image_keys(schema: pa.Schema) -> tuple[str, str]:
    """Return (front_key, wrist_key) image columns (struct<bytes, path>)."""
    img_keys = [
        f.name for f in schema if f.name.startswith("observation.image") and pa.types.is_struct(f.type)
    ]
    if not img_keys:
        raise ValueError("No image columns (observation.image*) found.")
    wrist = [k for k in img_keys if "wrist" in k or "eye_in_hand" in k]
    front = [k for k in img_keys if k not in wrist]
    front_key = front[0] if front else img_keys[0]
    wrist_key = wrist[0] if wrist else (img_keys[1] if len(img_keys) > 1 else img_keys[0])
    return front_key, wrist_key


def _image_from_cell(cell: dict) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


def _draw_action_panel(action: np.ndarray, width: int, height: int = 170) -> np.ndarray:
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(panel, "action", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    left, right = 132, width - 16
    bar_w = max(12, right - left)
    row_h = max(15, (height - 34) // len(action))
    center_x = left + bar_w // 2
    cv2.line(panel, (center_x, 32), (center_x, height - 8), (160, 160, 160), 1)
    labels = [f"a{i}" for i in range(len(action))]
    labels[-1] = "grip"
    for i, value in enumerate(action):
        y = 40 + i * row_h
        v = float(np.clip(value, -1.0, 1.0))
        end_x = int(center_x + v * (bar_w // 2))
        color = (40, 120, 230) if v >= 0 else (230, 90, 40)
        cv2.putText(panel, f"{labels[i]}:{value:+.2f}", (8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (35, 35, 35), 1)
        cv2.rectangle(panel, (min(center_x, end_x), y - 7), (max(center_x, end_x), y + 6), color, -1)
    return panel


def _make_side(row: dict, keys: tuple[str, str], title: str, subtitle: str, action_scale: float) -> np.ndarray:
    front = _image_from_cell(row[keys[0]])
    wrist = _image_from_cell(row[keys[1]])
    if front.shape != wrist.shape:
        wrist = cv2.resize(wrist, (front.shape[1], front.shape[0]), interpolation=cv2.INTER_AREA)
    top = cv2.cvtColor(np.concatenate([front, wrist], axis=1), cv2.COLOR_RGB2BGR)
    cv2.putText(top, "front", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(top, "wrist", (front.shape[1] + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(top, title, (10, top.shape[0] - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 230, 90), 2)
    cv2.putText(top, subtitle, (10, top.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    action = np.asarray(row["action"], dtype=np.float32) / action_scale
    panel = _draw_action_panel(action, width=top.shape[1])
    return np.concatenate([top, panel], axis=0)


def visualize(
    orig_root: Path,
    rewind_root: Path,
    episode_index: int,
    output: Path,
    action_scale: float,
    fps: float,
) -> Path:
    orig_dir = _resolve_dataset_dir(orig_root)
    rew_dir = _resolve_dataset_dir(rewind_root)
    keys = _detect_image_keys(pq.read_schema(_any_data_file(orig_dir)))
    o_eps = pq.read_table(sorted((orig_dir / "meta" / "episodes").glob("*/*.parquet"))).to_pandas()
    r_eps = pq.read_table(sorted((rew_dir / "meta" / "episodes").glob("*/*.parquet"))).to_pandas()

    o_rows = _load_episode_rows(orig_dir, o_eps, episode_index)
    r_rows = _load_episode_rows(rew_dir, r_eps, episode_index)
    length = min(len(o_rows), len(r_rows))
    print(f"image keys: {keys} | episode {episode_index} length: orig={len(o_rows)} rewind={len(r_rows)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    divider = None
    for i in range(length):
        left = _make_side(
            o_rows[i], keys, "ORIGINAL (forward)", f"frame {i} / {length - 1}", action_scale
        )
        right = _make_side(
            r_rows[i], keys, "REWIND (reverse)", f"frame {i}  == original frame {length - 1 - i}", action_scale
        )
        if divider is None:
            divider = np.full((left.shape[0], 6, 3), 30, dtype=np.uint8)
        frame = np.concatenate([left, divider, right], axis=1)
        if writer is None:
            writer = cv2.VideoWriter(
                str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0])
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter for {output}")
        writer.write(frame)
    writer.release()
    return output


def main() -> None:
    p = argparse.ArgumentParser(description="Side-by-side ORIGINAL vs REWIND episode visualization.")
    p.add_argument("--orig-root", type=Path, required=True)
    p.add_argument("--rewind-root", type=Path, required=True)
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--fps", type=float, default=10.0)
    args = p.parse_args()
    output = args.output or Path("/workspace/rewind_check") / f"episode_{args.episode_index:06d}_compare.mp4"
    out = visualize(
        args.orig_root, args.rewind_root, args.episode_index, output, args.action_scale, args.fps
    )
    print(out)


if __name__ == "__main__":
    main()
