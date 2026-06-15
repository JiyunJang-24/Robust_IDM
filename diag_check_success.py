#!/usr/bin/env python
"""Decisive diagnosis of the LIBERO check_success=0 problem.

Replays a demo's GROUND-TRUTH actions in the env (same init_state_id pinning as
rollout_eval) and, every step, prints:
  - the goal predicate object's joint qpos (e.g. the drawer slide) + its is_open threshold,
  - check_success(),
so we can tell apart THREE hypotheses:
  (A) init mismatch  : env starts from a different layout than the demo (reset img != demo[0]),
                       so the GT actions don't actually open the drawer.
  (B) genuine miss   : drawer qpos never crosses the is_open threshold.
  (C) judge bug      : qpos crosses threshold / state matches demo, yet check_success stays False.

Run from /workspace/lerobot via uv (needs MUJOCO_GL=egl, MUJOCO_EGL_DEVICE_ID=0, GPU0).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/lerobot/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.factory import make_env
from lerobot.envs.configs import LiberoEnv
from libero_task_map import dataset_to_benchmark_map, dataset_languages

ACTION = "action"
OBS_STATE = "observation.state"


def episode_task_map(ds):
    fr = ds.meta.episodes["dataset_from_index"]
    tasks = ds.hf_dataset["task_index"]
    out = {}
    for e in range(ds.meta.total_episodes):
        out[e] = int(tasks[int(fr[e])])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/workspace/data/libero_goal_image")
    ap.add_argument("--repo-id", default="libero_goal_image")
    ap.add_argument("--split", default="/workspace/data/libero_goal_split.json")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--ep-index", type=int, default=None, help="explicit episode; else first eval demo of task")
    ap.add_argument("--seed", type=int, default=10000)
    args = ap.parse_args()

    ds = LeRobotDataset(args.repo_id, root=args.dataset_root)
    ep_task = episode_task_map(ds)

    # within-task ordinal -> init_state index (mirror rollout_eval.py)
    task_eps = {}
    for e in range(ds.meta.total_episodes):
        task_eps.setdefault(ep_task[e], []).append(e)
    for tk in task_eps:
        task_eps[tk].sort()
    ordinal_of = {e: task_eps[ep_task[e]].index(e) for e in range(ds.meta.total_episodes)}

    # CRITICAL: dataset task_index is NOT the LIBERO benchmark task_id (they were
    # serialized in different orders). Map by language via the shared helper (1:1 checked).
    ds2bench = dataset_to_benchmark_map(ds.meta, "libero_goal")
    ds_lang = dataset_languages(ds.meta)
    print("dataset task_index -> benchmark task_id:", ds2bench)

    split = json.load(open(args.split))
    seen_eval = split.get("seen_eval", [])
    unseen_eval = split.get("unseen_eval", [])
    eval_eps = [e for e in seen_eval + unseen_eval if ep_task[e] == args.task_id]
    eval_eps.sort()

    ep_index = args.ep_index if args.ep_index is not None else (eval_eps[0] if eval_eps else task_eps[args.task_id][0])
    ordinal = ordinal_of[ep_index]
    print(f"task_id={args.task_id} ep_index={ep_index} ordinal={ordinal}  (eval demos for task: {eval_eps})")

    # --- load GT demo (images + actions) ---
    fr = int(ds.meta.episodes["dataset_from_index"][ep_index])
    to = int(ds.meta.episodes["dataset_to_index"][ep_index])
    T = to - fr
    img_key = "observation.images.image"
    demo_actions, demo_imgs, demo_states = [], [], []
    for k in range(fr, to):
        item = ds[k]
        demo_actions.append(item[ACTION].float().numpy())
        demo_states.append(item[OBS_STATE].float().numpy())
        img = item[img_key]
        if hasattr(img, "numpy"):
            img = img.numpy()
        # CHW float [0,1] -> HWC uint8
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.max() <= 1.01:
            img = (img * 255).clip(0, 255)
        demo_imgs.append(img.astype(np.uint8))
    demo_actions = np.stack(demo_actions)
    print(f"demo: T={T} action_shape={demo_actions.shape} state_shape={np.stack(demo_states).shape}")

    # --- build env with the CORRECT benchmark task_id, pin init_state to the demo ordinal ---
    bench_id = ds2bench[args.task_id]
    print(f"dataset task_index {args.task_id} ('{ds_lang[args.task_id]}') -> benchmark task_id {bench_id}")
    env_cfg = LiberoEnv(
        task="libero_goal", task_ids=[bench_id],
        obs_type="pixels_agent_pos", observation_height=256, observation_width=256,
    )
    suite = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = next(iter(suite["libero_goal"].values()))
    inner = env.envs[0].unwrapped
    inner.init_state_id = ordinal

    obs, _ = env.reset(seed=[args.seed])

    # underlying problem env (has parsed_problem, object_states_dict, sim)
    prob = inner._env.env  # OffScreenRenderEnv.env -> problem domain
    goal_state = prob.parsed_problem["goal_state"]
    print("GOAL STATE:", goal_state)

    # reset-image vs demo[0] image: init consistency check
    reset_img = env.call("render")[0]
    mse0 = float(np.mean((reset_img.astype(np.float32) - demo_imgs[0].astype(np.float32)) ** 2))
    print(f"[init check] reset_img vs demo[0] MSE = {mse0:.1f}  (low => same layout)")

    # identify the goal object
    obj_name = goal_state[0][1]
    obj_state = prob.object_states_dict[obj_name]

    def drawer_qpos():
        joints = getattr(obj_state, "joints", None)
        if not joints:
            return []
        qs = []
        for joint in joints:
            addr = prob.sim.model.get_joint_qpos_addr(joint)
            qs.append(float(prob.sim.data.qpos[addr]))
        return qs

    print(f"goal object={obj_name} type={type(obj_state).__name__} joints={getattr(obj_state,'joints',None)}")

    print(f"\n{'step':>4} {'drawer_qpos':>22} {'is_open':>8} {'check_success':>14}")
    ever = False
    q_max = -1e9
    for t in range(T):
        a = demo_actions[t][None]  # (1, action_dim) for the SyncVectorEnv
        obs, reward, terminated, truncated, infos = env.step(a)
        info = {k: (v[0] if hasattr(v, "__len__") and not isinstance(v, str) else v) for k, v in infos.items()}
        # NOTE: LiberoEnv.step auto-resets on terminated; to inspect raw sim we read BEFORE any reset.
        # But auto-reset already happened if terminated; so read success from info instead.
        qpos = drawer_qpos()
        q_max = max(q_max, max(qpos) if qpos else q_max)
        is_open = obj_state.is_open()
        succ = bool(info.get("is_success", False))
        ever = ever or succ
        if t < 5 or t >= T - 12 or succ:
            print(f"{t:>4} {str([round(q,4) for q in qpos]):>22} {str(is_open):>8} {str(succ):>14}"
                  + ("   <== auto-reset (terminated)" if terminated else ""))
        if terminated:
            print(f"  *** terminated at step {t}: is_success={succ} ***")
            break
    print(f"\nSUMMARY: ever_success={ever}  max_drawer_qpos={q_max:.4f}  init_mse={mse0:.1f}")
    env.close()


if __name__ == "__main__":
    main()
