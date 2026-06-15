#!/usr/bin/env python
"""Closed-loop 3-panel rollout eval for the goal-conditioned IDM (diffusion_idm) on LIBERO_GOAL.

For each (variant, task, reference-demo) cell this runs a closed-loop LIBERO rollout where the
goal image is supplied open-loop from the reference demo: at env step ``t`` the policy is
conditioned on ``goal = demo_img[min(t + H, T-1)]`` (the "demo clock"). Both variants therefore
receive the *identical* goal sequence, so any behavioral difference is attributable to the policy.

It writes a synchronized 3-panel mp4 per cell:

    [ O_t  (policy input obs) | O_{t+H} (goal input) | env rollout execution ]

Panels 1/2 freeze between policy re-plans (every ``n_action_steps``); panel 3 updates every step.
An overlay shows step ``t``, demo index ``g``, the open-loop goal-tracking error
``||state_robot_t - demo_state[min(t,T-1)]||`` (which makes the feared drift *measurable*), the
executed action bars, and SUCCESS/FAIL. A per-cell success flag and the tracking-error series are
recorded; ``summary.json`` / ``summary.csv`` aggregate success rate and mean tracking error per
(variant, seen|unseen, task_id).

The official rollout path is mirrored exactly (preprocess_observation -> env_preprocessor ->
preprocessor -> select_action -> postprocessor -> env_postprocessor -> env.step); the only addition
is injecting the (manually MEAN_STD-normalized) goal under OBS_GOAL_IMAGES between the policy
preprocessor and select_action. Input normalization and action un-normalization come from the
processor pipelines saved in the checkpoint.

Usage (see scripts/rollout_eval.sh):
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
    uv run python rollout_eval.py --variant rewind \
        --ckpt /workspace/lerobot/outputs/idm_rewind/checkpoints/last/pretrained_model \
        --dataset-root /workspace/data/libero_goal_image \
        --split /workspace/data/libero_goal_split.json \
        --out /workspace/rollout_eval_out --n-demos 1 --H 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

# MuJoCo EGL must be configured before the LIBERO env is imported/created. This system's EGL
# exposes a single device (id 0); the visible CUDA GPU is selected via CUDA_VISIBLE_DEVICES.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.envs.configs import LiberoEnv  # noqa: E402
from lerobot.envs.factory import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.diffusion_idm.modeling_diffusion_idm import (  # noqa: E402
    OBS_GOAL_IMAGES,
    DiffusionIDMPolicy,
)
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402
from lerobot.utils.io_utils import write_video  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from libero_task_map import dataset_to_benchmark_map  # noqa: E402

# Hang diagnosis: set ROLLOUT_FAULTHANDLER=<seconds> to dump every thread's Python stack
# every N seconds. If a rollout hangs, the same frame prints repeatedly -> that's the hang site.
if os.environ.get("ROLLOUT_FAULTHANDLER"):
    import faulthandler

    faulthandler.dump_traceback_later(int(os.environ["ROLLOUT_FAULTHANDLER"]), repeat=True)

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit("opencv (cv2) is required for the rollout video overlay.") from e


# --------------------------------------------------------------------------------------
# Normalization (goal image only — the saved preprocessor ignores OBS_GOAL_IMAGES).
# Mirrors goal_influence_probe.Normalizer: images use MEAN_STD from dataset stats.
# --------------------------------------------------------------------------------------
class GoalNormalizer:
    def __init__(self, ds: LeRobotDataset, image_keys: list[str]):
        s = ds.meta.stats
        self.img = {
            k: (
                torch.tensor(np.asarray(s[k]["mean"], dtype=np.float32)).reshape(3, 1, 1),
                torch.tensor(np.asarray(s[k]["std"], dtype=np.float32)).reshape(3, 1, 1),
            )
            for k in image_keys
        }

    def image(self, x: torch.Tensor, key: str) -> torch.Tensor:
        m, sd = self.img[key]
        return (x - m) / (sd + 1e-8)


# --------------------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------------------
def episode_task_map(ds: LeRobotDataset) -> dict[int, int]:
    """episode_index -> task_index (every frame in an episode shares the same task)."""
    ep = ds.meta.episodes
    tt = pq.read_table(Path(ds.root) / "meta" / "tasks.parquet")
    desc2i = dict(zip(tt.column("task").to_pylist(), tt.column("task_index").to_pylist()))
    out = {}
    for e in range(ds.meta.total_episodes):
        tasks = ep["tasks"][e]
        desc = tasks[0] if hasattr(tasks, "__len__") and not isinstance(tasks, str) else tasks
        out[e] = int(desc2i[desc])
    return out


def to_uint8_hwc(img_chw_float: torch.Tensor) -> np.ndarray:
    """(3,H,W) float in [0,1] (dataset / policy-input orientation) -> (H,W,3) uint8 RGB."""
    return (img_chw_float.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()


def slug(text: str) -> str:
    """Filesystem-safe slug of a task description, e.g. 'put the bowl on the plate' -> 'put_the_bowl_on_the_plate'."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# --------------------------------------------------------------------------------------
# Video composition
# --------------------------------------------------------------------------------------
PANEL = 256
DIV = 6
LABEL_H = 26
ACT_H = 96
GRIPPER_DIM = 6  # last action dim is the gripper
ACT_NAMES = ["dx", "dy", "dz", "dRx", "dRy", "dRz", "grip"]
INPUT_COLOR = (235, 200, 80)   # frozen policy inputs (O_t, O_t+H) — cyan/gold border
EXEC_COLOR = (245, 150, 50)    # live env execution — orange border


def _label_strip(width: int, text: str, color=(235, 235, 235)) -> np.ndarray:
    strip = np.full((LABEL_H, width, 3), 30, dtype=np.uint8)
    cv2.putText(strip, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return strip


def _panel(img_rgb: np.ndarray, label: str, border=None) -> np.ndarray:
    """Stack a label strip above a 256x256 RGB image, with an optional colored border."""
    if img_rgb.shape[:2] != (PANEL, PANEL):
        img_rgb = cv2.resize(img_rgb, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
    else:
        img_rgb = img_rgb.copy()
    if border is not None:
        cv2.rectangle(img_rgb, (0, 0), (PANEL - 1, PANEL - 1), border, 4)
    return np.concatenate([_label_strip(PANEL, label, border or (235, 235, 235)), img_rgb], axis=0)


def _attn_heatmap(pool, feat) -> np.ndarray:
    """SpatialSoftmax keypoint attention (mean over the 32 keypoints) as (Hf,Wf) in [0,1].

    `feat` is the feature map fed INTO the encoder's SpatialSoftmax pool (captured by a hook).
    The pool's 1x1 conv (`pool.nets`) maps it to K keypoint channels, then a spatial softmax over
    each channel gives where that keypoint attends. We average the K attention maps to get one
    "where the encoder looks" map for this image — the same op the model uses to read the image.
    """
    import torch.nn.functional as F  # noqa: PLC0415

    with torch.no_grad():
        feat = feat.clone()  # select_action runs under inference_mode; clone -> normal tensor
        kp = pool.nets(feat) if getattr(pool, "nets", None) is not None else feat
        b, k, hf, wf = kp.shape
        att = F.softmax(kp.reshape(b * k, hf * wf), dim=-1).reshape(b, k, hf, wf)
        h = att[0].mean(0).float().cpu().numpy()
    return h / (h.max() + 1e-8)


def _overlay_attn(panel_rgb: np.ndarray, heat: np.ndarray, flip: bool = False) -> np.ndarray:
    """Blend a JET heatmap (resized to the panel) onto an RGB panel."""
    if flip:  # env render() flips H,W vs the raw obs the encoder saw; align the heatmap to the panel
        heat = heat[::-1, ::-1]
    h = cv2.resize(heat.astype(np.float32), (panel_rgb.shape[1], panel_rgb.shape[0]))
    cm = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]  # BGR->RGB
    return (0.55 * panel_rgb.astype(np.float32) + 0.45 * cm.astype(np.float32)).clip(0, 255).astype(np.uint8)


def _input_saliency(policy, ob, image_keys, goal, n_steps=10):
    """Gradient saliency of the predicted action chunk w.r.t. the o_t image and the goal image.

    Returns (o_sal, g_sal) spatial maps (H,W) for camera 0, on a COMMON absolute scale (raw
    |grad| magnitudes), so brighter == that input drives the action more. This is decision-level:
    it flows through encoder -> U-Net -> action, so a region only lights up if it actually changes
    the output. Saliency uses a cheaper N-step denoising (memory + speed) than the rollout itself.
    """
    diff = policy.diffusion
    old_steps = diff.num_inference_steps
    diff.num_inference_steps = n_steps
    try:
        with torch.enable_grad():
            # ob carries single-timestep tensors (B, ...); generate_actions wants the n_obs dim (B, s, ...)
            state = ob[OBS_STATE].detach()
            state = state.unsqueeze(1) if state.ndim == 2 else state  # (B,1,sdim)
            imgs = [(ob[k].detach().unsqueeze(1) if ob[k].ndim == 4 else ob[k].detach()) for k in image_keys]
            obs_img = torch.stack(imgs, dim=-4).clone().requires_grad_(True)  # (B,1,n_cam,3,H,W)
            goal_img = goal.detach().clone().requires_grad_(True)  # (B,n_cam,3,H,W)
            batch = {OBS_STATE: state, OBS_IMAGES: obs_img, OBS_GOAL_IMAGES: goal_img}
            actions = diff.generate_actions(batch, noise=None)
            actions.float().pow(2).sum().sqrt().backward()
        # obs_img: (B, s, n_cam, 3, H, W); goal_img: (B, n_cam, 3, H, W). camera 0, sum |grad| over
        # the obs-history (s) and RGB channels to get one (H,W) map per input.
        o_sal = obs_img.grad[0, :, 0].abs().sum(dim=(0, 1)).float().cpu().numpy()
        g_sal = goal_img.grad[0, 0].abs().sum(dim=0).float().cpu().numpy()
    finally:
        diff.num_inference_steps = old_steps
    # smooth the per-pixel gradient noise so the maps are readable (common sigma, no rescaling)
    o_sal = cv2.GaussianBlur(o_sal, (0, 0), sigmaX=4)
    g_sal = cv2.GaussianBlur(g_sal, (0, 0), sigmaX=4)
    return o_sal, g_sal


def _share_label(panel: np.ndarray, text: str, dominant: bool) -> None:
    """Draw a saliency-share readout at the bottom of a panel (in place). Bright green when that
    input dominates (share >= 50%), dim gray otherwise."""
    color = (90, 240, 120) if dominant else (170, 170, 170)
    h = panel.shape[0]
    cv2.rectangle(panel, (0, h - 22), (96, h), (20, 20, 20), -1)
    cv2.putText(panel, text, (5, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def _action_bars(action: np.ndarray, width: int, gt_action: np.ndarray | None = None) -> np.ndarray:
    """Horizontal signed bars for the 7-d action (filled = model). If ``gt_action`` is given,
    the demo GT action is overlaid as a hollow caret '|' marker so model vs GT can be compared."""
    panel = np.full((ACT_H, width, 3), 22, dtype=np.uint8)
    n = len(action)
    row_h = ACT_H // n
    cx = width // 2
    half = width // 2 - 60
    cv2.line(panel, (cx, 0), (cx, ACT_H), (70, 70, 70), 1)
    for i, v in enumerate(action):
        y = i * row_h + row_h // 2
        val = float(np.clip(v, -1.0, 1.0))
        length = int(abs(val) * half)
        x2 = cx + length if val >= 0 else cx - length
        color = (90, 200, 90) if val >= 0 else (90, 90, 220)
        cv2.line(panel, (cx, y), (x2, y), color, max(2, row_h // 3))
        label = f"{ACT_NAMES[i]}:{v:+.2f}"
        if gt_action is not None:
            gv = float(np.clip(gt_action[i], -1.0, 1.0))
            gx = cx + int(gv * half)
            cv2.line(panel, (gx, y - row_h // 3), (gx, y + row_h // 3), (250, 230, 90), 1, cv2.LINE_AA)  # GT marker
            label += f" (gt{gt_action[i]:+.2f})"
        cv2.putText(panel, label, (6, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (210, 210, 210), 1, cv2.LINE_AA)
    return panel


def compose_frame(panel1_rgb, panel2_rgb, exec_rgb, *, t, g, err, action, status,
                  query_k, chunk_j, chunk_len, demo_T, gt_action=None, err_pos=None, err_rot=None,
                  reach_pos=None, reach_rot=None, reach_grip=None) -> np.ndarray:
    """Build one [O_t | O_t+H | exec] RGB frame with overlays. Fixed output size.

    Left two panels are the FROZEN policy inputs for the current query (re-plan); they update
    only when the policy re-plans. The right panel is the LIVE env execution of that query's
    action chunk and updates every step. Colored borders separate "frozen input" from "live".
    """
    div_v = np.full((PANEL + LABEL_H, DIV, 3), 15, dtype=np.uint8)
    top = np.concatenate(
        [
            _panel(panel1_rgb, f"O_t  INPUT (frozen) q#{query_k}", border=INPUT_COLOR),
            div_v,
            _panel(panel2_rgb, f"O_t+H  goal INPUT  demo[{g}]/{demo_T - 1}", border=INPUT_COLOR),
            div_v,
            _panel(exec_rgb, f"env execution  LIVE  step {chunk_j + 1}/{chunk_len}", border=EXEC_COLOR),
        ],
        axis=1,
    )
    width = top.shape[1]
    bars = _action_bars(action, width, gt_action=gt_action)
    frame = np.concatenate([top, bars], axis=0)
    color = (90, 220, 90) if status == "SUCCESS" else (235, 235, 235)
    trk = f"track_err={err:.3f}"
    if err_pos is not None and err_rot is not None:
        trk += f"(p{err_pos:.2f} r{err_rot:.2f})"
    line = (f"query#{query_k:3d}  t={t:3d}  goal=demo[{g:3d}]  chunk {chunk_j + 1}/{chunk_len}  "
            f"{trk}  {status}")
    if gt_action is not None:
        pe = float(np.linalg.norm(action[:3] - gt_action[:3]))
        re = float(np.linalg.norm(action[3:6] - gt_action[3:6]))
        line += f"  act_vs_gt[pos={pe:.2f} rot={re:.2f}]"
    cv2.putText(frame, line, (8, frame.shape[0] - ACT_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.44, color, 1, cv2.LINE_AA)
    if reach_pos is not None and reach_rot is not None:
        rg = f" grip={reach_grip:.3f}" if reach_grip is not None else ""
        cv2.putText(frame, f"goal-reach (eef vs O_t+H): pos={reach_pos:.3f}  rot={reach_rot:.3f}{rg}",
                    (8, frame.shape[0] - ACT_H - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (250, 230, 90), 1, cv2.LINE_AA)
    # even dims for libx264 yuv420p
    h, w = frame.shape[:2]
    return frame[: h - (h % 2), : w - (w % 2)]


def save_reach_plot(reach_pos, reach_rot, reach_grip, n_action_steps, out_path: Path, title: str = "") -> None:
    """Plot per-step goal reachability (|robot eef - goal eef|) for pos/rot/gripper.

    Vertical dotted lines mark re-plans (goal updates every n_action_steps). Within each chunk
    the curves should DECREASE if the IDM is steering toward the goal image, then jump at the
    next re-plan when a new (farther) goal is set.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(reach_pos)
    steps = list(range(n))
    replan = list(range(0, n, max(1, n_action_steps)))  # re-plan / goal-update positions
    # widen with rollout length so the per-step convergence within each chunk is legible
    fig, axs = plt.subplots(3, 1, figsize=(max(13, n * 0.06), 8.2), sharex=True)
    series = [(reach_pos, "eef pos (m)", "tab:blue"),
              (reach_rot, "eef axis-angle (rad)", "tab:orange"),
              (reach_grip, "gripper qpos", "tab:green")]
    for ax, (data, name, col) in zip(axs, series, strict=True):
        ax.plot(steps, data, lw=1.1, color=col)
        for c in replan:
            ax.axvline(c, color="red", ls=":", lw=0.5, alpha=0.4)
        ax.set_ylabel(f"|robot - goal|\n{name}")
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0)
    axs[0].set_title(f"goal reachability per step  —  re-plan every {n_action_steps} steps (red ticks)\n{title}")
    # put a numeric x tick at every re-plan position so it's obvious where each chunk starts
    axs[-1].set_xticks(replan)
    axs[-1].set_xticklabels(replan, rotation=90, fontsize=6)
    axs[-1].set_xlabel(f"env step  (ticks = re-plan / goal-update positions, every {n_action_steps})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=95)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# One rollout cell
# --------------------------------------------------------------------------------------
def run_cell(
    *,
    policy,
    preprocessor,
    postprocessor,
    env_pre,
    env_post,
    nz: GoalNormalizer,
    ds: LeRobotDataset,
    image_keys: list[str],
    task_id: int,
    bench_task_id: int,
    ep_index: int,
    ordinal: int,
    H: int,
    seed: int,
    out_path: Path,
    device: str,
    fps: int,
    attention: bool = False,
    saliency: bool = False,
) -> dict:
    # --- load reference demo: display images, normalized goals, states ---
    fr = int(ds.meta.episodes["dataset_from_index"][ep_index])
    to = int(ds.meta.episodes["dataset_to_index"][ep_index])
    T = to - fr
    demo_disp: list[np.ndarray] = []
    demo_goal: list[torch.Tensor] = []
    demo_state: list[torch.Tensor] = []
    demo_action: list[torch.Tensor] = []
    for k in range(fr, to):
        item = ds[k]
        demo_state.append(item[OBS_STATE].float())
        demo_action.append(item[ACTION].float())
        demo_goal.append(torch.stack([nz.image(item[ik], ik) for ik in image_keys], dim=0))
        demo_disp.append(to_uint8_hwc(item[image_keys[0]]))
    demo_state_t = torch.stack(demo_state)  # (T, state_dim)
    demo_action_t = torch.stack(demo_action)  # (T, action_dim), env scale (GT delta actions)

    # --- build single-task env, pin the init state to the demo's ordinal ---
    # NOTE: the dataset's task_index is NOT the LIBERO benchmark task_id — the two were
    # serialized in different orders. We MUST build the env from the benchmark id that
    # matches this demo's language, or the env would have a different BDDL goal + init
    # state and check_success could never fire. See main() for the language-based map.
    # If the policy consumes a wrist image, map the env's eye-in-hand camera to the SAME key the
    # dataset/policy use ("wrist_image"); otherwise the env emits "image2" and select_action
    # KeyErrors on observation.images.wrist_image. agentview always maps to "image".
    cam_mapping = None
    if any("wrist" in k for k in image_keys):
        cam_mapping = {"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"}
    env_cfg = LiberoEnv(
        task="libero_goal",
        task_ids=[bench_task_id],
        obs_type="pixels_agent_pos",
        observation_height=PANEL,
        observation_width=PANEL,
        camera_name_mapping=cam_mapping,
    )
    suite = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = next(iter(suite["libero_goal"].values()))
    try:
        env.envs[0].unwrapped.init_state_id = ordinal
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not pin init_state_id={ordinal}: {e}")

    n_action_steps = policy.config.n_action_steps
    max_steps = env.call("_max_episode_steps")[0]

    # Attention capture: hook the SpatialSoftmax pool of the o_t encoder and the goal encoder.
    # Each select_action(replan) runs both encoders; the hooks grab the feature map they pool over,
    # from which _attn_heatmap recovers "where the encoder looks" in o_t and in the goal image.
    attn_cap: dict[str, object] = {}
    obs_pool = goal_pool = None
    hooks = []
    if attention:
        diff = policy.diffusion
        oe = diff.rgb_encoder
        oe = oe[0] if isinstance(oe, torch.nn.ModuleList) else oe
        ge = diff.goal_rgb_encoder
        ge = ge[0] if isinstance(ge, torch.nn.ModuleList) else ge
        obs_pool, goal_pool = oe.pool, ge.pool
        hooks.append(obs_pool.register_forward_pre_hook(lambda m, i: attn_cap.__setitem__("obs", i[0].detach())))
        hooks.append(goal_pool.register_forward_pre_hook(lambda m, i: attn_cap.__setitem__("goal", i[0].detach())))

    policy.reset()
    obs, _ = env.reset(seed=[seed])

    frames: list[np.ndarray] = []
    errs: list[float] = []  # full 8-d state L2 (pos+rot+gripper)
    errs_pos: list[float] = []  # eef position part ||state[0:3]||
    errs_rot: list[float] = []  # eef axis-angle part ||state[3:6]||
    reach_pos: list[float] = []  # ||robot eef pos - GOAL frame eef pos|| (goal reachability)
    reach_rot: list[float] = []  # ||robot eef axisangle - GOAL frame eef axisangle||
    reach_grip: list[float] = []  # ||robot gripper_qpos - GOAL frame gripper_qpos||
    act_eef: list[float] = []  # per-step |eef delta| (env scale) actually applied in the rollout
    act_xyz: list[float] = []  # per-step mean |xyz delta|
    act_pos_err: list[float] = []  # per-step ||model xyz delta - demo[t] xyz delta||  (vs GT action)
    act_rot_err: list[float] = []  # per-step ||model axisangle delta - demo[t] axisangle delta||
    goal_shares: list[float] = []  # per-query goal_sens/(obs_sens+goal_sens) when --saliency
    panel1 = demo_disp[0].copy()  # frozen policy-input snapshot (updates only on re-plan)
    panel2 = demo_disp[min(H, T - 1)].copy()
    success = False
    steps_to_success = None

    # Query-block bookkeeping: the policy re-plans every n_action_steps. O_t / O_t+H are the
    # FROZEN inputs of the current query; the env panel shows that query's chunk executing.
    query_k = -1
    chunk_j = 0
    frozen_g = min(H, T - 1)

    t = 0
    done = False
    while not done and t < max_steps:
        replan = len(policy._queues[ACTION]) == 0  # True => this step the policy plans a new chunk
        if replan:
            query_k += 1
            chunk_j = 0
            frozen_g = min(t + H, T - 1)  # the goal index this query is conditioned on
        else:
            chunk_j += 1
        g = frozen_g
        goal = demo_goal[g].unsqueeze(0).to(device)  # (1, num_cam, 3, H, W)

        exec_rgb = env.call("render")[0]  # (256,256,3) uint8, dataset orientation

        ob = preprocess_observation(obs)
        try:
            ob["task"] = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            ob["task"] = [""] * env.num_envs
        ob = env_pre(ob)
        robot_state = ob[OBS_STATE].detach().clone().reshape(-1).cpu()  # (state_dim,), un-normalized
        ob = preprocessor(ob)
        ob[OBS_GOAL_IMAGES] = goal

        with torch.inference_mode():
            action = policy.select_action(ob)
        action = postprocessor(action)
        action = env_post({ACTION: action})[ACTION]
        act_np = action.to("cpu").numpy()  # (1, 7)

        if replan:
            panel1 = exec_rgb.copy()  # freeze the input the policy just conditioned on
            panel2 = demo_disp[g].copy()
            if saliency:
                # decision-level: |d action / d input pixel|, o_t and goal on a COMMON scale so
                # the brighter PANEL is the input the policy relies on more. 99th-pct shared norm
                # (robust to a few hot pixels). Also print goal_share = total goal-sens fraction —
                # the per-query number behind "the model relies on the goal X%".
                o_sal, g_sal = _input_saliency(policy, ob, image_keys, goal)
                o_sum, g_sum = float(o_sal.sum()), float(g_sal.sum())
                goal_share = g_sum / (o_sum + g_sum + 1e-8)
                goal_shares.append(goal_share)
                vmax = float(np.percentile(np.concatenate([o_sal.ravel(), g_sal.ravel()]), 99)) + 1e-8
                panel1 = _overlay_attn(panel1, np.clip(o_sal / vmax, 0, 1), flip=True)
                panel2 = _overlay_attn(panel2, np.clip(g_sal / vmax, 0, 1), flip=False)
                # share readout, color-coded (green when goal dominates) — baseline ~40%, rewind ~68%
                _share_label(panel1, f"O_t {1 - goal_share:.0%}", dominant=goal_share < 0.5)
                _share_label(panel2, f"goal {goal_share:.0%}", dominant=goal_share >= 0.5)
            elif attention and "obs" in attn_cap:
                # representation-level: each encoder's SpatialSoftmax attention (self-normalized)
                panel1 = _overlay_attn(panel1, _attn_heatmap(obs_pool, attn_cap["obs"]), flip=True)
                panel2 = _overlay_attn(panel2, _attn_heatmap(goal_pool, attn_cap["goal"]), flip=False)

        diff = robot_state - demo_state_t[min(t, T - 1)]
        err = float(diff.norm())  # full state (pos+rot+gripper)
        err_pos = float(diff[:3].norm())  # eef position
        err_rot = float(diff[3:6].norm())  # eef axis-angle
        errs.append(err)
        errs_pos.append(err_pos)
        errs_rot.append(err_rot)
        # goal reachability: distance from the GOAL frame's eef (the frozen_g this inference used),
        # split into position / rotation / gripper so we can watch each shrink within a chunk.
        gdiff = robot_state - demo_state_t[g]
        reach_pos.append(float(gdiff[:3].norm()))
        reach_rot.append(float(gdiff[3:6].norm()))
        reach_grip.append(float(gdiff[6:8].norm()))
        act_eef.append(float(np.linalg.norm(act_np[0, :6])))
        act_xyz.append(float(np.abs(act_np[0, :3]).mean()))
        demo_act = demo_action_t[min(t, T - 1)].numpy()  # GT action at the same (demo-clock) step
        act_pos_err.append(float(np.linalg.norm(act_np[0, :3] - demo_act[:3])))
        act_rot_err.append(float(np.linalg.norm(act_np[0, 3:6] - demo_act[3:6])))
        frames.append(
            compose_frame(panel1, panel2, exec_rgb, t=t, g=g, err=err, err_pos=err_pos, err_rot=err_rot,
                          action=act_np[0], gt_action=demo_act, status="SUCCESS" if success else "running",
                          query_k=query_k, chunk_j=chunk_j, chunk_len=n_action_steps, demo_T=T,
                          reach_pos=reach_pos[-1], reach_rot=reach_rot[-1], reach_grip=reach_grip[-1])
        )

        obs, _reward, terminated, truncated, info = env.step(act_np)

        if "final_info" in info and isinstance(info["final_info"], dict):
            successes = info["final_info"]["is_success"].tolist()
        elif "is_success" in info:
            iss = info["is_success"]
            successes = iss.tolist() if hasattr(iss, "tolist") else [bool(iss)] * env.num_envs
        else:
            successes = [False] * env.num_envs
        if successes[0] and not success:
            success = True
            steps_to_success = t + 1
        done = bool(terminated[0] or truncated[0])
        t += 1

    status = "SUCCESS" if success else "FAIL"
    # tag the final few frames with the outcome so the still is readable
    for j in range(max(0, len(frames) - 8), len(frames)):
        f = frames[j]
        cv2.putText(f, status, (f.shape[1] // 2 - 40, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (90, 220, 90) if success else (90, 90, 235), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(out_path), frames, fps)
    save_reach_plot(reach_pos, reach_rot, reach_grip, n_action_steps,
                    out_path.with_suffix(".reach.png"), title=out_path.stem)
    for h in hooks:
        h.remove()
    env.close()

    return {
        "task_id": task_id,
        "ep_index": ep_index,
        "ordinal": ordinal,
        "demo_len": T,
        "rollout_steps": t,
        "success": success,
        "steps_to_success": steps_to_success,
        "mean_track_err": float(np.mean(errs)) if errs else None,
        "mean_track_err_pos": float(np.mean(errs_pos)) if errs_pos else None,
        "mean_track_err_rot": float(np.mean(errs_rot)) if errs_rot else None,
        "final_track_err": float(errs[-1]) if errs else None,
        "final_track_err_pos": float(errs_pos[-1]) if errs_pos else None,
        "final_track_err_rot": float(errs_rot[-1]) if errs_rot else None,
        "mean_goal_share": float(np.mean(goal_shares)) if goal_shares else None,
        "mean_act_eef": float(np.mean(act_eef)) if act_eef else None,
        "mean_act_pos_err": float(np.mean(act_pos_err)) if act_pos_err else None,
        "mean_act_rot_err": float(np.mean(act_rot_err)) if act_rot_err else None,
        "mean_reach_pos": float(np.mean(reach_pos)) if reach_pos else None,
        "mean_reach_rot": float(np.mean(reach_rot)) if reach_rot else None,
        "mean_reach_grip": float(np.mean(reach_grip)) if reach_grip else None,
        "final_reach_pos": float(reach_pos[-1]) if reach_pos else None,
        "final_reach_rot": float(reach_rot[-1]) if reach_rot else None,
        "final_reach_grip": float(reach_grip[-1]) if reach_grip else None,
        "n_action_steps": n_action_steps,
        "track_err": errs,
        "track_err_pos": errs_pos,
        "track_err_rot": errs_rot,
        "act_eef": act_eef,
        "act_xyz": act_xyz,
        "act_pos_err": act_pos_err,
        "act_rot_err": act_rot_err,
        "reach_pos": reach_pos,
        "reach_rot": reach_rot,
        "reach_grip": reach_grip,
        "video": str(out_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Closed-loop 3-panel rollout eval for diffusion_idm.")
    p.add_argument("--variant", required=True, help="label for outputs (e.g. baseline / rewind)")
    p.add_argument("--ckpt", type=Path, required=True, help="path to .../checkpoints/<step>/pretrained_model")
    p.add_argument("--dataset-root", type=Path, default=Path("/workspace/data/libero_goal_image"))
    p.add_argument("--repo-id", default="lerobot/libero_goal_image")
    p.add_argument("--split", type=Path, default=Path("/workspace/data/libero_goal_split.json"))
    p.add_argument("--out", type=Path, required=True, help="output dir (videos + summary)")
    p.add_argument("--tasks", type=int, nargs="*", default=None, help="restrict to these task ids")
    p.add_argument("--n-demos", type=int, default=1, help="reference demos per task")
    p.add_argument("--demo-offset", type=int, default=0,
                   help="skip this many demos before taking n-demos (for per-demo process isolation)")
    p.add_argument("--H", type=int, default=20, help="goal offset in frames (mid of training window)")
    p.add_argument("--n-action-steps", type=int, default=None,
                   help="override how many predicted actions to execute per inference (default: policy config=4). "
                        "Set ~15 (=horizon-1, the max) to run a whole predicted chunk open-loop per inference, "
                        "so each inference goes (nearly) all the way to its goal. No retraining needed.")
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--fps", type=int, default=15, help="playback fps (lower => each query block is easier to see)")
    p.add_argument("--attention", action="store_true",
                   help="overlay each encoder's SpatialSoftmax attention on the O_t and goal panels")
    p.add_argument("--saliency", action="store_true",
                   help="overlay gradient saliency |d action/d input| on O_t and goal (COMMON scale, "
                        "decision-level; brighter input = drives the action more). Overrides --attention.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    split = json.load(open(args.split))
    seen_tasks = set(split["seen_task_ids"])
    unseen_tasks = set(split["unseen_task_ids"])
    seen_eval = split["seen_eval_episodes"]
    unseen_eval = split["unseen_eval_episodes"]

    policy = DiffusionIDMPolicy.from_pretrained(args.ckpt).to(args.device).eval()
    if args.n_action_steps is not None:
        # Inference-only override: execute more of the predicted horizon per inference (e.g. a whole
        # ~15-step chunk -> each inference goes (almost) to its goal). Weights are unchanged.
        policy.config.n_action_steps = args.n_action_steps
        print(f"[rollout_eval] OVERRIDE n_action_steps -> {args.n_action_steps} (inference-only, no retrain)")
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(args.ckpt))
    env_pre, env_post = make_env_pre_post_processors(
        env_cfg=LiberoEnv(task="libero_goal", obs_type="pixels_agent_pos"), policy_cfg=policy.config
    )
    image_keys = list(policy.config.image_features)
    print(f"[rollout_eval] variant={args.variant} image_keys={image_keys} "
          f"n_obs={policy.config.n_obs_steps} n_action={policy.config.n_action_steps} H={args.H}")

    ds = LeRobotDataset(args.repo_id, root=str(args.dataset_root))
    nz = GoalNormalizer(ds, image_keys)
    ep_task = episode_task_map(ds)

    # dataset task_index -> LIBERO benchmark task_id, matched by language (single source
    # of truth in libero_task_map). The dataset serialized tasks in a DIFFERENT order than
    # the benchmark, so feeding the dataset index straight into LiberoEnv(task_ids=...) would
    # build the WRONG task's env (wrong BDDL goal + init states) and check_success could never
    # fire. The helper matches by language and asserts the map is 1:1 (fails loud on mismatch).
    ds2bench = dataset_to_benchmark_map(ds.meta, "libero_goal")
    print(f"[task-id map] dataset task_index -> benchmark task_id: {ds2bench}")

    # per-task sorted episode lists (within-task ordinal -> init_state index)
    task_eps: dict[int, list[int]] = {}
    for e in range(ds.meta.total_episodes):
        task_eps.setdefault(ep_task[e], []).append(e)
    for tk in task_eps:
        task_eps[tk].sort()
    ordinal_of = {e: task_eps[ep_task[e]].index(e) for e in range(ds.meta.total_episodes)}

    eval_by_task: dict[int, list[int]] = {}
    for e in seen_eval + unseen_eval:
        eval_by_task.setdefault(ep_task[e], []).append(e)
    for tk in eval_by_task:
        eval_by_task[tk].sort()

    target_tasks = args.tasks if args.tasks is not None else sorted(seen_tasks | unseen_tasks)
    task_name = {int(k): v["desc"] for k, v in split.get("per_task", {}).items()}

    results = []
    for task_id in target_tasks:
        split_name = "seen" if task_id in seen_tasks else "unseen"
        demos = eval_by_task.get(task_id, [])[args.demo_offset : args.demo_offset + args.n_demos]
        if not demos:
            print(f"  [skip] task {task_id}: no eval episodes")
            continue
        name_slug = slug(task_name.get(task_id, f"task{task_id}"))
        for ep_index in demos:
            ordinal = ordinal_of[ep_index]
            out_path = args.out / args.variant / f"{split_name}_task{task_id}_{name_slug}_ep{ep_index}.mp4"
            print(f"  rolling out {split_name} task {task_id} demo ep{ep_index} (ordinal {ordinal}) -> {out_path.name}")
            r = run_cell(
                policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
                env_pre=env_pre, env_post=env_post, nz=nz, ds=ds, image_keys=image_keys,
                task_id=task_id, bench_task_id=ds2bench[task_id], ep_index=ep_index,
                ordinal=ordinal, H=args.H, seed=args.seed,
                out_path=out_path, device=args.device, fps=args.fps,
                attention=args.attention, saliency=args.saliency,
            )
            r.update({"variant": args.variant, "split": split_name})
            print(f"     -> success={r['success']} steps={r['rollout_steps']} "
                  f"mean_track_err={r['mean_track_err']:.3f}")
            results.append(r)

    # --- write summaries ---
    args.out.mkdir(parents=True, exist_ok=True)
    full = args.out / f"results_{args.variant}.json"
    with open(full, "w") as f:
        json.dump(results, f, indent=2)

    # compact per-(split,task) aggregation
    agg: dict[str, dict] = {}
    for r in results:
        key = f"{r['split']}/task{r['task_id']:02d}"
        a = agg.setdefault(key, {"n": 0, "succ": 0, "track_err": [], "reach_pos": [], "reach_rot": []})
        a["n"] += 1
        a["succ"] += int(r["success"])
        a["track_err"].append(r["mean_track_err"])
        a["reach_pos"].append(r["mean_reach_pos"])
        a["reach_rot"].append(r["mean_reach_rot"])

    def _m(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    summary = {
        "variant": args.variant,
        "ckpt": str(args.ckpt),
        "H": args.H,
        "cells": {
            k: {
                "n": a["n"],
                "success_rate": a["succ"] / a["n"],
                "mean_track_err": _m(a["track_err"]),
                "mean_reach_pos": _m(a["reach_pos"]),
                "mean_reach_rot": _m(a["reach_rot"]),
            }
            for k, a in sorted(agg.items())
        },
    }
    with open(args.out / f"summary_{args.variant}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[rollout_eval] {args.variant} summary:")
    print(f"{'cell':>16} | {'n':>3} | {'success':>8} | {'reach_pos':>10} | {'reach_rot':>10}")
    for k, v in summary["cells"].items():
        print(f"{k:>16} | {v['n']:>3} | {v['success_rate']*100:>6.1f}% | "
              f"{v['mean_reach_pos']:>10.3f} | {v['mean_reach_rot']:>10.3f}")
    print(f"\nwrote {full} and summary_{args.variant}.json")


if __name__ == "__main__":
    main()
