#!/usr/bin/env bash
# Closed-loop 3-panel rollout eval for the goal-conditioned IDM (diffusion_idm) on LIBERO_GOAL.
# Produces, per (variant, task, demo), a synchronized video:
#     [ O_t (policy input) | O_t+H (goal input) | env execution ]
# plus success rate + open-loop goal-tracking-error per (variant, seen|unseen, task).
#
# Goal schedule = open-loop demo clock: goal = demo_img[min(t+H, T-1)]. Both variants get the
# IDENTICAL goal sequence, so differences are attributable to the policy (see rollout_eval.py).
#
# Usage:
#   bash scripts/rollout_eval.sh                       # both variants, all 10 tasks, last ckpt
#   VARIANTS=rewind TASKS=7 N_DEMOS=1 bash scripts/rollout_eval.sh   # smoke: one cell
#   CKPT_STEP=020000 bash scripts/rollout_eval.sh      # pin a specific checkpoint step
# Env vars: VARIANTS, TASKS (space-separated task ids), N_DEMOS, H, CKPT_STEP, OUT_DIR, GPU, SEED.
# Output: $OUT_DIR/<variant>/<seen|unseen>_task<id>_ep<e>.mp4 + summary_<variant>.json
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

VARIANTS="${VARIANTS:-baseline rewind}"
TASKS="${TASKS:-}"                       # empty => all tasks from the split
N_DEMOS="${N_DEMOS:-1}"
H="${H:-20}"
CKPT_STEP="${CKPT_STEP:-last}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/rollout_eval_out}"
SEED="${SEED:-10000}"
GPU="${GPU:-0}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/libero_goal_image}"
SPLIT="${SPLIT:-$REPO_ROOT/data/libero_goal_split.json}"

# MuJoCo EGL: this system exposes ONLY EGL device 0, and robosuite also requires
# MUJOCO_EGL_DEVICE_ID to be IN CUDA_VISIBLE_DEVICES -> rollouts must run on GPU 0 (EGL id 0).
# GPU is kept configurable but should include 0; default 0.
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

TASK_ARGS=()
if [ -n "$TASKS" ]; then
  TASK_ARGS=(--tasks $TASKS)
fi

cd "$REPO_ROOT/lerobot"
for variant in $VARIANTS; do
  CKPT="$REPO_ROOT/lerobot/outputs/idm_${variant}/checkpoints/${CKPT_STEP}/pretrained_model"
  if [ ! -d "$CKPT" ]; then
    echo "[rollout_eval.sh] checkpoint not found: $CKPT  (skipping $variant)"
    continue
  fi
  echo "[rollout_eval.sh] variant=$variant ckpt=$CKPT H=$H n_demos=$N_DEMOS gpu=$GPU tasks=${TASKS:-all}"
  uv run python "$REPO_ROOT/rollout_eval.py" \
    --variant "$variant" \
    --ckpt "$CKPT" \
    --dataset-root "$DATASET_ROOT" \
    --split "$SPLIT" \
    --out "$OUT_DIR" \
    --n-demos "$N_DEMOS" \
    --H "$H" \
    --seed "$SEED" \
    "${TASK_ARGS[@]}"
done
echo "[rollout_eval.sh] done -> $OUT_DIR"
