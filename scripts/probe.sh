#!/usr/bin/env bash
# Run the goal-influence probe (open-loop, sim-free) on a trained diffusion_idm checkpoint.
#
# Usage:
#   bash scripts/probe.sh <baseline|rewind> [checkpoint]
#     checkpoint defaults to "last".  e.g. bash scripts/probe.sh baseline 030000
#
# Measures, at init vs mid-trajectory probe states (from SEEN-eval episodes) with o_t held
# in-distribution, how the predicted action changes when the goal image is swapped to a
# different SEEN task vs an UNSEEN task. Writes probe_<variant>.json in the repo root by default.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

VARIANT="${1:-baseline}"
CKPT="${2:-last}"
N_EPISODES="${N_EPISODES:-40}"
GOAL_OFFSET="${GOAL_OFFSET:-20}"
POLICY="${POLICY:-$REPO_ROOT/lerobot/outputs/idm_${VARIANT}/checkpoints/${CKPT}/pretrained_model}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/libero_goal_image}"
SPLIT="${SPLIT:-$REPO_ROOT/data/libero_goal_split.json}"
OUT="${OUT:-$REPO_ROOT/probe_${VARIANT}.json}"

cd "$REPO_ROOT/lerobot"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run python "$REPO_ROOT/goal_influence_probe.py" \
  --policy-path "$POLICY" \
  --dataset-root "$DATASET_ROOT" \
  --repo-id lerobot/libero_goal_image \
  --split "$SPLIT" \
  --positions 0.0 0.3 0.5 0.7 \
  --n-episodes "$N_EPISODES" \
  --goal-offset "$GOAL_OFFSET" \
  --output "$OUT"
