#!/usr/bin/env bash
# Launch the baseline (original data) and ReWiND (time-reversed data) IDM trainings in
# PARALLEL, one GPU each:  baseline -> GPU 0,  rewind -> GPU 1.
# Both use the same 270-episode seen-task train split and identical settings, so the only
# difference is the dataset -> a controlled comparison for the goal-attention hypothesis.
#
# Settings (see train_idm.sh): agentview-only camera, bf16, torch.compile, single-GPU
# (DDP was dropped — these A6000s have no NVLink, so multi-GPU sync overhead negated the
# gain; single-GPU is faster per run here).
#
# Usage:   bash scripts/run_training.sh
#          STEPS=15000 bash scripts/run_training.sh
# Logs:    logs/train_{baseline,rewind}.log
# Checkpoints: lerobot/outputs/idm_{baseline,rewind}/checkpoints/
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

STEPS="${STEPS:-30000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
BATCH="${BATCH:-64}"
MP="${MIXED_PRECISION:-bf16}"
COMPILE="${COMPILE:-1}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
mkdir -p "$LOG_DIR"

echo "Launching baseline (GPU0) + rewind (GPU1): single-GPU, mp=$MP, compile=$COMPILE, agentview-only, STEPS=$STEPS ..."
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 MIXED_PRECISION="$MP" COMPILE="$COMPILE" STEPS="$STEPS" SAVE_FREQ="$SAVE_FREQ" BATCH="$BATCH" \
  nohup bash "$SCRIPT_DIR/train_idm.sh" baseline > "$LOG_DIR/train_baseline.log" 2>&1 &
echo "  baseline PID $!  -> $LOG_DIR/train_baseline.log (GPU0)"
CUDA_VISIBLE_DEVICES=1 NUM_GPUS=1 MIXED_PRECISION="$MP" COMPILE="$COMPILE" STEPS="$STEPS" SAVE_FREQ="$SAVE_FREQ" BATCH="$BATCH" \
  nohup bash "$SCRIPT_DIR/train_idm.sh" rewind > "$LOG_DIR/train_rewind.log" 2>&1 &
echo "  rewind   PID $!  -> $LOG_DIR/train_rewind.log (GPU1)"
wait
echo "Both trainings finished."
