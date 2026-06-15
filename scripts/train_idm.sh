#!/usr/bin/env bash
# Train the goal-conditioned IDM (diffusion_idm) on the libero_goal SEEN-task train split.
#
# Usage:
#   bash scripts/train_idm.sh <baseline|rewind>
#   STEPS=300 OUT=/tmp/idm_test bash scripts/train_idm.sh baseline   # quick smoke
#
# Variant:
#   baseline -> trained on the original dataset           (data/libero_goal_image)
#   rewind   -> trained on the ReWiND (time-reversed) copy (data/libero_goal_image_rewind)
# Both use the SAME 270-episode train split (seen tasks, 80% of episodes) from
# data/libero_goal_split.json so the comparison is controlled.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

VARIANT="${1:-baseline}"
SPLIT="${SPLIT:-$REPO_ROOT/data/libero_goal_split.json}"
STEPS="${STEPS:-60000}"
BATCH="${BATCH:-64}"            # per-GPU batch size (effective = BATCH * NUM_GPUS under DDP)
SAVE_FREQ="${SAVE_FREQ:-10000}"
NUM_GPUS="${NUM_GPUS:-1}"       # >1 => accelerate multi-GPU data-parallel
MIXED_PRECISION="${MIXED_PRECISION:-no}"  # no | fp16 | bf16
OUT="${OUT:-$REPO_ROOT/lerobot/outputs/idm_${VARIANT}}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

# baseline: original 428-ep dataset, original 270 train episodes.
# rewind:   COMBINED original+reversed 856-ep dataset (libero_goal_image_combined), 540 train
#           episodes (270 forward + their 270 reversed copies) = the intended ReWiND set.
case "$VARIANT" in
  baseline) ROOT="$REPO_ROOT/data/libero_goal_image";          EPS_KEY=train_episodes ;;
  rewind)   ROOT="$REPO_ROOT/data/libero_goal_image_combined"; EPS_KEY=combined_train_episodes ;;
  *) echo "variant must be 'baseline' or 'rewind' (got '$VARIANT')"; exit 1 ;;
esac

cd "$REPO_ROOT/lerobot"
EPS=$(uv run python -c "import json; print(json.load(open('$SPLIT'))['$EPS_KEY'])")

echo "[train_idm] variant=$VARIANT root=$ROOT steps=$STEPS batch=$BATCH num_gpus=$NUM_GPUS mp=$MIXED_PRECISION out=$OUT gpu=$GPU"

TRAIN_ARGS=(
  --policy.type=diffusion_idm
  --policy.push_to_hub=false
  --policy.device=cuda
  --dataset.repo_id=lerobot/libero_goal_image
  --dataset.root="$ROOT"
  --dataset.episodes="$EPS"
  --batch_size="$BATCH"
  --steps="$STEPS"
  --save_freq="$SAVE_FREQ"
  --save_checkpoint=true
  --log_freq=200
  --num_workers="${NUM_WORKERS:-16}"
  --wandb.enable=false
  --output_dir="$OUT"
)

# Camera(s) the policy uses for obs + goal. Default: agentview only (wrist goal images are
# too end-effector-pose dependent). Set IMAGE_KEYS=all to use every camera.
IMAGE_KEYS="${IMAGE_KEYS:-[\"observation.images.image\"]}"
if [ "$IMAGE_KEYS" != "all" ]; then
  TRAIN_ARGS+=(--policy.image_keys="$IMAGE_KEYS")
fi

# Optional override of the goal-offset sampling concentration. Unset => use the config default
# (0.6, H=8-weighted). Set PEAK_PROB=0 to recover flat Uniform{1..8} (the h8unif setup).
if [ -n "${PEAK_PROB:-}" ]; then
  TRAIN_ARGS+=(--policy.goal_offset_peak_prob="$PEAK_PROB")
fi

# Optional torch.compile of the model (free compute speedup, no architecture change).
if [ "${COMPILE:-0}" = "1" ]; then
  TRAIN_ARGS+=(--policy.compile_model=true)
fi

# Launch via accelerate (handles mixed precision for single- and multi-GPU). num_processes=1
# is single-GPU (no DDP sync overhead); >1 adds --multi_gpu data parallelism.
ACC=(uv run accelerate launch --num_processes="$NUM_GPUS" --mixed_precision="$MIXED_PRECISION")
if [ "$NUM_GPUS" -gt 1 ]; then
  ACC+=(--multi_gpu)
fi
CUDA_VISIBLE_DEVICES="$GPU" "${ACC[@]}" "$REPO_ROOT/lerobot/.venv/bin/lerobot-train" "${TRAIN_ARGS[@]}"
