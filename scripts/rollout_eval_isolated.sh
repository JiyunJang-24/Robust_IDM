#!/usr/bin/env bash
# Crash-safe rollout eval: run EACH task in its OWN python process, then merge.
#
# WHY: robosuite's MuJoCo EGL render context leaks across env create/close cycles. In a single
# long process the leak accumulates (faster when rollouts fail and run to max_steps, emitting more
# render calls) until `binding_utils.render` blocks forever (~20 rollouts in). Running one task per
# process caps each process at ~n_demos rollouts (well under the leak threshold) and lets the OS
# reclaim all EGL state on process exit. Output is merged into the same results_<variant>.json
# layout that compare_eval.py / rollout_eval.py expect.
#
# Usage:
#   VARIANT=rewind CKPT=/path/to/pretrained_model OUT_DIR=$PWD/rollout_eval_out_X \
#     bash scripts/rollout_eval_isolated.sh
# Env: VARIANT, CKPT, OUT_DIR, TASKS ("0 1 .." default 0..9), N_DEMOS, H, SEED, GPU, DATASET_ROOT, SPLIT.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

VARIANT="${VARIANT:-rewind}"
CKPT="${CKPT:?set CKPT to a pretrained_model dir}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/rollout_eval_out_isolated}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
N_DEMOS="${N_DEMOS:-5}"
H="${H:-8}"
SEED="${SEED:-10000}"
GPU="${GPU:-0}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/libero_goal_image}"
SPLIT="${SPLIT:-$REPO_ROOT/data/libero_goal_split.json}"

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0          # this host exposes only EGL device 0
export CUDA_VISIBLE_DEVICES="$GPU"

# Tasks where the model tends to FAIL run every demo to max_steps, emitting far more render
# calls per rollout. There the EGL leak hits its threshold within a single task's n_demos, so
# those tasks must be split further: one process PER DEMO. UNSEEN_TASKS (default the libero_goal
# held-out 7 9) get per-demo isolation; all others get per-task isolation (success ends early,
# few render calls). Override UNSEEN_TASKS="" to force per-task everywhere.
UNSEEN_TASKS="${UNSEEN_TASKS:-7 9}"

cd "$REPO_ROOT/lerobot"
echo "[isolated] variant=$VARIANT ckpt=$CKPT out=$OUT_DIR tasks='$TASKS' n_demos=$N_DEMOS H=$H per_demo_tasks='$UNSEEN_TASKS'"

run_one() {  # $1=task $2=n_demos $3=demo_offset $4=out_subdir
  uv run python "$REPO_ROOT/rollout_eval.py" \
    --variant "$VARIANT" --ckpt "$CKPT" \
    --dataset-root "$DATASET_ROOT" --split "$SPLIT" \
    --out "$4" --tasks "$1" --n-demos "$2" --demo-offset "$3" --H "$H" --seed "$SEED" \
    || echo "[isolated] task $1 demo_offset $3 exited non-zero (continuing)"
}

for t in $TASKS; do
  if echo " $UNSEEN_TASKS " | grep -q " $t "; then
    # per-demo isolation: one fresh process per demo (EGL leak cannot accumulate across demos)
    for off in $(seq 0 $((N_DEMOS - 1))); do
      sub="$OUT_DIR/task${t}_d${off}"
      echo "[isolated] === task $t demo $off (own process) -> $sub ==="
      run_one "$t" 1 "$off" "$sub"
    done
  else
    sub="$OUT_DIR/task$t"
    echo "[isolated] === task $t (own process) -> $sub ==="
    run_one "$t" "$N_DEMOS" 0 "$sub"
  fi
done

# Merge per-task results_<variant>.json into one OUT_DIR/results_<variant>.json
uv run python "$REPO_ROOT/merge_eval_results.py" --dir "$OUT_DIR" --variant "$VARIANT"
echo "[isolated] done -> $OUT_DIR/results_${VARIANT}.json"
