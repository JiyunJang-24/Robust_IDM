# Robust IDM

Utilities for training and evaluating goal-conditioned inverse dynamics models on LIBERO goal tasks.

## Repository Layout

- `scripts/`: shell entry points for training, rollouts, isolated rollouts, and probing.
- `lerobot/`, `LIBERO/`, `susie/`: external dependencies managed as Git submodules.
- `data/`: local LIBERO datasets and train/eval split files. This directory is ignored by Git.
- `logs/` and `rollout*/`: generated training and evaluation outputs. These directories are ignored by Git.

After cloning, initialize the submodules:

```bash
git submodule update --init --recursive
```

Install the Python dependencies inside the `lerobot` environment before running these scripts. The scripts use `uv run` from `lerobot/`, so `uv` and the `lerobot` virtual environment must be available.

## Common Environment Variables

All shell scripts resolve the project root from their own location, so they can be run from any checkout path.

- `CUDA_VISIBLE_DEVICES` or `GPU`: GPU selection.
- `DATASET_ROOT`: dataset directory. Defaults to `data/libero_goal_image` for eval/probe.
- `SPLIT`: split file. Defaults to `data/libero_goal_split.json`.
- `OUT_DIR` or `OUT`: output location for generated results.

## Train IDM

Train one model variant:

```bash
bash scripts/train_idm.sh baseline
bash scripts/train_idm.sh rewind
```

Useful overrides:

```bash
STEPS=300 BATCH=16 SAVE_FREQ=100 OUT=/tmp/idm_test bash scripts/train_idm.sh baseline
```

Variants:

- `baseline`: trains on `data/libero_goal_image`.
- `rewind`: trains on `data/libero_goal_image_combined`.

The script writes checkpoints to `lerobot/outputs/idm_<variant>/` unless `OUT` is set.

## Run Parallel Training

Launch baseline on GPU 0 and rewind on GPU 1:

```bash
bash scripts/run_training.sh
```

Useful overrides:

```bash
STEPS=15000 SAVE_FREQ=5000 BATCH=64 MIXED_PRECISION=bf16 COMPILE=1 bash scripts/run_training.sh
```

Logs are written to `logs/train_baseline.log` and `logs/train_rewind.log` unless `LOG_DIR` is set.

## Rollout Evaluation

Run closed-loop rollouts for both variants:

```bash
bash scripts/rollout_eval.sh
```

Run a smoke test for one rewind task and one demo:

```bash
VARIANTS=rewind TASKS=7 N_DEMOS=1 bash scripts/rollout_eval.sh
```

Pin a checkpoint step:

```bash
CKPT_STEP=020000 bash scripts/rollout_eval.sh
```

Important variables:

- `VARIANTS`: space-separated variants. Defaults to `baseline rewind`.
- `TASKS`: space-separated task ids. Empty means all tasks from the split.
- `N_DEMOS`: demos per task. Defaults to `1`.
- `H`: goal horizon. Defaults to `20`.
- `CKPT_STEP`: checkpoint directory under `checkpoints/`. Defaults to `last`.
- `OUT_DIR`: output directory. Defaults to `rollout_eval_out`.
- `SEED`: rollout seed. Defaults to `10000`.

## Isolated Rollout Evaluation

Use this when long rollouts hang because of MuJoCo EGL render context leaks. It starts a fresh Python process per task, and for selected unseen tasks per demo.

```bash
VARIANT=rewind \
CKPT="$PWD/lerobot/outputs/idm_rewind/checkpoints/last/pretrained_model" \
OUT_DIR="$PWD/rollout_eval_out_isolated" \
bash scripts/rollout_eval_isolated.sh
```

Important variables:

- `CKPT`: required path to a `pretrained_model` directory.
- `TASKS`: task ids to evaluate. Defaults to `0 1 2 3 4 5 6 7 8 9`.
- `UNSEEN_TASKS`: tasks run with per-demo isolation. Defaults to `7 9`; set to an empty string for per-task isolation only.
- `N_DEMOS`, `H`, `SEED`, `GPU`, `DATASET_ROOT`, `SPLIT`: same purpose as in `rollout_eval.sh`.

Merged results are written to `OUT_DIR/results_<variant>.json`.

## Goal-Influence Probe

Run the sim-free probe on a trained checkpoint:

```bash
bash scripts/probe.sh baseline
bash scripts/probe.sh rewind 030000
```

Useful overrides:

```bash
N_EPISODES=20 GOAL_OFFSET=20 OUT=/tmp/probe_rewind.json bash scripts/probe.sh rewind last
```

The default policy path is `lerobot/outputs/idm_<variant>/checkpoints/<checkpoint>/pretrained_model`.
