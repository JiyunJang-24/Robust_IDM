# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this workspace is

`/workspace` is **not** a git repository. It is a working environment that bundles
several independent robotics / VLA (vision-language-action) projects that are used
together to train and evaluate robot manipulation policies:

- **`lerobot/`** — A clone of HuggingFace [LeRobot](https://github.com/huggingface/lerobot)
  (its own git repo, `origin/main`). The primary codebase where most work happens.
  PyTorch robot-learning library with datasets, training/eval CLIs, and many VLA
  policies (act, diffusion, tdmpc, smolvla, pi0/pi05/pi0_fast, groot, molmoact2,
  xvla, eo1, wall_x, vqbet, …). **It has its own authoritative agent guide — see below.**
- **`LIBERO/`** — A clone of [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
  (its own git repo), a benchmark of 130 lifelong/multitask manipulation tasks built on
  robosuite/MuJoCo. Used both standalone and as a LeRobot evaluation environment.
- **`tutorials/`** — Standalone Jupyter notebooks (Python/NumPy/pandas/matplotlib basics).
  Not part of the robotics pipeline.
- **`welcome.ipynb`** — Workspace landing notebook.

There is also `.venv-vllm/`, a separate Python 3.12 virtualenv that previously held
vLLM for serving VLM/VLA backends. It has been removed; recreate it only if you bring
back a vLLM-based serving workflow.

### How lerobot and LIBERO connect

LeRobot integrates LIBERO as a Gymnasium environment in
[lerobot/src/lerobot/envs/libero.py](lerobot/src/lerobot/envs/libero.py)
(registered as `--env.type=libero`, config `LiberoEnv` in
[envs/configs.py](lerobot/src/lerobot/envs/configs.py)). At runtime LeRobot's `[libero]`
extra pulls in the pip-packaged `hf-libero`, so the standalone `LIBERO/` clone here is
**separate** from what LeRobot imports — edits to `LIBERO/` do not affect a `lerobot-eval
--env.type=libero` run unless you install this clone into lerobot's environment.

## Python environments (important — they are not interchangeable)

There is **no single shared interpreter**. Match the tool to its environment:

| Environment | Python | Use for |
|---|---|---|
| `lerobot/.venv` via `uv` | 3.12+ | All lerobot work. Run with `uv run` from inside `lerobot/`. |
| `/opt/conda` (base, on `PATH`) | 3.8.10 | Matches LIBERO's required `python=3.8.13`. LIBERO standalone (robosuite 1.4.0, gym 0.25.2, torch 1.11+cu113). |

The bare `python` on `PATH` is conda's 3.8 — **do not** use it for lerobot.

## Working in `lerobot/`

`lerobot/AGENTS.md` (symlinked as `lerobot/CLAUDE.md`) is the **authoritative** guide for
that subtree — read it before doing lerobot work. It covers the draccus config system,
policy/processor/env architecture, and test layout. `lerobot/AGENT_GUIDE.md` is the
user-facing how-to (recording, picking a policy, training, eval). Key points:

```bash
cd lerobot
uv sync --locked --extra all          # install deps (use --extra test --extra dev for CI tooling)
uv run pytest tests -svv --maxfail=10 # run tests
uv run pytest tests/path::test_name   # run a single test
pre-commit run --all-files            # lint + format (ruff, typos, bandit)
DEVICE=cuda make test-end-to-end      # E2E train/eval for act/diffusion/tdmpc/smolvla (see Makefile)
```

Prefer `uv run <cmd>` over raw `python`/`pip`. CLI entry points (`lerobot-train`,
`lerobot-eval`, `lerobot-rollout`, `lerobot-record`, …) are defined in
`pyproject.toml [project.scripts]`. Note `lerobot-rollout` is the deployment engine for
running policies on real robots with pluggable strategies (base/sentry/highlight/dagger)
and inference backends (sync / rtc for slow VLAs).

## Working in `LIBERO/`

```bash
# LIBERO expects python 3.8 (conda base here matches)
cd LIBERO
pip install -r requirements.txt && pip install -e .
python benchmark_scripts/download_libero_datasets.py [--use-huggingface] [--datasets libero_10]

# Lifelong-learning train/eval (see README.md for full arg matrix)
python libero/lifelong/main.py seed=SEED benchmark_name=LIBERO_10 policy=bc_transformer_policy lifelong=base
python libero/lifelong/evaluate.py --benchmark LIBERO_10 --task_id 0 --algo base --policy bc_transformer_policy --seed 1 --ep 50
```

Task suites: `LIBERO_SPATIAL`, `LIBERO_OBJECT`, `LIBERO_GOAL`, `LIBERO_90`, `LIBERO_10`.
Task definitions are BDDL files under `libero/libero/bddl_files/`. MuJoCo EGL rendering
needs `MUJOCO_EGL_DEVICE_ID` set alongside `CUDA_VISIBLE_DEVICES`.

## Gotchas

- Each sub-project is its **own git repo**; `/workspace` is not. Commit inside the
  relevant subdirectory, and don't expect git operations at the workspace root.
- LIBERO (py3.8 / old torch) and lerobot (py3.12) have conflicting dependency stacks —
  keep their environments isolated; never `pip install` LIBERO requirements into the
  lerobot venv.
