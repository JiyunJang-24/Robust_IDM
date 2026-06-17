#!/usr/bin/env bash
# Reproduce the local environment needed to run scripts/train_idm.sh.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Avoid using an unrelated active virtualenv from the parent shell.
unset VIRTUAL_ENV
unset CONDA_PREFIX
export UV_LINK_MODE=copy

command -v uv >/dev/null
command -v python >/dev/null

git submodule update --init --recursive

cd "$SCRIPT_DIR/lerobot"
uv sync --extra training --extra diffusion

uv run python - <<'PY'
import json
from pathlib import Path

import accelerate
import diffusers
import torch

root = Path.cwd().parent
split = root / "data" / "libero_goal_split.json"
dataset = root / "data" / "libero_goal_image"

assert split.exists(), f"missing split file: {split}"
assert dataset.exists(), f"missing baseline dataset: {dataset}"
with split.open() as f:
    train_episodes = json.load(f)["train_episodes"]

print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"accelerate={accelerate.__version__}")
print(f"diffusers={diffusers.__version__}")
print(f"baseline_train_episodes={len(train_episodes)}")
PY
