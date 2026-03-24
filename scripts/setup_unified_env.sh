#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/project/DREAMDIFFUSION_RUNPOD}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/venv_unified}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

echo "[setup_unified_env] REPO_ROOT=$REPO_ROOT"
echo "[setup_unified_env] VENV_DIR=$VENV_DIR"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel

# Install torch family first (versions aligned with TensorFlow 2.13 typing constraints),
# then install the remaining unified stack.
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url "$TORCH_INDEX_URL"
pip install -r "$REPO_ROOT/requirements_unified.txt"

# Make repository modules importable.
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"

python "$REPO_ROOT/scripts/test_unified_imports.py"

echo "[setup_unified_env] Done."
