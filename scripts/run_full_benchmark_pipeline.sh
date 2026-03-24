#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/project/DREAMDIFFUSION_RUNPOD}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/benchmark_unified.yaml}"

cd "$REPO_ROOT"
source "$REPO_ROOT/venv/bin/activate"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"

python scripts/run_full_benchmark_pipeline.py --config "$CONFIG_PATH" "$@"
