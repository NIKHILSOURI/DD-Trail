#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/benchmark_unified.yaml}"

cd "$REPO_ROOT"
source "$REPO_ROOT/venv/bin/activate"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"

python scripts/run_full_benchmark_pipeline.py --config "$CONFIG_PATH" "$@"
