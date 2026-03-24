#!/usr/bin/env bash
# Run ThoughtViz Python with CUDA 11 libs on LD_LIBRARY_PATH (TensorFlow 2.13 GPU).
# Usage (from anywhere):
#   export REPO_ROOT=/path/to/DREAMDIFFUSION_RUNPOD
#   source venv_thoughtviz/bin/activate
#   bash "$REPO_ROOT/code/ThoughtViz/run_with_gpu.sh" training/thoughtviz_with_eeg.py
set -eu
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/../.." && pwd)}"
# shellcheck source=thoughtviz_gpu_env.sh
source "$HERE/thoughtviz_gpu_env.sh"
# Python puts the script's directory (training/) on sys.path[0], not ThoughtViz root — imports like `utils.*` need repo root.
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"
cd "$HERE"
exec python "$@"
