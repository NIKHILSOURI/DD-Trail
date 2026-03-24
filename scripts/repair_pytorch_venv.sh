#!/usr/bin/env bash
# Restore the main DreamDiffusion venv after mistakenly installing
# requirements-thoughtviz.txt (TensorFlow) into it. TF 2.13 pins typing_extensions<4.6,
# which breaks PyTorch 2.7+ (needs TypeIs). thoughtviz-gpu can also replace cudnn and
# break torch.
#
# Usage (from repo root, main venv active):
#   bash scripts/repair_pytorch_venv.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "[repair] WARNING: No active venv (VIRTUAL_ENV empty). Activate your project venv first, e.g.:"
  echo "         source \"$REPO_ROOT/venv/bin/activate\""
fi

echo "[repair] Removing TensorFlow / Keras stack from this venv (safe if not installed)..."
python -m pip uninstall -y \
  tensorflow tensorflow-estimator tensorboard keras \
  tensorflow-io-gcs-filesystem \
  2>/dev/null || true

echo "[repair] Restoring PyTorch-compatible pins..."
python -m pip install "typing_extensions>=4.12.0"
# Torch cu118 wheel expects this cudnn metapackage version on Linux x86_64
python -m pip install "nvidia-cudnn-cu11==9.1.0.70"

echo "[repair] Reinstalling core requirements (numpy/scipy/torch stack)..."
python -m pip install -r "$REPO_ROOT/requirements.txt"

echo "[repair] Verifying import torch..."
python -c "import torch; import typing_extensions as te; print('torch', torch.__version__, 'typing_extensions', getattr(te, '__version__', 'ok'))"

echo "[repair] Done. Do NOT run pip install -r requirements-thoughtviz.txt in this venv."
echo "         Use:  python3 -m venv venv_thoughtviz && source venv_thoughtviz/bin/activate && pip install -r requirements-thoughtviz.txt"
