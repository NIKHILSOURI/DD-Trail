#!/usr/bin/env bash
# Source with ThoughtViz venv active so TensorFlow finds CUDA libs from pip (nvidia-*-cu11).
#   source venv_thoughtviz/bin/activate
#   source "$REPO_ROOT/code/ThoughtViz/thoughtviz_gpu_env.sh"
#
# Install libs: pip install -r requirements-thoughtviz-gpu.txt
# If you edited on Windows: sed -i 's/\r$//' code/ThoughtViz/thoughtviz_gpu_env.sh

SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
TLP=""
for d in "$SITE"/nvidia/*/lib; do
  if [[ -d "$d" ]]; then
    TLP="${d}:${TLP}"
  fi
done
if [[ -d /usr/lib/x86_64-linux-gnu ]]; then
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${TLP}${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${TLP}${LD_LIBRARY_PATH:-}"
fi
echo "[thoughtviz_gpu_env] SITE=${SITE}"
echo "[thoughtviz_gpu_env] prepended $(echo "$TLP" | tr ':' '\n' | wc -l) nvidia lib dirs + /usr/lib/x86_64-linux-gnu"
