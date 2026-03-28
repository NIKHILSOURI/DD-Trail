# Unified Benchmark Environment

This repository now supports a single unified benchmark environment for:

- DreamDiffusion baseline
- SAR-HM
- ThoughtViz EEG + GAN
- Summary, segmentation, CLIP/FID-style metrics

## Files

- `requirements_unified.txt`
- `scripts/setup_unified_env.sh`
- `scripts/test_unified_imports.py`

## Paths

Example Linux clone on this workspace: `/workspace/DD-Trail`. Shell scripts default `REPO_ROOT` to the parent of `scripts/` when unset.

## Create Environment

From repo root:

```bash
export REPO_ROOT="/workspace/DD-Trail"
cd "$REPO_ROOT"
bash scripts/setup_unified_env.sh
```

Optional CUDA wheel index override:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash scripts/setup_unified_env.sh
```

## Validate Imports

```bash
source "$REPO_ROOT/venv_unified/bin/activate"
python scripts/test_unified_imports.py
```

On Linux, use `source "$REPO_ROOT/venv_unified/bin/activate"`. On Windows PowerShell, use `.\venv_unified\Scripts\Activate.ps1` instead.

## Compatibility Decision (Important)

For a true single-venv setup with ThoughtViz TensorFlow 2.13 and PyTorch together, the environment uses:

- `typing_extensions<4.6` (TensorFlow 2.13 requirement)
- PyTorch pinned to `2.1.2` line (pre-`TypeIs` requirement)

This is a deliberate conservative pin to avoid the `TypeIs` conflict seen with PyTorch 2.7+.

## Notes

- The unified stack keeps existing project conventions and avoids aggressive upgrades.
- ThoughtViz uses legacy Keras APIs; TensorFlow/Keras is pinned to `2.13.x`.
- If a specific GPU driver/runtime combination needs different torch wheels, set `TORCH_INDEX_URL`.
