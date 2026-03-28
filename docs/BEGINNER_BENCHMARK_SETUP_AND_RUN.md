# Beginner Guide: Clean Setup and Full Benchmark Run (Split Environments)

This guide is the **recommended, safe workflow** for this repository.

It is written for beginners and assumes you are starting from a messy state (old/broken virtual environments, conflicting installs, partial runs).

**This checkout:** Linux; repository root **`/workspace/DD-Trail`** (this workspace). Most steps use **bash** from that folder. Set `REPO_ROOT` to the same path when helpful; Python scripts under `scripts/` default to the repo that contains them if `REPO_ROOT` is not set. In `configs/benchmark_unified.yaml`, paths are absolute under `/workspace/DD-Trail/...`; change them if you clone elsewhere.

---

## 1) Overview

This thesis repo includes:

- DreamDiffusion baseline
- DreamDiffusion + SAR-HM
- ThoughtViz
- benchmark generation, summary/caption, segmentation, metrics, tables, and visualization panels

Two datasets are used in benchmark runs:

- `imagenet_eeg`
- `thoughtviz`

### Why this guide uses two environments

Do **not** run everything in one Python environment.

- DreamDiffusion/SAR-HM depend on a modern PyTorch stack.
- ThoughtViz depends on legacy TensorFlow/Keras requirements.
- Mixing them in one env causes dependency breakage and unstable GPU behavior.

### Correct design

- `venv` -> DreamDiffusion + SAR-HM + orchestration + post-processing
- `venv_thoughtviz_gpu` -> ThoughtViz inference only
- One **shared manifest** controls sample IDs/order
- Results are combined from disk in one output tree

---

## 2) Why the environments conflict

The core conflicts are:

1. `typing_extensions`
   - Modern torch ecosystem wants newer versions.
   - TensorFlow 2.13-era stacks force old ranges.
   - Pip resolves one side and breaks the other.

2. Torch stack vs legacy TF/Keras stack
   - DreamDiffusion stack includes torch, torchvision, timm, lightning, etc.
   - ThoughtViz stack expects legacy TF/Keras behavior and different transitive pins.

3. CUDA runtime expectations
   - Torch and TF often expect different user-space CUDA/cuDNN library combinations.
   - Installing one framework's GPU extras can destabilize the other in the same env.

4. Real-world effect
   - Imports fail or downgrade unexpectedly.
   - GPU not detected in one framework after fixing the other.
   - ThoughtViz can fall back to CPU and hit legacy Conv2D format issues.

---

## 3) Final recommended architecture

Use this flow:

```text
[venv]                [venv_thoughtviz_gpu]         [venv]
discover assets  ->   ThoughtViz generation    ->   summary/seg/metrics/tables/panels
build manifest   ->   from same manifest       ->   aggregate final outputs
DreamDiff+SAR-HM ->
```

Key idea: fairness comes from the **same manifest**, not from running in one env.

---

## 4) Delete old virtual environments safely

From the repo root in bash:

```bash
export REPO_ROOT="/workspace/DD-Trail"   # set to your actual clone path
cd "$REPO_ROOT"
ls -d venv* 2>/dev/null || true
rm -rf venv venv_thoughtviz venv_thoughtviz_gpu venv_unified
```

Expected state after cleanup:

- no `venv*` folders
- source code, `datasets/`, `exps/`, and `configs/` untouched

---

## 5) Recreate fresh environments

Use `python3` if your system does not provide `python` for the desired version.

### A) Main environment (`venv`)

```bash
export REPO_ROOT="/workspace/DD-Trail"
cd "$REPO_ROOT"
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
deactivate
```

### B) ThoughtViz environment (`venv_thoughtviz_gpu`)

```bash
export REPO_ROOT="/workspace/DD-Trail"
cd "$REPO_ROOT"
python3 -m venv venv_thoughtviz_gpu
source venv_thoughtviz_gpu/bin/activate
python -m pip install --upgrade pip setuptools wheel
deactivate
```

---

## 6) Install dependencies correctly

## Main env installs (DreamDiffusion/SAR-HM + benchmark orchestration)

```bash
export REPO_ROOT="/workspace/DD-Trail"
cd "$REPO_ROOT"
source venv/bin/activate
pip install -r requirements.txt
pip install pyyaml
deactivate
```

Notes:

- `requirements.txt` is the torch-first stack.
- `pyyaml` is required by benchmark config loading.

## ThoughtViz env installs (ThoughtViz only)

```bash
export REPO_ROOT="/workspace/DD-Trail"
cd "$REPO_ROOT"
source venv_thoughtviz_gpu/bin/activate
pip install -r requirements-thoughtviz.txt
pip install -r requirements-thoughtviz-benchmark.txt
```

If ThoughtViz must run on GPU in this env, install TF GPU-compatible stack (choose one path only):

- If you already validated a working TF GPU combo in this env, keep it.
- Otherwise install the repo's TF 2.13 GPU runtime helpers:

```bash
pip install -r requirements-thoughtviz-gpu.txt
```

Then deactivate:

```bash
deactivate
```

Important:

- Do not install `requirements.txt` into `venv_thoughtviz_gpu`.
- Do not install `requirements-thoughtviz*.txt` into `venv`.

---

## 7) Verify each environment before expensive runs

On Linux, `PYTHONPATH` uses `:` between entries.

## A) Verify main env

```bash
export REPO_ROOT="/workspace/DD-Trail"
cd "$REPO_ROOT"
source venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import benchmark.orchestrate_all as b; print('orchestrate_all ok')"
python -c "import yaml; print('yaml ok')"
deactivate
```

## B) Verify ThoughtViz env

```bash
export REPO_ROOT="/workspace/DD-Trail"
cd "$REPO_ROOT"
source venv_thoughtviz_gpu/bin/activate
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"
python -c "import tensorflow as tf; print('tf', tf.__version__); print(tf.config.list_physical_devices('GPU'))"
python -c "import thoughtviz_integration.model_wrapper as m; print('thoughtviz wrapper ok')"
deactivate
```

If TF shows no GPU, you can still run (CPU fallback), but it may be slower.

---

## 8) Shared manifest concept (critical)

A manifest is created per dataset at:

- `results/benchmark_unified/<run_name>/benchmark_outputs/<dataset>/manifest.json`

It stores:

- sample IDs
- EEG file paths
- GT image paths
- labels/metadata
- fixed sample ordering

Why it matters:

- Every model uses the same sample list/order.
- Fair comparison does not depend on sharing one Python env.

---

## 9) Correct execution order (recommended)

Use this preamble in **each** shell session (each block below repeats it so you can copy one stage alone):

```bash
export REPO_ROOT="/workspace/DD-Trail"
cd "$REPO_ROOT"
```

### Stage A: discover assets and config paths

```bash
export REPO_ROOT="/workspace/DD-Trail"
cd "$REPO_ROOT"
source venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"
python scripts/discover_benchmark_assets.py
deactivate
```

Check `configs/benchmark_unified.yaml` and confirm:

- `paths.baseline_ckpt`
- `paths.sarhm_ckpt`
- `paths.sarhm_proto`
- `paths.thoughtviz_eeg_model_path`
- `paths.thoughtviz_gan_model_path`

For `imagenet_eeg`, prefer image-domain ThoughtViz checkpoints (not char/digit checkpoints).

### Stage B: run DreamDiffusion + SAR-HM (main env)

Edit `configs/benchmark_unified.yaml`:

```yaml
models:
  - dreamdiffusion
  - sarhm
```

Run:

```bash
export REPO_ROOT="/workspace/DD-Trail"
cd "$REPO_ROOT"
source venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"
python -m benchmark.orchestrate_all --config configs/benchmark_unified.yaml --dataset both --max_samples 50 --skip_metrics --skip_tables --skip_panels
deactivate
```

This also builds the canonical manifest and writes DD/SAR-HM images.

### Stage C: run ThoughtViz separately (ThoughtViz env)

Edit `configs/benchmark_unified.yaml`:

```yaml
models:
  - thoughtviz
```

Run orchestrator from main env, but force ThoughtViz python:

```bash
export REPO_ROOT="/workspace/DD-Trail"
cd "$REPO_ROOT"
source venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"
tvPy="$REPO_ROOT/venv_thoughtviz_gpu/bin/python"
python -m benchmark.orchestrate_all \
  --config configs/benchmark_unified.yaml \
  --dataset both \
  --max_samples 50 \
  --thoughtviz_python "$tvPy" \
  --skip_metrics --skip_tables --skip_panels
deactivate
```

Why this works:

- Orchestrator stays in `venv`.
- ThoughtViz generation is launched as subprocess in `venv_thoughtviz_gpu`.
- Both use same manifest/output tree.

### Stage D: post-processing and final aggregation (main env)

Set all three models back:

```yaml
models:
  - thoughtviz
  - dreamdiffusion
  - sarhm
```

Run full pass (it will reuse existing images and compute evaluation artifacts):

```bash
export REPO_ROOT="/workspace/DD-Trail"
cd "$REPO_ROOT"
source venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"
tvPy="$REPO_ROOT/venv_thoughtviz_gpu/bin/python"
python -m benchmark.orchestrate_all \
  --config configs/benchmark_unified.yaml \
  --dataset both \
  --thoughtviz_python "$tvPy"
deactivate
```

Alternative one-command wrapper:

```bash
export REPO_ROOT="/workspace/DD-Trail"
cd "$REPO_ROOT"
source venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"
python scripts/run_full_benchmark_pipeline.py --max_samples 50
deactivate
```

Note: `benchmark.orchestrate_all` auto-selects `venv_thoughtviz_gpu/bin/python` or `venv_thoughtviz/bin/python` (POSIX layout) or the Windows `Scripts\python.exe` paths when `--thoughtviz_python` is omitted. Pass `--thoughtviz_python` explicitly if your layout differs. The orchestrator may expand `%CD%` in arguments on Windows; on Linux use absolute paths or `"$REPO_ROOT/venv_thoughtviz_gpu/bin/python"`.

---

## 10) Output structure

Main tree:

```text
results/benchmark_unified/<run_name>/
  benchmark_outputs/
    imagenet_eeg/
      manifest.json
      _prepared/
      sample_<id>/
        ground_truth.png
        thoughtviz.png
        dreamdiffusion.png
        sarhm.png
        metadata.json
        summaries/
        segmentation/
      panels/
    thoughtviz/
      manifest.json
      _prepared/
      sample_<id>/...
      panels/
  summary_metrics/
    imagenet_eeg/
    thoughtviz/
  segmentation_metrics/
    imagenet_eeg/
    thoughtviz/
  tables/
    table_imagenet_eeg.csv
    table_thoughtviz.csv
    table_summary_comparison.csv
    table_segmentation_comparison.csv
  combined/
    logs/
    final_report.json
```

---

## 11) How results are combined correctly without one env

You do **not** need one env for fair benchmarking.

Fairness is preserved by:

- same manifest (`sample_id`, ordering, EEG/GT references)
- same output format (`sample_<id>/<model>.png`)
- same evaluation scripts reading files from disk

Metrics/tables compare generated images and metadata, not live model objects.

---

## 12) How to inspect failures

Check per-sample status:

- `.../sample_<id>/metadata.json`
- field: `model_status.<model>.status`
- field: `model_status.<model>.reason`

Where to inspect pipeline-level issues:

- `results/benchmark_unified/<run_name>/combined/logs/`
- `results/benchmark_unified/combined/logs/asset_discovery.json`
- `results/benchmark_unified/<run_name>/combined/final_report.json`

Model-specific clues:

- ThoughtViz fail: missing output or TF/CUDA issues in reason text
- summary fallback: logs mention Florence fallback model
- segmentation fallback: logs/JSON show detector fallback or empty detections

---

## 13) Common issues and fixes

1. Torch CUDA unavailable in main env
   - Reinstall torch with correct CUDA wheel index for your machine.
   - Re-check `torch.cuda.is_available()`.

2. TensorFlow GPU unavailable in ThoughtViz env
   - Verify only ThoughtViz packages are installed there.
   - Reinstall TF GPU dependencies for that env only.
   - Re-test `tf.config.list_physical_devices('GPU')`.

3. ThoughtViz CPU NHWC/NCHW Conv2D error
   - Keep split-env workflow.
   - Ensure current patched wrapper is used.
   - Prefer GPU TF in ThoughtViz env for speed/stability.

4. Wrong ThoughtViz checkpoint auto-selected
   - Inspect `configs/benchmark_unified.yaml` after discovery.
   - For `imagenet_eeg`, use image-domain EEG/GAN checkpoints.

5. Summary model primary load fails
   - Pipeline has fallback chain; check logs for selected fallback.

6. Segmentation detector unavailable
   - Pipeline tries fallback detector then empty detections fallback.

7. Missing/corrupt dataset images
   - Dataset loader skips bad items and logs warnings.
   - Confirm dataset paths in config.

8. Wrong environment active
   - Before long jobs, run `which python` and `readlink -f "$(which python)"` to confirm you are using the intended venv.
   - For ThoughtViz stage, ensure `--thoughtviz_python` points to `venv_thoughtviz_gpu/bin/python` (or let the orchestrator auto-detect).

9. Missing `PYTHONPATH`
   - In bash: `export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"`

---

## 14) Safe rerun workflow

Dry-run cleanup candidate list:

```bash
export REPO_ROOT="/workspace/DD-Trail"
cd "$REPO_ROOT"
source venv/bin/activate
python scripts/clean_benchmark_outputs.py
deactivate
```

Delete generated benchmark artifacts:

```bash
export REPO_ROOT="/workspace/DD-Trail"
cd "$REPO_ROOT"
source venv/bin/activate
python scripts/clean_benchmark_outputs.py --yes
deactivate
```

Rerun one dataset only:

```bash
export REPO_ROOT="/workspace/DD-Trail"
cd "$REPO_ROOT"
source venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"
tvPy="$REPO_ROOT/venv_thoughtviz_gpu/bin/python"
python -m benchmark.orchestrate_all --config configs/benchmark_unified.yaml --dataset imagenet_eeg --thoughtviz_python "$tvPy"
deactivate
```

Smoke test before full run:

```bash
export REPO_ROOT="/workspace/DD-Trail"
cd "$REPO_ROOT"
source venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"
tvPy="$REPO_ROOT/venv_thoughtviz_gpu/bin/python"
python -m benchmark.orchestrate_all --config configs/benchmark_unified.yaml --dataset both --max_samples 10 --thoughtviz_python "$tvPy"
deactivate
```

Avoid cache redownloads:

- Keep same machine/user cache directories (Hugging Face, pip cache).
- Reuse existing environments when they are known good.

---

### Full benchmark one-liner style (copy-paste)

```bash
export REPO_ROOT="/workspace/DD-Trail"
cd "$REPO_ROOT"
source venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"
tv="$REPO_ROOT/venv_thoughtviz_gpu/bin/python"
python -m benchmark.orchestrate_all \
  --config configs/benchmark_unified.yaml \
  --dataset both \
  --thoughtviz_python "$tv"
deactivate
```

If the repository is not at `/workspace/DD-Trail`, set `REPO_ROOT` to your clone path (for example `export REPO_ROOT="$(pwd)"` from inside the repo).
