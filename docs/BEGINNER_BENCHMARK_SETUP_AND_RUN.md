# Beginner Guide: Clean Setup and Full Benchmark Run (Split Environments)

This guide is the **recommended, safe workflow** for this repository.

It is written for beginners and assumes you are starting from a messy state (old/broken virtual environments, conflicting installs, partial runs).

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

From repo root:

```bash
cd /workspace/project/DREAMDIFFUSION_RUNPOD
```

See likely environment folders first:

```bash
ls -d venv* 2>/dev/null || true
```

Remove only environment folders (safe):

```bash
rm -rf venv venv_thoughtviz venv_thoughtviz_gpu venv_unified
```

Expected state after cleanup:

- no `venv*` folders
- source code, `datasets/`, `exps/`, and `configs/` untouched

---

## 5) Recreate fresh environments

### A) Main environment (`venv`)

```bash
cd /workspace/project/DREAMDIFFUSION_RUNPOD
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### B) ThoughtViz environment (`venv_thoughtviz_gpu`)

```bash
cd /workspace/project/DREAMDIFFUSION_RUNPOD
python3 -m venv venv_thoughtviz_gpu
source venv_thoughtviz_gpu/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

---

## 6) Install dependencies correctly

## Main env installs (DreamDiffusion/SAR-HM + benchmark orchestration)

```bash
cd /workspace/project/DREAMDIFFUSION_RUNPOD
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
cd /workspace/project/DREAMDIFFUSION_RUNPOD
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

## A) Verify main env

```bash
cd /workspace/project/DREAMDIFFUSION_RUNPOD
source venv/bin/activate
export PYTHONPATH="$PWD:$PWD/code:$PWD/benchmark"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import benchmark.orchestrate_all as b; print('orchestrate_all ok')"
python -c "import yaml; print('yaml ok')"
deactivate
```

## B) Verify ThoughtViz env

```bash
cd /workspace/project/DREAMDIFFUSION_RUNPOD
source venv_thoughtviz_gpu/bin/activate
export PYTHONPATH="$PWD:$PWD/code:$PWD/benchmark"
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

Set root once:

```bash
export REPO_ROOT=/workspace/project/DREAMDIFFUSION_RUNPOD
cd "$REPO_ROOT"
```

### Stage A: discover assets and config paths

```bash
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
source venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"
python -m benchmark.orchestrate_all \
  --config configs/benchmark_unified.yaml \
  --dataset both \
  --max_samples 50 \
  --thoughtviz_python "$REPO_ROOT/venv_thoughtviz_gpu/bin/python" \
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
source venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"
python -m benchmark.orchestrate_all \
  --config configs/benchmark_unified.yaml \
  --dataset both \
  --thoughtviz_python "$REPO_ROOT/venv_thoughtviz_gpu/bin/python"
deactivate
```

Alternative one-command wrapper:

```bash
source venv/bin/activate
bash scripts/run_full_benchmark_pipeline.sh --max_samples 50
deactivate
```

Note: wrapper defaults ThoughtViz python to `venv_thoughtviz/bin/python`. If your env name is `venv_thoughtviz_gpu`, use `benchmark.orchestrate_all` directly (recommended).

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
   - Always run `which python` before long jobs.
   - For ThoughtViz stage, ensure `--thoughtviz_python` points to `venv_thoughtviz_gpu/bin/python`.

9. Missing `PYTHONPATH`
   - Export:
     - `export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"`

---

## 14) Safe rerun workflow

Dry-run cleanup candidate list:

```bash
cd /workspace/project/DREAMDIFFUSION_RUNPOD
source venv/bin/activate
python scripts/clean_benchmark_outputs.py
deactivate
```

Delete generated benchmark artifacts:

```bash
source venv/bin/activate
python scripts/clean_benchmark_outputs.py --yes
deactivate
```

Rerun one dataset only:

```bash
source venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"
python -m benchmark.orchestrate_all --config configs/benchmark_unified.yaml --dataset imagenet_eeg --thoughtviz_python "$REPO_ROOT/venv_thoughtviz_gpu/bin/python"
deactivate
```

Smoke test before full run:

```bash
source venv/bin/activate
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"
python -m benchmark.orchestrate_all --config configs/benchmark_unified.yaml --dataset both --max_samples 10 --thoughtviz_python "$REPO_ROOT/venv_thoughtviz_gpu/bin/python"
deactivate
```

Avoid cache redownloads:

- Keep same machine/user cache directories (Hugging Face, pip cache).
- Reuse existing environments when they are known good.

---

## 15) Final best-practice recommendation

Always do these four things:

1. Use split environments (`venv` + `venv_thoughtviz_gpu`).
2. Use manifest-based generation for fair sample alignment.
3. Verify both environments before expensive runs.
4. Do not force DreamDiffusion and ThoughtViz into one pip environment.

This is the most reliable way to get reproducible, conflict-free benchmark results in this repository.

