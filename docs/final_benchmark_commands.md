# Final Benchmark Commands (Beginner-Friendly)

This runbook executes a robust split-env benchmark:

- `venv` for DreamDiffusion + SAR-HM + aggregation
- `venv_thoughtviz` for ThoughtViz inference only

## 1) One-time environment setup

## Main env (`venv`)

```bash
export REPO_ROOT=/workspace/project/DREAMDIFFUSION_RUNPOD
cd "$REPO_ROOT"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ThoughtViz env (`venv_thoughtviz`)

```bash
cd "$REPO_ROOT"
python3 -m venv venv_thoughtviz
source venv_thoughtviz/bin/activate
pip install -r requirements-thoughtviz.txt
# Optional, only if ThoughtViz must use GPU:
# pip install -r requirements-thoughtviz-gpu.txt
deactivate
```

## 2) Configure benchmark assets

From main env:

```bash
cd "$REPO_ROOT"
source venv/bin/activate
python scripts/discover_benchmark_assets.py
```

Then verify `configs/benchmark_unified.yaml` has correct paths, especially:

- `paths.baseline_ckpt`
- `paths.sarhm_ckpt`
- `paths.sarhm_proto`
- `paths.thoughtviz_eeg_model_path`
- `paths.thoughtviz_gan_model_path` (must match image task, not char/digit mismatch)

## 3) Smoke test (both datasets, small sample count)

```bash
cd "$REPO_ROOT"
source venv/bin/activate
python -m benchmark.orchestrate_all \
  --config configs/benchmark_unified.yaml \
  --dataset both \
  --max_samples 10 \
  --thoughtviz_python "$REPO_ROOT/venv_thoughtviz/bin/python"
```

## 4) Full benchmark (thesis run)

```bash
cd "$REPO_ROOT"
source venv/bin/activate
python -m benchmark.orchestrate_all \
  --config configs/benchmark_unified.yaml \
  --dataset both \
  --thoughtviz_python "$REPO_ROOT/venv_thoughtviz/bin/python"
```

## 5) ThoughtViz-only run

```bash
cd "$REPO_ROOT"
source venv/bin/activate
python -m benchmark.orchestrate_all \
  --config configs/benchmark_unified.yaml \
  --dataset both \
  --thoughtviz_python "$REPO_ROOT/venv_thoughtviz/bin/python" \
  --skip_metrics --skip_tables --skip_panels
```

Then set models in config to only ThoughtViz:

```yaml
models:
  - thoughtviz
```

## 6) DreamDiffusion/SAR-HM-only run

Set models in config:

```yaml
models:
  - dreamdiffusion
  - sarhm
```

Then run:

```bash
cd "$REPO_ROOT"
source venv/bin/activate
python -m benchmark.orchestrate_all \
  --config configs/benchmark_unified.yaml \
  --dataset both \
  --max_samples 50
```

## 7) Where outputs go

Main output root:

```text
results/benchmark_unified/<run_name>/benchmark_outputs/
  imagenet_eeg/
    manifest.json
    _prepared/
    sample_<id>/
      ground_truth.png
      thoughtviz.png
      dreamdiffusion.png
      sarhm.png
      metadata.json
  thoughtviz/
    manifest.json
    _prepared/
    sample_<id>/
      ...
```

Tables:

```text
results/benchmark_unified/<run_name>/tables/
  table_imagenet_eeg.csv
  table_thoughtviz.csv
  table_summary_comparison.csv
  table_segmentation_comparison.csv
```

## 8) How to inspect failures

Per sample status is in:

- `.../sample_<id>/metadata.json`

Look for:

- `model_status.thoughtviz.status`
- `model_status.dreamdiffusion.status`
- `model_status.sarhm.status`
- `reason` fields for exact failure causes

## 9) Resume / rerun safely

- You can rerun the same command with the same `run_name`; statuses and outputs are updated in place.
- To avoid mixing old/new experiments, change `output.run_name` in `configs/benchmark_unified.yaml`.
- For fresh runs, optionally clean old outputs first with your cleanup script.

## 10) Recommended default command

```bash
cd "$REPO_ROOT"
source venv/bin/activate
python -m benchmark.orchestrate_all \
  --config configs/benchmark_unified.yaml \
  --dataset both \
  --thoughtviz_python "$REPO_ROOT/venv_thoughtviz/bin/python"
```
