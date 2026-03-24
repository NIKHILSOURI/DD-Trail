# Unified Benchmark Pipeline Guide

This guide documents the thesis-ready end-to-end benchmark pipeline in this repository.

## 1) Core Pipeline Files

- Config: `configs/benchmark_unified.yaml`
- Asset discovery: `scripts/discover_benchmark_assets.py`
- Safe cleanup: `scripts/clean_benchmark_outputs.py`
- Unified inference: `benchmark/run_unified_inference.py`
- Grids/panels: `benchmark/make_comparison_grids.py`
- Metrics + tables: `benchmark/compute_all_metrics.py`
- Master orchestration:
  - `scripts/run_full_benchmark_pipeline.py`
  - `scripts/run_full_benchmark_pipeline.sh`

## 2) Environment Setup

Use one venv (normal project venv is supported):

```bash
export REPO_ROOT=/workspace/project/DREAMDIFFUSION_RUNPOD
cd "$REPO_ROOT"
source "$REPO_ROOT/venv/bin/activate"
pip install -r requirements_unified.txt
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/code:$REPO_ROOT/benchmark"
python scripts/test_unified_imports.py
```

The unified file intentionally pins PyTorch to `2.1.2` and `typing_extensions<4.6` to stay compatible with ThoughtViz TensorFlow 2.13 in one environment.

## 3) Checkpoint and Dataset Discovery

```bash
python scripts/discover_benchmark_assets.py
```

This updates `configs/benchmark_unified.yaml` with discovered paths and writes:

- `results/benchmark_unified/combined/logs/asset_discovery.json`

## 4) Safe Cleanup (Generated Artifacts Only)

Preview only:

```bash
python scripts/clean_benchmark_outputs.py
```

Delete generated benchmark artifacts:

```bash
python scripts/clean_benchmark_outputs.py --yes
```

The cleanup script filters to generated-output-like paths and avoids checkpoint/source/dataset folders.

## 5) Run Full Pipeline

```bash
bash scripts/run_full_benchmark_pipeline.sh --max_samples 50
```

Or explicitly:

```bash
python scripts/run_full_benchmark_pipeline.py --config configs/benchmark_unified.yaml --max_samples 50
```

## 6) Inference Protocol

Per dataset and sample, outputs are standardized under:

- `results/benchmark_unified/<run_name>/benchmark_outputs/<dataset>/sample_<id>/`

Files include:

- `ground_truth.png`
- `thoughtviz.png`
- `dreamdiffusion.png`
- `sarhm.png`
- `metadata.json`

## 7) Metrics

The pipeline computes and aggregates:

- MSE / SSIM / PCC
- CLIP similarity (from existing metric utility)
- FID (if available in `utils_eval.compute_metrics`)
- Top-k retriever metrics (from existing metric utility when available)
- Instance segmentation label comparison
- Image summary/caption comparison
- Inference timing summaries

## 8) Segmentation Method

Implemented in existing benchmark modules:

- `benchmark/segmentation_model.py`
- `benchmark/segmentation_runner.py`

Current approach:

- Zero-shot detection via Grounding DINO pipeline
- SAM adapter attempt; bbox-mask fallback if unavailable
- Label precision/recall/F1, label IoU, bbox/mask overlap-derived scores, hallucination/omission rates

## 9) Image Summary Method

Implemented in:

- `benchmark/summary_model.py`
- `benchmark/summary_runner.py`

Current approach:

- Florence-2 captioning (with fallback to BLIP)
- Sentence embedding similarity + CLIP text-image score when dependencies available
- Missing optional dependencies are handled gracefully and recorded in status columns

## 10) Tables and Outputs

Combined outputs root:

- `results/benchmark_unified/<run_name>/combined/`

Key paths:

- Tables: `combined/tables/`
- Figures: `combined/figures/`
- Logs: `combined/logs/`
- Timing: `combined/timing/`
- Final report: `combined/final_report.json`

Tables are exported as CSV + Markdown + LaTeX where generated:

- `imagenet_benchmark.*`
- `thoughtviz_benchmark.*`
- `inference_timing.*`
- `model_comparison.*`

## 11) Troubleshooting

- ThoughtViz mismatch warnings on ImageNet:
  - Prefer image-family ThoughtViz checkpoints for ImageNet EEG.
  - For strict blocking behavior, set `evaluation.thoughtviz_strict_checkpoint_match: true`.

- Slow first run:
  - Summary and segmentation models may download weights on first execution.

- Optional metric dependency missing:
  - Pipeline now records missing dependency status and continues instead of hard failing.
