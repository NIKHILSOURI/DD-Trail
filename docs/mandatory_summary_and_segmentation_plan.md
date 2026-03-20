# Mandatory Summary + Segmentation Plan

## Scope lock

This implementation is thesis scope only:
- DreamDiffusion baseline
- DreamDiffusion + SAR-HM
- ThoughtViz

Out of scope:
- SAR-HM++

## What currently exists

- Unified benchmark generation exists in:
  - `benchmark/compare_all_models.py`
  - `benchmark/benchmark_runner.py`
  - `benchmark/dataset_registry.py`
  - `benchmark/model_registry.py`
  - `benchmark/output_standardizer.py`
- Standard sample outputs exist:
  - `results/.../benchmark_outputs/<dataset>/sample_<id>/ground_truth.png`
  - `thoughtviz.png`, `dreamdiffusion.png`, `sarhm.png`, `metadata.json`
- Core metrics runner exists:
  - `benchmark/metrics_runner.py` (SSIM/PCC/CLIP via `utils_eval.compute_metrics`)
- Table and visualization basic modules exist:
  - `benchmark/table_generator.py`
  - `benchmark/visualization_runner.py`

## What is missing

- Summary comparison is currently stubbed:
  - `benchmark/caption_eval.py` returns NA
- Segmentation comparison is currently stubbed:
  - `benchmark/segmentation_eval.py` returns NA
- No per-sample summary artifacts saved.
- No per-sample segmentation artifacts (masks/overlays/json) saved.
- No aggregate summary metrics CSV/JSON.
- No aggregate segmentation metrics CSV/JSON.
- No automatic integration into benchmark run.
- No thesis-ready setup doc for these dependencies/checkpoints.

## Modules to add/modify

### New modules

- `benchmark/summary_model.py`
  - Florence-2 wrapper + fallback caption model wrapper.
  - Local sentence-transformer embedding model for semantic similarity.
- `benchmark/summary_metrics.py`
  - Summary semantic cosine
  - CLIP text-image similarity
  - object mention overlap (P/R/F1)
  - attribute overlap
- `benchmark/summary_runner.py`
  - Per-sample summary generation and comparison
  - Aggregate summary CSV/JSON
- `benchmark/segmentation_model.py`
  - Grounding DINO detector adapter
  - SAM2 segmenter adapter
  - Label normalization + mask/overlay saving
- `benchmark/segmentation_metrics.py`
  - label-set precision/recall/F1/Jaccard
  - instance matching (greedy deterministic + IoU constraints)
  - mask IoU / Dice / bbox IoU
  - hallucination / omission rates
- `benchmark/segmentation_runner.py`
  - Per-sample segmentation inference and comparisons
  - Aggregate segmentation CSV/JSON

### Modified modules

- `benchmark/benchmark_config.py`
  - add mandatory summary/segmentation config fields and model/checkpoint paths
- `benchmark/compare_all_models.py`
  - add CLI flags and invoke mandatory summary/segmentation runs
- `benchmark/benchmark_runner.py`
  - keep generation flow, then trigger eval pipelines
- `benchmark/table_generator.py`
  - add summary and segmentation tables
- `benchmark/visualization_runner.py`
  - add summary and segmentation panel generation
- `docs/commands.md`
  - add summary-only, segmentation-only, and full smoke commands

### Replace/supersede stubs

- `benchmark/caption_eval.py`
- `benchmark/segmentation_eval.py`

These will become compatibility wrappers that call the full new runners.

## Config/checkpoint expectations

New explicit config fields:
- `florence2_model_id`
- `summary_sentence_model_id`
- `grounding_dino_model_id`
- `grounding_dino_checkpoint_path`
- `sam2_model_id`
- `sam2_config_path`
- `sam2_checkpoint_path`
- `summary_enabled` (default true)
- `segmentation_enabled` (default true)
- `strict_eval` (default false)

Checkpoint convention:
- `pretrains/eval_models/florence2/`
- `pretrains/eval_models/grounding_dino/`
- `pretrains/eval_models/sam2/`

Failure policy:
- If required model/checkpoint missing:
  - `strict_eval=true` => raise and stop
  - `strict_eval=false` => record failure in per-sample and aggregate reports, continue run

## Output schema

Per sample:

- `sample_<id>/summaries/`
  - `summary_gt.json`
  - `summary_thoughtviz.json`
  - `summary_dreamdiffusion.json`
  - `summary_sarhm.json`
  - `summary_comparison.json`
- `sample_<id>/segmentation/`
  - `gt/segmentation.json`, `gt/masks/*`, `gt/overlays/*`
  - `thoughtviz/...`
  - `dreamdiffusion/...`
  - `sarhm/...`
  - `segmentation_comparison.json`

Aggregate:

- `results/experiments/<run_name>/summary_metrics/<dataset>/summary_metrics.csv`
- `results/experiments/<run_name>/summary_metrics/<dataset>/summary_metrics.json`
- `results/experiments/<run_name>/segmentation_metrics/<dataset>/segmentation_metrics.csv`
- `results/experiments/<run_name>/segmentation_metrics/<dataset>/segmentation_metrics.json`

## Implementation notes

- Benchmark will call summary and segmentation pipelines by default (mandatory).
- Pipelines are deterministic where practical (seeded processing, deterministic parsing).
- Every failure is recorded with reason and sample/model context.
- Documentation additions:
  - `docs/summary_and_segmentation_setup.md`
  - `docs/thesis_summary_and_segmentation_reporting.md`
  - `docs/IMPLEMENTATION_DONE_SUMMARY.md`
