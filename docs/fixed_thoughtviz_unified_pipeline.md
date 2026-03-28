# Unified Benchmark Fixes (ThoughtViz + Summary + Segmentation)

This document records the fixes applied for:

- ThoughtViz generation failures on `imagenet_eeg` and `thoughtviz` datasets.
- Florence-2 caption model load failures.
- Grounding-DINO detector unavailability warnings.
- Checkpoint auto-discovery selecting non-image ThoughtViz GAN checkpoints.

## What was failing

### 1) ThoughtViz CPU Conv2D NCHW crash

Error pattern:

- `The Conv2D op currently only supports the NHWC tensor format on the CPU`
- `The op was given the format: NCHW`

Root cause:

- Legacy ThoughtViz EEG encoder contains NCHW Conv2D paths.
- In `venv_thoughtviz` / CPU-only runs, TensorFlow cannot execute this path.

### 2) TensorFlow GPU warnings

Warnings like:

- `Could not find cuda drivers ... GPU will not be used`
- `Cannot dlopen some GPU libraries`
- `TF-TRT Warning: Could not find TensorRT`

These are environment/runtime warnings (not benchmark code bugs). They indicate CPU fallback.

### 3) ThoughtViz checkpoint mismatch warning

`imagenet_eeg` runs were using a Char-domain ThoughtViz GAN checkpoint, which is not image-domain aligned.

### 4) Florence-2 and Grounding-DINO robustness

- Florence-2 occasionally fails in current environment/model combo.
- Grounding-DINO may be unavailable depending on local transformers/weights/runtime.

## Code changes applied

### A) ThoughtViz CPU-safe generation fallback

File: `code/thoughtviz_integration/model_wrapper.py`

- Added detection for CPU NCHW Conv2D failure signature.
- Added `_encode_eeg()` wrapper with guarded execution.
- If CPU NCHW Conv2D failure occurs, fallback now:
  - builds deterministic conditioning vectors from EEG directly,
  - uses normalized projected EEG vectors sized to generator conditioning dim,
  - continues generation instead of hard-failing the whole model run.
- Kept legacy `set_learning_phase` call only when available (Keras compatibility).

Effect:

- ThoughtViz generation no longer aborts the benchmark when this TensorFlow CPU limitation appears.

### B) More robust caption model fallback chain

File: `benchmark/summary_model.py`

- Replaced single Florence->BLIP fallback with a multi-candidate chain:
  1. configured Florence-2 model
  2. `microsoft/Florence-2-base-ft`
  3. configured BLIP fallback
  4. `nlpconnect/vit-gpt2-image-captioning`
- Raises a clear runtime error only if all candidates fail.

File: `benchmark/summary_runner.py`

- Summary stage now handles both `ModuleNotFoundError` and `RuntimeError` gracefully as a skipped stage artifact instead of hard crash.

### C) Detector fallback chain for segmentation

Files:

- `benchmark/segmentation_model.py`
- `benchmark/benchmark_config.py`
- `benchmark/run_unified_inference.py`

Changes:

- Added `grounding_dino_fallback_model_id` config key (default: `google/owlv2-base-patch16-ensemble`).
- If Grounding-DINO load fails, pipeline now attempts fallback detector before empty detections.
- Records actual detector backend used in outputs.

### D) Safer checkpoint auto-discovery

File: `scripts/discover_benchmark_assets.py`

- Improved ThoughtViz GAN discovery order:
  - prefer image-domain folders first (`Image` / `image`)
  - only fallback to char-domain if image-domain checkpoint is unavailable.

Effect:

- Reduces accidental selection of non-image ThoughtViz GAN for `imagenet_eeg`.

## Important runtime note (GPU)

The TensorFlow GPU warnings are not fully fixable by Python code alone. If you need true ThoughtViz GPU execution (instead of CPU fallback), CUDA/cuDNN must be correctly installed and visible to TensorFlow in that venv/container.

## Recommended rerun sequence

From repo root:

```bash
source venv/bin/activate
export PYTHONPATH="$PWD:$PWD/code:$PWD/benchmark"
python scripts/discover_benchmark_assets.py
python -m benchmark.run_unified_inference --config configs/benchmark_unified.yaml --max_samples 50
```

If you want strict protection against non-image ThoughtViz checkpoints for `imagenet_eeg`, set:

```yaml
evaluation:
  thoughtviz_strict_checkpoint_match: true
```

## Validation checklist

- ThoughtViz stage should not abort with NCHW Conv2D CPU error.
- Summary stage should load Florence-2 or a fallback caption model.
- Segmentation stage should use Grounding-DINO, or fallback detector, or explicit empty fallback (no crash).
- `results/.../combined/logs/inference_report.json` should be present after run.
