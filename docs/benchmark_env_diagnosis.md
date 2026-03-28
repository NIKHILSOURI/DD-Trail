# Benchmark Environment Diagnosis and Fix

## Current Architecture (Before Fix)

The previous benchmark flow was "unified" logically but mostly single-process in practice:

- `benchmark.compare_all_models` / `benchmark.run_unified_inference` loaded datasets and all model paths in one Python runtime.
- ThoughtViz, DreamDiffusion, SAR-HM, summary, segmentation, and later metrics could execute in tightly coupled steps.
- Sample failures were not consistently recorded per model in `metadata.json`.

## Confirmed Root Causes (From Code Inspection)

1. **Environment boundary conflict is real**
   - ThoughtViz requires TensorFlow/Keras 2.13 stack.
   - DreamDiffusion/SAR-HM are torch-first.
   - Running all model frameworks in one process is brittle and host-dependent.

2. **ImageNet -> ThoughtViz adaptation is shape-compatible but semantically mismatched**
   - `benchmark.model_registry._prepare_eeg_for_thoughtviz` resizes arbitrary EEG to `(14, 32, 1)`.
   - This enables execution but does not guarantee meaningful conditioning quality on ImageNet-EEG.

3. **Checkpoint mismatch risk existed in config/flow**
   - The old config may point ThoughtViz GAN to a non-image family path (`Char`) while running image datasets.
   - Existing guard was warning-only by default.

4. **Failure visibility was incomplete**
   - Missing generated files or invalid generations were not always captured as explicit per-sample/model statuses.
   - Metrics could proceed on whatever files existed, reducing transparency.

5. **Silent filtering exists at dataset level**
   - ImageNet-EEG loader intentionally filters invalid indices (missing/corrupt images, invalid EEG lengths).
   - This behavior is expected, but it needed explicit manifesting and status clarity for reproducibility.

## Why a Single In-Process Benchmark Is Not Robust

A strict one-process benchmark that imports TensorFlow + modern torch ecosystem + all optional post-processing reliably across machines is unrealistic for this repo's current dependency constraints. The robust choice is:

- one logical benchmark
- multiple runtime environments
- disk-based handoff (manifest + outputs + status)

## Implemented Fix

## 1) Canonical sample manifest

Added `benchmark/build_manifest.py`:

- Builds deterministic per-dataset manifest from benchmark datasets.
- Writes neutral prepared EEG files (`.npy`) under `benchmark_outputs/<dataset>/_prepared/`.
- Stores `manifest.json` with sample order and metadata.
- Initializes per-sample model status entries as `pending`.

## 2) ThoughtViz runtime isolation

Added `benchmark/run_thoughtviz_from_manifest.py`:

- Runs only ThoughtViz inference from manifest in `venv_thoughtviz`.
- Avoids DreamDiffusion/SAR-HM imports.
- Reads prepared EEG files, generates outputs, validates outputs, writes `thoughtviz.png`.
- Updates per-sample status with exact failure reason.

## 3) Explicit status tracking + image validation

Added `benchmark/status_utils.py`:

- Uniform metadata status updater per sample/model (`pending|success|failed` + reason).
- Basic image integrity checks (finite values, shape/channels, non-trivial variance).

Updated `benchmark/benchmark_runner.py`:

- On model load/generate failures: marks all relevant samples as failed with reasons.
- On per-sample output: validates generated image and records status.
- No implicit "success by existence"; status is explicit.

## 4) Orchestrated split-env benchmark

Added `benchmark/orchestrate_all.py`:

- Main env builds manifest and runs DreamDiffusion/SAR-HM.
- Calls ThoughtViz stage as subprocess in `venv_thoughtviz` Python using the same manifest.
- Runs summary, segmentation, core metrics, visualization, and tables from disk outputs.
- Keeps benchmark logically unified while respecting environment boundaries.

## 5) Pipeline script alignment

Updated `scripts/run_full_benchmark_pipeline.py`:

- Uses `benchmark.orchestrate_all` as the core stage.
- Keeps a cleaner end-to-end control flow.

## Files Changed

- `benchmark/status_utils.py` (new)
- `benchmark/build_manifest.py` (new)
- `benchmark/run_thoughtviz_from_manifest.py` (new)
- `benchmark/orchestrate_all.py` (new)
- `benchmark/benchmark_runner.py` (updated)
- `scripts/run_full_benchmark_pipeline.py` (updated)

## Resulting Execution Model

- **Main env (`venv`)**: dataset loading, manifest creation, DreamDiffusion/SAR-HM generation, post-processing, metrics/tables/panels.
- **ThoughtViz env (`venv_thoughtviz`)**: ThoughtViz-only inference from neutral prepared inputs.
- **Shared output tree**: canonical manifest + per-sample outputs + explicit per-model status in metadata.

This is reproducible, thesis-safe, and not dependent on lucky mixed imports.
