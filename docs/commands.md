# Benchmark Commands (ThoughtViz, DreamDiffusion Baseline, SAR-HM)

**Scope:** This document covers only **ThoughtViz**, **DreamDiffusion baseline**, and **DreamDiffusion + SAR-HM**. No SAR-HM++.

All commands assume you are at the **repository root** and that `code` is on `PYTHONPATH` (e.g. `pip install -e ./code` or `export PYTHONPATH=code:$PYTHONPATH`).

---

## A. Sanity Testing

### Dataset check
```bash
# Ensure ImageNet-EEG data and paths exist
export IMAGENET_PATH=/path/to/ILSVRC2012   # or your ImageNet root
python -c "
from pathlib import Path
import sys
sys.path.insert(0, 'code')
sys.path.insert(0, 'benchmark')
from benchmark.benchmark_config import BenchmarkConfig
from benchmark.dataset_registry import get_dataset
c = BenchmarkConfig()
c.imagenet_path = '$IMAGENET_PATH'
c.imagenet_eeg_eeg_path = 'datasets/eeg_5_95_std.pth'
c.imagenet_eeg_splits_path = 'datasets/block_splits_by_image_single.pth'
s = get_dataset('imagenet_eeg', c, max_samples=2)
print('ImageNet-EEG samples:', len(s))
"
```

### Model loading check
```bash
# DreamDiffusion baseline (requires checkpoint)
export BASELINE_CKPT=exps/results/generation/DD-MM-YYYY-HH-MM-SS/checkpoint.pth
python tests/test_full_pipeline_sanity.py
```

### One-batch inference check
```bash
# Set checkpoint and ImageNet path, then run sanity (includes one-sample inference)
export IMAGENET_PATH=/path/to/ILSVRC2012
export BASELINE_CKPT=path/to/baseline/checkpoint.pth
python tests/test_full_pipeline_sanity.py
```

### Metrics pipeline check
```bash
python tests/test_full_pipeline_sanity.py
# Includes tiny metric pass (compute_metrics on 2 fake images)
```

### Comparison pipeline check
```bash
# After running benchmark once, run metrics on output dir
python -c "
import sys; sys.path.insert(0, 'code'); sys.path.insert(0, 'benchmark')
from benchmark.metrics_runner import run_core_metrics
from pathlib import Path
r = run_core_metrics('results/benchmark_outputs', 'imagenet_eeg')
print('Metrics:', r)
"
```

---

## B. Small Benchmark Testing (10–20 samples)

### All three models on ImageNet-EEG (10 samples)
```bash
export IMAGENET_PATH=/path/to/ILSVRC2012
python -m benchmark.compare_all_models \
  --dataset imagenet_eeg \
  --max_samples 10 \
  --run_name smoke_test \
  --imagenet_path "$IMAGENET_PATH" \
  --baseline_ckpt path/to/baseline/checkpoint.pth \
  --sarhm_ckpt path/to/sarhm/checkpoint.pth \
  --sarhm_proto path/to/sarhm/prototypes.pt
```

### All three models on ThoughtViz (20 samples)
```bash
python -m benchmark.compare_all_models \
  --dataset thoughtviz \
  --max_samples 20 \
  --run_name smoke_thoughtviz \
  --thoughtviz_data_dir code/ThoughtViz/data \
  --thoughtviz_image_dir code/ThoughtViz/training/images \
  --baseline_ckpt path/to/baseline/checkpoint.pth \
  --sarhm_ckpt path/to/sarhm/checkpoint.pth \
  --sarhm_proto path/to/sarhm/prototypes.pt
```

### Single dataset, single model
```bash
# ThoughtViz only, 10 samples
python -m benchmark.compare_all_models --dataset thoughtviz --max_samples 10 --models thoughtviz --run_name tv_only
```

---

## C. ThoughtViz Training / Testing

### Smoke test (inference only, from ThoughtViz repo)
```bash
cd code/ThoughtViz
# Ensure data.pkl and models exist (see ThoughtViz README)
python testing/test.py
# Or from repo root with Keras available:
# python -c "
# import sys; sys.path.insert(0, 'code'); sys.path.insert(0, 'code/ThoughtViz')
# from testing.test import Tests
# t = Tests(); t.test_deligan_final('models/gan_models/final/image/generator.model', 'models/eeg_models/image/run_final.h5', 'data/eeg/image/data.pkl')
# "
```

### Small training run (ThoughtViz; from ThoughtViz dir)
```bash
cd code/ThoughtViz
# See training/thoughtviz_with_eeg.py for args; dataset 1 = image data
python training/thoughtviz_with_eeg.py  # adjust paths inside script or via args if supported
```

### Inference command (via benchmark wrapper)
```bash
export PYTHONPATH=code:benchmark:$PYTHONPATH
python -c "
from thoughtviz_integration.model_wrapper import ThoughtVizWrapper
from thoughtviz_integration.config import ThoughtVizConfig
import numpy as np
w = ThoughtVizWrapper(ThoughtVizConfig())
w.load_pretrained()
out = w.generate_from_eeg(np.random.randn(2, 128).astype(np.float32), num_samples=1)
print('Generated', len(out), 'images')
"
```

---

## D. DreamDiffusion Baseline Commands

### Sanity
```bash
python tests/test_full_pipeline_sanity.py
# Set BASELINE_CKPT and IMAGENET_PATH for full checks
```

### Small test (Stage C, 20 images)
```bash
python code/gen_eval_eeg.py --dataset EEG \
  --model_path exps/results/generation/BASELINE_RUN/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --imagenet_path "$IMAGENET_PATH" \
  --num_samples 20
```

### Final training (Stage B, baseline)
```bash
python code/eeg_ldm.py --dataset EEG --run_mode baseline \
  --pretrain_mbm_path pretrains/eeg_pretain/checkpoint.pth \
  --imagenet_path "$IMAGENET_PATH" \
  --num_epoch 100
```

---

## E. SAR-HM Commands

### Sanity
```bash
export SARHM_CKPT=path/to/sarhm/checkpoint.pth
export SARHM_PROTO=path/to/sarhm/prototypes.pt
python tests/test_full_pipeline_sanity.py
```

### Small test (Stage C)
```bash
python code/gen_eval_eeg.py --dataset EEG \
  --model_path exps/results/generation/SARHM_RUN/checkpoint.pth \
  --proto_path exps/results/generation/SARHM_RUN/prototypes.pt \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --imagenet_path "$IMAGENET_PATH" \
  --num_samples 20
```

### Final training (Stage B, SAR-HM)
```bash
python code/eeg_ldm.py --dataset EEG --run_mode sarhm \
  --pretrain_mbm_path pretrains/eeg_pretain/checkpoint.pth \
  --imagenet_path "$IMAGENET_PATH" \
  --num_epoch 100
```

### Final evaluation (compare baseline vs SAR-HM)
```bash
python code/compare_eval.py --dataset EEG \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --baseline_ckpt exps/.../baseline/checkpoint.pth \
  --sarhm_ckpt exps/.../sarhm/checkpoint.pth \
  --sarhm_proto exps/.../sarhm/prototypes.pt \
  --n_samples 100 --ddim_steps 250 --out_dir results/compare_eval
```

---

## F. Comparison Commands (Unified Benchmark)

### Run all-model benchmark (both datasets, 20 samples each)
```bash
python -m benchmark.compare_all_models --dataset both --max_samples 20 \
  --imagenet_path "$IMAGENET_PATH" \
  --baseline_ckpt path/to/baseline/checkpoint.pth \
  --sarhm_ckpt path/to/sarhm/checkpoint.pth \
  --sarhm_proto path/to/sarhm/prototypes.pt \
  --run_name run_001
```

### Run metrics on benchmark outputs
```bash
python -c "
import sys; sys.path.insert(0, 'code'); sys.path.insert(0, 'benchmark')
from benchmark.metrics_runner import run_core_metrics
from pathlib import Path
for ds in ('imagenet_eeg', 'thoughtviz'):
    run_core_metrics('results/experiments/run_001/benchmark_outputs', ds)
"
```

### Run timing
```bash
python -c "
import sys; sys.path.insert(0, 'code'); sys.path.insert(0, 'benchmark')
from benchmark.benchmark_config import BenchmarkConfig
from benchmark.timing_runner import run_timing, save_timing_table
c = BenchmarkConfig(); c.imagenet_path = '$IMAGENET_PATH'; c.dreamdiffusion_baseline_ckpt = 'path/to/baseline.pth'; c.sarhm_ckpt = 'path/to/sarhm.pth'; c.sarhm_proto_path = 'path/to/prototypes.pt'
r = run_timing('imagenet_eeg', c, max_samples=5)
save_timing_table(r, Path('results/experiments/run_001/timing/timing.json'))
"
```

### Generate tables
```bash
python -c "
import sys; sys.path.insert(0, 'benchmark')
from pathlib import Path
from benchmark.table_generator import generate_all_tables
generate_all_tables(Path('results/experiments/run_001/benchmark_outputs'), Path('results/experiments/run_001/tables'))
"
```

### Generate visual panels
```bash
python -c "
import sys; sys.path.insert(0, 'benchmark')
from pathlib import Path
from benchmark.visualization_runner import run_visualization
run_visualization(Path('results/experiments/run_001/benchmark_outputs'), 'imagenet_eeg', max_panels=10)
run_visualization(Path('results/experiments/run_001/benchmark_outputs'), 'thoughtviz', max_panels=10)
"
```

---

## Notes

- Replace `path/to/baseline/checkpoint.pth`, `path/to/sarhm/checkpoint.pth`, and `path/to/sarhm/prototypes.pt` with your actual run directories.
- ThoughtViz requires Keras and the ThoughtViz repo under `code/ThoughtViz` (or `codes/ThoughtViz`) with data and models as per ThoughtViz README.
- MSC is not defined in the codebase; tables use "NA" for MSC until a project-specific definition is added.

---

## G. Mandatory Summary + Segmentation Commands

These are compulsory thesis evaluation components in this workflow.

### Summary-only smoke test
```bash
python tests/test_summary_pipeline.py
```

### Segmentation-only smoke test
```bash
python tests/test_segmentation_pipeline.py
```

### Full benchmark smoke test (generation + summary + segmentation)
```bash
python -m benchmark.compare_all_models --dataset imagenet_eeg --max_samples 5 \
  --imagenet_path "$IMAGENET_PATH" \
  --baseline_ckpt path/to/baseline/checkpoint.pth \
  --sarhm_ckpt path/to/sarhm/checkpoint.pth \
  --sarhm_proto path/to/sarhm/prototypes.pt \
  --run_name smoke_mandatory_eval \
  --summary_enabled true \
  --segmentation_enabled true
```

### Full benchmark strict mode (fail loudly if summary/segmentation unavailable)
```bash
python -m benchmark.compare_all_models --dataset both --max_samples 20 \
  --imagenet_path "$IMAGENET_PATH" \
  --baseline_ckpt path/to/baseline/checkpoint.pth \
  --sarhm_ckpt path/to/sarhm/checkpoint.pth \
  --sarhm_proto path/to/sarhm/prototypes.pt \
  --run_name run_mandatory_eval \
  --summary_enabled true \
  --segmentation_enabled true \
  --strict_eval true
```
