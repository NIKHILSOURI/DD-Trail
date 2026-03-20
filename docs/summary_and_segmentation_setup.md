# Summary and Segmentation Setup (Mandatory)

This setup is for mandatory thesis evaluation components:
- Image summary comparison
- Instance segmentation label/mask comparison

Scope: DreamDiffusion baseline, DreamDiffusion + SAR-HM, ThoughtViz only.

## 1) Required Python packages

Install in your benchmark environment:

```bash
pip install transformers sentence-transformers pillow numpy scipy scikit-image
pip install torch torchvision
```

For Grounding DINO via Hugging Face pipeline and SAM adapter:

```bash
pip install transformers accelerate
```

Optional (if you have a dedicated SAM2 package setup):

```bash
# install your SAM2 package according to your environment
# and configure sam2_model_id / sam2_checkpoint_path in benchmark config
```

## 2) Model/checkpoint convention

Use this structure:

```text
pretrains/
  eval_models/
    florence2/
    grounding_dino/
    sam2/
```

Default model IDs used by benchmark config:
- Florence-2: `microsoft/Florence-2-base`
- Sentence embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Grounding DINO: `IDEA-Research/grounding-dino-base`
- SAM adapter default: `facebook/sam-vit-base`

## 3) How to verify setup

### Summary pipeline smoke test
```bash
python tests/test_summary_pipeline.py
```

### Segmentation pipeline smoke test
```bash
python tests/test_segmentation_pipeline.py
```

### Full smoke benchmark
```bash
python -m benchmark.compare_all_models --dataset imagenet_eeg --max_samples 5 \
  --imagenet_path <IMAGENET_PATH> \
  --baseline_ckpt <BASELINE_CKPT> \
  --sarhm_ckpt <SARHM_CKPT> \
  --sarhm_proto <SARHM_PROTO> \
  --run_name smoke_mandatory_eval \
  --summary_enabled true \
  --segmentation_enabled true
```

Expected outputs:
- Per-sample summary JSON files under `sample_<id>/summaries/`
- Per-sample segmentation JSON/masks/overlays under `sample_<id>/segmentation/`
- Aggregate files:
  - `results/experiments/<run_name>/summary_metrics/<dataset>/summary_metrics.csv|json`
  - `results/experiments/<run_name>/segmentation_metrics/<dataset>/segmentation_metrics.csv|json`

## 4) Strict vs non-strict behavior

- `--strict_eval true`: missing mandatory eval models/dependencies will raise error and stop.
- `--strict_eval false` (default): failures are recorded per sample/model and benchmark continues.

For thesis final runs, use strict mode after smoke tests pass.
