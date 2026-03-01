# SAR-HM: Semantic Associative Retrieval with Hopfield Memory

This document describes the SAR-HM extension to DreamDiffusion for thesis use. The baseline DreamDiffusion remains unchanged and is always available when `use_sarhm=False`.

## Target Architecture

```
EEG → MSM/MAE EEG Encoder → Semantic Projection → Hopfield Associative Memory
    → Confidence-Gated Fusion → Conditioning Adapter → Stable Diffusion (Frozen)
```

- **Stable Diffusion** (UNet, VAE, text encoder) is **never** finetuned.
- Only the **EEG encoder**, **projection**, **Hopfield memory**, and **adapter** are trained when SAR-HM is enabled.

## Configuration Flags

Set these in `Config_Generative_Model` (or via `config.py`) and pass the config as `main_config` into `eLDM` / `eLDM_eval`:

| Flag | Default | Description |
|------|--------|-------------|
| `use_sarhm` | `False` | Enable SAR-HM path in `cond_stage_model.forward`. |
| `ablation_mode` | `'baseline'` | One of: `'baseline'`, `'projection_only'`, `'hopfield_no_gate'`, `'full_sarhm'`. |
| `proto_mode` | `'class'` | Prototype mode (class-level by default). |
| `proto_path` | `None` | Path to load/save class prototypes. |
| `hopfield_tau` | `1.0` | Softmax temperature for Hopfield attention. |
| `gate_mode` | `'max'` | Confidence gating: `'max'` or `'entropy'`. |
| `num_classes` | `40` | Number of classes (for ImageNet-EEG 40 objects). |

## Ablation Modes

1. **`baseline`** – Standard DreamDiffusion: EEG → MAE → channel_mapper (if needed) → dim_mapper → conditioning.
2. **`projection_only`** – EEG → MAE → pool → projection → adapter → conditioning (no Hopfield, no gating).
3. **`hopfield_no_gate`** – EEG → MAE → pool → projection → Hopfield → **no gating** → adapter (use retrieved vector only).
4. **`full_sarhm`** – Full pipeline: projection → Hopfield → confidence-gated fusion → adapter.

## Datasets and Usage (Thesis)

- **ImageNet-EEG (primary)**  
  - 40 object classes, EEG paired with viewed images.  
  - Use for: main training, quantitative evaluation, ablations, retrieval accuracy, CLIP similarity.  
  - This supports the main thesis claims.

- **ThoughtViz (secondary / qualitative)**  
  - EEG from imagined images.  
  - Use only for: qualitative image generation and discussion.  
  - No strong quantitative claims.

- **MOABB (pretraining only)**  
  - Optional pretraining or regularization of the EEG encoder.  
  - No image-generation evaluation.

## Reproducibility Checklist

- [ ] Fix `seed` in config.
- [ ] Use the same split files (`block_splits_by_image_single.pth` or `block_splits_by_image_all.pth`).
- [ ] For SAR-HM: set `use_sarhm=True`, choose `ablation_mode`, and (if desired) `proto_path` for saving/loading prototypes.
- [ ] Logged metrics include: CLIP similarity, SSIM/PCC, Hopfield retrieval accuracy, attention entropy, confidence (when SAR-HM is used).
- [ ] Save checkpoints with `config` in the checkpoint so evaluation uses the same `main_config` (including SAR-HM flags).

## Code Layout

- **`code/sarhm/`**  
  - `sarhm_modules.py` – SemanticProjection, HopfieldRetrieval, ConfidenceGatedFusion, ConditioningAdapter, pool_eeg_tokens.  
  - `prototypes.py` – ClassPrototypes and `build_prototypes_from_loader`.  
  - `metrics_logger.py` – retrieval_accuracy, attention_entropy_mean, confidence_stats, CSV/JSON helpers.  
- **Integration**  
  - All SAR-HM logic is switched in **`code/dc_ldm/ldm_for_eeg.py`** inside `cond_stage_model.forward(...)`.  
  - Downstream DDPM/samplers are unchanged; conditioning is always `[B, 77, 768]`.

## Troubleshooting: "SAR-HM: OFF | baseline conditioning path"

You may see **"SAR-HM: OFF | baseline conditioning path"** even when your config has `use_sarhm=True`. This happens when the **`sarhm` package fails to import** (e.g. different machine, missing dependency, or `PYTHONPATH` not including `code/`). The conditioner then falls back to baseline.

- **Check:** At startup you should see either **"SAR-HM ACTIVE | mode=... | ..."** (SAR-HM on) or **"SAR-HM: OFF"** (baseline). If you requested SAR-HM but see OFF, look for the new warning:  
  **`[cond_stage_model] WARNING: use_sarhm=True but 'sarhm' failed to import; using baseline. Error: ...`**  
  The `Error:` part shows the import exception (e.g. `No module named 'sarhm'`).
- **Fix:** From the repo root, run with `PYTHONPATH` including `code/`, e.g.  
  `PYTHONPATH=code python code/eeg_ldm.py ...`  
  (Windows: `set PYTHONPATH=code` then `python code/eeg_ldm.py ...`).  
  Ensure `code/sarhm/` exists and all its dependencies are installed. After fixing the import, you should see **"SAR-HM ACTIVE"** and SAR-HM metrics in logs (e.g. `train/sarhm_retrieval_acc`, `train/sarhm_attention_entropy`).

If your **training** run already shows those SAR-HM metrics in `train_log.csv`, then SAR-HM was on for that run; "SAR-HM: OFF" may have been from another run (e.g. Stage C on a machine where `sarhm` was not importable).

## Architecture Diagram (Graphviz)

Render the diagram:

```bash
dot -Tpng assets/architecture_sarhm.dot -o assets/architecture_sarhm.png
dot -Tsvg assets/architecture_sarhm.dot -o assets/architecture_sarhm.svg
```

The `.dot` file is at `assets/architecture_sarhm.dot`.

## Experiment Table Template (Thesis)

| Experiment | use_sarhm | ablation_mode | Metrics to report |
|------------|-----------|---------------|-------------------|
| Baseline DreamDiffusion | False | baseline | CLIP sim, SSIM, PCC, FID/IS (if run), top-k class |
| EEG → Projection → Diffusion | True | projection_only | CLIP sim, SSIM, PCC, retrieval acc (N/A), attention entropy (N/A) |
| EEG → Hopfield (no gating) | True | hopfield_no_gate | CLIP sim, SSIM, PCC, Hopfield retrieval acc, attention entropy, confidence |
| Full SAR-HM | True | full_sarhm | CLIP sim, SSIM, PCC, Hopfield retrieval acc, attention entropy, confidence |

Within-subject vs subject-held-out is controlled by the split file and subject index in the dataset (e.g. `block_splits_by_image_single.pth` vs by-subject splits).

## Hybrid CLIP supervision (dataset-agnostic)

- **When a paired stimulus image exists**: The training loop uses `image_raw` and the frozen CLIP image encoder to obtain `image_embeds` as the semantic target. No change to the dataset is required.
- **When no paired image exists** (e.g. ThoughtViz): The dataset should provide `target_embeds` in the batch (e.g. from a frozen CLIP text encoder with class-name or prompt). The training step uses `batch.get('target_embeds')`; when present, it is used as the CLIP target instead of `self.image_embedder(image_raw)`.

So the contract is: the batch must contain either valid `image_raw` (for CLIP image target) or precomputed `target_embeds` (e.g. from CLIP text with class names). Implement this in the dataset `__getitem__` when building a loader that has no paired images.
