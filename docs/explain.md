# DreamDiffusion + SAR-HM: Project explained

This document explains **all config parameters**, **training vs validation vs inference**, **how each stage works**, and **typical timings**. Use it to learn the project end-to-end.

---

## 1. Project overview: three stages

| Stage | Script | What it does |
|-------|--------|--------------|
| **A1** | `stageA1_eeg_pretrain.py` | Pre-trains the **EEG encoder** (MAE) on EEG signals. No images. |
| **B**  | `eeg_ldm.py` | **Fine-tunes** the LDM: loads pretrained EEG encoder + Stable Diffusion, trains the conditioning path (EEG → latent) so the diffusion model can generate images from EEG. **Training + validation** happen here. |
| **C**  | `gen_eval_eeg.py` | **Inference only**: loads a Stage B checkpoint, **generates images** from EEG test set, saves images and can compute metrics. No training. |

**Data flow (high level):**  
EEG signals → (Stage A1) → pretrained EEG encoder → (Stage B) → EEG-conditioned LDM → (Stage C) → generated images + metrics.

---

## 2. Training, validation, and inference: definitions

### Training

- **What:** Update model weights using a loss (e.g. diffusion loss on noised latents).
- **Where:**  
  - **Stage A1:** One training loop over EEG data (MAE reconstruction).  
  - **Stage B:** PyTorch Lightning `trainer.fit()`: each **training step** takes a batch (EEG + image), encodes image to latent, noises it, and trains the model to predict noise conditioned on EEG. Only the **conditioning path** (EEG encoder, mapper, SAR-HM if on) is trained; Stable Diffusion UNet/VAE are frozen in “Stage One” of Stage B.
- **When:** Every batch in the training dataloader, for `num_epoch` epochs.

### Validation (during Stage B only)

- **What:** No weight updates. The model is run in eval mode; we **generate images** from EEG (using PLMS/DDIM) and compute metrics (e.g. MSE, PCC, SSIM, top-1-class) to monitor progress.
- **Where:** Inside Stage B, in `LatentDiffusion.validation_step()` (in `code/dc_ldm/models/diffusion/ddpm.py`). Validation runs every `check_val_every_n_epoch` epochs.
- **When:** Every `check_val_every_n_epoch` epochs; optionally a “full” validation every 5th validation run.
- **Important:** Validation uses **separate** config knobs so it can be fast: `val_gen_limit`, `val_ddim_steps`, `val_num_samples`. **Thesis-reported metrics** use Stage C with full `ddim_steps` and `num_samples`, not validation.

### Inference (generation)

- **What:** Given EEG, run the full pipeline: EEG → encoder → conditioning → diffusion sampler (PLMS) → decode latent to image. No training, no gradient.
- **Where:**  
  - **During Stage B:** Validation calls `generate()` with validation limits.  
  - **Stage C:** `gen_eval_eeg.py` loads checkpoint and calls `generative_model.generate(dataset, num_samples, ddim_steps, ...)` for train (limited) and test (full or limited) sets.  
  - **Thesis eval:** Same `generate()` API; can be driven from `code/eval/evaluate.py` with full `ddim_steps` and `num_samples` for final numbers.
- **When:** On demand: after Stage B (in script) or in Stage C; also every N epochs inside Stage B for validation.

---

## 3. Config reference (Stage B: `Config_Generative_Model`)

These are in `code/config.py` and can be overridden by CLI (e.g. `--num_epoch 500`).

### Paths and data

| Parameter | Default | Meaning |
|-----------|---------|--------|
| `root_path` | repo root | Project root. |
| `output_path` | `exps` under root | Not used as final output; Stage B writes to `results/generation/<timestamp>/`. |
| `eeg_signals_path` | `datasets/eeg_5_95_std.pth` | Preprocessed EEG signals. |
| `splits_path` | `datasets/block_splits_by_image_single.pth` | Train/test split. |
| `pretrain_gm_path` | `pretrains` | Folder for SD and config. |
| `pretrain_mbm_path` | `pretrains/eeg_pretain/checkpoint.pth` | Stage A1 EEG encoder checkpoint. |

### Stage B training (finetune)

| Parameter | Default | Meaning |
|-----------|---------|--------|
| `num_epoch` | **500** | Number of training epochs for Stage B. Thesis-level: 500; quick test: 10. |
| `batch_size` | 25 | Batch size for training (and for val dataloader). Reduce (e.g. 4) on small GPU; increase (e.g. 100) on A100 80GB if it fits. |
| `lr` | 5.3e-5 | Learning rate. |
| `precision` | 16 | Training precision: 16 (FP16 mixed), 32 (full), or `"bf16"` (BF16 mixed on A100). |
| `accumulate_grad` | 1 | Gradient accumulation steps (1 = no accumulation). |
| `num_workers` | 0 | DataLoader workers (0 on Windows; 4–8 on Linux/RunPod for faster data load). |
| `crop_ratio` | 0.2 | Random crop for training images (crop to `img_size - crop_ratio*img_size` with p=0.5). |
| `seed` | 2022 | Random seed for reproducibility. |

### Diffusion sampling (used for **inference** and **Stage C**; thesis metrics)

| Parameter | Default | Meaning |
|-----------|---------|--------|
| `num_samples` | **5** | Number of images to **generate per EEG** (e.g. 5 samples per test item). Used in Stage C and for final eval. |
| `ddim_steps` | **250** | Number of PLMS/DDIM denoising steps per generated image. More steps = better quality, slower. **Stage C and thesis metrics use this.** |
| `HW` | None | Optional (H, W) for non-default latent size; usually None. |

### Validation **during Stage B only** (does not change thesis metrics)

| Parameter | Default | Meaning |
|-----------|---------|--------|
| `val_gen_limit` | **2** | Max number of **items** (EEG samples) to run generation for in each validation run. Lower = faster validation. |
| `val_ddim_steps` | **50** | Denoising steps used **only in validation**. Full quality uses `ddim_steps` (250) in Stage C. |
| `val_num_samples` | **2** | Samples per item **only in validation**. Full quality uses `num_samples` (5) in Stage C. |
| `check_val_every_n_epoch` | **5** | Run validation every N epochs. Larger = fewer val runs = faster wall time per epoch. |

### Post–Stage B generation (limits in `eeg_ldm.py` after training)

| Parameter | Default | Meaning |
|-----------|---------|--------|
| `test_gen_limit` | 10 | Cap on test-set images generated at the end of Stage B (in-script). None = no cap. Stage C can generate full test set. |

### Thesis logging (optional)

| Parameter | Default | Meaning |
|-----------|---------|--------|
| `run_name` | None | Custom run name for `runs/<timestamp>_<run_name>/`. |
| `model` | None | `'baseline'` or `'sarhm'` for run folder and logging. |
| `eval_every` | 2 | Run thesis evaluation every N epochs (if MetricLogger is used). |
| `num_eval_samples` | 50 | Max samples per dataset for that evaluation. |

### SAR-HM (if `use_sarhm = True`)

| Parameter | Default | Meaning |
|-----------|---------|--------|
| `use_sarhm` | True | Enable SAR-HM (Hopfield + gated fusion). |
| `ablation_mode` | `'full_sarhm'` | `'baseline'` \| `'projection_only'` \| `'hopfield_no_gate'` \| `'full_sarhm'`. |
| `num_classes` | 40 | Number of classes for prototypes / retrieval. |
| `proto_freeze_epochs` | 5 | Epochs to keep prototypes frozen. |
| Others | (see config) | `alpha_mode`, `alpha_max`, `hopfield_tau`, `gate_mode`, etc. |

### Other

| Parameter | Default | Meaning |
|-----------|---------|--------|
| `global_pool` | False | Global pooling in conditioning. |
| `use_time_cond` | True | Use time embedding in diffusion. |
| `clip_tune` | True | Use CLIP tuning loss (if applicable). |
| `eval_avg` | True | Average over multiple samples in eval metrics. |
| `img_size` | 512 | Image size (e.g. 512×512). |
| `subject` | 4 | Subject index for dataset. |

---

## 4. Stage A1 config (EEG encoder pretrain): `Config_MBM_EEG`

Used by `stageA1_eeg_pretrain.py`. Key parameters:

| Parameter | Default | Meaning |
|-----------|---------|--------|
| `num_epoch` | **500** | Pretraining epochs. |
| `batch_size` | 100 | Batch size. |
| `lr` | 2.5e-4 | Learning rate. |
| `mask_ratio` | 0.1 | MAE mask ratio. |
| `patch_size` | 4 | Patch size. |
| `embed_dim` | 1024 | Encoder embedding dimension. |
| `depth` | 24 | Number of transformer blocks. |
| `num_heads` | 16 | Attention heads. |
| `warmup_epochs` | 40 | LR warmup epochs. |

Output: `results/eeg_pretrain/<timestamp>/checkpoints/checkpoint.pth` → copy to `pretrains/eeg_pretain/checkpoint.pth` for Stage B.

---

## 5. How Stage B runs (training + validation)

1. **Load data:** `create_EEG_dataset()` with `eeg_signals_path`, `splits_path` → train and test datasets.
2. **Load model:** eLDM (EEG encoder from Stage A1 + LDM with SD). Conditioning path is EEG → MAE → mapper → (optional SAR-HM) → diffusion.
3. **Stage One (only conditioning):**  
   - Freeze first_stage (VAE) and diffusion (UNet); train only cond_stage (EEG encoder + mapper + SAR-HM if on).  
   - `trainers.fit(model, train_dataloader, val_dataloaders=test_loader)`.
4. **Training step (each batch):**  
   - Batch = EEG + image.  
   - Image → VAE → latent; latent is noised; model predicts noise conditioned on EEG.  
   - Loss = diffusion loss (and optionally CLIP, etc.); backward, optimizer step.
5. **Validation (every `check_val_every_n_epoch`):**  
   - `validation_step()` is called with a batch from the test dataloader.  
   - It calls `generate(batch, ddim_steps=val_ddim_steps, num_samples=val_num_samples, limit=val_gen_limit)`.  
   - So we run **at most `val_gen_limit`** items, each with **`val_num_samples`** images, each with **`val_ddim_steps`** PLMS steps.  
   - Metrics (MSE, PCC, SSIM, top-1-class, etc.) are computed on these generated images; no backward.  
   - Every 5th validation can trigger a “full” validation (same but with slightly more items/samples).
6. **After training:** Model is unfrozen (for saving), checkpoint saved to `results/generation/<timestamp>/checkpoint.pth`. Optional: generate train/test images in-script (limited by `test_gen_limit`).

So: **training** = many steps of forward + loss + backward. **Validation** = a few generations (PLMS) + metrics, no training. Both are part of Stage B.

---

## 6. How inference (generation) works

- **Entry:** `eLDM.generate(fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, ...)`.  
  `fmri_embedding` is a dataset or list of items; each item has `'eeg'` and optionally `'image'` (for saving GT).
- **Per item (up to `limit`):**  
  1. Get EEG latent (and optional GT image).  
  2. Condition the diffusion model on EEG (encoder + mapper + optional SAR-HM).  
  3. Run **PLMS sampler** for `ddim_steps` steps to get a latent sample; repeat `num_samples` times per EEG.  
  4. Decode latents with VAE → images.  
  5. Append to list (and optionally save to disk).
- **Output:** Grid of images and array of samples (e.g. shape like `(N, 1+num_samples, C, H, W)` with GT in index 0).

So **one “inference”** = one call to `generate()` with a dataset and limits. **Cost** scales with:  
`(number of items) × (num_samples) × (ddim_steps)`.

---

## 7. Timings (typical)

Approximate; depend on GPU (e.g. A100 80GB vs 16 GB) and dataset size.

### Stage A1

- **Per epoch:** Depends on dataset size and `batch_size`. With large dataset and batch 100, order of minutes per epoch.
- **Total:** 500 epochs can be many hours; 10 epochs is a quick sanity run.

### Stage B

- **Training step:** Seconds per batch (e.g. 1–3 s on A100 with batch 24–100). One epoch = (train set size / batch_size) steps.
- **Validation (one run):** Dominated by **generation**.  
  - With **old** defaults (e.g. 250 steps × 5 samples × 5 items) ≈ **tens of minutes** (e.g. 20–35 min on A100).  
  - With **current** defaults (`val_ddim_steps=50`, `val_num_samples=2`, `val_gen_limit=2`, every 5 epochs): **~2–5 minutes** per validation run.  
- **Epoch time:** If validation runs this epoch: (training time) + (validation time). If not: only training. So “37 min per epoch” was usually **training + one heavy validation**; with fast validation and `check_val_every_n_epoch=5`, most epochs are much shorter.

### Inference (Stage C or single `generate()` call)

- **Per image:** Roughly proportional to `ddim_steps`.  
  - 250 steps: order of **tens of seconds** per sample on A100 (e.g. 30–60 s per sample depending on resolution and batch).  
  - 50 steps: roughly 5–10× faster per sample.
- **Per item with `num_samples=5`:** 5 × (time per sample).  
- **Full test set:** (number of test items) × (num_samples) × (time per sample). So “~70 min per item” in logs is a conservative ballpark for 250 steps × several samples on a typical GPU; on A100 it can be lower but still significant for many items.

---

## 8. Summary table: what affects what

| You want… | Parameters that matter |
|-----------|------------------------|
| **Thesis-reported metrics** | Stage C (or eval script) with **`ddim_steps=250`**, **`num_samples=5`**. Not validation. |
| **Faster Stage B epochs** | `val_gen_limit`, `val_ddim_steps`, `val_num_samples` (smaller = faster validation), `check_val_every_n_epoch` (larger = fewer val runs). |
| **Faster inference** | Lower `ddim_steps` (e.g. 50 for quick visuals; 250 for final numbers), lower `num_samples`, or fewer items (`limit`). |
| **More/fewer training epochs** | `num_epoch` (500 for thesis, 10 for quick test). |
| **Larger/smaller batches** | `batch_size` (e.g. 100 on A100 if it fits; 4 on 16 GB). |
| **Reproducibility** | Same `seed`, same `splits_path`, same dataset; when loading for eval, use same config (e.g. SAR-HM flags) as in training. |

---

## 9. Where in the code

| Concept | File / location |
|--------|------------------|
| Config (Stage B) | `code/config.py` → `Config_Generative_Model` |
| Config (Stage A1) | `code/config.py` → `Config_MBM_EEG` |
| Stage B entry, data, trainer | `code/eeg_ldm.py` → `main()`, `create_trainer()` |
| Stage B training/validation loop | PyTorch Lightning; model in `code/dc_ldm/models/diffusion/ddpm.py` |
| Training step | `ddpm.py` → `training_step()` → `shared_step()` |
| Validation step | `ddpm.py` → `validation_step()`, `full_validation()` |
| Generation (PLMS) | `code/dc_ldm/ldm_for_eeg.py` → `eLDM.generate()`; sampler in `code/dc_ldm/models/diffusion/plms.py` |
| Stage C | `code/gen_eval_eeg.py` → load checkpoint, then `generate()` on train/test sets |
| Thesis eval | `code/eval/evaluate.py` (if present); uses same `generate()` with config `ddim_steps` / `num_samples` |

This should give you a single place to look up parameters, understand training vs validation vs inference, and reason about timings and thesis-quality metrics.
