# Thesis Documentation — DreamDiffusion + SAR-HM + SAR-HM++ (EEG-to-Image)

**Complete implementation details, architecture, and step-by-step pipeline.**

---

## 1. Project Summary

This project implements **DreamDiffusion**: an EEG-conditioned Latent Diffusion Model (Stable Diffusion 1.5) that reconstructs or approximates visual stimuli from non-invasive brain signals (EEG). Two extensions are supported:

- **SAR-HM** (Semantic Associative Retrieval with Hopfield Memory): class-level prototypes in CLIP space + Hopfield retrieval + confidence-gated fusion.
- **SAR-HM++** (Multi-Level Semantic Prototype Retrieval): multi-level semantic memory (CLIP/scene/object/summary), semantic query from EEG, top-k retrieval, semantic adapter to SD conditioning, and optional **semantic teacher supervision** during training (L_sem_align, L_retr, L_clip_img, L_clip_text). **Inference remains EEG-only** (no ground-truth semantic leakage).

**In short:**
- **Input:** EEG recordings collected while a participant views natural images (ImageNet subset).
- **Stage A (A1):** Train an **EEG encoder** (Masked Autoencoder for EEG) that turns raw EEG into a compact latent representation.
- **Stage B:** Fine-tune a **Latent Diffusion Model** so that its conditioning path is driven by EEG and (optionally) SAR-HM or SAR-HM++; **Stable Diffusion UNet and VAE stay frozen**. For SAR-HM++, the train dataset can be wrapped with **SemanticTargetWrapper** so each batch includes `z_sem_gt`, `clip_img_embed_gt`, `summary_embed_gt`, and `has_semantic_gt`; losses L_sem_align, L_retr, and (optionally) L_clip_img / L_clip_text on **decoded generated images** are then applied.
- **Stage C:** Generate images from held-out EEG and evaluate; **no GT semantics are used at inference**. Optionally compare **baseline vs SAR-HM vs SAR-HM++** on the same data.
- **Unified Benchmark:** A separate benchmark subsystem compares **ThoughtViz**, **DreamDiffusion baseline**, and **DreamDiffusion + SAR-HM** (normal SAR-HM only; SAR-HM++ is excluded from this benchmark) on **ImageNet-EEG** and **ThoughtViz** datasets. It provides dataset adapters, model wrappers, standardized outputs, sanity tests, small (10–20 sample) runs, metrics, tables, and qualitative panels. See Section 11 and `docs/commands.md`, `docs/benchmark_workflow.md`, `docs/thoughtviz_integration.md`.

### What we perform in each stage (at a glance)

| Stage | What we do | How we do it | Main output |
|-------|------------|--------------|-------------|
| **A (A1)** | Pretrain EEG encoder | Masked autoencoding on EEG (mask patches → Transformer encode/decode → reconstruction loss). Data: `eeg_pretrain_dataset` from `.npy` or `eeg_5_95_std.pth`. | `checkpoint.pth` (MAE weights) → copied to `pretrains/eeg_pretain/checkpoint.pth` |
| **B** | Fine-tune conditioning only | Load SD 1.5 + MAE from A1. Freeze VAE and UNet. Train `cond_stage_model`: EEG → MAE → dim_mapper → c_base; **SAR-HM:** projection → Hopfield(prototypes) → fusion → adapter → c_sar; **SAR-HM++:** pooled EEG → SemanticQueryHead → q_sem → top-k retrieval over semantic_prototypes → m_sem → SemanticAdapter → c_sem; c_final = c_base + α*(c_sem - c_base). Loss: diffusion + optional CLIP + SAR-HM retrieval/stable; **SAR-HM++:** + L_sem_align(q_sem,z_sem_gt), L_retr(m_sem,z_sem_gt), and (when batch has semantic targets) L_clip_img / L_clip_text on decoded generated images (configurable frequency). | `checkpoint_best.pth`, `prototypes.pt` (SAR-HM) or `semantic_prototypes.pt` (SAR-HM++) |
| **C** | Generate and evaluate | Load Stage B checkpoint and (for SAR-HM/SAR-HM++) prototypes. **EEG-only:** no GT image or semantic targets. For each sample: EEG → cond_stage_model → c → PLMS → VAE decode → image. Compute SSIM, PCC, CLIP, etc. | Grids, per-image PNGs, metrics; compare_eval: baseline vs SAR-HM vs SAR-HM++ on same items |

---

## 2. Repository Map (Key Files and Folders)

| Path | Description |
|------|-------------|
| `README.md` | Main usage, quickstart, dataset layout, unified benchmark summary. |
| `code/stageA1_eeg_pretrain.py` | **Stage A1:** EEG MAE pretraining entry point. |
| `code/eeg_ldm.py` | **Stage B:** LDM fine-tuning (baseline + SAR-HM). |
| `code/gen_eval_eeg.py` | **Stage C:** Generation + evaluation from a Stage B checkpoint. |
| `code/compare_eval.py` | Dual evaluation: baseline vs SAR-HM (and optionally SAR-HM++) on same dataset/seed. |
| `code/dataset.py` | `eeg_pretrain_dataset`, `EEGDataset`, `Splitter`, `create_EEG_dataset`. |
| `code/config.py` | `Config_MBM_EEG`, `Config_EEG_finetune`, `Config_Generative_Model`. |
| `code/thoughtviz_integration/` | **Benchmark:** ThoughtViz wrapper: `get_thoughtviz_root`, `ThoughtVizConfig`, `ThoughtVizDatasetAdapter`, `ThoughtVizWrapper` (load, generate_from_eeg, save_outputs). |
| `code/dc_ldm/ldm_for_eeg.py` | `cond_stage_model`, `eLDM`, `eLDM_eval`; EEG → conditioning; PLMS generation. |
| `code/dc_ldm/models/diffusion/ddpm.py` | Latent Diffusion (UNet, training_step, p_losses, optional SAR-HM losses). |
| `code/dc_ldm/models/diffusion/plms.py` | PLMS sampler used for Stage C generation. |
| `code/sc_mbm/mae_for_eeg.py` | `MAEforEEG`, `eeg_encoder`, `PatchEmbed1D` (used in A1 and as encoder in B). |
| `code/sarhm/` | SAR-HM: `sarhm_modules.py`, `prototypes.py`. SAR-HM++: `semantic_targets.py`, `semantic_memory.py`, `semantic_query.py`, `semantic_adapter.py`, `semantic_losses.py`, `semantic_dataset_wrapper.py`. `metrics_logger.py`, `vis.py`. |
| `code/build_semantic_targets.py` | Offline: build `semantic_targets.pt` from EEG dataset images (CLIP/scene/object/summary). |
| `code/build_semantic_prototypes.py` | Offline: build `semantic_prototypes.pt` from `semantic_targets.pt`. |
| `benchmark/` | **Unified benchmark:** `benchmark_config.py`, `benchmark_runner.py`, `dataset_registry.py`, `model_registry.py`, `output_standardizer.py`, `metrics_runner.py`, `timing_runner.py`, `table_generator.py`, `visualization_runner.py`, `compare_all_models.py` (CLI), `segmentation_eval.py`, `caption_eval.py`. |
| `tests/test_full_pipeline_sanity.py` | Sanity test: dataset load, model load, one-sample inference, metrics for benchmark. |
| `datasets/eeg_5_95_std.pth` | ImageNet-EEG signals (required for B/C). |
| `datasets/block_splits_by_image_single.pth` | Train/test splits by image index. |
| `datasets/imageNet_images/` | Ground-truth images for EEG samples (or ILSVRC2012 root). |
| `pretrains/models/config15.yaml`, `v1-5-pruned.ckpt` | Stable Diffusion 1.5 config and weights. |
| `pretrains/eeg_pretain/checkpoint.pth` | EEG MAE checkpoint produced by Stage A1 (used by Stage B). |
| `docs/commands.md` | Runnable commands: sanity, small benchmark, ThoughtViz, DreamDiffusion, SAR-HM, comparison, metrics, tables. |
| `docs/thoughtviz_integration.md` | ThoughtViz wrapper API, data paths, dependencies. |
| `docs/benchmark_workflow.md` | Benchmark phases, folder structure, MSC/optional metrics. |
| `docs/BENCHMARK_INSPECTION_AND_PLAN.md` | Benchmark implementation plan and file map. |
| `docs/SARHM_README.md` | SAR-HM configuration, ablations, reproducibility. |
| `docs/SARHMPP_README.md` | SAR-HM++ modules, semantic targets in batch, CLIP losses, commands. |
| `docs/SARHMPP_IMPLEMENTATION_STATUS.md` | SAR-HM++ implementation status and remaining notes. |

---

## 3. End-to-End Architecture and Data Flow

### 3.1 High-Level Pipeline

```
EEG (raw) → preprocessing / storage (datasets/eeg_5_95_std.pth)
    │
    ├─► Stage A1: MAE for EEG (code/stageA1_eeg_pretrain.py)
    │       EEG → masked tokens → Transformer encoder/decoder → reconstruction loss
    │       Output: pretrains/eeg_pretain/checkpoint.pth
    │
    └─► Stage B: LDM fine-tuning (code/eeg_ldm.py)
            ├─ Baseline: EEG → MAE encoder → dim_mapper → c_base [B,77,768] → SD UNet cross-attention
            └─ SAR-HM:   EEG → MAE → projection → Hopfield(prototypes) → fusion → adapter → c_sar
                         c_final = c_base + α*(c_sar - c_base)  → SD UNet cross-attention
            Output: results/exps/results/generation/<timestamp>/checkpoint_best.pth (+ prototypes.pt)

Stage C: Generation & Eval
    EEG test set + trained model + PLMS
    ├─ code/gen_eval_eeg.py   (single model)
    └─ code/compare_eval.py   (baseline vs SAR-HM, same EEG items)
        → metrics (SSIM, PCC, CLIP, etc.) + grids under results/compare_eval_thesis/
```

### 3.2 Conditioning Path (Baseline vs SAR-HM)

- **Baseline:** `cond_stage_model` in `dc_ldm/ldm_for_eeg.py`:
  - EEG `[B, C, T]` → **MAE encoder** (`eeg_encoder` from `sc_mbm/mae_for_eeg.py`) → latent tokens.
  - If `global_pool == False`: `channel_mapper` reduces sequence to 77 tokens.
  - **dim_mapper** (linear) maps to `cond_dim` (768 for SD) → `c_base` shape `[B, 77, 768]`.
  - `_align_to_seq_len(c_base, 77)` ensures length 77 for UNet cross-attention.

- **SAR-HM (when `use_sarhm=True`):**
  - Same MAE path → `c_base`.
  - Pooled EEG tokens → **SemanticProjection** → CLIP-space vector `z_orig`.
  - **HopfieldRetrieval**: query `z_orig`, memory = class prototypes `P [K, 768]` → attention `attn [B, K]`, retrieved `z_ret`.
  - **ConfidenceGatedFusion**: `z_fused` from `z_orig` and `z_ret` using confidence (max or entropy).
  - **ConditioningAdapter**: `z_fused` → `c_sar` `[B, 77, cond_dim]`.
  - **Residual fusion:** `c_final = c_base_n + α * (c_sar_n - c_base_n)` with LayerNorm (or L2) normalization; α is gated by confidence and `conf_threshold` (low confidence → α=0).

- **SAR-HM++ (when `use_sarhmpp=True`):**
  - Same MAE path → `c_base`.
  - Pooled EEG tokens (via `pool_eeg_for_query`) → **SemanticQueryHead** → `q_sem` `[B, 768]` in CLIP space.
  - **SemanticRetrieval**: query `q_sem`, keys from **SemanticMemoryBank** (multi-level fused prototypes) → top-k retrieval → `m_sem` `[B, 768]`, confidence from attention.
  - **SemanticAdapter**: `m_sem` → `c_sem` `[B, 77, 768]` (MLP + optional transformer).
  - **Residual fusion:** `c_final = c_base_n + α * (c_sem_n - c_base_n)`; α from confidence (or constant when `no_confidence_gate`). Optionally `sarhmpp_projection_only`: skip retrieval, use `q_sem` as `m_sem`.
  - **Training-only (no inference):** When `semantic_targets_path` is set, the train dataset is wrapped with **SemanticTargetWrapper**, which loads `semantic_targets.pt` and adds per-sample `z_sem_gt`, `clip_img_embed_gt`, `summary_embed_gt`, `object_embed_gt`, `scene_embed_gt`, `has_semantic_gt`, `sample_id`. These are used only for losses; **inference uses only EEG and the preloaded semantic_prototypes.pt** (no GT semantics).

So **only the conditioning stack** (MAE, mappers, SAR-HM/SAR-HM++ modules) is trained in Stage B; **UNet and VAE are frozen**.

---

## 4. Stage A (Stage A1) — EEG Encoder Pretraining

**Goal:** Learn a compact, robust representation of EEG via masked autoencoding. This checkpoint is then used as the EEG encoder in Stage B.

### 4.1 Script and Config

- **Entry point:** `code/stageA1_eeg_pretrain.py` (default config: `Config_MBM_EEG` in `code/config.py`).
- **Dataset class:** `eeg_pretrain_dataset` in `code/dataset.py`.

### 4.2 Data Inputs

- **Preferred:** MNE-style `.npy` files under `datasets/mne_data/` (or path from `DREAMDIFFUSION_DATA_ROOT`).
- **Fallback:** If no `.npy` files exist, the loader reads `datasets/eeg_5_95_std.pth` and uses the list `loaded['dataset']`; each item must have key `'eeg'` (tensor).

**Per-sample processing in `eeg_pretrain_dataset.__getitem__`:**
- Load EEG: from `.npy` or from `_eeg_from_pth[index]`.
- If from `.pth` and shape is `(channels, time)` with more time than channels, transpose to `(channels, time)`; crop time to `[20:460]` if length ≥ 460.
- Resample/crop to fixed **time length 512** and **128 channels** (interpolation or replication/cropping as in `dataset.py`).
- Scale: `ret = ret / 10`, return `{'eeg': torch.FloatTensor}` of shape `[128, 512]`.

### 4.3 Model: MAEforEEG

- **File:** `code/sc_mbm/mae_for_eeg.py`.
- **Structure:**
  - **PatchEmbed1D:** 1D convolution over (channels, time) → patch embeddings; `num_patches = time_len // patch_size`.
  - **Encoder:** CLS token + positional embedding, stack of Transformer `Block`s (from timm), norm.
  - **Decoder:** mask token, decoder embedding, decoder blocks, `decoder_pred` → reconstruct patches (reconstruction loss).
  - Optional: nature image decoder branch if `use_nature_img_loss=True` (ResNet feature loss; often disabled).

**Config (Config_MBM_EEG):** `patch_size=4`, `embed_dim=1024`, `depth=24`, `num_heads=16`, `decoder_embed_dim=512`, `decoder_num_heads=16`, `mask_ratio=0.1`, etc.

### 4.4 Training Loop (stageA1_eeg_pretrain.py)

1. Create `eeg_pretrain_dataset(path=datasets/mne_data, ...)` and DataLoader (optionally `DistributedSampler` for multi-GPU).
2. Build **MAEforEEG** with `time_len=dataset_pretrain.data_len` (512), patch_size, embed_dim, depth, etc.
3. Optimizer: `timm.optim.optim_factory.param_groups_weight_decay` + `AdamW(lr=config.lr, betas=(0.9, 0.95))`.
4. For each epoch: `train_one_epoch(...)` from `sc_mbm.trainer` (reconstruction loss on masked patches; optional image loss if enabled).
5. Every 20 epochs (or last): `save_model(...)` to `results/eeg_pretrain/<timestamp>/checkpoints/checkpoint.pth`.
6. Optional: `plot_recon_figures(...)` for qualitative reconstruction plots.

**Output:** Checkpoint containing `model` (or `model_state_dict`), `config`, and optionally optimizer state. For Stage B, copy this to `pretrains/eeg_pretain/checkpoint.pth`.

---

## 5. Stage B — Diffusion Fine-Tuning (Baseline and SAR-HM)

**Goal:** Fine-tune the **conditioning path** so that EEG (and optionally SAR-HM) drives the pre-trained Stable Diffusion UNet. UNet and VAE remain **frozen**.

### 5.1 Entry Point and Data

- **Script:** `code/eeg_ldm.py` → `main(config)`.
- **Config:** `Config_Generative_Model` in `code/config.py`; overridable via CLI (e.g. `--splits_path`, `--eeg_signals_path`, `--imagenet_path`, `--use_sarhm`, `--ablation_mode`).

**Data:**
- `eeg_signals_path`: `datasets/eeg_5_95_std.pth`.
- `splits_path`: `datasets/block_splits_by_image_single.pth` (or `block_splits_by_image_all.pth`).
- `imagenet_path`: Required for `dataset=='EEG'`; root where ImageNet-style folders (e.g. `n02106662/`) and `image_name.JPEG` files live.

**Dataset creation:**
- `create_EEG_dataset(eeg_signals_path, splits_path, imagenet_path, image_transform=[train_transform, test_transform], subject=config.subject)` in `code/dataset.py`.
- This builds a full `EEGDataset` then wraps it with **Splitter** using `splits_path`: key `"splits"` → list of splits; each split has `"train"` and `"test"` indices. Splitter filters indices (e.g. EEG length 450–600, valid image paths) and returns `(split_train, split_test)`.
- **SAR-HM++ only:** When `use_sarhmpp` and `semantic_targets_path` are set and the file exists, the **train** split is further wrapped with **SemanticTargetWrapper** (`code/sarhm/semantic_dataset_wrapper.py`). The wrapper loads `semantic_targets.pt` (same order as train split) and, for each index, adds to the sample: `z_sem_gt` [768], `clip_img_embed_gt` [768], `summary_embed_gt` [768], `object_embed_gt` [768], `scene_embed_gt` [768], `has_semantic_gt` (bool), `has_region_semantics`, `sample_id`. If an item is missing, placeholders (zeros) and `has_semantic_gt=False` are used so collation stays consistent. The **test** split is never wrapped (inference remains EEG-only).

**EEGDataset** (dataset.py):
- Loads `eeg_5_95_std.pth` → `self.data = loaded['dataset']`, `self.labels`, `self.images`.
- Subject filter: if `subject != 0`, filter by `data[i]['subject']`.
- Each `__getitem__(i)`: EEG cropped to 440 time points `[20:460]`, resampled to 512, normalized; image loaded from `imagenet_path` using `self.images[self.data[i]["image"]]` (path `.../image_name_split_0/image_name.JPEG`). Returns dict with `eeg`, `image`, `label`, etc.

### 5.2 Model Construction (eLDM)

- **Pretrained MAE:** Load `config.pretrain_mbm_path` (e.g. `pretrains/eeg_pretain/checkpoint.pth`) → `metafile` with keys `config`, `model`/`model_state_dict`/`state_dict`.
- **SD config:** `pretrain_root/models/config15.yaml`; weights `v1-5-pruned.ckpt`. `eLDM` in `ldm_for_eeg.py` loads SD via `instantiate_from_config(config.model)` and `load_state_dict(pl_sd)`.
- **Conditioner:** `cond_stage_model(metafile, num_voxels, cond_dim=768, global_pool=config.global_pool, ..., main_config=config)`:
  - Builds **eeg_encoder** from MAE config (same as MAE encoder part), loads MAE weights from metafile.
  - **channel_mapper** (if `global_pool=False`): Conv1d to reduce sequence to 77.
  - **dim_mapper:** Linear(mae_embed_dim, cond_dim).
  - If SAR-HM: **SemanticProjection**, **ClassPrototypes**, **HopfieldRetrieval**, **ConfidenceGatedFusion**, **ConditioningAdapter**, LayerNorm for conditioning.

**Resume / init:**
- `resume_ckpt_path`: full resume (config + state_dict); must be Stage B checkpoint, not MAE.
- `init_from_ckpt`: weights-only load; training starts from epoch 0.

### 5.3 SAR-HM: Prototypes and Fusion

- **Prototypes:** If `use_sarhm` and `proto_source == 'baseline_centroids'`, **before** training:
  - `ensure_prototypes_loaded_or_built(...)` in `eeg_ldm.py`: build centroids from training set using baseline conditioner (pool EEG → SemanticProjection → per-class mean in CLIP space) and save to `output_path/prototypes.pt`; or load from `proto_path` / same dir as `init_from_ckpt`.
- **ClassPrototypes** holds a tensor `P` of shape `[num_classes, 768]`; optionally loaded/saved via `proto_path`.

**Forward (cond_stage_model.forward):**
- Always compute `c_base` (MAE → channel_mapper if needed → dim_mapper → align to 77).
- If SAR-HM and valid prototypes: pooled → projection → Hopfield(query, P) → fusion → adapter → `c_sar`; α from `compute_alpha_from_attention(attn, alpha_mode, alpha_max, conf_threshold)`; if confidence &lt; conf_threshold or prototypes invalid → α=0. Then `c_final = c_base_n + α*(c_sar_n - c_base_n)` (normalized by LayerNorm or L2).

### 5.4 Training Step (What Is Trained)

- **Frozen:** VAE (`freeze_first_stage()`), UNet / diffusion_model (`freeze_diffusion_model()`). All parameters with `first_stage` or diffusion_model (and not cond_stage) have `requires_grad=False`.
- **Trainable:** `cond_stage_model` (MAE encoder, channel_mapper, dim_mapper, and if SAR-HM: projection, Hopfield, fusion, adapter; optionally prototypes after `proto_freeze_epochs`).

**Training loop (PyTorch Lightning):**
- `trainer.fit(model, dataloader, val_dataloaders=test_loader)`.
- **training_step** (ddpm.py):
  1. Get batch: EEG, image, label, image_raw; optionally `target_embeds`, `z_sem_gt`, `clip_img_embed_gt`, `summary_embed_gt`, `has_semantic_gt` (when SemanticTargetWrapper is used).
  2. Encode image: `x = model.get_input(batch, model.first_stage_key)` → VAE encode → latent `x` in diffusion space.
  3. Get conditioning: `c, _ = model.get_learned_conditioning(repeat(eeg, ...))` → calls `cond_stage_model(eeg)` → returns `c_base` or `c_final` (and optional re_latent for CLIP).
  4. Sample timestep `t`. Optionally request `x0_pred` from the same forward when SAR-HM++ and CLIP-on-decoded loss are enabled (`need_x0` = use_sarhmpp, not no_clip_loss, batch has clip_img_embed_gt/summary_embed_gt, and `global_step % clip_loss_every_n_steps == 0` and at least one `has_semantic_gt`).
  5. Loss: `p_losses(x, c, t)` = noise prediction loss; if `need_x0`, also returns predicted clean latent `x0_pred`.
  6. **SAR-HM++ semantic losses (training-only):** When `z_sem_gt` is in the batch: L_sem_align(q_sem, z_sem_gt), L_retr(m_sem, z_sem_gt) via `compute_semantic_losses` (unless no_clip_loss / no_retrieval_loss zero the corresponding lambdas).
  7. **SAR-HM++ CLIP-on-decoded (training-only):** When `x0_pred` is available and batch has `clip_img_embed_gt` and `summary_embed_gt`: decode `x0_pred` with `_decode_first_stage_allow_grad` (gradients flow), convert to [0,1], compute CLIP image features of the **generated** image via `extract_clip_image_embed(..., allow_grad=True)`; optionally mask by `has_semantic_gt`. Then L_clip_img(clip_img_gen, clip_img_embed_gt) and L_clip_text(clip_img_gen, summary_embed_gt) are added (lambda_clip_img, lambda_clip_text). This improves actual CLIP similarity of generated outputs.
  8. Optional: existing CLIP loss (re_latent vs image_embeds); SAR-HM retrieval/stable losses if configured.
  9. Logged to `train_log.csv` when thesis logging is enabled (e.g. train/loss_total, train/loss_sem_align, train/loss_retr, train/loss_clip_img, train/loss_clip_text, train/sarhm_retrieval_acc, train/sarhm_attention_entropy).

**Validation:** Config options `disable_image_generation_in_val`, `val_image_gen_every_n_epoch`, `val_gen_limit` control whether validation runs PLMS generation; for thesis runs validation often skips image gen to save time; **Stage C** is used for final metrics.

### 5.5 Outputs of Stage B

- Checkpoint: `output_path/checkpoint.pth` and `checkpoint_best.pth` (model_state_dict, config, rng state).
- SAR-HM: `output_path/prototypes.pt` (class prototypes + metadata).
- SAR-HM++: Uses pre-built `semantic_prototypes.pt` (from `build_semantic_prototypes.py`); optionally `semantic_targets.pt` (from `build_semantic_targets.py`) for training-only wrapper. Checkpoint dir may contain a copy of `semantic_prototypes.pt` for evaluation.

---

## 6. Stage C — Generation, Evaluation, and Comparison

**Goal:** Load a trained Stage B checkpoint, generate images from EEG (test set), and compute metrics. Optionally compare baseline vs SAR-HM on the same samples.

### 6.1 Single-Model Generation and Eval: gen_eval_eeg.py

- **Inputs:** `--model_path` (Stage B checkpoint), `--dataset EEG`, `--splits_path`, `--eeg_signals_path`, `--config_patch` (SD YAML), `--imagenet_path`, optional `--proto_path` (for SAR-HM), `--num_samples`, `--ddim_steps`, `--split test`/`train`, `--max_test_items`, etc.
- **Flow:**
  1. Load checkpoint: `sd = torch.load(model_path)`; `config = sd['config']`.
  2. Build **eLDM_eval** with `config_patch` and `main_config=config` (no MAE metafile; conditioner built from config only). Load `sd['model_state_dict']` into `model` (strict=False). For SAR-HM: load prototypes from `proto_path` or checkpoint dir. For SAR-HM++: load **semantic_prototypes.pt** from `--proto_path` or auto-detected from checkpoint directory (`utils_eval.load_model` sets `config.semantic_prototypes_path` when `use_sarhmpp` and path unset).
  3. **create_EEG_dataset(...)** with same transforms as Stage B; **no SemanticTargetWrapper** is applied at evaluation (test split only; no GT semantic targets).
  4. Call **eLDM_eval.generate(...)** with PLMS sampler. For each EEG item (up to `limit`):
     - `latent = item['eeg']` (EEG tensor); repeat for `num_samples` copies.
     - `c, _ = model.get_learned_conditioning(latent_rep)` → cond_stage_model forward → conditioning `c` [num_samples, 77, 768].
     - `samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=c, batch_size=num_samples, shape=(C,H,W))` → denoised latent.
     - `x_samples = model.decode_first_stage(samples_ddim)` → VAE decode; clamp (x+1)/2 to [0,1].
     - Append [GT image, *generated samples] to list; optionally save per-image to output_path.
  5. Build grid from all samples; save `samples_train.png`, `samples_test.png`, and per-image `test<i>-<j>.png`. Compute metrics via `get_eval_metric` (MSE, PCC, SSIM, PSM, top-k class) and print/log.

### 6.2 Baseline vs SAR-HM vs SAR-HM++: compare_eval.py

- Loads **same** EEG dataset (same splits, seed). **Inference is EEG-only** for all models (no GT image or semantic targets).
- Loads **two** models: baseline checkpoint and a second checkpoint (SAR-HM or SAR-HM++). For SAR-HM: pass `--sarhm_proto` (or prototypes in checkpoint dir). For SAR-HM++: prototypes are auto-loaded from checkpoint dir as `semantic_prototypes.pt` or `prototypes.pt`, or set explicitly with `--sarhmpp_proto`.
- For a fixed subset of test items (`n_samples`), generates images with both models (same seed, same ddim_steps).
- Computes metrics (SSIM, PCC, CLIP similarity, etc.) for each and writes:
  - `results/compare_eval_thesis/metrics/metrics.csv`
  - `results/compare_eval_thesis/metrics/report.md`
  - Grids: `baseline_grid.png`, `sarhm_grid.png`, `side_by_side.png`; per-model images in `baseline/`, `sarhm/`.

---

## 7. Datasets and Splits (Summary)

| Stage | Dataset / file | Split | Loader |
|-------|----------------|-------|--------|
| A1 | `mne_data/*.npy` or `eeg_5_95_std.pth` | No split (shuffle by epoch) | `eeg_pretrain_dataset` |
| B | `eeg_5_95_std.pth` + `block_splits_by_image_single.pth` + ImageNet root | train / test from splits | `create_EEG_dataset` → Splitter |
| C | Same as B | test (or train) | Same |
| Compare-eval | Same + ImageNet for GT | Subset of test (n_samples) | `create_EEG_dataset` + Subset |

**eeg_5_95_std.pth:** `{'dataset': list of dicts (each has 'eeg', 'image', 'label', ...), 'labels', 'images'}`.  
**block_splits_*.pth:** `{'splits': [ { 'train': indices, 'test': indices }, ... ] }`.  
**Splitter** (dataset.py): Filters indices (e.g. EEG length 450–600, valid image path) and exposes `__getitem__(i)` → `dataset[split_idx[i]]`.

**Unified benchmark datasets:** ImageNet-EEG uses the same EEG/splits/ImageNet paths, adapted to unified sample format via `benchmark/dataset_registry.get_dataset('imagenet_eeg', ...)`. ThoughtViz uses `data.pkl` and class-based image folders under the ThoughtViz repo; `ThoughtVizDatasetAdapter` exposes the same interface (sample_id, eeg, ground_truth, label, metadata, split).

---

## 8. Metrics

- **eval_metrics.py:** MSE, PCC, SSIM, PSM (LPIPS), FID, top-k / n-way class accuracy (e.g. ViT/CLIP). Entry: `get_similarity_metric(..., method='pair-wise'|'n-way'|'class', metric_name=...)`.
- **Compare-eval:** Reports SSIM, PCC, CLIP similarity, mean variance, n_samples in `metrics.csv` and `report.md`.
- **SAR-HM training:** Retrieval accuracy, attention entropy, confidence (and optional stable/retrieval losses) logged when SAR-HM is enabled.

---

## 9. Outputs and Artifacts

| Stage | Artifact | Location |
|-------|----------|----------|
| A1 | EEG MAE checkpoint | `results/eeg_pretrain/<timestamp>/checkpoints/checkpoint.pth` → copy to `pretrains/eeg_pretain/checkpoint.pth` |
| B | LDM checkpoint | `results/exps/results/generation/<timestamp>/checkpoint.pth`, `checkpoint_best.pth` |
| B (SAR-HM) | Prototypes | `.../prototypes.pt` |
| B | Train logs | `results/runs/<timestamp>_<model>_<seed>/config.json`, `train_log.csv` |
| C | Grids, per-image PNGs | `results/eval/<timestamp>/` |
| Compare-eval | Metrics, grids, samples | `results/compare_eval_thesis/metrics/`, `grids/`, `baseline/`, `sarhm/` |
| **Unified benchmark** | Standardized outputs, metrics, timing, tables, panels | `results/benchmark_outputs/<dataset>/sample_<id>/` (ground_truth.png, thoughtviz.png, dreamdiffusion.png, sarhm.png, metadata.json); optional `results/experiments/<run_name>/` |

---

## 10. Step-by-Step Summary: What We Do in Each Stage

### Stage A (A1)
1. Load EEG from `mne_data/*.npy` or `eeg_5_95_std.pth`.
2. Normalize and fix shape to 128×512; optionally crop time [20:460].
3. Train MAEforEEG: mask patches → encode → decode → reconstruction loss.
4. Save encoder weights to `checkpoint.pth` and copy to `pretrains/eeg_pretain/checkpoint.pth` for Stage B.

### Stage B
1. Load EEG + splits + ImageNet path; build train/test via `create_EEG_dataset` and Splitter.
2. Load pretrained MAE from Stage A1; load SD 1.5 (VAE + UNet); build cond_stage_model (MAE → dim_mapper; optional SAR-HM path with prototypes).
3. Freeze VAE and UNet; train only cond_stage_model with diffusion loss (and optional CLIP/SAR-HM losses).
4. Optionally build/load prototypes for SAR-HM; run PyTorch Lightning training; save checkpoint and (for SAR-HM) prototypes.pt.

### Stage C
1. Load Stage B checkpoint and config; build eLDM_eval (no MAE metafile; load state_dict and optionally prototypes).
2. Build same EEG dataset (test or train split); for each item, get conditioning from EEG, run PLMS sampling, decode with VAE.
3. Save grids and per-image files; compute and report metrics (SSIM, PCC, CLIP, etc.). compare_eval.py does the same with two checkpoints (baseline and SAR-HM) on the same EEG subset.

---

## 11. Unified Benchmark (ThoughtViz, DreamDiffusion, SAR-HM)

A **unified benchmark** subsystem allows fair comparison of three EEG-to-image models on two datasets. **Only normal SAR-HM** is used in this benchmark; SAR-HM++ is excluded (separate/future work).

### 11.1 Scope

| Models | Description |
|--------|-------------|
| **ThoughtViz** | GAN-based; official repo under `code/ThoughtViz/` or `codes/ThoughtViz/`. EEG classifier encoding + generator; Keras/TF. Wrapped via `code/thoughtviz_integration/` (ThoughtVizWrapper: load_pretrained, generate_from_eeg, save_outputs). |
| **DreamDiffusion baseline** | EEG → MAE → conditioning → Stable Diffusion 1.5 (same as Stage B/C baseline). |
| **DreamDiffusion + SAR-HM** | Same as baseline + Hopfield retrieval over class prototypes + confidence-gated fusion (same as Stage B/C SAR-HM). |

| Datasets | Description |
|----------|-------------|
| **ImageNet-EEG** | EEG from `eeg_5_95_std.pth`, splits from `block_splits_*.pth`, GT images from ImageNet path. Exposed via `create_EEG_dataset` → unified sample format (sample_id, eeg, ground_truth, label, metadata, split). |
| **ThoughtViz** | ThoughtViz `data.pkl` (EEG) + images by class. Adapter: `ThoughtVizDatasetAdapter` in `thoughtviz_integration/dataset_adapter.py`; same unified sample interface. |

### 11.2 Architecture

- **Dataset registry** (`benchmark/dataset_registry.py`): `get_dataset(dataset_name, ...)` returns a list of unified sample dicts (sample_id, eeg, ground_truth path/tensor, label, metadata, split).
- **Model registry** (`benchmark/model_registry.py`): `get_model(model_name, config)` returns wrapper instances; `generate_dreamdiffusion` / `generate_thoughtviz` produce images from EEG samples. DreamDiffusion/SAR-HM reuse Stage C logic via `utils_eval.load_model` and LDM `generate()`.
- **Output standardizer** (`benchmark/output_standardizer.py`): Saves to `results/benchmark_outputs/<dataset>/sample_<id>/` with `ground_truth.png`, `thoughtviz.png`, `dreamdiffusion.png`, `sarhm.png`, `metadata.json`; standard eval size (e.g. 256×256); merge_with_existing so multiple models can write to the same sample dir.
- **Metrics**: Core metrics (SSIM, PCC, CLIP) via `metrics_runner.py`; timing via `timing_runner.py`; tables via `table_generator.py`; qualitative panels via `visualization_runner.py`. Instance segmentation and image caption comparison are stubbed (optional/future).

### 11.3 Workflow Phases

1. **Sanity:** Run `tests/test_full_pipeline_sanity.py` to verify dataset load, model load, one-sample inference, and metrics pipeline for all three models and both datasets.
2. **Small benchmark:** `python -m benchmark.compare_all_models --dataset imagenet_eeg --max_samples 10` (or 20) with paths for ImageNet, baseline checkpoint, SAR-HM checkpoint + prototypes. Same for `--dataset thoughtviz` when ThoughtViz data is present.
3. **Multi-run:** Results can be organized under `results/experiments/run_001/`, etc., with config, metrics, timing, tables, visualizations.
4. **Final comparison:** Full runs with consistent metrics, tables (ImageNet-EEG, ThoughtViz, timing, optional segmentation/caption), and qualitative panels (GT | ThoughtViz | DreamDiffusion | SAR-HM).

### 11.4 Commands and Documentation

- **Exact commands:** `docs/commands.md` (sanity, small benchmark, ThoughtViz train/test, DreamDiffusion/SAR-HM, comparison, metrics, tables).
- **ThoughtViz integration:** `docs/thoughtviz_integration.md` (wrapper API, data paths, Keras/TF dependencies).
- **Benchmark workflow:** `docs/benchmark_workflow.md` (phases, folder structure, MSC documented as NA if not in codebase).

---

**Inference safeguards (EEG-only):** At Stage C and in compare_eval, no ground-truth semantic targets (`z_sem_gt`, `clip_img_embed_gt`, `summary_embed_gt`) or GT images are used as input. Conditioning is: EEG → cond_stage_model → c. For SAR-HM++, the semantic path uses only EEG-derived `q_sem` and the preloaded `semantic_prototypes.pt`; the SemanticTargetWrapper is applied only to the **train** split in Stage B, and the test split is never wrapped.

This document reflects the implementation as of the codebase structure and configs described above. For SAR-HM ablation modes see `docs/SARHM_README.md`; for SAR-HM++ (semantic targets in batch, CLIP losses, commands) see `docs/SARHMPP_README.md` and `docs/SARHMPP_IMPLEMENTATION_STATUS.md`. For the unified benchmark (ThoughtViz, DreamDiffusion, SAR-HM on ImageNet-EEG and ThoughtViz datasets) see Section 11, `docs/commands.md`, `docs/benchmark_workflow.md`, and `docs/thoughtviz_integration.md`.
