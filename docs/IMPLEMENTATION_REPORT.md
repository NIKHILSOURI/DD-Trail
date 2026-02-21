# DreamDiffusion + SAR-HM: Complete Technical Implementation Report

**Thesis-ready documentation** · Extracted from repository · No code modifications.

---

## PART 1 — Repository Inventory

### 1.1 Scripts

| Script | Path | Role |
|--------|------|------|
| **Stage A1** | `code/stageA1_eeg_pretrain.py` | Pre-trains the EEG encoder (MAE) using masked signal modeling |
| **Stage B** | `code/eeg_ldm.py` | Fine-tunes the full generative model: EEG encoder + conditioning path + (optionally) SAR-HM |
| **Stage C** | `code/gen_eval_eeg.py` | Loads a trained checkpoint and generates images from EEG; computes evaluation metrics |
### 1.2 SAR-HM Modules

| Module | Path | Role |
|--------|------|------|
| `sarhm_modules.py` | `code/sarhm/sarhm_modules.py` | `pool_eeg_tokens`, `SemanticProjection`, `HopfieldRetrieval`, `ConfidenceGatedFusion`, `ConditioningAdapter` |
| `prototypes.py` | `code/sarhm/prototypes.py` | `ClassPrototypes`, `build_prototypes_from_loader` |
| `metrics_logger.py` | `code/sarhm/metrics_logger.py` | `retrieval_accuracy`, `attention_entropy_mean`, `confidence_stats`, `sarhm_metrics_from_extra`, `append_ablation_results_row` |
| `vis.py` | `code/sarhm/vis.py` | `save_hopfield_attention_bar`, `save_baseline_vs_sarhm_grid` |
### 1.3 Prototype Storage

| Location | When Used |
|----------|-----------|
| `output_path/prototypes.pt` | Saved after Stage B when `use_sarhm=True` (by `ldm_for_eeg.finetune`) |
| Checkpoint `model_state_dict` | Prototypes are part of `cond_stage_model.sarhm_prototypes.prototypes`; always saved/loaded with checkpoint |
| `config.proto_path` | Optional; if set, `ClassPrototypes.load_from_path()` loads at init; `save_to_path()` saves after training |

### 1.4 Ablation Modes

| Defined In | Enforced In |
|------------|-------------|
| `code/config.py` → `Config_Generative_Model.ablation_mode` | `code/dc_ldm/ldm_for_eeg.py` → `cond_stage_model.forward()` (lines 134–146) |

### 1.5 Major File Role Summaries

| File | Role |
|------|------|
| `code/config.py` | All hyperparameters: `Config_MBM_EEG` (Stage A1), `Config_Generative_Model` (Stage B/C), `Config_Cls_Model` |
| `code/dataset.py` | `eeg_pretrain_dataset`, `EEGDataset`, `Splitter`, `create_EEG_dataset`; EEG/image loading and splits |
| `code/dc_ldm/ldm_for_eeg.py` | `cond_stage_model`, `eLDM`, `eLDM_eval`; conditioning path (baseline + SAR-HM), training, generation |
| `code/dc_ldm/models/diffusion/ddpm.py` | `LatentDiffusion`; `get_learned_conditioning`, `shared_step`, `p_losses`, optimizer config |
| `code/sarhm/sarhm_modules.py` | SAR-HM building blocks: projection, Hopfield, fusion, adapter |
| `code/sarhm/prototypes.py` | `ClassPrototypes` (learnable [K, dim]); `build_prototypes_from_loader` (optional pre-build) |
| `code/eval_metrics.py` | `get_similarity_metric`, MSE/PCC/SSIM/PSM, `get_n_way_top_k_acc` (ViT-based class accuracy) |
| `code/gen_eval_eeg.py` | Stage C: load checkpoint, generate, save images, optional SAR-HM attention plot, `ablation_results.csv` |

---

## PART 2 — Baseline DreamDiffusion

### 2.1 What Baseline Is

Baseline DreamDiffusion: **EEG → MAE encoder → channel mapper (if `global_pool=False`) → dim mapper → Stable Diffusion conditioning**.

- No semantic projection.
- No Hopfield memory.
- Direct mapping from EEG latent space to diffusion conditioning.

### 2.2 Implementation Locations

| Step | File | Function | Approx. Lines |
|------|------|----------|---------------|
| Conditioning path | `code/dc_ldm/ldm_for_eeg.py` | `cond_stage_model.forward()` | 117–156 |
| Learned conditioning | `code/dc_ldm/models/diffusion/ddpm.py` | `get_learned_conditioning()` | 751–763 |
| Stage B entry | `code/eeg_ldm.py` | `main()` | 123–194 |
| Stage C entry | `code/gen_eval_eeg.py` | `__main__` | 71–186 |

### 2.3 Models Used

| Component | Type | Source |
|-----------|------|--------|
| EEG encoder | `eeg_encoder` (ViT-style, encoder-only, no masking) | `sc_mbm.mae_for_eeg.eeg_encoder` |
| Pre-training | `MAEforEEG` (Masked Autoencoder) | `sc_mbm.mae_for_eeg.MAEforEEG` |
| SD backbone | Stable Diffusion 1.5 | `pretrains/models/v1-5-pruned.ckpt` |
| SD config | `config15.yaml` | `pretrains/models/config15.yaml` |

**SD is frozen** (UNet, VAE, text encoder). Only the conditioning path (EEG encoder, channel mapper, dim mapper) is trained.

### 2.4 Trainable Parts (Baseline)

- `cond_stage_model.mae` (EEG encoder)
- `cond_stage_model.channel_mapper` (if `global_pool=False`)
- `cond_stage_model.dim_mapper`
- `cond_stage_model.mapping` (if `clip_tune=True`)
- Selected UNet params: `attn2`, `time_embed_condtion`, `norm2` (when `train_cond_stage_only=True`)

### 2.5 Training Hyperparameters (from Config)

| Parameter | Value | Config class |
|-----------|-------|--------------|
| Epochs | 500 | `Config_Generative_Model.num_epoch` |
| Batch size | 12 (EEG), 5 (GOD) | `Config_Generative_Model.batch_size` |
| Learning rate | 5.3e-5 | `Config_Generative_Model.lr` |
| Accumulate grad | 2 (effective batch 24) | `Config_Generative_Model.accumulate_grad` |
| Precision | 16 (FP16) | `Config_Generative_Model.precision` |
| DDIM steps (sampling) | 250 | `Config_Generative_Model.ddim_steps` |

“Iterations” = epochs; each epoch = one pass over the training set. Steps = batches per epoch.

### 2.6 Tensor Shapes (Baseline)

| Stage | Shape | Description |
|-------|-------|-------------|
| EEG input | `(B, C, 512)` | C = channels (e.g. 128); time = 512 (`data_len`); `EEGDataset` crops to 440 time then interpolates to 512 |
| MAE output | `(B, num_patches, 1024)` | `num_patches = time_len / patch_size`; e.g. 512/4 = 128 |
| Channel mapper output | `(B, 77, 1024)` | If `global_pool=False`; 1D conv reduces seq_len 128→77 |
| Dim mapper output | `(B, 77, 768)` | Linear 1024→768 (cond_dim from `config15.yaml`) |
| SD conditioning | `(B, 77, 768)` | Matches SD `context_dim` |

SD expects `[B, 77, 768]` via `conditioning_key: crossattn`; pipeline uses dataset-dependent EEG `(channels, 512)`.

---

## PART 3 — SAR-HM / Hopfield Semantic Associative Retrieval

### 3.1 What SAR-HM Means

**Semantic Associative Retrieval with Hopfield Memory**: the EEG-derived vector is projected into a shared semantic space, then used as a **query** into a **Hopfield-style associative memory** of class prototypes. The memory returns a mixture of prototypes; a **confidence-gated fusion** blends this with the raw projection.

### 3.2 Integration Point

- **File**: `code/dc_ldm/ldm_for_eeg.py`
- **Function**: `cond_stage_model.forward()`
- **Branch**: `if self.use_sarhm:` (lines 123–147)
- **Location**: Between MAE output and Stable Diffusion conditioning.

### 3.3 SAR-HM Modules

| Module | File | Input | Output | Config Params |
|--------|------|-------|--------|---------------|
| `pool_eeg_tokens` | `sarhm_modules.py` | `(B, seq, 1024)` | `(B, 1024)` | `global_pool`: mean vs `[:, 0]` |
| `SemanticProjection` | `sarhm_modules.py` | `(B, 1024)` | `(B, 768)` | `fmri_latent_dim`, `clip_dim` |
| `ClassPrototypes` | `prototypes.py` | — | `P`: `(K, 768)` | `num_classes`, `dim`, `proto_path` |
| `HopfieldRetrieval` | `sarhm_modules.py` | `query (B,768)`, `P (K,768)` | `(B,768)`, `(B,K)`, `(B,K)` | `hopfield_tau` |
| `ConfidenceGatedFusion` | `sarhm_modules.py` | `z_orig`, `z_ret`, `attn (B,K)` | `(B,768)`, `(B)` | `gate_mode` |
| `ConditioningAdapter` | `sarhm_modules.py` | `(B, 768)` | `(B, 77, 768)` | `clip_dim`, `cond_dim`, `seq_len` |

### 3.4 How Prototypes Are Built

- **Default**: Learnable `nn.Parameter` `(K, 768)`, initialized `normal_(std=0.02)`.
- **Optional**: `build_prototypes_from_loader()` in `prototypes.py` builds mean centroids from training data (not used by default).
- **Optional**: `ClassPrototypes.update_from_batch()` updates via EMA per class (not used by default).
- **When used**: Trained via backprop as part of `cond_stage_model`; loaded from `proto_path` if given; saved to `output_path/prototypes.pt` after Stage B.

### 3.5 Retrieval and Fusion

1. **Query**: `z_orig = SemanticProjection(pool(MAE(latent)))` → `(B, 768)`
2. **Similarity**: `logits = q @ P^T` → `(B, K)`
3. **Attention**: `attn = softmax(logits / tau)` → `(B, K)`
4. **Retrieval**: `z_ret = attn @ P` → `(B, 768)`
5. **Confidence** (`gate_mode='max'`): `confidence = max(attn, dim=-1)`
6. **Confidence** (`gate_mode='entropy'`): `confidence = 1 - entropy / max_entropy`
7. **Fusion**: `z_fused = confidence * z_ret + (1 - confidence) * z_orig`
8. **Adapter**: `ConditioningAdapter(z_fused)` → `(B, 77, 768)` for SD.

### 3.6 Ablation Modes

| Repo `ablation_mode` | SR/SAR Mapping | Active Modules | Trained | Config |
|----------------------|----------------|----------------|---------|--------|
| `baseline` | — | MAE, channel_mapper, dim_mapper | Same as baseline | `use_sarhm=False` |
| `projection_only` | **SR** (projection only) | MAE, pool, SemanticProjection, ConditioningAdapter | Projection + adapter | `use_sarhm=True` |
| `hopfield_no_gate` | **SAR** (retrieval only) | + HopfieldRetrieval; no fusion | + Hopfield + prototypes | `use_sarhm=True` |
| `full_sarhm` | **Full SAR-HM** | + ConfidenceGatedFusion | All SAR-HM params | `use_sarhm=True` |

---

## PART 4 — Training Plan

### 4.1 Stage Summary Table

| Stage | Script | Trained | Frozen | Epochs/Steps | Output |
|-------|--------|---------|--------|--------------|--------|
| **A1** | `stageA1_eeg_pretrain.py` | MAE encoder | — | 500 | `results/eeg_pretrain/<ts>/checkpoints/checkpoint.pth` |
| **B** | `eeg_ldm.py` | EEG encoder, conditioning path (incl. SAR-HM if on) | VAE, UNet (except attn2/time_embed/norm2), text encoder | 500 | `results/generation/<ts>/checkpoint.pth` |
| **C** | `gen_eval_eeg.py` | — | All | 0 | Images, metrics |
### 4.2 Losses

| Loss | When | Where |
|------|------|-------|
| Diffusion loss (simple + VLB) | Always | `ddpm.p_losses()` |
| CLIP alignment | If `clip_tune=True` | `cond_stage_model.get_clip_loss(re_latent, image_embeds)` |
| CLS loss | If `cls_tune=True` | `cls_loss(label, pre_cls)` |

CLIP: frozen image embedder; `image_embeds` from `FrozenImageEmbedder(image_raw)` or `batch['target_embeds']`.

### 4.3 Stage A1 Details

| Parameter | Value |
|-----------|-------|
| Epochs | 500 |
| Batch size | 100 |
| LR | 2.5e-4 |
| Mask ratio | 0.1 |
| Patch size | 4 |
| Embed dim | 1024 |

---

## PART 5 — Evaluation

### 5.1 SAR-HM Outputs

- Generated image (same as baseline)
- Attention weights `attn` over K classes
- Confidence per sample
- Retrieval accuracy: `argmax(attn) == label`

### 5.2 Metrics Table

| Metric | Script | Function | Notes |
|--------|--------|----------|-------|
| MSE | `eval_metrics.py` | `mse_metric` | Pair-wise, smaller better |
| PCC | `eval_metrics.py` | `pcc_metric` | Pair-wise, larger better |
| SSIM | `eval_metrics.py` | `ssim_metric` | Pair-wise, larger better |
| PSM (LPIPS) | `eval_metrics.py` | `psm_wrapper` | Pair-wise, smaller better |
| Top-1 class | `eval_metrics.py` | `get_n_way_top_k_acc` | ViT-H/14, n-way, top-k |
| Hopfield retrieval acc | `sarhm/metrics_logger.py` | `retrieval_accuracy` | SAR-HM only |
| Attention entropy | `sarhm/metrics_logger.py` | `attention_entropy_mean` | SAR-HM only |
| Confidence stats | `sarhm/metrics_logger.py` | `confidence_stats` | SAR-HM only |
| FID | `eval_metrics.py` | `fid_wrapper` | Implemented; not in default eval loop |

### 5.3 Evaluation Flow

1. `eeg_ldm.generate_images()` → `generative_model.generate()` → `get_eval_metric()` (MSE, PCC, SSIM, PSM, top-1-class).
2. `gen_eval_eeg.py` → same generate + eval; optional SAR-HM attention plot; `append_ablation_results_row()`.

### 5.4 Experiment Structure

| Split | File | Use |
|-------|------|-----|
| Within-subject | `block_splits_by_image_single.pth` | One subject (e.g. `subject=4`) |
| Subject-agnostic | `block_splits_by_image_all.pth` | All subjects |

Recommended: ablations (baseline, projection_only, hopfield_no_gate, full_sarhm), qualitative comparisons, attention/confidence plots.

---

## PART 6 — Glossary

| Term | Explanation |
|------|-------------|
| **EEG encoder / MAE** | Encoder-only ViT: patches EEG, applies Transformer blocks; pre-trained with masking (MAE) in Stage A1. |
| **Masked modeling** | Random masking of patches; encoder sees unmasked patches; decoder reconstructs masked patches. |
| **Conditioning tokens** | Vector sequence `[B, 77, 768]` fed to SD UNet cross-attention instead of text embeddings. |
| **Stable Diffusion conditioning** | `conditioning_key: crossattn`; conditioning as cross-attention context in the UNet. |
| **CLIP similarity** | Cosine similarity between generated and ground-truth CLIP embeddings; used as a loss and for evaluation. |
| **Prototypes** | Per-class vectors `(K, 768)` in semantic space; used as Hopfield memory. |
| **Hopfield retrieval** | Query–memory similarity → softmax attention → weighted sum of prototypes. |
| **Confidence gating** | Scalar from attention (max or entropy) blending retrieved vs. original projection. |
| **Entropy** | `-sum(p log p)` over attention; high entropy = uncertain; used for confidence in `gate_mode='entropy'`. |
| **Ablation study** | Experiments that disable parts of the model to measure their contribution. |
| **Within-subject** | Train and test on the same subject. |
| **Cross-subject** | Train on some subjects, test on others. |

---

*End of Implementation Report*
