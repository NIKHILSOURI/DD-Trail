# Debug: Pure Noise from EEG-to-Image Generation (DreamDiffusion + SAR-HM)

## 1) One-line Problem Statement

Generated images are **pure noise** (or totally static/unrecognizable even after many sampling steps) despite **training losses looking stable** (e.g. `train/loss_total` ~0.13–0.16, `train/sarhm_retrieval_acc` rising to ~1.0). The exact "10 approaches" or moment when noise started is **MISSING** from the repo (no changelog or dated notes). "Noise" here means: outputs that do not resolve into coherent images after 250 PLMS steps.

---

## 2) Repo Snapshot

- **Project root:** `DREAMDIFFUSION` (workspace path: `d:\STUDY\BTH\THESIS\DREAMDIFFUSION`).
- **Git:** Not a git repo (per user_info: `Is directory a git repo: No`). Branch / last commit hash: **MISSING**.
- **Key directories:**
  - **Configs:** `code/config.py`, `pretrains/models/config15.yaml`. Run configs saved under `runs/<timestamp>_<model>_<seed>/config.json`.
  - **Scripts:** `code/eeg_ldm.py` (Stage B training), `code/gen_eval_eeg.py` (Stage C inference), `code/stageA1_eeg_pretrain.py` (Stage A1).
  - **Checkpoints:** `pretrains/models/v1-5-pruned.ckpt` (SD 1.5), `pretrains/eeg_pretain/checkpoint.pth` (EEG encoder). Stage B outputs: `results/generation/<timestamp>/checkpoint.pth` (or path in config `output_path`).
  - **Logs:** `runs/<timestamp>_<model>_<seed>/train_log.csv`, `lightning_logs/version_*/metrics.csv`, `lightning_logs/version_*/hparams.yaml`.
  - **Outputs:** Stage C writes to `results/eval/<date-time>/` (samples_train.png, samples_test.png, test*.png).

---

## 3) Exact Inference Entry Point

- **Script:** `code/gen_eval_eeg.py` (Stage C: generate and evaluate).
- **CLI (from README):**
  ```bash
  python code/gen_eval_eeg.py --dataset EEG --model_path results/generation/<timestamp>/checkpoint.pth --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
  ```
  Optional: `--imagenet_path datasets/imageNet_images`, `--root`, `--config_patch` (SD config).
- **Checkpoint path:** From `--model_path`; also used in config as saved in checkpoint: `config = sd['config']` (so paths inside config may point to RunPod paths, e.g. `output_path`: `/workspace/DreamDiffusion_SAR-HM/exps/...`).
- **Output folder:** `config.root_path + 'results/eval/' + datetime` → e.g. `../dreamdiffusion/results/eval/28-02-2026-21-42-16` (or from `config.root_path` in checkpoint).

---

## 4) Model + Diffusion Stack Versions

| Item | Value | Where defined |
|------|--------|----------------|
| **Stable Diffusion** | 1.5 (v1-5-pruned), LDM latent diffusion | README; `pretrains/models/v1-5-pruned.ckpt`, `config15.yaml` |
| **VAE** | AutoencoderKL (LDM), embed_dim=4, ch_mult (1,2,4,4) | `pretrains/models/config15.yaml` → `first_stage_config.target: dc_ldm.models.autoencoder.AutoencoderKL` |
| **UNet** | in_channels=4, out_channels=4, model_channels=320, attention_resolutions [4,2,1], channel_mult [1,2,4,4], num_heads=8, context_dim=768 | `config15.yaml` → `unet_config.params` |
| **Scheduler / sampler** | PLMS (Pseudo Linear Multistep); DDIM-style timesteps from `make_ddim_timesteps` | `code/dc_ldm/models/diffusion/plms.py`; `ldm_for_eeg.py`: `sampler = PLMSSampler(model)` |
| **Prediction type** | **eps** (noise prediction) | `ddpm.py` line 94: `parameterization="eps"`; plms.py line 197: `assert self.model.parameterization == "eps"` |
| **EMA** | **use_ema: False** in config | `config15.yaml` line 17: `use_ema: False` → inference uses non-EMA weights |

---

## 5) Latent Scaling + Decode/Encode Pipeline (Critical)

- **Scale factor:** `0.18215` (fixed in SD config).  
  **File:** `pretrains/models/config15.yaml` line 16: `scale_factor: 0.18215`.  
  **Training:** `scale_by_std` is **False** (ddpm.py line 611); so no data-dependent rescaling; fixed `scale_factor` is used.
- **Encode (image → latent):**  
  `ddpm.py`: `get_first_stage_encoding(encoder_posterior)` returns `self.scale_factor * z` (line 764). So stored latent = scale_factor * VAE(z).
- **Decode (latent → image):**  
  `decode_first_stage` (line 944): `z = 1. / self.scale_factor * z` then `self.first_stage_model.decode(z)`. So decode path divides by scale_factor then VAE decode.
- **Consistency:** Same scale_factor is used at encode (train) and decode (inference); config is loaded from checkpoint in Stage C, but **scale_factor comes from config15.yaml** at model instantiation (eLDM_eval loads `config_path` YAML). So if Stage C is run with a different `config_patch` that had a different scale_factor, train vs inference would be inconsistent. **Check:** Ensure `config_patch` is the same `config15.yaml` used during training (scale_factor 0.18215).
- **Snippets:**
  - Encode (training): `code/dc_ldm/models/diffusion/ddpm.py` ~764: `return self.scale_factor * z`
  - Decode (inference): `code/dc_ldm/models/diffusion/ddpm.py` ~951: `z = 1. / self.scale_factor * z` then `return self.first_stage_model.decode(z)`

---

## 6) Conditioning Path (EEG + SAR-HM) Wiring

- **Tensor into UNet:** Cross-attention conditioning `c_crossattn`: list of one tensor shape `[B, 77, 768]` (context_dim=768, 77 tokens).  
  **Source:** `cond_stage_model.forward(x)` returns `(c_final, z_fused)` or `(c_base, latent_return)`; `get_learned_conditioning` returns the first element `c`.  
  **ddpm.py** `apply_model` (line 1184): `key = 'c_crossattn'`, `cond = {key: [cond]}`.
- **SAR-HM fusion:** In `ldm_for_eeg.py` `cond_stage_model.forward`: `c_final = c_base + alpha_bc * (c_sar - c_base)` with `alpha_bc = alpha.view(-1, 1, 1)`. Alpha from `compute_alpha_from_attention` (entropy-based) or `alpha_constant`; clamped by `alpha_max`, gating by `conf_threshold`. No extra LayerNorm/L2 on c_final in this file.
- **Expected shape:** `[B, 77, 768]` (B = batch, 77 = sequence, 768 = context_dim).  
- **Snippet (fusion):** `code/dc_ldm/ldm_for_eeg.py` lines 209–212:
  ```python
  alpha_bc = alpha.view(-1, 1, 1).to(c_base.dtype)
  c_final = c_base + alpha_bc * (c_sar - c_base)
  ```
  Defaults: `alpha_mode='entropy'`, `alpha_max=0.2`, `conf_threshold=0.2`, `alpha_constant=0.1` (config.py / checkpoint config).

---

## 7) Checkpoint Loading Integrity

- **Sampling script load:** `code/gen_eval_eeg.py` lines 87–133:
  ```python
  sd = torch.load(args.model_path, map_location='cpu', weights_only=False)
  config = sd['config']
  ...
  generative_model = eLDM_eval(..., main_config=config)
  generative_model.model.load_state_dict(sd['model_state_dict'], strict=False)
  ```
- **strict=False:** Yes. Missing/unexpected keys are **not** printed in gen_eval_eeg.py (no log of missing_keys / unexpected_keys). So we do not know from the script alone if SAR-HM or adapter weights failed to load.
- **SAR-HM / adapter:** Cond_stage_model is built from `main_config` (from checkpoint); then state_dict is applied. If checkpoint has `cond_stage_model.sarhm_*` keys, they are loaded into the same-named modules. If the run used SAR-HM, those keys should be present; if loading on a machine where `sarhm` failed to import, cond_stage_model would be baseline-only and might have unexpected keys or missing keys for SAR-HM.
- **Checkpoint formats:** Repo uses `.pth` (torch.save: `config`, `model_state_dict`, `state` for RNG). No `.ckpt` or `.safetensors` for Stage B output in the described flow.

---

## 8) Training vs Inference Config Diff

Run config used: `runs/20260228_214324_sarhm_2022/config.json`. Inference uses the same config from the checkpoint plus `config_patch` YAML for SD structure.

| Parameter | Training (config.json / config.py) | Inference (Stage C) | Match? |
|-----------|------------------------------------|----------------------|--------|
| scheduler / sampler | PLMS (training uses DDPM loss; val/sampling use PLMS) | PLMS | ✓ |
| timesteps | 1000 (DDPM), ddim_steps 250 | config.ddim_steps 250 | ✓ |
| guidance scale | Not used (unconditional_guidance_scale=1.0 in plms) | 1.0 | ✓ |
| VAE | From config15 (AutoencoderKL) | Same config_patch | ✓ |
| precision | bf16 | Script does not set autocast; model in eval() | Could differ (fp32 by default in inference) |
| image size | 512 (img_size); latent 64×64 | 64×64 latent (image_size 64 in config) | ✓ |
| latent scaling | 0.18215 (from config15) | 0.18215 (from config_patch) | ✓ (if same file) |
| conditioning dim | 768, 77 tokens | 768, 77 | ✓ |
| seed | 2022 (config); state in sd['state'] | Optional RNG state from checkpoint | ✓ |
| EMA | use_ema: False | ema_scope no-op | ✓ |

Possible mismatch: **precision** (training bf16, inference default fp32) — could affect numerics but usually not "pure noise." **config_patch** must point to the same config15.yaml (same scale_factor and UNet/VAE layout).

---

## 9) Metrics/Logs Evidence

- **train_log.csv** (last 30 rows, run 20260228_214324_sarhm_2022):  
  Rows 498–502 (epochs 497–500):

| step | epoch | train/loss_total | train/sarhm_retrieval_acc | train/sarhm_attention_entropy | train/loss_retrieval |
|------|--------|-------------------|----------------------------|--------------------------------|------------------------|
| 6972 | 497 | 0.15842 | 1.0 | 0.691824 | 0.15268 |
| 6986 | 498 | 0.151388 | 1.0 | 0.656268 | 0.139345 |
| 7000 | 499 | 0.157218 | 1.0 | 0.681095 | 0.146963 |
| 7014 | 500 | (last) | (last) | (last) | (last) |

- **Trends:**  
  - `train/loss_total`: stable ~0.13–0.16.  
  - `train/sarhm_retrieval_acc`: rises to ~1.0 by mid-training.  
  - `train/sarhm_attention_entropy`: decreases over time (e.g. 2.7 → ~0.66).  
  - `train/loss_retrieval`: decreases (e.g. 5.3 → ~0.14).  
  So training and SAR-HM metrics look healthy; no obvious collapse.
- **Val metrics:** With `disable_image_generation_in_val: true`, validation does not run full image generation; `val/skip_image_generation` is logged. So no val SSIM/CLIP in this run to compare to "good-detail" runs.

---

## 10) Minimal Sanity Checks We Should Run (No Code Changes)

**A) Text-only sampling baseline with same checkpoint + sampler**  
- Not directly supported: the pipeline is EEG-conditioned; there is no text encoder in the eval script. So **skip** or implement a separate "null" conditioning test (e.g. fixed zeros [B,77,768]) to see if the UNet+VAE alone produce non-noise.  
- *Implication:* If null cond gives structured images, problem is conditioning; if still noise, problem is UNet/VAE/scale_factor/sampler.

**B) Decode a known-good latent with current VAE pipeline**  
- Encode a real image with the same model (e.g. in Python: get batch from dataset, run `model.encode_first_stage`, then `model.decode_first_stage(z)`). Compare to original.  
- *Implication:* If decode is wrong, scale_factor or VAE state is wrong; if decode is correct, issue is earlier (sampling or conditioning).

**C) Run sampling with autocast disabled (fp32)**  
- Run Stage C with `torch.cuda.amp.autocast(enabled=False)` around the generate call (or set env `PYTORCH_AUTOCAST=0` if supported).  
- *Implication:* If images improve, bf16/fp16 numerical mismatch is likely.

**D) Run with CFG=1.0 vs 7.5 vs no-CFG**  
- PLMS sampler supports `unconditional_guidance_scale` and `unconditional_conditioning`. Currently they are not passed in `ldm_for_eeg.py` (so effectively 1.0 / no CFG). To test: pass `unconditional_conditioning` (e.g. zeros [B,77,768]) and `unconditional_guidance_scale=7.5` in `sampler.sample(...)`.  
- *Implication:* If CFG changes behavior, conditioning strength or format is relevant.

**E) Fixed seed, 1 sample**  
- Use a fixed RNG seed and `num_samples=1`, single test item.  
- *Implication:* Reproducibility and whether noise is per-sample or global.

---

## 11) Hypothesis Ranking (Based on Repo Evidence)

1. **Conditioning path / SAR-HM not active or wrong at inference**  
   - Evidence: `cond_stage_model` is built from checkpoint `main_config`; if `sarhm` fails to import on inference machine, "SAR-HM: OFF" and baseline path is used; checkpoint may have been trained with SAR-HM so cond shape/statistics could differ.  
   - Test: Run Stage C and check console for "SAR-HM ACTIVE" vs "SAR-HM: OFF"; ensure PYTHONPATH includes `code/` and `sarhm` imports.

2. **scale_factor or config_patch mismatch at inference**  
   - Evidence: Decode uses `z = 1. / self.scale_factor * z`; scale_factor comes from the YAML loaded in eLDM_eval (config_path). If a different YAML is used (e.g. different scale_factor), decoded images will be wrong.  
   - Test: Print `model.scale_factor` in gen_eval_eeg after model load; confirm 0.18215.

3. **Checkpoint missing keys or wrong device/dtype**  
   - Evidence: `load_state_dict(..., strict=False)` with no logging of missing/unexpected keys; SAR-HM/adapter weights might be missing or overwritten.  
   - Test: Temporarily log `missing_keys` and `unexpected_keys` after load_state_dict and inspect.

4. **Precision / autocast mismatch (train bf16 vs inference fp32)**  
   - Evidence: Training uses `precision: bf16`; gen_eval_eeg does not set autocast; get_learned_conditioning forces float32 for conditioner.  
   - Test: Run inference with autocast disabled (fp32) or with bf16 to match training.

5. **RNG state or seed**  
   - Evidence: Checkpoint stores `sd['state']` (CUDA RNG); gen_eval_eeg passes it to generate(); if state is from different GPU/PyTorch version, it can be skipped (script catches "wrong size" etc.).  
   - Test: Run with fixed seed and without loading state; compare to run with state.

---

## 12) Appendix: Key Snippets

**Sampler creation** (`code/dc_ldm/ldm_for_eeg.py` eLDM_eval.generate, ~634–636):
```python
model = self.model.to(self.device)
sampler = PLMSSampler(model)
```

**Timestep schedule** (`code/dc_ldm/models/diffusion/plms.py` make_schedule, ~25–29):
```python
self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                          num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
```
`make_ddim_timesteps` in `util.py`: uniform spacing `range(0, num_ddpm_timesteps, c)` with `c = num_ddpm_timesteps // num_ddim_timesteps`.

**VAE decode** (`code/dc_ldm/models/diffusion/ddpm.py` decode_first_stage, ~943–951):
```python
z = 1. / self.scale_factor * z
return self.first_stage_model.decode(z)
```

**VAE encode** (training, get_first_stage_encoding, ~756–764):
```python
def get_first_stage_encoding(self, encoder_posterior):
    ...
    return self.scale_factor * z
```

**Conditioning fusion** (`code/dc_ldm/ldm_for_eeg.py` cond_stage_model.forward, ~209–212):
```python
alpha_bc = alpha.view(-1, 1, 1).to(c_base.dtype)
c_final = c_base + alpha_bc * (c_sar - c_base)
return c_final, z_fused
```

**Checkpoint load** (`code/gen_eval_eeg.py` ~87–133):
```python
sd = torch.load(args.model_path, map_location='cpu', weights_only=False)
config = sd['config']
...
generative_model = eLDM_eval(args.config_patch, num_voxels, ..., main_config=config)
generative_model.model.load_state_dict(sd['model_state_dict'], strict=False)
```

**Guidance logic** (`code/dc_ldm/models/diffusion/plms.py` p_sample_plms get_model_output, ~187–196):
```python
if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
    e_t = self.model.apply_model(x, t, c)
else:
    x_in = torch.cat([x] * 2)
    t_in = torch.cat([t] * 2)
    c_in = torch.cat([unconditional_conditioning, c])
    e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
    e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
```
In ldm_for_eeg.generate(), `unconditional_conditioning` is not passed → effectively 1.0 (no CFG).

---

**Document path:** `docs/chat.md`
