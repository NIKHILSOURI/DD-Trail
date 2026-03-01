# Stage C inference – debug run instructions

Use these steps to run Stage C with the added diagnostics and interpret the logs.

## Exact inference command

From the **repository root**, with your environment activated:

```bash
python code/gen_eval_eeg.py --dataset EEG --model_path <PATH_TO_STAGE_B_CHECKPOINT.pth> --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
```

Example with a concrete checkpoint path:

```bash
python code/gen_eval_eeg.py --dataset EEG --model_path results/generation/28-02-2026-21-42-16/checkpoint.pth --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
```

Optional: `--imagenet_path datasets/imageNet_images`, `--root <path>`.

---

## What to look for in the logs

All diagnostic lines are tagged so you can search for `[DEBUG]`.

| Tag | What it means |
|-----|----------------|
| **`[DEBUG] [SD_CKPT]`** | Path to SD 1.5 checkpoint and whether the file exists. If `exists=False`, see the WARNING: weights may be random (pure noise) unless they were baked into the Stage B checkpoint. |
| **`[DEBUG] [CKPT_LOAD]`** | After loading the Stage B checkpoint: total missing/unexpected keys, counts by prefix (UNet, VAE, cond_stage, sarhm), and the first 20 missing and unexpected keys. Many missing keys in `first_stage_model` or `model.diffusion_model` suggest UNet/VAE not loaded correctly. |
| **`[DEBUG] [SCALE_FACTOR]`** | `scale_factor` used at inference. **Expected:** `value=0.18215 match=True`. If `match=False` or MISSING, decode scaling is wrong and images can be noise. |
| **`[DEBUG] [COND]`** | Conditioning sanity (one sample): `c_final` and, if SAR-HM, `c_base` and `alpha` (mean/std/min/max). `has_NaN_or_Inf=True` or extreme values indicate bad conditioning. |
| **`[DEBUG] [VAE_ROUNDTRIP]`** | Path to saved `vae_roundtrip.png` and a short sanity check. If you see the WARNING (near-constant, NaN or Inf), VAE/scale_factor or weights are wrong. |

---

## Outputs

- **Eval folder:** `results/eval/<dd-mm-yyyy-HH-MM-SS>/`
- **Debug artifact:** `vae_roundtrip.png` in that folder – VAE encode→decode of one real test image. If this image is garbage or noise, the VAE/scale pipeline or its weights are wrong before any sampling.

---

## Quick checklist

1. **`[SD_CKPT] exists=True`** (or accept that weights are in the .pth).
2. **`[CKPT_LOAD]`** – no large counts of missing keys for `first_stage_model` or `model.diffusion_model`.
3. **`[SCALE_FACTOR] match=True`**.
4. **`[COND]`** – no NaN/Inf; c_final/c_base in a reasonable range (e.g. mean near 0, std on the order of 0.1–1).
5. **`vae_roundtrip.png`** looks like a blurred but recognizable version of the test image, not noise or a flat field.
6. **Prototypes at Stage C:** The checkpoint already contains the **trained** prototypes (same as the run’s `prototypes.pt`). If you pass `--proto_path`, use the **run’s** `prototypes.pt` (e.g. next to `checkpoint.pth`), **not** `prototypes_baseline_centroids.pt`. Using baseline centroids at inference overwrites the trained prototypes and causes wrong Hopfield retrieval → noisy images. To use the checkpoint’s prototypes, omit `--proto_path` or set it to the same run folder’s `prototypes.pt`.

If any of these fail, fix that part before blaming the sampling loop.
