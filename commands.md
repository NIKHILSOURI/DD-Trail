# Training Commands: Baseline and SAR-HM

Run all commands from the **repository root** with your venv activated.  
Replace `<timestamp>` with the folder name under `results/generation/` after Stage B.

---

## 1. Stage B with image generation (quick check: 2 epochs, 3–5 images)

**Use case:** Run Stage B with SAR-HM for **2 epochs** and generate **3–5 images** during validation to check that images look correct (sanity check). Validation runs every epoch and generates images for **1 item** with **3 samples** (1 GT + 3 generated = 4 images per validation), saved under `exps/results/generation/<timestamp>/val/`.

```bash
python code/eeg_ldm.py --num_epoch 2 \
  --use_sarhm true --ablation_mode full_sarhm \
  --disable_image_generation_in_val false \
  --val_image_gen_every_n_epoch 1 \
  --val_gen_limit 1 \
  --val_num_samples 3 \
  --val_ddim_steps 50 \
  --check_val_every_n_epoch 1 \
  --batch_size 4
```

- **5 images per val** (1 GT + 4 generated): use `--val_num_samples 4`.
- **Post-training:** After the 2 epochs, the script also runs `generate_images()` (train + test) unless you set `--disable_image_generation_in_val true`; for a minimal run you can leave it as above to get a few train/test images too.

One-line (2 epochs, SAR-HM, 3 samples per item in val):
```bash
python code/eeg_ldm.py --num_epoch 2 --use_sarhm true --ablation_mode full_sarhm --disable_image_generation_in_val false --val_image_gen_every_n_epoch 1 --val_gen_limit 1 --val_num_samples 3 --val_ddim_steps 50 --check_val_every_n_epoch 1 --batch_size 4
```

---

## 2. Fast 10-epoch runs (thesis quality)

**Use case:** Quick thesis-grade comparison, sanity checks, ablations.  
**Quality:** Stage C uses 250 steps, 5 samples for final metrics.

### Baseline (10 epochs)

```bash
python code/eeg_ldm.py --num_epoch 10 \
  --model baseline --seed 2022 --run_name baseline_10ep \
  --eval_every 2 --num_eval_samples 50 \
  --check_val_every_n_epoch 5 --val_gen_limit 2 --val_ddim_steps 50 \
  --use_sarhm false \
  --num_workers 4 --batch_size 6
```

One-line:
```bash
python code/eeg_ldm.py --num_epoch 10 --model baseline --seed 2022 --run_name baseline_10ep --eval_every 2 --num_eval_samples 50 --check_val_every_n_epoch 5 --val_gen_limit 2 --val_ddim_steps 50 --use_sarhm false --num_workers 4 --batch_size 6
```

### SAR-HM (10 epochs)

```bash
python code/eeg_ldm.py --num_epoch 10 \
  --model sarhm --seed 2022 --run_name sarhm_10ep \
  --eval_every 2 --num_eval_samples 50 \
  --check_val_every_n_epoch 5 --val_gen_limit 2 --val_ddim_steps 50 \
  --use_sarhm true --ablation_mode full_sarhm \
  --num_workers 4 --batch_size 6
```

One-line:
```bash
python code/eeg_ldm.py --num_epoch 10 --model sarhm --seed 2022 --run_name sarhm_10ep --eval_every 2 --num_eval_samples 50 --check_val_every_n_epoch 5 --val_gen_limit 2 --val_ddim_steps 50 --use_sarhm true --ablation_mode full_sarhm --num_workers 4 --batch_size 6
```

---

## 3. Complete / full training: how many epochs?

**500 epochs** is the **config default** (full reproduction, original setup). It is a **safe upper bound** for best convergence but **not mandatory** for good thesis results. Comparable quality is often reached earlier (e.g. 200–350 epochs); use Stage C eval at fixed checkpoints to decide.

| Epochs | Use case | Typical quality / time |
|--------|----------|--------------------------|
| **10** | Fast comparison, ablations, debugging | Good for tables/figures; may underfit. |
| **50–100** | Strong thesis results, reasonable time | Often enough for good CLIP/SSIM; recommended starting point. |
| **150–200** | High quality, convergence on many setups | Safe choice for main thesis numbers. |
| **200–350** | Typical range where metrics plateau | Compare Stage C at 200 vs 300; if similar, stop. |
| **500** | Full reproduction, paper default | Safe upper bound; run if you need best possible convergence. |

**Recommendation:** Use **200–300 epochs** as a default target; run Stage C at 250 steps on a fixed test subset (same seed, e.g. `--max_test_items 20`, `--num_samples 1`). If SSIM/CLIP and qualitative images plateau, that is sufficient. Extend to 400–500 only if you need to squeeze out the last bit of quality. You can always extend later from a checkpoint.

**Evaluation checkpoints:** To decide manually when to stop (no early-stopping code): run Stage C on checkpoints at different epoch counts (e.g. 100, 200, 300, 400, 500) using the same seed and test subset; compare metrics (SSIM, PCC, CLIP) and qualitative images. See **docs/EPOCH_RECOMMENDATION.md** for a “How to decide if enough epochs” checklist.

---

## 4. Complete training commands (Baseline and SAR-HM)

Use the same structure as above; only `--num_epoch` and `--run_name` change. Examples below use **100 epochs** as a good balance; replace with `200` or `500` if you want longer training.

### Baseline (complete training, 100 epochs)

```bash
python code/eeg_ldm.py --num_epoch 100 \
  --model baseline --seed 2022 --run_name baseline_100ep \
  --eval_every 2 --num_eval_samples 50 \
  --check_val_every_n_epoch 10 --val_gen_limit 2 --val_ddim_steps 50 \
  --use_sarhm false \
  --num_workers 4 --batch_size 6
```

For **200 epochs**:
```bash
python code/eeg_ldm.py --num_epoch 200 --model baseline --seed 2022 --run_name baseline_200ep --eval_every 2 --num_eval_samples 50 --check_val_every_n_epoch 10 --val_gen_limit 2 --val_ddim_steps 50 --use_sarhm false --num_workers 4 --batch_size 6
```

For **500 epochs** (full default):
```bash
python code/eeg_ldm.py --num_epoch 500 --model baseline --seed 2022 --run_name baseline_500ep --eval_every 2 --num_eval_samples 50 --check_val_every_n_epoch 10 --val_gen_limit 2 --val_ddim_steps 50 --use_sarhm false --num_workers 4 --batch_size 6
```

### SAR-HM (complete training, 100 epochs)

```bash
python code/eeg_ldm.py --num_epoch 100 \
  --model sarhm --seed 2022 --run_name sarhm_100ep \
  --eval_every 2 --num_eval_samples 50 \
  --check_val_every_n_epoch 10 --val_gen_limit 2 --val_ddim_steps 50 \
  --use_sarhm true --ablation_mode full_sarhm \
  --num_workers 4 --batch_size 6
```

For **200 epochs**:
```bash
python code/eeg_ldm.py --num_epoch 200 --model sarhm --seed 2022 --run_name sarhm_200ep --eval_every 2 --num_eval_samples 50 --check_val_every_n_epoch 10 --val_gen_limit 2 --val_ddim_steps 50 --use_sarhm true --ablation_mode full_sarhm --num_workers 4 --batch_size 6
```

For **500 epochs** (full default):
```bash
python code/eeg_ldm.py --num_epoch 500 --model sarhm --seed 2022 --run_name sarhm_500ep --eval_every 2 --num_eval_samples 50 --check_val_every_n_epoch 10 --val_gen_limit 2 --val_ddim_steps 50 --use_sarhm true --ablation_mode full_sarhm --num_workers 4 --batch_size 6
```

---

## 5. Stage C: generate and evaluate (after Stage B)

Run **once per model** (baseline and SAR-HM). Uses saved config (250 steps, 5 samples) for thesis metrics.

**Baseline** (use baseline run’s `<timestamp>`):
```bash
python code/gen_eval_eeg.py --dataset EEG \
  --model_path results/generation/<timestamp>/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml
```

**SAR-HM** (use SAR-HM run’s `<timestamp>`; checkpoint stores `use_sarhm` / `ablation_mode`):
```bash
python code/gen_eval_eeg.py --dataset EEG \
  --model_path results/generation/<timestamp>/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml
```

---

## 6. GPU and speed notes

- **16–24 GB GPU:** Use `--batch_size 4` (optionally `--accumulate_grad 2`).
- **80 GB GPU:** Try `--batch_size 12` or `16` and `--num_workers 8`.
- **OOM during Stage B:** Reduce `--batch_size` (e.g. 4 or 2).
- Thesis metrics (FID, IS, CLIP, etc.) come from **Stage C** (250 steps, 5 samples). Stage B validation is kept light on purpose.
