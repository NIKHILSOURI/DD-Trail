# Stage B: Fast training (no image generation in validation)

Commands for **Baseline** and **SAR-HM** with **200 epochs**, **no image generation during validation** (faster, cheaper), on **A100 80GB**. Final quality is preserved: **Stage C** still uses full 250 steps and 5 samples.

---

## Prerequisites

- Stage A1 done; `pretrains/eeg_pretain/checkpoint.pth` in place.
- Data: `datasets/` with `eeg_5_95_std.pth`, `block_splits_by_image_single.pth` (or your splits).
- Run from repo root with environment activated.

---

## Baseline (200 epochs, no val generation)

```bash
python code/eeg_ldm.py \
  --num_epoch 200 \
  --batch_size 48 \
  --num_workers 8 \
  --precision bf16 \
  --disable_image_generation_in_val true \
  --check_val_every_n_epoch 10 \
  --model baseline \
  --seed 2022 \
  --use_sarhm false
```

- **48 batch:** Fits A100 80GB; try `--batch_size 100` for even fewer steps per epoch.
- **No image generation:** `--disable_image_generation_in_val true` so validation runs without PLMS/DDIM.
- **Stage C:** Run after training with the same checkpoint; it uses 250 steps, 5 samples (full quality).

---

## SAR-HM (200 epochs, no val generation)

```bash
python code/eeg_ldm.py \
  --num_epoch 200 \
  --batch_size 48 \
  --num_workers 8 \
  --precision bf16 \
  --disable_image_generation_in_val true \
  --check_val_every_n_epoch 10 \
  --model sarhm \
  --seed 2022 \
  --use_sarhm true \
  --ablation_mode full_sarhm
```

- Same speed settings as baseline; SAR-HM adds Hopfield + gated fusion.
- Use the same `--seed` as baseline for fair comparison.

---

## One-liners (copy-paste)

**Baseline:**
```bash
python code/eeg_ldm.py --num_epoch 200 --batch_size 48 --num_workers 8 --precision bf16 --disable_image_generation_in_val true --check_val_every_n_epoch 10 --model baseline --seed 2022 --use_sarhm false
```

**SAR-HM:**
```bash
python code/eeg_ldm.py --num_epoch 200 --batch_size 48 --num_workers 8 --precision bf16 --disable_image_generation_in_val true --check_val_every_n_epoch 10 --model sarhm --seed 2022 --use_sarhm true --ablation_mode full_sarhm
```

---

## After Stage B: Stage C (full quality)

Use the checkpoint from `results/generation/<timestamp>/checkpoint.pth` with **full-quality** generation (250 steps, 5 samples). Same command for both models; replace `<timestamp>` with the run folder.

```bash
python code/gen_eval_eeg.py --dataset EEG --model_path results/generation/<timestamp>/checkpoint.pth --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
```

---

## Optional: larger batch on A100 80GB

If VRAM allows, use `--batch_size 100` in both commands for fewer steps per epoch and faster wall time. If you see OOM, use 48 or 32.

---

## Summary

| Setting              | Value        | Purpose                          |
|----------------------|-------------|-----------------------------------|
| Epochs               | 200         | Shorter run, still good quality   |
| Batch size           | 48 (or 100) | Fast training on A100 80GB       |
| No val image gen     | `true`      | Faster, cheaper Stage B          |
| Val every N epochs   | 10          | Fewer val runs                   |
| Stage C              | Unchanged   | 250 steps, 5 samples = best quality |
