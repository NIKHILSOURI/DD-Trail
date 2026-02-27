# Training Commands: Baseline and SAR-HM

Run all commands from the **repository root** with your venv activated.  
Replace `<timestamp>` with the folder name under `results/generation/` after Stage B.

---

## 1. Fast 10-epoch runs (thesis quality)

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

## 2. Complete / full training: how many epochs?

**500 epochs** is the **default** in config (full reproduction, original setup). It is **not mandatory** for good thesis results.

| Epochs | Use case | Typical quality / time |
|--------|----------|--------------------------|
| **10** | Fast comparison, ablations, debugging | Good for tables/figures; may underfit. |
| **50–100** | Strong thesis results, reasonable time | Often enough for good FID/CLIP; recommended starting point. |
| **150–200** | High quality, full convergence on many setups | Safe choice for main thesis numbers. |
| **500** | Full reproduction, paper/default | Best convergence; long runtime. |

**Recommendation:** Use **100–150 epochs** for main thesis results unless you need exact reproduction of a 500-epoch run. You can always extend later from a checkpoint.

---

## 3. Complete training commands (Baseline and SAR-HM)

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

## 4. Stage C: generate and evaluate (after Stage B)

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

## 5. GPU and speed notes

- **16–24 GB GPU:** Use `--batch_size 4` (optionally `--accumulate_grad 2`).
- **80 GB GPU:** Try `--batch_size 12` or `16` and `--num_workers 8`.
- **OOM during Stage B:** Reduce `--batch_size` (e.g. 4 or 2).
- Thesis metrics (FID, IS, CLIP, etc.) come from **Stage C** (250 steps, 5 samples). Stage B validation is kept light on purpose.
