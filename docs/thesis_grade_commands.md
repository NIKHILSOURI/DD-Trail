
### Thesis-level vs 10-epoch testing (Stage B)

| | **Thesis-level (reported results)** | **10-epoch testing (sanity / comparison)** |
|--|--------------------------------------|--------------------------------------------|
| **Epochs** | **500** | **10** |
| **Purpose** | Numbers and figures for thesis; fair Baseline vs SAR-HM comparison. | Quick pipeline check; fast Baseline vs SAR-HM comparison. |
| **Batch size** | **32–100** on A100 80GB (use 100 if it fits; if OOM, try 48 or 32). Default in config is 25. | Same as thesis-level (e.g. 100 or 32). |
| **Stage C** | Run Stage C with full `ddim_steps=250`, `num_samples=5` for final metrics. | Same; Stage C defines reported metrics. |

- **Batch size 100:** On A100 80GB you can try `--batch_size 100`; it often fits and shortens wall time. If you get CUDA OOM, use `--batch_size 48` or `--batch_size 32`.
- **Thesis results:** Use **500 epochs** for any result you report. Use **10 epochs** only for debugging or to show “early training” comparison in the thesis if you want.

### Stage B: why it was slow and how it’s fixed (same thesis quality)

**Why one epoch took ~37 min on A100:** Validation during Stage B runs **image generation** (PLMS/DDIM) to compute metrics. With default settings this was: **250 steps × 3–5 samples × 5 items** every 2 epochs, which can take 20–30+ minutes per validation. So most of the “epoch” time was validation, not training.

**What we changed (no impact on thesis results):**
- **Validation-only** settings: `val_gen_limit=2`, `val_ddim_steps=50`, `val_num_samples=2`, `check_val_every_n_epoch=5`. These apply only to **training-time validation**. Your **final thesis numbers** come from **Stage C**, which still uses full `ddim_steps=250` and `num_samples=5`.
- So: Stage B runs much faster per epoch; Stage C and any reported metrics stay at full quality.

**Override from CLI if needed:** `--val_gen_limit`, `--val_ddim_steps`, `--val_num_samples`, `--check_val_every_n_epoch`.

---


## 2. Copy Stage A1 checkpoint for Stage B

**Windows:**
```cmd
copy results\eeg_pretrain\<timestamp>\checkpoints\checkpoint.pth pretrains\eeg_pretain\checkpoint.pth
```

**Linux/Mac:**
```bash
cp results/eeg_pretrain/<timestamp>/checkpoints/checkpoint.pth pretrains/eeg_pretain/checkpoint.pth
```

**Explanation:** Stage B expects the pretrained EEG encoder at `pretrains/eeg_pretain/checkpoint.pth`. Use the `<timestamp>` from the Stage A1 run.

---

## 3. Stage B: Fine-tune LDM (DreamDiffusion)

Stage B trains the EEG-conditioned LDM. For **thesis-grade** runs you want:
- **Logging:** `--model baseline` or `--model sarhm` (and optionally `--run_name <name>`) so that `runs/<timestamp>_<model>_<seed>/` is created with `config.json`, `train_log.csv`, `eval_log.csv`, and `artifacts/`.
- **Reproducibility:** Fixed `--seed` (e.g. `2022`).
- **Evaluation:** `--eval_every 2` and `--num_eval_samples 50` (or your choice) so periodic eval metrics are logged.

### 3a. Baseline (no SAR-HM)

**Thesis-level (500 epochs, batch 100 on A100 80GB):**
```bash
python code/eeg_ldm.py --num_epoch 500 --batch_size 100 --num_workers 8 --precision bf16 --model baseline --seed 2022 --eval_every 2 --num_eval_samples 50 --use_sarhm false
```
*(If OOM, use `--batch_size 48` or `--batch_size 32`.)*

**Thesis-level (500 epochs, default batch 25):**
```bash
python code/eeg_ldm.py --num_epoch 500 --model baseline --seed 2022 --eval_every 2 --num_eval_samples 50 --use_sarhm false
```

**10-epoch testing (batch 100 on A100 80GB):**
```bash
python code/eeg_ldm.py --num_epoch 10 --batch_size 100 --num_workers 8 --precision bf16 --model baseline --seed 2022 --eval_every 2 --num_eval_samples 50 --use_sarhm false
```

**Explanation:** Trains the original DreamDiffusion path (EEG → MAE → mapper → Stable Diffusion). No Hopfield/SAR-HM. Checkpoint: `results/generation/<timestamp>/checkpoint.pth`. If `--model baseline` is set, thesis logs go to `runs/<timestamp>_baseline_<seed>/`. Use **500 epochs** for thesis-reported results; **10 epochs** only for quick testing.

### 3b. SAR-HM (full_sarhm)

**Thesis-level (500 epochs, batch 100 on A100 80GB):**
```bash
python code/eeg_ldm.py --num_epoch 500 --batch_size 100 --num_workers 8 --precision bf16 --model sarhm --seed 2022 --eval_every 2 --num_eval_samples 50 --use_sarhm true --ablation_mode full_sarhm
```
*(If OOM, use `--batch_size 48` or `--batch_size 32`.)*

**Thesis-level (500 epochs, default batch):**
```bash
python code/eeg_ldm.py --num_epoch 500 --model sarhm --seed 2022 --eval_every 2 --num_eval_samples 50 --use_sarhm true --ablation_mode full_sarhm
```

**10-epoch testing (batch 100 on A100 80GB):**
```bash
python code/eeg_ldm.py --num_epoch 10 --batch_size 100 --num_workers 8 --precision bf16 --model sarhm --seed 2022 --eval_every 2 --num_eval_samples 50 --use_sarhm true --ablation_mode full_sarhm
```

**Explanation:** Trains with SAR-HM (Hopfield + confidence-gated fusion). Same outputs as baseline but with SAR-HM metrics in logs and run folder `runs/<timestamp>_sarhm_<seed>/`. Use the same `--seed` as baseline for fair comparison. Use **500 epochs** for thesis-reported results; **10 epochs** for quick testing.

### 3c. A100 80GB: optional faster settings

Defaults already use fast validation (`val_gen_limit=2`, `val_ddim_steps=50`, `check_val_every_n_epoch=5`). For even faster runs you can use **batch_size 100** (see 3a/3b); if OOM, use 48 or 32. No need to pass `--check_val_every_n_epoch` or `--val_gen_limit` unless you want to override.

### 3d. Tiny dataset (quick baseline vs SAR-HM comparison)

**Create tiny splits once:**
```bash
python code/make_tiny_splits.py
```
Optional: `--train 200 --test 50` to override default train/test sizes.

**Baseline, 10 epochs, tiny:**
```bash
python code/eeg_ldm.py --num_epoch 10 --splits_path datasets/block_splits_tiny.pth --val_gen_limit 2 --model baseline --seed 2022 --use_sarhm false
```

**SAR-HM, 10 epochs, tiny:**
```bash
python code/eeg_ldm.py --num_epoch 10 --splits_path datasets/block_splits_tiny.pth --val_gen_limit 2 --model sarhm --seed 2022 --use_sarhm true --ablation_mode full_sarhm
```

**Explanation:** Same pipeline on a small subset for fast comparison. Use the corresponding Stage B timestamp in Stage C with `--splits_path datasets/block_splits_tiny.pth`.

---

## 4. Stage C: Generate and evaluate

Stage C loads a Stage B checkpoint, generates images on the test set, and computes metrics. Use the **same splits and dataset** as in Stage B.

### 4a. Full dataset (single split)

**Baseline checkpoint:**
```bash
python code/gen_eval_eeg.py --dataset EEG --model_path results/generation/<timestamp_baseline>/checkpoint.pth --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
```

**SAR-HM checkpoint:**
```bash
python code/gen_eval_eeg.py --dataset EEG --model_path results/generation/<timestamp_sarhm>/checkpoint.pth --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
```

**Explanation:** Generates images from EEG test set and evaluates (e.g. SSIM, PCC, FID/IS/CLIP if wired in). Checkpoint contains the config (including SAR-HM), so no extra flags needed for SAR-HM. Outputs: e.g. `results/eval/<timestamp>/`, samples and metrics.

### 4b. Tiny dataset

**Baseline (trained with tiny splits):**
```bash
python code/gen_eval_eeg.py --dataset EEG --model_path results/generation/<timestamp_baseline>/checkpoint.pth --splits_path datasets/block_splits_tiny.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
```

**SAR-HM (trained with tiny splits):**
```bash
python code/gen_eval_eeg.py --dataset EEG --model_path results/generation/<timestamp_sarhm>/checkpoint.pth --splits_path datasets/block_splits_tiny.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
```

**Explanation:** Same as 4a but with `block_splits_tiny.pth` so test set matches the tiny-split training.

### 4c. All-image splits (if you trained with `block_splits_by_image_all.pth`)

Replace `block_splits_by_image_single.pth` with `block_splits_by_image_all.pth` in the Stage C command; keep everything else the same.

---

## Summary table (thesis-grade)

| Step | Purpose | Command (conceptual) |
|------|---------|----------------------|
| 0 | One-time setup | `pip install ...` + `pip install -e ./code` |
| 1 | EEG encoder pretrain | `python code/stageA1_eeg_pretrain.py` (add `--num_epoch 10` for quick) |
| 2 | Copy A1 → pretrains | `copy/cp ... checkpoint.pth pretrains/eeg_pretain/checkpoint.pth` |
| 3a | Stage B Baseline | **Thesis-level:** `--num_epoch 500 --batch_size 100 ... --model baseline --use_sarhm false`. **10-epoch testing:** `--num_epoch 10 --batch_size 100 ...` |
| 3b | Stage B SAR-HM | **Thesis-level:** `--num_epoch 500 --batch_size 100 ... --model sarhm --use_sarhm true --ablation_mode full_sarhm`. **10-epoch testing:** `--num_epoch 10 --batch_size 100 ...` |
| 4 | Stage C (generate + eval) | `python code/gen_eval_eeg.py --dataset EEG --model_path results/generation/<timestamp>/checkpoint.pth --splits_path ... --eeg_signals_path ... --config_patch ...` |

**Epochs for thesis:** Use **500** for any result you report. Use **10** only for quick testing or “early training” comparison.  
**Batch size:** **100** on A100 80GB is fine if it fits; else use 48 or 32.  
**Reproducibility:** Same `--seed`, same splits, same dataset for Baseline vs SAR-HM. For evaluation-only from a saved checkpoint, see `code/eval/evaluate.py` and `docs/logging.md` (if present).
