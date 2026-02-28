### Make Stage B faster (reduce time per epoch)

If Stage B is still **~40 min per epoch**, most of that is usually **validation** (PLMS generation) when it runs, plus **many training steps** if batch size is small. You can speed both up without changing final thesis quality (Stage C still uses 250 steps, 5 samples).

| Lever | Effect | What to do |
|-------|--------|------------|
| **Larger batch size** | Fewer steps per epoch → faster training | Use the largest `--batch_size` that fits (e.g. 32, 48, 100 on A100). |
| **Validate less often** | Validation runs less often → most epochs are training-only | `--check_val_every_n_epoch 10` or `20`. |
| **Lighter validation (PLMS)** | Shorter validation when it runs | `--val_gen_limit 1`, `--val_ddim_steps 25`, `--val_num_samples 1`. |

**Learning rate:** If you **increase batch size**, you can scale LR to keep training stable (e.g. linear scaling):  
`--lr 6.8e-5` for batch 32, or `--lr 2.1e-4` for batch 100 (≈ 5.3e-5 × batch_size/25). Try default `5.3e-5` first; increase slightly if loss is too smooth/slow to converge.

**Example – much faster per epoch (A100 80GB):**
```bash
# Baseline: large batch, validate every 10 epochs, minimal validation PLMS
python code/eeg_ldm.py --num_epoch 200 --batch_size 48 --num_workers 8 --precision bf16 --check_val_every_n_epoch 10 --val_gen_limit 1 --val_ddim_steps 25 --val_num_samples 1 --model baseline --seed 2022 --use_sarhm false

# SAR-HM: same
python code/eeg_ldm.py --num_epoch 200 --batch_size 48 --num_workers 8 --precision bf16 --check_val_every_n_epoch 10 --val_gen_limit 1 --val_ddim_steps 25 --val_num_samples 1 --model sarhm --seed 2022 --use_sarhm true --ablation_mode full_sarhm
```

If 48 fits, try `--batch_size 100` for even fewer steps per epoch. Thesis metrics still come from **Stage C** with full 250 steps and 5 samples.

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

### 3c. Very fast 10-epoch run (max speed, good quality)

Use these for **10 epochs only**, fastest wall time, while still getting a usable checkpoint for **Stage C with full quality** (Stage C uses 250 steps, 5 samples from the saved config).

**Baseline (10 epochs, very fast):**
```bash
python code/eeg_ldm.py --num_epoch 10 --batch_size 100 --num_workers 8 --precision bf16 --check_val_every_n_epoch 10 --model baseline --seed 2022 --eval_every 2 --num_eval_samples 50 --use_sarhm false
```

**SAR-HM (10 epochs, very fast):**
```bash
python code/eeg_ldm.py --num_epoch 10 --batch_size 100 --num_workers 8 --precision bf16 --check_val_every_n_epoch 10 --model sarhm --seed 2022 --eval_every 2 --num_eval_samples 50 --use_sarhm true --ablation_mode full_sarhm
```

- **Speed:** `batch_size 100`, `num_workers 8`, `precision bf16`, `check_val_every_n_epoch 10` (validation runs only once in 10 epochs). If OOM, use `--batch_size 48` or `--batch_size 32`.
- **Good quality:** After training, run **Stage C** with the checkpoint; it uses **250 PLMS steps** and **5 samples** per EEG from the saved config, so final images and metrics are full quality.

