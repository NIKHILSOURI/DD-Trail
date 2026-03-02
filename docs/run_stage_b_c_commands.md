# Stage B retrain + Stage C real check (real ImageNet GT)

All commands from repo root: `/workspace/DreamDiffusion_SAR-HM`

Pretrain MAE path in this repo: `pretrains/eeg_pretain/checkpoint.pth`

---

## Step 1 — Retrain Stage B baseline (10 epochs, real images)

```bash
cd /workspace/DreamDiffusion_SAR-HM && PYTHONPATH=code python code/eeg_ldm.py \
  --pretrain_mbm_path pretrains/eeg_pretain/checkpoint.pth \
  --use_sarhm false \
  --model baseline \
  --imagenet_path datasets/imageNet_images \
  --num_epoch 10 \
  --limit_train_items 400 \
  --limit_val_items 50 \
  --disable_image_generation_in_val true \
  --skip_post_train_generation true \
  --seed 2022 \
  --batch_size 48 \
  --num_workers 8 \
  --precision bf16
```

When it finishes, note the output directory, e.g.:
`exps/results/generation/01-03-2026-XX-XX-XX`
Use that timestamp as **BASELINE_TS** in Step 3.

---

## Step 2 — Retrain Stage B SAR-HM (10 epochs, real images)

```bash
cd /workspace/DreamDiffusion_SAR-HM && PYTHONPATH=code python code/eeg_ldm.py \
  --pretrain_mbm_path pretrains/eeg_pretain/checkpoint.pth \
  --use_sarhm true \
  --model sarhm \
  --proto_source baseline_centroids \
  --imagenet_path datasets/imageNet_images \
  --num_epoch 10 \
  --limit_train_items 400 \
  --limit_val_items 50 \
  --disable_image_generation_in_val true \
  --skip_post_train_generation true \
  --seed 2022 \
  --batch_size 48 \
  --num_workers 8 \
  --precision bf16
```

When it finishes, note the output directory, e.g.:
`exps/results/generation/01-03-2026-XX-XX-XX`
Use that timestamp as **SARHM_TS** in Step 3.

---

## Step 3a — Stage C real check: Baseline (250 steps)

Replace **BASELINE_TS** with the timestamp from Step 1 (e.g. `01-03-2026-23-45-00`).

```bash
cd /workspace/DreamDiffusion_SAR-HM && PYTHONPATH=code python code/gen_eval_eeg.py \
  --model_path exps/results/generation/BASELINE_TS/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --imagenet_path datasets/imageNet_images \
  --max_test_items 20 \
  --ddim_steps 250 \
  --num_samples 1 \
  --seed 2022
```

---

## Step 3b — Stage C real check: SAR-HM (250 steps)

Replace **SARHM_TS** with the timestamp from Step 2 (e.g. `01-03-2026-23-50-00`).

```bash
cd /workspace/DreamDiffusion_SAR-HM && PYTHONPATH=code python code/gen_eval_eeg.py \
  --model_path exps/results/generation/01-03-2026-23-55-15/checkpoint.pth \
  --proto_path exps/results/generation/01-03-2026-23-55-15/prototypes.pt \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --imagenet_path datasets/imageNet_images \
  --max_test_items 20 \
  --ddim_steps 250 \
  --num_samples 1 \
  --seed 2022
```

---

## Optional — Stage B with validation images (smoke test)

After retrain, if you want val images every 5 epochs, add to either Step 1 or Step 2:

```text
--disable_image_generation_in_val false \
--check_val_every_n_epoch 5 \
--val_ddim_steps 50 \
--val_gen_limit 2 \
--val_num_samples 2
```

Quality check is still Stage C with 250 steps (Step 3).

---

## Compare baseline vs SAR-HM (fewer images)

Same dataset/seed, side-by-side grids and metrics. Use **BASELINE_TS** (Step 1) and **SARHM_TS** (Step 2).

**Fewer images (e.g. 5 samples, 250 steps):**

```bash
cd /workspace/DreamDiffusion_SAR-HM && PYTHONPATH=code python code/compare_eval.py \
  --dataset EEG \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --imagenet_path datasets/imageNet_images \
  --baseline_ckpt exps/results/generation/01-03-2026-23-44-52/checkpoint.pth \
  --sarhm_ckpt exps/results/generation/01-03-2026-23-55-15/checkpoint.pth \
  --sarhm_proto exps/results/generation/01-03-2026-23-55-15/prototypes.pt \
  --n_samples 5 \
  --ddim_steps 250 \
  --seed 2022 \
  --out_dir results/compare_eval
```

Replace **BASELINE_TS** / **SARHM_TS** with your run timestamps (e.g. `01-03-2026-23-44-52` and `01-03-2026-23-55-15`). Use a larger `--n_samples` (e.g. 10 or 20) for more images.



cd /workspace/DreamDiffusion_SAR-HM && PYTHONPATH=code python code/gen_eval_eeg.py \
  --model_path exps/results/generation/02-03-2026-01-13-52/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --imagenet_path datasets/imageNet_images \
  --max_test_items 5 \
  --ddim_steps 250 \
  --num_samples 1 \
  --seed 2022