## FULL preset (full train + fast eval, no training-time images)

- **Stage B:** Full dataset (no item limits), more epochs (e.g. 100 or 500), no image gen in val, no post-train gen.  
- **Stage C:** Same fast eval as above (20 items, 1 sample, 250 steps).

### 1) Baseline Stage B (FULL)

```bash
cd /workspace/DreamDiffusion_SAR-HM && PYTHONPATH=code python code/eeg_ldm.py \
  --pretrain_mbm_path pretrains/eeg_pretain/checkpoint.pth \
  --use_sarhm false \
  --model baseline \
  --imagenet_path datasets/imageNet_images \
  --num_epoch 100 \
  --disable_image_generation_in_val true \
  --skip_post_train_generation true \
  --seed 2022 \
  --batch_size 48 \
  --num_workers 8 \
  --precision bf16
```

**→ Note run folder → <BASELINE_TIMESTAMP>.**

### 2) SAR-HM Stage B (FULL)

```bash
cd /workspace/DreamDiffusion_SAR-HM && PYTHONPATH=code python code/eeg_ldm.py \
  --pretrain_mbm_path pretrains/eeg_pretain/checkpoint.pth \
  --use_sarhm true \
  --model sarhm \
  --proto_source baseline_centroids \
  --imagenet_path datasets/imageNet_images \
  --num_epoch 100 \
  --disable_image_generation_in_val true \
  --skip_post_train_generation true \
  --seed 2022 \
  --batch_size 48 \
  --num_workers 8 \
  --precision bf16
```

**→ Note run folder → <SARHM_TIMESTAMP>. Ensure `prototypes.pt` is in that folder.**

### 3) Baseline Stage C (FAST eval)

```bash
cd /workspace/DreamDiffusion_SAR-HM && PYTHONPATH=code python code/gen_eval_eeg.py \
  --model_path exps/results/generation/<BASELINE_TIMESTAMP>/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --imagenet_path datasets/imageNet_images \
  --max_test_items 20 \
  --num_samples 1 \
  --ddim_steps 250 \
  --seed 2022
```

### 4) SAR-HM Stage C (FAST eval)

```bash
cd /workspace/DreamDiffusion_SAR-HM && PYTHONPATH=code python code/gen_eval_eeg.py \
  --model_path exps/results/generation/<SARHM_TIMESTAMP>/checkpoint.pth \
  --proto_path exps/results/generation/<SARHM_TIMESTAMP>/prototypes.pt \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --imagenet_path datasets/imageNet_images \
  --max_test_items 20 \
  --num_samples 1 \
  --ddim_steps 250 \
  --seed 2022
```

### 5) Compare (same as FAST block)

```bash
cd /workspace/DreamDiffusion_SAR-HM && PYTHONPATH=code python code/compare_eval.py \
  --dataset EEG \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --imagenet_path datasets/imageNet_images \
  --baseline_ckpt exps/results/generation/<BASELINE_TIMESTAMP>/checkpoint.pth \
  --sarhm_ckpt exps/results/generation/<SARHM_TIMESTAMP>/checkpoint.pth \
  --sarhm_proto exps/results/generation/<SARHM_TIMESTAMP>/prototypes.pt \
  --n_samples 5 \
  --ddim_steps 250 \
  --seed 2022 \
  --out_dir results/compare_eval
```

---

## How to decide if enough epochs (manual; no code changes)

1. **Run Stage C** with the same setup every time: `--ddim_steps 250`, `--seed 2022`, fixed test size (e.g. `--max_test_items 20`, `--num_samples 1`), and `--imagenet_path datasets/imageNet_images` so GT is real.
2. **Compare metrics** (SSIM, PCC, CLIP similarity) between runs at different epoch counts (e.g. 200 vs 300). If they plateau, extra epochs add little.
3. **Inspect generated images** at 100, 200, 300 (and optionally 400, 500). Early epochs are noisy; by 100–200 images become structured; further epochs refine. If 200 and 300 look similar, you can stop.
4. **Checkpoint strategy:** Run Stage B with different `--num_epoch` (e.g. 200, 300, 500) and run Stage C once per run; or use intermediate checkpoints if your setup saves them. No early-stopping logic required.

Full checklist and convergence reasoning: **docs/EPOCH_RECOMMENDATION.md**.

---

```bash
ls -t /workspace/DreamDiffusion_SAR-HM/exps/results/generation/
```

Top line = most recent run. Use that directory name for the matching Stage C and compare commands.

---

## Notes

- **Stage B:** No validation image generation (`--disable_image_generation_in_val true`), no post-train generation (`--skip_post_train_generation true`). All image generation is in Stage C.
- **SAR-HM:** Prototypes are built/saved during Stage B into the run folder as `prototypes.pt`. Stage C and compare must pass `--proto_path` / `--sarhm_proto` to that file; otherwise SAR-HM may fall back to dummy or fail (gen_eval_eeg asserts when proto_path is given but load fails).
- **Compare:** If you want to force failure when proto is missing for SAR-HM, add **`--fail_if_proto_missing`** to the compare command.
- **Epochs:** Config default is 500 (safe upper bound). Comparable final quality is often reached in **200–350 epochs**; 500 is not compulsory. See **docs/EPOCH_RECOMMENDATION.md** for a recommended range, convergence reasoning, and **How to decide if enough epochs** (Stage C @250 steps, same seed and test subset, metrics + qualitative images).
