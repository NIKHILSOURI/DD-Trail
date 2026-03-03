## FULL preset (full train + fast eval, no training-time images)

- **Stage B:** Full dataset (no item limits), 350 epochs, no image gen in val, no post-train gen.  
- **Stage C:** Same fast eval as above (20 items, 1 sample, 250 steps).

### 1) Baseline Stage B (FULL)

```bash
cd /workspace/DreamDiffusion_SAR-HM && PYTHONPATH=code python code/eeg_ldm.py \
  --pretrain_mbm_path pretrains/eeg_pretain/checkpoint.pth \
  --use_sarhm false \
  --model baseline \
  --imagenet_path datasets/imageNet_images \
  --num_epoch 350 \
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
  --num_epoch 350 \
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
  --model_path exps/results/generation/02-03-2026-01-49-36/checkpoint_best.pth \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --imagenet_path datasets/imageNet_images \
  --max_test_items 100 \
  --num_samples 2 \
  --ddim_steps 250 \
  --seed 2022
```

### 4) SAR-HM Stage C (FAST eval)

```bash
cd /workspace/DreamDiffusion_SAR-HM && PYTHONPATH=code python code/gen_eval_eeg.py \
  --model_path exps/results/generation/02-03-2026-09-57-39/checkpoint_best.pth \
  --proto_path exps/results/generation/02-03-2026-09-57-39/prototypes.pt \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --imagenet_path datasets/imageNet_images \
  --max_test_items 100 \
  --num_samples 2 \
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
  --baseline_ckpt exps/results/generation/02-03-2026-01-49-36/checkpoint_best.pth \
  --sarhm_ckpt exps/results/generation/02-03-2026-09-57-39/checkpoint_best.pth \
  --sarhm_proto exps/results/generation/02-03-2026-09-57-39/prototypes.pt \
  --n_samples 5 \
  --ddim_steps 250 \
  --seed 2022 \
  --out_dir results/compare_eval_thesis
```

---

For the baseline run (02-03-2026-18-20-51):
```bash
cd /workspace/DreamDiffusion_SAR-HM && PYTHONPATH=code python code/compute_metrics_from_images.py \    --eval_dir /workspace/dreamdiffusion/results/eval/02-03-2026-18-20-51 \    --num_samples 2
```

For the SAR-HM run (02-03-2026-21-33-14) once it finishes:
```bash
cd /workspace/DreamDiffusion_SAR-HM && PYTHONPATH=code python code/compute_metrics_from_images.py --eval_dir /workspace/dreamdiffusion/results/eval/02-03-2026-21-33-14 --num_samples 2
```