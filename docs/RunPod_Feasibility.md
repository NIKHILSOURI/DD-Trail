# Complete Training Info & Feasibility on RunPod GPU

This document gives **full training requirements**, **RunPod GPU feasibility**, and **copy-paste commands** to run the DreamDiffusion pipeline (Stage A → Stage B → Stage C) on RunPod. All commands assume a **Linux** RunPod environment.

---

## 1. Pipeline overview

| Stage | Script | What it does | Trains | Frozen |
|-------|--------|--------------|--------|--------|
| **A (A1)** | `stageA1_eeg_pretrain.py` | Pre-train EEG encoder (MAE) | Full MAE encoder + decoder | — |
| **B** | `eeg_ldm.py` | Fine-tune LDM conditioning | EEG encoder, mappers, (SAR-HM: projection, Hopfield, adapter) | VAE, UNet |
| **C** | `gen_eval_eeg.py` | Generate & evaluate | Nothing (inference only) | — |

- **Stage B** dominates total time and VRAM (frozen UNet still in memory; gradient checkpointing used by default).
- **Validation during Stage B** runs image generation (e.g. 20–250 DDIM steps per item); reduce `val_gen_limit` and `ddim_steps` to shorten runs.

---

## 2. Data and pretrains (required before training)

Place under the **repository root** (or set env vars below):

```
/datasets
  eeg_5_95_std.pth
  block_splits_by_image_single.pth   (or block_splits_tiny.pth for small runs)
  [optional] imageNet_images/

/pretrains
  models/
    config15.yaml
    v1-5-pruned.ckpt
  eeg_pretain/
    checkpoint.pth    ← from Stage A (or pre-downloaded)
  [optional] generation/checkpoint_best.pth
```

**Optional env vars** (if data/pretrains are not under repo root):

```bash
export DREAMDIFFUSION_DATA_ROOT=/path/to/datasets
export DREAMDIFFUSION_PRETRAIN_ROOT=/path/to/pretrains
```

On RunPod, put these on a **persistent volume** so you don’t re-download after each pod start.

For **dataset details** (file formats, sizes, splits) and **how much data you can run on an A100 80GB** (full dataset, batch 16–24), see **`docs/Datasets_and_A100.md`**.

---

## 3. RunPod GPU feasibility

### 3.1 VRAM and batch size (Stage A and B)

| RunPod GPU | VRAM | Stage A batch_size | Stage B batch_size | Stage B val_gen_limit | Notes |
|------------|------|--------------------|--------------------|------------------------|-------|
| RTX A5000  | 24 GB | 100 | 8–12 | 5 | Good balance |
| RTX 3090   | 24 GB | 100 | 8–12 | 5 | Good balance |
| RTX 4090   | 24 GB | 100 | 8–12 | 5 | Fast, good value |
| L4         | 24 GB | 64–100 | 6–8 | 3–5 | Slightly tighter |
| L40 / L40S | 48 GB | 100+ | 12–16 | 5 | Comfortable |
| A40 / A6000| 48 GB | 100+ | 12–16 | 5 | Comfortable |
| A100 80GB  | 80 GB | 100+ | 16–24 | 5 | Fastest full runs |
| H100 80GB  | 80 GB | 100+ | 16–24 | 5 | Fastest |

- **16 GB class** (e.g. T4, A10): Stage B with `batch_size 2–4`, `val_gen_limit 1–2`; use `--clip_tune false` if OOM. Possible but tight.
- **Gradient checkpointing** is on by default (saves VRAM, some extra compute). Keep it on for ≤24 GB.

### 3.2 Approximate runtime (single GPU)

Rough estimates; actual time depends on dataset size and exact settings.

| Run type | 24 GB (e.g. RTX 4090) | 48 GB (L40S) | 80 GB (A100) |
|----------|------------------------|--------------|--------------|
| Stage A: 10 epochs | ~15–30 min | ~10–20 min | ~8–15 min |
| Stage A: 500 epochs | ~2–4 h | ~1.5–3 h | ~1–2 h |
| Stage B: 10 epochs, val_gen_limit 2, batch 4 | ~1–2 h | ~45–90 min | ~30–60 min |
| Stage B: 10 epochs, val_gen_limit 2, batch 8 | — | ~1–1.5 h | ~25–50 min |
| Stage B: 500 epochs, val every 2, val_gen_limit 5 | ~12–24 h | ~8–16 h | ~5–10 h |
| Stage C: 20 test items, 250 steps | ~30–60 min | ~20–40 min | ~15–30 min |

### 3.3 RunPod pricing (on-demand, approximate)

| GPU | $/hr (on-demand) | 10-epoch Stage A+B (tiny data) | 500+500 full run (rough) |
|-----|-------------------|---------------------------------|---------------------------|
| RTX A5000 24GB | ~$0.16 | ~$0.5–1 | ~$4–8 |
| RTX 3090 24GB | ~$0.22 | ~$0.5–1.5 | ~$5–10 |
| RTX 4090 24GB | ~$0.34 | ~$0.5–1.5 | ~$6–12 |
| L4 24GB | ~$0.44 | ~$1–2 | ~$10–18 |
| L40S 48GB | ~$0.79 | ~$1–2 | ~$12–22 |
| A100 80GB | ~$1.19 | ~$1–2 | ~$8–16 |

**Spot instances** are cheaper but can be interrupted; use for long runs only if you save checkpoints often and can resume.

---

## 4. RunPod setup (one-time per pod)

### 4.1 Create a pod

1. RunPod Console → **Deploy** → choose **GPU** (e.g. RTX 4090 24GB or A100 80GB).
2. **Image:** e.g. **RunPod PyTorch 2.x** or any Linux image with Python 3.10/3.11 and CUDA.
3. **Container disk:** ≥ 50 GB (for repo, venv, datasets, checkpoints).
4. **Volume:** Attach a **persistent volume** (e.g. 100 GB) and mount at `/workspace` (or your path) so datasets and checkpoints survive restarts.

### 4.2 Clone repo and install (from workspace or home)

```bash
cd /workspace   # or your persistent mount
git clone https://github.com/YOUR_ORG/DREAMDIFFUSION.git
cd DREAMDIFFUSION
```

If you don’t use git, upload the project (e.g. zip) and unzip under `/workspace`.

### 4.3 Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e ./code
```

### 4.4 Data and pretrains

- **Option A:** Upload `datasets/` and `pretrains/` into the repo root (or into the persistent volume and set `DREAMDIFFUSION_DATA_ROOT` / `DREAMDIFFUSION_PRETRAIN_ROOT`).
- **Option B:** Use RunPod Network Storage and mount it, then point the env vars to the mount path.

Ensure at least:

- `datasets/eeg_5_95_std.pth`
- `datasets/block_splits_by_image_single.pth` (or `block_splits_tiny.pth`)
- `pretrains/models/config15.yaml`
- `pretrains/models/v1-5-pruned.ckpt`
- `pretrains/eeg_pretain/checkpoint.pth` (from Stage A or pre-downloaded)

### 4.5 Optional: reduce CUDA fragmentation (recommended for 24 GB)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Add to `~/.bashrc` or run before training.

---

## 5. Commands: full pipeline on RunPod

All commands from **repository root** with venv activated: `source venv/bin/activate`.

### 5.1 Tiny/small dataset (recommended for first RunPod test)

Create small splits once:

```bash
python code/make_tiny_splits.py
# Optional: python code/make_tiny_splits.py --train 200 --test 50
```

### 5.2 Stage A: EEG encoder pre-training (10 epochs)

```bash
python code/stageA1_eeg_pretrain.py --num_epoch 10
```

Full run (500 epochs):

```bash
python code/stageA1_eeg_pretrain.py
```

Copy Stage A checkpoint for Stage B (replace `<timestamp>` with the folder name under `results/eeg_pretrain/`):

```bash
cp results/eeg_pretrain/<timestamp>/checkpoints/checkpoint.pth pretrains/eeg_pretain/checkpoint.pth
```

### 5.3 Stage B: LDM fine-tuning (10 epochs, tiny data)

**Baseline (no SAR-HM):**

```bash
python code/eeg_ldm.py \
  --num_epoch 10 \
  --splits_path datasets/block_splits_tiny.pth \
  --val_gen_limit 2 \
  --model baseline \
  --run_name runpod_baseline_10ep_tiny \
  --eval_every 2 \
  --num_eval_samples 30
```

**SAR-HM (full):**

```bash
python code/eeg_ldm.py \
  --num_epoch 10 \
  --splits_path datasets/block_splits_tiny.pth \
  --val_gen_limit 2 \
  --model sarhm \
  --use_sarhm true \
  --ablation_mode full_sarhm \
  --run_name runpod_sarhm_10ep_tiny \
  --eval_every 2 \
  --num_eval_samples 30
```

**If you have 24 GB and want a bit more batch size:**

```bash
python code/eeg_ldm.py \
  --num_epoch 10 \
  --splits_path datasets/block_splits_tiny.pth \
  --val_gen_limit 2 \
  --batch_size 8 \
  --model baseline \
  --run_name runpod_baseline_10ep_tiny_b8
```

**If you hit OOM on 16–20 GB GPU:**

```bash
python code/eeg_ldm.py \
  --num_epoch 10 \
  --splits_path datasets/block_splits_tiny.pth \
  --val_gen_limit 1 \
  --batch_size 4 \
  --clip_tune false \
  --model baseline \
  --run_name runpod_baseline_10ep_tiny_safe
```

Stage B writes checkpoints to `results/generation/<timestamp>/` and (if thesis logging) to `runs/<timestamp>_<run_name>/`.

### 5.4 Stage C: Generate and evaluate

Replace `<timestamp>` with the Stage B output folder (e.g. from `results/generation/`):

```bash
python code/gen_eval_eeg.py \
  --dataset EEG \
  --model_path results/generation/<timestamp>/checkpoint.pth \
  --splits_path datasets/block_splits_tiny.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml
```

For full splits:

```bash
python code/gen_eval_eeg.py \
  --dataset EEG \
  --model_path results/generation/<timestamp>/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml
```

---

## 6. Saving work on RunPod

- **Checkpoints:** Stage A → `results/eeg_pretrain/<timestamp>/`. Stage B → `results/generation/<timestamp>/`. Copy these to your **persistent volume** or download via RunPod’s file browser / SCP.
- **Thesis logs:** `runs/<timestamp>_<run_name>/` (config, train_log.csv, eval_log.csv, artifacts). Copy the whole `runs/` folder to the volume.
- **Resume Stage B:** Use the same config and pass the last checkpoint:
  ```bash
  python code/eeg_ldm.py ... --checkpoint_path results/generation/<timestamp>/checkpoint.pth
  ```
  (If the checkpoint was saved with a different config, the script may load that config; keep paths consistent.)

---

## 7. Feasibility summary

| Question | Answer |
|----------|--------|
| Can the full pipeline run on RunPod? | **Yes.** Use a 24 GB+ GPU for comfortable runs; 16 GB is possible with smaller batch and val_gen_limit. |
| Cheapest viable GPU? | RTX A5000 (24 GB) or RTX 3090/4090; use batch 4–8 and val_gen_limit 2. |
| Fastest full 500+500 run? | A100 80 GB or H100; batch 12–24, val_gen_limit 5. |
| Spot vs on-demand? | Use **on-demand** for short runs (10 epochs) so you don’t lose progress. For long runs, use spot + frequent checkpointing and resume. |
| Persistent storage? | **Strongly recommended.** Attach a volume and keep datasets + pretrains + results there so you don’t re-download after each pod. |

---

## 8. One-page quick reference (RunPod)

```bash
# 1) Environment (once per pod)
cd /workspace/DREAMDIFFUSION && source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2) Tiny splits (once)
python code/make_tiny_splits.py

# 3) Stage A (10 ep)
python code/stageA1_eeg_pretrain.py --num_epoch 10
cp results/eeg_pretrain/*/checkpoints/checkpoint.pth pretrains/eeg_pretain/checkpoint.pth

# 4) Stage B baseline (10 ep)
python code/eeg_ldm.py --num_epoch 10 --splits_path datasets/block_splits_tiny.pth --val_gen_limit 2 --model baseline --run_name runpod_baseline_10ep

# 5) Stage B SAR-HM (10 ep)
python code/eeg_ldm.py --num_epoch 10 --splits_path datasets/block_splits_tiny.pth --val_gen_limit 2 --model sarhm --use_sarhm true --ablation_mode full_sarhm --run_name runpod_sarhm_10ep

# 6) Stage C (replace TIMESTAMP)
python code/gen_eval_eeg.py --dataset EEG --model_path results/generation/TIMESTAMP/checkpoint.pth --splits_path datasets/block_splits_tiny.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
```

This gives you complete training info and RunPod feasibility in one place.
