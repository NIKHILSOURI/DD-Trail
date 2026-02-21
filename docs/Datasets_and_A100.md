# Dataset Details & How Much You Can Run on A100 80GB

This document describes the **datasets** used in the pipeline and **recommended data scale and settings for an A100 80GB** GPU.

---

## 1. Dataset details

### 1.1 Primary dataset: ImageNet-EEG

| Item | Description |
|------|--------------|
| **Source** | Processed EEG recorded while subjects viewed ImageNet images (e.g. perceivelab/eeg_visual_classification, EEG-ImageNet). |
| **Main file** | `datasets/eeg_5_95_std.pth` – a PyTorch pickle containing a **list of samples** and metadata. |
| **Structure** | Each sample has `eeg` (EEG tensor), `image` (image index), `label` (class), `subject` (optional). The file also has `"dataset"`, `"labels"`, `"images"` keys. |
| **EEG shape** | After loading: typically **(channels, time)** e.g. (128, 512) or (62, 440) depending on processing. The code uses **512 time points** and **128 channels** (padded/cropped as needed). |
| **Total size** | Depends on the specific `.pth` you use. Common setups: **~2k–40k+ samples** (e.g. 40 classes × 50 images, possibly multiple trials per image). The **splits file** decides how many go to train vs test. |
| **Disk size** | `eeg_5_95_std.pth` is often **hundreds of MB to a few GB** depending on number of samples and preprocessing. |

### 1.2 Splits files

| File | Role | Typical size |
|------|------|--------------|
| **block_splits_by_image_single.pth** | One split: `splits[0]["train"]` and `splits[0]["test"]` are **lists of indices** into the full dataset. “Single” usually means one block/trial per image. | Small (KB). |
| **block_splits_by_image_all.pth** | Same format; “all” can mean all blocks/trials per image → **larger train/test sets**. | Small (KB). |
| **block_splits_tiny.pth** | Subset for fast runs: **150 train, 40 test** by default (created by `make_tiny_splits.py`). | Small (KB). |

**Format:**  
`torch.load(splits_path)` gives a dict with key `"splits"`: a list of dicts, each with `"train"` and `"test"` (iterables of integer indices). The code uses `split_num=0` (first split) and filters indices so that the corresponding EEG has length between 450 and 600 (then interpolated to 512).

### 1.3 Optional: ImageNet images on disk

- **Path:** `datasets/imageNet_images/` (or path set via dataset code).
- **Use:** If provided, the dataset loads **real images** from disk (by image name); otherwise it can use placeholder/noise. Needed for image-conditioned losses (e.g. CLIP) and for proper SSIM/CLIP evaluation.
- **Size:** Depends on subset (e.g. 4,000 images × ~100–500 KB each → **hundreds of MB to a few GB**).

### 1.4 Other datasets (ThoughtViz, MOABB)

- **ThoughtViz:** Can be integrated for qualitative/secondary experiments; not required for the default pipeline.
- **MOABB:** Used in some work for pretraining or regularization; not required for the default ImageNet-EEG training/eval.

---

## 2. How much you can take on A100 80GB

On an **A100 80GB** GPU you can run the **full pipeline** with **large batch sizes** and **full dataset** (whatever is in your `eeg_5_95_std.pth` and splits). VRAM is not the bottleneck; time and disk are.

### 2.1 Recommended settings (A100 80GB)

| Setting | Recommended value | Notes |
|---------|--------------------|--------|
| **Stage A batch_size** | **100** (default) or **128** | Fits easily; can try larger if dataset is huge. |
| **Stage B batch_size** | **16–24** | 16 is safe; 24 uses more VRAM but fewer steps per epoch. |
| **Stage B accumulate_grad** | **2** (or 1 if batch 24) | Effective batch 32–48. |
| **Stage B val_gen_limit** | **5** (or more) | Full validation; 80 GB can handle it. |
| **Stage B ddim_steps** | **50–250** | 250 for best quality; 50 for faster val. |
| **Dataset size** | **Full dataset** | Use `block_splits_by_image_single.pth` or `block_splits_by_image_all.pth` (no need for tiny). |
| **Gradient checkpointing** | Can **disable** for speed | 80 GB is enough; turn off in UNet config if you want faster training. |

### 2.2 Data scale vs VRAM (A100 80GB)

- **Number of samples:** The A100 80 GB does **not** limit how many **dataset samples** you can use. All samples are loaded from the `.pth` (or from disk) and the DataLoader feeds **batches**. You can use 10k, 50k, or 100k+ samples; training time scales with dataset size and epochs, not VRAM.
- **Batch size** is what uses VRAM. On A100 80 GB:
  - **Stage A:** batch 100–128 is fine; even 200 often fits.
  - **Stage B:** batch 16–24 is comfortable; some configs can try 32 with gradient checkpointing.
- **Validation generation:** `val_gen_limit` items × `num_samples` images × DDIM steps. With 5 items, 5 samples each, 250 steps, 80 GB is enough; you can even increase `val_gen_limit` (e.g. 10) if you want more validation samples.

### 2.3 Example: full run on A100 80GB

**Use the full dataset** (no tiny splits):

```bash
# Stage A (full dataset, large batch)
python code/stageA1_eeg_pretrain.py --num_epoch 500 --batch_size 100

# Copy checkpoint
cp results/eeg_pretrain/<timestamp>/checkpoints/checkpoint.pth pretrains/eeg_pretain/checkpoint.pth

# Stage B (full dataset, batch 16–24)
python code/eeg_ldm.py \
  --num_epoch 500 \
  --splits_path datasets/block_splits_by_image_single.pth \
  --batch_size 16 \
  --val_gen_limit 5 \
  --model baseline \
  --run_name a10080_baseline_full

# Or SAR-HM
python code/eeg_ldm.py \
  --num_epoch 500 \
  --splits_path datasets/block_splits_by_image_single.pth \
  --batch_size 16 \
  --val_gen_limit 5 \
  --model sarhm \
  --use_sarhm true \
  --ablation_mode full_sarhm \
  --run_name a10080_sarhm_full
```

**Rough time (full 500+500 epochs):** about **5–10 hours** on A100 80 GB depending on dataset size and validation frequency.

### 2.4 Summary table: dataset scale vs GPU

| GPU | Max recommended batch (Stage B) | Suggested dataset usage |
|-----|---------------------------------|--------------------------|
| 16 GB | 4 | Tiny (150/40) or small subset; full dataset is fine but more epochs take longer. |
| 24 GB | 8–12 | Full dataset; batch 8–12. |
| 48 GB | 12–16 | Full dataset; batch 12–16. |
| **A100 80 GB** | **16–24** | **Full dataset; batch 16–24; no need to reduce data.** |

**Bottom line:** On **A100 80GB** you can take the **full dataset** (whatever is in your splits), use **batch_size 16–24** for Stage B, and run **500 epochs** for both stages without reducing dataset size. Only reduce data (e.g. tiny splits) for quick experiments or when using smaller GPUs.
