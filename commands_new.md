# commands_new

**Repository root (this checkout):** `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail`

All paths below use that root. Replace it if your clone lives elsewhere.

| Symbol | Path / value |
|--------|----------------|
| **MAE checkpoint** | `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/pretrains/eeg_pretain/checkpoint.pth` (after §1.2) |
| **ImageNet JPEG root** | `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/imageNet_images` |
| **ImageNet-EEG** | `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/eeg_5_95_std.pth` |
| **ImageNet splits** | `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_by_image_single.pth` |
| **ThoughtViz pickle** | `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/data/eeg/image/data.pkl` |
| **ThoughtViz images** | `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/training/images` |
| **ThoughtViz EEG .pth** | `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/thoughtviz_eeg.pth` (build with §1.3a) |
| **ThoughtViz splits** | `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_thoughtviz_single.pth` (from §1.3a) |
| **Prototype output dir** | `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/exps/prototypes` |
| **`N` (ImageNet-EEG)** | `40` — must match `code/config.py` / your manifest |
| **`K` (EEG2Image subset)** | e.g. `40` — same across §2.2–§2.4 and benchmark `eeg2image_n_classes` |
| **ThoughtViz classes** | `10` |

**PowerShell vs bash:** In **PowerShell**, line continuation is a **backtick** (`` ` ``) at the **end** of each line, not `\`. A stray `\` is passed to Python and causes `unrecognized arguments: \`. **Git Bash** and **WSL** use `\` as below. Alternatively, use the **one-line** commands (no continuation) in the PowerShell blocks.

---

## 1. DreamDiffusion + SAR-HM

### 1.1 Environment setup

**Linux / Git Bash:**

```bash
cd D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e ./code
```

**Windows (PowerShell):**

```powershell
cd D:\STUDY\BTH\THESIS\NEW_DREAM\DD-Trail
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e ./code
```

Provide `datasets/eeg_5_95_std.pth`, `datasets/block_splits_by_image_single.pth`, `pretrains/models/v1-5-pruned.ckpt`, and `pretrains/models/config15.yaml`.

### 1.2 Stage A1 — EEG encoder pretraining

```bash
cd D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail
python code/stageA1_eeg_pretrain.py
```

Copy the newest run’s checkpoint (adjust the source folder to your `results/eeg_pretrain/<date-time>/checkpoints/`):

```bash
cp D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/results/eeg_pretrain/<TIMESTAMP_A1>/checkpoints/checkpoint.pth D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/pretrains/eeg_pretain/checkpoint.pth
```

**PowerShell:**

```powershell
Copy-Item "D:\STUDY\BTH\THESIS\NEW_DREAM\DD-Trail\results\eeg_pretrain\<TIMESTAMP_A1>\checkpoints\checkpoint.pth" `
  "D:\STUDY\BTH\THESIS\NEW_DREAM\DD-Trail\pretrains\eeg_pretain\checkpoint.pth"
```

### 1.3a ThoughtViz — build `thoughtviz_eeg.pth` + splits (once)

DreamDiffusion expects an ImageNet-style `.pth` plus a split file whose indices match that manifest. **Do not** use `block_splits_by_image_single.pth` for ThoughtViz.

**Layout:** class JPEGs live under `code/ThoughtViz/training/images/ImageNet-Filtered/<Class>/` (e.g. `Tiger/*.JPEG`). Pass `--images-root` as `.../training/images`; the exporter scans `ImageNet-Filtered` automatically and stores paths like `ImageNet-Filtered/Tiger/....JPEG` in the `.pth`.

**Linux / Git Bash:**

```bash
cd D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail
python scripts/export_thoughtviz_pkl_to_eeg_pth.py \
  --data-pkl D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/data/eeg/image/data.pkl \
  --images-root D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/training/images \
  --output D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/thoughtviz_eeg.pth \
  --splits-out D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_thoughtviz_single.pth
```

**Windows PowerShell** (backtick `` ` `` continues the line; or use the one-liner):

```powershell
cd D:\STUDY\BTH\THESIS\NEW_DREAM\DD-Trail
python scripts/export_thoughtviz_pkl_to_eeg_pth.py `
  --data-pkl D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/data/eeg/image/data.pkl `
  --images-root D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/training/images `
  --output D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/thoughtviz_eeg.pth `
  --splits-out D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_thoughtviz_single.pth
```



### 1.3 Prototype generation

SAR-HM class centroids (CLIP space), **before** SAR-HM Stage B, via `scripts/build_prototypes_by_domain.py`.

**ImageNet-EEG (`N=40`):**

```bash
cd D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail
python scripts/build_prototypes_by_domain.py \
  --domain imagenet_eeg \
  --output D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/exps/prototypes/proto_imagenet_eeg.pt \
  --eeg_signals_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/eeg_5_95_std.pth \
  --splits_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_by_image_single.pth \
  --imagenet_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/imageNet_images \
  --pretrain_mbm_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/pretrains/eeg_pretain/checkpoint.pth \
  --num_classes 40
```

**PowerShell (one line):**

```powershell
cd D:\STUDY\BTH\THESIS\NEW_DREAM\DD-Trail; python scripts/build_prototypes_by_domain.py --domain imagenet_eeg --output D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/exps/prototypes/proto_imagenet_eeg.pt --eeg_signals_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/eeg_5_95_std.pth --splits_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_by_image_single.pth --imagenet_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/imageNet_images --pretrain_mbm_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/pretrains/eeg_pretain/checkpoint.pth --num_classes 40
```

**ThoughtViz (`N=10`; requires §1.3a):**

```bash
cd D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail
python scripts/build_prototypes_by_domain.py \
  --domain thoughtviz \
  --output D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/exps/prototypes/proto_thoughtviz.pt \
  --thoughtviz_eeg_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/thoughtviz_eeg.pth \
  --thoughtviz_images_root D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/training/images \
  --splits_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_thoughtviz_single.pth \
  --pretrain_mbm_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/pretrains/eeg_pretain/checkpoint.pth \
  --num_classes 10
```

**PowerShell (one line):**

```powershell
cd D:\STUDY\BTH\THESIS\NEW_DREAM\DD-Trail; python scripts/build_prototypes_by_domain.py --domain thoughtviz --output D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/exps/prototypes/proto_thoughtviz.pt --thoughtviz_eeg_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/thoughtviz_eeg.pth --thoughtviz_images_root D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/training/images --splits_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_thoughtviz_single.pth --pretrain_mbm_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/pretrains/eeg_pretain/checkpoint.pth --num_classes 10
```

### 1.4 DreamDiffusion baseline training

**Training-only Stage B (stable, lower GPU load):** every `eeg_ldm.py` command in §1.4–§1.5 includes `--disable_image_generation_in_val true`, `--val_image_gen_every_n_epoch 0`, and `--skip_post_train_generation true`. That skips validation DDIM/PLMS sampling, post-training sample grids, and the heavy val CLIP/SSIM metrics tied to generated images. **Image generation and full metrics:** **§1.6** (`gen_eval_eeg.py`). To turn val sampling back on for debugging, use e.g. `--disable_image_generation_in_val false` and `--val_image_gen_every_n_epoch 1` (and consider `--smoke_test`).

**Windows PowerShell:** the bash examples use ``\`` line continuation; **PowerShell does not**. Use the **PowerShell one-liners** below, or use a backtick `` ` `` at the end of each line instead of ``\``.

**ImageNet-EEG:**

```bash
cd D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail
python code/eeg_ldm.py \
  --eeg_dataset_name imagenet_eeg \
  --imagenet_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/imageNet_images \
  --eeg_signals_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/eeg_5_95_std.pth \
  --splits_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_by_image_single.pth \
  --num_epoch 500 \
  --batch_size 100 \
  --num_workers 8 \
  --precision bf16 \
  --model baseline \
  --seed 2022 \
  --eval_every 2 \
  --num_eval_samples 50 \
  --use_sarhm false \
  --disable_image_generation_in_val true \
  --val_image_gen_every_n_epoch 0 \
  --skip_post_train_generation true
```

**PowerShell (one line) — ImageNet-EEG baseline:**

```powershell
cd D:\STUDY\BTH\THESIS\NEW_DREAM\DD-Trail; python code/eeg_ldm.py --eeg_dataset_name imagenet_eeg --imagenet_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/imageNet_images --eeg_signals_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/eeg_5_95_std.pth --splits_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_by_image_single.pth --num_epoch 500 --batch_size 100 --num_workers 8 --precision bf16 --model baseline --seed 2022 --eval_every 2 --num_eval_samples 50 --use_sarhm false --disable_image_generation_in_val true --val_image_gen_every_n_epoch 0 --skip_post_train_generation true
```

**ThoughtViz** (`--imagenet_path` must match the image root used for GT):

```bash
cd D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail
python code/eeg_ldm.py \
  --eeg_dataset_name thoughtviz \
  --imagenet_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/training/images \
  --thoughtviz_eeg_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/thoughtviz_eeg.pth \
  --thoughtviz_images_root D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/training/images \
  --splits_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_thoughtviz_single.pth \
  --num_epoch 500 \
  --batch_size 100 \
  --num_workers 8 \
  --precision bf16 \
  --model baseline \
  --seed 2022 \
  --eval_every 2 \
  --num_eval_samples 50 \
  --use_sarhm false \
  --disable_image_generation_in_val true \
  --val_image_gen_every_n_epoch 0 \
  --skip_post_train_generation true
```

**PowerShell (one line) — ThoughtViz baseline:**

```powershell
cd D:\STUDY\BTH\THESIS\NEW_DREAM\DD-Trail; python code/eeg_ldm.py --eeg_dataset_name thoughtviz --imagenet_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/training/images --thoughtviz_eeg_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/thoughtviz_eeg.pth --thoughtviz_images_root D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/training/images --splits_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_thoughtviz_single.pth --num_epoch 500 --batch_size 100 --num_workers 8 --precision bf16 --model baseline --seed 2022 --eval_every 2 --num_eval_samples 50 --use_sarhm false --disable_image_generation_in_val true --val_image_gen_every_n_epoch 0 --skip_post_train_generation true
```

Each run writes `checkpoint_best.pth` under `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/exps/results/generation/<TIMESTAMP>/`.

### 1.5 SAR-HM training

There is **no** `--proto_path` CLI for single-domain SAR-HM. Before **each** SAR-HM run, set `self.proto_path` in `code/config.py` (`Config_Generative_Model`) to:

- `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/exps/prototypes/proto_imagenet_eeg.pt` or
- `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/exps/prototypes/proto_thoughtviz.pt`

**ImageNet-EEG:**

```bash
cd D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail
python code/eeg_ldm.py \
  --eeg_dataset_name imagenet_eeg \
  --imagenet_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/imageNet_images \
  --eeg_signals_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/eeg_5_95_std.pth \
  --splits_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_by_image_single.pth \
  --num_epoch 500 \
  --batch_size 100 \
  --num_workers 8 \
  --precision bf16 \
  --model sarhm \
  --run_mode sarhm \
  --seed 2022 \
  --eval_every 2 \
  --num_eval_samples 50 \
  --ablation_mode full_sarhm \
  --disable_image_generation_in_val true \
  --val_image_gen_every_n_epoch 0 \
  --skip_post_train_generation true
```

**PowerShell (one line) — ImageNet-EEG SAR-HM:**

```powershell
cd D:\STUDY\BTH\THESIS\NEW_DREAM\DD-Trail; python code/eeg_ldm.py --eeg_dataset_name imagenet_eeg --imagenet_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/imageNet_images --eeg_signals_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/eeg_5_95_std.pth --splits_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_by_image_single.pth --num_epoch 500 --batch_size 100 --num_workers 8 --precision bf16 --model sarhm --run_mode sarhm --seed 2022 --eval_every 2 --num_eval_samples 50 --ablation_mode full_sarhm --disable_image_generation_in_val true --val_image_gen_every_n_epoch 0 --skip_post_train_generation true
```

**ThoughtViz:**

```bash
cd D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail
python code/eeg_ldm.py \
  --eeg_dataset_name thoughtviz \
  --imagenet_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/training/images \
  --thoughtviz_eeg_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/thoughtviz_eeg.pth \
  --thoughtviz_images_root D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/training/images \
  --splits_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_thoughtviz_single.pth \
  --num_epoch 500 \
  --batch_size 100 \
  --num_workers 8 \
  --precision bf16 \
  --model sarhm \
  --run_mode sarhm \
  --seed 2022 \
  --eval_every 2 \
  --num_eval_samples 50 \
  --ablation_mode full_sarhm \
  --disable_image_generation_in_val true \
  --val_image_gen_every_n_epoch 0 \
  --skip_post_train_generation true
```

**PowerShell (one line) — ThoughtViz SAR-HM:**

```powershell
cd D:\STUDY\BTH\THESIS\NEW_DREAM\DD-Trail; python code/eeg_ldm.py --eeg_dataset_name thoughtviz --imagenet_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/training/images --thoughtviz_eeg_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/thoughtviz_eeg.pth --thoughtviz_images_root D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/training/images --splits_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_thoughtviz_single.pth --num_epoch 500 --batch_size 100 --num_workers 8 --precision bf16 --model sarhm --run_mode sarhm --seed 2022 --eval_every 2 --num_eval_samples 50 --ablation_mode full_sarhm --disable_image_generation_in_val true --val_image_gen_every_n_epoch 0 --skip_post_train_generation true
```

Each run writes `checkpoint_best.pth` and `prototypes.pt` under `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/exps/results/generation/<TIMESTAMP>/`.

### 1.6 Generation / inference

Use this section after **training-only** Stage B (§1.4–§1.5): checkpoints from `exps/results/generation/<TIMESTAMP>/`, not the skipped val/post-train samples.

**ImageNet-EEG** (Stage C, `code/gen_eval_eeg.py`). Replace checkpoint paths with your run folders:

```bash
cd D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail
python code/gen_eval_eeg.py --dataset EEG \
  --model_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/exps/results/generation/<IE_BASELINE_TS>/checkpoint_best.pth \
  --splits_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_by_image_single.pth \
  --eeg_signals_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/eeg_5_95_std.pth \
  --imagenet_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/imageNet_images \
  --config_patch D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/pretrains/models/config15.yaml

python code/gen_eval_eeg.py --dataset EEG \
  --model_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/exps/results/generation/<IE_SARHM_TS>/checkpoint_best.pth \
  --splits_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_by_image_single.pth \
  --eeg_signals_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/eeg_5_95_std.pth \
  --imagenet_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/imageNet_images \
  --config_patch D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/pretrains/models/config15.yaml \
  --proto_path D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/exps/results/generation/<IE_SARHM_TS>/prototypes.pt
```

**ThoughtViz:** `gen_eval_eeg.py` currently builds the EEG dataset **without** `build_stage_b_datasets` ThoughtViz mode. For ThoughtViz-conditioned checkpoints, use **§3** `benchmark.orchestrate_all` with TV `checkpoint_best.pth` / `prototypes.pt` in `configs/benchmark_unified.yaml`.

### 1.7 Required outputs before benchmarking

- `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/pretrains/eeg_pretain/checkpoint.pth`
- **ImageNet-EEG:** baseline + SAR-HM `checkpoint_best.pth` and SAR-HM `prototypes.pt` from the IE generation run dirs
- **ThoughtViz:** same for TV runs

---

## 2. EEG2Image

**Windows PowerShell:** blocks that use bash ``\`` line breaks (§2.2, §2.5) will **fail** if pasted into PowerShell — use the **PowerShell one-liners** under each section, or end each continued line with a backtick `` ` `` instead of ``\``.

### 2.1 Environment setup

**Linux / Git Bash:**

```bash
cd D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail
python -m venv third_party/eeg-venv
source third_party/eeg-venv/bin/activate
pip install -r requirements-eeg2image.txt
```

**Windows:**

```powershell
cd D:\STUDY\BTH\THESIS\NEW_DREAM\DD-Trail
python -m venv third_party\eeg-venv
.\third_party\eeg-venv\Scripts\Activate.ps1
pip install -r requirements-eeg2image.txt
```

### 2.2 Encode / export ImageNet-EEG

Example: `K=40`, export under `results/eeg2image_export_ie` (create any empty dir name you prefer):

```bash
cd D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail
python scripts/export_imagenet_eeg_for_eeg2image.py \
  --output-dir D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/results/eeg2image_export_ie \
  --split-names train test \
  --class-count 40 \
  --imagenet-root D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/imageNet_images
```

**PowerShell (one line — use this on Windows):**

```powershell
cd D:\STUDY\BTH\THESIS\NEW_DREAM\DD-Trail; python scripts/export_imagenet_eeg_for_eeg2image.py --output-dir D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/results/eeg2image_export_ie --split-names train test --class-count 40 --imagenet-root D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/imageNet_images
```

Writes `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/results/eeg2image_export_ie/data.pkl` and `export_metadata.json`.

### 2.3 Train TripleNet on ImageNet-EEG subset

Edit `configs/eeg2image/triplet_imagenet_eeg.json`: set `imagenet_export_pkl` to the export `data.pkl`, `n_classes` = `K`, and a non-empty `run_id` (e.g. `imagenet_eeg`).

```bash
cd D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/third_party/EEG2Image/lstm_kmean
python train.py --config D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/configs/eeg2image/triplet_imagenet_eeg.json --n-classes 40
```

### 2.4 Train DCGAN on ImageNet-EEG subset

Edit `configs/eeg2image/dcgan_imagenet_eeg.json`: export paths, `imagenet_root`, `n_classes`, same `run_id` as §2.3.

```bash
cd D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/third_party/EEG2Image
python train.py --config D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/configs/eeg2image/dcgan_imagenet_eeg.json --n-classes 40
```

### 2.5 ThoughtViz training path

```bash
cd D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/third_party/EEG2Image/lstm_kmean
python train.py \
  --dataset-mode thoughtviz \
  --thoughtviz-pkl D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/data/eeg/image/data.pkl \
  --n-classes 10

cd D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/third_party/EEG2Image
python train.py \
  --dataset-mode thoughtviz \
  --thoughtviz-pkl D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/data/eeg/image/data.pkl \
  --b2i-images-train-root D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/training/images \
  --n-classes 10
```

**PowerShell (one line each):**

```powershell
cd D:\STUDY\BTH\THESIS\NEW_DREAM\DD-Trail\third_party\EEG2Image\lstm_kmean; python train.py --dataset-mode thoughtviz --thoughtviz-pkl D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/data/eeg/image/data.pkl --n-classes 10
```

```powershell
cd D:\STUDY\BTH\THESIS\NEW_DREAM\DD-Trail\third_party\EEG2Image; python train.py --dataset-mode thoughtviz --thoughtviz-pkl D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/data/eeg/image/data.pkl --b2i-images-train-root D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/training/images --n-classes 10
```

### 2.6 Required outputs before benchmarking

- **ImageNet-EEG:** `results/eeg2image_export_ie/data.pkl`, `export_metadata.json`, checkpoints under `third_party/EEG2Image/experiments/runs/<run_id>/` and `third_party/EEG2Image/lstm_kmean/experiments/runs/<run_id>/triplet_ckpt`
- **ThoughtViz:** legacy layout under `lstm_kmean/experiments/best_ckpt` and `experiments/ckpt` (or `experiments/best_ckpt`) when `run_id` is empty, or matching `run_id` dirs

---

## 3. Benchmarking

### 3.1 Benchmark environment assumptions

```bash
cd D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail
source venv/bin/activate
```

**Windows:** `.\venv\Scripts\Activate.ps1`

The file `configs/benchmark_unified.yaml` is already wired for this tree, for example:

- `paths.repo_root`: `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail`
- `paths.imagenet_path`: `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/imageNet_images`
- `paths.eeg_signals_path`: `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/eeg_5_95_std.pth`
- `paths.splits_path`: `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/datasets/block_splits_by_image_single.pth`
- `paths.thoughtviz_data_dir`: `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/data`
- `paths.thoughtviz_image_dir`: `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/code/ThoughtViz/training/images`
- `paths.eeg2image_root`: `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/third_party/EEG2Image`

Update `baseline_ckpt`, `sarhm_ckpt`, and `sarhm_proto` to your actual `exps/results/generation/<TIMESTAMP>/` files.

`orchestrate_all` uses **one** `baseline_ckpt` and **one** `sarhm_ckpt` / `sarhm_proto` pair per run. For **dataset-matched** LDM weights: IE checkpoints + `--dataset imagenet_eeg`, or TV checkpoints + `--dataset thoughtviz`, or edit YAML between runs.

For EEG2Image on the ImageNet-EEG campaign, set `eeg2image_n_classes: 40` (or your `K`) and `eeg2image_run_id` to the §2.3 `run_id`. For ThoughtViz defaults (10 classes, legacy dirs), omit `eeg2image_run_id` unless you trained with a `run_id`.

Set `posthoc.enabled: false` unless the JSON paths under `posthoc` exist.

### 3.2 Main benchmark command

**Linux / Git Bash:**

```bash
cd D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail
source venv/bin/activate
python -m benchmark.orchestrate_all \
  --config D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/configs/benchmark_unified.yaml \
  --dataset both \
  --eeg2image_python D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/third_party/eeg-venv/bin/python
```

**Windows (PowerShell):**

```powershell
cd D:\STUDY\BTH\THESIS\NEW_DREAM\DD-Trail
.\venv\Scripts\Activate.ps1
python -m benchmark.orchestrate_all `
  --config D:\STUDY\BTH\THESIS\NEW_DREAM\DD-Trail\configs\benchmark_unified.yaml `
  --dataset both `
  --eeg2image_python D:\STUDY\BTH\THESIS\NEW_DREAM\DD-Trail\third_party\eeg-venv\Scripts\python.exe
```

### 3.3 Expected outputs

Under `output.root` / `run_name` in the YAML (e.g. `D:/STUDY/BTH/THESIS/NEW_DREAM/DD-Trail/results/benchmark_unified/thesis_unified/`): `benchmark_outputs/<dataset>/sample_<id>/` with `dreamdiffusion.png`, `sarhm.png`, `eeg2image.png`, `ground_truth.png` where applicable.

---

## 4. Notes

- **Stage B training-only:** `eeg_ldm.py` examples in §1.4–§1.5 use `--disable_image_generation_in_val true --val_image_gen_every_n_epoch 0 --skip_post_train_generation true` (same as `code/config.py` defaults). Generation: §1.6.
- **Windows:** use `venv\Scripts\activate` or `Activate.ps1` and `third_party\eeg-venv\Scripts\python.exe` instead of `source …/bin/activate` and `…/bin/python`.
- **ThoughtViz LDM / prototypes:** always use `datasets/thoughtviz_eeg.pth` + `datasets/block_splits_thoughtviz_single.pth` from §1.3a, not ImageNet splits.
- **One EEG2Image checkpoint pair per orchestrator run:** use separate runs or YAML edits if IE-trained and TV-trained EEG2Image weights differ.
- **`config.proto_path`:** reset in `code/config.py` when switching between ImageNet-EEG and ThoughtViz SAR-HM training so each run loads the correct §1.3 prototype file.
