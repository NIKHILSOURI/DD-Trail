

### Datasets and Usage (Thesis-Oriented)

| Dataset | Role | Use for | Claims |
|--------|------|---------|--------|
| **ImageNet-EEG** | Primary | Main training, quantitative evaluation, ablations, retrieval accuracy, CLIP similarity | Primary thesis claims |
| **ThoughtViz** | Secondary / qualitative | Qualitative image generation, discussion | No heavy quantitative claims |
| **MOABB** | Pretraining only | Optional EEG encoder pretraining/regularization | No image-generation evaluation |

See `docs/SARHM_README.md` for full dataset policy and SAR-HM usage.

### How to Switch Modes (Baseline vs SAR-HM)

- **Baseline DreamDiffusion**: In `code/config.py`, keep `use_sarhm = False` (default). Training and generation use the original EEG → MAE → channel_mapper → dim_mapper → SD path.
- **SAR-HM**: Set `use_sarhm = True` and choose `ablation_mode` in `Config_Generative_Model`:
  - `'projection_only'` – EEG → projection → adapter → SD
  - `'hopfield_no_gate'` – add Hopfield retrieval, no gating
  - `'full_sarhm'` – Hopfield + confidence-gated fusion
- Pass the same config as `main_config` into `eLDM(..., main_config=config)` and `eLDM_eval(..., main_config=config)` so evaluation matches training.

The **Stable Diffusion** stack (UNet, VAE, text encoder) is never finetuned; only the EEG encoder, projection, Hopfield memory, and adapter are trained when SAR-HM is enabled.

**Reproducibility checklist**: Same seed, same splits, same checkpoint config (including SAR-HM flags when loading for eval). See `docs/SARHM_README.md` for the full list.

---

The **datasets** folder and **pretrains** folder are not included in this repository. 
Please download them from [eeg](https://github.com/perceivelab/eeg_visual_classification) and put them in the root directory of this repository as shown below. We also provide a copy of the Imagenet subset [imagenet](https://drive.google.com/file/d/1y7I9bG1zKYqBM94odcox_eQjnP9HGo9-/view?usp=drive_link).

For Stable Diffusion, we just use standard SD1.5. You can download it from the [official page of Stability](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main). You want the file ["v1-5-pruned.ckpt"](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main).

File path | Description
```

/pretrains
┣ 📂 models
┃   ┗ 📜 config15.yaml
┃   ┗ 📜 v1-5-pruned.ckpt

┣ 📂 generation  
┃   ┗ 📜 checkpoint_best.pth 

┣ 📂 eeg_pretain
┃   ┗ 📜 checkpoint.pth  (pre-trained EEG encoder)

┣ 📂 sarhm         (optional; created automatically when using SAR-HM)
┃   ┗ 📜 prototypes_dummy.pt  (dummy prototypes if none provided)

/datasets
┣ 📂 imageNet_images (subset of Imagenet)

┗  📜 block_splits_by_image_all.pth
┗  📜 block_splits_by_image_single.pth 
┗  📜 eeg_5_95_std.pth  

/code
┣ 📂 sc_mbm
┃   ┗ 📜 mae_for_eeg.py
┃   ┗ 📜 trainer.py
┃   ┗ 📜 utils.py

┣ 📂 sarhm
┃   ┗ 📜 sarhm_modules.py
┃   ┗ 📜 prototypes.py
┃   ┗ 📜 metrics_logger.py
┃   ┗ 📜 vis.py

┣ 📂 dc_ldm
┃   ┗ 📜 ldm_for_eeg.py
┃   ┗ 📜 utils.py
┃   ┣ 📂 models
┃   ┃   ┗ (adopted from LDM)
┃   ┣ 📂 modules
┃   ┃   ┗ (adopted from LDM)

┗  📜 stageA1_eeg_pretrain.py   (main script for EEG pre-training)
┗  📜 eeg_ldm.py                (main script for fine-tuning stable diffusion)
┗  📜 gen_eval_eeg.py           (main script for generating images)
┗  📜 dataset.py                (functions for loading datasets)
┗  📜 eval_metrics.py           (functions for evaluation metrics)
┗  📜 config.py                 (configurations for the main scripts)

┣  📂 docs
┃   ┗  📜 SARHM_README.md       (SAR-HM config, ablations, dataset policy)
┃   ┗  📜 logging.md            (thesis logging: metrics, run layout, eval-only)

```

---

## Quick start (10 epochs + test)

**Prerequisites:** Data in `datasets/` and `pretrains/` (see file tree below). From repo root with venv activated:

```sh
# One-time setup
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e ./code
```

**Run for 10 epochs only (quick sanity run):**

```sh
# Stage A1: 10 epochs
python code/stageA1_eeg_pretrain.py --num_epoch 10
# Copy checkpoint (replace <timestamp> with the folder name under results/eeg_pretrain/)
# Windows: copy results\eeg_pretrain\<timestamp>\checkpoints\checkpoint.pth pretrains\eeg_pretain\checkpoint.pth
# Linux/Mac: cp results/eeg_pretrain/<timestamp>/checkpoints/checkpoint.pth pretrains/eeg_pretain/checkpoint.pth

# Stage B: 10 epochs
python code/eeg_ldm.py --num_epoch 10

# Stage C: generate and evaluate (use the Stage B timestamp in model_path)
python code/gen_eval_eeg.py --dataset EEG --model_path results/generation/<timestamp>/checkpoint.pth --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
```

### 10 epochs + tiny dataset: baseline vs SAR-HM (for comparison)

Use a small subset of the data to compare **baseline** and **SAR-HM** in 10 epochs. From repo root with venv activated:

```sh
# 1) Create tiny splits once (150 train, 40 test; optional: --train 200 --test 50)
python code/make_tiny_splits.py

# 2) Stage A1 (optional; skip if you already have pretrains/eeg_pretain/checkpoint.pth)
python code/stageA1_eeg_pretrain.py --num_epoch 10
# Copy checkpoint to pretrains/eeg_pretain/checkpoint.pth (see copy commands earlier in README)

# 3a) Stage B — BASELINE: 10 epochs, tiny dataset, no SAR-HM
# On 16GB GPU add: --batch_size 4 (and optionally --accumulate_grad 6)
python code/eeg_ldm.py --num_epoch 10 --splits_path datasets/block_splits_tiny.pth --val_gen_limit 2 --use_sarhm false

# 3b) Stage B — SAR-HM: 10 epochs, tiny dataset, full_sarhm (run after or in parallel with 3a)
python code/eeg_ldm.py --num_epoch 10 --splits_path datasets/block_splits_tiny.pth --val_gen_limit 2 --use_sarhm true --ablation_mode full_sarhm

# 4) Stage C: generate and evaluate (run once per model; replace <timestamp> with the folder from 3a or 3b)
# Baseline:
python code/gen_eval_eeg.py --dataset EEG --model_path results/generation/<timestamp_baseline>/checkpoint.pth --splits_path datasets/block_splits_tiny.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
# SAR-HM:
python code/gen_eval_eeg.py --dataset EEG --model_path results/generation/<timestamp_sarhm>/checkpoint.pth --splits_path datasets/block_splits_tiny.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
```

- **Tiny splits:** `make_tiny_splits.py` writes `datasets/block_splits_tiny.pth` (150 train, 40 test; override with `--train N --test M`).
- **Baseline:** `--use_sarhm false` uses only the EEG→MAE→mapper→SD path.
- **SAR-HM:** `--use_sarhm true --ablation_mode full_sarhm` uses Hopfield + confidence-gated fusion on the same tiny data.
- **Val gen:** `--val_gen_limit 2` keeps validation generation short. Each run creates a new folder under `results/generation/` (e.g. `13-02-2026-10-11-59`); use that as `<timestamp_baseline>` or `<timestamp_sarhm>` in Stage C.
- **16GB GPU (CUDA OOM):** Use a smaller batch size so training fits in memory, e.g. `--batch_size 4` (optionally `--accumulate_grad 6` to keep effective batch size similar). You can also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before running to reduce fragmentation.

### Faster training (RunPod / A100 / multi-GPU servers)

To **maximize throughput** and GPU utilization:

| Option | Effect | Example |
|--------|--------|--------|
| **Mixed precision** | Less VRAM, faster steps | Default is `16` (FP16). On A100 use `--precision bf16`. |
| **Larger batch** | More GPU utilization | `--batch_size 24` or `32` (tune to fit VRAM). |
| **DataLoader workers** | Avoid CPU bottleneck | `--num_workers 8` (Linux/RunPod; use `0` on Windows). |
| **Validate less often** | Fewer slow val runs | Default `check_val_every_n_epoch=5`; override with `--check_val_every_n_epoch 10` for even faster. |
| **Fewer val items/samples/steps** | Shorter val generation | Defaults: `val_gen_limit=2`, `val_ddim_steps=50`, `val_num_samples=2` (validation only; **Stage C** still uses full 250 steps, 5 samples for thesis metrics). |
| **torch.compile** | Not used | Disabled for Stage B (validation/generation incompatible). |

**Why Stage B was slow:** Each validation run does image generation (PLMS steps × samples × items). Defaults above make validation fast; final thesis numbers come from Stage C with full quality.

**Example (fast run on A100 80GB):**

```sh
python code/eeg_ldm.py --num_epoch 10 --batch_size 24 --num_workers 8 --precision bf16 --check_val_every_n_epoch 5 --val_gen_limit 2
```

Use `--precision 32` if you need full precision (e.g. for debugging).

**Tests:** See [Tests](#tests) for the preflight script and smoke test.

---

### Thesis logging and evaluation (Baseline vs SAR-HM)

To get **thesis-grade run folders** with `config.json`, `train_log.csv`, `eval_log.csv`, and per-dataset artifacts (grids, optional samples), pass `--model` and optionally `--run_name`:

```sh
# Baseline with logging (run folder: runs/<timestamp>_baseline_2022/)
python code/eeg_ldm.py --run_name my_baseline --model baseline --seed 2022 --eval_every 2 --num_eval_samples 50

# SAR-HM with logging (run folder: runs/<timestamp>_sarhm_2022/)
python code/eeg_ldm.py --run_name my_sarhm --model sarhm --seed 2022 --eval_every 2 --num_eval_samples 50 --use_sarhm true --ablation_mode full_sarhm
```

- **Run folder:** `runs/<timestamp>_<model>_<seed>/` (or `runs/<timestamp>_<run_name>/` if you set `--run_name`). Contains `config.json`, `train_log.csv`, `eval_log.csv`, and `artifacts/imagenet_eeg/`, `artifacts/thoughtviz/`, `artifacts/moabb/`.
- **Train log:** Per-epoch metrics (e.g. `train/loss_total`, `train/loss_gen`, and SAR-HM metrics when applicable).
- **Eval log:** Per-dataset, per-epoch metrics (FID, IS, SSIM, CLIP similarity, `n_samples`). Inapplicable metrics (e.g. SSIM on MOABB) are stored as `NA`.
- **Artifacts:** For each evaluated dataset, `artifacts/<dataset>/grid_epochXX.png` and optionally `samples/`.

**Evaluation-only (saved checkpoint):** To run evaluation without re-training, load the checkpoint and test dataset, create a run dir and `MetricLogger`, then call the same `evaluate()` used in training (see `code/eval/evaluate.py`). Full metric descriptions and when each applies (e.g. MOABB caveats) are in **`docs/logging.md`**.

**Full command reference (thesis-grade):** See **`docs/thesis_grade_commands.md`**.  
**Project explainer (config, training vs validation vs inference, timings):** See **`docs/explain.md`**.

---

## Complete Steps to Run the System

Run all commands from the **repository root**. On Windows use `python` (not `python3`). Activate the environment first when using a venv.

### Step 0: One-time setup

1. **Create environment and install dependencies**
   ```sh
   py -3.11 -m venv venv
   venv\Scripts\Activate.ps1
   pip install --upgrade pip setuptools wheel
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   pip install -e ./code
   ```
   Or, with an existing venv, from repo root: **`scripts\install_venv_deps.bat`** (Windows) to install `requirements.txt` and the code package into `venv`. If you use Python 3.13 and `scipy` fails to install, create the venv with Python 3.10 or 3.11 (e.g. `py -3.11 -m venv venv` on Windows).

2. **Place data and checkpoints**
   - `datasets/`: `eeg_5_95_std.pth`, `block_splits_by_image_single.pth` (or `block_splits_by_image_all.pth`), and optionally `imageNet_images/` for ImageNet-EEG.
   - `pretrains/models/`: `v1-5-pruned.ckpt` (Stable Diffusion 1.5), `config15.yaml`.
   - **If you see `ModuleNotFoundError`** (e.g. `einops`, `scipy`, `matplotlib`, `wandb`): with the venv activated, run `pip install -r requirements.txt` from the repo root. Use the venv’s pip (e.g. `.\venv\Scripts\pip.exe install -r requirements.txt` if needed).

### Step 1: Pre-train EEG encoder (Stage A1)

```sh
python code/stageA1_eeg_pretrain.py
```

- Checkpoint is saved under `results/eeg_pretrain/<timestamp>/checkpoints/checkpoint.pth`.
- Copy it for Stage B:
  ```sh
  copy results\eeg_pretrain\<timestamp>\checkpoints\checkpoint.pth pretrains\eeg_pretain\checkpoint.pth
  ```

### Step 2: Fine-tune LDM (Stage B)

**Baseline DreamDiffusion (default)**  
- In `code/config.py`, keep `use_sarhm = False` in `Config_Generative_Model`.
- Run:
  ```sh
  python code/eeg_ldm.py
  ```
- Checkpoint: `results/generation/<timestamp>/checkpoint.pth`.

**SAR-HM**  
- In `code/config.py`, set in `Config_Generative_Model`:
  - `use_sarhm = True`
  - `ablation_mode = 'full_sarhm'` (or `'projection_only'`, `'hopfield_no_gate'`)
  - `num_classes = 40`
  - Optionally `proto_path` to load/save class prototypes.
- Run the same command:
  ```sh
  python code/eeg_ldm.py
  ```
- The script passes `main_config=config` into `eLDM`, so SAR-HM flags are applied automatically.

### Step 3: Generate and evaluate (Stage C)

Use the Stage B checkpoint path and the same dataset/splits as in training:

```sh
python code/gen_eval_eeg.py --dataset EEG --model_path results/generation/<timestamp>/checkpoint.pth --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
```

- If you use a checkpoint trained with SAR-HM, the saved `config` in the checkpoint is loaded and passed as `main_config` into `eLDM_eval`, so generation uses the same SAR-HM setup.
- Optionally pass `--imagenet_path datasets/imageNet_images` if you use ImageNet-EEG with stimulus images.

### Summary: commands in order

| Step | What | Command |
|------|------|--------|
| 1 | Pre-train EEG encoder | `python code/stageA1_eeg_pretrain.py` |
| 2 | Fine-tune LDM (baseline or SAR-HM per config) | `python code/eeg_ldm.py` |
| 3 | Generate and evaluate | `python code/gen_eval_eeg.py --dataset EEG --model_path <Stage_B_ckpt> --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml` |

For a **quick 10-epoch run**, add `--num_epoch 10` to the Stage 1 and Stage 2 commands. For SAR-HM options, ablations, and dataset roles, see `docs/SARHM_README.md`.

---

## Environment setup

### Option 1: Using venv (Recommended for Windows)

1. **Create a virtual environment:**
```sh
# On Windows PowerShell
py -3.11 -m venv venv
venv\Scripts\Activate.ps1

# On Windows CMD
python -m venv venv
venv\Scripts\activate.bat

# On Windows Git Bash / Linux / Mac
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
# or: source venv/bin/activate  # Linux/Mac
```

2. **Upgrade pip and install PyTorch first** (important for CUDA support):
```sh
# Upgrade pip first
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (recommended if you have an NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# OR install CPU-only version (if no GPU available)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3. **Install remaining dependencies:**
```sh
pip install -r requirements.txt
pip install -e ./code
```

4. **Verify GPU/CPU availability:**
```sh
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"
```

5. **Set up data and checkpoints:**
   - Place `datasets/` folder at repository root (containing `eeg_5_95_std.pth`, `block_splits_by_image_*.pth`, and `imageNet_images/` subdirectory)
   - Place `pretrains/` folder at repository root:
     - `pretrains/models/v1-5-pruned.ckpt` (download from [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main))
     - `pretrains/models/config15.yaml` (should already exist)
     - Optionally: `pretrains/generation/checkpoint_best.pth` and `pretrains/eeg_pretrain/checkpoint.pth` for training

## Complete Training Pipeline

This guide covers the complete pipeline from scratch. The training consists of **3 main stages**:

1. **Stage A1: EEG Encoder Pre-training** - Train the EEG encoder using masked signal modeling
2. **Stage B: LDM Fine-tuning** - Fine-tune Stable Diffusion with the pretrained EEG encoder
3. **Stage C: Generation & Evaluation** - Generate images from EEG signals

**Prerequisites:**
- All dependencies installed (see Environment setup above)
- Datasets downloaded and placed in `datasets/` folder
- Stable Diffusion 1.5 checkpoint (`v1-5-pruned.ckpt`) placed in `pretrains/models/`
- Environment activated: `venv\Scripts\Activate.ps1` (PowerShell) or `venv\Scripts\activate.bat` (CMD)

**Note:** 
- Run all commands from the **repository root directory**
- On **Windows**: Use `python` (not `python3`)
- **Activate your environment first** before running any commands

---

### Stage A1: EEG Encoder Pre-training

This stage trains the EEG encoder using Masked Autoencoder (MAE) approach.

**Command:**
```sh
# Activate environment first
venv\Scripts\Activate.ps1  # PowerShell
# or: venv\Scripts\activate.bat  # CMD

# Run Stage A1 training
python code/stageA1_eeg_pretrain.py
```

**What this does:**
- Trains the EEG encoder on EEG signals using masked signal modeling
- Saves checkpoint to `results/eeg_pretrain/[timestamp]/checkpoints/checkpoint.pth`
- Configurable parameters in `code/config.py` (Config_MBM_EEG class)

**Output:**
- Trained EEG encoder checkpoint saved to: `results/eeg_pretrain/[timestamp]/checkpoints/checkpoint.pth`
- You will need this checkpoint path for Stage B

**Move checkpoint for Stage B:**
```sh
# Copy the checkpoint to pretrains folder (adjust timestamp to your actual folder)
copy results\eeg_pretrain\[timestamp]\checkpoints\checkpoint.pth pretrains\eeg_pretrain\checkpoint.pth
```

---

### Stage B: Fine-tuning Stable Diffusion with EEG Encoder

This stage fine-tunes Stable Diffusion using the pretrained EEG encoder from Stage A1.

**Prerequisites:**
- Stage A1 checkpoint: `pretrains/eeg_pretrain/checkpoint.pth`
- Stable Diffusion checkpoint: `pretrains/models/v1-5-pruned.ckpt`
- Stable Diffusion config: `pretrains/models/config15.yaml`

**Command:**
```sh
# Ensure environment is activated
venv\Scripts\Activate.ps1  # PowerShell

# Run Stage B training
python code/eeg_ldm.py
```

**What this does:**
- Loads the pretrained EEG encoder from Stage A1
- Loads Stable Diffusion 1.5 checkpoint
- Fine-tunes the model to generate images from EEG signals
- Saves checkpoint to `results/generation/[timestamp]/checkpoint.pth`

**Output:**
- Fine-tuned model checkpoint: `results/generation/[timestamp]/checkpoint.pth`
- Training logs and sample images in the same directory

**Move checkpoint for Stage C (optional):**
```sh
# Copy checkpoint to pretrains folder (adjust timestamp)
copy results\generation\[timestamp]\checkpoint.pth pretrains\generation\checkpoint_best.pth
```

---

### Stage C: Generation & Evaluation

This stage generates images from EEG signals using the trained model.

**Prerequisites:**
- **Stage B checkpoint must exist** - You need to complete Stage B training first, OR download a pre-trained checkpoint
- Dataset files in `datasets/` folder

**Option 1: Use checkpoint from Stage B training**
```sh
# After Stage B completes, find your checkpoint in results/generation/[timestamp]/checkpoint.pth
# Then run:
venv\Scripts\Activate.ps1  # PowerShell
python code/gen_eval_eeg.py --dataset EEG --model_path results/generation/[timestamp]/checkpoint.pth --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
```

**Option 2: Use pre-trained checkpoint (if available)**
```sh
# If you have a pre-trained checkpoint, place it at pretrains/generation/checkpoint_best.pth
# Then run:
venv\Scripts\Activate.ps1  # PowerShell
python code/gen_eval_eeg.py --dataset EEG --model_path pretrains/generation/checkpoint_best.pth --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
```

**Note:** If you get "Model checkpoint not found" error, you must complete Stage B training first or provide a valid checkpoint path.

**What this does:**
- Loads the trained model from Stage B
- Generates images from EEG signals in the test set
- Computes evaluation metrics (SSIM, PCC, etc.)
- Saves generated images to `results/eval/[timestamp]/`

**Output:**
- Generated images: `results/eval/[timestamp]/samples_test.png`
- Individual test images: `results/eval/[timestamp]/test*.png`
- Evaluation metrics printed to console

---

## Quick Reference: Complete Pipeline

See **"Complete Steps to Run the System"** above for the full sequence and SAR-HM options. Minimal sequence:

```sh
# 0. One-time setup
python -m venv venv
venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e ./code

# 1. Stage A1: Train EEG encoder
python code/stageA1_eeg_pretrain.py
# Copy: copy results\eeg_pretrain\[timestamp]\checkpoints\checkpoint.pth pretrains\eeg_pretain\checkpoint.pth

# 2. Stage B: Fine-tune (baseline or SAR-HM per code/config.py)
python code/eeg_ldm.py

# 3. Stage C: Generate and evaluate
python code/gen_eval_eeg.py --dataset EEG --model_path results/generation/[timestamp]/checkpoint.pth --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
```

**Quick run (10 epochs):** Add `--num_epoch 10` to Stage A1 and Stage B commands above.

---

## Tests

The project includes a **preflight script** that checks required files, imports, and runs a minimal smoke generation (baseline + SAR-HM). Run from the **repository root** with your environment activated:

```sh
python code/ci_preflight.py
```

- **Smoke only (skip compile/import):**  
  `python code/ci_preflight.py --smoke_only`
- **Run on GPU:** The script uses CUDA automatically if available. It prints `Using GPU: <name>` or `Using CPU`. To **require** GPU (exit if no CUDA):  
  `python code/ci_preflight.py --require_gpu`  
  Ensure PyTorch is installed with CUDA (e.g. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`).

If the script reports missing files, install the datasets and checkpoints as in Step 0. If imports fail, activate the same venv/conda environment you use for training and run again.

---

## Generating Images with Pre-trained Checkpoints (Optional)

If you have pre-trained checkpoints (e.g., from the paper authors), you can skip Stages A1 and B and run generation directly. If the checkpoint was trained with SAR-HM, `gen_eval_eeg.py` loads the saved config and uses the same SAR-HM setup automatically.

```sh
# On Windows PowerShell (venv - recommended)
venv\Scripts\Activate.ps1
python code/gen_eval_eeg.py --dataset EEG --model_path pretrains/generation/checkpoint_best.pth --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml

# On Windows CMD (venv)
venv\Scripts\activate.bat
python code/gen_eval_eeg.py --dataset EEG --model_path pretrains/generation/checkpoint_best.pth --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml

# On Windows with conda
conda activate dreamdiffusion
python code/gen_eval_eeg.py --dataset EEG --model_path pretrains/generation/checkpoint_best.pth --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml

# On Linux/Mac (with venv activated)
source venv/bin/activate
python code/gen_eval_eeg.py --dataset EEG --model_path pretrains/generation/checkpoint_best.pth --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
```

**Note:** On Windows, always use `python` (not `python3`). The `python3` command may not be available.



Commands you can use
#Thesis-level (500 epochs, batch 100) – Baseline:

```sh
python code/eeg_ldm.py --num_epoch 500 --batch_size 100 --num_workers 8 --precision bf16 --model baseline --seed 2022 --eval_every 2 --num_eval_samples 50 --use_sarhm false


#Thesis-level (500 epochs, batch 100) – SAR-HM:

python code/eeg_ldm.py --num_epoch 500 --batch_size 100 --num_workers 8 --precision bf16 --model sarhm --seed 2022 --eval_every 2 --num_eval_samples 50 --use_sarhm true --ablation_mode full_sarhm


#10-epoch testing (batch 100) – Baseline:

python code/eeg_ldm.py --num_epoch 10 --batch_size 100 --num_workers 8 --precision bf16 --model baseline --seed 2022 --eval_every 2 --num_eval_samples 50 --use_sarhm false

#10-epoch testing (batch 100) – SAR-HM:

python code/eeg_ldm.py --num_epoch 10 --batch_size 100 --num_workers 8 --precision bf16 --model sarhm --seed 2022 --eval_every 2 --num_eval_samples 50 --use_sarhm true --ablation_mode full_sarhm
```
All of this is written in docs/thesis_grade_commands.md with the comparison table and the full command set.