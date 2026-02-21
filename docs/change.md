# Running DreamDiffusion on Arch Linux

This document describes what to change or keep in mind when running the existing DreamDiffusion project on **Arch Linux** (or other Linux distributions).

---

## 1. No code changes required (already portable)

The project is already written in a portable way:

- **Paths**: `code/config.py` uses `pathlib.Path` and `os.environ` for `DREAMDIFFUSION_DATA_ROOT` and `DREAMDIFFUSION_PRETRAIN_ROOT`. No hardcoded Windows drive letters or backslashes.
- **Data/pretrain locations**: Default to `./datasets` and `./pretrains` relative to the repo root; you can override with env vars if needed.
- **Python**: Standard library and cross-platform packages only in the main code paths.

So you do **not** need to modify Python source files for Arch Linux.

---

## 2. What you need to change: environment and shell commands

### 2.1 Python and virtual environment

| On Windows (README) | On Arch Linux |
|---------------------|----------------|
| `py -3.11 -m venv venv` | `python -m venv venv` or `python3 -m venv venv` |
| `venv\Scripts\Activate.ps1` (PowerShell) | `source venv/bin/activate` |
| `venv\Scripts\activate.bat` (CMD) | (same: `source venv/bin/activate`) |
| `python` in commands | Use `python` or `python3` (ensure the venv is activated so it’s the same interpreter) |

**Arch:** Install Python and venv if needed:

```bash
sudo pacman -S python python-pip
# Create and activate venv from repo root
cd /path/to/DREAMDIFFUSION
python -m venv venv
source venv/bin/activate
```

### 2.2 Installing dependencies

Use the same dependency list; only the **shell** and **PyTorch** install differ.

**Arch Linux (from repo root, with venv activated):**

```bash
pip install --upgrade pip setuptools wheel
# PyTorch with CUDA (Arch / Linux) – pick the index that matches your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Or CPU-only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e ./code
```

- Ignore Windows-only helpers such as `scripts\install_venv_deps.bat`; use the `pip` commands above instead.
- If `scipy` or other build deps fail, install Arch build packages, e.g.:

  ```bash
  sudo pacman -S base-devel openblas
  ```

### 2.3 Copying files (post-training)

| On Windows (README) | On Arch Linux |
|---------------------|----------------|
| `copy results\eeg_pretrain\<timestamp>\checkpoints\checkpoint.pth pretrains\eeg_pretain\checkpoint.pth` | `cp results/eeg_pretrain/<timestamp>/checkpoints/checkpoint.pth pretrains/eeg_pretain/checkpoint.pth` |
| `copy results\generation\<timestamp>\checkpoint.pth pretrains\generation\checkpoint_best.pth` | `cp results/generation/<timestamp>/checkpoint.pth pretrains/generation/checkpoint_best.pth` |

Use forward slashes and `cp`; replace `<timestamp>` with your actual folder name.

### 2.4 Running the pipeline

Commands are the same; run them from the **repository root** with the venv **activated**:

```bash
source venv/bin/activate

# Stage A1
python code/stageA1_eeg_pretrain.py

# Stage B (after copying Stage A1 checkpoint to pretrains/eeg_pretain/)
python code/eeg_ldm.py

# Stage C (replace <timestamp> with your Stage B output folder)
python code/gen_eval_eeg.py --dataset EEG --model_path results/generation/<timestamp>/checkpoint.pth --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
```

Use `python` or `python3` consistently (venv ensures it’s the same).

**10 epochs + tiny dataset (baseline vs SAR-HM):** See README “10 epochs + tiny dataset: baseline vs SAR-HM”. On Arch use the same Python commands; paths use `/`, copy uses `cp`:

```bash
python code/make_tiny_splits.py
# Baseline
python code/eeg_ldm.py --num_epoch 10 --splits_path datasets/block_splits_tiny.pth --val_gen_limit 2 --use_sarhm false
# SAR-HM
python code/eeg_ldm.py --num_epoch 10 --splits_path datasets/block_splits_tiny.pth --val_gen_limit 2 --use_sarhm true --ablation_mode full_sarhm
# Stage C: replace <timestamp> with the run’s folder under results/generation/
python code/gen_eval_eeg.py --dataset EEG --model_path results/generation/<timestamp>/checkpoint.pth --splits_path datasets/block_splits_tiny.pth --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml
```

### 2.5 Preflight / smoke test

```bash
python code/ci_preflight.py
# Or smoke only:
python code/ci_preflight.py --smoke_only
```

---

## 3. Optional: faster data loading on Linux

The code uses `num_workers=0` in several DataLoaders (to avoid Windows multiprocessing issues). On Linux you can safely use multiple workers for faster training.

- **Stage A1** (`code/stageA1_eeg_pretrain.py`): Already uses `num_workers = 4` when CUDA is available and sets `persistent_workers=(num_workers > 0 and os.name != 'nt')`, so it will use 4 workers on Linux.
- **Stage B** (`code/dc_ldm/ldm_for_eeg.py`), **eeg_ldm** (`code/eeg_ldm.py`), **ci_preflight** (`code/ci_preflight.py`): Use `num_workers=0`. If you want to try faster loading on Arch, you can change those to e.g. `num_workers=2` or `4` only in those files; not required for correctness.

---

## 4. Arch-specific system packages (if needed)

If `pip install -r requirements.txt` fails on some packages (e.g. build errors):

```bash
sudo pacman -S base-devel openblas python-numpy
```

For CUDA / GPU support, install NVIDIA drivers and optionally CUDA:

```bash
# NVIDIA driver (if using NVIDIA GPU)
sudo pacman -S nvidia nvidia-utils
# Optional: CUDA toolkit (PyTorch often ships with its own CUDA libs)
# sudo pacman -S cuda
```

Reboot after installing the driver if the GPU is not detected.

---

## 5. Summary table

| Topic | Windows (current docs) | Arch Linux |
|-------|------------------------|------------|
| Create venv | `py -3.11 -m venv venv` | `python -m venv venv` |
| Activate venv | `venv\Scripts\Activate.ps1` or `activate.bat` | `source venv/bin/activate` |
| Copy command | `copy src\path dest\path` | `cp src/path dest/path` |
| Path separators | `\` (backslash) | `/` (forward slash) |
| Python executable | `python` | `python` or `python3` (with venv active) |
| PyTorch install | `--index-url .../cu118` (or cpu) | Same; use Linux/CUDA or CPU wheel |
| Code changes | — | None required |
| Data/pretrain layout | `datasets/`, `pretrains/` at repo root | Same; optional env vars for custom paths |

---

## 6. Optional env vars (any OS)

You can override data and pretrain roots without changing code:

```bash
export DREAMDIFFUSION_DATA_ROOT=/path/to/datasets
export DREAMDIFFUSION_PRETRAIN_ROOT=/path/to/pretrains
```

Then run the same Python commands as above. Useful if you keep data outside the repo on Arch.

---

**Bottom line:** Use `source venv/bin/activate`, `cp` and `/` in shell commands, and install PyTorch/deps with `pip` as on Windows. No edits to the project’s Python files are needed for Arch Linux.
