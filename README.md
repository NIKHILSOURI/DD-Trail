# DreamDiffusion + SAR-HM: EEG-to-Image Generation

DreamDiffusion is an **EEG-conditioned** variant of Stable Diffusion that reconstructs visual stimuli from brain signals. This repository contains:

- **Baseline DreamDiffusion** (EEG → conditioning → SD 1.5 UNet)
- **SAR-HM** (*Semantic Associative Retrieval with Hopfield Memory*) extension
- End-to-end scripts for **training** (Stage A1 + Stage B), **generation/evaluation** (Stage C), and **baseline vs SAR-HM comparison + plotting** (no retraining needed once results exist)

> **Note:** `datasets/` and `pretrains/` are not included in the repo. See **Data & Pretrained Weights** below.

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Quickstart](#quickstart)
- [Stages](#stages)
  - [Stage A1: EEG Encoder Pretraining](#stage-a1-eeg-encoder-pretraining)
  - [Stage B: Fine-tuning Stable Diffusion](#stage-b-fine-tuning-stable-diffusion)
  - [Stage C: EEG-to-Image Generation & Evaluation](#stage-c-eeg-to-image-generation--evaluation)
  - [Compare-Eval: Baseline vs SAR-HM](#compare-eval-baseline-vs-sar-hm)
  - [Graphs: Thesis-Quality Plots](#graphs-thesis-quality-plots)
- [Results (Thesis Runs)](#results-thesis-runs)
- [Repository Structure](#repository-structure)
- [Data & Pretrained Weights](#data--pretrained-weights)
- [Reproducibility Checklist](#reproducibility-checklist)
- [Switching Modes (Baseline vs SAR-HM)](#switching-modes-baseline-vs-sar-hm)
- [Architecture Diagram Placeholder](#architecture-diagram-placeholder)

---

## Pipeline Overview

**Stages**

- **Stage A1**: Pre-train an EEG encoder (masked modeling / representation learning).
- **Stage B**: Fine-tune a Latent Diffusion Model (Stable Diffusion 1.5 backbone) using EEG conditioning:
  - **Baseline**: original DreamDiffusion EEG-conditioning path
  - **SAR-HM**: adds CLIP-space projection + Hopfield retrieval over class prototypes + confidence-gated fusion
- **Stage C**: Generate images from EEG and compute metrics using saved checkpoints.
- **Compare-eval + graphs**: Create **final tables/figures** from existing outputs (no retraining).

---

## Quickstart

All commands assume you are in the **repository root**.

### 1) Environment (Python 3.10/3.11, GPU recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or: venv\Scripts\Activate.ps1 # Windows PowerShell

python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e ./code
```

### 2) Data & checkpoint layout (required paths)

Place the following at the **repository root**:

- `datasets/`
  - `eeg_5_95_std.pth`
  - `block_splits_by_image_single.pth` (and/or `block_splits_by_image_all.pth`)
  - optional: `imageNet_images/` (ImageNet subset used for EEG ground-truth images)
- `pretrains/models/`
  - `v1-5-pruned.ckpt` (Stable Diffusion 1.5 weights)
  - `config15.yaml` (Stable Diffusion config)
- `pretrains/eeg_pretain/checkpoint.pth`
  - EEG encoder checkpoint produced by Stage A1 (see below)

---

## Stages

### Stage A1: EEG Encoder Pretraining

```bash
python code/stageA1_eeg_pretrain.py
```

Output (example):

- `results/eeg_pretrain/<timestamp>/checkpoints/checkpoint.pth`

Copy it into the standard path used by Stage B:

```bash
cp results/eeg_pretrain/<timestamp>/checkpoints/checkpoint.pth pretrains/eeg_pretain/checkpoint.pth
```

---

### Stage B: Fine-tuning Stable Diffusion

Stage B is run via:

- `python code/eeg_ldm.py ...`

This produces:

- `results/runs/<timestamp>_<mode>_<seed>/` (configs + `train_log.csv`)
- `results/exps/results/generation/<timestamp>/` (checkpoints, Lightning logs, SAR-HM prototypes)

#### Baseline DreamDiffusion (thesis-level)

In `code/config.py`, keep:

- `use_sarhm = False`

Example thesis-level command:

```bash
python code/eeg_ldm.py \
  --num_epoch 500 \
  --batch_size 100 \
  --num_workers 8 \
  --precision bf16 \
  --model baseline \
  --seed 2022 \
  --eval_every 2 \
  --num_eval_samples 50 \
  --use_sarhm false
```

#### SAR-HM (full_sarhm, thesis-level)

In `code/config.py` set:

- `use_sarhm = True`
- `ablation_mode = "full_sarhm"`
- `num_classes = 40`

Then run:

```bash
python code/eeg_ldm.py \
  --num_epoch 500 \
  --batch_size 100 \
  --num_workers 8 \
  --precision bf16 \
  --model sarhm \
  --seed 2022 \
  --eval_every 2 \
  --num_eval_samples 50 \
  --use_sarhm true \
  --ablation_mode full_sarhm
```

> For smoke tests: lower `--num_epoch` (e.g., `10`) and/or `--batch_size` (e.g., `4–8`).

---

### Stage C: EEG-to-Image Generation & Evaluation

Baseline Stage C:

```bash
python code/gen_eval_eeg.py --dataset EEG \
  --model_path results/exps/results/generation/02-03-2026-01-49-36/checkpoint_best.pth \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml
```

SAR-HM Stage C (checkpoint + matching prototypes):

```bash
python code/gen_eval_eeg.py --dataset EEG \
  --model_path results/exps/results/generation/02-03-2026-09-57-39/checkpoint_best.pth \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --proto_path results/exps/results/generation/02-03-2026-09-57-39/prototypes.pt
```

Outputs:

- `results/eval/<timestamp>/...`

---

### Compare-Eval: Baseline vs SAR-HM

Re-run side-by-side comparison (metrics + grids):

```bash
python code/compare_eval.py --dataset EEG \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --baseline_ckpt results/exps/results/generation/02-03-2026-01-49-36/checkpoint_best.pth \
  --sarhm_ckpt   results/exps/results/generation/02-03-2026-09-57-39/checkpoint_best.pth \
  --sarhm_proto  results/exps/results/generation/02-03-2026-09-57-39/prototypes.pt \
  --n_samples 5 \
  --ddim_steps 250 \
  --seed 2022 \
  --out_dir results/compare_eval_thesis
```

Outputs:

- `results/compare_eval_thesis/metrics/metrics.csv`
- `results/compare_eval_thesis/grids/*.png`

---

### Graphs: Thesis-Quality Plots

Generate main thesis plots:

```bash
python tools/make_graphs.py \
  --runs_dir results/runs \
  --compare_dir results/compare_eval_thesis \
  --out_dir graphs
```

Generate optional / advanced plots:

```bash
python tools/make_optional_graphs.py \
  --runs_dir results/runs \
  --results_dir results \
  --out_dir graphs/optional
```

---

## Results (Thesis Runs)

### Final checkpoints used

For the final thesis comparison:

- **Baseline**
  - `results/exps/results/generation/02-03-2026-01-49-36/checkpoint_best.pth`
- **SAR-HM (full_sarhm)**
  - `results/exps/results/generation/02-03-2026-09-57-39/checkpoint_best.pth`
  - `results/exps/results/generation/02-03-2026-09-57-39/prototypes.pt`

### Quantitative metrics (compare_eval_thesis)

From `results/compare_eval_thesis/metrics/metrics.csv`:

| mode     | ssim_mean | pcc_mean  | clip_sim_mean | mean_variance | n_samples |
|----------|-----------|-----------|---------------|---------------|-----------|
| baseline | 0.1620    | -0.0790   | 0.5782        | 2936.12       | 5         |
| sarhm    | 0.1769    | -0.0578   | 0.5955        | 3078.70       | 5         |
| delta    | +0.0149   | +0.0212   | +0.0173       | +142.58       | 5         |

**Interpretation:** higher **SSIM**, **PCC**, and **CLIP similarity** are better (closer to ground-truth).  
`mean_variance` is a proxy for diversity/contrast across samples.

---

## Qualitative Evidence (Grids)

These are the primary qualitative grids produced by `compare_eval.py`.

> If these images are missing (e.g., `.gitignore`), regenerate them by re-running `code/compare_eval.py`.

```markdown
![Baseline vs SAR-HM (side-by-side)](results/compare_eval_thesis/grids/side_by_side.png)

![Baseline grid](results/compare_eval_thesis/grids/baseline_grid.png)

![SAR-HM grid](results/compare_eval_thesis/grids/sarhm_grid.png)
```

### Rendered (if files exist)

![Baseline vs SAR-HM (side-by-side)](results/compare_eval_thesis/grids/side_by_side.png)

![Baseline grid](results/compare_eval_thesis/grids/baseline_grid.png)

![SAR-HM grid](results/compare_eval_thesis/grids/sarhm_grid.png)

---

## Graphs Showcase

Main graphs produced under `graphs/`:

```markdown
![Training loss](graphs/training_loss_total.png)
![SAR-HM retrieval accuracy](graphs/retrieval_acc.png)
![SAR-HM attention entropy](graphs/attention_entropy.png)
![Metrics (bars)](graphs/metrics_bars.png)
![Metrics delta](graphs/metrics_delta.png)
![Variance comparison](graphs/variance_comparison.png)
```

Optional graphs under `graphs/optional/`:

```markdown
![Loss curve](graphs/optional/loss_curve.png)
![Metrics with std across seeds](graphs/optional/metrics_with_std.png)
![Ablation comparison](graphs/optional/ablation_comparison.png)
```

### Rendered (if files exist)

![Training loss](graphs/training_loss_total.png)

![SAR-HM retrieval accuracy](graphs/retrieval_acc.png)

![SAR-HM attention entropy](graphs/attention_entropy.png)

![Metrics (bars)](graphs/metrics_bars.png)

![Metrics delta](graphs/metrics_delta.png)

![Variance comparison](graphs/variance_comparison.png)

![Loss curve](graphs/optional/loss_curve.png)

![Metrics with std across seeds](graphs/optional/metrics_with_std.png)

![Ablation comparison](graphs/optional/ablation_comparison.png)

---

## Repository Structure

High-level structure (key files only):

```text
.
├── code/
│   ├── stageA1_eeg_pretrain.py        # Stage A1 (EEG encoder pretraining)
│   ├── eeg_ldm.py                     # Stage B (baseline or SAR-HM training)
│   ├── gen_eval_eeg.py                # Stage C (generation + evaluation)
│   ├── compare_eval.py                # Baseline vs SAR-HM metrics + grids
│   ├── dataset.py                     # Dataset loading helpers
│   ├── eval_metrics.py                # Metrics utilities
│   ├── config.py                      # Main configuration (including SAR-HM flags)
│   ├── sc_mbm/
│   │   ├── mae_for_eeg.py
│   │   ├── trainer.py
│   │   └── utils.py
│   ├── sarhm/
│   │   ├── sarhm_modules.py           # Hopfield retrieval + gating modules
│   │   ├── prototypes.py              # Prototype creation/IO
│   │   ├── metrics_logger.py          # Logging hooks (retrieval acc, entropy, etc.)
│   │   └── vis.py                     # Visualizations
│   └── dc_ldm/
│       ├── ldm_for_eeg.py
│       ├── utils.py
│       ├── models/                    # adopted from LDM
│       └── modules/                   # adopted from LDM
│
├── tools/
│   ├── make_graphs.py                 # main thesis graphs from logs + compare_eval outputs
│   └── make_optional_graphs.py        # optional/advanced plots (ablations, std across seeds)
│
├── docs/
│   ├── SARHM_README.md                # SAR-HM configuration + dataset policy
│   ├── explain.md                     # narrative explanations / debugging notes
│   └── architecture_diagram.png       # (recommended) architecture figure used in README
│
├── datasets/                          # NOT included in repo (download separately)
│   ├── eeg_5_95_std.pth
│   ├── block_splits_by_image_single.pth
│   ├── block_splits_by_image_all.pth
│   └── imageNet_images/               # optional ImageNet subset
│
├── pretrains/                         # NOT included in repo (download separately)
│   ├── models/
│   │   ├── config15.yaml
│   │   └── v1-5-pruned.ckpt
│   ├── eeg_pretain/
│   │   └── checkpoint.pth             # Stage A1 EEG encoder checkpoint
│   └── sarhm/                         # optional; created/used when SAR-HM enabled
│       └── prototypes_dummy.pt        # dummy prototypes (if none provided)
│
├── results/                           # generated outputs (training, eval, compare-eval)
│   ├── runs/
│   │   └── <timestamp>_<mode>_<seed>/
│   │       ├── config.json
│   │       └── train_log.csv
│   ├── exps/results/generation/
│   │   └── <timestamp>/
│   │       ├── checkpoint_best.pth
│   │       ├── prototypes.pt          # SAR-HM only
│   │       └── lightning_logs/...
│   ├── eval/
│   │   └── <timestamp>/...
│   └── compare_eval_thesis/
│       ├── metrics/metrics.csv
│       └── grids/
│           ├── side_by_side.png
│           ├── baseline_grid.png
│           └── sarhm_grid.png
│
├── graphs/                            # generated figures (main)
│   ├── training_loss_total.png
│   ├── retrieval_acc.png
│   ├── attention_entropy.png
│   ├── metrics_bars.png
│   ├── metrics_delta.png
│   └── variance_comparison.png
│
└── graphs/optional/                   # generated figures (optional/advanced)
    ├── loss_curve.png
    ├── metrics_with_std.png
    └── ablation_comparison.png
```

---

## Data & Pretrained Weights

### EEG datasets / splits

Download EEG tensors + split files from the EEG Visual Classification repo and place them under `datasets/`:

- EEG repo: https://github.com/perceivelab/eeg_visual_classification

Optional: ImageNet subset (for ImageNet-EEG) provided via Drive:

- ImageNet subset (Drive): https://drive.google.com/file/d/1y7I9bG1zKYqBM94odcox_eQjnP9HGo9-/view?usp=drive_link

> If you cannot access Drive from your environment, download locally and upload/mount into your compute instance.

### Stable Diffusion 1.5

Download SD 1.5 weights and config and place them under:

- `pretrains/models/config15.yaml`
- `pretrains/models/v1-5-pruned.ckpt`

Recommended source:

- RunwayML SD 1.5 (Hugging Face): https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main

---

## Reproducibility Checklist

To reproduce the reported thesis results:

- Use **seed 2022** across training and evaluation:
  - `--seed 2022` in Stage B
  - `--seed 2022` in Stage C / compare-eval
- Use the same dataset files:
  - `datasets/eeg_5_95_std.pth`
  - `datasets/block_splits_by_image_single.pth`
  - `datasets/imageNet_images/` for ImageNet-EEG (if used)
- Use the same diffusion evaluation settings:
  - `--ddim_steps 250`
  - `--n_samples 5` (per EEG item) for compare-eval
- **SAR-HM requires matching prototypes**:
  - Example: `.../02-03-2026-09-57-39/checkpoint_best.pth` must be paired with  
    `.../02-03-2026-09-57-39/prototypes.pt`

---

## Switching Modes (Baseline vs SAR-HM)

### Baseline DreamDiffusion

- In `code/config.py`: `use_sarhm = False`
- Training and generation use:  
  EEG → MAE → channel/latent mapper → Stable Diffusion conditioning

### SAR-HM

- In `code/config.py`: `use_sarhm = True`
- Choose SAR-HM ablation mode (`Config_Generative_Model.ablation_mode`):
  - `projection_only` – EEG → projection → adapter → SD
  - `hopfield_no_gate` – add Hopfield retrieval, no gating
  - `full_sarhm` – Hopfield + confidence-gated fusion (recommended)

**Important:** Ensure you pass the same config into both:
- `eLDM(..., main_config=config)` for training
- `eLDM_eval(..., main_config=config)` for evaluation

> The Stable Diffusion stack (UNet, VAE, text encoder) is **not** fine-tuned; only EEG encoder + SAR-HM modules are trained when enabled.

---

## Architecture Diagram Placeholder

Add your final architecture diagram at:

- `docs/architecture_diagram.png`

Then it will render here:

![Architecture diagram](docs/architecture_diagram.png)

---

## Notes

- If `graphs/` is empty in a fresh clone, run:
  ```bash
  python tools/make_graphs.py --runs_dir results/runs --compare_dir results/compare_eval_thesis --out_dir graphs
  python tools/make_optional_graphs.py --runs_dir results/runs --results_dir results --out_dir graphs/optional
  ```
- If `results/compare_eval_thesis/grids/` images are missing, regenerate with `code/compare_eval.py` (see above).
