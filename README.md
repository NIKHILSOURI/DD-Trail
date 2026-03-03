
## DreamDiffusion + SAR-HM: EEG-to-Image Generation

DreamDiffusion is an EEG-conditioned variant of Stable Diffusion that reconstructs visual stimuli from brain signals. This repository contains the **baseline DreamDiffusion model**, the **SAR-HM (Semantic Associative Retrieval with Hopfield Memory) extension**, and all scripts needed for training (Stage A1 + Stage B), evaluation (Stage C), and baseline vs SAR-HM comparison and plotting.

The code is organized so that:
- **Stage A1** pre-trains an EEG encoder.
- **Stage B** fine-tunes Stable Diffusion using the EEG encoder (baseline or SAR-HM).
- **Stage C** runs EEG-to-image generation and evaluation from saved checkpoints.
- **Compare-eval + graphs** use existing outputs to create thesis-quality tables and figures (no retraining needed).

---

### A. Quickstart (end-to-end, minimal commands)

All commands assume the repository root as the working directory.

#### 1) Environment (Python 3.10/3.11, GPU recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or: venv\Scripts\Activate.ps1 # Windows PowerShell

python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e ./code
```

#### 2) Data and checkpoints layout

Place the following at the **repository root**:

- `datasets/`
  - `eeg_5_95_std.pth`
  - `block_splits_by_image_single.pth` (and/or `block_splits_by_image_all.pth`)
  - optional: `imageNet_images/` (ImageNet subset used for EEG ground-truth images)
- `pretrains/models/`
  - `v1-5-pruned.ckpt` (Stable Diffusion 1.5 weights)
  - `config15.yaml` (Stable Diffusion config)
- `pretrains/eeg_pretain/checkpoint.pth`
  - EEG encoder checkpoint produced by Stage A1 (see below).

#### 3) Stage A1 – EEG encoder pre-training

```bash
python code/stageA1_eeg_pretrain.py
```

This writes a checkpoint under `results/eeg_pretrain/<timestamp>/checkpoints/checkpoint.pth`. Copy it for Stage B:

```bash
cp results/eeg_pretrain/<timestamp>/checkpoints/checkpoint.pth pretrains/eeg_pretain/checkpoint.pth
```

#### 4) Stage B – Baseline DreamDiffusion (thesis-level run)

Use `Config_Generative_Model` in `code/config.py` with `use_sarhm = False`. A typical **thesis-level** command (500 epochs, large batch, A100) is:

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

This produces:
- Training logs and configs under `results/runs/<timestamp_baseline>_baseline_2022/`.
- Checkpoints and Lightning logs under  
  `results/exps/results/generation/02-03-2026-01-49-36/` (for the run used in our final compare-eval).

#### 5) Stage B – SAR-HM (full_sarhm, thesis-level run)

In `Config_Generative_Model` (in `code/config.py`), enable SAR-HM:

- `use_sarhm = True`
- `ablation_mode = 'full_sarhm'`
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

This produces:
- Training logs and configs under `results/runs/<timestamp_sarhm>_sarhm_2022/`.
- Checkpoints and prototypes under  
  `results/exps/results/generation/02-03-2026-09-57-39/` (for the run used in our final compare-eval).

For quick smoke tests, you can reduce `--num_epoch` (e.g. 10) and/or `--batch_size` (e.g. 4–8).

#### 6) Stage C – EEG-to-image generation & evaluation

Use the Stage B checkpoints and the same dataset/splits:

```bash
python code/gen_eval_eeg.py --dataset EEG \
  --model_path results/exps/results/generation/02-03-2026-01-49-36/checkpoint_best.pth \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml
```

For SAR-HM you use the SAR-HM checkpoint and matching prototypes:

```bash
python code/gen_eval_eeg.py --dataset EEG \
  --model_path results/exps/results/generation/02-03-2026-09-57-39/checkpoint_best.pth \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --proto_path results/exps/results/generation/02-03-2026-09-57-39/prototypes.pt
```

Stage C writes evaluation outputs under `results/eval/<timestamp>/`.

#### 7) Baseline vs SAR-HM compare-eval

To re-run the side-by-side comparison and metrics:

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

This reproduces the metrics and grids under `results/compare_eval_thesis/`.

#### 8) Plotting (graphs)

Once logs and compare-eval outputs exist, you can generate graphs:

```bash
python tools/make_graphs.py \
  --runs_dir results/runs \
  --compare_dir results/compare_eval_thesis \
  --out_dir graphs

python tools/make_optional_graphs.py \
  --runs_dir results/runs \
  --results_dir results \
  --out_dir graphs/optional
```

Graphs are saved under `graphs/` and `graphs/optional/` (see **Graphs** section below).

---

### B. Results

#### 1) Final checkpoints used

For the final thesis comparison we used:

- **Baseline**  
  - Checkpoint: `results/exps/results/generation/02-03-2026-01-49-36/checkpoint_best.pth`  
  - We report metrics from this `checkpoint_best.pth`.
- **SAR-HM (full_sarhm)**  
  - Checkpoint: `results/exps/results/generation/02-03-2026-09-57-39/checkpoint_best.pth`  
  - Prototypes: `results/exps/results/generation/02-03-2026-09-57-39/prototypes.pt`  
  - Metrics are reported from this checkpoint and prototype set.

#### 2) Quantitative metrics (compare_eval_thesis)

From `results/compare_eval_thesis/metrics/metrics.csv`:

| mode     | ssim_mean | pcc_mean   | clip_sim_mean | mean_variance | n_samples |
|----------|-----------|------------|----------------|---------------|-----------|
| baseline | 0.1620    | -0.0790    | 0.5782         | 2936.12       | 5         |
| sarhm    | 0.1769    | -0.0578    | 0.5955         | 3078.70       | 5         |
| delta    | +0.0149   | +0.0212    | +0.0173        | +142.58       | 5         |

- **Interpretation:** Higher **SSIM**, **PCC**, and **CLIP similarity** are better (more similar to ground-truth images).  
  `mean_variance` acts as a proxy for image diversity/contrast across samples.

#### 3) Qualitative comparison (grids)

If present in the repository, these images are the primary qualitative evidence:

```markdown
![Baseline vs SAR-HM (side-by-side)](results/compare_eval_thesis/grids/side_by_side.png)

![Baseline grid](results/compare_eval_thesis/grids/baseline_grid.png)

![SAR-HM grid](results/compare_eval_thesis/grids/sarhm_grid.png)
```

If these files are missing (e.g. `.gitignore`d), regenerate them by re-running `code/compare_eval.py` as above and copying the resulting grids into `results/compare_eval_thesis/grids/`.

#### 4) Graphs

The following graphs are generated by `tools/make_graphs.py` and `tools/make_optional_graphs.py`:

```markdown
![Training loss](graphs/training_loss_total.png)

![Retrieval accuracy](graphs/optional/loss_curve.png)

![SAR-HM retrieval accuracy](graphs/retrieval_acc.png)

![SAR-HM attention entropy](graphs/attention_entropy.png)

![Metrics (bars)](graphs/metrics_bars.png)

![Metrics delta](graphs/metrics_delta.png)

![Variance comparison](graphs/variance_comparison.png)

![Metrics with std across seeds](graphs/optional/metrics_with_std.png)

![Ablation comparison](graphs/optional/ablation_comparison.png)
```

If `graphs/` is empty in a fresh clone, run:

```bash
python tools/make_graphs.py --runs_dir results/runs --compare_dir results/compare_eval_thesis --out_dir graphs
python tools/make_optional_graphs.py --runs_dir results/runs --results_dir results --out_dir graphs/optional
```

#### 5) Architecture diagram (placeholder)

We recommend including a diagram at `docs/architecture_diagram.png`:

```markdown
![Architecture diagram](docs/architecture_diagram.png)
```

The diagram should illustrate:

- EEG encoder (Stage A1) producing latent representations.
- Baseline DreamDiffusion conditioning path: EEG → MAE → channel/latent mapper → Stable Diffusion UNet cross-attention.
- SAR-HM extension: projection into CLIP space, Hopfield retrieval over class prototypes, confidence-gated fusion with baseline conditioning, and adapter back into SD conditioning space.
- Where Stage B (training) and Stage C (generation) touch the pipeline.

---

### C. Repository structure (high-level)

Key directories and what they contain:

- `code/`
  - Core training and evaluation scripts:
    - `stageA1_eeg_pretrain.py` – Stage A1 EEG encoder pre-training.
    - `eeg_ldm.py` – Stage B LDM finetuning (baseline or SAR-HM).
    - `gen_eval_eeg.py` – Stage C EEG-to-image generation + evaluation.
    - `compare_eval.py` – baseline vs SAR-HM compare-eval (metrics + grids).
- `pretrains/`
  - `models/` – Stable Diffusion v1-5 checkpoint (`v1-5-pruned.ckpt`) + `config15.yaml`.
  - `eeg_pretain/` – EEG encoder checkpoint used for Stage B.
- `datasets/`
  - EEG tensors and splits; optionally ImageNet images.
- `results/`
  - `runs/` – per-run folders with `config.json` + `train_log.csv` (baseline, sarhm, etc.).
  - `exps/results/generation/` – Stage B checkpoints, Lightning logs, and SAR-HM prototypes.
  - `compare_eval_thesis/` – compare-eval metrics (`metrics/metrics.csv`) and grids (`grids/*.png`).
- `Training eval/`
  - `Baseline Outputs/` and `SARHM Outputs/` – per-image qualitative outputs for many test items.
- `graphs/` and `graphs/optional/`
  - Generated figures from `tools/make_graphs.py` and `tools/make_optional_graphs.py`.
- `docs/`
  - `SARHM_README.md` – SAR-HM configuration, ablations, and dataset usage policy.
  - `explain.md` – narrative explanation of the pipeline and debugging notes.

For more detail, see the existing sections below and the SAR-HM-specific docs.

---

### D. Reproducibility notes

To reproduce the reported results:

- Use **seed 2022** across training and evaluation:
  - `--seed 2022` in Stage B; `--seed 2022` or default in Stage C and compare-eval.
- Use the same **test split** and dataset files:
  - `datasets/eeg_5_95_std.pth`
  - `datasets/block_splits_by_image_single.pth`
  - `datasets/imageNet_images/` for the ImageNet-EEG setting.
- Use the same **diffusion settings** in Stage C and compare-eval:
  - `ddim_steps = 250`
  - `n_samples = 5` (per EEG item).
- For SAR-HM, always pair the checkpoint with its **matching prototypes**:
  - Example: `results/exps/results/generation/02-03-2026-09-57-39/checkpoint_best.pth` must be used with  
    `results/exps/results/generation/02-03-2026-09-57-39/prototypes.pt`.
- Stage B validation is intentionally lightweight (fewer items and steps); all **final quality claims** (metrics + grids) come from **Stage C** and **compare_eval**.

If you change the number of epochs, batch size, or precision, you will not exactly reproduce the thesis runs but the pipeline will remain valid.

---

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
┗  📂 tools
   ┗ 📜 make_graphs.py          (generate thesis-quality graphs from existing logs)
   ┗ 📜 make_optional_graphs.py (generate advanced / optional evaluation graphs)

### Graphs / plots

To generate thesis-quality plots comparing the **Baseline** and **SAR-HM** models from existing logs and evaluation outputs, run:

```bash
python tools/make_graphs.py --runs_dir runs --compare_dir results/compare_eval_thesis --out_dir graphs
```

If you run this from the repo root with the standard structure (`runs/` and `results/compare_eval_thesis/` present), all flags are optional. Figures are saved into `graphs/` as PNG (and, when possible, PDF). The script automatically looks for:

- Baseline and SAR-HM `train_log*.csv` files under `runs/`
- Compare-eval metrics under `results/compare_eval_thesis/metrics/metrics.csv` (or any `metrics.csv` there)
- Existing qualitative grids (`baseline_grid.png`, `sarhm_grid.png`, `side_by_side.png`) under `results/compare_eval_thesis/grids/`

For additional **optional** evaluations (training dynamics, multi-seed statistics, ablations), run:

```bash
python tools/make_optional_graphs.py --runs_dir runs --results_dir results --out_dir graphs/optional
```

This reads the same logs and any `results/**/metrics.csv` files and writes advanced plots (loss curves, SAR-HM retrieval/entropy curves, metrics with standard deviation across seeds, and ablation comparisons) into `graphs/optional/`.
┗  📜 eval_metrics.py           (functions for evaluation metrics)
┗  📜 config.py                 (configurations for the main scripts)

┣  📂 docs
┃   ┗  📜 SARHM_README.md       (SAR-HM config, ablations, dataset policy)
┃   ┗  📜 logging.md            (thesis logging: metrics, run layout, eval-only)

```

---
