
## Quality test: both approaches, minimal images, best quality

Use these to compare **baseline** vs **SAR-HM** with the **fewest generated images** and **best image quality** (250 DDIM steps).

### Stage B (training)

**Baseline (no SAR-HM) – 1 val image per epoch, 250 steps:**

```bash
python code/eeg_ldm.py --num_epoch 5 --use_sarhm false --ablation_mode baseline \
  --pretrain_mbm_path pretrains/eeg_pretain/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth \
  --pretrain_gm_path pretrains --check_val_every_n_epoch 1 --batch_size 4 --num_workers 8 \
  --val_image_gen_every_n_epoch 1 --val_gen_limit 1 --val_num_samples 1 --val_ddim_steps 250
```

**SAR-HM – 1 val image per epoch, 250 steps (prototypes built from train set):**

```bash
python code/eeg_ldm.py --num_epoch 5 --use_sarhm true --ablation_mode full_sarhm --proto_source baseline_centroids \
  --normalize_conditioning true --pretrain_mbm_path pretrains/eeg_pretain/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth \
  --pretrain_gm_path pretrains --check_val_every_n_epoch 1 --batch_size 4 --num_workers 8 \
  --val_image_gen_every_n_epoch 1 --val_gen_limit 1 --val_num_samples 1 --val_ddim_steps 250
```

### Stage C (inference)

Use the checkpoint (and same-folder `prototypes.pt` for SAR-HM) from your Stage B run. Config default is `ddim_steps=250` for best quality.

**Baseline – minimal images (5 test items × 1 sample = 5 images), 250 steps:**

```bash
python code/gen_eval_eeg.py --dataset EEG --model_path exps/results/generation/<timestamp>/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml --split test --num_samples 1 --max_test_items 5 --no_sarhm
```

**SAR-HM – minimal images (5 test items × 1 sample = 5 images), 250 steps (prototypes auto-loaded from checkpoint dir):**

```bash
python code/gen_eval_eeg.py --dataset EEG --model_path exps/results/generation/<timestamp>/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml --split test --num_samples 1 --max_test_items 5
```

- **Stage B:** `val_gen_limit=1`, `val_num_samples=1`, `val_ddim_steps=250` → one high-quality val image per epoch.
- **Stage C:** `num_samples=1` (one image per EEG), `max_test_items=5` → five test images total; checkpoint’s `ddim_steps` (default 250) gives best quality.
- For even fewer Stage C images, use e.g. `--max_test_items 1` (single image).

---

## Determining if SAR-HM is better than baseline (after Stage B)

After running Stage B you can compare **baseline** vs **SAR-HM** by running **Stage C** for each checkpoint with the **same test setup** and comparing metrics.

### 1. Train both variants (Stage B)

- **Baseline run:** e.g. `exps/results/generation/<timestamp_baseline>/`  
  Use `--use_sarhm false --ablation_mode baseline`.
- **SAR-HM run:** e.g. `exps/results/generation/<timestamp_sarhm>/`  
  Use `--use_sarhm true --ablation_mode full_sarhm --proto_source baseline_centroids` (and `--normalize_conditioning true`).

### 2. Run Stage C with the same test setup

Use the **same** `--split test`, `--num_samples`, `--max_test_items`, and same data paths so the comparison is fair.

```bash
# Baseline checkpoint (add --no_sarhm so Stage C uses baseline conditioning)
python code/gen_eval_eeg.py --dataset EEG --model_path exps/results/generation/<timestamp_baseline>/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml --split test --num_samples 1 --max_test_items 20 --no_sarhm

# SAR-HM checkpoint (prototypes.pt auto-loaded from same folder)
python code/gen_eval_eeg.py --dataset EEG --model_path exps/results/generation/<timestamp_sarhm>/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml --split test --num_samples 1 --max_test_items 20
```

### 3. Read the metrics

Stage C prints **image-quality metrics** after test generation:

- **`[EVAL] mse=... pcc=... ssim=... psm=... top-1-class=... top-1-class (max)=...`**

Interpretation:

- **Higher is better:** `pcc`, `ssim`, `top-1-class` (and `top-1-class (max)`).  
  **Lower is better:** `mse`, `psm`.

So **SAR-HM is better than baseline** on this run if, for the same test setup, it has **higher** PCC/SSIM/top-1-class and/or **lower** MSE/PSM.

### 4. Ablation CSV

Each Stage C run appends a row to **`<output_path>/ablation_results.csv`** with `mode` (e.g. `baseline` or `full_sarhm`) and the computed metrics (ssim, pcc, retrieval_acc). You can open the CSV from the two runs (or from `results/eval/<timestamp>/`) and compare the rows side by side.

### Summary

| Step | What to do |
|------|------------|
| 1 | Run Stage B for **baseline** and for **SAR-HM** (two separate runs). |
| 2 | Run Stage C for **each** checkpoint with the **same** test split and limits. |
| 3 | Compare the **`[EVAL]`** line or **ablation_results.csv**: higher PCC/SSIM/top-1-class and lower MSE/PSM ⇒ better. |
| 4 | If SAR-HM improves those metrics over baseline on the same test set, SAR-HM is better for this setup. |

---

## Dual evaluation: BASELINE vs SAR-HM in one run (compare_eval.py)

**`code/compare_eval.py`** runs both models on the **same dataset, seed, and samples**, generates exactly **N=20** test images per mode, and writes comparison grids + a single metrics CSV and report.

### Options

| Argument | Description |
|----------|-------------|
| `--baseline_ckpt` | Path to baseline Stage-B checkpoint |
| `--sarhm_ckpt` | Path to SAR-HM Stage-B checkpoint |
| `--sarhm_proto` | Optional: path to `prototypes.pt` for SAR-HM. If missing and SAR-HM needs it: use `--fail_if_proto_missing` to error, or leave unset for alpha=0 fallback |
| `--n_samples` | Number of test images per mode (default 20) |
| `--ddim_steps` | DDIM steps (default 250) |
| `--out_dir` | Output directory (default `results/compare_eval/<timestamp>`) |
| `--use_train_split` | Use train split instead of test |
| `--imagenet_path` | Optional: ImageNet root for real GT images; if unset, CLIP/SSIM/PCC use dataset image or NA |

### Output layout

- `out_dir/baseline/samples/*.png` – baseline images
- `out_dir/sarhm/samples/*.png` – SAR-HM images  
- `out_dir/grids/baseline_grid.png`, `sarhm_grid.png`, `side_by_side.png` (each row: baseline left, SAR-HM right, same sample)
- `out_dir/metrics/metrics.csv` – one row per mode + delta row
- `out_dir/metrics/report.md` – human-readable summary

### Example

```bash
python code/compare_eval.py --dataset EEG \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --baseline_ckpt exps/results/generation/<baseline_run>/checkpoint.pth \
  --sarhm_ckpt exps/results/generation/<sarhm_run>/checkpoint.pth \
  --sarhm_proto exps/results/generation/<sarhm_run>/prototypes.pt \
  --n_samples 20 --ddim_steps 250 --seed 2022 --out_dir results/compare_eval
```

Logs use prefixes `[COMPARE] [BASELINE]`, `[COMPARE] [SARHM]`, `[COMPARE] [PROTO]`, and `[SCALE_FACTOR]` / `[CKPT_LOAD]` for debugging.

---

## Stage B: correct checkpoint paths (MAE pretrain vs Stage-B resume)

- **`--pretrain_mbm_path`** – Path to the **EEG encoder / MAE pretrain** checkpoint (e.g. `pretrains/eeg_pretain/checkpoint.pth`). Required for Stage B unless resuming from a full run that already has the encoder loaded.
- **`--resume_ckpt_path`** – Path to a **Stage-B LatentDiffusion** checkpoint to **resume** training. Do **not** point this at the MAE pretrain; use `--pretrain_mbm_path` for that.
- **`--checkpoint_path`** is **deprecated** but still supported: if the file looks like an MAE pretrain (has `config`/`state`, MAE keys, no LDM keys), it is treated as `--pretrain_mbm_path`; otherwise as `--resume_ckpt_path`. You will see `[ARGS]` warnings; prefer the explicit args.

## Stage B: 5-epoch quality check (correct-by-default)

Stage B generates validation images every epoch when not disabled. Defaults: `disable_image_generation_in_val=False`, `val_image_gen_every_n_epoch=1`, `check_val_every_n_epoch=5`, `val_gen_limit=2`, `val_ddim_steps=50`, `val_num_samples=2`.

**4) Baseline 5-epoch quality check (no SAR-HM)**

```bash
python code/eeg_ldm.py --num_epoch 5 --use_sarhm false --ablation_mode baseline \
  --pretrain_mbm_path pretrains/eeg_pretain/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth \
  --pretrain_gm_path pretrains --check_val_every_n_epoch 1 --batch_size 4 --num_workers 8
```

**5) Full SAR-HM 5-epoch quality check (prototypes built from train set)**

```bash
python code/eeg_ldm.py --num_epoch 5 --use_sarhm true --ablation_mode full_sarhm --proto_source baseline_centroids \
  --normalize_conditioning true --pretrain_mbm_path pretrains/eeg_pretain/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth \
  --pretrain_gm_path pretrains --check_val_every_n_epoch 1 \
  --disable_image_generation_in_val false --val_image_gen_every_n_epoch 1 --val_gen_limit 1 --val_num_samples 1 \
  --val_ddim_steps 250 --batch_size 4 --num_workers 8
```

**6) Smoke test (minimal run: 1 val item, 4 samples, 25 steps, val every epoch)**

```bash
python code/eeg_ldm.py --dataset EEG --pretrain_mbm_path pretrains/eeg_pretain/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth \
  --pretrain_gm_path pretrains --num_epoch 3 --smoke_test
```

**Resuming Stage B** (continue from a previous run):

```bash
python code/eeg_ldm.py ... --resume_ckpt_path exps/results/generation/<timestamp>/checkpoint.pth \
  --pretrain_mbm_path pretrains/eeg_pretain/checkpoint.pth ...
```

---

## Best quality quickly: cond_ln + alpha warmup

For **best image quality quickly** you want the conditioning LayerNorm (`cond_ln`) to be trained and α to stay small early so SAR-HM doesn’t overpower the baseline. You can do either of the following.

### Option 1: Resume from a checkpoint that already has cond_ln

Use a **Stage B** run that was trained with **`normalize_conditioning=true`** (default) for **at least ~10–15 epochs**. That checkpoint will contain trained `cond_stage_model.cond_ln` weights. Then **resume** from it (e.g. to fine-tune or extend training) so you start with a good cond_ln and don’t need to train it from scratch.

```bash
# Resume from a run that already has cond_ln (e.g. 01-03-2026-18-00-04 after 10+ epochs)
python code/eeg_ldm.py --resume_ckpt_path exps/results/generation/01-03-2026-18-00-04/checkpoint.pth \
  --pretrain_mbm_path pretrains/eeg_pretain/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth \
  --pretrain_gm_path pretrains --num_epoch 20 --check_val_every_n_epoch 1 --batch_size 4 --num_workers 8
```

- **Requirement:** The checkpoint must be from a **SAR-HM** run with **`--normalize_conditioning true`** (or default) so it includes `cond_ln`. If you load a checkpoint that doesn’t have cond_ln, you’ll see:  
  `[DEBUG] [CKPT_LOAD] cond_ln missing is OK: using random init` — then cond_ln is untrained and you need Option 2 or more epochs.

### Option 2: Run Stage B long enough so cond_ln learns (alpha small early)

Train from scratch with **`normalize_conditioning=true`** (default) and run **at least 10–15 epochs** so `cond_ln` gets trained. α is already kept small early via **alpha_max warmup**:

- **Default:** `alpha_max_start=0.05` (epoch 0), ramping to `alpha_max_end=0.2` over `warmup_epochs=10`.
- So for the first 10 epochs α is capped low (0.05 → 0.2); after that you get full α. That lets the baseline and cond_ln stabilize before SAR-HM has large effect.

**Minimal command (no resume):**

```bash
python code/eeg_ldm.py --num_epoch 15 --use_sarhm true --ablation_mode full_sarhm --proto_source baseline_centroids \
  --normalize_conditioning true \
  --pretrain_mbm_path pretrains/eeg_pretain/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth \
  --pretrain_gm_path pretrains --check_val_every_n_epoch 1 --batch_size 4 --num_workers 8
```

**Override alpha warmup (optional):** keep α smaller for longer or shorten warmup:

```bash
# Longer warmup (15 epochs), same start/end
python code/eeg_ldm.py ... --warmup_epochs 15 --alpha_max_start 0.05 --alpha_max_end 0.2

# Shorter warmup, still small at start
python code/eeg_ldm.py ... --warmup_epochs 5 --alpha_max_start 0.05 --alpha_max_end 0.2
```

- **cond_ln:** Only created when `normalize_conditioning=true` and `normalization_type=layernorm` (default). It normalizes `c_base` and `c_sar` before fusion so their scales match.
- **Summary:** For best quality quickly, either **(1)** resume from a checkpoint that already includes cond_ln, or **(2)** run Stage B at least 10–15 epochs with `normalize_conditioning true` and use the default alpha warmup (or the CLI overrides above).

---

- **Stage B** – enable conditioning stats + VAE round-trip + save intermediate decoded images at steps 250,200,150,100,50 for the first val sample:
  ```bash
  python code/eeg_ldm.py ... --num_epoch 5 --debug
  ```
  Or only sampling intermediates: `--debug_sampling_steps 250,200,150,100,50`.

- **Stage C** – conditioning stats, VAE round-trip, and attention plot:
  ```bash
  python code/gen_eval_eeg.py ... --debug
  ```

Interpretation:
- **Conditioning stats** (`[COND_STATS]` / `[COND]`): mean/std/min/max/norm for c_base, c_sar, c_final and alpha/conf/entropy. NaN/Inf or extreme scales → conditioning or SAR-HM retrieval issue.
- **VAE round-trip** (`vae_roundtrip.png`): encode then decode one real image. If this is noise → VAE/scale_factor or dtype issue.
- **Sampling intermediates** (`val_intermediates/step*_sample*.png`): first val sample decoded at given steps. If structure never appears → sampling or conditioning; if it appears late → step count or schedule.

---

When correct, you should see:

- `[DEBUG] [PROTO] loaded path=... source=loaded shape=(40, 768) ... finite=True`
- `SAR-HM ACTIVE | ... | proto_source=loaded ...`
- Alpha stats (e.g. `[COND] alpha min=... mean=... max=...`) with min &lt; max unless ablation forces fixed alpha.
