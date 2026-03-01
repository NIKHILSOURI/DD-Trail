# SAR-HM Safe by Design – Run Guide & Troubleshooting

## Safe SAR-HM rules

- **Baseline residual fusion (only fusion used):**  
  `c_final = c_base + α * (c_sar - c_base)`.  
  Cross-attention uses key `c_crossattn`, shape `[B, 77, 768]`.  
  α gating and fallbacks make SAR-HM **non-degrading by design**: when prototypes are invalid or confidence is low, α=0 and we use baseline only.

- **Hard fallbacks:**  
  - Prototypes invalid (`proto_source` in `{"dummy", None}` or `P` missing/wrong shape/non-finite) → **α = 0**, log warning.  
  - `conf < conf_threshold` → **α = 0**.

- **α must vary:**  
  α = `alpha_max * conf` (entropy/max-derived confidence), then clamped and gated by `conf_threshold`. If prototypes are missing → α forced to 0.

- **Baseline never degrades:**  
  Baseline-only path is unchanged; SAR is additive. Use `--ablation baseline` or `--no_sarhm` to run pure baseline.

---

## How prototypes are loaded in Stage C

- The Stage B checkpoint already contains trained prototype weights.  
- If you pass **`--proto_path <path>`**, that file **overwrites** the checkpoint’s prototypes.  
- **Use the same run’s `prototypes.pt`** (next to `checkpoint.pth`), not a different file (e.g. not `prototypes_baseline_centroids.pt` for a SAR-HM run), or you may get wrong retrieval and noisy images.

- **`--latest_run_dir <dir>`**  
  If set, the script finds the latest subdir under `<dir>` (e.g. `exps/results/generation`), copies `checkpoint.pth` and `prototypes.pt` to `exps/latest/`, and uses those paths so Stage C always has matching checkpoint + prototypes.

- **Acceptance:**  
  When `--proto_path` is set, logs must show `proto_source=loaded` and `has_valid_prototypes=True`. If you see `proto_source=dummy`, fix `--proto_path` or use the latest-run copier.

---

## Minimal Stage C (10–20 images)

From repo root (or with `code` on PYTHONPATH):

```bash
python code/gen_eval_eeg.py --dataset EEG --model_path exps/results/generation/<timestamp>/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --num_samples 20 --split test
```

- **Baseline only (no SAR-HM):**  
  Add `--no_sarhm` or `--ablation baseline`.

- **Full SAR-HM with prototypes loaded:**  
  Add `--proto_path exps/results/generation/<timestamp>/prototypes.pt` (same run as checkpoint).

- **Using latest run:**  
  Use `--latest_run_dir exps/results/generation` so the script copies the latest run to `exps/latest/` and uses `exps/latest/checkpoint.pth` and `exps/latest/prototypes.pt`.

---

## Full Stage C run

```bash
python code/gen_eval_eeg.py --dataset EEG --model_path exps/results/generation/<timestamp>/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth \
  --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --proto_path exps/results/generation/<timestamp>/prototypes.pt
```

Omit `--num_samples` to use config default; omit `--max_items` / `--max_test_items` to run on full test set.

---

## Ablation matrix

| Ablation              | CLI / behavior |
|-----------------------|----------------|
| **Baseline**          | `--ablation baseline` or `--no_sarhm` or `--disable_sarhm` |
| **Projection only**   | `--ablation projection_only` |
| **Hopfield, no gate** | `--ablation hopfield_no_gate` (fixed α=0.05) |
| **Full SAR-HM**       | `--ablation full` (default) |

Override fusion: `--force_alpha 0.2`, `--conf_threshold 0.2`, `--alpha_max 0.2` (use `-1` for force_alpha to disable).

---

## Sanity checks

1. **Prototypes:**  
   Log line: `[DEBUG] [PROTO] loaded path=... source=loaded shape=(K,768) dtype=... finite=True`.  
   With `--proto_path` set, do **not** see `proto_source=dummy`.

2. **Alpha:**  
   With full SAR-HM and entropy mode, alpha should **vary** across samples (min/mean/max in `[COND]` or `[COND_STATS]`). If alpha is always equal to `alpha_max`, check prototype loading and confidence.

3. **Conditioning scale:**  
   With `--debug` or `config.debug_cond_stats=True`, `[COND_STATS]` should show similar scales for c_base, c_sar, c_final after normalization.

4. **Baseline safety:**  
   Run with `--ablation baseline` or `--no_sarhm`; output should be non-noise if the baseline checkpoint is valid.

5. **VAE / scale_factor:**  
   `[SCALE_FACTOR] value=0.18215 match=True`. With `--debug`, `vae_roundtrip.png` should look like a blurred but recognizable image, not noise.

---

## Logs to inspect

- **`[DEBUG] [PROTO]`** – path, source, shape, dtype, finite. Expect `source=loaded` and `has_valid_prototypes=True` when `--proto_path` is provided.
- **`SAR-HM ACTIVE | ... | proto_source=...`** – must not be `dummy` when using real prototypes.
- **`[DEBUG] [COND]`** / **`[COND_STATS]`** – c_final, c_base, c_sar, alpha min/mean/max; no NaN/Inf.
- **`[DEBUG] [SCALE_FACTOR]`** – should match 0.18215.
- **`[DEBUG] [CKPT_LOAD]`** – missing/unexpected keys; no large missing counts for UNet/VAE.

---

## Troubleshooting

- **proto_source dummy**  
  Provide `--proto_path` to the **run’s** `prototypes.pt`, or use `--latest_run_dir` so the script copies the latest run. Do not use a different run’s or baseline_centroids file unless you intend to test that setup.

- **Pure noise even with losses stable**  
  Usually conditioning or decode: wrong prototypes (wrong run or baseline_centroids), scale_factor mismatch, or missing keys in checkpoint. Check `[PROTO]`, `[SCALE_FACTOR]`, `[CKPT_LOAD]`, and `vae_roundtrip.png`.

- **Alpha always at alpha_max**  
  Entropy-based confidence may be 1 if attention is peaked (e.g. one class). Ensure prototypes are loaded and from the same run; try `--conf_threshold` or different `alpha_max`.

- **Baseline worse than before**  
  Baseline path is unchanged; if you see regression, ensure you are not overwriting checkpoint prototypes with a bad file and that you are comparing same dataset/splits.

---

## Expected behavior

- **Baseline:** Never degraded; same as before when SAR-HM is off or α=0.
- **SAR-HM:** Helps when prototypes are valid and confidence is high; when in doubt (invalid prototypes or low conf), α=0 and we fall back to baseline.

---

## Exact commands summary

**1) Minimal Stage C (20 images) – baseline**

```bash
python code/gen_eval_eeg.py --dataset EEG --model_path exps/results/generation/<timestamp>/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml --num_samples 20 --split test --no_sarhm
```

**2) Minimal Stage C (20 images) – full SAR-HM with prototypes**

```bash
python code/gen_eval_eeg.py --dataset EEG --model_path exps/results/generation/<timestamp>/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml --num_samples 20 --split test \
  --proto_path exps/results/generation/<timestamp>/prototypes.pt
```

**3) Full Stage C**

```bash
python code/gen_eval_eeg.py --dataset EEG --model_path exps/results/generation/<timestamp>/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth \
  --config_patch pretrains/models/config15.yaml \
  --proto_path exps/results/generation/<timestamp>/prototypes.pt
```

Replace `<timestamp>` with your run folder name (e.g. `28-02-2026-21-42-16`).

---

## Stage B: 5-epoch quality check (correct-by-default)

Stage B now generates validation images every epoch when not disabled. Defaults: `disable_image_generation_in_val=False`, `val_image_gen_every_n_epoch=1`, `check_val_every_n_epoch=5`, `val_gen_limit=2`, `val_ddim_steps=50`, `val_num_samples=2`.

**4) Baseline 5-epoch quality check (no SAR-HM)**

```bash
python code/eeg_ldm.py --dataset EEG --checkpoint_path pretrains/eeg_pretain/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth \
  --pretrain_gm_path pretrains --num_epoch 5 --check_val_every_n_epoch 1 \
  --use_sarhm false --ablation_mode baseline
```

**5) Full SAR-HM 5-epoch quality check**

```bash
python code/eeg_ldm.py --dataset EEG --checkpoint_path pretrains/eeg_pretain/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth \
  --pretrain_gm_path pretrains --num_epoch 5 --check_val_every_n_epoch 1 \
  --use_sarhm true --ablation_mode full_sarhm
```

**6) Smoke test (minimal run: 1 val item, 4 samples, 25 steps, val every epoch)**

```bash
python code/eeg_ldm.py --dataset EEG --checkpoint_path pretrains/eeg_pretain/checkpoint.pth \
  --splits_path datasets/block_splits_by_image_single.pth --eeg_signals_path datasets/eeg_5_95_std.pth \
  --pretrain_gm_path pretrains --num_epoch 3 --smoke_test
```

---

## If outputs are still noisy: debug flags

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
