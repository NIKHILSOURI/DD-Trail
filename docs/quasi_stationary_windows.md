# Quasi-Stationary Windows in This Project

This document explains the quasi-stationary EEG preprocessing that was added to this repository, why it exists, where it runs, and how to use it safely.

---

## 1) What problem this solves

EEG is highly non-stationary over long time ranges. If we process the full signal as one global sequence, fast local temporal patterns can be blurred by global resampling.

The quasi-stationary (QS) approach assumes that **short local segments are closer to stationary** than the full recording. Instead of one global interpolation over all time points, the signal is split into many local windows and each window is normalized to a fixed local length before concatenation.

In short:

- **Without QS:** one global resample from `(C, T_source)` to `(C, data_len)`.
- **With QS:** split into `num_windows`, resample each window to `samples_per_window`, concatenate, then (if needed) do a final global adjustment to `data_len`.

---

## 2) Where it is implemented

Core logic is centralized in `code/eeg_preprocessing.py`:

- `quasi_stationary_windowing(...)`
- `resample_time_axis(...)`
- `preprocess_eeg_numpy(...)`
- `adapt_channels(...)`

Main dataset integration points are in `code/dataset.py`:

- `EEGDataset.__getitem__()` uses `preprocess_eeg_numpy(...)` for Stage B data.
- `eeg_pretrain_dataset.__getitem__()` optionally applies `quasi_stationary_windowing(...)` for Stage A1.
- `build_stage_b_datasets(...)` forwards QS settings from config into `EEGDataset`.

Config defaults are in `code/config.py`:

- `Config_MBM_EEG` (Stage A1 pretrain)
- `Config_Generative_Model` (Stage B DreamDiffusion)

CLI toggles:

- `code/stageA1_eeg_pretrain.py` has `--use_quasi_stationary_windows`, `--qs_num_windows`, `--qs_samples_per_window`.
- `code/eeg_ldm.py` has `--use_quasi_stationary_windows` (boolean toggle).  
  Window size/count values come from config defaults unless changed in `code/config.py`.

---

## 3) Exact algorithm used

For input EEG shaped `(C, T_source)`:

1. Build integer window edges with `np.linspace(0, T_source, num_windows + 1, dtype=int)`.
2. For each window `w`:
   - Slice `seg = eeg_tc[:, a:b]`.
   - If a segment is too short (`< 2` samples), pad in time with edge values.
   - Resample `seg` to `samples_per_window` using linear interpolation (`interp1d`).
3. Concatenate all resampled segments along time.
4. Expected output time length becomes:
   - `expected = num_windows * samples_per_window`
5. Safety check:
   - If concatenation does not exactly match `expected`, it is resampled again to `expected`.
6. In `preprocess_eeg_numpy(...)`:
   - If QS is enabled, the QS output is optionally resampled to `data_len` when needed.
   - Channels are then adapted to `channel_target` (tile or truncate).
   - Final scaling divides by `scale_div` (default `10.0`).

---

## 4) Current defaults and what they imply

Default QS-related values:

- `use_quasi_stationary_windows = False`
- `qs_num_windows = 64`
- `qs_samples_per_window = 8`
- `data_len = 512`

Important relationship:

- `64 * 8 = 512`

So with current defaults, if QS is enabled, the concatenated length already matches `data_len` and no extra post-QS resample is needed.

---

## 5) Stage A1 vs Stage B behavior

### Stage A1 (`code/stageA1_eeg_pretrain.py`)

- QS can be enabled directly from CLI.
- Stage A1 dataset (`eeg_pretrain_dataset`) applies optional QS before the model receives EEG.
- Good for aligning MAE pretraining with the same temporal preprocessing assumption.

### Stage B (`code/eeg_ldm.py`)

- Stage B can toggle QS with `--use_quasi_stationary_windows true|false`.
- Datasets are built through `build_stage_b_datasets(...)`, which forwards QS settings.
- Works for both supported dataset modes (`imagenet_eeg` and `thoughtviz`) because preprocessing is in shared dataset pipeline.

Practical consistency guideline:

- If Stage B runs with QS enabled, it is generally better to have Stage A1 pretraining done with QS too, so encoder assumptions match better.

---

## 6) Shapes and data flow (simplified)

Stage B item path (`EEGDataset`):

1. Load raw EEG tensor from `.pth`.
2. Temporal crop to `[20:460]` (project’s existing behavior).
3. Convert to numpy `(C, T)`.
4. Run `preprocess_eeg_numpy(...)`:
   - QS branch or legacy global branch.
   - Channel adapt to target channels.
   - Scale.
5. Convert back to torch float tensor.

Final model-facing EEG shape remains consistent with configured `data_len` and `channel_target` regardless of QS on/off.

---

## 7) Why this is backward compatible

The feature is strictly flag-gated:

- Defaults keep QS disabled.
- Existing checkpoints/commands continue to work unless QS is explicitly enabled.
- With QS disabled, behavior remains legacy global-resample preprocessing.

---

## 8) Choosing parameters safely

Recommended baseline:

- `qs_num_windows=64`
- `qs_samples_per_window=8`
- `data_len=512`

Rules of thumb:

- Keep `qs_num_windows * qs_samples_per_window` close to `data_len`.
- Too many windows with very small local sample count may oversmooth local structure.
- Too few windows moves behavior closer to global resampling.

If you change one parameter, verify:

- final time length entering model is what you expect.
- Stage A1 and Stage B settings are aligned if reusing pretrained encoder checkpoints.

---

## 9) Enabling it in practice

### Stage A1

Example:

```bash
python code/stageA1_eeg_pretrain.py \
  --use_quasi_stationary_windows true \
  --qs_num_windows 64 \
  --qs_samples_per_window 8
```

### Stage B

Example:

```bash
python code/eeg_ldm.py \
  --use_quasi_stationary_windows true \
  ...
```

If you also want to change `qs_num_windows`/`qs_samples_per_window` for Stage B, set them in `Config_Generative_Model` in `code/config.py` (current Stage B CLI only exposes the on/off switch).

---

## 10) Common pitfalls and troubleshooting

- **Pitfall: Stage mismatch**
  - Stage A1 trained without QS, Stage B run with QS (or vice versa) can reduce transfer quality.

- **Pitfall: unexpected lengths**
  - If `num_windows * samples_per_window` differs from `data_len`, an extra resample occurs after concatenation.

- **Pitfall: assuming QS changes channel behavior**
  - QS affects temporal axis only; channel adaptation is a separate step.

- **Debug tip**
  - Use temporary shape logging around `preprocess_eeg_numpy(...)` output to verify `(channel_target, data_len)` and compare QS on/off runs.

---

## 11) Summary

Quasi-stationary windows in this project are a **temporal preprocessing option** that:

- preserves local temporal structure better than one global interpolation,
- is shared across Stage A1 and Stage B pipelines,
- is backward compatible by default,
- and is controlled by explicit config/CLI flags.

For stable experiments, keep parameter products aligned with `data_len`, and keep Stage A1/Stage B preprocessing assumptions consistent.

