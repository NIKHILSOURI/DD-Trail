# Quick evaluation checklist — ThoughtViz image + EEG GAN

Run training from the `code/ThoughtViz` directory (or set `PYTHONPATH` to this folder).

## What success looks like after ~5 epochs (sanity / smoke)

- Losses are **finite** (no NaN/Inf in console or `outputs/sanity_check/loss_log.jsonl`).
- `d_loss` and `g_loss` **change** from epoch 0 to 4 (not stuck at identical floats).
- Preview grid (`*_g.png`) **changes** slightly epoch-to-epoch (same fixed noise/EEG in sanity mode).
- Debug lines (first epochs) show real batch **min/max** consistent with `--image_range` (e.g. ~[-1, 1] for `tanh`).

## What success looks like after ~20 epochs (sanity mode)

- **Discriminator:** `d_loss` does not collapse to exactly 0 for all batches (some variation is normal).
- **Generator:** `g_loss` moves; previews show **coarser structure** (blobs, color regions) even if not photorealistic.
- **One-hot mode:** If one-hot learns faster or looks cleaner than EEG mode, EEG encoding or pairing may need work; if **neither** learns, suspect data range, batching, or LR.

## Broken preprocessing

- Real **min/max** outside expected range for chosen `--image_range` (e.g. reals looking like [0, 1] when you intended `tanh`).
- Fake **min/max** stuck at exactly -1 or 1 everywhere (dead `tanh`) or wildly outside [-1, 1] (bug elsewhere).

## Discriminator overpowering

- `g_loss` rises consistently; previews become uniform noise or single color.
- `D(fake_prob) mean` stays near 0 while real near 1 with no G progress (strong D — try lower `d_adam_lr` or fewer D steps; this stack uses 1 D step per G step).

## Generator collapse / mode collapse

- All preview tiles look **identical** across classes.
- Inception diversity collapses (if you run inception): single-mode scores.

## Previews look “static” (unchanging) or like TV noise

**Unchanging grid:** Full training used to save **one fixed** noise+EEG batch every time, so PNGs could look the same for many epochs even while the network trained. **Default now:** each preview on **full** runs uses a **new random** batch (seed `preview_seed + epoch`). Use **`--preview_fixed`** if you want the old fixed grid for comparisons.

**Noise / no clear objects:** **Sanity** with **2 images** is far too small for a GAN to learn photoreal content; noisy or blob-like outputs after thousands of epochs are **expected**. Evaluate appearance on **full-data** runs, not tiny sanity.

## Preview bug vs training failure

- **Preview bug:** Structure visible in **logged tensor stats** (fake mean/std reasonable) but PNG is neon garbage → mapping bug; fixed by `tensor_to_image_uint8` path.
- **Training failure:** Stats show fake batch **constant** or nonsensical; previews match (no structure).

## Verify generator + discriminator before a long run

**A. Wiring (no training)** — confirms shapes, EEG features, and D forward pass:

```bash
bash run_with_gpu.sh training/verify_g_and_d.py
# optional: load your saved generator
bash run_with_gpu.sh training/verify_g_and_d.py --checkpoint saved_models/thoughtviz_image_with_eeg/Image/run_1/generator_0
```

Check `outputs/smoke_verify/generator_one_sample.png` (one full 64×64 decode) and console lines for `D(real)` / `D(fake)` means.

**B. Learning (short)** — confirms gradients and memorization:

```bash
bash run_with_gpu.sh training/thoughtviz_image_with_eeg.py --overfit_one_batch --epochs 25 --image_range tanh
```

Previews should start changing; on a single batch, tiles often get sharper over time.

**C. Full training** only after A + B look reasonable.

## Before launching 500+ or 5000+ epochs

1. Run **`--sanity_check`** (20–30 epochs) with default `tanh` normalization.
2. Run **`--overfit_one_batch --epochs 10`** — previews should show **memorization** of the small batch; if not, fix pipeline before scaling.
3. Run **`--conditioning_mode onehot --sanity_check`** — confirms core GAN + conditioning path without EEG.
4. If **resuming** an old checkpoint trained on **[0, 1]** reals, use **`--image_range unit`** or retrain from scratch with `tanh`.
5. Confirm `loss_log.jsonl` (under your output dir) shows smooth finite curves.

## Resuming after sanity → full training

Sanity runs now save **final** weights as `saved_models/thoughtviz_image_with_eeg/Image/run_1/generator_<last_ep>` (same for `discriminator_<last_ep>`). If you trained **100** sanity epochs, indices are **0 … 99**, so the last file is **`generator_99`**.

Continue on the **full** dataset (no `--sanity_check`), same `--image_range` as before:

```bash
bash run_with_gpu.sh training/thoughtviz_image_with_eeg.py \
  --resume_from_epoch 99 \
  --epochs 1000 \
  --image_range tanh \
  --skip_inception
```

`--epochs` here is **how many more epoch steps** to run (e.g. 1000 → epochs 100 … 1099 on disk).

**Note:** A sanity run **before** this save logic shipped did **not** write checkpoints; in that case either re-run a short sanity once (to produce `generator_99`) or start full training **without** `--resume_from_epoch`.

**Extend an existing sanity run** (same tiny subset), e.g. 1000 epochs done → continue 5000 more (last saved index is `999`):

```bash
bash run_with_gpu.sh training/thoughtviz_image_with_eeg.py \
  --sanity_check \
  --sanity_classes 0 \
  --sanity_images_per_class 2 \
  --sanity_epochs 5000 \
  --resume_from_epoch 999
```

Requires `saved_models/.../generator_999` and `discriminator_999` from the previous run.

## Command reference

```bash
# Sanity (fast, previews every epoch; length = --sanity_epochs)
python training/thoughtviz_image_with_eeg.py --sanity_check --sanity_epochs 25

# One-batch overfit (memorization test)
python training/thoughtviz_image_with_eeg.py --overfit_one_batch --epochs 15

# One-hot baseline (no EEG features)
python training/thoughtviz_image_with_eeg.py --sanity_check --conditioning_mode onehot --sanity_epochs 25

# Normal long run (from repo root with your GPU wrapper)
bash run_with_gpu.sh training/thoughtviz_image_with_eeg.py --epochs 5000

# Resume (same image_range as original run)
bash run_with_gpu.sh training/thoughtviz_image_with_eeg.py --resume_from_epoch 4950 --epochs 5000 --image_range unit
```

Adjust `--image_range` for resume to match how the checkpoint was trained.
