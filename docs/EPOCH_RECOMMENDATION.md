# Stage B epoch recommendation and when to stop

**Summary:** 500 epochs is **not compulsory**. Comparable final quality is often reached in **200–350 epochs**. Use Stage C eval at fixed checkpoints and the same test setup to decide manually whether to continue or stop.

---

## A) Clear recommendation

- **500 epochs is likely not compulsory** for this repo’s workflow (Stage B train + Stage C eval at 250 steps).
- **Recommended epoch range for “final” quality:** **250–350 epochs** for baseline; **300–400 epochs** for SAR-HM if you want the fusion and prototypes fully settled. Many runs plateau in SSIM/CLIP by 200–250 epochs.
- **Practical schedule (no code changes):** Run Stage B for a target (e.g. 200 or 300), run Stage C once with the same seed and a fixed test subset. If metrics and qualitative images are acceptable, treat that as final. If not, resume or start a longer run (e.g. to 400/500). This is a **manual decision point**, not automatic early stopping.

---

## B) Evidence to look at (manual stopping checklist)

Use these to justify stopping or continuing; all without changing any training or eval code.

1. **Stage B training logs**
   - **train/loss** (or `train/loss_simple`, `train/loss_gen`, `train/loss_clip`): trend should flatten; large drops are early, small oscillations later. Plateau suggests conditioning has converged.
   - **SAR-HM only:** `alpha_max` warmup over `warmup_epochs` (default 10); after warmup, loss should stabilize. If loss is still dropping sharply after 50–100 epochs, more epochs may help.

2. **Stage C evaluation (source of “final quality”)**
   - Run Stage C at **250 steps**, **same seed** (e.g. `--seed 2022`), **same test subset** (e.g. `--max_test_items 20`, `--num_samples 1`) for fair comparison across runs.
   - **Metrics (from compare_eval or gen_eval):** **SSIM**, **PCC**, **CLIP similarity** (pair-wise gen vs GT when `--imagenet_path` is set). When these stop improving between checkpoint A and B (e.g. epoch 200 vs 300), extra epochs add little.
   - **Qualitative:** Inspect generated images at 100, 200, 300 (and optionally 400, 500). Early epochs (e.g. 10–50) are noisy; by 100–200 images usually become structured; further epochs refine detail. If images at 200 and 300 look similar, more epochs are optional.

3. **Fair comparison**
   - Same **seed**, same **splits_path**, same **max_test_items** / **n_samples** in Stage C and compare.
   - Use **ddim_steps 250** for all “final” checks.

4. **Checkpoint strategy**
   - Stage B saves the last epoch by default. To compare epochs 100 vs 200 vs 300, either:
     - Run multiple Stage B runs with different `--num_epoch` (e.g. 100, 200, 300), or
     - Use a run that saves checkpoints every N epochs (if your setup does so) and point Stage C at the desired epoch’s checkpoint.
   - No code changes: manual runs with different `--num_epoch` and one Stage C per run is sufficient.

---

## C) Why this range (convergence)

- **DreamDiffusion-style setup:** Only the **conditioning path** (EEG encoder + mapper + optional SAR-HM) is trained; diffusion UNet/VAE are frozen. So convergence is dominated by the conditioner, not full diffusion fine-tuning.
- **Steps per epoch:** With full train split (~668 items after valid_indices) and `batch_size=48`, one epoch ≈ 14 steps. So 200 epochs ≈ 2800 steps, 500 ≈ 7000 steps. Conditioning typically stabilizes in a few thousand steps.
- **Early epochs:** First 20–50 epochs often look noisy (EEG→latent mapping not yet aligned). By 100–150, images usually become structured; 200–300 often reach a plateau in SSIM/CLIP.
- **SAR-HM:** `proto_freeze_epochs` (default 5) and `warmup_epochs` (default 10) delay full SAR-HM influence. Allowing 50–100 epochs after warmup for prototypes and gate to settle is reasonable; hence 300–400 can be safer than 200 for SAR-HM if you want maximum convergence.

---

## D) Where this is reflected in the repo

- **commands.md:** Section “Complete / full training: how many epochs?” states 500 is default but not mandatory; recommends 100–150 for thesis; table gives 150–200 as “safe choice.” Eval checkpoints and “how to decide” are reinforced there.
- **docs/explain.md:** `num_epoch` default 500 described as “Thesis-level; quick test: 10” with a note that comparable quality may be reached earlier.
- **docs/FINAL_RUN_COMMANDS.md:** Notes clarify 500 as safe upper bound; “How to decide if enough epochs” points to this doc and Stage C + metrics + same seed/subset.

No training or eval code is changed; only comments and documentation.
