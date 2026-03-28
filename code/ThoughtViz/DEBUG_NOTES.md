# ThoughtViz `thoughtviz_image_with_eeg` — debugging notes

## Image ranges

| Source | Range | Where set |
|--------|--------|-----------|
| Real images (ImageNet-Filtered) | **[-1, 1]** when `--image_range tanh` (default) | `utils/data_input_util.load_image_data(..., normalize_to_tanh=True)` maps `x/255 → (2x-1)` |
| Real images (legacy) | **[0, 1]** | `normalize_to_tanh=False` or `--image_range unit` |
| Generator output | **[-1, 1]** (final `tanh`) | `training/models/thoughtviz.py` → `generator_model_rgb` |

**Match:** With default `--image_range tanh`, real and fake tensors presented to the discriminator are both in **[-1, 1]**, which matches `tanh` generator outputs. Legacy runs that trained with **[0, 1]** reals vs **[-1, 1]** fakes were mismatched; use `--image_range unit` only when **resuming old checkpoints** trained under that regime.

## Preview saving

- Previously: `grid * 255` on `tanh` outputs → invalid `uint8` mapping and garbled colors.
- Now: `utils/gan_image_norm.tensor_to_image_uint8` and `utils/image_utils.combine_rgb_preview_grid` apply `(x+1)/2`, clip to [0, 1], then scale to **uint8**; tiled grids stay uint8 for `PIL.Image.fromarray(..., mode="RGB")` (float32 grids are rejected).

## Discriminator training (actual behavior)

1. `discriminator_model_rgb` builds D with a **frozen** pretrained `Sequential` classifier (`classifier_model.trainable = False`).
2. `generator_containing_discriminator` calls `freeze_discriminator_for_generator_training(d)` so **all** D layers are non-trainable in the combined model `d_on_g` (only G updates).
3. After building `d_on_g`, training calls `unfreeze_discriminator_for_d_step(d)` so conv + fake head train; aux branch stays frozen (`utils/gan_training_utils.py`).
4. Each iteration:
   - `unfreeze_discriminator_for_d_step(d)` → `d.train_on_batch` (real, then fake).
   - `freeze_discriminator_for_generator_training(d)` → `d_on_g.train_on_batch`.
   - `unfreeze_discriminator_for_d_step(d)` again for the next iteration.

`model.summary()` printed **after** building `d_on_g` used to show **0 trainable params** on D because the whole D was frozen for the combined model; that was misleading. Startup now prints **actual trainable weight counts** via `print_trainable_param_report`.

## Other design notes

- **Inception score:** `get_inception_score` expects **uint8** images ~[0, 255]; generated tensors are converted with `tensor_to_image_uint8` before scoring.
- **EEG conditioning:** Class `layer_index = 9` features from the EEG classifier; `--conditioning_mode onehot` replaces that with a **fixed** 10→100 linear map to isolate GAN stability from EEG quality.

## Suspicious patterns to watch

- D loss → 0 and G loss exploding: D too strong or LR.
- D loss ~ log(2), G not moving: frozen D or wrong targets (check trainable report).
- Previews improve but inception does not: preview uses fixed mapping; check both.
