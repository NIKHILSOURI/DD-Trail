import argparse
import json
import os
import sys
# Quieter TensorFlow logs (INFO lines look like hangs). 1=hide INFO, 2=hide INFO+WARNING
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# Before Keras builds graphs: legacy GAN + dropout can trigger Grappler layout errors on GPU
# ("TransposeNHWCToNCHW-LayoutOptimizer"). Disabling avoids noisy errors; training is unchanged.
import tensorflow as tf

try:
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
except Exception:
    pass

import numpy as np
from keras import backend as K
import random
from PIL import Image
from keras.models import load_model
from keras.utils import to_categorical
import utils.data_input_util as inutil
from utils.keras_compat import adam_opt, set_learning_phase_inference
from utils.pickle_compat import load_pickle_compat
from training.models.thoughtviz import *
from utils.image_utils import *
from utils.gan_image_norm import tensor_to_image_uint8
from utils.gan_training_utils import (
    append_jsonl,
    build_onehot_projection,
    freeze_discriminator_for_generator_training,
    onehot_labels_to_feature_vectors,
    print_trainable_param_report,
    unfreeze_discriminator_for_d_step,
)
from utils.eval_utils import *
from utils import thoughtviz_paths as tv_paths

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

# Single place to set run length (edit here for shorter smoke tests or long runs).
DEFAULT_TRAIN_EPOCHS = 5000
# Samples for inception score during training (50k = paper-style but prints one "." per forward pass and is very slow).
INCEPTION_EVAL_SAMPLES = 2048


def _pbar(it, **kwargs):
    if tqdm is None:
        return it
    return tqdm(it, **kwargs)


def _checkpoint_paths(model_save_dir: str, epoch: int) -> tuple[str, str]:
    g = os.path.join(model_save_dir, f"generator_{epoch}")
    d = os.path.join(model_save_dir, f"discriminator_{epoch}")
    return g, d


def _adversarial_bce_targets(batch_size: int, label_smoothing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real / fake targets for D; third tensor is what G matches for the BCE head (one-sided smoothing)."""
    bs = batch_size
    a = float(label_smoothing)
    if a <= 0.0:
        return (
            np.ones((bs, 1), dtype=np.float32),
            np.zeros((bs, 1), dtype=np.float32),
            np.ones((bs, 1), dtype=np.float32),
        )
    return (
        np.full((bs, 1), 1.0 - a, dtype=np.float32),
        np.full((bs, 1), a, dtype=np.float32),
        np.full((bs, 1), 1.0 - a, dtype=np.float32),
    )


def _subset_classes_and_limit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    class_ids: list[int],
    max_per_class: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only given classes; cap images per class; shuffle."""
    ys = np.argmax(y_train, axis=1)
    parts_x, parts_y = [], []
    for c in class_ids:
        idx = np.where(ys == c)[0]
        rng.shuffle(idx)
        idx = idx[:max_per_class]
        parts_x.append(x_train[idx])
        parts_y.append(y_train[idx])
    x = np.concatenate(parts_x, axis=0)
    y = np.concatenate(parts_y, axis=0)
    perm = rng.permutation(len(x))
    return x[perm], y[perm]


def _build_fixed_preview_batch(
    batch_size: int,
    input_noise_dim: int,
    num_classes: int,
    conditioning_mode: str,
    layer_output: np.ndarray,
    y_test_int: np.ndarray,
    onehot_proj: np.ndarray | None,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fixed noise + conditioning for preview grids so epochs are comparable."""
    noise = rng.uniform(-1, 1, (batch_size, input_noise_dim))
    meta: dict = {"preview_seed": int(rng.randint(0, 2**31 - 1))}
    if conditioning_mode == "onehot":
        labels = rng.randint(0, num_classes, size=batch_size)
        assert onehot_proj is not None
        eeg_vecs = onehot_labels_to_feature_vectors(labels, onehot_proj)
    else:
        # Must match classes still present in y_test_int after sanity filtering (e.g. only class 0).
        allowed = np.unique(np.asarray(y_test_int, dtype=int))
        if allowed.size == 0:
            raise ValueError(
                "EEG preview: no test rows left for the selected classes — "
                "check --sanity_classes vs EEG data in data.pkl."
            )
        labels = rng.choice(allowed, size=batch_size, replace=True)
        eeg_vecs = np.array(
            [layer_output[rng.choice(np.where(y_test_int == int(lab))[0])] for lab in labels],
            dtype=np.float32,
        )
        meta["eeg_row_indices"] = [
            int(rng.choice(np.where(y_test_int == int(lab))[0])) for lab in labels
        ]
    meta["preview_labels"] = labels.tolist()
    return noise, eeg_vecs, meta


def train_gan(
    input_noise_dim: int,
    batch_size: int,
    epochs: int,
    data_dir: str,
    saved_classifier_model_file: str,
    model_save_dir: str,
    output_dir: str,
    classifier_model_file: str,
    *,
    resume_from_epoch: int | None = None,
    conditioning_mode: str = "eeg",
    sanity_check: bool = False,
    sanity_epochs: int = 25,
    sanity_class_ids: list[int] | None = None,
    sanity_images_per_class: int = 40,
    overfit_one_batch: bool = False,
    overfit_steps_per_epoch: int = 100,
    debug_log_first_epochs: int = 3,
    debug_log_batches: int = 2,
    image_range: str = "tanh",
    preview_seed: int = 42,
    skip_inception: bool = False,
    preview_use_fixed_batch: bool | None = None,
    label_smoothing: float = 0.1,
    g_lr: float | None = None,
    d_lr: float | None = None,
    g_steps: int = 1,
    d_aux_loss_weight: float = 0.5,
):
    """
    Train GAN. If `resume_from_epoch` is set, loads that checkpoint and continues from epoch `resume_from_epoch + 1`.

    conditioning_mode:
      - "eeg": use EEG classifier features (paper-style).
      - "onehot": fixed linear map from class id to 100-D features (isolates GAN vs EEG encoding).

    image_range:
      - "tanh": real images in [-1,1] to match generator tanh (recommended).
      - "unit": legacy [0,1] reals (use when resuming very old checkpoints trained on [0,1]).

    preview_use_fixed_batch:
      - None (default): fixed noise/EEG for sanity/overfit; **new random batch each preview** on full runs
        (seeded by preview_seed+epoch) so PNGs are not misleadingly "frozen".
      - True: always the same preview batch (apples-to-apples across epochs).
    """
    if resume_from_epoch is not None and overfit_one_batch:
        raise ValueError("overfit_one_batch is for debugging; do not combine with resume.")

    if preview_use_fixed_batch is None:
        preview_use_fixed_batch = bool(sanity_check or overfit_one_batch)

    set_learning_phase_inference()
    imagenet_folder = tv_paths.training_images("ImageNet-Filtered")
    num_classes = 10
    feature_encoding_dim = 100

    normalize_to_tanh = image_range == "tanh"
    x_train, y_train, x_test, y_test = inutil.load_image_data(
        imagenet_folder, patch_size=(64, 64), normalize_to_tanh=normalize_to_tanh
    )

    rng = np.random.RandomState(preview_seed)
    # In sanity mode, length is controlled by --sanity_epochs (not --epochs).
    effective_epochs = sanity_epochs if sanity_check else epochs
    if sanity_check:
        class_ids = sanity_class_ids if sanity_class_ids is not None else [0, 1]
        x_train, y_train = _subset_classes_and_limit(
            x_train, y_train, class_ids, sanity_images_per_class, rng
        )
        skip_inception = True

    n_batch = max(1, int(x_train.shape[0] / batch_size))
    print("\n=== thoughtviz_image_with_eeg ===", flush=True)
    print(
        f"  epochs: {effective_epochs}  |  batches/epoch: {n_batch}  |  batch_size: {batch_size}",
        flush=True,
    )
    print(f"  conditioning_mode={conditioning_mode}  image_range={image_range}", flush=True)
    print(
        f"  g_lr={g_adam_lr}  d_lr={d_adam_lr}  label_smoothing={label_smoothing}  "
        f"g_steps={g_steps}  d_aux_loss_weight={d_aux_loss_weight}",
        flush=True,
    )
    if sanity_check:
        print(f"  [sanity_check] subset classes={class_ids} images/class<={sanity_images_per_class}", flush=True)
    if overfit_one_batch:
        print(
            f"  [overfit_one_batch] steps/epoch={overfit_steps_per_epoch} (single batch)",
            flush=True,
        )
    print("==================================\n", flush=True)

    g_adam_lr = g_lr if g_lr is not None else 0.00003
    g_adam_beta_1 = 0.5
    d_adam_lr = d_lr if d_lr is not None else 0.00005
    d_adam_beta_1 = 0.5

    c = load_model(classifier_model_file)
    g_optim = adam_opt(g_adam_lr, beta_1=g_adam_beta_1)
    d_optim = adam_opt(d_adam_lr, beta_1=d_adam_beta_1)
    lw = [1.0, float(d_aux_loss_weight)]

    if resume_from_epoch is not None:
        g_dir, d_dir = _checkpoint_paths(model_save_dir, resume_from_epoch)
        if not os.path.isdir(g_dir) or not os.path.isdir(d_dir):
            raise FileNotFoundError(
                f"Resume checkpoint not found. Expected directories:\n  {g_dir}\n  {d_dir}"
            )
        print(f"\n[resume] Loading generator/discriminator from epoch {resume_from_epoch}", flush=True)
        g = load_model(g_dir)
        d = load_model(d_dir)
        unfreeze_discriminator_for_d_step(d)
        d.compile(
            loss=["binary_crossentropy", "categorical_crossentropy"],
            optimizer=d_optim,
            loss_weights=lw,
        )
        g.compile(loss="categorical_crossentropy", optimizer=g_optim)
    else:
        d = discriminator_model_rgb((64, 64), c)
        unfreeze_discriminator_for_d_step(d)
        d.compile(
            loss=["binary_crossentropy", "categorical_crossentropy"],
            optimizer=d_optim,
            loss_weights=lw,
        )

        g = generator_model_rgb(input_noise_dim, feature_encoding_dim)
        g.compile(loss="categorical_crossentropy", optimizer=g_optim)

    d_on_g = generator_containing_discriminator(input_noise_dim, feature_encoding_dim, g, d)
    d_on_g.compile(
        loss=["binary_crossentropy", "categorical_crossentropy"],
        optimizer=g_optim,
        loss_weights=lw,
    )
    # After building d_on_g, D was fully frozen for the combined graph; restore D step behavior.
    unfreeze_discriminator_for_d_step(d)

    print_trainable_param_report(g, d, d_on_g)

    g.summary()
    d.summary()

    eeg_data = load_pickle_compat(os.path.join(data_dir, "data.pkl"))
    classifier = load_model(saved_classifier_model_file)
    classifier.summary()
    x_test_eeg = eeg_data[b"x_test"]
    y_test_eeg = eeg_data[b"y_test"]
    y_test_int = np.array([np.argmax(y) for y in y_test_eeg])
    layer_index = 9

    get_nth_layer_output = K.function(
        [classifier.layers[0].input], [classifier.layers[layer_index].output]
    )
    layer_output = get_nth_layer_output([x_test_eeg])[0]

    if sanity_check:
        mask = np.isin(y_test_int, class_ids)
        layer_output = layer_output[mask]
        y_test_int = y_test_int[mask]
        manifest = {
            "class_ids": class_ids,
            "sanity_images_per_class": sanity_images_per_class,
            "preview_seed": preview_seed,
            "conditioning_mode": conditioning_mode,
        }
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "sanity_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    # Classes that still have ≥1 EEG test row (after sanity filter). Fake labels must stay in this set.
    eeg_label_pool = np.unique(y_test_int.astype(int))
    if conditioning_mode == "eeg" and eeg_label_pool.size == 0:
        raise ValueError(
            "EEG conditioning: no test rows in data.pkl for the selected classes — "
            "widen --sanity_classes or check EEG labels."
        )

    onehot_proj = None
    if conditioning_mode == "onehot":
        onehot_proj = build_onehot_projection(num_classes, feature_encoding_dim, seed=preview_seed)

    # Fixed preview batch (esp. sanity_check) — same noise/conditioning every epoch for comparison
    prev_n = min(batch_size, x_train.shape[0])
    prev_noise, prev_eeg, prev_meta = _build_fixed_preview_batch(
        prev_n,
        input_noise_dim,
        num_classes,
        conditioning_mode,
        layer_output,
        y_test_int,
        onehot_proj,
        np.random.RandomState(preview_seed),
    )
    if sanity_check:
        manifest.update(prev_meta)
        with open(os.path.join(output_dir, "sanity_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    # Overfit: cache one real batch + matching conditioning (built after onehot_proj exists)
    of_labels: np.ndarray | None = None
    eeg_fix: np.ndarray | None = None
    noise_fix: np.ndarray | None = None
    real_fix: np.ndarray | None = None
    real_lab_fix: np.ndarray | None = None
    if overfit_one_batch:
        ob = min(batch_size, x_train.shape[0])
        noise_fix = np.random.uniform(-1, 1, (ob, input_noise_dim))
        if conditioning_mode == "onehot":
            of_labels = np.random.randint(0, num_classes, ob)
        else:
            of_labels = np.random.choice(eeg_label_pool, size=ob, replace=True)
        if conditioning_mode == "onehot":
            assert onehot_proj is not None
            eeg_fix = onehot_labels_to_feature_vectors(of_labels, onehot_proj)
        else:
            eeg_fix = np.array(
                [
                    layer_output[random.choice(np.where(y_test_int == int(rl))[0])]
                    for rl in of_labels
                ]
            )
        real_fix = x_train[0:ob].astype(np.float32)
        real_lab_fix = y_train[0:ob]

    start_epoch = (resume_from_epoch + 1) if resume_from_epoch is not None else 0
    end_epoch = start_epoch + effective_epochs
    print(
        f"  epoch range: {start_epoch} .. {end_epoch - 1}  ({effective_epochs} steps)",
        flush=True,
    )

    # Optional: assert ranges on startup (first real batch; may be smaller than batch_size)
    _ri = x_train[0 : min(batch_size, x_train.shape[0])].astype(np.float32)
    if normalize_to_tanh:
        assert float(_ri.min()) >= -1.01 and float(_ri.max()) <= 1.01, "real batch should be ~[-1,1]"
    else:
        assert float(_ri.min()) >= -0.01 and float(_ri.max()) <= 1.01, "real batch should be ~[0,1]"

    loss_log_path = os.path.join(output_dir, "loss_log.jsonl")

    epoch_iter = _pbar(
        range(start_epoch, start_epoch + effective_epochs),
        desc="Epoch",
        unit="ep",
        position=0,
        file=sys.stdout,
        dynamic_ncols=True,
    )
    first_preview_mean: float | None = None
    last_preview_mean: float | None = None

    for epoch in epoch_iter:
        batch_iter = _pbar(
            range(n_batch if not overfit_one_batch else 1),
            desc=f"  ep {epoch:04d}",
            unit="batch",
            leave=False,
            file=sys.stdout,
            dynamic_ncols=True,
        )
        inner_steps = 1 if not overfit_one_batch else overfit_steps_per_epoch

        for index in batch_iter:
            for step_i in range(inner_steps):
                if overfit_one_batch:
                    assert noise_fix is not None and of_labels is not None and eeg_fix is not None
                    assert real_fix is not None and real_lab_fix is not None
                    noise = noise_fix
                    one_hot_vectors = [to_categorical(int(lb), num_classes) for lb in of_labels]
                    eeg_feature_vectors = eeg_fix
                    real_images = real_fix
                    real_labels = real_lab_fix
                else:
                    # Match noise / labels to this slice (dataset may be smaller than batch_size).
                    start = index * batch_size
                    end = min(start + batch_size, x_train.shape[0])
                    bs = end - start
                    noise = np.random.uniform(-1, 1, (bs, input_noise_dim))
                    if conditioning_mode == "onehot":
                        random_labels = np.random.randint(0, num_classes, bs)
                    else:
                        random_labels = np.random.choice(eeg_label_pool, size=bs, replace=True)
                    one_hot_vectors = [to_categorical(label, num_classes) for label in random_labels]
                    if conditioning_mode == "onehot":
                        assert onehot_proj is not None
                        eeg_feature_vectors = onehot_labels_to_feature_vectors(
                            np.asarray(random_labels), onehot_proj
                        )
                    else:
                        eeg_feature_vectors = np.array(
                            [
                                layer_output[
                                    random.choice(np.where(y_test_int == random_label)[0])
                                ]
                                for random_label in random_labels
                            ]
                        )
                    real_images = x_train[start:end]
                    real_labels = y_train[start:end]

                unfreeze_discriminator_for_d_step(d)
                generated_images = g.predict([noise, eeg_feature_vectors], verbose=0)
                bs = int(real_images.shape[0])

                t_real, t_fake, t_gen = _adversarial_bce_targets(bs, label_smoothing)
                oh = np.array(one_hot_vectors).reshape(bs, num_classes)
                d_loss_real = d.train_on_batch(real_images, [t_real, np.array(real_labels)])
                d_loss_fake = d.train_on_batch(generated_images, [t_fake, oh])
                d_loss = (d_loss_fake[0] + d_loss_real[0]) * 0.5
                d_aux_real = float(d_loss_real[1]) if len(d_loss_real) > 1 else 0.0
                d_aux_fake = float(d_loss_fake[1]) if len(d_loss_fake) > 1 else 0.0

                freeze_discriminator_for_generator_training(d)
                g_loss = None
                for _gs in range(max(1, g_steps)):
                    g_loss = d_on_g.train_on_batch(
                        [noise, eeg_feature_vectors],
                        [t_gen, oh],
                    )
                unfreeze_discriminator_for_d_step(d)

                gl = float(g_loss[0]) if isinstance(g_loss, (list, tuple)) else float(g_loss)
                g_aux = float(g_loss[1]) if isinstance(g_loss, (list, tuple)) and len(g_loss) > 1 else 0.0

                dbg_ep = epoch - start_epoch
                if dbg_ep < debug_log_first_epochs and index < debug_log_batches and step_i == 0:
                    ri = real_images.astype(np.float32)
                    fi = generated_images.astype(np.float32)
                    print(
                        f"  [debug] ep{epoch} b{index}  real min/max/mean="
                        f"{ri.min():.3f}/{ri.max():.3f}/{ri.mean():.3f}  "
                        f"fake min/max/mean={fi.min():.3f}/{fi.max():.3f}/{fi.mean():.3f}",
                        flush=True,
                    )
                    print(
                        f"  [debug]         d_loss_real={float(d_loss_real[0]):.4f} "
                        f"d_loss_fake={float(d_loss_fake[0]):.4f} d_aux_r={d_aux_real:.4f} "
                        f"d_aux_f={d_aux_fake:.4f} g_loss={gl:.4f} g_aux={g_aux:.4f}",
                        flush=True,
                    )
                    try:
                        dr = d.predict(real_images, verbose=0)
                        df = d.predict(generated_images, verbose=0)
                        print(
                            f"  [debug]         D(fake_prob) mean real={float(np.mean(dr[0])):.4f} "
                            f"fake={float(np.mean(df[0])):.4f}",
                            flush=True,
                        )
                    except Exception as ex:
                        print(f"  [debug]         D mean stats skipped: {ex}", flush=True)

                if tqdm is not None and hasattr(batch_iter, "set_postfix") and step_i == inner_steps - 1:
                    batch_iter.set_postfix(d_loss=f"{float(d_loss):.3f}", g_loss=f"{gl:.3f}")

        # Save previews
        preview_every = 1 if (sanity_check or overfit_one_batch) else 100
        if epoch % preview_every == 0:
            if preview_use_fixed_batch:
                gen_prev = g.predict([prev_noise, prev_eeg], verbose=0)
            else:
                # Different noise/EEG each save (reproducible per epoch) — avoids "static" looking PNGs on full runs.
                rng_p = np.random.RandomState(preview_seed + int(epoch))
                pn, pe, _ = _build_fixed_preview_batch(
                    prev_n,
                    input_noise_dim,
                    num_classes,
                    conditioning_mode,
                    layer_output,
                    y_test_int,
                    onehot_proj,
                    rng_p,
                )
                gen_prev = g.predict([pn, pe], verbose=0)
            grid = combine_rgb_preview_grid(gen_prev, from_tanh=True)
            img_save_path = os.path.join(output_dir, str(epoch) + "_g.png")
            # grid is uint8 HWC — PIL rejects float32 RGB
            Image.fromarray(np.asarray(grid), mode="RGB").save(img_save_path)
            pm = float(np.mean(gen_prev))
            if first_preview_mean is None:
                first_preview_mean = pm
            last_preview_mean = pm
            print(f"\n  [epoch {epoch}] Saving preview → {img_save_path}", flush=True)

        if not skip_inception and epoch % 100 == 0 and not sanity_check:
            test_image_count = INCEPTION_EVAL_SAMPLES
            print(
                f"  [epoch {epoch}] Generating {test_image_count} samples for inception score (slow)…",
                flush=True,
            )
            test_noise = np.random.uniform(-1, 1, (test_image_count, input_noise_dim))
            if conditioning_mode == "onehot":
                test_labels = np.random.randint(0, num_classes, test_image_count)
            else:
                test_labels = np.random.choice(eeg_label_pool, size=test_image_count, replace=True)
            if conditioning_mode == "onehot":
                assert onehot_proj is not None
                eeg_feature_vectors_test = onehot_labels_to_feature_vectors(test_labels, onehot_proj)
            else:
                eeg_feature_vectors_test = np.array(
                    [
                        layer_output[random.choice(np.where(y_test_int == tl)[0])]
                        for tl in test_labels
                    ]
                )
            test_images = g.predict([test_noise, eeg_feature_vectors_test], verbose=0)
            # Inception expects uint8 ~ [0,255]
            ims = [tensor_to_image_uint8(test_images[i], from_tanh=True) for i in range(test_image_count)]
            inception_score = get_inception_score(ims, splits=10)
            print(
                f"  [epoch {epoch}] inception_score (mean): {inception_score[0]:.4f}  (std: {inception_score[1]:.4f})\n",
                flush=True,
            )

        gl = float(g_loss[0]) if isinstance(g_loss, (list, tuple)) else float(g_loss)
        if tqdm is not None and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(last_d=f"{float(d_loss):.3f}", last_g=f"{gl:.3f}")
        print(f"  summary  ep {epoch:04d}  d_loss={float(d_loss):.4f}  g_loss={gl:.4f}", flush=True)

        rec = {
            "epoch": int(epoch),
            "d_loss": float(d_loss),
            "g_loss": float(gl),
            "d_loss_real": float(d_loss_real[0]),
            "d_loss_fake": float(d_loss_fake[0]),
        }
        append_jsonl(loss_log_path, rec)

        ckpt_every = 1 if (sanity_check or overfit_one_batch) else 50
        if epoch % ckpt_every == 0 and not sanity_check:
            g.save(os.path.join(model_save_dir, "generator_" + str(epoch)), overwrite=True, include_optimizer=True)
            d.save(
                os.path.join(model_save_dir, "discriminator_" + str(epoch)),
                overwrite=True,
                include_optimizer=True,
            )

    # Sanity/overfit skip per-epoch saves above — always persist final weights so you can
    # `--resume_from_epoch <last>` into full training on the full dataset.
    if (sanity_check or overfit_one_batch) and effective_epochs > 0:
        last_ep = start_epoch + effective_epochs - 1
        g.save(
            os.path.join(model_save_dir, "generator_" + str(last_ep)),
            overwrite=True,
            include_optimizer=True,
        )
        d.save(
            os.path.join(model_save_dir, "discriminator_" + str(last_ep)),
            overwrite=True,
            include_optimizer=True,
        )
        print(
            f"\n[checkpoint] Saved final weights for resume: "
            f"{model_save_dir}/generator_{last_ep}  (and discriminator_{last_ep})\n",
            flush=True,
        )

    # End-of-run interpretation (sanity / overfit)
    if sanity_check or overfit_one_batch:
        print("\n--- run interpretation ---", flush=True)
        try:
            with open(loss_log_path, encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            if lines:
                d0, g0 = lines[0]["d_loss"], lines[0]["g_loss"]
                d1, g1 = lines[-1]["d_loss"], lines[-1]["g_loss"]
                print(f"  loss first epoch:  d={d0:.4f} g={g0:.4f}", flush=True)
                print(f"  loss last epoch:   d={d1:.4f} g={g1:.4f}", flush=True)
                finite = all(np.isfinite([d0, g0, d1, g1]))
                print(f"  losses finite: {finite}", flush=True)
                if first_preview_mean is not None and last_preview_mean is not None:
                    print(
                        f"  preview batch mean (tanh space): first={first_preview_mean:.4f} last={last_preview_mean:.4f}",
                        flush=True,
                    )
                if abs(d1 - d0) < 1e-6 and abs(g1 - g0) < 1e-6:
                    print("  warning: losses barely changed — check LR, batch, or broken graph.", flush=True)
        except Exception as ex:
            print(f"  (could not read loss log: {ex})", flush=True)
        print("--- end interpretation ---\n", flush=True)


def train(
    epochs: int | None = None,
    resume_from_epoch: int | None = None,
    *,
    conditioning_mode: str = "eeg",
    sanity_check: bool = False,
    sanity_epochs: int = 25,
    sanity_class_ids: list[int] | None = None,
    sanity_images_per_class: int = 40,
    overfit_one_batch: bool = False,
    overfit_steps_per_epoch: int = 100,
    debug_log_first_epochs: int = 3,
    debug_log_batches: int = 2,
    image_range: str = "tanh",
    preview_seed: int = 42,
    skip_inception: bool = False,
    preview_fixed: bool = False,
    label_smoothing: float = 0.1,
    g_lr: float | None = None,
    d_lr: float | None = None,
    g_steps: int = 1,
    d_aux_loss_weight: float = 0.5,
    output_dir_override: str | None = None,
    model_save_dir_override: str | None = None,
):
    dataset = "Image"
    batch_size = 100
    run_id = 1
    if epochs is None:
        epochs = DEFAULT_TRAIN_EPOCHS
    subset = dataset.lower()
    imagenet_folder = tv_paths.training_images("ImageNet-Filtered")
    classifier_model_file = tv_paths.trained_image_classifier(subset)
    eeg_data_dir = tv_paths.data_eeg(subset)
    eeg_classifier_model_file = tv_paths.eeg_classifier_model(subset)
    tv_paths.validate_image_eeg_prereqs(
        imagenet_folder=imagenet_folder,
        classifier_h5=classifier_model_file,
        eeg_data_dir=eeg_data_dir,
        eeg_classifier_h5=eeg_classifier_model_file,
    )

    model_save_dir = model_save_dir_override or tv_paths.saved_models(
        "thoughtviz_image_with_eeg", dataset, "run_" + str(run_id)
    )
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    if output_dir_override is not None:
        output_dir = output_dir_override
    elif sanity_check:
        output_dir = tv_paths.outputs_dir("sanity_check")
    else:
        output_dir = tv_paths.outputs_dir("thoughtviz_image_with_eeg", dataset, "run_" + str(run_id))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_gan(
        input_noise_dim=100,
        batch_size=batch_size,
        epochs=epochs,
        data_dir=eeg_data_dir,
        saved_classifier_model_file=eeg_classifier_model_file,
        model_save_dir=model_save_dir,
        output_dir=output_dir,
        classifier_model_file=classifier_model_file,
        resume_from_epoch=resume_from_epoch,
        conditioning_mode=conditioning_mode,
        sanity_check=sanity_check,
        sanity_epochs=sanity_epochs,
        sanity_class_ids=sanity_class_ids,
        sanity_images_per_class=sanity_images_per_class,
        overfit_one_batch=overfit_one_batch,
        overfit_steps_per_epoch=overfit_steps_per_epoch,
        debug_log_first_epochs=debug_log_first_epochs,
        debug_log_batches=debug_log_batches,
        image_range=image_range,
        preview_seed=preview_seed,
        skip_inception=skip_inception,
        preview_use_fixed_batch=True if preview_fixed else None,
        label_smoothing=label_smoothing,
        g_lr=g_lr,
        d_lr=d_lr,
        g_steps=g_steps,
        d_aux_loss_weight=d_aux_loss_weight,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ThoughtViz image+EEG GAN. Checkpoints: saved_models/.../generator_<ep>, discriminator_<ep>."
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_TRAIN_EPOCHS,
        help=f"How many epoch steps to run in this process (default: {DEFAULT_TRAIN_EPOCHS}).",
    )
    p.add_argument(
        "--resume_from_epoch",
        type=int,
        default=None,
        metavar="E",
        help="Load generator_E and discriminator_E from the current run's model_save_dir, then continue from epoch E+1.",
    )
    p.add_argument(
        "--conditioning_mode",
        choices=["eeg", "onehot"],
        default="eeg",
        help="eeg: EEG features (default). onehot: fixed class→100D projection (GAN baseline without EEG path).",
    )
    p.add_argument("--sanity_check", action="store_true", help="Tiny subset, fast epochs, previews every epoch.")
    p.add_argument(
        "--sanity_epochs",
        type=int,
        default=25,
        help="Number of epochs when --sanity_check is set (ignores --epochs).",
    )
    p.add_argument("--sanity_images_per_class", type=int, default=40)
    p.add_argument(
        "--sanity_classes",
        type=str,
        default="0,1",
        help="Comma-separated class ids for sanity mode (default: 0,1).",
    )
    p.add_argument(
        "--overfit_one_batch",
        action="store_true",
        help="Memorize a single batch (many steps/epoch). Use short --epochs.",
    )
    p.add_argument("--overfit_steps_per_epoch", type=int, default=100)
    p.add_argument("--debug_log_first_epochs", type=int, default=3)
    p.add_argument("--debug_log_batches", type=int, default=2)
    p.add_argument(
        "--image_range",
        choices=["tanh", "unit"],
        default="tanh",
        help="tanh: reals in [-1,1] (recommended). unit: [0,1] legacy for old checkpoints.",
    )
    p.add_argument("--preview_seed", type=int, default=42)
    p.add_argument(
        "--preview_fixed",
        action="store_true",
        help="Use the same noise+EEG for every saved preview (default on sanity). "
        "On full training, default is OFF: new random batch each preview (epoch-seeded) so images look less 'stuck'.",
    )
    p.add_argument(
        "--skip_inception",
        action="store_true",
        help="Skip slow inception score during long runs.",
    )
    p.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Smooth BCE targets for D (e.g. 0.1 → real=0.9, fake=0.1). Reduces D overconfidence; helps G learn structure. Use 0 to match older runs.",
    )
    p.add_argument(
        "--g_lr",
        type=float,
        default=None,
        help="Generator Adam LR (default: 3e-5). Try 1e-4 with --d_lr 4e-5 if previews stay noisy.",
    )
    p.add_argument(
        "--d_lr",
        type=float,
        default=None,
        help="Discriminator Adam LR (default: 5e-5). Slightly lower than G often stabilizes.",
    )
    p.add_argument(
        "--g_steps",
        type=int,
        default=1,
        help="Generator train_on_batch steps per batch after each D update (default 1). Try 2 if D is too strong.",
    )
    p.add_argument(
        "--d_aux_loss_weight",
        type=float,
        default=0.5,
        help="Weight on auxiliary classifier loss vs BCE (default 0.5). Lower = more emphasis on real/fake.",
    )
    return p.parse_args()


if __name__ == "__main__":
    _a = _parse_args()
    sc = [int(x.strip()) for x in _a.sanity_classes.split(",") if x.strip()]
    train(
        epochs=_a.epochs,
        resume_from_epoch=_a.resume_from_epoch,
        conditioning_mode=_a.conditioning_mode,
        sanity_check=_a.sanity_check,
        sanity_epochs=_a.sanity_epochs,
        sanity_class_ids=sc,
        sanity_images_per_class=_a.sanity_images_per_class,
        overfit_one_batch=_a.overfit_one_batch,
        overfit_steps_per_epoch=_a.overfit_steps_per_epoch,
        debug_log_first_epochs=_a.debug_log_first_epochs,
        debug_log_batches=_a.debug_log_batches,
        image_range=_a.image_range,
        preview_seed=_a.preview_seed,
        skip_inception=_a.skip_inception,
        preview_fixed=_a.preview_fixed,
        label_smoothing=_a.label_smoothing,
        g_lr=_a.g_lr,
        d_lr=_a.d_lr,
        g_steps=_a.g_steps,
        d_aux_loss_weight=_a.d_aux_loss_weight,
    )
