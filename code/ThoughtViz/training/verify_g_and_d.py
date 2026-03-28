#!/usr/bin/env python3
"""
One-shot check that generator + discriminator forward passes work with real data shapes.

Does NOT train — use this before a long run to confirm wiring, EEG features, and image ranges.
Also run `thoughtviz_image_with_eeg.py --overfit_one_batch` for a short learning smoke test.

Usage (from code/ThoughtViz, same as run_with_gpu.sh):
  bash run_with_gpu.sh training/verify_g_and_d.py
  python training/verify_g_and_d.py --checkpoint saved_models/thoughtviz_image_with_eeg/Image/run_1/generator_0
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import tensorflow as tf

try:
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
except Exception:
    pass

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image

import utils.data_input_util as inutil
from utils.keras_compat import set_learning_phase_inference
from utils.pickle_compat import load_pickle_compat
from training.models.thoughtviz import (
    discriminator_model_rgb,
    generator_model_rgb,
)
from utils.gan_image_norm import tensor_to_image_uint8
from utils import thoughtviz_paths as tv_paths


def main() -> int:
    p = argparse.ArgumentParser(description="Verify G and D forward passes (no training).")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to SavedModel dir for generator (e.g. saved_models/.../generator_0). "
        "If omitted, uses a freshly initialized generator.",
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_learning_phase_inference()
    rng = np.random.RandomState(args.seed)

    subset = "image"
    imagenet_folder = tv_paths.training_images("ImageNet-Filtered")
    classifier_model_file = tv_paths.trained_image_classifier(subset)
    eeg_data_dir = tv_paths.data_eeg("Image")
    eeg_classifier_model_file = tv_paths.eeg_classifier_model("Image")
    tv_paths.validate_image_eeg_prereqs(
        imagenet_folder=imagenet_folder,
        classifier_h5=classifier_model_file,
        eeg_data_dir=eeg_data_dir,
        eeg_classifier_h5=eeg_classifier_model_file,
    )

    input_noise_dim = 100
    feature_dim = 100
    num_classes = 10

    # Real images in [-1, 1] to match training + generator tanh
    x_train, _, _, _ = inutil.load_image_data(imagenet_folder, patch_size=(64, 64), normalize_to_tanh=True)
    bs = min(args.batch_size, x_train.shape[0])
    real = x_train[:bs].astype(np.float32)

    eeg_data = load_pickle_compat(os.path.join(eeg_data_dir, "data.pkl"))
    classifier = load_model(eeg_classifier_model_file)
    x_test_eeg = eeg_data[b"x_test"]
    y_test_eeg = eeg_data[b"y_test"]
    y_test_int = np.array([np.argmax(y) for y in y_test_eeg])
    layer_index = 9
    get_nth_layer_output = K.function(
        [classifier.layers[0].input], [classifier.layers[layer_index].output]
    )
    layer_output = get_nth_layer_output([x_test_eeg])[0]

    labels = rng.randint(0, num_classes, size=bs)
    eeg_vecs = np.array(
        [layer_output[rng.choice(np.where(y_test_int == int(lb))[0])] for lb in labels],
        dtype=np.float32,
    )
    noise = rng.uniform(-1, 1, (bs, input_noise_dim)).astype(np.float32)

    c_img = load_model(classifier_model_file)

    if args.checkpoint:
        print(f"[verify] Loading generator from {args.checkpoint}", flush=True)
        g = load_model(args.checkpoint)
    else:
        print("[verify] Building fresh generator (random weights)", flush=True)
        g = generator_model_rgb(input_noise_dim, feature_dim)

    print("[verify] Building discriminator (random weights)", flush=True)
    d = discriminator_model_rgb((64, 64), c_img)

    print("[verify] Generator forward…", flush=True)
    fake = g.predict([noise, eeg_vecs], verbose=0)
    assert fake.shape == (bs, 64, 64, 3), fake.shape
    print(f"  fake tensor: shape={fake.shape} min/max/mean={fake.min():.4f}/{fake.max():.4f}/{fake.mean():.4f}", flush=True)

    # Single-tile PNG (first sample) + small grid if bs>1
    out_dir = tv_paths.outputs_dir("smoke_verify")
    os.makedirs(out_dir, exist_ok=True)
    one = tensor_to_image_uint8(fake[0], from_tanh=True)
    path_one = os.path.join(out_dir, "generator_one_sample.png")
    Image.fromarray(one, mode="RGB").save(path_one)
    print(f"[verify] Saved one full decode: {path_one}", flush=True)

    if bs > 1:
        from utils.image_utils import combine_rgb_preview_grid

        grid = combine_rgb_preview_grid(fake, from_tanh=True)
        path_grid = os.path.join(out_dir, "generator_batch_grid.png")
        Image.fromarray(np.asarray(grid), mode="RGB").save(path_grid)
        print(f"[verify] Saved mini grid: {path_grid}", flush=True)

    print("[verify] Discriminator forward (real then fake)…", flush=True)
    dr = d.predict(real, verbose=0)
    df = d.predict(fake, verbose=0)
    p_real = float(np.mean(dr[0]))
    p_fake = float(np.mean(df[0]))
    print(f"  D(real) sigmoid mean: {p_real:.4f}  (expect ~0.5 for untrained D)", flush=True)
    print(f"  D(fake) sigmoid mean: {p_fake:.4f}", flush=True)
    if hasattr(dr[1], "shape"):
        print(f"  aux real shape: {dr[1].shape}  aux fake shape: {df[1].shape}", flush=True)

    print("\n[verify] OK — generator produced RGB images; discriminator ran on real and fake.", flush=True)
    print("  Next: run a short learning test, e.g. --overfit_one_batch --epochs 20", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
