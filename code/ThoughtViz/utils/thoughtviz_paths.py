"""Paths relative to the ThoughtViz repo root (independent of process cwd).

Layout (see README): training/images/, data/eeg/, models/eeg_models/, training/trained_classifier_models/
"""
from __future__ import annotations

import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def root() -> str:
    return _ROOT


def training_images(*parts: str) -> str:
    return os.path.join(_ROOT, "training", "images", *parts)


def data_eeg(subset: str) -> str:
    return os.path.join(_ROOT, "data", "eeg", subset.lower())


def eeg_classifier_model(subset: str, filename: str = "run_final.h5") -> str:
    return os.path.join(_ROOT, "models", "eeg_models", subset.lower(), filename)


def trained_image_classifier(name: str) -> str:
    return os.path.join(
        _ROOT, "training", "trained_classifier_models", f"classifier_{name.lower()}.h5"
    )


def saved_models(*parts: str) -> str:
    return os.path.join(_ROOT, "saved_models", *parts)


def outputs_dir(*parts: str) -> str:
    return os.path.join(_ROOT, "outputs", *parts)


def _readme_hint() -> str:
    return "See code/ThoughtViz/README.md for Google Drive links (EEG data, images, pretrained models)."


def validate_eeg_gan_prereqs(
    dataset: int,
    *,
    char_font_dir: str | None,
    classifier_h5: str | None,
    eeg_data_dir: str | None,
    eeg_classifier_h5: str | None,
) -> None:
    """Raise FileNotFoundError with actionable text before a long TF import/train."""
    errors: list[str] = []

    if dataset == 1 and char_font_dir and not os.path.isdir(char_font_dir):
        errors.append(
            f"Character font images not found: {char_font_dir}\n"
            "  Extract the image pack to training/images/Char-Font (under the ThoughtViz folder)."
        )

    if classifier_h5 and not os.path.isfile(classifier_h5):
        errors.append(
            f"Image classifier not found: {classifier_h5}\n"
            "  Place classifier_*.h5 under training/trained_classifier_models/."
        )

    if eeg_data_dir:
        pkl = os.path.join(eeg_data_dir, "data.pkl")
        if not os.path.isfile(pkl):
            errors.append(
                f"EEG data not found: {pkl}\n"
                "  Download EEG data and extract to data/eeg/<digit|char|image>/."
            )

    if eeg_classifier_h5 and not os.path.isfile(eeg_classifier_h5):
        errors.append(
            f"EEG classifier model not found: {eeg_classifier_h5}\n"
            "  Place run_final.h5 under models/eeg_models/<subset>/."
        )

    if errors:
        raise FileNotFoundError("\n\n".join(errors) + "\n\n" + _readme_hint())


def validate_label_gan_prereqs(dataset: int, *, char_font_dir: str | None, classifier_h5: str | None) -> None:
    """GAN training with class labels only (no EEG assets)."""
    errors: list[str] = []
    if dataset == 1 and char_font_dir and not os.path.isdir(char_font_dir):
        errors.append(
            f"Character font images not found: {char_font_dir}\n"
            "  Extract the image pack to training/images/Char-Font (under the ThoughtViz folder)."
        )
    if classifier_h5 and not os.path.isfile(classifier_h5):
        errors.append(
            f"Image classifier not found: {classifier_h5}\n"
            "  Place classifier_*.h5 under training/trained_classifier_models/."
        )
    if errors:
        raise FileNotFoundError("\n\n".join(errors) + "\n\n" + _readme_hint())


def validate_imagenet_filtered(imagenet_folder: str) -> None:
    if not os.path.isdir(imagenet_folder):
        raise FileNotFoundError(
            f"ImageNet-Filtered folder not found: {imagenet_folder}\n"
            "  Extract the image pack to training/images/ImageNet-Filtered.\n\n"
            + _readme_hint()
        )


def validate_image_eeg_prereqs(
    *,
    imagenet_folder: str,
    classifier_h5: str | None,
    eeg_data_dir: str | None,
    eeg_classifier_h5: str | None,
) -> None:
    validate_imagenet_filtered(imagenet_folder)
    validate_eeg_gan_prereqs(
        0,
        char_font_dir=None,
        classifier_h5=classifier_h5,
        eeg_data_dir=eeg_data_dir,
        eeg_classifier_h5=eeg_classifier_h5,
    )
