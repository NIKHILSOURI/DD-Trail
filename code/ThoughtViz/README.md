# ThoughtViz
Implementation for the paper https://dl.acm.org/citation.cfm?doid=3240508.3240641

## Python environment (TensorFlow / Keras)

ThoughtViz uses **legacy Keras 2** APIs (`keras`, MoG layer, GAN training). Install **TensorFlow + Keras** in a **separate virtualenv** from the main DreamDiffusion PyTorch project (avoids dependency clashes).

From the **repository root** (parent of `code/`):

```bash
python3 -m venv venv_thoughtviz
source venv_thoughtviz/bin/activate   # Windows: venv_thoughtviz\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-thoughtviz.txt
```

Do **not** run `pip install -e ./code` in this environment — it installs `dreamdiffusion`, which expects PyTorch (`torch`, `timm`, `tqdm`) and will conflict with TensorFlow. If you see pip errors about missing `torch`/`timm` for `dreamdiffusion`, run: `pip uninstall dreamdiffusion -y`.

Verify:

```bash
python -c "import keras, tensorflow as tf; print('keras', keras.__version__, 'tf', tf.__version__)"
```

**GPU:** If `nvidia-smi` works but TensorFlow prints `Could not find cuda drivers`, install CUDA 11 libraries via pip and export `LD_LIBRARY_PATH` (TensorFlow 2.13 is built against CUDA 11.8, not only the host driver):

```bash
pip install -r ../../requirements-thoughtviz-gpu.txt   # from repo root, or use full path
source ./thoughtviz_gpu_env.sh
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Pinned stack** (see `requirements-thoughtviz.txt` in repo root): `tensorflow==2.13.1`, `keras==2.13.1`, plus `numpy`, `Pillow`, `scikit-learn`, `six`, `h5py`, `protobuf`. Optional GPU wheels: `requirements-thoughtviz-gpu.txt`.

Then run training from `code/ThoughtViz` with that venv active. Use **`bash run_with_gpu.sh training/thoughtviz_with_eeg.py`** — the launcher sets **CUDA `LD_LIBRARY_PATH`** and **`PYTHONPATH`** to the ThoughtViz root (required so `import utils.*` resolves; running `python training/...` alone puts only `training/` on `sys.path` and breaks imports).

## EEG Data

* Download the EEG data from [here](https://drive.google.com/file/d/1atP9CsjWIT-hg3fX--fcC1hg0uvg9bEH/view?usp=drive_link)
* Extract it and place it in the project folder (.../ThoughtViz/data)

## Image Data

* Downwload the images used to train the models from [here](https://drive.google.com/file/d/1x32IulYeBVmkshEKweijMX3DK1Wu8odx/view?usp=drive_link)
* Extract them and place them under `code/ThoughtViz/training/images/` (e.g. `Char-Font/`, `ImageNet-Filtered/`). Training scripts resolve paths from the ThoughtViz root, not the shell cwd.

## Trained Models

* Download the trained EEG Classification models from [here](https://drive.google.com/file/d/1cq8RTBiwqO-Jo0TZjBNlRHZEhjKDknKP/view?usp=drive_link)
* Extract them and place in the models folder (.../ThoughtViz/models/eeg_models)
* Download the trained image classifier models used in training from [here](https://drive.google.com/file/d/1U9qtN1SlOS3dzd2BwWWHhJiMz_0yNW9U/view?usp=drive_link)
* Extract them and place in the training folder (.../ThoughtViz/training/trained_classifier_models)

## Training

1. EEG Classification

2. GAN Training

## Testing

* Download the sample trained GAN models from [here](https://drive.google.com/open?id=1uFFhvTsU2nmdaecR2WPWsiGJfgI3as1_)
* Extract them and place in the models folder (.../ThoughtViz/models/gan_models)

* Run test.py to run the sample tests 

   1. Baseline Evaluation

      * DeLiGAN : Uses 1-hot class label as conditioning with MoGLayer at the input.

   2. Final Evaluation

      * Our Approach : Uses EEG encoding from the trained EEG classifier as conditioning. The encoding is used as weights in the MoGLayer
 
**NOTE** : Currently we have uploaded only one baseline model and our final model. Other models can be obtained by running the training code. 



