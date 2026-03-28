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

**GPU:** If `nvidia-smi` works but TensorFlow prints `Could not find cuda drivers` or lists no GPUs, install CUDA 11 libraries via pip **and** set `LD_LIBRARY_PATH` in the **same shell** before starting Python. TensorFlow 2.13 is built against CUDA 11.x from pip; activating the venv alone does **not** load those `.so` files.

```bash
# from repository root (parent of code/):
pip install -r requirements-thoughtviz-gpu.txt
cd code/ThoughtViz
source ./thoughtviz_gpu_env.sh   # required every new terminal session
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

You should see `[PhysicalDevice(name='/physical_device:GPU:0', ...)]`. The script prints something like `prepended 8 nvidia lib dirs` (or more); if it only prepends **one** or **zero**, the pip CUDA packages are missing or `SITE` is wrong. **Do not** run the `python -c` check without `source ./thoughtviz_gpu_env.sh` first — that will falsely show no GPU even after a correct install. Training with `bash run_with_gpu.sh ...` sources this file for you.

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

### 1. EEG classification (prerequisite for EEG-conditioned GANs)

Train or obtain `run_final.h5` under `models/eeg_models/<char|image>/`. The training entry point in this tree is `training/eeg_classification.py` (run from the ThoughtViz root with the same `PYTHONPATH` / GPU setup as below). You must already have the matching `data/eeg/.../data.pkl` from the README data links.

### 2. Image-domain EEG-conditioned GAN → `models/gan_models/final/image/generator.model`

The benchmark and `testing/test.py` expect a **SavedModel-style** generator directory at:

`models/gan_models/final/image/generator.model`

That folder is **not** created by default; you either download it (Testing section below) or **train** it and then copy the trained generator into that path.

**Training script:** `training/thoughtviz_image_with_eeg.py`

**Required files on disk before training** (see `utils/thoughtviz_paths.py`):

| Item | Path |
|------|------|
| ImageNet-Filtered 64×64 images | `training/images/ImageNet-Filtered/` |
| EEG pickle for the image task | `data/eeg/image/data.pkl` |
| Trained EEG classifier (image) | `models/eeg_models/image/run_final.h5` |
| Image classifier for the discriminator | `training/trained_classifier_models/classifier_image.h5` |

**Run (Linux / Git Bash, GPU env):**

```bash
cd /path/to/DD-Trail/code/ThoughtViz
source /path/to/venv_thoughtviz_gpu/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
bash run_with_gpu.sh training/thoughtviz_image_with_eeg.py
```

**Run (Windows PowerShell):** `cd` to `code/ThoughtViz`, then (adjust path to your `venv_thoughtviz_gpu` if needed):

```powershell
cd <your-repo>\code\ThoughtViz
$env:PYTHONPATH = (Get-Location).Path
..\..\venv_thoughtviz_gpu\Scripts\python.exe training\thoughtviz_image_with_eeg.py
```

(If TensorFlow needs pip CUDA DLLs, install `requirements-thoughtviz-gpu.txt` in that venv first.)

**Checkpoints:** training writes under `saved_models/thoughtviz_image_with_eeg/Image/run_1/generator_<epoch>` (and `discriminator_<epoch>`). The default `epochs` in `train()` is **10000** — reasonable for paper-quality runs; for a smoke test, temporarily lower `epochs` in `thoughtviz_image_with_eeg.py` (e.g. 100–500) to verify the pipeline.

**Install into the layout the benchmark expects** (pick your best `generator_<epoch>` directory, e.g. the last one):

```text
models/gan_models/final/image/generator.model   ← copy contents of generator_<epoch> here (folder = SavedModel)
```

PowerShell example (run from `code/ThoughtViz`; change `generator_499` to your best epoch):

```powershell
$tvRoot = (Get-Location).Path
$dst = Join-Path $tvRoot "models\gan_models\final\image\generator.model"
$src = Join-Path $tvRoot "saved_models\thoughtviz_image_with_eeg\Image\run_1\generator_499"
New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
Remove-Item -Recurse -Force $dst -ErrorAction SilentlyContinue
Copy-Item -Recurse -Path $src -Destination $dst
```

Then set `paths.thoughtviz_gan_model_path` in `configs/benchmark_unified.yaml` to that `generator.model` path (or keep the default if it already matches).

**Note:** `saved_models/thoughtviz_with_eeg/Char/...` is a **different** task (characters). Training the **image** GAN requires the **ImageNet-Filtered** image pack and `data/eeg/image/data.pkl`, not the Char assets alone.

## Testing

* Download the sample trained GAN models from [here](https://drive.google.com/open?id=1uFFhvTsU2nmdaecR2WPWsiGJfgI3as1_)
* Extract them and place in the models folder (.../ThoughtViz/models/gan_models)

* Run test.py to run the sample tests 

   1. Baseline Evaluation

      * DeLiGAN : Uses 1-hot class label as conditioning with MoGLayer at the input.

   2. Final Evaluation

      * Our Approach : Uses EEG encoding from the trained EEG classifier as conditioning. The encoding is used as weights in the MoGLayer
 
**NOTE** : Currently we have uploaded only one baseline model and our final model. Other models can be obtained by running the training code. 



