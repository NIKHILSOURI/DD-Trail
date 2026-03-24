# Benchmark, Environment Issues, and Dependency Conflicts

This document explains:

1. What the unified benchmark runs
2. Why there are still environment issues
3. Why two virtual environments (`venv` + `venv_thoughtviz`) are used
4. What the unified environment (`venv_unified`) changes
5. Dependencies per model and known conflicts

## 1) Benchmark Scope (What is being compared)

The unified benchmark compares three EEG-to-image model paths:

- ThoughtViz
- DreamDiffusion baseline
- DreamDiffusion + SAR-HM

Datasets used:

- `imagenet_eeg`
- `thoughtviz`

Standard output tree per run:

- `results/experiments/<run_name>/benchmark_outputs/<dataset>/sample_<id>/`
  - `ground_truth.png`
  - `thoughtviz.png`
  - `dreamdiffusion.png`
  - `sarhm.png`
  - `metadata.json`
  - optional summary/segmentation artifacts

Main post-processing:

- core metrics JSON (`metrics_summary.json`)
- tables (`table_imagenet_eeg.csv`, `table_thoughtviz.csv`, and optional summary/segmentation tables)
- visualization panels

## 2) Main Issues (Why environment problems keep happening)

### A) `typing_extensions` hard conflict

- TensorFlow 2.13 (used by ThoughtViz legacy Keras flow) requires `typing_extensions<4.6`.
- Newer PyTorch lines (notably 2.7+) need newer `typing_extensions` and rely on symbols such as `TypeIs`.
- Result: a single environment can break either TensorFlow/ThoughtViz or PyTorch imports depending on what pip resolved last.

This is the most important root conflict.

### B) TensorFlow GPU runtime expectations vs PyTorch CUDA stack

- ThoughtViz TensorFlow 2.13 expects CUDA 11.x runtime libraries (often installed as pip NVIDIA packages and exported in `LD_LIBRARY_PATH`).
- PyTorch environments are often built against a different CUDA wheel/index strategy.
- Result: GPU works for one stack but not the other, or installing TF GPU libs destabilizes torch CUDA expectations.

### C) Legacy ThoughtViz API surface

- ThoughtViz relies on legacy Keras/TensorFlow 2.13 style APIs.
- Upgrading TensorFlow/Keras aggressively to match modern stacks risks code breakage.
- So ThoughtViz pins are conservative, which increases conflict pressure with modern torch-first environments.

### D) Benchmark process coupling

- Unified benchmark CLI can include summary/segmentation pipelines that add Hugging Face and other dependencies.
- Running all benchmark stages in one process magnifies dependency resolution risk.

## 3) Why Two Venvs Exist (`venv` and `venv_thoughtviz`)

## `venv` (main PyTorch env)

Used for:

- DreamDiffusion baseline
- SAR-HM
- torch-first benchmarking and metrics

Typical package source: `requirements.txt`.

## `venv_thoughtviz` (legacy TF/Keras env)

Used for:

- ThoughtViz model path
- optional TF GPU extras via `requirements-thoughtviz-gpu.txt`
- benchmark bridge deps via `requirements-thoughtviz-benchmark.txt` (when needed)

Typical package source:

- `requirements-thoughtviz.txt`
- optional `requirements-thoughtviz-gpu.txt`
- optional `requirements-thoughtviz-benchmark.txt`

## Why this split is still needed

Two envs remain the safer setup because they isolate:

- TensorFlow legacy constraints
- PyTorch modern constraints
- conflicting CUDA runtime packaging choices

This reduces break/fix loops from mixed pip resolutions.

## 4) Unified Venv (`requirements_unified.txt`) and Why It Is Still an Issue

The unified env attempts one-venv compatibility by using conservative pins:

- `torch==2.1.2` (pre-newest TypeIs pressure)
- `typing_extensions>=4.5.0,<4.6.0`
- `tensorflow==2.13.1`
- `keras==2.13.1`

This can work for a single benchmark venv, but it is still an issue because:

1. It intentionally prevents moving to newer torch lines quickly.
2. It keeps the project on older compatibility boundaries for core packages.
3. GPU runtime expectations for TF and torch can still diverge on some hosts.
4. If additional packages pull incompatible transitive dependencies, the lock can destabilize.
5. Legacy ThoughtViz API compatibility still limits upgrade flexibility.

So unified venv is a practical compromise, not a permanent conflict-free end state.

## 5) Dependencies by Model/Path

## DreamDiffusion baseline (PyTorch stack)

From `requirements.txt`:

- `setuptools==68.0.0`
- `typing_extensions>=4.12.0`
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `torchaudio>=2.0.0`
- `numpy>=1.24.0`
- `scipy>=1.10.0,<2.0.0`
- `pytorch-lightning>=2.0.5,<2.1.0`
- `torchmetrics>=0.11.0,<1.0.0`
- `Pillow>=10.0.0,<11.0.0`
- `opencv-python>=4.8.0,<5.0.0`
- `scikit-image>=0.21.0,<1.0.0`
- `kornia>=0.7.0,<1.0.0`
- `transformers>=4.30.0,<5.0.0`
- `accelerate>=0.30.0,<1.0.0`
- `sentence-transformers>=2.2.2,<4.0.0`
- `omegaconf>=2.3.0,<3.0.0`
- `einops>=0.6.1,<1.0.0`
- `tqdm>=4.65.0,<5.0.0`
- `pyyaml>=6.0,<7.0.0`
- `timm>=0.9.0,<1.0.0`
- `wandb>=0.15.0,<1.0.0`
- `torch-fidelity>=0.3.0,<1.0.0`
- `lpips>=0.1.4,<1.0.0`
- `natsort>=8.0.0,<9.0.0`
- `matplotlib>=3.7.0,<4.0.0`
- `scikit-learn>=1.3.0,<2.0.0`

## SAR-HM path

SAR-HM runs in the same PyTorch ecosystem as DreamDiffusion. It therefore uses the same core dependency family as above.

## ThoughtViz path (legacy TF/Keras stack)

From `requirements-thoughtviz.txt`:

- `tensorflow==2.13.1`
- `keras==2.13.1`
- `numpy>=1.23.0,<2.0.0`
- `Pillow>=9.0.0`
- `scikit-learn>=1.0.0`
- `six>=1.16.0`
- `h5py>=3.7.0`
- `tqdm>=4.65.0`
- `protobuf>=3.20.3,<5.0.0`

Optional TF GPU runtime extras (`requirements-thoughtviz-gpu.txt`):

- `nvidia-cuda-runtime-cu11==11.8.89`
- `nvidia-cudnn-cu11==8.6.0.163`
- `nvidia-cublas-cu11==11.11.3.6`
- `nvidia-cuda-nvrtc-cu11==11.8.89`
- `nvidia-cusolver-cu11==11.4.1.48`
- `nvidia-cufft-cu11==10.9.0.58`
- `nvidia-curand-cu11==10.3.0.86`
- `nvidia-cusparse-cu11==11.7.5.86`

Optional benchmark bridge deps (`requirements-thoughtviz-benchmark.txt`):

- `torch` (CPU index in file)
- `transformers>=4.35.0,<5`
- `accelerate>=0.20.0`
- `safetensors>=0.4.0`
- `sentence-transformers>=2.6.0`

## Unified benchmark venv

From `requirements_unified.txt`:

- `setuptools==68.0.0`
- `typing_extensions>=4.5.0,<4.6.0`
- `torch==2.1.2`
- `torchvision==0.16.2`
- `torchaudio==2.1.2`
- `tensorflow==2.13.1`
- `keras==2.13.1`
- `numpy>=1.24.0,<2.0.0`
- `scipy>=1.10.0,<2.0.0`
- `pandas>=2.0.0,<3.0.0`
- `scikit-learn>=1.3.0,<2.0.0`
- `scikit-image>=0.21.0,<1.0.0`
- `matplotlib>=3.7.0,<4.0.0`
- `Pillow>=10.0.0,<11.0.0`
- `opencv-python>=4.8.0,<5.0.0`
- `pytorch-lightning>=2.0.5,<2.1.0`
- `torchmetrics>=0.11.0,<1.0.0`
- `kornia>=0.7.0,<1.0.0`
- `timm>=0.9.0,<1.0.0`
- `einops>=0.6.1,<1.0.0`
- `omegaconf>=2.3.0,<3.0.0`
- `pyyaml>=6.0,<7.0.0`
- `tqdm>=4.65.0,<5.0.0`
- `natsort>=8.0.0,<9.0.0`
- `transformers==4.38.2`
- `accelerate>=0.20.0,<2.0.0`
- `safetensors>=0.4.0`
- `sentence-transformers==2.6.1`
- `wandb>=0.15.0,<1.0.0`
- `torch-fidelity>=0.3.0,<1.0.0`
- `lpips>=0.1.4,<1.0.0`
- `h5py>=3.7.0`
- `protobuf>=3.20.3,<5.0.0`

## 6) Conflict Matrix (Important)

### Confirmed/high-risk conflicts

1. `typing_extensions`
   - ThoughtViz TF 2.13 path: `<4.6`
   - modern torch path (esp. 2.7+): needs newer versions
   - impact: torch import failure or TF constraint break

2. CUDA runtime package expectations
   - TF 2.13 GPU often expects explicit CUDA 11.x pip libs + `LD_LIBRARY_PATH`
   - torch installs can rely on different wheel/index/runtime assumptions
   - impact: one framework sees GPU while the other fails

3. Mixed editable/package installs in ThoughtViz env
   - installing the `dreamdiffusion` package inside TF-only env can pull torch-family deps
   - impact: pip resolver churn, missing/overwritten deps, runtime instability

### Lower-risk but common friction points

1. Transformers/accelerate/sentence-transformers version drift between files
2. Optional metrics stacks adding transitive constraints
3. Different numpy ranges across stacks (generally manageable here, but watch upgrades)

## 7) Practical Recommendation

For reliability today:

- Keep **two-env execution** as the default operational path:
  - `venv` for DreamDiffusion/SAR-HM
  - `venv_thoughtviz` for ThoughtViz
- Use **`venv_unified` only when one-process benchmarking is required**, and accept conservative pins.
- Treat unified env as a controlled benchmark runtime, not as a general development environment.
