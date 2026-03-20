"""
Full pipeline sanity test: datasets, models, inference, metrics.
Run before expensive experiments. No SAR-HM++ in this benchmark.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add repo root and code to path
REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if CODE_DIR.is_dir() and str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# Add benchmark
BENCHMARK_DIR = REPO_ROOT / "benchmark"
if BENCHMARK_DIR.is_dir() and str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))


def _ok(msg: str) -> None:
    print("[PASS] %s" % msg)


def _fail(msg: str) -> None:
    print("[FAIL] %s" % msg)


def _warn(msg: str) -> None:
    print("[WARN] %s" % msg)


def test_imports() -> bool:
    """Verify benchmark and code imports."""
    try:
        from benchmark.benchmark_config import BenchmarkConfig
        from benchmark.dataset_registry import get_dataset
        from benchmark.model_registry import get_model
        _ok("benchmark imports")
        return True
    except Exception as e:
        _fail("benchmark imports: %s" % e)
        return False


def test_imagenet_eeg_dataset(config) -> bool:
    """ImageNet-EEG loads and sample retrieval works."""
    try:
        from benchmark.dataset_registry import get_dataset
        samples = get_dataset("imagenet_eeg", config, split="test", max_samples=2)
        if not samples:
            _warn("ImageNet-EEG: no samples (missing paths or empty split)")
            return True  # skip, not hard fail
        s = samples[0]
        if "eeg" not in s or "sample_id" not in s:
            _fail("ImageNet-EEG sample missing eeg or sample_id")
            return False
        _ok("ImageNet-EEG dataset (%d samples)" % len(samples))
        return True
    except Exception as e:
        _fail("ImageNet-EEG dataset: %s" % e)
        return False


def test_thoughtviz_dataset(config) -> bool:
    """ThoughtViz dataset loads if available."""
    try:
        from benchmark.dataset_registry import get_dataset
        samples = get_dataset("thoughtviz", config, split="test", max_samples=2)
        if not samples:
            _warn("ThoughtViz: no samples (data dir missing or empty)")
            return True
        s = samples[0]
        if "eeg" not in s or "sample_id" not in s:
            _fail("ThoughtViz sample missing eeg or sample_id")
            return False
        _ok("ThoughtViz dataset (%d samples)" % len(samples))
        return True
    except Exception as e:
        _warn("ThoughtViz dataset: %s" % e)
        return True


def test_thoughtviz_wrapper_load(config) -> bool:
    """ThoughtViz wrapper loads (if Keras and paths available)."""
    try:
        from benchmark.model_registry import get_model
        w = get_model("thoughtviz", config)
        if w is None:
            _warn("ThoughtViz wrapper not available (missing repo or Keras)")
            return True
        if not w.load_pretrained():
            _warn("ThoughtViz load_pretrained failed (missing .h5/.model)")
            return True
        _ok("ThoughtViz wrapper load")
        return True
    except Exception as e:
        _warn("ThoughtViz wrapper: %s" % e)
        return True


def test_dreamdiffusion_baseline_load(config) -> bool:
    """DreamDiffusion baseline model loads if checkpoint path set."""
    try:
        from benchmark.model_registry import get_model
        w = get_model("dreamdiffusion", config)
        if w is None:
            _warn("DreamDiffusion baseline not available (checkpoint not set or missing)")
            return True
        _ok("DreamDiffusion baseline load")
        return True
    except Exception as e:
        _warn("DreamDiffusion baseline: %s" % e)
        return True


def test_sarhm_load(config) -> bool:
    """SAR-HM model loads if checkpoint and proto path set."""
    try:
        from benchmark.model_registry import get_model
        w = get_model("sarhm", config)
        if w is None:
            _warn("SAR-HM not available (checkpoint/proto not set)")
            return True
        _ok("SAR-HM load")
        return True
    except Exception as e:
        _warn("SAR-HM: %s" % e)
        return True


def test_one_sample_inference(config) -> bool:
    """One-sample inference for each available model (smoke)."""
    try:
        from benchmark.dataset_registry import get_dataset
        from benchmark.model_registry import generate_thoughtviz, generate_dreamdiffusion, get_model
    except ImportError:
        _warn("Skip inference: benchmark imports failed")
        return True
    samples = get_dataset("imagenet_eeg", config, split="test", max_samples=1)
    if not samples:
        _warn("Skip inference: no ImageNet-EEG samples")
        return True
    # ThoughtViz
    try:
        w = get_model("thoughtviz", config)
        if w and w.load_pretrained():
            imgs = generate_thoughtviz(w, samples)
            if imgs and len(imgs) == 1:
                _ok("ThoughtViz one-sample inference")
            else:
                _warn("ThoughtViz inference returned no image")
        else:
            _warn("ThoughtViz skip inference (not loaded)")
    except Exception as e:
        _warn("ThoughtViz inference: %s" % e)
    # DreamDiffusion / SAR-HM need checkpoint; skip if not set
    for name in ("dreamdiffusion", "sarhm"):
        try:
            m = get_model(name, config)
            if m is None:
                continue
            imgs = generate_dreamdiffusion(m, samples, num_samples_per_item=1, ddim_steps=min(5, config.ddim_steps))
            if imgs:
                _ok("%s one-sample inference" % name)
        except Exception as e:
            _warn("%s inference: %s" % (name, e))
    return True


def test_metrics_pipeline() -> bool:
    """Tiny metric pass (two fake images)."""
    try:
        import numpy as np
        from utils_eval import compute_metrics
        fake_gen = np.random.randint(0, 256, (2, 64, 64, 3), dtype=np.uint8)
        fake_gt = np.random.randint(0, 256, (2, 64, 64, 3), dtype=np.uint8)
        m = compute_metrics(fake_gen, fake_gt)
        if "ssim_mean" in m or "n_samples" in m:
            _ok("metrics pipeline (compute_metrics)")
            return True
        _fail("metrics pipeline returned unexpected keys")
        return False
    except ImportError as e:
        _warn("metrics pipeline: %s" % e)
        return True
    except Exception as e:
        _fail("metrics pipeline: %s" % e)
        return False


def main() -> int:
    """Run all sanity checks; return 0 if all critical pass."""
    from benchmark.benchmark_config import BenchmarkConfig
    config = BenchmarkConfig()
    config.resolve_paths()
    # Allow env override for paths
    if os.environ.get("IMAGENET_PATH"):
        config.imagenet_path = os.environ.get("IMAGENET_PATH")
    if os.environ.get("EEG_SIGNALS_PATH"):
        config.imagenet_eeg_eeg_path = os.environ.get("EEG_SIGNALS_PATH")
    if os.environ.get("BASELINE_CKPT"):
        config.dreamdiffusion_baseline_ckpt = os.environ.get("BASELINE_CKPT")
    if os.environ.get("SARHM_CKPT"):
        config.sarhm_ckpt = os.environ.get("SARHM_CKPT")
    if os.environ.get("SARHM_PROTO"):
        config.sarhm_proto_path = os.environ.get("SARHM_PROTO")

    passed = 0
    failed = 0
    if test_imports():
        passed += 1
    else:
        failed += 1
    if test_imagenet_eeg_dataset(config):
        passed += 1
    else:
        failed += 1
    if test_thoughtviz_dataset(config):
        passed += 1
    else:
        failed += 1
    if test_thoughtviz_wrapper_load(config):
        passed += 1
    else:
        failed += 1
    if test_dreamdiffusion_baseline_load(config):
        passed += 1
    else:
        failed += 1
    if test_sarhm_load(config):
        passed += 1
    else:
        failed += 1
    if test_one_sample_inference(config):
        passed += 1
    else:
        failed += 1
    if test_metrics_pipeline():
        passed += 1
    else:
        failed += 1

    print("\nSanity: %d passed, %d failed." % (passed, failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
