"""
Unified benchmark for ThoughtViz, DreamDiffusion baseline, and DreamDiffusion + SAR-HM.
No SAR-HM++ in this benchmark (separate paper).
"""
from pathlib import Path

BENCHMARK_MODELS = ("thoughtviz", "dreamdiffusion", "sarhm")
BENCHMARK_DATASETS = ("imagenet_eeg", "thoughtviz")
BENCHMARK_ROOT = Path(__file__).resolve().parent.parent

__all__ = ["BENCHMARK_MODELS", "BENCHMARK_DATASETS", "BENCHMARK_ROOT"]
