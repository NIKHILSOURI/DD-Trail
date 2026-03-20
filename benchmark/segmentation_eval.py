"""
Instance segmentation label/mask comparison (mandatory).
Compatibility wrapper around segmentation_runner.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .benchmark_config import BenchmarkConfig
from .segmentation_runner import run_segmentation_eval as _run_seg
from .utils import setup_logger

log = setup_logger(__name__)


def run_segmentation_eval(
    output_dir: Path,
    dataset_name: str,
    max_samples: Optional[int] = None,
    config: Optional[BenchmarkConfig] = None,
) -> Dict[str, Any]:
    """Run mandatory segmentation pipeline and return aggregate metrics."""
    cfg = config or BenchmarkConfig()
    return _run_seg(output_dir, dataset_name, cfg)
