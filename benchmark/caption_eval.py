"""
Image summary comparison (mandatory).
Compatibility wrapper around summary_runner.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .benchmark_config import BenchmarkConfig
from .summary_runner import run_summary_eval
from .utils import setup_logger

log = setup_logger(__name__)


def run_caption_eval(
    output_dir: Path,
    dataset_name: str,
    max_samples: Optional[int] = None,
    config: Optional[BenchmarkConfig] = None,
) -> Dict[str, Any]:
    """Run mandatory summary/caption pipeline and return aggregate metrics."""
    cfg = config or BenchmarkConfig()
    return run_summary_eval(output_dir, dataset_name, cfg)
