"""
Benchmark utilities: path checks, logging, safe I/O.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent


def setup_logger(name: str = "benchmark", level: int = logging.INFO) -> logging.Logger:
    """Create a benchmark logger."""
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
        log.addHandler(h)
    log.setLevel(level)
    return log


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def check_path(
    path: Optional[str | Path],
    label: str = "path",
    must_exist: bool = True,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Return True if path is valid; log warning and return False otherwise."""
    if path is None or (isinstance(path, str) and not path.strip()):
        if logger:
            logger.warning("%s is not set.", label)
        return False
    p = Path(path)
    if must_exist and not p.exists():
        if logger:
            logger.warning("%s does not exist: %s", label, p)
        return False
    return True


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load JSON file; return {} on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save dict to JSON file."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
