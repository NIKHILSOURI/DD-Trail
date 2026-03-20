from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmark"))

from benchmark.benchmark_config import BenchmarkConfig
from benchmark.segmentation_runner import run_segmentation_eval


def main() -> int:
    cfg = BenchmarkConfig()
    cfg.resolve_paths()
    out = REPO / "results" / "benchmark_outputs"
    ds = "imagenet_eeg"
    try:
        r = run_segmentation_eval(out, ds, cfg)
        print("[PASS] segmentation pipeline", r.get("n_rows", 0))
        return 0
    except Exception as e:
        print("[FAIL] segmentation pipeline:", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
