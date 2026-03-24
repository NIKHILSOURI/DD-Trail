from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from benchmark.utils import ensure_dir, setup_logger
from benchmark.visualization_runner import run_visualization

log = setup_logger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate per-sample and dataset-level comparison grids")
    ap.add_argument("--config", type=str, default="configs/benchmark_unified.yaml")
    ap.add_argument("--max_panels", type=int, default=50)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    out = cfg.get("output", {})
    run_name = out.get("run_name", "thesis_unified")
    root = Path(out.get("root", "results/benchmark_unified")).resolve()
    bench_out = root / run_name / "benchmark_outputs"

    for ds in ("imagenet_eeg", "thoughtviz"):
        run_visualization(bench_out, ds, max_panels=args.max_panels)

    combined = root / run_name / "combined" / "figures"
    ensure_dir(combined)
    log.info("Comparison grids generated under %s and dataset panel dirs", combined)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
