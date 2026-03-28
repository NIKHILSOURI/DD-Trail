from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
from tqdm import tqdm

from benchmark.benchmark_config import BenchmarkConfig
from benchmark.metrics_runner import run_all_metrics
from benchmark.table_generator import generate_all_tables
from benchmark.timing_runner import run_timing, save_timing_table
from benchmark.utils import ensure_dir, setup_logger

log = setup_logger(__name__)


def _rows_to_markdown(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def _rows_to_latex(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    cols = list(rows[0].keys())
    lines = [
        "\\begin{tabular}{" + "l" * len(cols) + "}",
        " \\hline",
        " & ".join(cols) + " \\\\",
        " \\hline",
    ]
    for r in rows:
        lines.append(" & ".join(str(r.get(c, "")) for c in cols) + " \\\\")
    lines.extend([" \\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _csv_to_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_multi_format(csv_path: Path, out_stem: Path) -> None:
    rows = _csv_to_rows(csv_path)
    if not rows:
        return
    out_stem.with_suffix(".csv").write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    out_stem.with_suffix(".md").write_text(_rows_to_markdown(rows), encoding="utf-8")
    out_stem.with_suffix(".tex").write_text(_rows_to_latex(rows), encoding="utf-8")


def make_thesis_tables(root: Path, run_name: str) -> None:
    bench_out = root / run_name / "benchmark_outputs"
    combined_tables = root / run_name / "combined" / "tables"
    ensure_dir(combined_tables)
    temp_tables = root / run_name / "tables_tmp"
    ensure_dir(temp_tables)
    generate_all_tables(bench_out, temp_tables)

    mapping = {
        "table_imagenet_eeg.csv": "imagenet_benchmark",
        "table_thoughtviz.csv": "thoughtviz_benchmark",
        "table_summary_comparison.csv": "model_comparison",
        "table_segmentation_comparison.csv": "segmentation_comparison",
    }
    for src, dst in mapping.items():
        write_multi_format(temp_tables / src, combined_tables / dst)

    timing_src = root / run_name / "combined" / "timing" / "timing_summary.csv"
    write_multi_format(timing_src, combined_tables / "inference_timing")


def config_from_yaml(path: Path) -> tuple[BenchmarkConfig, Dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    p = raw.get("paths", {})
    out = raw.get("output", {})
    ev = raw.get("evaluation", {})
    c = BenchmarkConfig()
    c.imagenet_path = p.get("imagenet_path")
    c.imagenet_eeg_eeg_path = p.get("eeg_signals_path")
    c.imagenet_eeg_splits_path = p.get("splits_path")
    c.thoughtviz_data_dir = p.get("thoughtviz_data_dir")
    c.thoughtviz_image_dir = p.get("thoughtviz_image_dir")
    c.thoughtviz_eeg_model_path = p.get("thoughtviz_eeg_model_path")
    c.thoughtviz_gan_model_path = p.get("thoughtviz_gan_model_path")
    c.dreamdiffusion_baseline_ckpt = p.get("baseline_ckpt")
    c.sarhm_ckpt = p.get("sarhm_ckpt")
    c.sarhm_proto_path = p.get("sarhm_proto")
    c.models = list(raw.get("models", c.models))
    c.max_samples = raw.get("max_samples", c.max_samples)
    c.ddim_steps = int(raw.get("ddim_steps", c.ddim_steps))
    c.show_progress = bool(raw.get("show_progress", c.show_progress))
    c.summary_enabled = bool(ev.get("summary_enabled", True))
    c.segmentation_enabled = bool(ev.get("segmentation_enabled", True))
    c.strict_eval = bool(ev.get("strict_eval", False))
    run_name = out.get("run_name", "thesis_unified")
    root = Path(out.get("root", "results/benchmark_unified")).resolve()
    c.output_dir = str(root / run_name / "benchmark_outputs")
    c.resolve_paths()
    return c, raw


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute all benchmark metrics and thesis tables")
    ap.add_argument("--config", type=str, default="configs/benchmark_unified.yaml")
    args = ap.parse_args()

    cfg, raw = config_from_yaml(Path(args.config))
    out = raw.get("output", {})
    run_name = out.get("run_name", "thesis_unified")
    root = Path(out.get("root", "results/benchmark_unified")).resolve()
    bench_out = Path(cfg.output_dir)

    all_results: Dict[str, Any] = {}
    for ds in tqdm(("imagenet_eeg", "thoughtviz"), desc="Metrics datasets", unit="dataset"):
        log.info("Metrics stage: %s", ds)
        all_results[ds] = run_all_metrics(bench_out, ds, config=cfg)
        t = run_timing(ds, cfg, max_samples=min(cfg.max_samples or 10, 10))
        timing_dir = root / run_name / "combined" / "timing"
        ensure_dir(timing_dir)
        save_timing_table(t, timing_dir / f"{ds}_timing")

    summary_json = root / run_name / "combined" / "logs" / "metrics_all.json"
    ensure_dir(summary_json.parent)
    summary_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    log.info("Wrote metrics bundle: %s", summary_json)

    # Merge per-dataset timing csv to timing_summary.csv
    timing_rows: List[Dict[str, Any]] = []
    for ds in ("imagenet_eeg", "thoughtviz"):
        p = root / run_name / "combined" / "timing" / f"{ds}_timing.csv"
        timing_rows.extend(_csv_to_rows(p))
    timing_summary = root / run_name / "combined" / "timing" / "timing_summary.csv"
    if timing_rows:
        with timing_summary.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
            w.writeheader()
            w.writerows(timing_rows)

    make_thesis_tables(root, run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
