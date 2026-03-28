from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from tqdm import tqdm


def run(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run full unified thesis benchmark pipeline")
    ap.add_argument("--config", type=str, default="configs/benchmark_unified.yaml")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    cfg_path = (repo / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    out = cfg.get("output", {})
    run_name = out.get("run_name", "thesis_unified")
    root = Path(out.get("root", "results/benchmark_unified")).resolve()

    stages: list[tuple[str, list[str]]] = [("discover_assets", [sys.executable, "scripts/discover_benchmark_assets.py"])]
    if args.clean:
        stages.append(("clean_outputs", [sys.executable, "scripts/clean_benchmark_outputs.py", "--yes"]))
    inf_cmd = [sys.executable, "-m", "benchmark.orchestrate_all", "--config", str(cfg_path)]
    if args.max_samples is not None:
        inf_cmd += ["--max_samples", str(args.max_samples)]
    stages.append(("orchestrated_benchmark", inf_cmd))
    for name, cmd in tqdm(stages, desc="Pipeline stages", unit="stage"):
        print(f"[stage] {name}")
        run(cmd, repo)

    final_report = {
        "config": str(cfg_path),
        "run_name": run_name,
        "root": str(root / run_name),
        "benchmark_outputs": str(root / run_name / "benchmark_outputs"),
        "tables_dir": str(root / run_name / "tables"),
        "logs_dir": str(root / run_name / "combined" / "logs"),
    }
    out_json = root / run_name / "combined" / "final_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print("[done] final report:", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
