"""
Thesis-grade experiment logging: run folder, config dump, train/eval CSV logs.
Use MetricLogger.log_train() and log_eval(); config is written once at init.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Repo root (parent of code/)
_CODE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CODE_DIR.parent


def _sanitize_value(v: Any) -> Any:
    """Convert to JSON/CSV-serializable value. Use 'NA' for non-finite or inapplicable."""
    if v is None:
        return "NA"
    if isinstance(v, float):
        if v != v or v == float("inf") or v == float("-inf"):  # nan, inf
            return "NA"
        return round(v, 6)
    if isinstance(v, (int, str, bool)):
        return v
    if hasattr(v, "item"):
        return _sanitize_value(v.item())
    return str(v)


def _config_to_dict(config: Any) -> Dict[str, Any]:
    """Extract a JSON-serializable dict from a config object (e.g. with __dict__)."""
    if hasattr(config, "__dict__"):
        d = dict(config.__dict__)
    elif isinstance(config, dict):
        d = dict(config)
    else:
        return {"config": str(config)}
    out = {}
    for k, v in d.items():
        try:
            if isinstance(v, (type(Path), Path)):
                v = str(v)
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


class RunConfig:
    """Minimal run identifier: run_dir, model type, seed. Used to build run folder path."""

    def __init__(self, run_dir: str, model: str = "baseline", seed: int = 2022):
        self.run_dir = run_dir
        self.model = model
        self.seed = seed

    def artifacts_dir(self, dataset_name: str) -> Path:
        return Path(self.run_dir) / "artifacts" / dataset_name


class MetricLogger:
    """
    Writes train_log.csv, eval_log.csv, and config.json under a single run directory.
    CSV columns expand dynamically when new metric keys appear. Flushes after each write.
    """

    NA = "NA"

    def __init__(self, run_dir: str, config: Any, run_config: Optional[RunConfig] = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "artifacts").mkdir(exist_ok=True)
        self._run_config = run_config or RunConfig(run_dir, "baseline", 2022)
        self._train_headers: list = []
        self._eval_headers: list = []
        self._train_file: Optional[Any] = None
        self._eval_file: Optional[Any] = None
        self._train_writer: Optional[csv.DictWriter] = None
        self._eval_writer: Optional[csv.DictWriter] = None
        # Write config once
        config_path = self.run_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(_config_to_dict(config), f, indent=2)
        self._train_path = self.run_dir / "train_log.csv"
        self._eval_path = self.run_dir / "eval_log.csv"

    def _ensure_train_csv(self, row: Dict[str, Any]) -> None:
        if self._train_file is None:
            self._train_file = open(self._train_path, "w", newline="", encoding="utf-8")
            self._train_headers = ["step", "epoch"] + sorted(k for k in row if k not in ("step", "epoch"))
            self._train_writer = csv.DictWriter(
                self._train_file, fieldnames=self._train_headers, extrasaction="ignore"
            )
            self._train_writer.writeheader()
            self._train_file.flush()
        else:
            extra = set(row) - set(self._train_headers)
            if extra:
                self._train_headers = sorted(set(self._train_headers) | set(row))
                self._train_file.close()
                # Rewrite file with new header + existing rows (read back)
                rows = []
                with open(self._train_path, "r", newline="", encoding="utf-8") as fr:
                    rdr = csv.DictReader(fr)
                    rows = list(rdr)
                with open(self._train_path, "w", newline="", encoding="utf-8") as fw:
                    w = csv.DictWriter(fw, fieldnames=self._train_headers, extrasaction="ignore")
                    w.writeheader()
                    for r in rows:
                        w.writerow(r)
                self._train_file = open(self._train_path, "a", newline="", encoding="utf-8")
                self._train_writer = csv.DictWriter(
                    self._train_file, fieldnames=self._train_headers, extrasaction="ignore"
                )

    def _ensure_eval_csv(self, row: Dict[str, Any]) -> None:
        if self._eval_file is None:
            self._eval_file = open(self._eval_path, "w", newline="", encoding="utf-8")
            self._eval_headers = ["epoch", "dataset"] + sorted(k for k in row if k not in ("epoch", "dataset"))
            self._eval_writer = csv.DictWriter(
                self._eval_file, fieldnames=self._eval_headers, extrasaction="ignore"
            )
            self._eval_writer.writeheader()
            self._eval_file.flush()
        else:
            extra = set(row) - set(self._eval_headers)
            if extra:
                self._eval_headers = sorted(set(self._eval_headers) | set(row))
                self._eval_file.close()
                rows = []
                with open(self._eval_path, "r", newline="", encoding="utf-8") as fr:
                    rdr = csv.DictReader(fr)
                    rows = list(rdr)
                with open(self._eval_path, "w", newline="", encoding="utf-8") as fw:
                    w = csv.DictWriter(fw, fieldnames=self._eval_headers, extrasaction="ignore")
                    w.writeheader()
                    for r in rows:
                        w.writerow(r)
                self._eval_file = open(self._eval_path, "a", newline="", encoding="utf-8")
                self._eval_writer = csv.DictWriter(
                    self._eval_file, fieldnames=self._eval_headers, extrasaction="ignore"
                )

    def log_train(self, step: int, metrics_dict: Dict[str, Any], epoch: Optional[int] = None) -> None:
        """Append one row to train_log.csv. metrics_dict can include train/loss_total, train/alpha_mean, etc."""
        row = {"step": step}
        if epoch is not None:
            row["epoch"] = epoch
        for k, v in metrics_dict.items():
            row[k] = _sanitize_value(v) if v is not None and v != self.NA else self.NA
        self._ensure_train_csv(row)
        self._train_writer.writerow(row)
        self._train_file.flush()

    def log_eval(self, epoch: int, dataset_name: str, metrics_dict: Dict[str, Any]) -> None:
        """Append one row to eval_log.csv. Keys can be eval/<dataset>/fid, etc."""
        row = {"epoch": epoch, "dataset": dataset_name}
        for k, v in metrics_dict.items():
            row[k] = _sanitize_value(v) if v is not None and v != self.NA else self.NA
        self._ensure_eval_csv(row)
        self._eval_writer.writerow(row)
        self._eval_file.flush()

    def artifacts_dir(self, dataset_name: str) -> Path:
        p = self.run_dir / "artifacts" / dataset_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def close(self) -> None:
        if self._train_file is not None:
            self._train_file.close()
            self._train_file = None
        if self._eval_file is not None:
            self._eval_file.close()
            self._eval_file = None


def create_run_dir(
    base_dir: Optional[str] = None,
    model: str = "baseline",
    seed: int = 2022,
    run_name: Optional[str] = None,
) -> tuple[str, RunConfig]:
    """
    Create runs/<timestamp>_<model>_<seed>/ (or run_name if provided).
    Returns (run_dir path string, RunConfig).
    """
    from datetime import datetime

    base = Path(base_dir) if base_dir else _REPO_ROOT
    runs = base / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        folder = f"{ts}_{run_name}"
    else:
        folder = f"{ts}_{model}_{seed}"
    run_dir = str(runs / folder)
    os.makedirs(run_dir, exist_ok=True)
    (Path(run_dir) / "artifacts").mkdir(exist_ok=True)
    for d in ("imagenet_eeg", "thoughtviz", "moabb"):
        (Path(run_dir) / "artifacts" / d).mkdir(parents=True, exist_ok=True)
    return run_dir, RunConfig(run_dir, model=model, seed=seed)
