from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List


SAFE_TOKENS = (
    "benchmark_outputs",
    "benchmark_unified",
    "summary_metrics",
    "segmentation_metrics",
    "tables",
    "panels",
    "temp_eval",
    "generated_samples",
)
PROTECTED_TOKENS = ("checkpoint", "pretrain", "dataset", "code", "venv")


def looks_generated(path: Path) -> bool:
    s = str(path).lower()
    return any(tok in s for tok in SAFE_TOKENS) and not any(tok in s for tok in PROTECTED_TOKENS)


def collect_candidates(repo_root: Path) -> List[Path]:
    roots: Iterable[Path] = [
        repo_root / "results",
        repo_root / "benchmark_outputs",
    ]
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_dir() and looks_generated(p):
                out.append(p)
    # Deduplicate nested paths by keeping shortest parents.
    out = sorted(set(out), key=lambda p: len(str(p)))
    filtered: List[Path] = []
    for p in out:
        if not any(str(p).startswith(str(parent) + "/") for parent in filtered):
            filtered.append(p)
    return filtered


def main() -> int:
    ap = argparse.ArgumentParser(description="Safely clean generated benchmark artifacts")
    ap.add_argument("--repo_root", type=str, default="/workspace/project/DREAMDIFFUSION_RUNPOD")
    ap.add_argument("--yes", action="store_true", help="actually delete")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    candidates = collect_candidates(repo_root)

    print("[clean] candidate generated folders:")
    for p in candidates:
        print(f"  - {p}")

    if not args.yes:
        print("[clean] dry-run complete. Re-run with --yes to delete.")
        return 0

    for p in candidates:
        if p.exists():
            shutil.rmtree(p, ignore_errors=False)
    print(f"[clean] deleted {len(candidates)} folders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
