#!/usr/bin/env python3
"""
Optional wrapper to document and smoke-test ThoughtViz GAN validation flows.

Does not run full training by default — prints the recommended commands.
Use --run_smoke to execute a tiny subprocess import check only.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_TV_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description="ThoughtViz GAN pipeline validation helper")
    p.add_argument(
        "--run_smoke",
        action="store_true",
        help="Verify training module imports (no data, no training).",
    )
    args = p.parse_args()

    cmds = [
        "# From code/ThoughtViz (set PYTHONPATH=. or use run_with_gpu.sh):",
        "python training/thoughtviz_image_with_eeg.py --sanity_check --sanity_epochs 25",
        "python training/thoughtviz_image_with_eeg.py --overfit_one_batch --epochs 10",
        "python training/thoughtviz_image_with_eeg.py --sanity_check --conditioning_mode onehot --sanity_epochs 25",
    ]
    print("\n".join(cmds))

    if args.run_smoke:
        import os

        env = {**os.environ, "PYTHONPATH": str(_TV_ROOT)}
        r = subprocess.run(
            [sys.executable, str(_TV_ROOT / "training" / "thoughtviz_image_with_eeg.py"), "--help"],
            cwd=str(_TV_ROOT),
            env=env,
            check=False,
        )
        return int(r.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
