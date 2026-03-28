from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "code"))


def check(name: str, timeout_sec: int = 120) -> None:
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys,importlib;"
            f"sys.path.insert(0,{repr(str(REPO_ROOT))});"
            f"sys.path.insert(0,{repr(str(REPO_ROOT / 'code'))});"
            f"importlib.import_module({name!r});"
            "print('ok')"
        ),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        print(f"[err] {name}: TimeoutExpired after {timeout_sec}s")
        raise
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        print(f"[err] {name}: {err}")
        raise RuntimeError(f"Import failed: {name}")
    print(f"[ok] {name}")


def main() -> int:
    modules = [
        "torch",
        "tensorflow",
        "keras",
        "benchmark.compare_all_models",
        "benchmark.metrics_runner",
        "benchmark.segmentation_runner",
        "benchmark.summary_runner",
        "thoughtviz_integration.model_wrapper",
        "thoughtviz_integration.dataset_adapter",
        "utils_eval",
        "transformers",
        "sentence_transformers",
    ]
    for mod in modules:
        check(mod)
    print("[ok] unified import smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
