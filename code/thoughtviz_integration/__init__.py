"""
ThoughtViz integration for the unified benchmark.
Wraps the official ThoughtViz repo (code/ThoughtViz or codes/ThoughtViz) for inference and dataset access.
"""
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent  # code/thoughtviz_integration
_REPO_ROOT = _CODE_DIR.parent.parent  # repo root (code/thoughtviz_integration -> code -> repo)


def get_thoughtviz_root() -> Path | None:
    """Return ThoughtViz root if it exists under code/ThoughtViz or codes/ThoughtViz."""
    for sub in ("code", "codes"):
        candidate = _REPO_ROOT / sub / "ThoughtViz"
        if candidate.is_dir():
            return candidate
    if (_CODE_DIR.parent / "ThoughtViz").is_dir():
        return _CODE_DIR.parent / "ThoughtViz"
    return None


__all__ = ["get_thoughtviz_root"]
