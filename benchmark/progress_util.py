"""
Optional tqdm. Minimal envs (e.g. venv_thoughtviz with TF only) may omit tqdm;
benchmark still runs without progress bars.
"""
from __future__ import annotations

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable=None, *args, **kwargs):  # type: ignore[no-redef]
        return iterable
