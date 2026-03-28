"""Load pickles produced with Python 2 / legacy protocols (ThoughtViz `data.pkl`)."""
from __future__ import annotations

import pickle


def load_pickle_compat(path: str):
    """Try default unpickling; on UnicodeDecodeError use ``encoding='bytes'`` (Py2 str → bytes keys)."""
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="bytes", fix_imports=True)
