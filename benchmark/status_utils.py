from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .utils import ensure_dir


def _sample_dir(output_dir: str | Path, dataset_name: str, sample_id: str) -> Path:
    return Path(output_dir) / dataset_name / ("sample_%s" % str(sample_id).replace("/", "_"))


def load_sample_metadata(output_dir: str | Path, dataset_name: str, sample_id: str) -> Dict[str, Any]:
    d = _sample_dir(output_dir, dataset_name, sample_id)
    p = d / "metadata.json"
    if not p.exists():
        return {"sample_id": sample_id, "dataset": dataset_name}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"sample_id": sample_id, "dataset": dataset_name}


def update_model_status(
    output_dir: str | Path,
    dataset_name: str,
    sample_id: str,
    model_name: str,
    status: str,
    reason: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    d = _sample_dir(output_dir, dataset_name, sample_id)
    ensure_dir(d)
    p = d / "metadata.json"
    meta = load_sample_metadata(output_dir, dataset_name, sample_id)
    st = dict(meta.get("model_status") or {})
    item: Dict[str, Any] = {"status": status}
    if reason:
        item["reason"] = reason
    if extra:
        item.update(extra)
    st[model_name] = item
    meta["model_status"] = st
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def validate_image_array(img: Any) -> Tuple[bool, str]:
    if img is None:
        return False, "image is None"
    arr = np.asarray(img)
    if arr.size == 0:
        return False, "image is empty"
    if not np.isfinite(arr).all():
        return False, "image contains non-finite values"
    if arr.ndim not in (2, 3):
        return False, "image has invalid ndim=%s" % arr.ndim
    if arr.ndim == 3 and arr.shape[-1] not in (1, 3, 4):
        return False, "image has invalid channel count %s" % arr.shape[-1]
    # Quick trivial-image guard: near-constant outputs are not useful generations.
    arrf = arr.astype(np.float32)
    if float(arrf.std()) < 1e-3:
        return False, "image is near-constant"
    return True, "ok"
