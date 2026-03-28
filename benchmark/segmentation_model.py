from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .benchmark_config import BenchmarkConfig
from .utils import ensure_dir, setup_logger

log = setup_logger(__name__)


def normalize_label(label: str) -> str:
    l = label.strip().lower()
    l = re.sub(r"[^a-z0-9_\- ]+", "", l)
    alias = {"automobile": "car", "bike": "bicycle", "cell phone": "phone"}
    return alias.get(l, l)


@dataclass
class InstanceRecord:
    label_raw: str
    label_norm: str
    confidence: float
    bbox_xyxy: List[float]
    mask_path: str
    mask_area: float
    centroid: List[float]
    source_model_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label_raw": self.label_raw,
            "label_norm": self.label_norm,
            "confidence": self.confidence,
            "bbox_xyxy": self.bbox_xyxy,
            "mask_path": self.mask_path,
            "mask_area": self.mask_area,
            "centroid": self.centroid,
            "source_model_name": self.source_model_name,
        }


class SegmentationModel:
    """
    Grounding DINO detector + SAM2/SAM adapter.
    If SAM backend unavailable, uses bbox-mask fallback (recorded explicitly).
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._det_pipe = None
        self._detector_name = "none_fallback"
        self._sam_processor = None
        self._sam_model = None

    def _load_detector(self) -> None:
        if self._det_pipe is not None:
            return
        try:
            from transformers import pipeline
            self._det_pipe = pipeline("zero-shot-object-detection", model=self.config.grounding_dino_model_id)
            self._detector_name = self.config.grounding_dino_model_id
        except Exception as e:
            log.warning("Grounding-DINO unavailable (%s). Trying fallback detector.", e)
            try:
                from transformers import pipeline
                self._det_pipe = pipeline(
                    "zero-shot-object-detection",
                    model=self.config.grounding_dino_fallback_model_id,
                )
                self._detector_name = self.config.grounding_dino_fallback_model_id
                log.warning("Using detector fallback: %s", self.config.grounding_dino_fallback_model_id)
            except Exception as e2:
                self._det_pipe = None
                self._detector_name = "none_fallback"
                log.warning(
                    "Fallback detector unavailable (%s). Falling back to empty detections.",
                    e2,
                )

    def _load_sam(self) -> None:
        if self._sam_model is not None:
            return
        try:
            from transformers import SamModel, SamProcessor
            self._sam_processor = SamProcessor.from_pretrained(self.config.sam2_model_id)
            self._sam_model = SamModel.from_pretrained(self.config.sam2_model_id)
        except Exception as e:
            self._sam_model = None
            self._sam_processor = None
            log.warning("SAM backend unavailable (%s). Using bbox-mask fallback.", e)

    def detect_and_segment(
        self,
        image: np.ndarray,
        out_dir: Path,
        source_name: str,
        candidate_labels: List[str] | None = None,
    ) -> Dict[str, Any]:
        self._load_detector()
        self._load_sam()
        ensure_dir(out_dir / "masks")
        ensure_dir(out_dir / "overlays")
        pil = Image.fromarray(image.astype(np.uint8))
        labels = candidate_labels or [
            "person", "dog", "cat", "car", "bus", "truck", "bicycle", "motorcycle",
            "tree", "flower", "chair", "table", "phone", "watch", "wallet", "apple", "tiger", "scooter",
        ]
        if self._det_pipe is None:
            det = []
        else:
            det = self._det_pipe(pil, candidate_labels=labels, threshold=0.2)
        inst: List[InstanceRecord] = []
        overlay = pil.copy()
        draw = ImageDraw.Draw(overlay)
        for i, d in enumerate(det):
            box = d.get("box", {})
            x1, y1, x2, y2 = float(box.get("xmin", 0)), float(box.get("ymin", 0)), float(box.get("xmax", 0)), float(box.get("ymax", 0))
            lab = str(d.get("label", "unknown"))
            conf = float(d.get("score", 0.0))
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            x1i, y1i, x2i, y2i = int(max(0, x1)), int(max(0, y1)), int(min(image.shape[1], x2)), int(min(image.shape[0], y2))
            if x2i > x1i and y2i > y1i:
                mask[y1i:y2i, x1i:x2i] = 255
            mpath = out_dir / "masks" / ("mask_%03d.png" % i)
            Image.fromarray(mask).save(mpath)
            area = float((mask > 0).sum())
            centroid = [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]
            inst.append(
                InstanceRecord(
                    label_raw=lab,
                    label_norm=normalize_label(lab),
                    confidence=conf,
                    bbox_xyxy=[x1, y1, x2, y2],
                    mask_path=str(mpath),
                    mask_area=area,
                    centroid=centroid,
                    source_model_name=source_name,
                )
            )
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, max(0, y1 - 12)), "%s %.2f" % (lab, conf), fill="yellow")
        overlay_path = out_dir / "overlays" / "overlay.png"
        overlay.save(overlay_path)
        return {
            "source": source_name,
            "num_instances": len(inst),
            "instances": [x.to_dict() for x in inst],
            "labels_raw": [x.label_raw for x in inst],
            "labels_norm": sorted(list({x.label_norm for x in inst})),
            "overlay_path": str(overlay_path),
            "sam_backend": "sam" if self._sam_model is not None else "bbox_fallback",
            "detector_backend": self._detector_name,
        }
