from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from .benchmark_config import BenchmarkConfig
from .utils import setup_logger

log = setup_logger(__name__)


@dataclass
class SummaryArtifacts:
    short_caption: str
    detailed_caption: str
    objects_mentioned: List[str]
    scene_type: str
    attributes: List[str]
    uncertainty_notes: str
    model_name: str
    runtime_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "short_caption": self.short_caption,
            "detailed_caption": self.detailed_caption,
            "objects_mentioned": self.objects_mentioned,
            "scene_type": self.scene_type,
            "attributes": self.attributes,
            "uncertainty_notes": self.uncertainty_notes,
            "model_name": self.model_name,
            "runtime_sec": self.runtime_sec,
        }


def _normalize_token(t: str) -> str:
    t = t.strip().lower()
    t = re.sub(r"[^a-z0-9_\- ]+", "", t)
    return t


def _extract_objects_and_attributes(text: str) -> Dict[str, List[str]]:
    text_l = text.lower()
    object_vocab = {
        "person", "man", "woman", "child", "dog", "cat", "car", "bus", "truck", "bike", "bicycle",
        "motorcycle", "bird", "horse", "cow", "sheep", "table", "chair", "tree", "flower",
        "phone", "watch", "wallet", "apple", "tiger", "scooter", "road", "building", "sky", "water",
    }
    attr_vocab = {
        "red", "blue", "green", "yellow", "black", "white", "small", "large", "big", "bright",
        "dark", "sunny", "cloudy", "indoor", "outdoor", "wooden", "metal", "plastic", "old", "new",
    }
    found_obj = sorted({w for w in object_vocab if re.search(r"\b%s\b" % re.escape(w), text_l)})
    found_attr = sorted({w for w in attr_vocab if re.search(r"\b%s\b" % re.escape(w), text_l)})
    return {"objects": found_obj, "attributes": found_attr}


def _infer_scene_type(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("street", "road", "car", "bus", "building")):
        return "urban"
    if any(k in t for k in ("tree", "forest", "mountain", "river", "lake")):
        return "nature"
    if any(k in t for k in ("room", "table", "chair", "bed", "kitchen")):
        return "indoor"
    return "unknown"


class SummaryModel:
    """
    Caption/summary generator with Florence-2 default and BLIP fallback.
    Also exposes sentence embeddings and CLIP text-image scoring.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._device = "cpu"
        self._caption_pipe = None
        self._caption_model_name = None
        self._sent_model = None
        self._clip_model = None
        self._clip_processor = None

    def _load_caption_model(self) -> None:
        if self._caption_pipe is not None:
            return
        from transformers import pipeline
        try:
            self._caption_pipe = pipeline("image-to-text", model=self.config.florence2_model_id)
            self._caption_model_name = self.config.florence2_model_id
        except Exception as e:
            log.warning("Florence-2 load failed (%s). Falling back to %s", e, self.config.summary_fallback_model_id)
            self._caption_pipe = pipeline("image-to-text", model=self.config.summary_fallback_model_id)
            self._caption_model_name = self.config.summary_fallback_model_id

    def _load_sentence_model(self) -> None:
        if self._sent_model is None:
            from sentence_transformers import SentenceTransformer
            self._sent_model = SentenceTransformer(self.config.summary_sentence_model_id)

    def _load_clip(self) -> None:
        if self._clip_model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self._device)
        self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def summarize(self, image: np.ndarray, tag: str) -> SummaryArtifacts:
        self._load_caption_model()
        pil = Image.fromarray(image.astype(np.uint8))
        t0 = time.perf_counter()
        out = self._caption_pipe(pil)
        runtime = time.perf_counter() - t0
        text = ""
        if isinstance(out, list) and out:
            text = out[0].get("generated_text", "") or out[0].get("caption", "")
        text = (text or "").strip()
        if not text:
            text = "No reliable caption generated."
        parsed = _extract_objects_and_attributes(text)
        return SummaryArtifacts(
            short_caption=text[:180],
            detailed_caption=text,
            objects_mentioned=parsed["objects"],
            scene_type=_infer_scene_type(text),
            attributes=parsed["attributes"],
            uncertainty_notes="auto-generated summary",
            model_name=self._caption_model_name or "unknown",
            runtime_sec=runtime,
        )

    def sentence_cosine(self, a: str, b: str) -> float:
        import numpy as np
        self._load_sentence_model()
        va = self._sent_model.encode([a])[0]
        vb = self._sent_model.encode([b])[0]
        denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-8
        return float(np.dot(va, vb) / denom)

    def clip_text_image_score(self, image: np.ndarray, text: str) -> float:
        import torch
        self._load_clip()
        pil = Image.fromarray(image.astype(np.uint8))
        inputs = self._clip_processor(text=[text], images=[pil], return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            outputs = self._clip_model(**inputs)
            logits = outputs.logits_per_image  # [1,1]
        return float(logits[0, 0].item())
