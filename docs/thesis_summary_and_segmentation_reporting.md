# Thesis Reporting Notes: Summary + Segmentation Metrics

This document explains what can be honestly claimed from mandatory summary and segmentation outputs.

## 1) Summary metrics meaning

- `summary_semantic_similarity`:
  cosine similarity between sentence embeddings of GT summary text vs generated-image summary text.
  - Type: proxy semantic metric (text-text).
- `clip_text_image_score`:
  CLIP compatibility score between generated summary text and its image.
  - Type: proxy multimodal consistency metric.
- `object_mention_precision/recall/F1`:
  overlap of extracted object mentions from summary text.
  - Type: proxy object semantics from language.
- `attribute_overlap`:
  overlap of extracted visual attributes from summary text.
  - Type: proxy attribute semantics from language.

## 2) Segmentation metrics meaning

- `label_precision/recall/F1`, `label_set_iou`:
  set-level object label agreement between generated and GT detections.
  - Type: direct detector-output comparison metric.
- `matched_bbox_iou_mean`, `matched_mask_iou_mean`, `matched_dice_mean`:
  geometric overlap quality for matched instances.
  - Type: approximate spatial consistency metric (matching is algorithmic).
- `hallucination_rate`, `omission_rate`:
  generated-only objects and missed GT objects rates.
  - Type: direct count discrepancy metric.

## 3) Exact vs approximate

Exact:
- File existence checks, per-sample artifact saving, deterministic post-processing logic.
- Label normalization and overlap formulas.

Approximate / proxy:
- Open-vocabulary detections from Grounding DINO can vary with threshold/model.
- Instance matching is deterministic but approximate.
- Summary structuring is parser-based from model-generated text.
- CLIP-based compatibility is a proxy for semantic grounding, not full semantic correctness.

## 4) Honest claims guidance

You can claim:
- Mandatory summary and segmentation evaluations were run for all benchmark samples where images existed.
- Aggregate model comparisons were computed with consistent code paths.

You should avoid claiming:
- Absolute object truth from open-vocabulary detectors.
- Perfect instance correspondences.
- Human-equivalent caption understanding.

Use wording like:
- "proxy semantic metrics"
- "detector-based object consistency"
- "approximate instance matching"
