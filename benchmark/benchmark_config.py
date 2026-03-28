"""
Benchmark configuration: paths, model names, limits, output layout.
All options are optional with safe defaults; no SAR-HM++ in this benchmark.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class BenchmarkConfig:
    """Configuration for the unified benchmark (ThoughtViz, DreamDiffusion, SAR-HM only)."""

    # Datasets
    root_path: Path = field(default_factory=lambda: BENCHMARK_ROOT)
    imagenet_eeg_eeg_path: Optional[str] = None  # e.g. datasets/eeg_5_95_std.pth
    imagenet_eeg_splits_path: Optional[str] = None  # e.g. datasets/block_splits_by_image_single.pth
    imagenet_path: Optional[str] = None  # ImageNet root for EEG GT images
    thoughtviz_data_dir: Optional[str] = None  # ThoughtViz EEG/data dir
    thoughtviz_image_dir: Optional[str] = None  # ThoughtViz training/images
    thoughtviz_root: Optional[str] = None  # code/ThoughtViz or codes/ThoughtViz

    # Models (paths)
    thoughtviz_eeg_model_path: Optional[str] = None
    thoughtviz_gan_model_path: Optional[str] = None
    dreamdiffusion_baseline_ckpt: Optional[str] = None
    sarhm_ckpt: Optional[str] = None
    sarhm_proto_path: Optional[str] = None
    config_patch: str = "pretrains/models/config15.yaml"
    pretrain_root: str = "pretrains"

    # Run control
    max_samples: Optional[int] = None  # None = all; 10, 20 for smoke
    seed: int = 2022
    models: List[str] = field(default_factory=lambda: ["thoughtviz", "dreamdiffusion", "sarhm"])
    datasets: List[str] = field(default_factory=lambda: ["imagenet_eeg", "thoughtviz"])

    # Output
    output_dir: str = "results/benchmark_outputs"
    experiments_dir: str = "results/experiments"
    run_name: Optional[str] = None
    eval_size: int = 256  # Standard size for metric evaluation (e.g. 256x256)

    # Optional
    show_progress: bool = True  # tqdm + stage logs for benchmark CLI
    ddim_steps: int = 250
    num_samples_per_item: int = 1  # Generated samples per EEG sample
    # Mandatory evaluation components
    summary_enabled: bool = True
    segmentation_enabled: bool = True
    strict_eval: bool = False
    # If True, refuse ThoughtViz load when checkpoint family mismatches dataset.
    # Keep False for "run all models without skipping" workflows.
    thoughtviz_strict_checkpoint_match: bool = False
    # Summary/caption models
    florence2_model_id: str = "microsoft/Florence-2-base"
    summary_fallback_model_id: str = "Salesforce/blip-image-captioning-base"
    summary_sentence_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Segmentation models/checkpoints
    grounding_dino_model_id: str = "IDEA-Research/grounding-dino-base"
    grounding_dino_fallback_model_id: str = "google/owlv2-base-patch16-ensemble"
    grounding_dino_checkpoint_path: Optional[str] = None
    sam2_model_id: str = "facebook/sam-vit-base"
    sam2_config_path: Optional[str] = None
    sam2_checkpoint_path: Optional[str] = None

    def resolve_paths(self) -> None:
        """Resolve relative paths against root_path."""
        r = self.root_path
        if self.imagenet_eeg_eeg_path and not Path(self.imagenet_eeg_eeg_path).is_absolute():
            self.imagenet_eeg_eeg_path = str(r / self.imagenet_eeg_eeg_path)
        if self.imagenet_eeg_splits_path and not Path(self.imagenet_eeg_splits_path).is_absolute():
            self.imagenet_eeg_splits_path = str(r / self.imagenet_eeg_splits_path)
        if self.config_patch and not Path(self.config_patch).is_absolute():
            self.config_patch = str(r / self.config_patch)
        if self.pretrain_root and not Path(self.pretrain_root).is_absolute():
            self.pretrain_root = str(r / self.pretrain_root)
        if self.output_dir and not Path(self.output_dir).is_absolute():
            self.output_dir = str(r / self.output_dir)
        if self.experiments_dir and not Path(self.experiments_dir).is_absolute():
            self.experiments_dir = str(r / self.experiments_dir)
        if self.grounding_dino_checkpoint_path and not Path(self.grounding_dino_checkpoint_path).is_absolute():
            self.grounding_dino_checkpoint_path = str(r / self.grounding_dino_checkpoint_path)
        if self.sam2_config_path and not Path(self.sam2_config_path).is_absolute():
            self.sam2_config_path = str(r / self.sam2_config_path)
        if self.sam2_checkpoint_path and not Path(self.sam2_checkpoint_path).is_absolute():
            self.sam2_checkpoint_path = str(r / self.sam2_checkpoint_path)
