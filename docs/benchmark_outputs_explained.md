# Outputs from `new commands.md` §12–§15

Commands referenced: **§12** unified benchmark, **§13** metrics, **§14** tables, **§15** panels.  
Default run name in those snippets: **`thesis_final_a100`** (smoke uses **`thesis_smoke_a100`**).

If you run §12 with **`--models dreamdiffusion sarhm`** (recommended in the main PyTorch venv), **`thoughtviz.png`** is **not** produced per sample; only DreamDiffusion + SAR-HM images appear.

Root layout:

```text
$REPO_ROOT/results/experiments/<run_name>/
├── benchmark_outputs/          # §12 primary tree; §13–§15 read/write here
├── summary_metrics/            # written during §12 when summary is enabled; used by §14
└── segmentation_metrics/       # written during §12 when segmentation is enabled; used by §14
```

Replace `<run_name>` with your `--run_name` (e.g. `thesis_final_a100`).

---

## §12 — `python -m benchmark.compare_all_models`

Runs generation for **ImageNet-EEG** and **ThoughtViz** when `--dataset both`, then optional caption/summary and segmentation passes.

### How many images are generated?

After loading, the log line **`Dataset <name>: N EEG samples × G gen/sample → … images per model`** gives **N** (samples in that split) and **G** (`num_samples_per_item`, default 1). Each selected model produces **N × G** images for that dataset. Example: **`--max_samples 50`** with `--dataset both` → **N = 50** for `imagenet_eeg` and **N = 50** for `thoughtviz` (100 samples total across datasets, per model).

| Dataset | What **N** is |
|--------|----------------|
| **`imagenet_eeg`** | Test split from `block_splits_by_image_single.pth`, further restricted to indices with a **readable ImageNet** file (`valid_indices`). Typical order ~**164** with the stock split + dataset; not the full 1995 EEG rows. |
| **`thoughtviz`** | All **`x_test`** rows in `data.pkl`, unless `--max_samples K` with **K > 0** (then `min(len(x_test), K)`). **`--max_samples 0`** means “no cap” (use full test set). |

### Per-dataset, per-sample (`benchmark_outputs/<dataset>/sample_<id>/`)

| File / folder | Content |
|----------------|--------|
| `ground_truth.png` | Reference image (256 px eval size when saved). |
| `thoughtviz.png` | ThoughtViz generator output. |
| `dreamdiffusion.png` | DreamDiffusion baseline output. |
| `sarhm.png` | SAR-HM output. |
| `metadata.json` | Sample id, dataset, timing/errors metadata. |
| `summaries/` | **If** `--summary_enabled true`: `summary_gt.json`, `summary_<model>.json`, `summary_comparison.json` per sample. |
| `segmentation/` | **If** `--segmentation_enabled true`: GT/pred masks, JSON, overlays under `gt/`, `thoughtviz/`, `dreamdiffusion/`, `sarhm/`, plus `segmentation_comparison.json`. |

`<dataset>` is `imagenet_eeg` or `thoughtviz`.

### Aggregates next to `benchmark_outputs/` (same experiment folder)

| Path | Content |
|------|--------|
| `summary_metrics/<dataset>/summary_metrics.csv` | Per-row summary metrics per sample/model. |
| `summary_metrics/<dataset>/summary_metrics.json` | Means by model (`by_model`). |
| `segmentation_metrics/<dataset>/segmentation_metrics.csv` | Per-sample segmentation metrics. |
| `segmentation_metrics/<dataset>/segmentation_metrics.json` | Aggregated means by model. |

---

## §13 — `run_all_metrics` on `benchmark_outputs`

Extends what is already there:

| Path | Content |
|------|--------|
| `benchmark_outputs/<dataset>/metrics_summary.json` | Core image metrics (MSE, SSIM, PCC, CLIP, FID, Top-1/5 when computed) **per model**, from `ground_truth` vs each `*.png`. |

Re-runs summary and segmentation eval if enabled in `BenchmarkConfig` (same outputs as §12 for those pipelines).

---

## §14 — `generate_all_tables`

Reads metrics under `benchmark_outputs` and (if present) `summary_metrics` / `segmentation_metrics` beside it.

| Path | Content |
|------|--------|
| `results/experiments/<run_name>/tables/table_imagenet_eeg.csv` | Core metrics (MSE, SSIM, PCC, CLIP, FID, Top-1/5, …) per model. |
| `results/experiments/<run_name>/tables/table_thoughtviz.csv` | Same for ThoughtViz dataset. |
| `.../tables/table_summary_comparison.csv` | Built when `summary_metrics/<dataset>/summary_metrics.json` exists. |
| `.../tables/table_segmentation_comparison.csv` | Built when `segmentation_metrics/<dataset>/segmentation_metrics.json` exists. |

---

## §15 — `run_visualization`

| Path | Content |
|------|--------|
| `benchmark_outputs/<dataset>/panels/sample_<id>.png` | Horizontal strip: GT \| ThoughtViz \| DreamDiffusion \| SAR-HM. |
| `.../panels/sample_<id>_summary.png` | Image row + short caption text from `summaries/`. |
| `.../panels/sample_<id>_segmentation.png` | Concatenated segmentation overlays (if present). |

Up to `max_panels` samples per dataset (50 in the snippet).

---

## Notes

- **§13** assumes `BENCH_OUT` points at the same `benchmark_outputs` tree produced by **§12** for that run (e.g. `thesis_final_a100`). If you only ran smoke (`thesis_smoke_a100`), set `BENCH_OUT` to that run’s `benchmark_outputs` or change paths in §14–§15.
- ThoughtViz needs valid EEG + GAN paths and a working TensorFlow env in the process that runs the benchmark; otherwise `thoughtviz.png` may be missing for those samples.
