# New Commands (Thesis Execution Playbook)

Scope: **baseline + SAR-HM + ThoughtViz** (no SAR-HM++).

All commands assume:
- repo root: `D:\STUDY\BTH\THESIS\DREAMDIFFUSION`
- venv is active
- A100 80GB available

---

## 0) One-time environment setup

```powershell
cd D:\STUDY\BTH\THESIS\DREAMDIFFUSION
$env:PYTHONPATH="D:\STUDY\BTH\THESIS\DREAMDIFFUSION;D:\STUDY\BTH\THESIS\DREAMDIFFUSION\code;D:\STUDY\BTH\THESIS\DREAMDIFFUSION\benchmark"
$env:IMAGENET_PATH="D:\STUDY\BTH\THESIS\DREAMDIFFUSION\datasets\imageNet_images"
```

Optional GPU sanity:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

---

## 1) Short training runs (10–15 epochs)

## 1.1 Baseline (15 epochs)

```powershell
python code/eeg_ldm.py --dataset EEG --run_mode baseline --pretrain_mbm_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\pretrains\eeg_pretain\checkpoint.pth --imagenet_path $env:IMAGENET_PATH --splits_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\datasets\block_splits_by_image_single.pth --eeg_signals_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\datasets\eeg_5_95_std.pth --num_epoch 15 --batch_size 16 --precision bf16 --seed 2022
```

## 1.2 SAR-HM (15 epochs)

```powershell
python code/eeg_ldm.py --dataset EEG --run_mode sarhm --pretrain_mbm_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\pretrains\eeg_pretain\checkpoint.pth --imagenet_path $env:IMAGENET_PATH --splits_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\datasets\block_splits_by_image_single.pth --eeg_signals_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\datasets\eeg_5_95_std.pth --num_epoch 15 --batch_size 16 --precision bf16 --seed 2022 --save_epoch_stats true --save_best_by_clip true
```

## 1.3 ThoughtViz (10–15 epochs)

ThoughtViz script currently has epochs hardcoded in `code/ThoughtViz/training/thoughtviz_with_eeg.py` (`epochs = 500`).  
For short run, first edit that value to `10` or `15`, then run:

```powershell
cd D:\STUDY\BTH\THESIS\DREAMDIFFUSION\code\ThoughtViz
python training\thoughtviz_with_eeg.py
cd D:\STUDY\BTH\THESIS\DREAMDIFFUSION
```

---

## 2) Full training

## 2.1 Baseline full (500 epochs)

```powershell
python code/eeg_ldm.py --dataset EEG --run_mode baseline --pretrain_mbm_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\pretrains\eeg_pretain\checkpoint.pth --imagenet_path $env:IMAGENET_PATH --splits_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\datasets\block_splits_by_image_single.pth --eeg_signals_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\datasets\eeg_5_95_std.pth --num_epoch 500 --batch_size 32 --precision bf16 --seed 2022
```

## 2.2 SAR-HM full (500 epochs)

```powershell
python code/eeg_ldm.py --dataset EEG --run_mode sarhm --pretrain_mbm_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\pretrains\eeg_pretain\checkpoint.pth --imagenet_path $env:IMAGENET_PATH --splits_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\datasets\block_splits_by_image_single.pth --eeg_signals_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\datasets\eeg_5_95_std.pth --num_epoch 500 --batch_size 32 --precision bf16 --seed 2022 --save_epoch_stats true --save_best_by_clip true
```

## 2.3 ThoughtViz full (max quality)

Use the native ThoughtViz training script with high epoch count (default currently 500).  
For max quality, keep `epochs = 500` (or higher only if validated by your loss curves and outputs).

```powershell
cd D:\STUDY\BTH\THESIS\DREAMDIFFUSION\code\ThoughtViz
python training\thoughtviz_with_eeg.py
cd D:\STUDY\BTH\THESIS\DREAMDIFFUSION
```

---

## 3) Compare baseline vs SAR-HM (Stage C compare)

```powershell
python code/compare_eval.py --dataset EEG --splits_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\datasets\block_splits_by_image_single.pth --eeg_signals_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\datasets\eeg_5_95_std.pth --config_patch D:\STUDY\BTH\THESIS\DREAMDIFFUSION\pretrains\models\config15.yaml --imagenet_path $env:IMAGENET_PATH --baseline_ckpt D:\STUDY\BTH\THESIS\DREAMDIFFUSION\results\exps\results\generation\Baseline\checkpoint_best.pth --sarhm_ckpt D:\STUDY\BTH\THESIS\DREAMDIFFUSION\results\exps\results\generation\Sarhm\checkpoint_best.pth --sarhm_proto D:\STUDY\BTH\THESIS\DREAMDIFFUSION\results\exps\results\generation\Sarhm\prototypes.pt --n_samples 100 --ddim_steps 250 --seed 2022 --out_dir D:\STUDY\BTH\THESIS\DREAMDIFFUSION\results\compare_eval_final
```

---

## 4) Full unified benchmark (all datasets, all models)

Includes mandatory image summary and instance segmentation evaluation.

```powershell
python -m benchmark.compare_all_models --dataset both --max_samples 0 --run_name thesis_final_a100 --imagenet_path $env:IMAGENET_PATH --baseline_ckpt D:\STUDY\BTH\THESIS\DREAMDIFFUSION\results\exps\results\generation\Baseline\checkpoint_best.pth --sarhm_ckpt D:\STUDY\BTH\THESIS\DREAMDIFFUSION\results\exps\results\generation\Sarhm\checkpoint_best.pth --sarhm_proto D:\STUDY\BTH\THESIS\DREAMDIFFUSION\results\exps\results\generation\Sarhm\prototypes.pt --thoughtviz_data_dir D:\STUDY\BTH\THESIS\DREAMDIFFUSION\code\ThoughtViz\data --thoughtviz_image_dir D:\STUDY\BTH\THESIS\DREAMDIFFUSION\code\ThoughtViz\training\images --summary_enabled true --segmentation_enabled true --strict_eval true
```

Note:
- `--max_samples 0` means no explicit benchmark limit; set to `10/20` for smoke test.

---

## 5) Run metrics for final benchmark

```powershell
python -c "import sys; sys.path.insert(0,'code'); sys.path.insert(0,'benchmark'); from benchmark.metrics_runner import run_all_metrics; from benchmark.benchmark_config import BenchmarkConfig; c=BenchmarkConfig(); c.summary_enabled=True; c.segmentation_enabled=True; out=r'results/experiments/thesis_final_a100/benchmark_outputs'; print(run_all_metrics(out,'imagenet_eeg',c)); print(run_all_metrics(out,'thoughtviz',c))"
```

This runs:
- core metrics
- summary metrics
- segmentation metrics

---

## 6) Inference timing measurement (per model)

```powershell
python -c "import sys; sys.path.insert(0,'benchmark'); from pathlib import Path; from benchmark.benchmark_config import BenchmarkConfig; from benchmark.timing_runner import run_timing, save_timing_table; c=BenchmarkConfig(); c.imagenet_path=r'D:\STUDY\BTH\THESIS\DREAMDIFFUSION\datasets\imageNet_images'; c.dreamdiffusion_baseline_ckpt=r'D:\STUDY\BTH\THESIS\DREAMDIFFUSION\results\exps\results\generation\Baseline\checkpoint_best.pth'; c.sarhm_ckpt=r'D:\STUDY\BTH\THESIS\DREAMDIFFUSION\results\exps\results\generation\Sarhm\checkpoint_best.pth'; c.sarhm_proto_path=r'D:\STUDY\BTH\THESIS\DREAMDIFFUSION\results\exps\results\generation\Sarhm\prototypes.pt'; c.thoughtviz_data_dir=r'D:\STUDY\BTH\THESIS\DREAMDIFFUSION\code\ThoughtViz\data'; c.thoughtviz_image_dir=r'D:\STUDY\BTH\THESIS\DREAMDIFFUSION\code\ThoughtViz\training\images'; c.resolve_paths(); r1=run_timing('imagenet_eeg', c, max_samples=20); save_timing_table(r1, Path(r'results/experiments/thesis_final_a100/timing/imagenet_timing')); r2=run_timing('thoughtviz', c, max_samples=20); save_timing_table(r2, Path(r'results/experiments/thesis_final_a100/timing/thoughtviz_timing')); print('done')"
```

---

## 7) Training timing measurement (wall-clock per model)

PowerShell `Measure-Command` examples:

## 7.1 Baseline training timing
```powershell
Measure-Command { python code/eeg_ldm.py --dataset EEG --run_mode baseline --pretrain_mbm_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\pretrains\eeg_pretain\checkpoint.pth --imagenet_path $env:IMAGENET_PATH --splits_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\datasets\block_splits_by_image_single.pth --eeg_signals_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\datasets\eeg_5_95_std.pth --num_epoch 15 --batch_size 16 --precision bf16 --seed 2022 } | Tee-Object -FilePath results\experiments\thesis_final_a100\timing\baseline_train_time_15ep.txt
```

## 7.2 SAR-HM training timing
```powershell
Measure-Command { python code/eeg_ldm.py --dataset EEG --run_mode sarhm --pretrain_mbm_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\pretrains\eeg_pretain\checkpoint.pth --imagenet_path $env:IMAGENET_PATH --splits_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\datasets\block_splits_by_image_single.pth --eeg_signals_path D:\STUDY\BTH\THESIS\DREAMDIFFUSION\datasets\eeg_5_95_std.pth --num_epoch 15 --batch_size 16 --precision bf16 --seed 2022 } | Tee-Object -FilePath results\experiments\thesis_final_a100\timing\sarhm_train_time_15ep.txt
```

## 7.3 ThoughtViz training timing
```powershell
cd D:\STUDY\BTH\THESIS\DREAMDIFFUSION\code\ThoughtViz
Measure-Command { python training\thoughtviz_with_eeg.py } | Tee-Object -FilePath D:\STUDY\BTH\THESIS\DREAMDIFFUSION\results\experiments\thesis_final_a100\timing\thoughtviz_train_time.txt
cd D:\STUDY\BTH\THESIS\DREAMDIFFUSION
```

---

## 8) Generate result tables

```powershell
python -c "import sys; sys.path.insert(0,'benchmark'); from benchmark.table_generator import generate_all_tables; generate_all_tables(r'results/experiments/thesis_final_a100/benchmark_outputs', r'results/experiments/thesis_final_a100/tables')"
```

---

## 9) Generate qualitative visualization panels

```powershell
python -c "import sys; sys.path.insert(0,'benchmark'); from pathlib import Path; from benchmark.visualization_runner import run_visualization; out=Path(r'results/experiments/thesis_final_a100/benchmark_outputs'); run_visualization(out,'imagenet_eeg',max_panels=50); run_visualization(out,'thoughtviz',max_panels=50)"
```

Panels include:
- core image panel
- summary panel
- segmentation overlay panel

---

## 10) A100 80GB practical settings

- Use `--precision bf16` for baseline/SAR-HM training.
- Increase `--batch_size` cautiously:
  - baseline/SAR-HM: try 16/24/32 and validate stability
- Keep `--ddim_steps 250` for final quality comparisons.
- Use `--strict_eval true` in final benchmark to ensure mandatory summary + segmentation do not silently skip.
- For reproducibility: always include `--seed 2022`.

