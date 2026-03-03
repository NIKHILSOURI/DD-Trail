import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _find_train_logs(runs_dir: str, keyword: str) -> List[str]:
    """Find all train_log*.csv under runs_dir whose path contains keyword."""
    matches: List[str] = []
    if not os.path.isdir(runs_dir):
        return matches
    for dirpath, _, filenames in os.walk(runs_dir):
        if keyword.lower() not in dirpath.lower():
            continue
        for f in filenames:
            if f.startswith("train_log") and f.endswith(".csv"):
                matches.append(os.path.join(dirpath, f))
    return matches


def _pick_latest(path_list: List[str]) -> Optional[str]:
    if not path_list:
        return None
    return max(path_list, key=lambda p: os.path.getmtime(p))


def _infer_epoch(df: pd.DataFrame) -> pd.Series:
    for col in df.columns:
        if col.lower() == "epoch":
            return df[col].reset_index(drop=True)
    return pd.Series(np.arange(len(df)), name="epoch")


def _get_loss_col(df: pd.DataFrame) -> Optional[str]:
    for name in ("train/loss_total", "train/loss"):
        if name in df.columns:
            return name
    return None


def plot_loss_curve(
    baseline_log: Optional[str],
    sarhm_log: Optional[str],
    out_path: str,
) -> bool:
    if baseline_log is None and sarhm_log is None:
        return False

    plt.figure(figsize=(10, 6))

    def add_curve(path: Optional[str], label: str) -> bool:
        if path is None or not os.path.isfile(path):
            return False
        df = pd.read_csv(path)
        loss_col = _get_loss_col(df)
        if loss_col is None:
            return False
        epoch = _infer_epoch(df)
        plt.plot(epoch, df[loss_col], label=label)
        return True

    any_plotted = False
    any_plotted |= add_curve(baseline_log, "Baseline")
    any_plotted |= add_curve(sarhm_log, "SAR-HM")
    if not any_plotted:
        plt.close()
        return False

    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Training loss vs epoch (Baseline vs SAR-HM)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _ensure_dir(os.path.dirname(out_path))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def plot_single_metric_curve(
    log_path: Optional[str],
    column: str,
    ylabel: str,
    title: str,
    out_path: str,
) -> bool:
    if log_path is None or not os.path.isfile(log_path):
        return False
    df = pd.read_csv(log_path)
    if column not in df.columns:
        return False

    epoch = _infer_epoch(df)
    plt.figure(figsize=(10, 6))
    plt.plot(epoch, df[column])
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    _ensure_dir(os.path.dirname(out_path))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def _find_metrics_csvs(results_dir: str) -> List[str]:
    """Find all metrics.csv files under results_dir."""
    paths: List[str] = []
    if not os.path.isdir(results_dir):
        return paths
    for dirpath, _, filenames in os.walk(results_dir):
        for f in filenames:
            if f.endswith(".csv") and f == "metrics.csv":
                paths.append(os.path.join(dirpath, f))
    return paths


def _load_mode_rows(csv_path: str) -> Dict[str, pd.Series]:
    """Load a metrics.csv and return dict mode -> row."""
    df = pd.read_csv(csv_path)
    if "mode" not in df.columns:
        return {}
    rows: Dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        mode = str(row["mode"]).lower()
        rows[mode] = row
    return rows


def _canonical_mode_name(mode: str) -> str:
    m = mode.lower()
    if m in ("baseline",):
        return "baseline"
    if m in ("sarhm", "sar-hm", "sar_hm"):
        return "sarhm"
    return m


def aggregate_metrics_across_seeds(metrics_paths: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """
    Aggregate metrics across multiple metrics.csv files.

    Returns:
        dict[mode][metric_name] -> list of values (one per seed/file).
    """
    agg: Dict[str, Dict[str, List[float]]] = {}
    target_metrics = ["ssim_mean", "pcc_mean", "clip_sim_mean", "mean_variance"]

    for path in metrics_paths:
        rows = _load_mode_rows(path)
        for mode, row in rows.items():
            cname = _canonical_mode_name(mode)
            mode_dict = agg.setdefault(cname, {})
            for m in target_metrics:
                if m in row.index:
                    mode_dict.setdefault(m, []).append(float(row[m]))
    return agg


def plot_metrics_with_std(
    agg: Dict[str, Dict[str, List[float]]],
    out_path: str,
) -> bool:
    metrics = ["ssim_mean", "pcc_mean", "clip_sim_mean"]
    modes = ["baseline", "sarhm"]

    # Check if we have at least two seeds (any metric) for significance
    multi_seed = False
    for mode in modes:
        md = agg.get(mode, {})
        for m in metrics:
            if len(md.get(m, [])) > 1:
                multi_seed = True
                break
        if multi_seed:
            break

    _ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(10, 6))

    labels: List[str] = []
    x_indices: List[int] = []
    baseline_means: List[float] = []
    baseline_stds: List[float] = []
    sarhm_means: List[float] = []
    sarhm_stds: List[float] = []

    for i, m in enumerate(metrics):
        labels.append(m)
        x_indices.append(i)
        base_vals = agg.get("baseline", {}).get(m, [])
        sar_vals = agg.get("sarhm", {}).get(m, [])
        if base_vals:
            baseline_means.append(float(np.mean(base_vals)))
            baseline_stds.append(float(np.std(base_vals)))  # type: ignore[arg-type]
        else:
            baseline_means.append(0.0)
            baseline_stds.append(0.0)
        if sar_vals:
            sarhm_means.append(float(np.mean(sar_vals)))
            sarhm_stds.append(float(np.std(sar_vals)))  # type: ignore[arg-type]
        else:
            sarhm_means.append(0.0)
            sarhm_stds.append(0.0)

    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width / 2, baseline_means, width, yerr=baseline_stds, label="Baseline")
    plt.bar(x + width / 2, sarhm_means, width, yerr=sarhm_stds, label="SAR-HM")
    plt.xticks(x, labels)
    plt.ylabel("Score")
    if multi_seed:
        plt.title("Metrics with standard deviation across seeds")
    else:
        plt.title("Metrics (single seed)\nStatistical significance requires multiple seeds.")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def plot_ablation_comparison(
    agg: Dict[str, Dict[str, List[float]]],
    out_path: str,
) -> bool:
    """
    Compare baseline, sarhm, and any additional modes using ssim_mean and clip_sim_mean.
    """
    if not agg:
        return False

    metrics = ["ssim_mean", "clip_sim_mean"]
    modes = sorted(agg.keys())

    # Compute means per mode/metric
    values: Dict[str, Dict[str, float]] = defaultdict(dict)
    for mode in modes:
        for m in metrics:
            vals = agg.get(mode, {}).get(m, [])
            if vals:
                values[mode][m] = float(np.mean(vals))

    if not values:
        return False

    _ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(10, 6))

    metric_labels = metrics
    x = np.arange(len(metric_labels))
    width = 0.8 / max(len(modes), 1)

    for i, mode in enumerate(modes):
        mode_vals = [values.get(mode, {}).get(m, 0.0) for m in metric_labels]
        offset = (i - (len(modes) - 1) / 2.0) * width
        plt.bar(x + offset, mode_vals, width, label=mode)

    plt.xticks(x, metric_labels)
    plt.ylabel("Score")
    plt.title("Ablation comparison (Baseline, SAR-HM, variants)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate optional advanced evaluation visualizations for DreamDiffusion (Baseline vs SAR-HM)."
    )
    parser.add_argument("--runs_dir", type=str, default="results/runs", help="Directory containing training runs.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing results/*/metrics.csv.")
    parser.add_argument("--out_dir", type=str, default="graphs/optional", help="Output directory for optional graphs.")
    args = parser.parse_args()

    runs_dir = args.runs_dir
    # Backward-compatible fallback: if requested runs_dir does not exist,
    # try common alternatives.
    if not os.path.isdir(runs_dir):
        if os.path.isdir("results/runs"):
            runs_dir = "results/runs"
        elif os.path.isdir("runs"):
            runs_dir = "runs"
    results_dir = args.results_dir
    out_dir = args.out_dir
    _ensure_dir(out_dir)

    # -----------------------------
    # PART 1 — TRAINING DYNAMICS
    # -----------------------------
    baseline_logs = _find_train_logs(runs_dir, "baseline")
    sarhm_logs = _find_train_logs(runs_dir, "sarhm")
    baseline_log = _pick_latest(baseline_logs)
    sarhm_log = _pick_latest(sarhm_logs)

    generated: List[str] = []
    skipped: List[str] = []

    # 1) loss_curve.png
    loss_curve_path = os.path.join(out_dir, "loss_curve.png")
    if plot_loss_curve(baseline_log, sarhm_log, loss_curve_path):
        generated.append(loss_curve_path)
    else:
        skipped.append("loss_curve.png (missing or incompatible train logs)")

    # 2) retrieval_accuracy.png
    ra_path = os.path.join(out_dir, "retrieval_accuracy.png")
    if plot_single_metric_curve(
        sarhm_log,
        "train/sarhm_retrieval_acc",
        "Retrieval accuracy",
        "SAR-HM retrieval accuracy vs epoch",
        ra_path,
    ):
        generated.append(ra_path)
    else:
        skipped.append("retrieval_accuracy.png (column train/sarhm_retrieval_acc missing or log not found)")

    # 3) attention_entropy.png
    ae_path = os.path.join(out_dir, "attention_entropy.png")
    if plot_single_metric_curve(
        sarhm_log,
        "train/sarhm_attention_entropy",
        "Attention entropy",
        "SAR-HM attention entropy vs epoch",
        ae_path,
    ):
        generated.append(ae_path)
    else:
        skipped.append("attention_entropy.png (column train/sarhm_attention_entropy missing or log not found)")

    # ---------------------------------------------
    # PART 2 — STATISTICAL SIGNIFICANCE (MULTI-SEED)
    # ---------------------------------------------
    metrics_csvs = _find_metrics_csvs(results_dir)
    agg = aggregate_metrics_across_seeds(metrics_csvs)

    metrics_with_std_path = os.path.join(out_dir, "metrics_with_std.png")
    if agg:
        plot_metrics_with_std(agg, metrics_with_std_path)
        generated.append(metrics_with_std_path)
    else:
        # Generate placeholder figure
        _ensure_dir(out_dir)
        plt.figure(figsize=(10, 6))
        plt.text(
            0.5,
            0.5,
            "Statistical significance requires multiple seeds.\nNo metrics.csv files found.",
            ha="center",
            va="center",
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(metrics_with_std_path, dpi=200)
        plt.close()
        generated.append(metrics_with_std_path)

    # ---------------------------------------------
    # PART 3 — ABLATION STUDY VISUALIZATION
    # ---------------------------------------------
    ablation_path = os.path.join(out_dir, "ablation_comparison.png")
    if agg:
        if plot_ablation_comparison(agg, ablation_path):
            generated.append(ablation_path)
        else:
            skipped.append("ablation_comparison.png (metrics missing for ablation modes)")
    else:
        # Placeholder: same as previous message but specific to ablations
        _ensure_dir(out_dir)
        plt.figure(figsize=(10, 6))
        plt.text(
            0.5,
            0.5,
            "Ablation comparison requires metrics.csv for baseline, sarhm, and variants.\nNo metrics found.",
            ha="center",
            va="center",
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(ablation_path, dpi=200)
        plt.close()
        generated.append(ablation_path)

    # ---------------------------------------------
    # CONSOLE SUMMARY
    # ---------------------------------------------
    print("=== make_optional_graphs.py summary ===")
    print(f"runs_dir:    {os.path.abspath(runs_dir)}")
    print(f"results_dir: {os.path.abspath(results_dir)}")
    print(f"out_dir:     {os.path.abspath(out_dir)}")
    print(f"baseline_logs found: {len(baseline_logs)}")
    print(f"sarhm_logs found:    {len(sarhm_logs)}")
    print(f"metrics.csv files found: {len(metrics_csvs)}")
    if generated:
        print("Generated plots:")
        for g in generated:
            print(f"  - {g}")
    if skipped:
        print("Skipped plots due to missing data:")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()

