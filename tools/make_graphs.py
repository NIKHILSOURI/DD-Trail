import argparse
import os
import shutil
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _find_files(root: str, name_substring: str) -> List[str]:
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if name_substring in f:
                matches.append(os.path.join(dirpath, f))
    return matches


def _pick_latest(paths: List[str]) -> Optional[str]:
    if not paths:
        return None
    return max(paths, key=lambda p: os.path.getmtime(p))


def find_train_log(runs_dir: str, model_keyword: str) -> Optional[str]:
    """
    Find a train_log CSV under runs/ whose path contains model_keyword
    (e.g. 'baseline' or 'sarhm'). Prefer train_log*.csv and the most recent file.
    """
    if not os.path.isdir(runs_dir):
        return None
    candidate_logs: List[str] = []
    for dirpath, _, filenames in os.walk(runs_dir):
        if model_keyword.lower() not in dirpath.lower():
            continue
        for f in filenames:
            if f.startswith("train_log") and f.endswith(".csv"):
                candidate_logs.append(os.path.join(dirpath, f))
    return _pick_latest(candidate_logs)


def find_metrics_csv(compare_dir: str) -> Optional[str]:
    """
    Find metrics.csv under compare_eval_thesis (or similar). Prefer
    files named 'metrics.csv' but accept any *.csv if needed.
    """
    if not os.path.isdir(compare_dir):
        return None
    # Prefer exact metrics.csv names
    exact = []
    any_csv = []
    for dirpath, _, filenames in os.walk(compare_dir):
        for f in filenames:
            if f.endswith(".csv"):
                path = os.path.join(dirpath, f)
                any_csv.append(path)
                if f == "metrics.csv":
                    exact.append(path)
    if exact:
        return _pick_latest(exact)
    return _pick_latest(any_csv)


def infer_epoch_column(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series representing epoch index.
    Prefer explicit 'epoch' column; otherwise use row index (0..N-1).
    """
    for col in df.columns:
        if col.lower() == "epoch":
            return df[col].reset_index(drop=True)
    return pd.Series(np.arange(len(df)), name="epoch")


def get_loss_column(df: pd.DataFrame) -> Optional[str]:
    """Return the name of the loss column to use."""
    for name in ("train/loss_total", "train/loss"):
        if name in df.columns:
            return name
    return None


def plot_training_loss(
    baseline_log: str,
    sarhm_log: str,
    out_dir: str,
) -> Optional[str]:
    if baseline_log is None and sarhm_log is None:
        return None

    plt.figure(figsize=(10, 6))

    def add_curve(path: str, label: str):
        if path is None:
            return
        df = pd.read_csv(path)
        epoch = infer_epoch_column(df)
        loss_col = get_loss_column(df)
        if loss_col is None:
            return
        plt.plot(epoch, df[loss_col], label=label)

    add_curve(baseline_log, "Baseline")
    add_curve(sarhm_log, "SAR-HM")

    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Training loss vs epoch (Baseline vs SAR-HM)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "training_loss_total.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    try:
        plt.savefig(os.path.join(out_dir, "training_loss_total.pdf"))
    except Exception:
        pass
    plt.close()
    return png_path


def plot_sarhm_curves(sarhm_log: str, out_dir: str) -> Tuple[Optional[str], Optional[str]]:
    if sarhm_log is None or not os.path.isfile(sarhm_log):
        return None, None

    df = pd.read_csv(sarhm_log)
    epoch = infer_epoch_column(df)

    # retrieval accuracy
    acc_col = "train/sarhm_retrieval_acc"
    acc_path = None
    if acc_col in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(epoch, df[acc_col])
        plt.xlabel("Epoch")
        plt.ylabel("Retrieval accuracy")
        plt.title("SAR-HM retrieval accuracy vs epoch")
        plt.grid(True, alpha=0.3)
        os.makedirs(out_dir, exist_ok=True)
        acc_path = os.path.join(out_dir, "retrieval_acc.png")
        plt.tight_layout()
        plt.savefig(acc_path, dpi=200)
        try:
            plt.savefig(os.path.join(out_dir, "retrieval_acc.pdf"))
        except Exception:
            pass
        plt.close()

    # attention entropy
    ent_col = "train/sarhm_attention_entropy"
    ent_path = None
    if ent_col in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(epoch, df[ent_col])
        plt.xlabel("Epoch")
        plt.ylabel("Attention entropy")
        plt.title("SAR-HM attention entropy vs epoch")
        plt.grid(True, alpha=0.3)
        os.makedirs(out_dir, exist_ok=True)
        ent_path = os.path.join(out_dir, "attention_entropy.png")
        plt.tight_layout()
        plt.savefig(ent_path, dpi=200)
        try:
            plt.savefig(os.path.join(out_dir, "attention_entropy.pdf"))
        except Exception:
            pass
        plt.close()

    return acc_path, ent_path


def load_metrics(metrics_csv: str) -> Optional[pd.DataFrame]:
    if metrics_csv is None or not os.path.isfile(metrics_csv):
        return None
    df = pd.read_csv(metrics_csv)
    # Normalize mode column name
    if "mode" not in df.columns:
        return None
    return df


def extract_metrics_rows(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    base = None
    sarhm = None
    delta = None
    # try exact string matches first
    for _, row in df.iterrows():
        m = str(row["mode"]).lower()
        if m == "baseline":
            base = row
        elif m in ("sarhm", "sar-hm", "sar_hm"):
            sarhm = row
        elif m == "delta":
            delta = row
    return base, sarhm, delta


def plot_metrics_bars(df: pd.DataFrame, out_dir: str) -> Optional[str]:
    base, sarhm, delta_row = extract_metrics_rows(df)
    if base is None or sarhm is None:
        return None

    metrics = ["ssim_mean", "pcc_mean", "clip_sim_mean"]
    values = []
    labels = []
    for m in metrics:
        if m in base.index and m in sarhm.index:
            labels.append(m)
            values.append((float(base[m]), float(sarhm[m])))
    if not values:
        return None

    baseline_vals = [v[0] for v in values]
    sarhm_vals = [v[1] for v in values]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, baseline_vals, width, label="Baseline")
    plt.bar(x + width / 2, sarhm_vals, width, label="SAR-HM")
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Metrics comparison (Baseline vs SAR-HM)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "metrics_bars.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    try:
        plt.savefig(os.path.join(out_dir, "metrics_bars.pdf"))
    except Exception:
        pass
    plt.close()
    return png_path


def plot_metrics_delta(df: pd.DataFrame, out_dir: str) -> Optional[str]:
    base, sarhm, delta_row = extract_metrics_rows(df)
    if base is None or sarhm is None:
        return None

    metrics = ["ssim_mean", "pcc_mean", "clip_sim_mean"]
    deltas = []
    labels = []
    for m in metrics:
        if m in base.index and m in sarhm.index:
            labels.append(m)
            if delta_row is not None and m in delta_row.index:
                deltas.append(float(delta_row[m]))
            else:
                deltas.append(float(sarhm[m]) - float(base[m]))
    if not deltas:
        return None

    x = np.arange(len(labels))
    plt.figure(figsize=(10, 6))
    plt.bar(x, deltas)
    plt.xticks(x, labels)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Delta (SAR-HM - Baseline)")
    plt.title("Metric deltas (SAR-HM minus Baseline)")
    plt.grid(True, axis="y", alpha=0.3)
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "metrics_delta.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    try:
        plt.savefig(os.path.join(out_dir, "metrics_delta.pdf"))
    except Exception:
        pass
    plt.close()
    return png_path


def plot_variance_comparison(df: pd.DataFrame, out_dir: str) -> Optional[str]:
    base, sarhm, _ = extract_metrics_rows(df)
    if base is None or sarhm is None:
        return None
    if "mean_variance" not in base.index or "mean_variance" not in sarhm.index:
        return None

    labels = ["baseline", "sarhm"]
    vals = [float(base["mean_variance"]), float(sarhm["mean_variance"])]
    x = np.arange(len(labels))

    plt.figure(figsize=(10, 6))
    plt.bar(x, vals)
    plt.xticks(x, labels)
    plt.ylabel("Mean variance")
    plt.title("Image variance comparison")
    plt.grid(True, axis="y", alpha=0.3)
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "variance_comparison.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    try:
        plt.savefig(os.path.join(out_dir, "variance_comparison.pdf"))
    except Exception:
        pass
    plt.close()
    return png_path


def copy_qualitative_grids(compare_dir: str, out_dir: str) -> List[str]:
    copied = []
    if not os.path.isdir(compare_dir):
        return copied
    grid_dir = os.path.join(compare_dir, "grids")
    candidates = [
        "baseline_grid.png",
        "sarhm_grid.png",
        "side_by_side.png",
    ]
    os.makedirs(out_dir, exist_ok=True)
    for name in candidates:
        src = os.path.join(grid_dir, name)
        if os.path.isfile(src):
            dst = os.path.join(out_dir, name)
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def main():
    parser = argparse.ArgumentParser(description="Generate thesis-quality graphs for DreamDiffusion Baseline vs SAR-HM.")
    parser.add_argument("--runs_dir", type=str, default="results/runs", help="Directory containing training runs.")
    parser.add_argument("--compare_dir", type=str, default="results/compare_eval_thesis", help="Directory with compare-eval metrics and grids.")
    parser.add_argument("--out_dir", type=str, default="graphs", help="Output directory for graphs.")
    args = parser.parse_args()

    runs_dir = args.runs_dir
    # Backward-compatible fallback: if requested runs_dir does not exist,
    # try common alternatives.
    if not os.path.isdir(runs_dir):
        if os.path.isdir("results/runs"):
            runs_dir = "results/runs"
        elif os.path.isdir("runs"):
            runs_dir = "runs"
    compare_dir = args.compare_dir
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    # 1) Find train logs
    baseline_log = find_train_log(runs_dir, "baseline")
    sarhm_log = find_train_log(runs_dir, "sarhm")

    # 2) Find metrics CSV
    metrics_csv = find_metrics_csv(compare_dir)
    metrics_df = load_metrics(metrics_csv) if metrics_csv else None

    generated = []

    # A) training_loss_total.png
    loss_png = plot_training_loss(baseline_log, sarhm_log, out_dir)
    if loss_png:
        generated.append(loss_png)

    # B) sarhm_retrieval_curves: retrieval_acc.png, attention_entropy.png
    acc_png, ent_png = plot_sarhm_curves(sarhm_log, out_dir)
    for p in (acc_png, ent_png):
        if p:
            generated.append(p)

    # C, D, E) metrics-based plots
    if metrics_df is not None:
        mb = plot_metrics_bars(metrics_df, out_dir)
        if mb:
            generated.append(mb)
        md = plot_metrics_delta(metrics_df, out_dir)
        if md:
            generated.append(md)
        var_png = plot_variance_comparison(metrics_df, out_dir)
        if var_png:
            generated.append(var_png)

    # F) copy qualitative grids
    copied_grids = copy_qualitative_grids(compare_dir, out_dir)

    # Summary
    print("=== make_graphs.py summary ===")
    print(f"runs_dir: {os.path.abspath(runs_dir)}")
    print(f"compare_dir: {os.path.abspath(compare_dir)}")
    print(f"out_dir: {os.path.abspath(out_dir)}")
    print(f"baseline_log: {baseline_log or 'NOT FOUND'}")
    print(f"sarhm_log: {sarhm_log or 'NOT FOUND'}")
    print(f"metrics_csv: {metrics_csv or 'NOT FOUND'}")
    if generated:
        print("Generated figures:")
        for p in generated:
            print(f"  - {p}")
    else:
        print("No figures were generated (missing inputs?).")
    if copied_grids:
        print("Copied qualitative grids:")
        for p in copied_grids:
            print(f"  - {p}")
    else:
        print("No qualitative grids were copied (grids directory missing or empty).")


if __name__ == "__main__":
    main()

