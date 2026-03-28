"""Recompute eval metrics (MSE, PCC, SSIM, PSM, top-1-class) from saved test images.

Usage:
    PYTHONPATH=code python code/compute_metrics_from_images.py \
        --eval_dir path/to/repo/results/eval/<run_id> \
        --num_samples 2
"""
import argparse
import glob
import os
import re
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_images(eval_dir, num_samples):
    pattern = re.compile(r'test(\d+)-(\d+)\.png')
    files = {}
    for f in glob.glob(os.path.join(eval_dir, 'test*-*.png')):
        m = pattern.search(os.path.basename(f))
        if m:
            idx, sample = int(m.group(1)), int(m.group(2))
            files[(idx, sample)] = f

    test_indices = sorted(set(idx for idx, _ in files.keys()))
    print(f"Found {len(test_indices)} test items, {num_samples} samples each")

    gt_images = []
    pred_images_per_sample = [[] for _ in range(num_samples)]

    for idx in tqdm(test_indices, desc='[1/2] Loading images', unit='img'):
        gt_path = files.get((idx, 0))
        if gt_path is None:
            continue
        gt = np.array(Image.open(gt_path).convert('RGB'))
        gt_images.append(gt)
        for s in range(num_samples):
            pred_path = files.get((idx, s + 1))
            if pred_path is None:
                print(f"WARNING: missing test{idx}-{s+1}.png")
                pred_images_per_sample[s].append(gt)
            else:
                pred_images_per_sample[s].append(np.array(Image.open(pred_path).convert('RGB')))

    gt_images = np.stack(gt_images)
    pred_images_per_sample = [np.stack(p) for p in pred_images_per_sample]
    return gt_images, pred_images_per_sample


def pair_wise_score_with_progress(pred_imgs, gt_imgs, metric_func, decision_func, metric_name, sample_idx):
    """pair_wise_score with per-image progress bar."""
    n = len(pred_imgs)
    corrects = []
    total_comparisons = n * (n - 1)
    pbar = tqdm(total=total_comparisons,
                desc=f'    {metric_name} sample={sample_idx} comparisons',
                unit='cmp', leave=False)
    for idx in range(n):
        pred = pred_imgs[idx]
        gt = gt_imgs[idx]
        gt_score = metric_func(pred, gt)
        count = 0
        for j in range(n):
            if j == idx:
                continue
            comp_score = metric_func(pred, gt_imgs[j])
            if decision_func(gt_score, comp_score):
                count += 1
            pbar.update(1)
        corrects.append(count / (n - 1))
    pbar.close()
    return corrects


def get_metric_func(metric_name):
    from eval_metrics import mse_metric, pcc_metric, ssim_metric, psm_wrapper
    from eval_metrics import larger_the_better, smaller_the_better

    if metric_name == 'mse':
        return mse_metric, smaller_the_better
    elif metric_name == 'pcc':
        return pcc_metric, larger_the_better
    elif metric_name == 'ssim':
        return ssim_metric, larger_the_better
    elif metric_name == 'psm':
        return psm_wrapper(), smaller_the_better
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=2)
    parser.add_argument('--skip_topk', action='store_true', help='Skip slow top-1-class metric')
    args = parser.parse_args()

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

    gt_images, pred_per_sample = load_images(args.eval_dir, args.num_samples)
    print(f"GT shape: {gt_images.shape}, Pred shape: {pred_per_sample[0].shape}")

    metric_list = ['mse', 'pcc', 'ssim', 'psm']
    total_steps = len(metric_list) + (0 if args.skip_topk else 1)
    overall = tqdm(total=total_steps, desc='[2/2] Overall metrics', unit='metric',
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]')
    results = {}

    for m in metric_list:
        metric_func, decision_func = get_metric_func(m)
        tqdm.write(f"\n--- Computing {m.upper()} ---")
        sample_scores = []
        for s in range(args.num_samples):
            tqdm.write(f"  Sample {s+1}/{args.num_samples}:")
            res = pair_wise_score_with_progress(
                pred_per_sample[s], gt_images,
                metric_func, decision_func,
                metric_name=m, sample_idx=s+1)
            sample_scores.append(np.mean(res))
            tqdm.write(f"    -> {m} sample {s+1} = {sample_scores[-1]:.4f}")
        results[m] = np.mean(sample_scores)
        overall.set_postfix_str(f'{m}={results[m]:.4f}')
        overall.update(1)
        tqdm.write(f"  >> {m} avg = {results[m]:.4f}")

    if not args.skip_topk:
        import torch
        from eval_metrics import get_similarity_metric
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tqdm.write(f"\n--- Computing TOP-1-CLASS (n_way=50, trials=50) ---")
        topk_scores = []
        for s in range(args.num_samples):
            tqdm.write(f"  Sample {s+1}/{args.num_samples}:")
            res = get_similarity_metric(pred_per_sample[s], gt_images, 'class', None,
                                        n_way=50, num_trials=50, top_k=1, device=device)
            topk_scores.append(np.mean(res))
            tqdm.write(f"    -> top-1-class sample {s+1} = {topk_scores[-1]:.4f}")
        results['top-1-class'] = np.mean(topk_scores)
        results['top-1-class (max)'] = np.max(topk_scores)
        overall.set_postfix_str(f'top-1={results["top-1-class"]:.4f}')
        overall.update(1)
        tqdm.write(f"  >> top-1-class avg = {results['top-1-class']:.4f}")
        tqdm.write(f"  >> top-1-class max = {results['top-1-class (max)']:.4f}")

    overall.close()

    print("\n===== FINAL RESULTS =====")
    for k, v in results.items():
        print(f"  {k:20s} = {v:.4f}")

    out_path = os.path.join(args.eval_dir, 'metrics.txt')
    with open(out_path, 'w') as f:
        for k, v in results.items():
            f.write(f"{k}\t{v:.6f}\n")
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
