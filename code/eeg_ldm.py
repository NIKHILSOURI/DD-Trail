import os, sys
# Reduce TensorFlow log noise (e.g. from wandb or deps that load TF)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# Reduce CUDA fragmentation on 16 GB GPUs (avoids OOM during backward)
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*", category=UserWarning)
# Lightning / PyTorch deprecation and logger warnings (Stage B runs correctly)
warnings.filterwarnings("ignore", message=".*GradScaler.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*tensorboardX.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*LeafSpec.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*duplicate parameters.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*dataloader.*does not have many workers.*", category=UserWarning)

import numpy as np
import torch
import argparse
import datetime
import wandb
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import copy
from tqdm import tqdm

# own code
from config import Config_Generative_Model
from dataset import  create_EEG_dataset
from dc_ldm.ldm_for_eeg import eLDM
from eval_metrics import get_similarity_metric
from utils.state_dict_utils import filter_state_dict_for_model, log_filter_info, is_mae_pretrain_ckpt

# Thesis logging and evaluation (optional)
try:
    from logger import MetricLogger, create_run_dir, RunConfig
    from eval.evaluate import evaluate
except ImportError:
    MetricLogger = None
    create_run_dir = None
    RunConfig = None
    evaluate = None


class ExperimentLoggingCallback(pl.Callback):
    """Logs train metrics per epoch and runs evaluation every eval_every epochs."""
    def __init__(self, metric_logger, generative_model, eval_datasets, config):
        self.metric_logger = metric_logger
        self.generative_model = generative_model
        self.eval_datasets = eval_datasets or []
        self.config = config
        self.eval_every = getattr(config, "eval_every", 2)
        self.num_eval_samples = getattr(config, "num_eval_samples", 50)
        self.run_dir = getattr(config, "run_dir", None)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.metric_logger is None or self.run_dir is None:
            return
        epoch = trainer.current_epoch
        step = trainer.global_step
        # Collect train metrics from Lightning (keys like train/loss, train/loss_simple, etc.)
        cm = trainer.callback_metrics or {}
        train_metrics = {}
        for k, v in cm.items():
            if isinstance(k, str) and ("loss" in k or "alpha" in k or "entropy" in k or "acc" in k or "retrieval" in k):
                try:
                    train_metrics[k] = float(v.item()) if hasattr(v, "item") else float(v)
                except (TypeError, ValueError):
                    train_metrics[k] = v
        # Normalize keys to match spec (train/loss_total, train/loss_gen, ...)
        if "train/loss" in cm:
            train_metrics["train/loss_total"] = float(cm["train/loss"].item() if hasattr(cm["train/loss"], "item") else cm["train/loss"])
        if "train/loss_simple" in cm:
            train_metrics["train/loss_gen"] = float(cm["train/loss_simple"].item() if hasattr(cm["train/loss_simple"], "item") else cm["train/loss_simple"])
        if not train_metrics and "train/loss" in cm:
            train_metrics["train/loss_total"] = float(cm["train/loss"].item() if hasattr(cm["train/loss"], "item") else cm["train/loss"])
        self.metric_logger.log_train(step, train_metrics, epoch=epoch)
        # Stage B: alpha_max schedule (linear warmup so SAR-HM can beat baseline consistently)
        config = self.config
        csm = getattr(self.generative_model.model, "cond_stage_model", None)
        if csm is not None and getattr(config, "use_sarhm", False):
            warmup = getattr(config, "warmup_epochs", 10)
            start = getattr(config, "alpha_max_start", 0.05)
            end = getattr(config, "alpha_max_end", 0.2)
            alpha_max_epoch = min(end, start + (epoch / max(warmup, 1)) * (end - start))
            csm.alpha_max = float(alpha_max_epoch)
        # Stage B: no PLMS/image generation. Use Stage C (gen_eval_eeg.py / evaluate.py) for generation and metrics.

    def on_train_end(self, trainer, pl_module):
        """Close CSV files so all data is flushed."""
        if self.metric_logger is not None and hasattr(self.metric_logger, "close"):
            self.metric_logger.close()

    def on_train_start(self, trainer, pl_module):
        """Set alpha_max to schedule start (epoch 0)."""
        config = self.config
        csm = getattr(self.generative_model.model, "cond_stage_model", None)
        if csm is not None and getattr(config, "use_sarhm", False):
            start = getattr(config, "alpha_max_start", 0.05)
            csm.alpha_max = float(start)


def wandb_init(config, output_path):
    # wandb.init( project='dreamdiffusion',
    #             group="stageB_dc-ldm",
    #             anonymous="allow",
    #             config=config,
    #             reinit=True)
    create_readme(config, output_path)

def wandb_finish():
    wandb.finish()

def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))

def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')

def _has_samples(samples):
    """Safe check: avoid truth value of array (use a.any()/a.all() error)."""
    if samples is None:
        return False
    if hasattr(samples, '__len__'):
        return len(samples) > 0
    return True


def get_eval_metric(samples, avg=True):
    _nan_return = ([float('nan')] * 6, ['mse', 'pcc', 'ssim', 'psm', 'top-1-class', 'top-1-class (max)'])
    if not _has_samples(samples):
        return _nan_return[0], _nan_return[1]
    try:
        first_len = len(samples[0]) if hasattr(samples[0], '__len__') else 0
    except (TypeError, IndexError):
        return _nan_return[0], _nan_return[1]
    if first_len < 2 and avg:
        return _nan_return[0], _nan_return[1]
    metric_list = ['mse', 'pcc', 'ssim', 'psm']
    res_list = []
    try:
        gt_images = [img[0] for img in samples]
        gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
        samples_to_run = np.arange(1, first_len) if avg else [1]
        for m in tqdm(metric_list, desc='Eval metrics (mse/pcc/ssim/psm)', unit='metric'):
            res_part = []
            for s in samples_to_run:
                pred_images = [img[s] for img in samples]
                pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
                res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m)
                res_part.append(np.mean(res))
            res_list.append(np.mean(res_part))
        res_part = []
        for s in tqdm(samples_to_run, desc='Eval top-1-class', unit='sample'):
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            res = get_similarity_metric(pred_images, gt_images, 'class', None,
                            n_way=50, num_trials=50, top_k=1,
                            device='cuda' if torch.cuda.is_available() else 'cpu')
            res_part.append(np.mean(res))
        res_list.append(np.mean(res_part))
        res_list.append(np.max(res_part))
        metric_list.append('top-1-class')
        metric_list.append('top-1-class (max)')
        return res_list, metric_list
    except Exception as e:
        print(f"[get_eval_metric] Failed: {e}")
        return _nan_return[0], _nan_return[1]
               
def generate_images(generative_model, eeg_latents_dataset_train, eeg_latents_dataset_test, config):
    train_limit = getattr(config, 'test_gen_limit', 10) or 10  # cap train samples for speed
    try:
        grid, _ = generative_model.generate(eeg_latents_dataset_train, config.num_samples,
                    config.ddim_steps, config.HW, train_limit)
    except Exception as e:
        print(f"[generate_images] Train generation failed: {e}")
        grid = None
    if grid is not None and getattr(grid, 'size', 0) > 0:
        grid_imgs = Image.fromarray(grid.astype(np.uint8))
        grid_imgs.save(os.path.join(config.output_path, 'samples_train.png'))

    test_limit = getattr(config, 'test_gen_limit', 10) or 10  # None -> 10 for fast; set 5 or 3 for very fast
    samples = None
    try:
        grid, samples = generative_model.generate(eeg_latents_dataset_test, config.num_samples,
                    config.ddim_steps, config.HW, limit=test_limit)
    except Exception as e:
        print(f"[generate_images] Test generation failed: {e}")
        grid = None
        samples = None
    if grid is not None and getattr(grid, 'size', 0) > 0:
        grid_imgs = Image.fromarray(grid.astype(np.uint8))
        grid_imgs.save(os.path.join(config.output_path, f'./samples_test.png'))
    if _has_samples(samples):
        try:
            for sp_idx, imgs in enumerate(samples):
                # imgs: (1+num_samples, C, H, W); imgs[0]=gt, imgs[1:]=generated
                for copy_idx, img in enumerate(imgs[1:]):
                    img = rearrange(img, 'c h w -> h w c')
                    img_np = np.asarray(img)
                    if img_np.dtype != np.uint8:
                        img_np = (255.0 * np.clip(img_np.astype(np.float64), 0, 1)).astype(np.uint8)
                    Image.fromarray(img_np).save(os.path.join(config.output_path, f'./test{sp_idx}-{copy_idx}.png'))
        except Exception as e:
            print(f"[generate_images] Saving test images failed: {e}")

    samples_for_metric = samples if _has_samples(samples) else []
    metric, metric_list = get_eval_metric(samples_for_metric, avg=config.eval_avg)
    # Build metric_dict safely (metric/metric_list length may vary on error path)
    metric_dict = {}
    n_pair = min(len(metric_list) - 2, len(metric) - 2, 4)
    if n_pair > 0:
        metric_dict.update({f'summary/pair-wise_{k}': v for k, v in zip(metric_list[:n_pair], metric[:n_pair])})
    if len(metric_list) >= 2 and len(metric) >= 2:
        metric_dict[f'summary/{metric_list[-2]}'] = metric[-2]
        metric_dict[f'summary/{metric_list[-1]}'] = metric[-1]
    # wandb.log(metric_dict)

def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

def _print_gpu_banner():
    """Print a large, visible banner that GPU is available and in use."""
    name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA"
    bar = "=" * 56
    lines = [
        "",
        bar,
        "",
        "              **   GPU   **",
        "",
        "           **   IT IS AVAILABLE   **",
        "",
        "              Device: %s" % name[:36],
        "              PLMS: very fast (20 steps). Use --ddim_steps 50 or 250 for better quality.",
        "",
        bar,
        "",
    ]
    for L in lines:
        print(L)


def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)


def ensure_prototypes_loaded_or_built(model, config, train_dataset, output_dir, device):
    """
    Build or load baseline_centroids prototypes after dataloader/dataset exists, before first forward.
    Only runs if config.use_sarhm and config.proto_source == 'baseline_centroids'.
    Validates shape (K, 768), finite, float32. Raises RuntimeError if invalid after build/load.
    Returns path used.
    """
    if not getattr(config, 'use_sarhm', False):
        return None
    proto_source = getattr(config, 'proto_source', '') or 'baseline_centroids'
    if proto_source != 'baseline_centroids':
        return None
    from pathlib import Path
    from torch.utils.data import DataLoader
    from sarhm.prototypes import build_baseline_centroids
    csm = model.model.cond_stage_model
    K = getattr(config, 'num_classes', 40)
    dim = 768
    proto_path = getattr(config, 'proto_path', None)
    if proto_path and os.path.isfile(proto_path):
        if csm.sarhm_prototypes.load_from_path(proto_path):
            csm._proto_source = "baseline_centroids"
            config.proto_path = proto_path
            path_used = proto_path
        else:
            path_used = None
    else:
        save_path = os.path.join(output_dir, 'prototypes.pt')
        os.makedirs(output_dir, exist_ok=True)
        batch_size_proto = max(1, min(32, len(train_dataset)))
        loader = DataLoader(train_dataset, batch_size=batch_size_proto, shuffle=False, num_workers=0)
        build_baseline_centroids(loader, csm, K, dim, device, save_path=save_path)
        csm.sarhm_prototypes.load_from_path(save_path)
        csm._proto_source = "baseline_centroids"
        config.proto_path = save_path
        path_used = save_path
        print('[PROTO] built and saved baseline_centroids -> %s' % save_path)
    if path_used is None and proto_path:
        raise RuntimeError("[SAR-HM] prototypes load failed from %s (shape mismatch or invalid file). Aborting training." % proto_path)
    # Validate
    P = getattr(csm.sarhm_prototypes, 'P', None)
    if P is None:
        raise RuntimeError("[SAR-HM] prototypes invalid after build/load (P is None). Aborting training.")
    if P.shape != (K, dim):
        raise RuntimeError("[SAR-HM] prototypes invalid shape %s; expected (%d, %d). Aborting training." % (tuple(P.shape), K, dim))
    if not torch.isfinite(P).all():
        raise RuntimeError("[SAR-HM] prototypes contain non-finite values. Aborting training.")
    P.data = P.data.float()
    return path_used


def main(config):
    # GPU only: require CUDA, no CPU fallback
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU required. No CUDA device found. "
            "Install PyTorch with CUDA (e.g. pip install torch --index-url https://download.pytorch.org/whl/cu118) and ensure a GPU is available."
        )
    device = torch.device('cuda')
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')  # use Tensor Cores on A100 etc.
    _print_gpu_banner()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    pbar = tqdm(total=3, desc='Stage B total', unit='stage',
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} stages [{elapsed}<{remaining}]')

    crop_pix = int(config.crop_ratio*config.img_size)
    img_transform_train = transforms.Compose([
        normalize,

        transforms.Resize((512, 512)),
        random_crop(config.img_size-crop_pix, p=0.5),

        transforms.Resize((512, 512)),
        channel_last
    ])
    img_transform_test = transforms.Compose([
        normalize, 

        transforms.Resize((512, 512)),
        channel_last
    ])
    if config.dataset == 'EEG':

        eeg_latents_dataset_train, eeg_latents_dataset_test = create_EEG_dataset(eeg_signals_path = config.eeg_signals_path, splits_path = config.splits_path,
                image_transform=[img_transform_train, img_transform_test], subject = getattr(config, 'subject', 4))
        # eeg_latents_dataset_train, eeg_latents_dataset_test = create_EEG_dataset_viz( image_transform=[img_transform_train, img_transform_test])
        if len(eeg_latents_dataset_train) == 0:
            raise RuntimeError("Training split is empty after filtering. Check splits_path and eeg_signals_path, or relax Splitter filter (e.g. EEG length 450–600).")
        if len(eeg_latents_dataset_test) == 0:
            raise RuntimeError("Test split is empty after filtering. Check splits_path and eeg_signals_path.")
        num_voxels = eeg_latents_dataset_train.data_len

    else:
        raise NotImplementedError
    # print(num_voxels)

    # prepare pretrained mbm 
    pretrain_mbm_path = getattr(config, 'pretrain_mbm_path', None)
    if not pretrain_mbm_path or not os.path.isfile(pretrain_mbm_path):
        _def = Config_Generative_Model()
        pretrain_mbm_path = _def.pretrain_mbm_path
        config.pretrain_mbm_path = pretrain_mbm_path
        if not os.path.isfile(pretrain_mbm_path):
            raise FileNotFoundError('pretrain_mbm_path not found: %s. Set --pretrain_mbm_path (MAE checkpoint).' % pretrain_mbm_path)
    print(f"Loading pretrained checkpoint from: {config.pretrain_mbm_path}")
    pretrain_mbm_metafile = torch.load(config.pretrain_mbm_path, map_location='cpu', weights_only=False)
    print(f"Checkpoint keys: {list(pretrain_mbm_metafile.keys())}")

    # create generative model
    if getattr(config, 'use_sarhm', False):
        print(f"SAR-HM enabled: ablation_mode={getattr(config, 'ablation_mode', 'full_sarhm')}, num_classes={getattr(config, 'num_classes', 40)}")
        from pathlib import Path
        proto_dir = Path(config.pretrain_gm_path) / 'prototypes'
        proto_dir.mkdir(parents=True, exist_ok=True)
    generative_model = eLDM(pretrain_mbm_metafile, num_voxels,
                device=device, pretrain_root=config.pretrain_gm_path, logger=config.logger,
                ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond,
                clip_tune=config.clip_tune, cls_tune=config.cls_tune, main_config=config)

    # Resume Stage-B LDM from config.resume_ckpt_path only (MAE loaded above from pretrain_mbm_path).
    # Only resume_ckpt_path triggers full resume (config + weights). init_from_ckpt is weights-only below.
    resume_path = getattr(config, 'resume_ckpt_path', None) or getattr(config, 'checkpoint_path', None)
    if resume_path is not None:
        model_meta = torch.load(resume_path, map_location='cpu', weights_only=False)
        if is_mae_pretrain_ckpt(model_meta):
            raise RuntimeError(
                "[ERROR] resume_ckpt_path points to a MAE/MBM pretrain checkpoint. "
                "Use --pretrain_mbm_path for that file and --resume_ckpt_path for Stage-B LDM only."
            )
        state_dict = model_meta.get('model_state_dict', model_meta.get('state_dict'))
        if state_dict is None:
            print('[CKPT_LOAD] no model_state_dict in checkpoint; skipping load.')
        else:
            model_keys = set(generative_model.model.state_dict().keys())
            filtered_sd, filter_info = filter_state_dict_for_model(
                state_dict,
                model_state_keys=model_keys,
                drop_exact_keys=None,
                drop_prefixes=None,
                prune_unexpected_keys=getattr(config, 'prune_unexpected_keys', False),
            )
            log_filter_info(filter_info, tag='[CKPT_FILTER]')
            missing, unexpected = generative_model.model.load_state_dict(filtered_sd, strict=False)
            n_miss, n_unex = len(missing), len(unexpected)
            print('[CKPT_LOAD] missing=%d unexpected=%d' % (n_miss, n_unex))
            sarhm_prefixes = ('cond_stage_model.cond_ln.', 'cond_stage_model.sarhm_projection.',
                              'cond_stage_model.sarhm_prototypes.prototypes', 'cond_stage_model.sarhm_adapter.')
            sarhm_missing = [k for k in missing if any(k.startswith(p) for p in sarhm_prefixes)]
            if sarhm_missing:
                print('[CKPT_LOAD] SAR-HM keys missing (ok): %s' % (sarhm_missing[:10] if len(sarhm_missing) > 10 else sarhm_missing))
            if n_miss:
                print('[CKPT_LOAD] missing (first 20): %s' % list(missing)[:20])
            if n_unex:
                print('[CKPT_LOAD] unexpected (first 20): %s' % list(unexpected)[:20])
            print('model resumed')

    # Weights-only init (no config/optimizer/epoch): load from init_from_ckpt, start fresh at epoch 0
    # Only when NOT resuming (resume_ckpt_path not set).
    init_path = getattr(config, 'init_from_ckpt', None)
    if resume_path is None and init_path is not None and os.path.isfile(init_path):
        ckpt = torch.load(init_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        if sd is None:
            print('[INIT_CKPT] no model state in %s; skipping' % init_path)
        else:
            model_keys = set(generative_model.model.state_dict().keys())
            filtered_sd, filter_info = filter_state_dict_for_model(
                sd,
                model_state_keys=model_keys,
                drop_exact_keys=None,
                drop_prefixes=None,
                prune_unexpected_keys=False,
            )
            missing, unexpected = generative_model.model.load_state_dict(filtered_sd, strict=False)
            print('[INIT_CKPT] loaded weights from %s (strict=False) missing=%d unexpected=%d' % (
                init_path, len(missing), len(unexpected)))
        config.start_epoch = 0
        print('[INIT_CKPT] starting fresh from epoch 0 (no optimizer/scheduler/step loaded).')

    # Prototypes: build or load after dataset exists, before trainer.fit (correct time)
    if getattr(config, 'use_sarhm', False):
        ensure_prototypes_loaded_or_built(
            generative_model, config, eeg_latents_dataset_train, config.output_path, device
        )

    # Thesis logging: run_dir, MetricLogger, eval_datasets
    run_dir = None
    metric_logger = None
    eval_datasets = []
    if (create_run_dir is not None and MetricLogger is not None) and (getattr(config, 'run_name', None) or getattr(config, 'model', None)):
        model_type = getattr(config, 'model', None) or ('sarhm' if getattr(config, 'use_sarhm', False) else 'baseline')
        run_dir, run_config = create_run_dir(
            base_dir=getattr(config, 'root_path', None),
            model=model_type,
            seed=getattr(config, 'seed', 2022),
            run_name=getattr(config, 'run_name', None),
        )
        config.run_dir = run_dir
        metric_logger = MetricLogger(run_dir, config, run_config=run_config)
        config.metric_logger = metric_logger
        config.eval_every = getattr(config, 'eval_every', 2)
        config.num_eval_samples = getattr(config, 'num_eval_samples', 50)
        eval_datasets = [("imagenet_eeg", eeg_latents_dataset_test)]
        config.eval_datasets = eval_datasets
        print(f"Thesis logging: run_dir={run_dir}")

    pbar.update(1)
    pbar.set_description('Finetune')
    # Use CSVLogger when no logger set to avoid Lightning's tensorboardX warning
    if config.logger is None:
        try:
            from pytorch_lightning.loggers import CSVLogger
            config.logger = CSVLogger(save_dir=config.output_path, name='')
        except Exception:
            pass
    # finetune the model
    extra_callbacks = []
    if metric_logger is not None and run_dir and ExperimentLoggingCallback is not None:
        extra_callbacks.append(ExperimentLoggingCallback(metric_logger, generative_model, eval_datasets, config))
    check_val = getattr(config, 'check_val_every_n_epoch', 2) or 2
    trainer = create_trainer(config.num_epoch, config.precision, config.accumulate_grad, config.logger, check_val_every_n_epoch=check_val, extra_callbacks=extra_callbacks)
    val_limit = getattr(config, 'val_gen_limit', 2)
    print(f'Stage B remaining (approx): {config.num_epoch} epochs, val every {check_val} epoch(s). Full val <={val_limit} items. Progress bars show remaining.\n')
    if getattr(config, 'disable_image_generation_in_val', False):
        print('Stage B: skipping image generation during validation (disable_image_generation_in_val=True).\n')
    elif getattr(config, 'val_image_gen_every_n_epoch', 0) == 0:
        print('Stage B: skipping image generation during validation (val_image_gen_every_n_epoch=0).\n')
    elif getattr(config, 'val_image_gen_every_n_epoch', 0) > 0:
        print(f'Stage B: image generation in validation only every {config.val_image_gen_every_n_epoch} epoch(s).\n')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    generative_model.finetune(trainer, eeg_latents_dataset_train, eeg_latents_dataset_test,
                config.batch_size, config.lr, config.output_path, config=config)

    pbar.update(1)
    pbar.set_description('Generate & evaluate')
    # Skip post-training image generation when disabled (same flag as val; Stage C does full-quality gen)
    if getattr(config, 'disable_image_generation_in_val', False):
        print('Stage B: skipping post-training image generation (disable_image_generation_in_val=True). Run Stage C for full-quality generation.\n')
    else:
        generate_images(generative_model, eeg_latents_dataset_train, eeg_latents_dataset_test, config)
    pbar.update(1)
    pbar.set_description('Complete')
    pbar.close()

    return

def get_args_parser():
    parser = argparse.ArgumentParser('Double Conditioning LDM Finetuning', add_help=False)
    # project parameters
    parser.add_argument('--root_path', type=str, default=None, help='Repo root; default from config.')
    parser.add_argument('--splits_path', type=str, default=None, help='Override splits file (e.g. datasets/block_splits_tiny.pth)')
    parser.add_argument('--eeg_signals_path', type=str, default=None, help='Path to EEG signals (e.g. datasets/eeg_5_95_std.pth)')
    parser.add_argument('--pretrain_mbm_path', type=str, help='EEG encoder/MAE pretrain checkpoint (required for Stage B)')
    parser.add_argument('--resume_ckpt_path', type=str, default=None, help='Stage-B checkpoint to resume (loads config + weights + trainer state). Only flag that triggers resume.')
    parser.add_argument('--init_from_ckpt', type=str, default=None, help='Load model weights ONLY from this path; no config/optimizer/epoch. Fresh run from epoch 0.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Deprecated: use --pretrain_mbm_path and --resume_ckpt_path')
    parser.add_argument('--crop_ratio', type=float)
    parser.add_argument('--dataset', type=str)

    # finetune parameters
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int, default=None, help='DataLoader workers; 0=Windows-safe, 4–8 on Linux for faster training')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--precision', default=None, help='32 (full) | 16 (mixed, faster) | bf16 (A100); default 16')
    parser.add_argument('--accumulate_grad', type=int)
    parser.add_argument('--global_pool', type=bool)

    # diffusion sampling parameters
    parser.add_argument('--pretrain_gm_path', type=str)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--ddim_steps', type=int)
    parser.add_argument('--use_time_cond', type=bool)
    parser.add_argument('--eval_avg', type=bool)
    parser.add_argument('--val_gen_limit', type=int, default=None, help='Limit val generation items (default 2)')
    parser.add_argument('--val_ddim_steps', type=int, default=None, help='DDIM/PLMS steps during validation only (default 50; Stage C uses ddim_steps)')
    parser.add_argument('--val_num_samples', type=int, default=None, help='Samples per item during validation only (default 2; Stage C uses num_samples)')
    parser.add_argument('--test_gen_limit', type=int, default=None, help='Limit test images generated after training (e.g. 5 or 10 for fast; default 10)')
    parser.add_argument('--use_sarhm', type=str, default=None, choices=['true','false'], help='Enable SAR-HM')
    parser.add_argument('--ablation_mode', type=str, default=None, help='baseline|projection_only|hopfield_no_gate|full_sarhm')
    parser.add_argument('--proto_source', type=str, default=None, help='baseline_centroids|clip_text|learnable')
    parser.add_argument('--normalize_conditioning', type=str, default=None, choices=['true','false'], help='LayerNorm before fusion (true=layernorm)')
    parser.add_argument('--warmup_epochs', type=int, default=None, help='Epochs to ramp alpha_max from alpha_max_start to alpha_max_end (default 10)')
    parser.add_argument('--alpha_max_start', type=float, default=None, help='SAR-HM alpha_max at epoch 0 (default 0.05; keep small early)')
    parser.add_argument('--alpha_max_end', type=float, default=None, help='SAR-HM alpha_max after warmup (default 0.2)')
    parser.add_argument('--freeze_diffusion_until_epoch', type=int, default=None, help='Freeze UNet/VAE until this epoch (0=never). Bootstrap: set to num_epoch to train only cond_ln/SAR-HM.')
    parser.add_argument('--warm_start_from_baseline', type=str, default=None, choices=['true', 'false'], help='If true, init SAR adapter near zero (weights-only). Default false for fresh SAR-HM.')
    parser.add_argument('--clip_tune', type=str, default=None, choices=['true','false'], help='CLIP loss; false saves VRAM, works without ImageNet')

    # Thesis logging and evaluation
    parser.add_argument('--run_name', type=str, default=None, help='Optional run name for runs/<timestamp>_<model>_<seed>/')
    parser.add_argument('--model', type=str, default=None, choices=['baseline', 'sarhm'], help='Model type for run folder and logging')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed for run folder and reproducibility')
    parser.add_argument('--eval_every', type=int, default=2, help='Run evaluation every N epochs')
    parser.add_argument('--num_eval_samples', type=int, default=50, help='Max samples per dataset for evaluation')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=None, help='Validate every N epochs (e.g. 5 to fit 10 epochs in ~1 h)')
    parser.add_argument('--start_epoch', type=int, default=None, help='Skip training for epochs < this (e.g. 2 to test epoch 2 quickly)')
    parser.add_argument('--disable_image_generation_in_val', type=str, default=None, choices=['true', 'false'], help='If true, skip PLMS/DDIM image generation during Stage B validation (faster, cheaper). Stage C unchanged.')
    parser.add_argument('--val_image_gen_every_n_epoch', type=int, default=None, help='Generate images in val only every N epochs; 0 = never. Ignored if disable_image_generation_in_val=true.')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Resume from this checkpoint path (same as --checkpoint_path; loads model state and config from .pth)')
    parser.add_argument('--use_compile', action='store_true', help='Use torch.compile (PyTorch 2+) for faster training if available')
    parser.add_argument('--smoke_test', action='store_true', help='Short quality check: val_gen_limit=1, num_samples=4, val_ddim_steps=25, check_val_every_n_epoch=1, val_image_gen_every_n_epoch=1')
    parser.add_argument('--debug', action='store_true', help='Enable debug toggles: debug_cond_stats, debug_vae_roundtrip')
    parser.add_argument('--debug_sampling_steps', type=str, default=None, help='Comma-separated steps to save intermediate val images (e.g. 250,200,150,100,50)')
    parser.add_argument('--prune_unexpected_keys', action='store_true', help='When loading ckpt, drop keys not in model (default False)')
    # # distributed training parameters
    # parser.add_argument('--local_rank', type=int)

    return parser

def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    # Resume: --resume_ckpt overrides resume_ckpt_path; --checkpoint_path deprecated (resolved in __main__)
    if getattr(args, 'resume_ckpt', None) is not None:
        config.resume_ckpt_path = args.resume_ckpt
    # --smoke_test: short quality check run
    if getattr(args, 'smoke_test', False):
        config.val_gen_limit = 1
        config.num_samples = 4
        config.val_ddim_steps = 25
        config.check_val_every_n_epoch = 1
        config.val_image_gen_every_n_epoch = 1
        if getattr(config, 'disable_image_generation_in_val', True) == True:
            config.disable_image_generation_in_val = False
        if getattr(config, 'val_image_gen_every_n_epoch', 0) == 0:
            config.val_image_gen_every_n_epoch = 1
        print("[SMOKE_TEST] val_gen_limit=1 num_samples=4 val_ddim_steps=25 check_val_every_n_epoch=1 val_image_gen_every_n_epoch=1")
    # --debug: enable conditioning stats and VAE round-trip
    if getattr(args, 'debug', False):
        config.debug_cond_stats = True
        config.debug_vae_roundtrip = True
        if getattr(config, 'debug_sampling_steps', None) is None:
            config.debug_sampling_steps = [250, 200, 150, 100, 50]
        print("[DEBUG] debug_cond_stats=True debug_vae_roundtrip=True")
    if getattr(args, 'debug_sampling_steps', None):
        try:
            config.debug_sampling_steps = [int(x.strip()) for x in args.debug_sampling_steps.split(',') if x.strip()]
        except Exception:
            config.debug_sampling_steps = None
    # Normalize CLI bools
    if hasattr(args, 'disable_image_generation_in_val') and args.disable_image_generation_in_val is not None:
        config.disable_image_generation_in_val = (args.disable_image_generation_in_val == 'true')
    if hasattr(args, 'normalize_conditioning') and args.normalize_conditioning is not None:
        config.normalize_conditioning = (args.normalize_conditioning == 'true')
    if getattr(args, 'warm_start_from_baseline', None) is not None:
        config.warm_start_from_baseline = (args.warm_start_from_baseline == 'true')
    elif getattr(args, 'use_sarhm', False):
        config.warm_start_from_baseline = False  # default false for SAR-HM fresh runs
    if getattr(args, 'init_from_ckpt', None) is not None:
        config.init_from_ckpt = args.init_from_ckpt
    # When validation image gen is enabled, default to every epoch if not set
    if not getattr(config, 'disable_image_generation_in_val', True):
        if getattr(config, 'val_image_gen_every_n_epoch', 0) == 0:
            config.val_image_gen_every_n_epoch = 1
    # Normalize precision: CLI may pass "16", "32", or "bf16"
    if hasattr(config, 'precision') and isinstance(getattr(config, 'precision'), str):
        p = config.precision.strip().lower()
        if p == 'bf16':
            config.precision = 'bf16'
        elif p.isdigit():
            config.precision = int(p)
    return config

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)


def create_trainer(num_epoch, precision=32, accumulate_grad_batches=2, logger=None, check_val_every_n_epoch=0, extra_callbacks=None):
    from pytorch_lightning.callbacks import TQDMProgressBar
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required. No CUDA device found.")
    # GPU only: use single GPU (device 0)
    acc, devices = 'gpu', 1
    # Lightning 2.x: mixed precision strings
    if precision == 16 or (isinstance(precision, str) and precision == "16"):
        precision = "16-mixed"
    elif isinstance(precision, str) and precision.strip().lower() == "bf16":
        precision = "bf16-mixed"
    # 32 (or string "32") stays full precision; Lightning accepts 32
    progress_bar = TQDMProgressBar(refresh_rate=1)
    callbacks = [progress_bar]
    if extra_callbacks:
        callbacks = callbacks + list(extra_callbacks)
    return pl.Trainer(accelerator=acc, devices=devices, max_epochs=num_epoch, logger=logger,
            precision=precision, accumulate_grad_batches=accumulate_grad_batches,
            enable_checkpointing=False, enable_model_summary=False, gradient_clip_val=0.5,
            check_val_every_n_epoch=check_val_every_n_epoch, callbacks=callbacks,
            log_every_n_steps=1)
  
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.use_sarhm is not None:
        args.use_sarhm = args.use_sarhm == 'true'
    if args.clip_tune is not None:
        args.clip_tune = args.clip_tune == 'true'
    config = Config_Generative_Model()
    config = update_config(args, config)

    # Deprecate --checkpoint_path: resolve to pretrain_mbm_path (MAE) or resume_ckpt_path (Stage-B LDM)
    if getattr(args, 'checkpoint_path', None) is not None:
        try:
            peek = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print("[ARGS] WARNING: could not open --checkpoint_path: %s" % e)
            peek = None
        if peek is not None and is_mae_pretrain_ckpt(peek):
            config.pretrain_mbm_path = args.checkpoint_path
            print("[ARGS] --checkpoint_path looks like MBM/MAE pretrain; using it as --pretrain_mbm_path. Use --resume_ckpt_path for Stage-B resume.")
        else:
            config.resume_ckpt_path = args.checkpoint_path
            print("[ARGS] --checkpoint_path is deprecated; use --resume_ckpt_path.")
        config.checkpoint_path = args.checkpoint_path  # keep for any legacy reads

    # Load config from Stage-B checkpoint when resuming (so paths/epochs etc. can be restored)
    if config.resume_ckpt_path is not None:
        model_meta = torch.load(config.resume_ckpt_path, map_location='cpu', weights_only=False)
        ckp = config.resume_ckpt_path
        if is_mae_pretrain_ckpt(model_meta):
            print("[ERROR] resume_ckpt_path points to a MAE/MBM pretrain checkpoint. Use --pretrain_mbm_path for that file.")
            sys.exit(1)
        if 'config' not in model_meta:
            print('Checkpoint has no "config" key; using current config (only resume_ckpt_path set).')
        else:
            config = model_meta['config']
        config.resume_ckpt_path = ckp
        config.checkpoint_path = ckp
        # Re-apply CLI and fix paths only when we loaded config from checkpoint (e.g. from another machine)
        if 'config' in model_meta:
            config = update_config(args, config)
            default_cfg = Config_Generative_Model()
            for attr in default_cfg.__dict__:
                if not hasattr(config, attr):
                    setattr(config, attr, getattr(default_cfg, attr))
            config.root_path = default_cfg.root_path
            config.output_path = default_cfg.output_path
            if not os.path.isfile(getattr(config, 'pretrain_mbm_path', '')):
                config.pretrain_mbm_path = default_cfg.pretrain_mbm_path
                print('Using default pretrain_mbm_path (loaded path not found).')
            for path_attr in ['eeg_signals_path', 'splits_path', 'pretrain_gm_path', 'pretrain_mbm_path']:
                p = getattr(config, path_attr, None)
                if p and not os.path.isabs(p) and not os.path.isfile(p):
                    resolved = os.path.join(config.root_path, p)
                    if os.path.isfile(resolved) or os.path.isdir(resolved) or path_attr == 'pretrain_gm_path':
                        setattr(config, path_attr, resolved)
            print('Resuming from checkpoint: {}'.format(config.resume_ckpt_path))
        if getattr(args, 'disable_image_generation_in_val', None) is not None:
            config.disable_image_generation_in_val = (args.disable_image_generation_in_val == 'true')
        else:
            config.disable_image_generation_in_val = True
        if getattr(args, 'val_image_gen_every_n_epoch', None) is not None:
            config.val_image_gen_every_n_epoch = args.val_image_gen_every_n_epoch

    output_path = os.path.join(config.output_path, 'results', 'generation',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    config.output_path = output_path
    os.makedirs(output_path, exist_ok=True)
    
    wandb_init(config, output_path)

    # logger = WandbLogger()
    config.logger = None # logger
    main(config)
