"""
Stage C: Generate and evaluate images from EEG using a trained Stage B checkpoint.

Usage (full eval):
  python code/gen_eval_eeg.py --dataset EEG --model_path <ckpt.pth> \\
    --splits_path datasets/block_splits_by_image_single.pth \\
    --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml

Minimal Stage C (20 images):
  python code/gen_eval_eeg.py --dataset EEG --model_path <ckpt.pth> ... --num_samples 20 --split test

Load real prototypes (required for SAR-HM; use run's prototypes.pt, not baseline_centroids):
  python code/gen_eval_eeg.py ... --proto_path exps/results/generation/28-02-2026-21-42-16/prototypes.pt

SAR-HM++: When config.use_sarhmpp is True (from checkpoint), semantic_prototypes_path is set from
  --proto_path or from the checkpoint directory (semantic_prototypes.pt or prototypes.pt).

Or use latest run: --latest_run_dir exps/results/generation → copies to exps/latest/ and uses that.

Ablations: --ablation baseline | projection_only | hopfield_no_gate | full
  --no_sarhm / --disable_sarhm  force baseline
  --baseline_only   alpha=0
  --force_alpha 0.2   fix alpha
  --debug   extra logs + VAE roundtrip
"""
import os, sys
import shutil
import numpy as np
import torch
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms
from config import *
import wandb
import datetime
import argparse
from tqdm import tqdm

from config import Config_Generative_Model
from dataset import create_EEG_dataset
from dc_ldm.ldm_for_eeg import eLDM_eval
from dc_ldm.models.diffusion.plms import PLMSSampler
from utils.state_dict_utils import filter_state_dict_for_model, log_filter_info


def find_latest_generation_run(base_dir="exps/results/generation", require_both=False):
    """Return path to the newest subdir by mtime. If require_both, only consider dirs containing both checkpoint.pth and prototypes.pt."""
    if not os.path.isdir(base_dir):
        return None
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d))]
    if require_both:
        subdirs = [d for d in subdirs
                   if os.path.isfile(os.path.join(d, 'checkpoint.pth'))
                   and os.path.isfile(os.path.join(d, 'prototypes.pt'))]
    if not subdirs:
        return None
    def mtime(p):
        try:
            return os.path.getmtime(p)
        except OSError:
            return 0
    return max(subdirs, key=mtime)


def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))

def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')

def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

def wandb_init(config):
    wandb.init( project="dreamdiffusion",
                group='eval',
                anonymous="allow",
                config=config,
                reinit=True)

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

def get_args_parser():
    parser = argparse.ArgumentParser('Double Conditioning LDM Finetuning', add_help=False)
    # project parameters
    parser.add_argument('--root', type=str, default='../dreamdiffusion/')
    parser.add_argument('--dataset', type=str, default='GOD')
    parser.add_argument('--model_path', type=str)

    parser.add_argument('--splits_path', type=str, default=None,
                        help='Path to dataset splits.')
    parser.add_argument('--eeg_signals_path', type=str, default=None,
                        help='Path to EEG signals data.')

    parser.add_argument('--config_patch', type=str, default=None,
                        help='sd config path.')
    
    parser.add_argument('--imagenet_path', type=str, default=None,
                        help='ImageNet root for EEG GT images (required for dataset EEG). Set or use env IMAGENET_PATH.')

    # Stage C debug / mini run
    parser.add_argument('--proto_path', type=str, default=None,
                        help='Load prototypes from this path (e.g. .pt file) so proto_source is not dummy.')
    parser.add_argument('--max_items', type=int, default=None,
                        help='Limit number of EEG samples to generate (train and test if not overridden).')
    parser.add_argument('--max_train_items', type=int, default=None,
                        help='Limit train set generation (overrides max_items for train).')
    parser.add_argument('--max_test_items', type=int, default=0,
                        help='Mini eval: use only first K test samples (0=no limit). Deterministic Subset(range(K)).')
    parser.add_argument('--no_sarhm', action='store_true',
                        help='Force SAR-HM off in conditioning (baseline only).')
    parser.add_argument('--force_alpha', type=float, default=-1.0,
                        help='Override alpha with fixed value; use -1 to disable.')
    parser.add_argument('--baseline_only', action='store_true',
                        help='Equivalent to alpha=0 (no SAR-HM fusion).')
    parser.add_argument('--debug', action='store_true',
                        help='Extra prints and VAE roundtrip saved to figures/vae_roundtrip.png.')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples per EEG (default 20 for minimal run).')
    parser.add_argument('--ddim_steps', type=int, default=None,
                        help='Override DDIM/PLMS steps for generation (default: use checkpoint config).')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'train'],
                        help='Which split to generate (default test).')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Start index into dataset (default 0).')
    parser.add_argument('--save_intermediates', action='store_true',
                        help='Save decoded images at intermediate steps (if supported).')
    parser.add_argument('--ablation', type=str, default='full',
                        choices=['baseline', 'projection_only', 'hopfield_no_gate', 'full'],
                        help='Ablation mode (default full).')
    parser.add_argument('--conf_threshold', type=float, default=None,
                        help='Override conf_threshold (if conf < this, alpha=0).')
    parser.add_argument('--alpha_max', type=float, default=None,
                        help='Override alpha_max.')
    parser.add_argument('--disable_sarhm', action='store_true',
                        help='Same as --no_sarhm: force baseline only.')
    parser.add_argument('--cfg_scale', type=float, default=1.0,
                        help='Classifier-free guidance scale (default 1.0 = no CFG).')
    parser.add_argument('--cfg_uncond', type=str, default='zeros', choices=['zeros', 'baseline'],
                        help='Unconditional conditioning for CFG: zeros or baseline (default zeros).')
    parser.add_argument('--latest_run_dir', type=str, default=None,
                        help='Base dir for generation runs (e.g. exps/results/generation); copy latest to exps/latest/ and use it.')
    parser.add_argument('--no_auto_use_latest', action='store_true', help='Disable auto-load of prototypes.pt from checkpoint dir (default: auto-load).')
    parser.add_argument('--latest_root', type=str, default=None,
                        help='Root to scan for latest run containing checkpoint.pth and prototypes.pt.')
    parser.add_argument('--prune_unexpected_keys', action='store_true', help='When loading ckpt, drop keys not in model (default False).')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed for reproducibility (mini eval).')

    # Stage C sanity ladder (diagnose noise: VAE/uncond/EEG/latent)
    parser.add_argument('--sanity_checks', action='store_true', help='Run sanity ladder only (VAE roundtrip, uncond sample, EEG sample, latent stats).')
    parser.add_argument('--sanity_items', type=int, default=2, help='Number of items for sanity checks (default 2).')
    parser.add_argument('--sanity_ddim_steps', type=int, default=30, help='DDIM/PLMS steps for sanity sampling (default 30).')
    parser.add_argument('--dump_first_batch', action='store_true', default=True, help='Dump first batch tensor stats when sanity_checks (default True).')
    parser.add_argument('--no_dump_first_batch', action='store_true', help='Disable dump_first_batch.')
    parser.add_argument('--also_run_eval', action='store_true', help='If set with --sanity_checks, run full eval after sanity; otherwise exit after sanity.')

    return parser


def run_sanity_ladder(generative_model, dataset_test, output_path, device, args):
    """Run Stage C sanity ladder: VAE roundtrip, unconditional sample, EEG sample, latent stats, cond stats, weights check."""
    sanity_dir = os.path.join(output_path, 'sanity')
    os.makedirs(sanity_dir, exist_ok=True)
    n_items = min(getattr(args, 'sanity_items', 2), len(dataset_test))
    steps = getattr(args, 'sanity_ddim_steps', 30)
    dump_batch = not getattr(args, 'no_dump_first_batch', False)

    model = generative_model.model.to(device)
    model.eval()
    cond_dim = getattr(generative_model, 'cond_dim', 768)
    ldm_config = generative_model.ldm_config
    C, H, W = ldm_config.model.params.channels, ldm_config.model.params.image_size, ldm_config.model.params.image_size
    shape = (C, H, W)

    # ----- (5) Weights check -----
    unet_params = []
    vae_params = []
    for name, p in model.named_parameters():
        if 'model.diffusion_model' in name:
            unet_params.append(p.data.flatten())
        elif 'first_stage_model' in name:
            vae_params.append(p.data.flatten())
    unet_flat = torch.cat(unet_params, 0) if unet_params else torch.tensor(0.0)
    vae_flat = torch.cat(vae_params, 0) if vae_params else torch.tensor(0.0)
    unet_mean = unet_flat.float().mean().item() if unet_flat.numel() else float('nan')
    vae_mean = vae_flat.float().mean().item() if vae_flat.numel() else float('nan')
    print("[SANITY][WEIGHTS] unet_param_mean=%.6f vae_param_mean=%.6f" % (unet_mean, vae_mean))

    # ----- (3) Conditioning stats (one batch) -----
    if n_items > 0:
        item = dataset_test[0]
        eeg = item['eeg']
        if not isinstance(eeg, torch.Tensor):
            eeg = torch.as_tensor(eeg, dtype=torch.float32, device=device)
        else:
            eeg = eeg.to(device)
        if eeg.dim() == 2:
            eeg = eeg.unsqueeze(0)
        with torch.no_grad():
            c, _ = model.get_learned_conditioning(eeg)
        c_tensor = c if isinstance(c, torch.Tensor) else c.get(list(c.keys())[0], c)
        if isinstance(c_tensor, (list, tuple)):
            c_tensor = c_tensor[0]
        csm = getattr(model, 'cond_stage_model', None)
        if csm is not None:
            extra = getattr(csm, '_sarhm_extra', {})
            for name, t in [('c_base', extra.get('c_base')), ('c_sar', extra.get('c_sar')), ('c_final', c_tensor)]:
                if t is not None:
                    t = t.float()
                    nan_inf = torch.isnan(t).any().item() or torch.isinf(t).any().item()
                    if nan_inf:
                        print("[SANITY][COND] *** %s contains NaN or Inf" % name)
                    print("[SANITY][COND] %s mean=%.4f std=%.4f min=%.4f max=%.4f" % (
                        name, t.mean().item(), t.std().item(), t.min().item(), t.max().item()))
            alpha = extra.get('alpha')
            if alpha is not None:
                a = alpha.float()
                print("[SANITY][COND] alpha min=%.4f mean=%.4f max=%.4f" % (a.min().item(), a.mean().item(), a.max().item()))
        else:
            c_tensor = c_tensor.float() if isinstance(c_tensor, torch.Tensor) else c_tensor
            if torch.isnan(c_tensor).any() or torch.isinf(c_tensor).any():
                print("[SANITY][COND] *** c_final contains NaN or Inf")
            print("[SANITY][COND] c_final mean=%.4f std=%.4f min=%.4f max=%.4f" % (
                c_tensor.mean().item(), c_tensor.std().item(), c_tensor.min().item(), c_tensor.max().item()))

    # ----- (A) VAE roundtrip -----
    if n_items > 0:
        item = dataset_test[0]
        img = item['image']
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        else:
            img = torch.tensor(img, dtype=torch.float32)
        if img.dim() == 3 and img.shape[-1] == 3:
            img = rearrange(img, 'h w c -> 1 c h w')
        elif img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(device)
        if img.max() <= 1.0 and img.min() >= 0.0:
            img = img * 2.0 - 1.0
        with torch.no_grad():
            enc = model.encode_first_stage(img)
            z = model.get_first_stage_encoding(enc)
            rec = model.decode_first_stage(z.float())
        rec = torch.clamp((rec + 1.0) / 2.0, 0.0, 1.0)
        img_01 = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)
        gt_np = (255. * rearrange(img_01[0], 'c h w -> h w c').cpu().numpy()).astype(np.uint8)
        rec_np = (255. * rearrange(rec[0], 'c h w -> h w c').cpu().numpy()).astype(np.uint8)
        Image.fromarray(gt_np).save(os.path.join(sanity_dir, 'vae_roundtrip_gt.png'))
        Image.fromarray(rec_np).save(os.path.join(sanity_dir, 'vae_roundtrip_recon.png'))
        print("[SANITY][VAE] gt min=%.4f max=%.4f mean=%.4f std=%.4f" % (
            img_01.min().item(), img_01.max().item(), img_01.mean().item(), img_01.std().item()))
        print("[SANITY][VAE] recon min=%.4f max=%.4f mean=%.4f std=%.4f" % (
            rec.min().item(), rec.max().item(), rec.mean().item(), rec.std().item()))
        rec_finite = torch.isfinite(rec).all().item()
        print("[SANITY][VAE] recon finite=%s" % rec_finite)
        if not rec_finite:
            print("[SANITY][VAE] *** NaN/Inf detected in VAE reconstruction")

    # ----- (B) Unconditional sample (zero conditioning) -----
    sampler = PLMSSampler(model)
    dtype = next(model.parameters()).dtype
    c_zeros = torch.zeros(1, 77, cond_dim, device=device, dtype=dtype)
    with model.ema_scope():
        model.eval()
        samples_z, _ = sampler.sample(S=steps, conditioning=c_zeros, batch_size=1, shape=shape, verbose=False)
    z_final = samples_z
    print("[SANITY][LATENT] z min=%.4f max=%.4f mean=%.4f std=%.4f finite=%s shape=%s" % (
        z_final.min().item(), z_final.max().item(), z_final.mean().item(), z_final.std().item(),
        torch.isfinite(z_final).all().item(), tuple(z_final.shape)))
    if not torch.isfinite(z_final).all().item():
        print("[SANITY][LATENT] *** NaN/Inf detected in latent z")
    with torch.no_grad():
        x_dec = model.decode_first_stage(z_final.float())
    x_dec = torch.clamp((x_dec + 1.0) / 2.0, 0.0, 1.0)
    print("[SANITY][DECODE] img min=%.4f max=%.4f mean=%.4f std=%.4f finite=%s" % (
        x_dec.min().item(), x_dec.max().item(), x_dec.mean().item(), x_dec.std().item(), torch.isfinite(x_dec).all().item()))
    if not torch.isfinite(x_dec).all().item():
        print("[SANITY][DECODE] *** NaN/Inf detected in decoded image")
    uncond_path = os.path.join(sanity_dir, 'uncond_sample_steps%d.png' % steps)
    Image.fromarray((255. * rearrange(x_dec[0], 'c h w -> h w c').cpu().numpy()).astype(np.uint8)).save(uncond_path)
    print("[SANITY][UNCOND] saved=%s steps=%d" % (uncond_path, steps))
    print("[SANITY][UNCOND] mode=zero_cond (conditioning=zeros [1,77,%d])" % cond_dim)

    # ----- (C) EEG-conditioned sample -----
    if n_items > 0:
        from torch.utils.data import Subset
        subset_one = Subset(dataset_test, [0])
        grid, samples = generative_model.generate(
            subset_one, num_samples=1, ddim_steps=steps, HW=None, limit=1, state=None, output_path=None,
            cfg_scale=1.0, cfg_uncond='zeros')
        if samples is not None and samples.size > 0:
            # samples shape (1, 2, C, H, W) -> gt + 1 gen
            gen = samples[0, 1]
            if gen.shape[0] == 3:
                arr = (255. * np.transpose(gen, (1, 2, 0))).astype(np.uint8)
            else:
                arr = (255. * gen).astype(np.uint8)
            eeg_path = os.path.join(sanity_dir, 'eeg_cond_sample_steps%d.png' % steps)
            Image.fromarray(arr).save(eeg_path)
            print("[SANITY][EEG] saved=%s steps=%d" % (eeg_path, steps))

    # ----- dump_first_batch -----
    if dump_batch and n_items > 0:
        item = dataset_test[0]
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().float()
                print("[SANITY][BATCH] %s shape=%s min=%.4f max=%.4f mean=%.4f" % (
                    k, tuple(v.shape), v.min().item(), v.max().item(), v.mean().item()))
            elif isinstance(v, np.ndarray):
                print("[SANITY][BATCH] %s shape=%s dtype=%s" % (k, v.shape, v.dtype))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if getattr(args, 'imagenet_path', None) is None and os.environ.get('IMAGENET_PATH'):
        args.imagenet_path = os.environ.get('IMAGENET_PATH')
    root = args.root
    target = args.dataset
    if target == 'EEG' and (not getattr(args, 'imagenet_path', None) or not str(getattr(args, 'imagenet_path', '')).strip()):
        print("ERROR: dataset=EEG requires imagenet_path for real GT images.")
        print("  Pass --imagenet_path /path/to/ILSVRC2012 or set IMAGENET_PATH.")
        sys.exit(1)

    seed = getattr(args, 'seed', 2022)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Optional: use latest run from generation dir and copy to exps/latest/
    if getattr(args, 'latest_run_dir', None):
        base = os.path.abspath(args.latest_run_dir)
        run_dir = find_latest_generation_run(base)
        if run_dir is None:
            print("[DEBUG] [LATEST] No subdirs in %s; ignoring --latest_run_dir" % base)
        else:
            latest_dir = os.path.join(root, 'exps', 'latest')
            os.makedirs(latest_dir, exist_ok=True)
            ckpt_src = os.path.join(run_dir, 'checkpoint.pth')
            proto_src = os.path.join(run_dir, 'prototypes.pt')
            for src, name in [(ckpt_src, 'checkpoint.pth'), (proto_src, 'prototypes.pt')]:
                if os.path.isfile(src):
                    dst = os.path.join(latest_dir, name)
                    shutil.copy2(src, dst)
                    print("[DEBUG] [LATEST] copied %s -> %s" % (src, dst))
            if os.path.isfile(ckpt_src):
                args.model_path = os.path.join(latest_dir, 'checkpoint.pth')
            if os.path.isfile(proto_src):
                args.proto_path = os.path.join(latest_dir, 'prototypes.pt')

    if getattr(args, 'disable_sarhm', False):
        args.no_sarhm = True

    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found: {args.model_path}")
        print("\nThis checkpoint is required for generation. You need to either:")
        print("1. Run Stage B training first: python code/eeg_ldm.py")
        print("   Then use the checkpoint from: results/generation/[timestamp]/checkpoint.pth")
        print("2. Or download a pre-trained checkpoint and place it at the specified path")
        sys.exit(1)

    sd = torch.load(args.model_path, map_location='cpu', weights_only=False)
    config = sd['config']
    config.root_path = root
    if getattr(args, 'ddim_steps', None) is not None:
        config.ddim_steps = args.ddim_steps
        print('[MINI_EVAL] ddim_steps overridden to %d' % args.ddim_steps)
    # SAR-HM++: set semantic_prototypes_path so eLDM_eval loads the semantic memory
    if getattr(config, 'use_sarhmpp', False):
        if getattr(args, 'proto_path', None) and os.path.isfile(args.proto_path):
            config.semantic_prototypes_path = args.proto_path
            print('[SAR-HM++] semantic_prototypes_path=%s' % args.proto_path)
        else:
            model_dir = os.path.dirname(os.path.abspath(args.model_path))
            for name in ('semantic_prototypes.pt', 'prototypes.pt'):
                candidate = os.path.join(model_dir, name)
                if os.path.isfile(candidate):
                    config.semantic_prototypes_path = candidate
                    print('[SAR-HM++] semantic_prototypes_path from checkpoint dir: %s' % candidate)
                    break
        if getattr(config, 'semantic_prototypes_path', None) is None:
            print('[SAR-HM++] WARNING: no semantic_prototypes.pt found; pass --proto_path or place file in checkpoint dir. Inference will fall back to baseline (alpha=0).')

    output_path = os.path.join(config.root_path, 'results', 'eval',
                    '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    os.makedirs(output_path, exist_ok=True)

    # ----- DEBUG: config_patch path -----
    _config_patch = getattr(args, 'config_patch', None)
    if _config_patch:
        print("[DEBUG] [CONFIG_PATCH] path=%s exists=%s" % (_config_patch, os.path.exists(_config_patch)))
    else:
        print("[DEBUG] [CONFIG_PATCH] path=MISSING (not provided; required for eLDM_eval)")

    # ----- DEBUG: Pretrained SD checkpoint existence (B) -----
    try:
        _sd_ckpt_path = os.path.join(getattr(config, 'pretrain_gm_path', ''), 'models', 'v1-5-pruned.ckpt')
        _sd_exists = os.path.exists(_sd_ckpt_path)
        print("[DEBUG] [SD_CKPT] path=%s exists=%s" % (_sd_ckpt_path, _sd_exists))
        if not _sd_exists:
            print("[DEBUG] [SD_CKPT] WARNING: SD 1.5 checkpoint file not found. If Stage B was trained on this machine, UNet/VAE were loaded from this path; missing file can mean random-init UNet/VAE and thus pure noise. If this checkpoint was trained elsewhere, weights are inside the .pth and this warning may be benign.")
    except Exception as _e:
        print("[DEBUG] [SD_CKPT] MISSING (exception): %s" % _e)

    # GPU only: require CUDA, no CPU fallback
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU required. No CUDA device found. "
            "Install PyTorch with CUDA and ensure a GPU is available."
        )
    device = torch.device('cuda')
    # Large banner: GPU IS AVAILABLE
    _name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA"
    _bar = "=" * 56
    print("\n" + _bar + "\n\n              **   GPU   **\n\n           **   IT IS AVAILABLE   **\n\n              Device: %s\n\n" % _name[:36] + _bar + "\n")
    pbar = tqdm(total=5, desc='Stage C total', unit='phase',
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} phases [{elapsed}<{remaining}]')

    pbar.set_description('Load data')
    crop_pix = int(config.crop_ratio*config.img_size)
    img_transform_train = transforms.Compose([
        normalize,
        transforms.Resize((512, 512)),
        channel_last
    ])
    img_transform_test = transforms.Compose([
        normalize, transforms.Resize((512, 512)),
        channel_last
    ])
    dataset_train, dataset_test = create_EEG_dataset(eeg_signals_path=args.eeg_signals_path,
                splits_path=args.splits_path, imagenet_path=args.imagenet_path,
                image_transform=[img_transform_train, img_transform_test], subject=4)
    num_voxels = dataset_test.dataset.data_len
    max_test = getattr(args, 'max_test_items', 0) or 0
    if max_test > 0:
        from torch.utils.data import Subset
        n_test = min(max_test, len(dataset_test))
        if n_test == 0:
            raise ValueError("[MINI_EVAL] max_test_items=%d but test split is empty." % max_test)
        dataset_test = Subset(dataset_test, list(range(n_test)))
        print('[MINI_EVAL] test_items=%d' % len(dataset_test))
    pbar.update(1)

    pbar.set_description('Load model')
    generative_model = eLDM_eval(args.config_patch, num_voxels,
                device=device, pretrain_root=config.pretrain_gm_path, logger=getattr(config, 'logger', None),
                ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond,
                clip_tune=getattr(config, 'clip_tune', True), cls_tune=getattr(config, 'cls_tune', False),
                main_config=config)
    # ----- DEBUG: Checkpoint load integrity (A) -----
    model_keys = set(generative_model.model.state_dict().keys())
    ckpt_raw = sd['model_state_dict']
    ckpt_sd, filter_info = filter_state_dict_for_model(
        ckpt_raw,
        model_state_keys=model_keys,
        drop_exact_keys=None,
        drop_prefixes=None,
        prune_unexpected_keys=getattr(args, 'prune_unexpected_keys', False),
    )
    log_filter_info(filter_info, tag='[CKPT_FILTER]')
    try:
        missing, unexpected = generative_model.model.load_state_dict(ckpt_sd, strict=False)
        missing_list = list(missing)
        unexpected_list = list(unexpected)
        print("[DEBUG] [CKPT_LOAD] total missing keys=%d total unexpected keys=%d" % (len(missing_list), len(unexpected_list)))
        def _count_prefix(keys, prefix):
            return sum(1 for k in keys if k.startswith(prefix))
        print("[DEBUG] [CKPT_LOAD] missing by prefix: model.diffusion_model=%d first_stage_model=%d cond_stage_model=%d (sarhm in key)=%d" % (
            _count_prefix(missing_list, "model.diffusion_model"),
            _count_prefix(missing_list, "first_stage_model"),
            _count_prefix(missing_list, "cond_stage_model"),
            _count_prefix(missing_list, "cond_stage_model.sarhm") + _count_prefix(missing_list, "sarhm")))
        print("[DEBUG] [CKPT_LOAD] unexpected by prefix: model.diffusion_model=%d first_stage_model=%d cond_stage_model=%d (sarhm in key)=%d" % (
            _count_prefix(unexpected_list, "model.diffusion_model"),
            _count_prefix(unexpected_list, "first_stage_model"),
            _count_prefix(unexpected_list, "cond_stage_model"),
            _count_prefix(unexpected_list, "cond_stage_model.sarhm") + _count_prefix(unexpected_list, "sarhm")))
        print("[DEBUG] [CKPT_LOAD] first 20 missing keys: %s" % (missing_list[:20],))
        print("[DEBUG] [CKPT_LOAD] first 20 unexpected keys: %s" % (unexpected_list[:20],))
        # If only cond_ln is missing, checkpoint was saved without LayerNorm conditioning (expected with older config)
        cond_ln_only = (len(missing_list) <= 2 and all(
            k in ('cond_stage_model.cond_ln.weight', 'cond_stage_model.cond_ln.bias') for k in missing_list))
        if cond_ln_only and missing_list:
            print("[DEBUG] [CKPT_LOAD] cond_ln missing is OK: using random init (checkpoint from run without normalize_conditioning/layernorm).")
    except Exception as e:
        print("[DEBUG] [CKPT_LOAD] load_state_dict failed or no return: %s" % e)
        generative_model.model.load_state_dict(ckpt_sd, strict=False)
    print('load ldm successfully')
    state = sd.get('state')
    pbar.update(1)

    # ----- Resolve proto_path: explicit, else same dir as checkpoint, else latest_root -----
    proto_path = getattr(args, 'proto_path', None)
    if not proto_path and not getattr(args, 'no_auto_use_latest', False):
        ckpt_dir = os.path.dirname(os.path.abspath(args.model_path))
        candidate = os.path.join(ckpt_dir, 'prototypes.pt')
        if os.path.isfile(candidate):
            proto_path = candidate
            args.proto_path = candidate
            print("[PROTO] auto_use_latest: using %s" % candidate)
    if not proto_path and getattr(args, 'latest_root', None):
        latest_root = os.path.abspath(args.latest_root)
        run_dir = find_latest_generation_run(latest_root, require_both=True)
        if run_dir:
            candidate = os.path.join(run_dir, 'prototypes.pt')
            if os.path.isfile(candidate):
                proto_path = candidate
                args.proto_path = candidate
                print("[PROTO] latest_root: using %s" % candidate)
    if not proto_path:
        proto_path = getattr(args, 'proto_path', None)

    # ----- Prototype loading (so proto_source is not dummy) -----
    csm = getattr(generative_model.model, 'cond_stage_model', None)
    if proto_path and csm is not None:
        if not os.path.isfile(proto_path):
            print("[DEBUG] [PROTO] WARNING: proto_path not found: %s (keeping proto_source as-is)" % proto_path)
        else:
            try:
                if getattr(csm, 'sarhm_prototypes', None) is not None:
                    ok = csm.sarhm_prototypes.load_from_path(proto_path)
                    if ok:
                        csm._proto_source = "loaded"
                        dev = getattr(generative_model.model, 'device', None) or next(generative_model.model.parameters()).device
                        dtype = next(generative_model.model.parameters()).dtype
                        csm.sarhm_prototypes.prototypes.data = csm.sarhm_prototypes.prototypes.data.to(device=dev, dtype=dtype)
                        P = csm.sarhm_prototypes.P
                        if P is not None:
                            p = P.float()
                            finite = torch.isfinite(P).all().item()
                            print("[DEBUG] [PROTO] loaded path=%s source=loaded shape=%s dtype=%s finite=%s" % (
                                proto_path, tuple(P.shape), str(P.dtype), finite))
                            print("[PROTO] using path=%s source=loaded exists=True shape=%s finite=%s" % (
                                proto_path, tuple(P.shape), finite))
                            print("[DEBUG] [PROTO] P mean=%.6f std=%.6f min=%.6f max=%.6f" % (
                                p.mean().item(), p.std().item(), p.min().item(), p.max().item()))
                    else:
                        print("[DEBUG] [PROTO] WARNING: load_from_path returned False for %s (shape mismatch or invalid file)" % proto_path)
                else:
                    print("[DEBUG] [PROTO] WARNING: cond_stage_model has no sarhm_prototypes (SAR-HM off?)")
            except Exception as e:
                print("[DEBUG] [PROTO] WARNING: failed to load %s: %s" % (proto_path, e))
    elif proto_path:
        print("[DEBUG] [PROTO] WARNING: proto_path given but cond_stage_model not found")

    if csm is not None and getattr(csm, 'use_sarhm', False) and not csm.has_valid_prototypes():
        print("[PROTO] missing or invalid -> SAR-HM alpha forced 0 (baseline only)")

    # ----- Acceptance: when proto_path was provided, must have valid prototypes and not dummy -----
    if proto_path and csm is not None and getattr(csm, 'use_sarhm', False):
        valid = csm.has_valid_prototypes()
        src = csm.get_proto_source()
        if not valid:
            print("[DEBUG] [PROTO] *** ASSERTION: proto_path provided but has_valid_prototypes()=False. Fix proto file or use --baseline_only.")
        if src == "dummy":
            print("[DEBUG] [PROTO] *** ASSERTION: proto_path provided but proto_source still dummy. Load failed or path wrong.")
        # Optional strict assert (can be disabled for debugging):
        if getattr(args, 'debug', False):
            assert valid, "proto_path set but has_valid_prototypes() is False"
            assert src != "dummy", "proto_path set but proto_source is still dummy"

    # ----- Ablation flags (no_sarhm, baseline_only, force_alpha) -----
    if csm is not None:
        csm._no_sarhm = getattr(args, 'no_sarhm', False)
        csm._baseline_only = getattr(args, 'baseline_only', False)
        csm._force_alpha = float(args.force_alpha) if getattr(args, 'force_alpha', -1) >= 0 else None
        csm.debug_cond_stats = getattr(args, 'debug', False) or getattr(config, 'debug_cond_stats', False)
        if getattr(args, 'conf_threshold', None) is not None:
            csm.conf_threshold = float(args.conf_threshold)
        if getattr(args, 'alpha_max', None) is not None:
            csm.alpha_max = float(args.alpha_max)
        ab = getattr(args, 'ablation', 'full')
        if ab == 'baseline':
            csm._baseline_only = True
        else:
            csm.ablation_mode = {'projection_only': 'projection_only', 'hopfield_no_gate': 'hopfield_no_gate', 'full': 'full_sarhm'}.get(ab, 'full_sarhm')

    # ----- Warn if proto_source is dummy and SAR-HM is on -----
    if csm is not None and getattr(csm, 'use_sarhm', False) and not getattr(csm, '_no_sarhm', False):
        ps = csm.get_proto_source() if hasattr(csm, 'get_proto_source') else getattr(csm, '_proto_source', 'dummy')
        if ps == 'dummy':
            print("[DEBUG] [PROTO] *** WARNING: proto_source=dummy and SAR-HM is ON. Provide --proto_path <prototypes.pt> for real prototypes, or use --baseline_only / --no_sarhm for safe debug.")
            if getattr(args, 'debug', False):
                print("[DEBUG] [PROTO] debug=True: forcing alpha=0 as safety (baseline-only fusion).")
                csm._baseline_only = True

    # ----- DEBUG: scale_factor at inference (C) -----
    try:
        sf = getattr(generative_model.model, 'scale_factor', None)
        if sf is not None and hasattr(sf, 'item'):
            sf_val = float(sf.item())
        else:
            sf_val = float(sf) if sf is not None else None
        print("[DEBUG] [SCALE_FACTOR] value=%s expected=0.18215 match=%s" % (sf_val, (abs(sf_val - 0.18215) < 1e-5) if sf_val is not None else "MISSING"))
        if sf_val is not None and abs(sf_val - 0.18215) >= 1e-5:
            print("[DEBUG] [SCALE_FACTOR] *** WARNING: scale_factor != 0.18215; decode may produce noise.")
        if sf_val is None:
            print("[DEBUG] [SCALE_FACTOR] MISSING: generative_model.model.scale_factor not found")
    except Exception as e:
        print("[DEBUG] [SCALE_FACTOR] MISSING (exception): %s" % e)

    # ----- DEBUG: Conditioning sanity once per run (D) -----
    try:
        if len(dataset_test) == 0:
            print("[DEBUG] [COND] skip (test set empty)")
        else:
            first_item = dataset_test[0]
            eeg = first_item['eeg']
            if hasattr(eeg, 'dim') and eeg.dim() == 2:
                eeg = eeg.unsqueeze(0).to(device)
            else:
                eeg = torch.as_tensor(eeg, dtype=torch.float32, device=device) if not isinstance(eeg, torch.Tensor) else eeg.to(device)
            if eeg.dim() == 2:
                eeg = eeg.unsqueeze(0)
            with torch.no_grad():
                c, _ = generative_model.model.get_learned_conditioning(eeg)
            c_final = c if isinstance(c, torch.Tensor) else (c[list(c.keys())[0]][0] if isinstance(c, dict) else None)
            if c_final is not None:
                cf = c_final.float()
                print("[DEBUG] [COND] c_final mean=%.6f std=%.6f min=%.6f max=%.6f" % (cf.mean().item(), cf.std().item(), cf.min().item(), cf.max().item()))
                nan_inf = torch.isnan(cf).any().item() or torch.isinf(cf).any().item()
                print("[DEBUG] [COND] c_final has_NaN_or_Inf=%s" % nan_inf)
            csm_inner = getattr(generative_model.model, 'cond_stage_model', None)
            if csm_inner is not None and getattr(csm_inner, '_sarhm_extra', None):
                extra = csm_inner._sarhm_extra
                c_base = extra.get('c_base')
                if c_base is not None:
                    cb = c_base.float()
                    print("[DEBUG] [COND] c_base mean=%.6f std=%.6f min=%.6f max=%.6f" % (cb.mean().item(), cb.std().item(), cb.min().item(), cb.max().item()))
                c_sar = extra.get('c_sar')
                if c_sar is not None:
                    cs = c_sar.float()
                    print("[DEBUG] [COND] c_sar mean=%.6f std=%.6f min=%.6f max=%.6f" % (cs.mean().item(), cs.std().item(), cs.min().item(), cs.max().item()))
                alpha = extra.get('alpha')
                if alpha is not None:
                    ah = alpha.float()
                    print("[DEBUG] [COND] alpha min=%.6f mean=%.6f max=%.6f" % (ah.min().item(), ah.mean().item(), ah.max().item()))
            else:
                print("[DEBUG] [COND] c_base/c_sar/alpha not available (baseline or _sarhm_extra missing)")
    except Exception as e:
        print("[DEBUG] [COND] MISSING (exception): %s" % e)

    # ----- DEBUG: VAE round-trip sanity (E); only when --debug -----
    if getattr(args, 'debug', False) and len(dataset_test) > 0:
        try:
            model = generative_model.model.to(device)
            model.eval()
            sample_item = dataset_test[0]
            img = sample_item['image']
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            else:
                img = img.float() if isinstance(img, torch.Tensor) else torch.tensor(img, dtype=torch.float32)
            if img.dim() == 3 and img.shape[-1] == 3:
                img = rearrange(img, 'h w c -> 1 c h w')
            elif img.dim() == 3:
                img = img.unsqueeze(0)
            img = img.to(device=device, dtype=torch.float32)
            if img.max() <= 1.0 and img.min() >= 0.0:
                img = img * 2.0 - 1.0
            with torch.no_grad():
                enc = model.encode_first_stage(img)
                z = model.get_first_stage_encoding(enc)
                z = z.float()
                rec = model.decode_first_stage(z)
            rec = rec.float()
            rec = torch.clamp((rec + 1.0) / 2.0, 0.0, 1.0)
            rec_np = (255. * rearrange(rec[0], 'c h w -> h w c').cpu().numpy()).astype(np.uint8)
            fig_dir = os.path.join(output_path, 'figures')
            os.makedirs(fig_dir, exist_ok=True)
            vae_path = os.path.join(fig_dir, 'vae_roundtrip.png')
            Image.fromarray(rec_np).save(vae_path)
            print("[DEBUG] [VAE_ROUNDTRIP] saved %s" % vae_path)
            rec_mean = rec.float().mean().item()
            rec_std = rec.float().std().item()
            if rec_std < 1e-5 or torch.isnan(rec).any() or torch.isinf(rec).any():
                print("[DEBUG] [VAE_ROUNDTRIP] *** WARNING: Reconstruction looks wrong (near-constant, NaN or Inf). VAE/scale_factor pipeline or weights may be incorrect.")
            else:
                print("[DEBUG] [VAE_ROUNDTRIP] recon mean=%.4f std=%.4f (sanity check)" % (rec_mean, rec_std))
        except Exception as e:
            print("[DEBUG] [VAE_ROUNDTRIP] MISSING (exception): %s" % e)
            print("[DEBUG] [VAE_ROUNDTRIP] *** WARNING: Could not run VAE round-trip. If reconstructions are noise, VAE/scale_factor pipeline is wrong or weights not loaded.")

    # C) Professor-friendly attention plot when SAR-HM is on
    if getattr(config, 'use_sarhm', False) and len(dataset_test) > 0:
        try:
            fig_dir = os.path.join(output_path, 'figures')
            os.makedirs(fig_dir, exist_ok=True)
            first_item = dataset_test[0]
            eeg = first_item['eeg']
            if eeg.dim() == 2:
                eeg = eeg.unsqueeze(0).to(device)
            else:
                eeg = eeg.to(device)
            with torch.no_grad():
                _ = generative_model.model.get_learned_conditioning(eeg)
            csm = getattr(generative_model.model, 'cond_stage_model', None)
            if csm is not None and getattr(csm, '_sarhm_extra', None):
                attn = csm._sarhm_extra.get('attn')
                if attn is not None:
                    _code_dir = os.path.dirname(os.path.abspath(__file__))
                    if _code_dir not in sys.path:
                        sys.path.insert(0, _code_dir)
                    from sarhm.vis import save_hopfield_attention_bar
                    p = save_hopfield_attention_bar(
                        attn[0], None,
                        out_path=os.path.join(fig_dir, 'hopfield_attention_sample0.png'),
                        top_k=5,
                    )
                    if p:
                        print('attention plot:', p)
        except Exception as e:
            print('attention plot skip:', e)

    # ----- Sanity ladder: run first, then exit unless also_run_eval -----
    if getattr(args, 'sanity_checks', False):
        print("[SANITY] Running Stage C sanity ladder (sanity_items=%d, sanity_ddim_steps=%d) ..." % (
            getattr(args, 'sanity_items', 2), getattr(args, 'sanity_ddim_steps', 30)))
        run_sanity_ladder(generative_model, dataset_test, output_path, device, args)
        if not getattr(args, 'also_run_eval', False):
            print("[SANITY] Ladder done. Exiting (--also_run_eval not set).")
            sys.exit(0)

    pbar.set_description('Generate train')
    num_samples = getattr(args, 'num_samples', None) or getattr(config, 'num_samples', 5)
    split = getattr(args, 'split', 'test')
    start_index = getattr(args, 'start_index', 0)
    cfg_scale = getattr(args, 'cfg_scale', 1.0)
    cfg_uncond = getattr(args, 'cfg_uncond', 'zeros')
    limit_train = 10
    if getattr(args, 'max_train_items', None) is not None:
        limit_train = args.max_train_items
    elif getattr(args, 'max_items', None) is not None:
        limit_train = args.max_items
    grid, _ = generative_model.generate(dataset_train, num_samples,
                config.ddim_steps, config.HW, limit_train, cfg_scale=cfg_scale, cfg_uncond=cfg_uncond)
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(output_path, f'./samples_train.png'))
    pbar.update(1)

    pbar.set_description('Generate test')
    limit_test = None
    if getattr(args, 'max_test_items', 0) and args.max_test_items > 0:
        limit_test = None  # already sliced dataset_test in [MINI_EVAL]; generate full subset
    elif getattr(args, 'max_test_items', None) is not None and args.max_test_items != 0:
        limit_test = args.max_test_items
    elif getattr(args, 'max_items', None) is not None:
        limit_test = args.max_items
    # Optional: use only a slice of the dataset (start_index, start_index+limit_test)
    ds_test = dataset_test
    if start_index > 0 and hasattr(dataset_test, '__getitem__'):
        try:
            from torch.utils.data import Subset
            n = len(dataset_test)
            indices = list(range(start_index, min(start_index + (limit_test or n), n)))
            if indices:
                ds_test = Subset(dataset_test, indices)
                limit_test = None  # generate all of the subset
        except Exception:
            pass
    grid, samples = generative_model.generate(ds_test, num_samples,
                config.ddim_steps, config.HW, limit=limit_test, state=state, output_path=output_path,
                cfg_scale=cfg_scale, cfg_uncond=cfg_uncond)
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(output_path, f'./samples_test.png'))
    pbar.update(1)

    # E) Image quality metrics (pair-wise MSE, PCC, SSIM, PSM, top-1-class) for baseline vs SAR-HM comparison
    eval_metrics_row = None  # if set, block D will use it for ablation_results.csv
    if samples is not None and len(samples) > 0:
        try:
            first = samples[0]
            if hasattr(first, 'shape') and first.shape[0] >= 2:
                from eeg_ldm import get_eval_metric
                metric_vals, metric_names = get_eval_metric(samples, avg=True)
                if metric_vals and metric_names:
                    print("[EVAL] " + " ".join("%s=%.4f" % (k, v) for k, v in zip(metric_names, metric_vals)))
                    mdict = dict(zip(metric_names, metric_vals))
                    eval_metrics_row = dict(
                        ssim=mdict.get('ssim'), pcc=mdict.get('pcc'),
                        retrieval_acc=mdict.get('top-1-class'),
                    )
                    print("[EVAL] metrics will be written to ablation_results.csv")
        except Exception as e:
            print("[EVAL] metrics failed: %s" % e)

    pbar.set_description('Complete')
    pbar.update(1)
    pbar.close()

    # D) Ablation results table (professor-friendly artifact)
    try:
        _code_dir = os.path.dirname(os.path.abspath(__file__))
        if _code_dir not in sys.path:
            sys.path.insert(0, _code_dir)
        from sarhm.metrics_logger import append_ablation_results_row
        mode = getattr(config, 'ablation_mode', 'baseline') if getattr(config, 'use_sarhm', False) else 'baseline'
        kwargs = dict(run_dir=output_path, mode=mode)
        if eval_metrics_row:
            kwargs.update(eval_metrics_row)
        append_ablation_results_row(**kwargs)
        print("ablation_results.csv written under", output_path)
    except Exception as e:
        print("ablation_results skip:", e)
