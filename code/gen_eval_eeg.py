import os, sys
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
                        help='imagenet path.')

    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    root = args.root
    target = args.dataset

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

    output_path = os.path.join(config.root_path, 'results', 'eval',
                    '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    os.makedirs(output_path, exist_ok=True)

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
    pbar.update(1)

    pbar.set_description('Load model')
    generative_model = eLDM_eval(args.config_patch, num_voxels,
                device=device, pretrain_root=config.pretrain_gm_path, logger=getattr(config, 'logger', None),
                ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond,
                clip_tune=getattr(config, 'clip_tune', True), cls_tune=getattr(config, 'cls_tune', False),
                main_config=config)
    # ----- DEBUG: Checkpoint load integrity (A) -----
    try:
        missing, unexpected = generative_model.model.load_state_dict(sd['model_state_dict'], strict=False)
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
    except Exception as e:
        print("[DEBUG] [CKPT_LOAD] load_state_dict failed or no return: %s" % e)
        generative_model.model.load_state_dict(sd['model_state_dict'], strict=False)
    print('load ldm successfully')
    state = sd.get('state')
    pbar.update(1)

    # ----- DEBUG: scale_factor at inference (C) -----
    try:
        sf = getattr(generative_model.model, 'scale_factor', None)
        if sf is not None and hasattr(sf, 'item'):
            sf_val = float(sf.item())
        else:
            sf_val = float(sf) if sf is not None else None
        print("[DEBUG] [SCALE_FACTOR] value=%s expected=0.18215 match=%s" % (sf_val, (sf_val == 0.18215) if sf_val is not None else "MISSING"))
        if sf_val is None:
            print("[DEBUG] [SCALE_FACTOR] MISSING: generative_model.model.scale_factor not found")
    except Exception as e:
        print("[DEBUG] [SCALE_FACTOR] MISSING (exception): %s" % e)

    # ----- DEBUG: Conditioning sanity once per run (D) -----
    try:
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
        csm = getattr(generative_model.model, 'cond_stage_model', None)
        if csm is not None and getattr(csm, '_sarhm_extra', None):
            extra = csm._sarhm_extra
            c_base = extra.get('c_base')
            if c_base is not None:
                cb = c_base.float()
                print("[DEBUG] [COND] c_base mean=%.6f std=%.6f min=%.6f max=%.6f" % (cb.mean().item(), cb.std().item(), cb.min().item(), cb.max().item()))
            alpha = extra.get('alpha')
            if alpha is not None:
                ah = alpha.float()
                print("[DEBUG] [COND] alpha mean=%.6f min=%.6f max=%.6f" % (ah.mean().item(), ah.min().item(), ah.max().item()))
        else:
            print("[DEBUG] [COND] c_base/alpha not available (baseline or _sarhm_extra missing)")
    except Exception as e:
        print("[DEBUG] [COND] MISSING (exception): %s" % e)

    # ----- DEBUG: VAE round-trip sanity (E) -----
    try:
        model = generative_model.model.to(device)
        model.eval()
        sample_item = dataset_test[0]
        img = sample_item['image']
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
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
            rec = model.decode_first_stage(z)
        rec = torch.clamp((rec + 1.0) / 2.0, 0.0, 1.0)
        rec_np = (255. * rearrange(rec[0], 'c h w -> h w c').cpu().numpy()).astype(np.uint8)
        vae_path = os.path.join(output_path, 'vae_roundtrip.png')
        Image.fromarray(rec_np).save(vae_path)
        print("[DEBUG] [VAE_ROUNDTRIP] saved %s" % vae_path)
        rec_mean = rec.float().mean().item()
        rec_std = rec.float().std().item()
        if rec_std < 1e-5 or torch.isnan(rec).any() or torch.isinf(rec).any():
            print("[DEBUG] [VAE_ROUNDTRIP] WARNING: Reconstruction looks wrong (near-constant, NaN or Inf). VAE/scale_factor pipeline or weights may be incorrect.")
        else:
            print("[DEBUG] [VAE_ROUNDTRIP] recon mean=%.4f std=%.4f (sanity check)" % (rec_mean, rec_std))
    except Exception as e:
        print("[DEBUG] [VAE_ROUNDTRIP] MISSING (exception): %s" % e)
        print("[DEBUG] [VAE_ROUNDTRIP] WARNING: Could not run VAE round-trip. If reconstructions are noise, VAE/scale_factor pipeline is wrong or weights not loaded.")

    # C) Professor-friendly attention plot when SAR-HM is on
    if getattr(config, 'use_sarhm', False):
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

    pbar.set_description('Generate train')
    grid, _ = generative_model.generate(dataset_train, config.num_samples,
                config.ddim_steps, config.HW, 10)
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(output_path, f'./samples_train.png'))
    pbar.update(1)

    pbar.set_description('Generate test')
    grid, samples = generative_model.generate(dataset_test, config.num_samples,
                config.ddim_steps, config.HW, limit=None, state=state, output_path=output_path)
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(output_path, f'./samples_test.png'))
    pbar.update(1)

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
        append_ablation_results_row(run_dir=output_path, mode=mode)
        print("ablation_results.csv written under", output_path)
    except Exception as e:
        print("ablation_results skip:", e)
