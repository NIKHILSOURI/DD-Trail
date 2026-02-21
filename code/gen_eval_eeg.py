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
    generative_model.model.load_state_dict(sd['model_state_dict'], strict=False)
    print('load ldm successfully')
    state = sd['state']
    pbar.update(1)

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
