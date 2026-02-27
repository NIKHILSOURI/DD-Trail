import os
import numpy as np
from pathlib import Path

# Get repository root directory (parent of 'code' directory)
# This works whether script is run from repo root or code directory
_CODE_DIR = Path(__file__).parent.absolute()
_REPO_ROOT = _CODE_DIR.parent.absolute()
# Optional env vars for custom data locations (no hardcoded drive letters)
_DATA_ROOT = Path(os.environ.get("DREAMDIFFUSION_DATA_ROOT", str(_REPO_ROOT / "datasets")))
_PRETRAIN_ROOT = Path(os.environ.get("DREAMDIFFUSION_PRETRAIN_ROOT", str(_REPO_ROOT / "pretrains")))

class Config_MAE_fMRI: # back compatibility
    pass
class Config_MBM_finetune: # back compatibility
    pass 

class Config_MBM_EEG(Config_MAE_fMRI):
    # configs for fmri_pretrain.py
    def __init__(self):
    # --------------------------------------------
    # MAE for fMRI
        # Training Parameters
        self.lr = 2.5e-4
        self.min_lr = 0.
        self.weight_decay = 0.05
        self.num_epoch = 500
        self.warmup_epochs = 40
        self.batch_size = 100
        self.clip_grad = 0.8
        
        # Model Parameters
        self.mask_ratio = 0.1
        self.patch_size = 4 #  1
        self.embed_dim = 1024 #256 # has to be a multiple of num_heads
        self.decoder_embed_dim = 512 #128
        self.depth = 24
        self.num_heads = 16
        self.decoder_num_heads = 16
        self.mlp_ratio = 1.0

        # Project setting - use repo root by default, Windows-safe
        self.root_path = str(_REPO_ROOT)
        self.output_path = os.path.join(self.root_path, 'exps')
        self.seed = 2022
        self.roi = 'VC'
        self.aug_times = 1
        self.num_sub_limit = None
        self.include_hcp = True
        self.include_kam = True
        self.accum_iter = 1

        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6

        # distributed training
        self.local_rank = 0


class Config_EEG_finetune(Config_MBM_finetune):
    def __init__(self):
        
        # Project setting - use repo root by default, Windows-safe
        self.root_path = str(_REPO_ROOT)
        self.output_path = os.path.join(self.root_path, 'exps')

        self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_5_95_std.pth')
        self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')

        self.dataset = 'EEG' 
        self.pretrain_mbm_path = os.path.join(self.root_path, 'pretrains', 'eeg_pretrain', 'checkpoint.pth') 

        self.include_nonavg_test = True


        # Training Parameters
        self.lr = 5.3e-5
        self.weight_decay = 0.05
        self.num_epoch = 15
        self.batch_size = 16 if self.dataset == 'GOD' else 4 
        self.mask_ratio = 0.5
        self.accum_iter = 1
        self.clip_grad = 0.8
        self.warmup_epochs = 2
        self.min_lr = 0.
        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6

        # distributed training
        self.local_rank = 0
        
class Config_Generative_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        # Use repo root by default, Windows-safe
        self.root_path = str(_REPO_ROOT)
        self.output_path = os.path.join(self.root_path, 'exps')

        self.eeg_signals_path = str(_DATA_ROOT / 'eeg_5_95_std.pth')
        self.splits_path = str(_DATA_ROOT / 'block_splits_by_image_single.pth')
        # self.splits_path = str(_DATA_ROOT / 'block_splits_by_image_all.pth')
        self.roi = 'VC'
        self.patch_size = 4 # 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 1.0

        self.pretrain_gm_path = str(_PRETRAIN_ROOT)
        
        self.dataset = 'EEG' 
        self.pretrain_mbm_path = str(_PRETRAIN_ROOT / 'eeg_pretain' / 'checkpoint.pth')

        self.img_size = 512

        np.random.seed(self.seed)
        # finetune parameters (defaults from original config; for 16 GB GPU use --batch_size 4)
        # Default 4 avoids OOM in attention (64x64 latent => 4096 tokens; batch 25 needs ~50 GiB for full attn)
        self.batch_size = 5 if self.dataset == 'GOD' else 4
        self.lr = 5.3e-5
        self.num_epoch = 500
        
        # 16 = mixed precision (faster, less VRAM); 32 = full; "bf16" on A100 for best speed
        self.precision = 16
        self.accumulate_grad = 1
        self.crop_ratio = 0.2
        self.global_pool = False
        self.use_time_cond = True
        self.clip_tune = True #False
        self.cls_tune = False
        self.subject = 4
        self.eval_avg = True

        # diffusion sampling parameters (used for Stage C / final eval; thesis metrics use these)
        self.num_samples = 5
        self.ddim_steps = 250
        self.HW = None
        # validation during Stage B (only affects training-time val; Stage C uses num_samples/ddim_steps above)
        self.val_gen_limit = 1
        self.val_ddim_steps = 25
        self.val_num_samples = 1
        # limit test set generation after training (e.g. 10 = only 10 images; None = all)
        self.test_gen_limit = 10
        # validate every N epochs (10 = fewer val runs for faster Stage B)
        self.check_val_every_n_epoch = 10
        # skip training for epochs < start_epoch (e.g. --start_epoch 2 to test epoch 2 only)
        self.start_epoch = 0
        # DataLoader: 0 = main process only (Windows-safe); 4–8 on Linux/RunPod for faster GPU feed
        self.num_workers = 0
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None
        # torch.compile (PyTorch 2+) for faster training; enable with --use_compile
        self.use_compile = False

        # Thesis logging / evaluation (optional; set via CLI --run_name, --model, --seed, --eval_every, --num_eval_samples)
        self.run_name = None
        self.model = None  # 'baseline' | 'sarhm'
        self.eval_every = 2
        self.num_eval_samples = 50

        # ---------- SAR-HM (Semantic Associative Retrieval with Hopfield Memory) ----------
        self.use_sarhm = True
        self.proto_mode = 'class'
        self.proto_path = None  # Set at runtime to output_path/prototypes.pt when use_sarhm
        self.hopfield_tau = 1.0
        self.gate_mode = 'max'
        self.ablation_mode = 'full_sarhm'  # 'baseline' | 'projection_only' | 'hopfield_no_gate' | 'full_sarhm'
        self.num_classes = 40
        # Safe fusion: c_final = c_base + alpha * (c_sar - c_base)
        self.alpha_mode = 'entropy'  # 'entropy' | 'max' | 'constant'
        self.alpha_max = 0.2
        self.conf_threshold = 0.2  # if conf < this -> alpha=0 (pure baseline)
        self.alpha_constant = 0.1  # fallback when alpha_mode=='constant'
        # Warm start: init SAR adapter so c_sar ~ c_base at epoch 0
        self.warm_start_from_baseline = True
        # Prototypes: stable memory first
        self.proto_source = 'baseline_centroids'  # 'baseline_centroids' | 'clip_text' | 'learnable'
        self.proto_freeze_epochs = 5
        # Progressive training
        self.training_phase = 1  # 0=baseline sanity, 1=proj only, 2=hopfield, 3=proto EMA
        self.enable_hopfield = True
        self.enable_gate = True
        # Extra losses (optional)
        self.lambda_stable = 0.1
        self.stable_loss_epochs = 5
        self.lambda_retrieval = 0.1



class Config_Cls_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        # Use repo root by default, Windows-safe
        self.root_path = str(_REPO_ROOT)
        self.output_path = os.path.join(self.root_path, 'exps')

        # self.eeg_signals_path = os.path.join(self.root_path, 'datasets', 'eeg_5_95_std.pth')
        self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_14_70_std.pth')
        # self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_single.pth')
        self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')
        self.roi = 'VC'
        self.patch_size = 4 # 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 1.0

        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains')
        
        self.dataset = 'EEG' 
        self.pretrain_mbm_path = None

        self.img_size = 512

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 5 if self.dataset == 'GOD' else 25
        self.lr = 5.3e-5
        self.num_epoch = 50
        
        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.15
        self.global_pool = False
        self.use_time_cond = False
        self.clip_tune = False
        self.subject = 4
        self.eval_avg = True

        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None 