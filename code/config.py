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
        self.imagenet_path = None  # Must set via --imagenet_path or IMAGENET_PATH for EEG
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
        self.val_gen_limit = 2
        self.val_ddim_steps = 50
        self.val_num_samples = 2
        # limit test set generation after training (e.g. 10 = only 10 images; None = all)
        self.test_gen_limit = 10
        # validate every N epochs
        self.check_val_every_n_epoch = 5
        # Stage B correct-by-default: generate validation images when not disabled
        self.disable_image_generation_in_val = False
        # When image gen is enabled, generate every N epochs (1 = every val run)
        self.val_image_gen_every_n_epoch = 1
        # skip training for epochs < start_epoch (e.g. --start_epoch 2 to test epoch 2 only)
        self.start_epoch = 0
        # Debug toggles (minimal high-signal): conditioning stats, VAE round-trip, sampling intermediates
        self.debug_cond_stats = False
        self.debug_vae_roundtrip = False
        self.debug_sampling_steps = None  # e.g. [250,200,150,100,50] to save intermediates for first val sample
        # DataLoader: 0 = Windows-safe; 4 on Linux to avoid "does not have many workers" warning
        self.num_workers = 4 if os.name != 'nt' else 0
        # resume check util (resume_ckpt_path = Stage-B LDM; checkpoint_path deprecated)
        self.model_meta = None
        self.checkpoint_path = None
        self.resume_ckpt_path = None
        self.prune_unexpected_keys = False
        # torch.compile (PyTorch 2+) for faster training; enable with --use_compile
        self.use_compile = False

        # Mini experiment: limit dataset size for fast sanity runs (0 = no limit)
        self.limit_train_items = 0
        self.limit_val_items = 0
        self.skip_post_train_generation = True  # skip end-of-training image gen unless explicitly false

        # Thesis logging / evaluation (optional; set via CLI --run_name, --model, --seed, --eval_every, --num_eval_samples)
        self.run_name = None
        self.model = None  # 'baseline' | 'sarhm'
        self.eval_every = 2
        self.num_eval_samples = 50

        # ---------- Run mode (overrides use_sarhm/use_sarhmpp when set) ----------
        self.run_mode = None  # None | 'baseline' | 'sarhm' | 'sarhmpp'

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
        # Alpha schedule (training): linear warmup from alpha_max_start to alpha_max_end over warmup_epochs
        self.warmup_epochs = 10
        self.alpha_max_start = 0.05
        self.alpha_max_end = 0.2
        # Conditioning normalization before fusion (match c_base / c_sar scales)
        self.normalize_conditioning = True
        self.normalization_type = 'layernorm'  # 'layernorm' | 'l2'
        # debug_cond_stats moved to top-level debug toggles above
        self.warm_start_from_baseline = True
        # Prototypes: stable memory first
        self.proto_source = 'baseline_centroids'  # 'baseline_centroids' | 'clip_text' | 'learnable'
        self.proto_freeze_epochs = 5
        # Progressive training (optional)
        self.freeze_diffusion_until_epoch = 0  # 0 = no freeze; >0 freeze UNet/VAE until this epoch
        self.train_memory_only_until_epoch = 0  # 0 = off; >0 train only conditioner/memory until this epoch
        self.training_phase = 1  # 0=baseline sanity, 1=proj only, 2=hopfield, 3=proto EMA
        self.enable_hopfield = True
        self.enable_gate = True
        # Prototype diversity regularization (optional)
        self.proto_diversity_weight = 0.0  # 1e-3 recommended if learnable prototypes
        # Extra losses (optional)
        self.lambda_stable = 0.1
        self.stable_loss_epochs = 5
        self.lambda_retrieval = 0.1

        # ---------- SAR-HM improved (optional; backward compatible) ----------
        # A. Residual fusion: final_cond = baseline_cond + alpha * (sarhm_cond - baseline_cond)
        self.use_residual_fusion = True  # False = original (legacy) fusion behavior for ablation
        self.sarhm_fusion_mode = 'residual'  # 'residual' | 'original' (original = replace with c_sar when not residual)
        self.sarhm_alpha = 0.2  # Optional fixed alpha when use_residual_fusion; None = use confidence-based alpha
        self.sarhm_residual_clamp = None  # Optional clamp on residual (e.g. 2.0); None = no clamp
        self.residual_scale = 1.0  # Scale factor on (c_sar - c_base) before adding
        self.residual_clamp_value = None  # Clamp residual magnitude; None = no clamp
        # B. Confidence-gated fusion: alpha * g * (sarhm_cond - baseline_cond)
        self.use_confidence_gate = True  # When True, alpha is modulated by retrieval confidence
        self.confidence_source = 'entropy'  # 'max' | 'entropy' | 'margin' (top1 - top2 similarity)
        self.confidence_min = 0.0
        self.confidence_max = 1.0
        # C. Warm-start / safe training (warm_start_from_baseline above inits adapter near zero)
        self.freeze_sarhm_until_epoch = 0  # Freeze SAR-HM params (projection, hopfield, adapter) until this epoch; 0 = never
        self.progressive_unfreeze = False  # Unfreeze SAR-HM progressively (e.g. projection first, then hopfield)
        # D. Epoch stats and best checkpoint
        self.save_epoch_stats = True  # Save train/val stats to CSV every epoch
        self.save_every_epoch_ckpt = False  # Save checkpoint every epoch (epoch_XXX.ckpt)
        self.save_best_by_clip = True  # Save best_clip.ckpt when val CLIP is computed
        self.save_best_by_balanced_score = False  # Save best_balanced.ckpt
        self.balanced_score_weights = (0.5, 0.25, 0.25)  # (w_clip, w_ssim, w_pcc) for balanced score
        # E. Early stopping
        self.early_stopping_monitor = None  # None | 'val_loss' | 'clip_sim_mean' | 'balanced_score'
        self.early_stopping_patience = 5
        self.early_stopping_min_delta = 1e-4
        # F. Reduce semantic drift (alpha schedule, freeze SAR-HM later)
        self.alpha_schedule_type = 'warmup'  # 'warmup' | 'constant' | 'decay_after'
        self.alpha_start = 0.05
        self.alpha_end = 0.2
        self.alpha_decay_start_epoch = None  # When set, decay alpha after this epoch (e.g. 0.8 * alpha_end)
        self.freeze_sarhm_after_epoch = None  # Freeze SAR-HM after this epoch; None = never
        self.sarhm_lr_multiplier = 1.0  # LR multiplier for SAR-HM params only (e.g. 0.5 for gentler updates)
        # G. Prototype / memory quality (SAR-HM only; no semantic classes)
        self.normalize_memory_embeddings = True  # L2-normalize prototype vectors when loading
        self.top_k_retrieval = None  # None = use all K; int = use only top-k by similarity
        self.min_retrieval_confidence = 0.0  # Downweight or zero alpha when confidence below this
        self.use_similarity_margin = False  # Use (top1 - top2) as confidence proxy when confidence_source=='margin'
        self.retrieval_temperature = 1.0  # Same as hopfield_tau; softmax temperature for attention
        # H. Limit retrieval strength
        self.attention_temperature = 1.0  # Softmax temperature in Hopfield (same as hopfield_tau if not set)
        # I. Loss weights (modular; for high-CLIP-priority set lambda_clip_* higher)
        self.lambda_diffusion = 1.0  # Main diffusion noise prediction loss
        self.lambda_recon = 0.0  # Optional reconstruction term (if any)
        self.lambda_clip = 0.2  # CLIP loss (EEG->cond vs image embed); increase for higher CLIP priority
        self.lambda_pcc = 0.0
        self.lambda_ssim = 0.0
        # J/K. Inference-time controls (used by eval scripts when loading config)
        self.eval_ddim_steps = None  # None = use ddim_steps from config
        self.eval_guidance_scale = 1.0
        self.eval_retrieval_top_k = None
        self.eval_fusion_alpha = None
        self.eval_confidence_gate = True
        # L. Reproducibility
        self.save_full_config = True  # Save full config with checkpoint and in run dir
        self.save_git_hash = True  # Save git commit hash in run dir when available

        # ---------- SAR-HM++ (Multi-Level Semantic Prototype Retrieval) ----------
        self.use_sarhmpp = False
        self.semantic_targets_path = None
        self.semantic_prototypes_path = None
        self.semantic_memory_mode = 'per_sample'
        self.semantic_fusion_mode = 'concat_project'
        self.semantic_target_fusion_mode = 'concat_project'
        self.semantic_query_pooling = 'mean'
        self.semantic_query_dim = 768
        self.semantic_topk = 5
        self.semantic_temperature = 1.0
        self.semantic_adapter_mode = 'mlp_tokens_plus_transformer'
        self.semantic_transformer_layers = 1
        self.conf_w1 = 0.5
        self.conf_w2 = 0.3
        self.conf_w3 = 0.2
        self.lambda_clip_img = 0.2
        self.lambda_clip_text = 0.1
        self.lambda_sem = 0.1
        self.lambda_retr = 0.1
        self.lambda_ssim = 0.05
        self.lambda_obj = 0.0
        self.clip_loss_every_n_steps = 1
        self.use_region_semantics = False
        self.use_summary_semantics = True
        self.use_scene_semantics = True
        self.use_object_semantics = True
        self.training_stage_mode = 'joint'
        self.freeze_prototypes = True
        self.prototype_update_after_epoch = 0
        self.enable_semantic_logging = True
        # Ablation flags (SAR-HM++ and build scripts): when True, disable that component
        self.no_summary = False
        self.no_region = False
        self.no_scene = False
        self.no_object = False
        self.no_clip_loss = False
        self.no_retrieval_loss = False
        self.no_confidence_gate = False
        self.sarhmpp_projection_only = False  # SAR-HM++: use q_sem only (adapter from query), no retrieval


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