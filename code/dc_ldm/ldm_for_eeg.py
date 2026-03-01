import numpy as np
import wandb
import torch
from dc_ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
import os
from dc_ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sc_mbm.mae_for_eeg import eeg_encoder, classify_network, mapping
from PIL import Image
from tqdm import tqdm

# Optional SAR-HM (Semantic Associative Retrieval with Hopfield Memory)
try:
    from sarhm.sarhm_modules import (
        pool_eeg_tokens,
        SemanticProjection,
        HopfieldRetrieval,
        ConfidenceGatedFusion,
        ConditioningAdapter,
        compute_alpha_from_attention,
    )
    from sarhm.prototypes import ClassPrototypes
    _SARHM_AVAILABLE = True
except ImportError as e:
    _SARHM_AVAILABLE = False
    compute_alpha_from_attention = None
    _SARHM_IMPORT_ERROR = str(e)


def create_model_from_config(config, num_voxels, global_pool):
    model = eeg_encoder(time_len=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, global_pool=global_pool) 
    return model

def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0

class cond_stage_model(nn.Module):
    def __init__(self, metafile, num_voxels=440, cond_dim=1280, global_pool=True, clip_tune=True, cls_tune=False,
                 use_sarhm=False, ablation_mode='baseline', num_classes=40, hopfield_tau=1.0, gate_mode='max',
                 proto_path=None, proto_mode='class',
                 alpha_mode='entropy', alpha_max=0.2, conf_threshold=0.2, alpha_constant=0.1,
                 warm_start_from_baseline=True):
        super().__init__()
        self.use_sarhm = use_sarhm and _SARHM_AVAILABLE
        self.ablation_mode = ablation_mode if self.use_sarhm else 'baseline'
        if use_sarhm and not _SARHM_AVAILABLE:
            import sys
            err = getattr(sys.modules[__name__], '_SARHM_IMPORT_ERROR', 'unknown')
            print("[cond_stage_model] WARNING: use_sarhm=True but 'sarhm' failed to import; using baseline. Error: %s" % err)
        self._sarhm_extra = {}  # optional diagnostics (attention, confidence, alpha, entropy)
        self._sarhm_header_printed = False
        self._baseline_header_printed = False
        self.alpha_mode = alpha_mode
        self.alpha_max = alpha_max
        self.conf_threshold = conf_threshold
        self.alpha_constant = alpha_constant
        self.warm_start_from_baseline = warm_start_from_baseline

        # prepare pretrained fmri mae
        if metafile is not None:
            # Handle different checkpoint structures
            if 'config' in metafile:
                config = metafile['config']
            else:
                raise KeyError(f"Checkpoint missing 'config' key. Available keys: {list(metafile.keys())}")
            
            # Try different possible keys for the model state dict
            if 'model' in metafile:
                model_state_dict = metafile['model']
            elif 'model_state_dict' in metafile:
                model_state_dict = metafile['model_state_dict']
            elif 'state_dict' in metafile:
                model_state_dict = metafile['state_dict']
            else:
                raise KeyError(f"Checkpoint missing 'model', 'model_state_dict', or 'state_dict' key. Available keys: {list(metafile.keys())}")
            
            model = create_model_from_config(config, num_voxels, global_pool)
            model.load_checkpoint(model_state_dict)
        else:
            model = eeg_encoder(time_len=num_voxels, global_pool=global_pool)
        self.mae = model
        if clip_tune:
            self.mapping = mapping()
        if cls_tune:
            self.cls_net = classify_network()

        self.fmri_seq_len = model.num_patches
        self.fmri_latent_dim = model.embed_dim
        if global_pool == False:
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True),
                nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True)
            )
        self.dim_mapper = nn.Linear(self.fmri_latent_dim, cond_dim, bias=True)
        self.global_pool = global_pool

        if self.use_sarhm:
            clip_dim = 768
            self.sarhm_projection = SemanticProjection(self.fmri_latent_dim, clip_dim=clip_dim)
            self.sarhm_prototypes = ClassPrototypes(num_classes=num_classes, dim=clip_dim, proto_path=proto_path)
            self.sarhm_hopfield = HopfieldRetrieval(tau=hopfield_tau)
            self.sarhm_fusion = ConfidenceGatedFusion(gate_mode=gate_mode)
            self.sarhm_adapter = ConditioningAdapter(clip_dim=clip_dim, cond_dim=cond_dim, seq_len=77)
            if self.warm_start_from_baseline:
                self.sarhm_adapter.init_near_zero_delta(scale=0.01)
            self._proto_source = "dummy"
            if proto_path:
                if self.sarhm_prototypes.load_from_path(proto_path):
                    self._proto_source = "loaded"
            # else: train-built prototypes use "train" when set by training script

        # self.image_embedder = FrozenImageEmbedder()

    # def forward(self, x):
    #     # n, c, w = x.shape
    #     latent_crossattn = self.mae(x)
    #     if self.global_pool == False:
    #         latent_crossattn = self.channel_mapper(latent_crossattn)
    #     latent_crossattn = self.dim_mapper(latent_crossattn)
    #     out = latent_crossattn
    #     return out

    def _baseline_conditioning(self, latent_crossattn):
        """Always compute baseline conditioning [B, L, cond_dim]. L may be 77 or num_patches."""
        lat = latent_crossattn
        if self.global_pool == False:
            lat = self.channel_mapper(lat)
        return self.dim_mapper(lat)

    def _align_to_seq_len(self, c, target_len=77):
        """Ensure c is [B, target_len, D]. Repeat or interpolate if needed."""
        if c.dim() == 2:
            # [B, D] -> [B, 1, D] then expand to [B, target_len, D]
            c = c.unsqueeze(1).expand(-1, target_len, -1)
            return c
        if c.dim() != 3:
            return c
        B, L, D = c.shape
        if L == target_len:
            return c
        if L == 1:
            return c.expand(B, target_len, D)
        # Interpolate along sequence dimension
        c = c.transpose(1, 2)  # [B, D, L]
        c = F.interpolate(c, size=target_len, mode='linear', align_corners=False)
        return c.transpose(1, 2)  # [B, target_len, D]

    def forward(self, x):
        latent_crossattn = self.mae(x)
        latent_return = latent_crossattn

        # A1: Always compute baseline conditioning c_base [B, L, cond_dim]
        c_base = self._baseline_conditioning(latent_crossattn)
        target_seq = 77
        c_base = self._align_to_seq_len(c_base, target_seq)

        if self.use_sarhm:
            # === SAR-HM path: c_sar then residual fusion c_final = c_base + alpha * (c_sar - c_base) ===
            if not self._sarhm_header_printed:
                self._sarhm_header_printed = True
                K = self.sarhm_prototypes.P.shape[0]
                tau = getattr(self.sarhm_hopfield, "tau", 1.0)
                gate = getattr(self.sarhm_fusion, "gate_mode", "max")
                proto_src = getattr(self, "_proto_source", "dummy")
                print("SAR-HM ACTIVE | mode=%s | proto_source=%s | K=%s | tau=%s | alpha_mode=%s | alpha_max=%s | conf_threshold=%s" % (
                    self.ablation_mode, proto_src, K, tau, self.alpha_mode, self.alpha_max, self.conf_threshold))
            self._last_mae_latent = latent_crossattn
            pooled = pool_eeg_tokens(latent_crossattn, self.global_pool)
            z_orig = self.sarhm_projection(pooled)
            logits = None
            if self.ablation_mode == 'projection_only':
                z_fused = z_orig
                attn = None
                confidence = torch.zeros(z_orig.shape[0], device=z_orig.device, dtype=z_orig.dtype)
            else:
                z_ret, attn, logits = self.sarhm_hopfield(z_orig, self.sarhm_prototypes)
                if self.ablation_mode == 'hopfield_no_gate':
                    z_fused = z_ret
                    confidence = torch.ones(z_orig.shape[0], device=z_orig.device, dtype=z_orig.dtype)
                else:
                    z_fused, confidence = self.sarhm_fusion(z_orig, z_ret, attn)
            c_sar = self.sarhm_adapter(z_fused)  # [B, 77, cond_dim]

            # A2–A3: Residual fusion; alpha from Hopfield confidence (clamped, fallback)
            if compute_alpha_from_attention is not None and attn is not None:
                alpha, conf_out, entropy = compute_alpha_from_attention(
                    attn, alpha_mode=self.alpha_mode, alpha_max=self.alpha_max,
                    conf_threshold=self.conf_threshold, alpha_constant=self.alpha_constant)
            elif attn is None:
                alpha = torch.full((c_sar.shape[0],), self.alpha_constant, device=c_sar.device, dtype=c_sar.dtype)
                conf_out = alpha.clone()
                entropy = torch.zeros(c_sar.shape[0], device=c_sar.device, dtype=c_sar.dtype)
            else:
                alpha = torch.full((c_sar.shape[0],), self.alpha_constant, device=c_sar.device, dtype=c_sar.dtype)
                conf_out = confidence
                entropy = torch.zeros(c_sar.shape[0], device=c_sar.device, dtype=c_sar.dtype)

            # alpha per sample: [B] -> [B, 1, 1] for broadcasting
            alpha_bc = alpha.view(-1, 1, 1).to(c_base.dtype)
            c_final = c_base + alpha_bc * (c_sar - c_base)

            self._sarhm_extra = {
                'confidence': conf_out,
                'attn': attn,
                'alpha': alpha,
                'entropy': entropy if attn is not None else torch.zeros_like(conf_out),
                'logits': logits,
                'c_base': c_base,
                'c_sar': c_sar,
            }
            return c_final, z_fused

        if not self._baseline_header_printed:
            self._baseline_header_printed = True
            print("SAR-HM: OFF | baseline conditioning path")
        return c_base, latent_return

    # def recon(self, x):
    #     recon = self.decoder(x)
    #     return recon

    def get_cls(self, x):
        if getattr(self, 'use_sarhm', False) and getattr(self, '_last_mae_latent', None) is not None:
            x = self._last_mae_latent
        if not hasattr(self, 'cls_net') or self.cls_net is None:
            raise RuntimeError("get_cls called but cls_net not built (cls_tune=False)")
        return self.cls_net(x)

    def get_clip_loss(self, x, image_embeds):
        if image_embeds is None or x is None:
            dev = getattr(x, 'device', None) or getattr(image_embeds, 'device', torch.device('cpu'))
            return torch.tensor(0.0, device=dev, dtype=torch.float32)
        if getattr(self, 'use_sarhm', False) and x.shape[-1] == image_embeds.shape[-1]:
            loss = 1 - torch.cosine_similarity(x, image_embeds, dim=-1).mean()
            return loss
        if not hasattr(self, 'mapping') or self.mapping is None:
            return torch.tensor(0.0, device=x.device, dtype=torch.float32)
        target_emb = self.mapping(x)
        loss = 1 - torch.cosine_similarity(target_emb, image_embeds, dim=-1).mean()
        return loss
    


class eLDM:

    def __init__(self, metafile, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=False, clip_tune=True, cls_tune=False,
                 main_config=None):
        # self.ckp_path = os.path.join(pretrain_root, 'model.ckpt')
        self.ckp_path = os.path.join(pretrain_root, 'models/v1-5-pruned.ckpt')
        self.config_path = os.path.join(pretrain_root, 'models/config15.yaml')
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim

        model = instantiate_from_config(config.model)
        pl_sd = torch.load(self.ckp_path, map_location="cpu", weights_only=False)['state_dict']

        # Remap or drop raw MAE keys (no "cond_stage_model.mae." prefix) to avoid unexpected-key noise
        def is_raw_mae_key(k):
            if k in ('cls_token', 'pos_embed', 'mask_token') or k in ('norm.weight', 'norm.bias'):
                return True
            if k.startswith('patch_embed.') or k.startswith('blocks.'):
                return True
            return False
        raw_mae_sd = {k: pl_sd.pop(k) for k in list(pl_sd.keys()) if is_raw_mae_key(k)}

        m, u = model.load_state_dict(pl_sd, strict=False)
        if m or u:
            print(f"LDM checkpoint load: {len(m)} missing, {len(u)} unexpected keys (cond_stage_model replaced below)")
        model.cond_stage_trainable = True
        sarhm_kw = {}
        if main_config is not None:
            sarhm_kw = {
                'use_sarhm': getattr(main_config, 'use_sarhm', False),
                'ablation_mode': getattr(main_config, 'ablation_mode', 'baseline'),
                'num_classes': getattr(main_config, 'num_classes', 40),
                'hopfield_tau': getattr(main_config, 'hopfield_tau', 1.0),
                'gate_mode': getattr(main_config, 'gate_mode', 'max'),
                'proto_path': getattr(main_config, 'proto_path', None),
                'proto_mode': getattr(main_config, 'proto_mode', 'class'),
                'alpha_mode': getattr(main_config, 'alpha_mode', 'entropy'),
                'alpha_max': getattr(main_config, 'alpha_max', 0.2),
                'conf_threshold': getattr(main_config, 'conf_threshold', 0.2),
                'alpha_constant': getattr(main_config, 'alpha_constant', 0.1),
                'warm_start_from_baseline': getattr(main_config, 'warm_start_from_baseline', True),
            }
        model.cond_stage_model = cond_stage_model(
            metafile, num_voxels, self.cond_dim, global_pool=global_pool, clip_tune=clip_tune, cls_tune=cls_tune,
            **sarhm_kw
        )
        # If checkpoint had raw MAE keys, load them into the conditioner's MAE
        if raw_mae_sd:
            m_mae, u_mae = model.cond_stage_model.mae.load_state_dict(raw_mae_sd, strict=False)
            if m_mae or u_mae:
                print(f"Cond MAE load from ckpt: {len(m_mae)} missing, {len(u_mae)} unexpected")
        if sarhm_kw.get('use_sarhm'):
            try:
                from sarhm.metrics_logger import sarhm_metrics_from_extra
                model._sarhm_metrics_hook = lambda csm, lbl: sarhm_metrics_from_extra(
                    getattr(csm, '_sarhm_extra', {}), lbl
                )
            except Exception:
                model._sarhm_metrics_hook = None
        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        
        self.device = device    
        self.model = model
        
        self.model.clip_tune = clip_tune
        self.model.cls_tune = cls_tune

        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.cond_stage_model.fmri_latent_dim
        self.metafile = metafile

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one
      
        # # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        num_workers = getattr(config, 'num_workers', 0) if config else 0
        use_cuda = next(self.model.parameters()).is_cuda
        dataloader = DataLoader(
            dataset, batch_size=bs1, shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=(num_workers > 0),
        )
        test_loader = DataLoader(
            test_dataset, batch_size=bs1, shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=(num_workers > 0),
        )
        self.model.unfreeze_whole_model()
        self.model.freeze_first_stage()
        self.model.freeze_diffusion_model()
        # E: Assert SD (VAE + UNet) is frozen
        for name, p in self.model.named_parameters():
            if "first_stage" in name or (name.startswith("model.") and "cond_stage" not in name):
                assert not p.requires_grad, "SD must be frozen: %s has requires_grad=True" % name
        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        print("trainable_modules (%d): %s" % (len(trainable), trainable[:20] if len(trainable) > 20 else trainable))
        if getattr(config, 'use_sarhm', False):
            proto_freeze = getattr(config, 'proto_freeze_epochs', 5)
            csm = getattr(self.model, 'cond_stage_model', None)
            if csm is not None and getattr(csm, 'sarhm_prototypes', None):
                csm.sarhm_prototypes.prototypes.requires_grad = False
                if proto_freeze > 0:
                    self.model._proto_freeze_epochs = proto_freeze
                    self.model._proto_unfrozen = False

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg
        # torch.compile disabled: validation_step runs generate()/PLMS which can break under Dynamo
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)

        self.model.unfreeze_whole_model()

        from dc_ldm.util import pickle_safe_config
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': pickle_safe_config(config),
                'state': torch.random.get_rng_state()

            },
            os.path.join(output_path, 'checkpoint.pth')
        )
        # Save SAR-HM prototypes for reproducibility (checkpoint already contains them)
        if getattr(config, 'use_sarhm', False):
            csm = getattr(self.model, 'cond_stage_model', None)
            if csm is not None and getattr(csm, 'sarhm_prototypes', None):
                proto_path = getattr(config, 'proto_path', None)
                if proto_path is None:
                    proto_path = os.path.join(output_path, 'prototypes.pt')
                os.makedirs(os.path.dirname(proto_path) or '.', exist_ok=True)
                csm.sarhm_prototypes.save_to_path(proto_path)
                print(f"SAR-HM prototypes saved to {proto_path}")

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path = None):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            try:
                torch.cuda.set_rng_state(state)
            except (RuntimeError, TypeError) as e:
                if "wrong size" in str(e).lower() or "state" in str(e).lower():
                    pass  # RNG state incompatible (e.g. different GPU/PyTorch); skip
                else:
                    raise
            
        with model.ema_scope():
            model.eval()
            total_items = len(fmri_embedding) if limit is None else min(limit, len(fmri_embedding))
            pbar = tqdm(enumerate(fmri_embedding), total=total_items, desc='Generate',
                        unit='item', bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            for count, item in pbar:
                if limit is not None:
                    if count >= limit:
                        break
                latent = item['eeg']
                if not isinstance(latent, torch.Tensor):
                    latent = torch.as_tensor(latent, dtype=torch.float32, device=self.device)
                else:
                    latent = latent.to(self.device)
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w')  # h w c
                if isinstance(gt_image, np.ndarray):
                    gt_image = torch.from_numpy(gt_image).float().to(self.device)
                pbar.set_postfix_str(f'item {count+1}/{total_items}, {ddim_steps} steps')
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'
                latent_rep = repeat(latent, 'h w -> c h w', c=num_samples)
                c, re_latent = model.get_learned_conditioning(latent_rep)
                # c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                if output_path is not None:
                    samples_t = (255. * torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(np.uint8)
                    for copy_idx, img_t in enumerate(samples_t):
                        img_t = rearrange(img_t, 'c h w -> h w c')
                        Image.fromarray(img_t).save(os.path.join(output_path, 
                            f'./test{count}-{copy_idx}.png'))
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to(self.device)  # keep on device for next training step (was .to('cpu') and caused epoch 2 crash)
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)




class eLDM_eval:

    def __init__(self, config_path, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=False, clip_tune=True, cls_tune=False,
                 main_config=None):
        self.config_path = config_path
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim

        model = instantiate_from_config(config.model)

        model.cond_stage_trainable = True
        sarhm_kw = {}
        if main_config is not None:
            sarhm_kw = {
                'use_sarhm': getattr(main_config, 'use_sarhm', False),
                'ablation_mode': getattr(main_config, 'ablation_mode', 'baseline'),
                'num_classes': getattr(main_config, 'num_classes', 40),
                'hopfield_tau': getattr(main_config, 'hopfield_tau', 1.0),
                'gate_mode': getattr(main_config, 'gate_mode', 'max'),
                'proto_path': getattr(main_config, 'proto_path', None),
                'proto_mode': getattr(main_config, 'proto_mode', 'class'),
                'alpha_mode': getattr(main_config, 'alpha_mode', 'entropy'),
                'alpha_max': getattr(main_config, 'alpha_max', 0.2),
                'conf_threshold': getattr(main_config, 'conf_threshold', 0.2),
                'alpha_constant': getattr(main_config, 'alpha_constant', 0.1),
                'warm_start_from_baseline': getattr(main_config, 'warm_start_from_baseline', True),
            }
        model.cond_stage_model = cond_stage_model(
            None, num_voxels, self.cond_dim, global_pool=global_pool, clip_tune=clip_tune, cls_tune=cls_tune,
            **sarhm_kw
        )

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        
        self.device = device    
        self.model = model
        
        self.model.clip_tune = clip_tune
        self.model.cls_tune = cls_tune

        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.cond_stage_model.fmri_latent_dim

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one
      
        # # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        num_workers = getattr(config, 'num_workers', 0) if config else 0
        use_cuda = next(self.model.parameters()).is_cuda
        dataloader = DataLoader(
            dataset, batch_size=bs1, shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=(num_workers > 0),
        )
        test_loader = DataLoader(
            test_dataset, batch_size=bs1, shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=(num_workers > 0),
        )
        self.model.unfreeze_whole_model()
        self.model.freeze_first_stage()
        self.model.freeze_diffusion_model()
        # E: Assert SD (VAE + UNet) is frozen
        for name, p in self.model.named_parameters():
            if "first_stage" in name or (name.startswith("model.") and "cond_stage" not in name):
                assert not p.requires_grad, "SD must be frozen: %s has requires_grad=True" % name
        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        print("trainable_modules (%d): %s" % (len(trainable), trainable[:20] if len(trainable) > 20 else trainable))
        if getattr(config, 'use_sarhm', False):
            proto_freeze = getattr(config, 'proto_freeze_epochs', 5)
            csm = getattr(self.model, 'cond_stage_model', None)
            if csm is not None and getattr(csm, 'sarhm_prototypes', None):
                csm.sarhm_prototypes.prototypes.requires_grad = False
                if proto_freeze > 0:
                    self.model._proto_freeze_epochs = proto_freeze
                    self.model._proto_unfrozen = False

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg
        # torch.compile disabled: validation_step runs generate()/PLMS which can break under Dynamo
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)

        self.model.unfreeze_whole_model()

        from dc_ldm.util import pickle_safe_config
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': pickle_safe_config(config),
                'state': torch.random.get_rng_state()

            },
            os.path.join(output_path, 'checkpoint.pth')
        )
        # Save SAR-HM prototypes for reproducibility (checkpoint already contains them)
        if getattr(config, 'use_sarhm', False):
            csm = getattr(self.model, 'cond_stage_model', None)
            if csm is not None and getattr(csm, 'sarhm_prototypes', None):
                proto_path = getattr(config, 'proto_path', None)
                if proto_path is None:
                    proto_path = os.path.join(output_path, 'prototypes.pt')
                os.makedirs(os.path.dirname(proto_path) or '.', exist_ok=True)
                csm.sarhm_prototypes.save_to_path(proto_path)
                print(f"SAR-HM prototypes saved to {proto_path}")

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path = None):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            try:
                torch.cuda.set_rng_state(state)
            except (RuntimeError, TypeError) as e:
                if "wrong size" in str(e).lower() or "state" in str(e).lower():
                    pass  # RNG state incompatible (e.g. different GPU/PyTorch); skip
                else:
                    raise
            
        with model.ema_scope():
            model.eval()
            total_items = len(fmri_embedding) if limit is None else min(limit, len(fmri_embedding))
            wrap = tqdm(enumerate(fmri_embedding), total=total_items, desc='Generate test',
                        unit='item', bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            for count, item in wrap:
                if limit is not None:
                    if count >= limit:
                        break
                latent = item['eeg']
                if not isinstance(latent, torch.Tensor):
                    latent = torch.as_tensor(latent, dtype=torch.float32, device=self.device)
                else:
                    latent = latent.to(self.device)
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w')  # h w c
                if isinstance(gt_image, np.ndarray):
                    gt_image = torch.from_numpy(gt_image).float().to(self.device)
                wrap.set_description(f'Generate test (item {count+1}/{total_items}, {ddim_steps} steps)')
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'
                latent_rep = repeat(latent, 'h w -> c h w', c=num_samples)
                c, re_latent = model.get_learned_conditioning(latent_rep)
                # c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                if output_path is not None:
                    samples_t = (255. * torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(np.uint8)
                    for copy_idx, img_t in enumerate(samples_t):
                        img_t = rearrange(img_t, 'c h w -> h w c')
                        Image.fromarray(img_t).save(os.path.join(output_path, 
                            f'./test{count}-{copy_idx}.png'))
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to(self.device)  # keep on device for next training step (was .to('cpu') and caused epoch 2 crash)
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)
