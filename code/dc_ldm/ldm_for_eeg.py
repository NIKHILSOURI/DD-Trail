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

# Optional SAR-HM++ (Multi-Level Semantic Prototype Retrieval)
try:
    from sarhm.semantic_query import SemanticQueryHead, pool_eeg_for_query
    from sarhm.semantic_retrieval import SemanticRetrieval, confidence_from_attention
    from sarhm.semantic_adapter import SemanticAdapter as SemanticAdapterPP
    from sarhm.semantic_memory import SemanticMemoryBank
    _SARHMPP_AVAILABLE = True
except ImportError as e:
    _SARHMPP_AVAILABLE = False
    SemanticQueryHead = None
    SemanticRetrieval = None
    SemanticAdapterPP = None
    SemanticMemoryBank = None
    _SARHMPP_IMPORT_ERROR = str(e)


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
                 warm_start_from_baseline=True,
                 normalize_conditioning=True, normalization_type='layernorm', debug_cond_stats=False,
                 use_sarhmpp=False, semantic_prototypes_path=None, semantic_topk=5, semantic_temperature=1.0,
                 semantic_query_pooling='mean', semantic_adapter_mode='mlp_tokens_plus_transformer',
                 semantic_transformer_layers=1, conf_w1=0.5, conf_w2=0.3, conf_w3=0.2,
                 no_confidence_gate=False, sarhmpp_projection_only=False,
                 # SAR-HM improved (optional; backward compatible)
                 use_residual_fusion=True, sarhm_fusion_mode='residual', sarhm_alpha=None,
                 sarhm_residual_clamp=None, residual_scale=1.0, residual_clamp_value=None,
                 use_confidence_gate=True, confidence_source='entropy', confidence_min=0.0, confidence_max=1.0,
                 normalize_memory_embeddings=False, top_k_retrieval=None):
        super().__init__()
        self.no_confidence_gate = no_confidence_gate
        self.sarhmpp_projection_only = sarhmpp_projection_only
        self.use_sarhmpp = use_sarhmpp and _SARHMPP_AVAILABLE
        if use_sarhmpp and not _SARHMPP_AVAILABLE:
            import sys
            err = getattr(sys.modules[__name__], '_SARHMPP_IMPORT_ERROR', 'unknown')
            print("[cond_stage_model] WARNING: use_sarhmpp=True but sarhm++ modules failed to import; using baseline/sarhm. Error: %s" % err)
        self.use_sarhm = use_sarhm and _SARHM_AVAILABLE and not self.use_sarhmpp
        self.ablation_mode = ablation_mode if self.use_sarhm else 'baseline'
        if use_sarhm and not _SARHM_AVAILABLE:
            import sys
            err = getattr(sys.modules[__name__], '_SARHM_IMPORT_ERROR', 'unknown')
            print("[cond_stage_model] WARNING: use_sarhm=True but 'sarhm' failed to import; using baseline. Error: %s" % err)
        self._sarhm_extra = {}  # optional diagnostics (attention, confidence, alpha, entropy)
        self._sarhm_header_printed = False
        self._baseline_header_printed = False
        self._proto_invalid_warned = False
        self.alpha_mode = alpha_mode
        self.alpha_max = float(alpha_max)
        self.conf_threshold = conf_threshold
        self.alpha_constant = alpha_constant
        self.warm_start_from_baseline = warm_start_from_baseline
        self.normalize_conditioning = normalize_conditioning
        self.normalization_type = normalization_type if normalization_type in ('layernorm', 'l2') else 'layernorm'
        self.debug_cond_stats = debug_cond_stats
        # SAR-HM improved: residual fusion and confidence gating (optional)
        self.use_residual_fusion = use_residual_fusion
        self.sarhm_fusion_mode = sarhm_fusion_mode if sarhm_fusion_mode in ('residual', 'original') else 'residual'
        self.sarhm_alpha_fixed = sarhm_alpha  # None = use confidence-based alpha
        self.sarhm_residual_clamp = sarhm_residual_clamp
        self.residual_scale = residual_scale
        self.residual_clamp_value = residual_clamp_value
        self.use_confidence_gate = use_confidence_gate and not no_confidence_gate
        self.confidence_source = confidence_source if confidence_source in ('max', 'entropy', 'margin') else 'entropy'
        self.confidence_min = confidence_min
        self.confidence_max = confidence_max
        self.normalize_memory_embeddings = normalize_memory_embeddings
        self.top_k_retrieval = top_k_retrieval
        if self.normalize_conditioning and self.normalization_type == 'layernorm':
            self.cond_ln = nn.LayerNorm(768)
        else:
            self.cond_ln = None

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
            self.sarhm_hopfield = HopfieldRetrieval(tau=hopfield_tau, top_k=top_k_retrieval)
            self.sarhm_fusion = ConfidenceGatedFusion(gate_mode=gate_mode)
            self.sarhm_adapter = ConditioningAdapter(clip_dim=clip_dim, cond_dim=cond_dim, seq_len=77)
            if self.warm_start_from_baseline:
                self.sarhm_adapter.init_near_zero_delta(scale=0.01)
            self._proto_source = "dummy"
            if proto_path:
                if self.sarhm_prototypes.load_from_path(proto_path):
                    self._proto_source = "loaded"
            # else: train-built prototypes use "train" when set by training script

        # SAR-HM++: multi-level semantic prototype retrieval
        if self.use_sarhmpp and SemanticQueryHead is not None and SemanticRetrieval is not None and SemanticAdapterPP is not None and SemanticMemoryBank is not None:
            self.semantic_query_head = SemanticQueryHead(
                self.fmri_latent_dim,
                output_dim=768,
                hidden_dim=512,
                dropout=0.1,
            )
            # Load or create semantic memory bank
            keys_tensor = None
            if semantic_prototypes_path and os.path.isfile(semantic_prototypes_path):
                state = torch.load(semantic_prototypes_path, map_location='cpu', weights_only=False)
                keys_tensor = state.get('keys', state.get('prototypes'))
                if keys_tensor is not None:
                    keys_tensor = torch.as_tensor(keys_tensor, dtype=torch.float32)
            if keys_tensor is not None:
                self.semantic_memory_bank = SemanticMemoryBank(keys=keys_tensor, dim=768)
                self._sarhmpp_proto_loaded = True
            else:
                self.semantic_memory_bank = SemanticMemoryBank(num_prototypes=0, dim=768)
                self._sarhmpp_proto_loaded = False
            self.semantic_retrieval = SemanticRetrieval(
                top_k=semantic_topk,
                temperature=semantic_temperature,
                conf_w1=conf_w1, conf_w2=conf_w2, conf_w3=conf_w3,
            )
            self.semantic_adapter_pp = SemanticAdapterPP(
                input_dim=768, cond_dim=cond_dim, seq_len=77,
                mode=semantic_adapter_mode,
                num_transformer_layers=semantic_transformer_layers,
            )
            if warm_start_from_baseline:
                self.semantic_adapter_pp.init_near_zero_delta(scale=0.01)
            self._sarhmpp_header_printed = False
        else:
            self._sarhmpp_header_printed = False

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

    # -------------------------------------------------------------------------
    # Baseline residual fusion (ONLY fusion used): c_final = c_base + alpha*(c_sar - c_base).
    # Cross-attention input: ddpm.apply_model key 'c_crossattn', cond as list [c], shape [B,77,768].
    # Alpha gating + fallback (invalid prototypes or low confidence -> alpha=0) makes SAR-HM
    # non-degrading by design: when in doubt, we use baseline only.
    # -------------------------------------------------------------------------

    def has_valid_prototypes(self):
        """Return True iff prototypes tensor P exists, shape [K,768], and is finite."""
        if not self.use_sarhm or getattr(self, 'sarhm_prototypes', None) is None:
            return False
        P = getattr(self.sarhm_prototypes, 'P', None)
        if P is None or not isinstance(P, torch.Tensor):
            return False
        if P.ndim != 2:
            return False
        K, D = P.shape
        if D != 768:
            return False
        if not torch.isfinite(P).all():
            return False
        if P.numel() == 0:
            return False
        return True

    def get_proto_source(self):
        """Return current prototype source string; 'dummy' if never loaded."""
        return getattr(self, '_proto_source', 'dummy')

    def forward(self, x):
        latent_crossattn = self.mae(x)
        latent_return = latent_crossattn

        # A1: Always compute baseline conditioning c_base [B, L, cond_dim]
        c_base = self._baseline_conditioning(latent_crossattn)
        target_seq = 77
        c_base = self._align_to_seq_len(c_base, target_seq)

        # SAR-HM++ path: semantic query -> top-k retrieval -> adapter -> residual fusion
        if self.use_sarhmpp and getattr(self, 'semantic_memory_bank', None) is not None:
            if self.semantic_memory_bank.num_prototypes == 0 or not getattr(self, '_sarhmpp_proto_loaded', False):
                if not getattr(self, '_sarhmpp_header_printed', False):
                    self._sarhmpp_header_printed = True
                    print("[SAR-HM++] No valid semantic prototypes; alpha=0 (baseline only).")
                self._sarhm_extra = {'c_base': c_base, 'alpha': None, 'q_sem': None, 'm_sem': None}
                return c_base, latent_return
            if not getattr(self, '_sarhmpp_header_printed', False):
                self._sarhmpp_header_printed = True
                print("SAR-HM++ ACTIVE | top_k=%s | semantic adapter | confidence-gated fusion" % getattr(self.semantic_retrieval, 'top_k', 5))
            self._last_mae_latent = latent_crossattn
            # Pool MAE tokens to [B, embed_dim] for semantic query (same pooling as SAR-HM for consistency)
            pooled = pool_eeg_tokens(latent_crossattn, self.global_pool)
            q_sem = self.semantic_query_head(pooled)
            if getattr(self, 'sarhmpp_projection_only', False):
                m_sem = q_sem
                attn = None
                conf = torch.ones(q_sem.shape[0], device=q_sem.device, dtype=q_sem.dtype)
                topk_idx = None
            else:
                m_sem, attn, conf, topk_idx = self.semantic_retrieval(q_sem, self.semantic_memory_bank.keys)
            c_sem = self.semantic_adapter_pp(m_sem)
            effective_alpha_max = getattr(self, '_effective_alpha_max', self.alpha_max)
            if getattr(self, 'no_confidence_gate', False):
                alpha = torch.full((c_sem.shape[0],), effective_alpha_max, device=c_sem.device, dtype=c_sem.dtype)
            else:
                alpha = (effective_alpha_max * conf).clamp(0, effective_alpha_max)
                alpha = alpha.clone()
                alpha[conf < self.conf_threshold] = 0.0
            if self.normalize_conditioning:
                if self.cond_ln is not None:
                    c_base_n = self.cond_ln(c_base)
                    c_sem_n = self.cond_ln(c_sem)
                else:
                    eps = 1e-12
                    c_base_n = F.normalize(c_base, dim=-1, eps=eps)
                    c_sem_n = F.normalize(c_sem, dim=-1, eps=eps)
            else:
                c_base_n = c_base
                c_sem_n = c_sem
            alpha_bc = alpha.view(-1, 1, 1).to(c_base_n.dtype)
            c_final = c_base_n + alpha_bc * (c_sem_n - c_base_n)
            self._sarhm_extra = {
                'q_sem': q_sem, 'm_sem': m_sem, 'attn': attn, 'confidence': conf,
                'alpha': alpha, 'c_base': c_base, 'c_sem': c_sem, 'topk_idx': topk_idx,
            }
            return c_final, q_sem

        if self.use_sarhm:
            # Ablation: force baseline-only path (no SAR-HM fusion)
            if getattr(self, "_no_sarhm", False):
                if not getattr(self, "_no_sarhm_header_printed", False):
                    self._no_sarhm_header_printed = True
                    print("SAR-HM: OFF | _no_sarhm=True (CLI override)")
                self._sarhm_extra = {'c_base': c_base, 'alpha': None, 'c_sar': None}
                return c_base, latent_return

            # Hard fallback: invalid prototypes OR proto_source dummy/None -> alpha=0 (non-degrading by design)
            valid_proto = self.has_valid_prototypes()
            proto_src = self.get_proto_source()
            if proto_src in ("dummy", None) or not valid_proto:
                valid_proto = False
                if not getattr(self, '_proto_invalid_warned', False):
                    self._proto_invalid_warned = True
                    print("[SAR-HM] invalid prototypes -> alpha forced 0 (baseline only). proto_source=%s" % proto_src)
                # We still compute c_sar path but will force alpha=0 below

            # === SAR-HM path: c_sar then residual fusion c_final = c_base + alpha * (c_sar - c_base) ===
            if not self._sarhm_header_printed:
                self._sarhm_header_printed = True
                K = self.sarhm_prototypes.P.shape[0] if getattr(self.sarhm_prototypes, 'P', None) is not None else 0
                tau = getattr(self.sarhm_hopfield, "tau", 1.0)
                gate = getattr(self.sarhm_fusion, "gate_mode", "max")
                proto_src = self.get_proto_source()
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
                # Optional: normalize prototype vectors before retrieval (improves memory quality)
                memory = self.sarhm_prototypes.P
                if getattr(self, 'normalize_memory_embeddings', False):
                    memory = F.normalize(memory.float(), dim=-1, eps=1e-12).to(memory.dtype)
                top_k = getattr(self, 'top_k_retrieval', None)
                z_ret, attn, logits = self.sarhm_hopfield(z_orig, memory, top_k=top_k)
                if self.ablation_mode == 'hopfield_no_gate':
                    z_fused = z_ret
                    confidence = torch.ones(z_orig.shape[0], device=z_orig.device, dtype=z_orig.dtype) * 0.05
                else:
                    z_fused, confidence = self.sarhm_fusion(z_orig, z_ret, attn)
            c_sar = self.sarhm_adapter(z_fused)  # [B, 77, cond_dim]

            # Alpha from confidence (entropy/max/constant); alpha = alpha_max * conf, conf < conf_threshold -> 0
            effective_alpha_max = getattr(self, '_effective_alpha_max', self.alpha_max)
            if compute_alpha_from_attention is not None and attn is not None:
                alpha, conf_out, entropy = compute_alpha_from_attention(
                    attn, alpha_mode=self.alpha_mode, alpha_max=float(effective_alpha_max),
                    conf_threshold=self.conf_threshold, alpha_constant=self.alpha_constant)
            elif attn is None:
                alpha = torch.full((c_sar.shape[0],), self.alpha_constant, device=c_sar.device, dtype=c_sar.dtype)
                conf_out = alpha.clone()
                entropy = torch.zeros(c_sar.shape[0], device=c_sar.device, dtype=c_sar.dtype)
            else:
                alpha = torch.full((c_sar.shape[0],), self.alpha_constant, device=c_sar.device, dtype=c_sar.dtype)
                conf_out = confidence
                entropy = torch.zeros(c_sar.shape[0], device=c_sar.device, dtype=c_sar.dtype)
            # Optional: confidence from similarity margin (top1 - top2) when confidence_source == 'margin'
            if getattr(self, 'confidence_source', 'entropy') == 'margin' and logits is not None and logits.shape[-1] >= 2:
                top1_sim = logits.max(dim=-1).values
                top2_sim = logits.topk(2, dim=-1).values[:, 1]
                margin = (top1_sim - top2_sim).clamp(min=0).float()
                m_max = margin.max().clamp(min=1e-8)
                conf_out = (margin / m_max).clamp(0, 1).to(conf_out.dtype)

            if self.ablation_mode == 'hopfield_no_gate':
                alpha = torch.full((c_sar.shape[0],), 0.05, device=c_sar.device, dtype=c_sar.dtype)

            # Hard fallback: invalid prototypes -> force alpha=0
            if not valid_proto:
                alpha = torch.zeros(alpha.shape[0], device=alpha.device, dtype=alpha.dtype)
            # Also force alpha=0 when proto_source is dummy/None (already set valid_proto=False above)
            # Ablation overrides (Stage C debug flags)
            if getattr(self, "_baseline_only", False):
                alpha = torch.zeros(alpha.shape[0], device=alpha.device, dtype=alpha.dtype)
            if getattr(self, "_force_alpha", None) is not None:
                alpha = torch.full((alpha.shape[0],), float(self._force_alpha), device=alpha.device, dtype=alpha.dtype)

            # SAR-HM improved: confidence gating — clamp confidence and optionally modulate alpha
            conf_out_clamped = conf_out.clamp(
                getattr(self, 'confidence_min', 0.0),
                getattr(self, 'confidence_max', 1.0)
            )
            if getattr(self, 'use_confidence_gate', True) and conf_out_clamped is not None:
                alpha = alpha * conf_out_clamped
            if getattr(self, 'sarhm_alpha_fixed', None) is not None:
                alpha = torch.full((alpha.shape[0],), float(self.sarhm_alpha_fixed), device=alpha.device, dtype=alpha.dtype)

            # Normalize c_base and c_sar before fusion (match scales; reduces distribution mismatch)
            if self.normalize_conditioning:
                if self.cond_ln is not None:
                    c_base_n = self.cond_ln(c_base)
                    c_sar_n = self.cond_ln(c_sar)
                else:
                    eps = 1e-12
                    c_base_n = F.normalize(c_base, dim=-1, eps=eps)
                    c_sar_n = F.normalize(c_sar, dim=-1, eps=eps)
            else:
                c_base_n = c_base
                c_sar_n = c_sar

            # SAR-HM improved: residual fusion or original (replace). final_cond = baseline + alpha * (sarhm - baseline)
            use_residual = getattr(self, 'use_residual_fusion', True)
            if use_residual:
                residual = (c_sar_n - c_base_n).float()
                scale = getattr(self, 'residual_scale', 1.0)
                if scale != 1.0:
                    residual = residual * scale
                if getattr(self, 'sarhm_residual_clamp', None) is not None:
                    residual = residual.clamp(-float(self.sarhm_residual_clamp), float(self.sarhm_residual_clamp))
                if getattr(self, 'residual_clamp_value', None) is not None:
                    rnorm = residual.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    residual = residual * (rnorm.clamp(max=float(self.residual_clamp_value)) / rnorm)
                alpha_bc = alpha.view(-1, 1, 1).to(c_base_n.dtype)
                c_final = c_base_n + alpha_bc * residual.to(c_base_n.dtype)
            else:
                # Original (replace): c_final = c_sar when not residual; optional blend still via alpha
                if getattr(self, 'sarhm_fusion_mode', 'residual') == 'original':
                    c_final = c_sar_n
                else:
                    alpha_bc = alpha.view(-1, 1, 1).to(c_base_n.dtype)
                    c_final = c_base_n + alpha_bc * (c_sar_n - c_base_n)

            if self.debug_cond_stats:
                def _stats(t, name):
                    t = t.float()
                    return (name, t.mean().item(), t.std().item(), t.min().item(), t.max().item(),
                            t.norm().item() if t.numel() > 0 else 0.0)
                cb = _stats(c_base_n, 'c_base')
                cs = _stats(c_sar_n, 'c_sar')
                cf = _stats(c_final, 'c_final')
                ah = alpha.float()
                print("[COND_STATS] %s mean=%.4f std=%.4f min=%.4f max=%.4f norm=%.4f" % cb)
                print("[COND_STATS] %s mean=%.4f std=%.4f min=%.4f max=%.4f norm=%.4f" % cs)
                print("[COND_STATS] %s mean=%.4f std=%.4f min=%.4f max=%.4f norm=%.4f" % cf)
                print("[COND_STATS] alpha min=%.4f mean=%.4f max=%.4f | conf min=%.4f mean=%.4f max=%.4f | proto_valid=%s" % (
                    ah.min().item(), ah.mean().item(), ah.max().item(),
                    conf_out.min().item(), conf_out.mean().item(), conf_out.max().item(), valid_proto))

            self._sarhm_extra = {
                'confidence': conf_out,
                'confidence_clamped': conf_out_clamped,
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
                'normalize_conditioning': getattr(main_config, 'normalize_conditioning', True),
                'normalization_type': getattr(main_config, 'normalization_type', 'layernorm'),
                'debug_cond_stats': getattr(main_config, 'debug_cond_stats', False),
                'use_sarhmpp': getattr(main_config, 'use_sarhmpp', False),
                'semantic_prototypes_path': getattr(main_config, 'semantic_prototypes_path', None),
                'semantic_topk': getattr(main_config, 'semantic_topk', 5),
                'semantic_temperature': getattr(main_config, 'semantic_temperature', 1.0),
                'semantic_query_pooling': getattr(main_config, 'semantic_query_pooling', 'mean'),
                'semantic_adapter_mode': getattr(main_config, 'semantic_adapter_mode', 'mlp_tokens_plus_transformer'),
                'semantic_transformer_layers': getattr(main_config, 'semantic_transformer_layers', 1),
                'conf_w1': getattr(main_config, 'conf_w1', 0.5),
                'conf_w2': getattr(main_config, 'conf_w2', 0.3),
                'conf_w3': getattr(main_config, 'conf_w3', 0.2),
                'no_confidence_gate': getattr(main_config, 'no_confidence_gate', False),
                'sarhmpp_projection_only': getattr(main_config, 'sarhmpp_projection_only', False),
                # SAR-HM improved
                'use_residual_fusion': getattr(main_config, 'use_residual_fusion', True),
                'sarhm_fusion_mode': getattr(main_config, 'sarhm_fusion_mode', 'residual'),
                'sarhm_alpha': getattr(main_config, 'sarhm_alpha', None),
                'sarhm_residual_clamp': getattr(main_config, 'sarhm_residual_clamp', None),
                'residual_scale': getattr(main_config, 'residual_scale', 1.0),
                'residual_clamp_value': getattr(main_config, 'residual_clamp_value', None),
                'use_confidence_gate': getattr(main_config, 'use_confidence_gate', True),
                'confidence_source': getattr(main_config, 'confidence_source', 'entropy'),
                'confidence_min': getattr(main_config, 'confidence_min', 0.0),
                'confidence_max': getattr(main_config, 'confidence_max', 1.0),
                'normalize_memory_embeddings': getattr(main_config, 'normalize_memory_embeddings', False),
                'top_k_retrieval': getattr(main_config, 'top_k_retrieval', None),
            }
        model.cond_stage_model = cond_stage_model(
            metafile, num_voxels, self.cond_dim, global_pool=global_pool, clip_tune=clip_tune, cls_tune=cls_tune,
            **sarhm_kw
        )
        # Load cond_stage_model state from checkpoint (cond_ln, mae, etc.) so missing cond_ln keys are restored when present
        csm_prefix = 'cond_stage_model.'
        csm_sd = {k[len(csm_prefix):]: v for k, v in pl_sd.items() if k.startswith(csm_prefix)}
        if csm_sd:
            m_csm, u_csm = model.cond_stage_model.load_state_dict(csm_sd, strict=False)
            if m_csm or u_csm:
                print(f"Cond stage model load from ckpt: {len(m_csm)} missing, {len(u_csm)} unexpected")
            if sarhm_kw.get('use_sarhm') and m_csm and any(k.startswith('cond_ln') for k in m_csm):
                print("[SAR-HM] WARNING: checkpoint missing cond_ln weights; LayerNorm is randomly initialized. Fine-tune or load a SAR-HM ckpt for correct normalization.")
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
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'config': pickle_safe_config(config),
            'state': torch.random.get_rng_state()
        }
        torch.save(ckpt, os.path.join(output_path, 'checkpoint.pth'))
        # When validation doesn't save checkpoint_best (e.g. cls_tune=False or disable_image_generation_in_val=True),
        # write checkpoint_best.pth at end of Stage B so downstream scripts that expect it still get a file.
        if getattr(config, 'disable_image_generation_in_val', False) or not getattr(self.model, 'cls_tune', False):
            torch.save(ckpt, os.path.join(output_path, 'checkpoint_best.pth'))
            print('Stage B: saved checkpoint_best.pth (final checkpoint; validation did not run classification-based best save).\n')
        # Save SAR-HM prototypes for reproducibility (checkpoint already contains them)
        if getattr(config, 'use_sarhm', False):
            csm = getattr(self.model, 'cond_stage_model', None)
            if csm is not None and getattr(csm, 'sarhm_prototypes', None):
                proto_path = getattr(config, 'proto_path', None)
                if proto_path is None:
                    proto_path = os.path.join(output_path, 'prototypes.pt')
                os.makedirs(os.path.dirname(proto_path) or '.', exist_ok=True)
                csm.sarhm_prototypes.save_to_path_with_metadata(
                    proto_path,
                    proto_source=getattr(csm, '_proto_source', 'train'),
                    normalization_type=getattr(csm, 'normalization_type', 'layernorm'),
                )
                print(f"SAR-HM prototypes saved to {proto_path}")
                # One-time prototype stats (mean/std/min/max, finite only)
                with torch.no_grad():
                    P = csm.sarhm_prototypes.P.detach().cpu().float()
                    P_fin = P[torch.isfinite(P)]
                    if P_fin.numel() > 0:
                        print("[PROTO_STATS] shape=%s mean=%.4f std=%.4f min=%.4f max=%.4f (finite)" % (
                            tuple(P.shape), P_fin.mean().item(), P_fin.std().item(), P_fin.min().item(), P_fin.max().item()))
                    else:
                        print("[PROTO_STATS] shape=%s (no finite values)" % (tuple(P.shape),))

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
                if latent.dim() == 3 and latent.shape[0] == 1:
                    latent = latent.squeeze(0)
                while latent.dim() > 2 and latent.shape[-1] == 1:
                    latent = latent.squeeze(-1)
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w')  # h w c
                if isinstance(gt_image, np.ndarray):
                    gt_image = torch.from_numpy(gt_image).float().to(self.device)
                # Benchmark / raw PIL paths often pass [0,1]; training uses [-1,1]
                if float(gt_image.max()) <= 1.0 and float(gt_image.min()) >= 0.0:
                    gt_image = gt_image * 2.0 - 1.0
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
                if gt_image.shape[-2:] != x_samples_ddim.shape[-2:]:
                    gt_image = F.interpolate(
                        gt_image,
                        size=x_samples_ddim.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                if output_path is not None:
                    samples_t = (255. * torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(np.uint8)
                    for copy_idx, img_t in enumerate(samples_t):
                        img_t = rearrange(img_t, 'c h w -> h w c')
                        Image.fromarray(img_t).save(os.path.join(output_path, 
                            f'./test{count}-{copy_idx}.png'))
        
        # display as grid (handle empty dataset: no test samples)
        if not all_samples:
            ph_c = shape[0]
            ph_h, ph_w = 64, 64
            placeholder = torch.ones(1, 1, ph_c, ph_h, ph_w, device=self.device) * 0.5
            grid = rearrange(placeholder, 'n b c h w -> (n b) c h w', n=1, b=1)
            grid = make_grid(grid, nrow=1)
            grid_np = (255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()).astype(np.uint8)
            model = model.to(self.device)
            return grid_np, np.zeros((0,), dtype=np.uint8)

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
                'normalize_conditioning': getattr(main_config, 'normalize_conditioning', True),
                'normalization_type': getattr(main_config, 'normalization_type', 'layernorm'),
                'debug_cond_stats': getattr(main_config, 'debug_cond_stats', False),
                'use_sarhmpp': getattr(main_config, 'use_sarhmpp', False),
                'semantic_prototypes_path': getattr(main_config, 'semantic_prototypes_path', None),
                'semantic_topk': getattr(main_config, 'semantic_topk', 5),
                'semantic_temperature': getattr(main_config, 'semantic_temperature', 1.0),
                'semantic_query_pooling': getattr(main_config, 'semantic_query_pooling', 'mean'),
                'semantic_adapter_mode': getattr(main_config, 'semantic_adapter_mode', 'mlp_tokens_plus_transformer'),
                'semantic_transformer_layers': getattr(main_config, 'semantic_transformer_layers', 1),
                'conf_w1': getattr(main_config, 'conf_w1', 0.5),
                'conf_w2': getattr(main_config, 'conf_w2', 0.3),
                'conf_w3': getattr(main_config, 'conf_w3', 0.2),
                'no_confidence_gate': getattr(main_config, 'no_confidence_gate', False),
                'sarhmpp_projection_only': getattr(main_config, 'sarhmpp_projection_only', False),
                # SAR-HM improved (inference-time)
                'use_residual_fusion': getattr(main_config, 'use_residual_fusion', True),
                'sarhm_fusion_mode': getattr(main_config, 'sarhm_fusion_mode', 'residual'),
                'sarhm_alpha': getattr(main_config, 'eval_fusion_alpha', None) or getattr(main_config, 'sarhm_alpha', None),
                'sarhm_residual_clamp': getattr(main_config, 'sarhm_residual_clamp', None),
                'residual_scale': getattr(main_config, 'residual_scale', 1.0),
                'residual_clamp_value': getattr(main_config, 'residual_clamp_value', None),
                'use_confidence_gate': getattr(main_config, 'eval_confidence_gate', getattr(main_config, 'use_confidence_gate', True)),
                'confidence_source': getattr(main_config, 'confidence_source', 'entropy'),
                'confidence_min': getattr(main_config, 'confidence_min', 0.0),
                'confidence_max': getattr(main_config, 'confidence_max', 1.0),
                'normalize_memory_embeddings': getattr(main_config, 'normalize_memory_embeddings', False),
                'top_k_retrieval': getattr(main_config, 'eval_retrieval_top_k', None) or getattr(main_config, 'top_k_retrieval', None),
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
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'config': pickle_safe_config(config),
            'state': torch.random.get_rng_state()
        }
        torch.save(ckpt, os.path.join(output_path, 'checkpoint.pth'))
        # When validation doesn't save checkpoint_best (e.g. cls_tune=False or disable_image_generation_in_val=True),
        # write checkpoint_best.pth at end of Stage B so downstream scripts that expect it still get a file.
        if getattr(config, 'disable_image_generation_in_val', False) or not getattr(self.model, 'cls_tune', False):
            torch.save(ckpt, os.path.join(output_path, 'checkpoint_best.pth'))
            print('Stage B: saved checkpoint_best.pth (final checkpoint; validation did not run classification-based best save).\n')
        # Save SAR-HM prototypes for reproducibility (checkpoint already contains them)
        if getattr(config, 'use_sarhm', False):
            csm = getattr(self.model, 'cond_stage_model', None)
            if csm is not None and getattr(csm, 'sarhm_prototypes', None):
                proto_path = getattr(config, 'proto_path', None)
                if proto_path is None:
                    proto_path = os.path.join(output_path, 'prototypes.pt')
                os.makedirs(os.path.dirname(proto_path) or '.', exist_ok=True)
                csm.sarhm_prototypes.save_to_path_with_metadata(
                    proto_path,
                    proto_source=getattr(csm, '_proto_source', 'train'),
                    normalization_type=getattr(csm, 'normalization_type', 'layernorm'),
                )
                print(f"SAR-HM prototypes saved to {proto_path}")
                # One-time prototype stats (mean/std/min/max, finite only)
                with torch.no_grad():
                    P = csm.sarhm_prototypes.P.detach().cpu().float()
                    P_fin = P[torch.isfinite(P)]
                    if P_fin.numel() > 0:
                        print("[PROTO_STATS] shape=%s mean=%.4f std=%.4f min=%.4f max=%.4f (finite)" % (
                            tuple(P.shape), P_fin.mean().item(), P_fin.std().item(), P_fin.min().item(), P_fin.max().item()))
                    else:
                        print("[PROTO_STATS] shape=%s (no finite values)" % (tuple(P.shape),))

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path=None,
                 cfg_scale=1.0, cfg_uncond='zeros', pbar_desc=None):
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
            _desc = pbar_desc if pbar_desc else 'Generate test'
            wrap = tqdm(enumerate(fmri_embedding), total=total_items, desc=_desc,
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
                # Benchmark / callers may pass (1, H, W); einops repeat expects 2D (H, W).
                if latent.dim() == 3 and latent.shape[0] == 1:
                    latent = latent.squeeze(0)
                while latent.dim() > 2 and latent.shape[-1] == 1:
                    latent = latent.squeeze(-1)
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w')  # h w c
                if isinstance(gt_image, np.ndarray):
                    gt_image = torch.from_numpy(gt_image).float().to(self.device)
                if float(gt_image.max()) <= 1.0 and float(gt_image.min()) >= 0.0:
                    gt_image = gt_image * 2.0 - 1.0
                wrap.set_description(f'{_desc} (item {count+1}/{total_items}, {ddim_steps} PLMS steps)')
                latent_rep = repeat(latent, 'h w -> c h w', c=num_samples)
                c, re_latent = model.get_learned_conditioning(latent_rep)
                # CFG: unconditional conditioning tensor [B,77,768] for PLMS torch.cat([uc, c])
                uc = None
                if cfg_scale != 1.0:
                    c_tensor = c if isinstance(c, torch.Tensor) else c[list(c.keys())[0]][0]
                    dtype, dev = c_tensor.dtype, c_tensor.device
                    if cfg_uncond == 'zeros':
                        uc = torch.zeros(num_samples, 77, 768, device=dev, dtype=dtype)
                    else:
                        uc = c_tensor.detach().clone()
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=cfg_scale,
                                                unconditional_conditioning=uc)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                if gt_image.shape[-2:] != x_samples_ddim.shape[-2:]:
                    gt_image = F.interpolate(
                        gt_image,
                        size=x_samples_ddim.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                if output_path is not None:
                    samples_t = (255. * torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(np.uint8)
                    for copy_idx, img_t in enumerate(samples_t):
                        img_t = rearrange(img_t, 'c h w -> h w c')
                        Image.fromarray(img_t).save(os.path.join(output_path, 
                            f'./test{count}-{copy_idx}.png'))
        
        # display as grid (handle empty dataset: no test samples)
        if not all_samples:
            ph_c = shape[0]
            ph_h, ph_w = 64, 64
            placeholder = torch.ones(1, 1, ph_c, ph_h, ph_w, device=self.device) * 0.5
            grid = rearrange(placeholder, 'n b c h w -> (n b) c h w', n=1, b=1)
            grid = make_grid(grid, nrow=1)
            grid_np = (255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()).astype(np.uint8)
            model = model.to(self.device)
            return grid_np, np.zeros((0,), dtype=np.uint8)

        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to(self.device)  # keep on device for next training step (was .to('cpu') and caused epoch 2 crash)
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)
