"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager, nullcontext
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
try:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except ImportError:
    try:
        from pytorch_lightning.utilities.distributed import rank_zero_only
    except ImportError:
        from pytorch_lightning.utilities import rank_zero_only

from dc_ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config, pickle_safe_config
from dc_ldm.modules.ema import LitEma
from dc_ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from dc_ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from dc_ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from dc_ldm.models.diffusion.ddim import DDIMSampler
from dc_ldm.models.diffusion.plms import PLMSSampler
from PIL import Image
import torch.nn.functional as F
from eval_metrics import get_similarity_metric

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def _safe_zero_loss_for_backward(module, device):
    """Return a scalar tensor that has grad_fn so Lightning backward() works. Uses first trainable param."""
    trainable = [p for p in module.parameters() if p.requires_grad]
    if trainable:
        return (0.0 * trainable[0].sum()).to(device)
    return torch.tensor(0.0, device=device, dtype=torch.float32)


def _loss_ok_for_backward(loss):
    """True if loss can be used in loss.backward() without RuntimeError."""
    if loss is None or not isinstance(loss, torch.Tensor):
        return False
    return loss.requires_grad or (getattr(loss, 'grad_fn', None) is not None)


from dc_ldm.modules.encoders.modules import FrozenImageEmbedder
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ddim_steps=300
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        self.validation_count = 0
        self.ddim_steps = ddim_steps
        self.return_cond = False
        self.output_path = None
        self.main_config = None
        self.best_val = 0.0 
        self.run_full_validation_threshold = 0.0
        self.eval_avg = True

    def re_init_ema(self):
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")   

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing (first 5): {list(missing)[:5]}")
        if len(unexpected) > 0:
            print(f"Unexpected (first 5): {list(unexpected)[:5]}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t.cpu()].to(loss.device) * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        self.train()
        self.cond_stage_model.train()  ###到底是在哪里训练的

        def _log_and_return(loss, loss_dict=None):
            if loss_dict is not None:
                self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)
            else:
                ld = loss.detach() if getattr(loss, 'requires_grad', False) else loss
                self.log_dict({'train/loss': ld, 'train/loss_simple': ld, 'train/loss_vlb': ld, 'train/loss_clip': ld}, prog_bar=True, logger=True, on_step=False, on_epoch=True)
            if self.use_scheduler and loss_dict is not None:
                lr = self.optimizers().param_groups[0]['lr']
                self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=False, on_epoch=True)
            return loss

        # Optional: skip real training for epochs < start_epoch (e.g. --start_epoch 2 to test epoch 2 only)
        start_epoch = getattr(getattr(self, 'main_config', None), 'start_epoch', 0) or 0
        if start_epoch > 0 and getattr(self, 'trainer', None) is not None and self.trainer.current_epoch < start_epoch:
            loss = _safe_zero_loss_for_backward(self, self.device)
            return _log_and_return(loss, loss_dict=None)

        try:
            loss, loss_dict = self.shared_step(batch)
        except Exception as e:
            import traceback
            rank_zero_only(lambda: print(f"[DDPM] training_step shared_step failed (epoch {getattr(self.trainer, 'current_epoch', '?')} batch {batch_idx}), using safe zero loss and continuing: {e}"))()
            rank_zero_only(lambda: traceback.print_exc())()
            loss = _safe_zero_loss_for_backward(self, self.device)
            return _log_and_return(loss, loss_dict=None)

        if not _loss_ok_for_backward(loss):
            rank_zero_only(lambda: print(f"[DDPM] loss has no grad_fn (epoch {getattr(self.trainer, 'current_epoch', '?')} batch {batch_idx}), using safe zero loss and continuing"))()
            loss = _safe_zero_loss_for_backward(self, self.device)
            return _log_and_return(loss, loss_dict=None)
        return _log_and_return(loss, loss_dict)

    
    @torch.no_grad()
    def generate(self, data, num_samples, ddim_steps=300, HW=None, limit=None, state=None,
                 debug_sampling_steps=None, debug_save_dir=None):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.p_channels, 
                self.p_image_size, self.p_image_size)
        else:
            num_resolutions = len(self.ch_mult)
            shape = (self.p_channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        model.eval()
        if torch.cuda.is_available():
            state = torch.cuda.get_rng_state() if state is None else state
            torch.cuda.set_rng_state(state)
        else:
            state = torch.get_rng_state() if state is None else state
            torch.set_rng_state(state)

        # rng = torch.Generator(device=self.device).manual_seed(2022).set_state(state)

        # state = torch.cuda.get_rng_state()
        n_items = len(data['eeg'])
        n_run = min(limit, n_items) if limit is not None else n_items
        if n_run > 0:
            print(f"Validation generate: {n_run} items (~70 min/item on typical GPU). Remaining in bar below.")
        debug_steps_set = set(int(s) for s in debug_sampling_steps) if debug_sampling_steps else None
        with model.ema_scope():
            items_iter = list(zip(data['eeg'], data['image']))
            if limit is not None:
                items_iter = items_iter[:limit]
            for count, item in tqdm(enumerate(items_iter), total=len(items_iter), desc='Val generate',
                                    unit='item', bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
                latent = item[0] # fmri embedding
                gt_image = rearrange(item[1], 'h w c -> 1 c h w') # h w c
                csm = getattr(model, 'cond_stage_model', None)
                if csm is not None and hasattr(csm, 'set_batch_context') and isinstance(data, dict):
                    ctx = {k: data[k] for k in ('domain_id', 'domain_key') if k in data}
                    if 'domain_id' in data and isinstance(data['domain_id'], torch.Tensor) and data['domain_id'].dim() > 0:
                        ctx['domain_id'] = data['domain_id'][count:count + 1]
                    if 'domain_key' in data and isinstance(data['domain_key'], (list, tuple)) and len(data['domain_key']) > count:
                        ctx['domain_key'] = data['domain_key'][count]
                    if ctx:
                        csm.set_batch_context(ctx)
                # Optional: save intermediate decoded images at given steps for first item (noise debugging)
                img_cb = None
                if count == 0 and debug_steps_set and debug_save_dir:
                    import os
                    os.makedirs(debug_save_dir, exist_ok=True)
                    def _save_step(pred_x0, i, step):
                        if int(step) in debug_steps_set:
                            x = model.decode_first_stage(pred_x0.float())
                            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                            for b in range(x.shape[0]):
                                arr = (255. * x[b].detach().cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
                                Image.fromarray(arr).save(
                                    os.path.join(debug_save_dir, 'step%d_sample%d.png' % (int(step), b)))
                    img_cb = _save_step
                try:
                    c, re_latent = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                    samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                    conditioning=c,
                                                    batch_size=num_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    generator=None,
                                                    img_callback=img_cb)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0,min=0.0, max=1.0)
                    gt_image = torch.clamp((gt_image+1.0)/2.0,min=0.0, max=1.0)
                    all_samples.append(torch.cat([gt_image.detach().cpu(), x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                finally:
                    if csm is not None and hasattr(csm, 'clear_batch_context'):
                        csm.clear_batch_context()
        
        # Guard: return safe values when no samples (e.g. empty batch)
        if not all_samples:
            return None, np.array([]), state
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8), state

    def save_images(self, all_samples, suffix=0):
        # print('output_path')
        # print(self.output_path)
        if self.output_path is not None:
            os.makedirs(os.path.join(self.output_path, 'val', f'{self.validation_count}_{suffix}'), exist_ok=True)
            for sp_idx, imgs in enumerate(all_samples):
                # for copy_idx, img in enumerate(imgs[1:]):
                for copy_idx, img in enumerate(imgs):
                    img = rearrange(img, 'c h w -> h w c')
                    Image.fromarray(img).save(os.path.join(self.output_path, 'val', 
                                    f'{self.validation_count}_{suffix}', f'test{sp_idx}-{copy_idx}.png'))
                                    
    def full_validation(self, batch, state=None):
        # Stage B: never run PLMS/DDIM (saves time/VRAM). Use Stage C for generation.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if batch_idx != 0:
            return
        
        main_cfg = getattr(self, 'main_config', None)
        disable_gen = getattr(main_cfg, 'disable_image_generation_in_val', True)
        every_n = getattr(main_cfg, 'val_image_gen_every_n_epoch', 0)
        current_epoch = self.trainer.current_epoch if getattr(self, 'trainer', None) else 0
        
        do_gen = not disable_gen and every_n > 0 and (current_epoch % every_n == 0)
        val_limit = getattr(main_cfg, 'val_gen_limit', 2)
        val_num_samples = getattr(main_cfg, 'val_num_samples', 2)
        val_ddim_steps = getattr(main_cfg, 'val_ddim_steps', 50)
        rank_zero_only(lambda: print("[VAL_GEN] generating images: %s, num_items=%s, ddim_steps=%s, num_samples=%s" % (
            'yes' if do_gen else 'no', val_limit, val_ddim_steps, val_num_samples)))()
        
        if do_gen:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                debug_steps = getattr(main_cfg, 'debug_sampling_steps', None)
                debug_dir = None
                if debug_steps and getattr(self, 'trainer', None) and getattr(self.trainer, 'log_dir', None):
                    debug_dir = os.path.join(self.trainer.log_dir, 'val_intermediates')
                grid, all_samples, _ = self.generate(
                    batch, num_samples=val_num_samples, ddim_steps=val_ddim_steps,
                    HW=getattr(main_cfg, 'HW', None), limit=val_limit,
                    debug_sampling_steps=debug_steps, debug_save_dir=debug_dir)
                if all_samples is not None and len(all_samples) > 0:
                    # Store for epoch-stats callback (SAR-HM improved: best-by-CLIP, epoch_stats CSV)
                    setattr(self, '_last_val_samples', all_samples)
                    setattr(self, '_last_val_epoch', current_epoch)
                    # Single-image mode: save exactly one final image per epoch to val_images/epoch_XXX.png
                    output_path = getattr(self, 'output_path', None) or (getattr(main_cfg, 'output_path', None) if main_cfg else None)
                    if output_path and val_limit == 1 and val_num_samples == 1:
                        val_images_dir = os.path.join(output_path, 'val_images')
                        os.makedirs(val_images_dir, exist_ok=True)
                        single = all_samples[0][1]
                        if single.ndim == 3:
                            path = os.path.join(val_images_dir, 'epoch_%03d.png' % current_epoch)
                            Image.fromarray(rearrange(single, 'c h w -> h w c')).save(path)
                            rank_zero_only(lambda: print("[VAL_IMG] epoch=%d saved path=%s" % (current_epoch, path)))()
                        else:
                            self.save_images(all_samples, suffix='epoch%d' % current_epoch)
                            rank_zero_only(lambda: print(f'Stage B: validation (epoch {current_epoch}): saved to val/'))()
                    else:
                        self.save_images(all_samples, suffix='epoch%d' % current_epoch)
                        rank_zero_only(lambda: print(f'Stage B: validation image generation (epoch {current_epoch}): saved {len(all_samples)} items x {val_num_samples+1} images to val/'))()
                elif do_gen:
                    rank_zero_only(lambda: print(f'Stage B: validation image generation (epoch {current_epoch}): no samples produced (empty batch or limit=0).'))()
                self.log('val/skip_image_generation', 0.0, on_step=False, on_epoch=True)
            except Exception as e:
                rank_zero_only(lambda: print(f'Stage B: validation image generation failed (epoch {current_epoch}): {e}'))()
                self.log('val/skip_image_generation', 1.0, on_step=False, on_epoch=True)
        else:
            rank_zero_only(lambda: print(f'Stage B: skipping image generation during validation (epoch {current_epoch}).'))()
            self.log('val/skip_image_generation', 1.0, on_step=False, on_epoch=True)
            # Avoid stale _last_val_samples from a previous epoch (would trigger heavy val metrics wrongly)
            setattr(self, '_last_val_samples', None)
            setattr(self, '_last_val_epoch', None)
        
        self.validation_count += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    def get_eval_metric(self, samples, avg=True):
        metric_list = ['mse', 'pcc', 'ssim', 'psm']
        res_list = []
        
        gt_images = [img[0] for img in samples]
        gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
        samples_to_run = np.arange(1, len(samples[0])) if avg else [1]
        for m in tqdm(metric_list, desc='Eval metrics (mse/pcc/ssim/psm)', unit='metric', leave=False):
            res_part = []
            for s in samples_to_run:
                pred_images = [img[s] for img in samples]
                pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
                res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m)
                res_part.append(np.mean(res))
            res_list.append(np.mean(res_part))
        res_part = []
        for s in tqdm(samples_to_run, desc='Eval top-1-class', unit='sample', leave=False):
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

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                first_stage_config,
                cond_stage_config,
                num_timesteps_cond=None,
                cond_stage_key="image",
                cond_stage_trainable=True,
                concat_mode=True,
                cond_stage_forward=None,
                conditioning_key=None,
                scale_factor=1.0,
                scale_by_std=False,
                *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
      
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None  

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True
        self.train_cond_stage_only = False
        self.clip_tune = True
        if self.clip_tune:
            self.image_embedder = FrozenImageEmbedder()
        self.cls_tune = False

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()

    def freeze_diffusion_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_diffusion_model(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_cond_stage(self):
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

    def unfreeze_cond_stage(self):
        for param in self.cond_stage_model.parameters():
            param.requires_grad = True
   

    def freeze_first_stage(self):
        self.first_stage_model.trainable = False
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def unfreeze_first_stage(self):
        self.first_stage_model.trainable = True
        for param in self.first_stage_model.parameters():
            param.requires_grad = True

    def freeze_whole_model(self):
        self.first_stage_model.trainable = False
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_whole_model(self):
        self.first_stage_model.trainable = True
        for param in self.parameters():
            param.requires_grad = True
        
    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                # self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                            force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        # self.cond_stage_model.eval()
        # Ensure conditioner on same device as input (e.g. after validation moved model to CPU)
        if isinstance(c, torch.Tensor):
            self.cond_stage_model.to(c.device)
        # Run conditioner in float32 to avoid AMP Half vs float32 mismatch (MAE patch_embed)
        try:
            autocast_ctx = torch.amp.autocast(
                device_type="cuda" if (isinstance(c, torch.Tensor) and c.is_cuda) else "cpu",
                enabled=False,
            )
        except (AttributeError, TypeError):
            autocast_ctx = torch.cuda.amp.autocast(enabled=False) if (isinstance(c, torch.Tensor) and c.is_cuda) else nullcontext()
        with autocast_ctx:
            if isinstance(c, torch.Tensor) and c.is_floating_point() and c.dtype != torch.float32:
                c = c.float()
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c, re_latent = self.cond_stage_model.encode(c)
            else:
                c, re_latent = self.cond_stage_model(c)
        if not getattr(self, '_logged_cond_shape', False):
            self._logged_cond_shape = True
        return c, re_latent

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        # print('encoder_posterior.shape')
        # print(encoder_posterior.shape)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        # print('z.shape')
        # print(z.shape)
        # print(cond_key)
        # print(self.cond_stage_key)
        # print(cond_key)
        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            # SAR-HM alpha warmup: set effective_alpha_max on cond_stage_model before conditioning
            main_cfg = getattr(self, 'main_config', None)
            if main_cfg and getattr(self, 'cond_stage_model', None):
                csm = self.cond_stage_model
                if getattr(csm, 'use_sarhm', False):
                    warmup = getattr(main_cfg, 'warmup_epochs', 10)
                    start = getattr(main_cfg, 'alpha_max_start', 0.05)
                    end = getattr(main_cfg, 'alpha_max_end', 0.2)
                    cur = getattr(self.trainer, 'current_epoch', 0) if getattr(self, 'trainer', None) else 0
                    if warmup <= 0:
                        csm._effective_alpha_max = end
                    else:
                        frac = min(1.0, cur / warmup)
                        csm._effective_alpha_max = start + frac * (end - start)
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox','fmri', 'eeg']:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            # print('get input')
            # print(not self.cond_stage_trainable)
            # print(force_c_encode)
            if not self.cond_stage_trainable or force_c_encode :
                # print('get learned condition')
                if isinstance(xc, dict) or isinstance(xc, list):
                    # import pudb; pudb.set_trace()
                    c, re_latent = self.get_learned_conditioning(xc)
                    # c = self.get_learned_conditioning(xc)
                else:
                    c, re_latent = self.get_learned_conditioning(xc.to(self.device))
                    # c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        out = [z, c , batch['label'], batch['image_raw']]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):  
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    def _decode_first_stage_allow_grad(self, z):
        """Decode latent to image allowing gradients (for SAR-HM++ generated-image CLIP loss)."""
        z = 1. / self.scale_factor * z
        if isinstance(self.first_stage_model, VQModelInterface):
            return self.first_stage_model.decode(z, force_not_quantize=True)
        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        self.freeze_first_stage()
        csm = getattr(self, 'cond_stage_model', None)
        if csm is not None and hasattr(csm, 'set_batch_context'):
            csm.set_batch_context(batch)
        try:
            x, c, label, image_raw = self.get_input(batch, self.first_stage_key)
            target_embeds = batch.get('target_embeds')
            z_sem_gt = batch.get('z_sem_gt')
            clip_img_embed_gt = batch.get('clip_img_embed_gt')
            summary_embed_gt = batch.get('summary_embed_gt')
            has_semantic_gt = batch.get('has_semantic_gt')
            if self.return_cond:
                loss, cc = self(x, c, label, image_raw, target_embeds=target_embeds, z_sem_gt=z_sem_gt,
                                clip_img_embed_gt=clip_img_embed_gt, summary_embed_gt=summary_embed_gt, has_semantic_gt=has_semantic_gt)
                return loss, cc
            else:
                loss = self(x, c, label, image_raw, target_embeds=target_embeds, z_sem_gt=z_sem_gt,
                            clip_img_embed_gt=clip_img_embed_gt, summary_embed_gt=summary_embed_gt, has_semantic_gt=has_semantic_gt)
                return loss
        finally:
            if csm is not None and hasattr(csm, 'clear_batch_context'):
                csm.clear_batch_context()

    def forward(self, x, c, label, image_raw, target_embeds=None, z_sem_gt=None,
                clip_img_embed_gt=None, summary_embed_gt=None, has_semantic_gt=None, *args, **kwargs):
        # print(self.num_timesteps)
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        # print('t.shape')
        # print(t.shape)
        re_latent = None  # only set when cond_stage_trainable and get_learned_conditioning is called
        if self.model.conditioning_key is not None:
            assert c is not None
            imgs = c
            if self.cond_stage_trainable:
                # c = self.get_learned_conditioning(c)
                c, re_latent = self.get_learned_conditioning(c)
                # print('c.shape')
                # print(c.shape)

        prefix = 'train' if self.training else 'val'
        main_cfg_pp = getattr(self, 'main_config', None)
        need_x0 = (
            main_cfg_pp is not None
            and getattr(main_cfg_pp, 'use_sarhmpp', False)
            and not getattr(main_cfg_pp, 'no_clip_loss', False)
            and clip_img_embed_gt is not None
            and summary_embed_gt is not None
            and (getattr(getattr(self, 'trainer', None), 'global_step', 0) % max(1, getattr(main_cfg_pp, 'clip_loss_every_n_steps', 1)) == 0)
            and (has_semantic_gt is None or (has_semantic_gt.dim() > 0 and has_semantic_gt.any()))
        )
        if need_x0:
            loss, loss_dict, x0_pred = self.p_losses(x, c, t, return_x0_pred=True)
        else:
            loss, loss_dict = self.p_losses(x, c, t)
            x0_pred = None
        # SAR-HM improved: optional scaling of diffusion loss (default 1.0 for backward compat)
        main_cfg = getattr(self, 'main_config', None)
        lam_diff = getattr(main_cfg, 'lambda_diffusion', 1.0) if main_cfg else 1.0
        if lam_diff != 1.0:
            loss = loss * lam_diff
            loss_dict.update({f'{prefix}/loss_diffusion_scaled': loss.detach()})
        if hasattr(self, '_sarhm_metrics_hook') and callable(self._sarhm_metrics_hook):
            try:
                sarhm = self._sarhm_metrics_hook(self.cond_stage_model, label)
                loss_dict.update({f'{prefix}/' + k: v for k, v in sarhm.items()})
            except Exception:
                pass
        # G: Optional L_stable (first N epochs) and L_retrieval (Hopfield modes)
        main_cfg = getattr(self, 'main_config', None)
        if main_cfg and getattr(main_cfg, 'use_sarhm', False):
            extra = getattr(self.cond_stage_model, '_sarhm_extra', {})
            ce = getattr(getattr(self, 'trainer', None), 'current_epoch', 0)
            lam_s = getattr(main_cfg, 'lambda_stable', 0.0)
            stable_epochs = getattr(main_cfg, 'stable_loss_epochs', 5)
            if lam_s > 0 and ce < stable_epochs and 'c_base' in extra and 'c_sar' in extra:
                cb, cs = extra['c_base'], extra['c_sar']
                L_stable = lam_s * (cb - cs).float().pow(2).mean()
                loss = loss + L_stable
                loss_dict.update({f'{prefix}/loss_stable': L_stable})
            lam_r = getattr(main_cfg, 'lambda_retrieval', 0.0)
            if lam_r > 0 and 'logits' in extra and extra['logits'] is not None and label is not None:
                logits = extra['logits']
                num_classes = logits.shape[-1]
                lbl = label.long().to(logits.device).clamp(0, num_classes - 1)
                if lbl.dim() == 0:
                    lbl = lbl.unsqueeze(0)
                L_retrieval = F.cross_entropy(logits, lbl)
                loss = loss + lam_r * L_retrieval
                loss_dict.update({f'{prefix}/loss_retrieval': L_retrieval})
        # SAR-HM++: L_sem_align(q_sem, z_sem_gt) and L_retr(m_sem, z_sem_gt). Training-only; batch must contain z_sem_gt.
        main_cfg_pp = getattr(self, 'main_config', None)
        if main_cfg_pp and getattr(main_cfg_pp, 'use_sarhmpp', False) and z_sem_gt is not None:
            try:
                from sarhm.semantic_losses import compute_semantic_losses
                extra = getattr(self.cond_stage_model, '_sarhm_extra', {})
                q_sem = extra.get('q_sem')
                m_sem = extra.get('m_sem')
                if q_sem is not None and m_sem is not None:
                    z_sem_gt = z_sem_gt.to(q_sem.device).float()
                    if z_sem_gt.dim() == 2 and q_sem.dim() == 2 and z_sem_gt.shape[0] == q_sem.shape[0]:
                        lam_retr = 0.0 if getattr(main_cfg_pp, 'no_retrieval_loss', False) else getattr(main_cfg_pp, 'lambda_retr', 0.1)
                        lam_clip_img = 0.0 if getattr(main_cfg_pp, 'no_clip_loss', False) else getattr(main_cfg_pp, 'lambda_clip_img', 0.2)
                        lam_clip_text = 0.0 if getattr(main_cfg_pp, 'no_clip_loss', False) else getattr(main_cfg_pp, 'lambda_clip_text', 0.1)
                        L_sem_total, sem_loss_dict = compute_semantic_losses(
                            q_sem=q_sem, m_sem=m_sem, z_sem_gt=z_sem_gt,
                            lambda_sem=getattr(main_cfg_pp, 'lambda_sem', 0.1),
                            lambda_retr=lam_retr,
                            lambda_clip_img=lam_clip_img,
                            lambda_clip_text=lam_clip_text,
                        )
                        loss = loss + L_sem_total
                        for k, v in sem_loss_dict.items():
                            loss_dict.update({f'{prefix}/{k}': v})
            except Exception:
                pass
        # SAR-HM++: L_clip_img and L_clip_text from decoded generated image (when x0_pred and GT embeds available)
        if x0_pred is not None and clip_img_embed_gt is not None and summary_embed_gt is not None and main_cfg_pp is not None and getattr(main_cfg_pp, 'use_sarhmpp', False) and not getattr(main_cfg_pp, 'no_clip_loss', False):
            try:
                decoded = self._decode_first_stage_allow_grad(x0_pred)
                if decoded is not None:
                    decoded_01 = (decoded.float() + 1.0).mul(0.5).clamp(0.0, 1.0)
                    if not hasattr(self, '_clip_gen_cache') or self._clip_gen_cache is None:
                        from sarhm.semantic_targets import _get_clip_encoder
                        self._clip_gen_cache = _get_clip_encoder(self.device)
                    clip_model, clip_processor = self._clip_gen_cache
                    from sarhm.semantic_targets import extract_clip_image_embed
                    clip_img_gen = extract_clip_image_embed(decoded_01, clip_model, clip_processor, self.device, allow_grad=True)
                    gt_img = clip_img_embed_gt.to(self.device).float()
                    gt_text = summary_embed_gt.to(self.device).float()
                    if has_semantic_gt is not None and has_semantic_gt.dim() > 0:
                        mask = has_semantic_gt.to(self.device)
                        if mask.any():
                            clip_img_gen = clip_img_gen[mask]
                            gt_img = gt_img[mask]
                            gt_text = gt_text[mask]
                        else:
                            clip_img_gen = None
                    if clip_img_gen is not None and clip_img_gen.shape[0] > 0:
                        from sarhm.semantic_losses import compute_semantic_losses
                        lam_i = getattr(main_cfg_pp, 'lambda_clip_img', 0.2)
                        lam_t = getattr(main_cfg_pp, 'lambda_clip_text', 0.1)
                        L_clip_total, clip_loss_dict = compute_semantic_losses(
                            lambda_clip_img=lam_i, lambda_clip_text=lam_t,
                            clip_img_gen=clip_img_gen, clip_img_gt=gt_img, clip_text_gt=gt_text,
                        )
                        for k, v in clip_loss_dict.items():
                            loss_dict.update({f'{prefix}/{k}': v})
                        loss = loss + L_clip_total
            except Exception:
                pass
        # pre_cls = self.cond_stage_model.get_cls(re_latent)
        # rencon = self.cond_stage_model.recon(re_latent)
        # CLIP loss (EEG->cond vs image embed). To give higher priority to CLIP, increase lambda_clip (e.g. 0.5 or 1.0).
        main_cfg_clip = getattr(self, 'main_config', None)
        lam_clip = getattr(main_cfg_clip, 'lambda_clip', 0.2) if main_cfg_clip else 0.2
        if self.clip_tune and re_latent is not None and not getattr(main_cfg_clip, 'no_clip_loss', False):
            if target_embeds is not None:
                image_embeds = target_embeds.to(self.device) if hasattr(target_embeds, 'to') else target_embeds
            else:
                if hasattr(self, 'image_embedder') and self.image_embedder is not None:
                    image_embeds = self.image_embedder(image_raw)
                else:
                    image_embeds = None
            if image_embeds is not None:
                loss_clip = self.cond_stage_model.get_clip_loss(re_latent, image_embeds)
                loss_dict.update({f'{prefix}/loss_clip_raw': loss_clip})
                loss_clip_weighted = lam_clip * loss_clip
                loss += loss_clip_weighted
                loss_dict.update({f'{prefix}/loss_clip': loss_clip_weighted})
        # Optional: InfoNCE (CLIP space) — requires mapping + target_embeds; B>=2 recommended
        main_cfg_ctr = getattr(self, 'main_config', None)
        lam_ctr = getattr(main_cfg_ctr, 'lambda_contrast', 0.0) if main_cfg_ctr else 0.0
        if (
            lam_ctr > 0
            and re_latent is not None
            and target_embeds is not None
            and hasattr(self.cond_stage_model, 'mapping')
            and self.cond_stage_model.mapping is not None
        ):
            try:
                from sarhm.sarhm_modules import pool_eeg_tokens
                from semantic_losses_extra import infonce_clip_style
                pooled = pool_eeg_tokens(re_latent, self.cond_stage_model.global_pool)
                q = self.cond_stage_model.mapping(pooled)
                k = target_embeds.to(q.device).float()
                temp = float(getattr(main_cfg_ctr, 'contrast_temperature', 0.07))
                if q.shape[0] > 1 and q.shape[0] == k.shape[0]:
                    L_ctr = infonce_clip_style(q, k, temperature=temp)
                    loss = loss + lam_ctr * L_ctr
                    loss_dict.update({f'{prefix}/loss_contrast': L_ctr})
            except Exception:
                pass
        if self.cls_tune and re_latent is not None and hasattr(self.cond_stage_model, 'cls_net') and self.cond_stage_model.cls_net is not None:
            pre_cls = self.cond_stage_model.get_cls(re_latent)
            loss_cls = self.cls_loss(label, pre_cls)
            loss += loss_cls
            loss_dict.update({f'{prefix}/loss_cls': loss_cls})
                # if self.return_cond:
                    # return self.p_losses(x, c, t, *args, **kwargs), c
        # return self.p_losses(x, c, t, *args, **kwargs)
        if self.return_cond:
            return loss, loss_dict, c
        return loss, loss_dict
    # def recon_loss(self, )
    def recon_loss(self, imgs, pred):
        """
        imgs: [N, 1, num_voxels]
        pred: [N, L, p]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # target = self.patchify(imgs)

        loss = (pred - imgs) ** 2
        loss = loss.mean()
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        # loss = (loss * mask).sum() / mask.sum()  if mask.sum() != 0 else (loss * mask).sum() # mean loss on removed patches
        return loss
    def cls_loss(self, label, pred):
        return torch.nn.CrossEntropyLoss()(pred, label)

    def _rescale_annotations(self, bboxes, crop_coordinates):  # TODO: move to dataset
        def rescale_bbox(bbox):
            x0 = torch.clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = torch.clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return x0, y0, w, h

        return [rescale_bbox(b) for b in bboxes]

    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond)
        # print('x_recon')
        # if isinstance(x_recon, tuple):
        #     print('is tuple')
        #     # print(len(x_recon))
        #     # print(x_recon[0].shape)
        # else:
        #     print(x_recon.shape)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, t, noise=None, return_x0_pred=False):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        t_cpu = t.cpu()
        logvar_t = self.logvar[t_cpu].to(t.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t_cpu].to(loss_vlb.device) * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        if return_x0_pred:
            if self.parameterization == "eps":
                x0_pred = self.predict_start_from_noise(x_noisy, t=t, noise=model_output)
            else:
                x0_pred = model_output
            return loss, loss_dict, x0_pred
        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim, ddim_steps,**kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates


    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                             ddim_steps=ddim_steps,eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):

                    samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        if self.train_cond_stage_only:
            print(f"{self.__class__.__name__}: Only optimizing conditioner params!")
            # Cond_parms from diffusion model only (attn2, time_embed, norm2) to avoid duplicate params with cond_stage_model
            cond_parms = [p for n, p in self.model.named_parameters()
                         if 'attn2' in n or 'time_embed_condtion' in n or 'norm2' in n]
            params = list(self.cond_stage_model.parameters()) + cond_parms
            # Deduplicate by id() in case of any overlap
            seen = set()
            params = [p for p in params if id(p) not in seen and not seen.add(id(p))]
        
            for p in params:
                p.requires_grad = True

        else:
            params = list(self.model.parameters())
            if self.cond_stage_trainable:
                print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
                params = params + list(self.cond_stage_model.parameters())
            if self.learn_logvar:
                print('Diffusion model optimizing logvar')
                params.append(self.logvar)

        opt = torch.optim.AdamW(params, lr=lr)

        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
            
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + [c_concat], dim=1)
            cc = torch.cat([c_crossattn], dim=1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


class EEGClassifier(pl.LightningModule):
    """main class"""
    def __init__(self,
                first_stage_config,
                cond_stage_config,
                num_timesteps_cond=None,
                cond_stage_key="image",
                cond_stage_trainable=True,
                concat_mode=True,
                cond_stage_forward=None,
                conditioning_key=None,
                scale_factor=1.0,
                scale_by_std=False,
                *args, **kwargs):
        super().__init__()
        # self.use_scheduler = scheduler_config is not None
        # if self.use_scheduler:
        #     self.scheduler_config = scheduler_config
        self.cond_stage_trainable = True
        self.main_config = None
        self.best_val = 0.0 
        self.cond_stage_model = None
        self.validation_count = 0
        
    def forward(self, x, c, label, image_raw, *args, **kwargs):
        # print(self.num_timesteps)
        # t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        # print('t.shape')
        # print(t.shape)
        # if self.model.conditioning_key is not None:
        #     assert c is not None
        #     imgs = c
        #     if self.cond_stage_trainable:
                # c = self.get_learned_conditioning(c)
        c, re_latent = self.get_learned_conditioning(c)
                # print('c.shape')
                # print(c.shape)

        prefix = 'train' if self.training else 'val'
        # loss, loss_dict = self.p_losses(x, c, t, *args, **kwargs)
        pre_cls = self.cond_stage_model.get_cls(re_latent)

        loss = self.cls_loss(label, pre_cls)

        loss_dict = {}
        loss_dict.update({f'{prefix}/loss_cls': loss})
        # rencon = self.cond_stage_model.recon(re_latent)
        if self.clip_tune:
            image_embeds = self.image_embedder(image_raw)
            loss_clip = self.cond_stage_model.get_clip_loss(re_latent, image_embeds)
        # loss_recon = self.recon_loss(imgs, rencon)
        
            loss += loss_clip
        # loss += loss_cls # loss_recon +  #(self.original_elbo_weight * loss_vlb)
        # loss_dict.update({f'{prefix}/loss_recon': loss_recon})
        # loss_dict.update({f'{prefix}/loss_cls': loss_cls})
            loss_dict.update({f'{prefix}/loss_clip': loss_clip})
                # if self.return_cond:
                    # return self.p_losses(x, c, t, *args, **kwargs), c
        # return self.p_losses(x, c, t, *args, **kwargs)
        # if self.return_cond:
        #     return loss, loss_dict, c
        return loss, loss_dict

    def shared_step(self, batch):
        x,c, label, image_raw  = self.get_input(batch)
        loss, loss_dict = self(x,c, label, image_raw)
        return loss, loss_dict

    def cls_loss(self, label, pred):
        return torch.nn.CrossEntropyLoss()(pred, label)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        self.train()
        self.cond_stage_model.train()  ###到底是在哪里训练的

        # Optional: skip real training for epochs < start_epoch (e.g. --start_epoch 2 to test epoch 2 only)
        start_epoch = getattr(getattr(self, 'main_config', None), 'start_epoch', 0) or 0
        if start_epoch > 0 and getattr(self, 'trainer', None) is not None and self.trainer.current_epoch < start_epoch:
            loss = _safe_zero_loss_for_backward(self, self.device)
            ld = loss.detach() if getattr(loss, 'requires_grad', False) else loss
            self.log_dict({'train/loss': ld, 'train/loss_simple': ld, 'train/loss_vlb': ld, 'train/loss_clip': ld}, prog_bar=True, logger=True, on_step=False, on_epoch=True)
            return loss

        # C1: Unfreeze SAR-HM prototypes after proto_freeze_epochs
        if not getattr(self, "_proto_unfrozen", True) and getattr(self, "_proto_freeze_epochs", 0) > 0:
            ce = getattr(self.trainer, "current_epoch", 0)
            if ce >= self._proto_freeze_epochs:
                csm = getattr(self, "cond_stage_model", None)
                if csm is not None and getattr(csm, "sarhm_prototypes", None):
                    csm.sarhm_prototypes.prototypes.requires_grad = True
                    self._proto_unfrozen = True
                    print("SAR-HM: prototypes unfrozen at epoch %d" % ce)
        
        try:
            loss, loss_dict = self.shared_step(batch)
        except Exception as e:
            import traceback
            rank_zero_only(lambda: print(f"[EEGClassifier] training_step shared_step failed (epoch {getattr(self.trainer, 'current_epoch', '?')} batch {batch_idx}), using safe zero loss and continuing: {e}"))()
            rank_zero_only(lambda: traceback.print_exc())()
            loss = _safe_zero_loss_for_backward(self, self.device)
            ld = loss.detach() if getattr(loss, 'requires_grad', False) else loss
            self.log_dict({'train/loss': ld, 'train/loss_simple': ld, 'train/loss_vlb': ld, 'train/loss_clip': ld}, prog_bar=True, logger=True, on_step=False, on_epoch=True)
            return loss

        if not _loss_ok_for_backward(loss):
            rank_zero_only(lambda: print(f"[EEGClassifier] loss has no grad_fn (epoch {getattr(self.trainer, 'current_epoch', '?')} batch {batch_idx}), using safe zero loss and continuing"))()
            loss = _safe_zero_loss_for_backward(self, self.device)
            ld = loss.detach() if getattr(loss, 'requires_grad', False) else loss
            self.log_dict({'train/loss': ld, 'train/loss_simple': ld, 'train/loss_vlb': ld, 'train/loss_clip': ld}, prog_bar=True, logger=True, on_step=False, on_epoch=True)
            return loss

        # B2: Hopfield stats every 50 steps when SAR-HM is on (professor-friendly evidence)
        if getattr(self, "main_config", None) and getattr(self.main_config, "use_sarhm", False):
            try:
                from sarhm.metrics_logger import log_hopfield_stats_once
                extra = getattr(self.cond_stage_model, "_sarhm_extra", {})
                gs = getattr(self, "global_step", 0)
                log_hopfield_stats_once(
                    extra, batch.get("label"), step=gs, log_every=50, is_smoke_test=False
                )
            except Exception:
                pass

        self.log_dict(loss_dict, prog_bar=True,
                    logger=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        # if self.train_cond_stage_only:
        #     print(f"{self.__class__.__name__}: Only optimizing conditioner params!")
        #     cond_parms = [p for n, p in self.named_parameters() 
        #             if 'attn2' in n or 'time_embed_condtion' in n or 'norm2' in n]
        #     # cond_parms = [p for n, p in self.named_parameters() 
        #             # if 'time_embed_condtion' in n]
        #     # cond_parms = []
            
        params = list(self.cond_stage_model.parameters()) # + cond_parms
        
        for p in params:
            p.requires_grad = True

        # else:
        #     params = list(self.model.parameters())
        #     if self.cond_stage_trainable:
        #         print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
        #         params = params + list(self.cond_stage_model.parameters())
        #     if self.learn_logvar:
        #         print('Diffusion model optimizing logvar')
        #         params.append(self.logvar)

        opt = torch.optim.AdamW(params, lr=lr)

        # if self.use_scheduler:
        #     assert 'target' in self.scheduler_config
        #     scheduler = instantiate_from_config(self.scheduler_config)

        #     print("Setting up LambdaLR scheduler...")
        #     scheduler = [
        #         {
        #             'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
        #             'interval': 'step',
        #             'frequency': 1
        #         }]
        #     return [opt], scheduler
            
        return opt

    @torch.no_grad()
    def get_input(self, batch, k='image', return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        # x = super().get_input(batch, k)
        x = batch['image']
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        # print('z.shape')
        # print(z.shape)
        # print(cond_key)
        # print(self.cond_stage_key)
        # print(cond_key)
        xc = batch['eeg']
        c = xc
        # if self.model.conditioning_key is not None:
        #     if cond_key is None:
        #         cond_key = self.cond_stage_key
        #     if cond_key != self.first_stage_key:
        #         if cond_key in ['caption', 'coordinates_bbox','fmri', 'eeg']:
        #             xc = batch[cond_key]
        #         elif cond_key == 'class_label':
        #             xc = batch
        #         else:
        #             xc = super().get_input(batch, cond_key).to(self.device)
        #     else:
        #         xc = x
        #     # print('get input')
        #     # print(not self.cond_stage_trainable)
        #     # print(force_c_encode)
        #     if not self.cond_stage_trainable or force_c_encode :
        #         # print('get learned condition')
        #         if isinstance(xc, dict) or isinstance(xc, list):
        #             # import pudb; pudb.set_trace()
        #             c, re_latent = self.get_learned_conditioning(xc)
        #             # c = self.get_learned_conditioning(xc)
        #         else:
        #             c, re_latent = self.get_learned_conditioning(xc.to(self.device))
        #             # c = self.get_learned_conditioning(xc.to(self.device))
        #     else:
        #         c = xc
        #     if bs is not None:
        #         c = c[:bs]

        #     if self.use_positional_encodings:
        #         pos_x, pos_y = self.compute_latent_shifts(batch)
        #         ckey = __conditioning_keys__[self.model.conditioning_key]
        #         c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        # else:
        #     c = None
        #     xc = None
        #     if self.use_positional_encodings:
        #         pos_x, pos_y = self.compute_latent_shifts(batch)
        #         c = {'pos_x': pos_x, 'pos_y': pos_y}
        out = [x, c , batch['label'], batch['image_raw']]
        # if return_first_stage_outputs:
        #     xrec = self.decode_first_stage(z)
        #     out.extend([x, xrec])
        # if return_original_cond:
        #     out.append(xc)
        return out


    @torch.no_grad()

    def accuracy(self, output, target, topk=(1, )):

        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        print('val step')
        print('batch_idx:', batch_idx)
        # if batch_idx != 0:
        #     return
        
        if self.validation_count % 1 == 0 and self.trainer.current_epoch != 0:
            self.full_validation(batch)
        # else:
        #     # pass
        #     grid, all_samples, state = self.generate(batch, ddim_steps=self.ddim_steps, num_samples=3, limit=5)
        #     metric, metric_list = self.get_eval_metric(all_samples, avg=self.eval_avg)
        #     grid_imgs = Image.fromarray(grid.astype(np.uint8))
        #     # self.logger.log_image(key=f'samples_test', images=[grid_imgs])
        #     metric_dict = {f'val/{k}':v for k, v in zip(metric_list, metric)}
        #     # self.logger.log_metrics(metric_dict)
        #     if metric[-1] > self.run_full_validation_threshold:
        #         self.full_validation(batch, state=state)
        self.validation_count += 1


    def full_validation(self, batch, state=None):
        print('###### run full validation! ######\n')
        c = batch['eeg']

        c, re_latent = self.get_learned_conditioning(c)

        # Save checkpoint_best only when classification head is available (cls_tune=True).
        # With cls_tune=False, get_cls would raise; save best at end of Stage B instead (see finetune).
        if getattr(self, 'cls_tune', False) and getattr(self.cond_stage_model, 'cls_net', None) is not None:
            pre_cls = self.cond_stage_model.get_cls(re_latent)
            acc1, acc5 = self.accuracy(pre_cls, batch['label'], topk=(1, 5))
            print(acc1, acc5)
            if acc1[0] > self.best_val:
                self.best_val = acc1[0]
                torch.save(
                    {
                        'model_state_dict': self.state_dict(),
                        'config': self.main_config,
                        'state': state

                    },
                    os.path.join(self.output_path, 'checkpoint_best.pth')
                )
        else:
            print('Stage B: skipping classification-based best checkpoint (cls_tune=False). checkpoint_best.pth will be written at end of training if disable_image_generation_in_val=True.\n')
    def get_learned_conditioning(self, c):
        # self.cond_stage_model.eval()
        if isinstance(c, torch.Tensor):
            self.cond_stage_model.to(c.device)
        try:
            autocast_ctx = torch.amp.autocast(
                device_type="cuda" if (isinstance(c, torch.Tensor) and c.is_cuda) else "cpu",
                enabled=False,
            )
        except (AttributeError, TypeError):
            autocast_ctx = torch.cuda.amp.autocast(enabled=False) if (isinstance(c, torch.Tensor) and c.is_cuda) else nullcontext()
        with autocast_ctx:
            if isinstance(c, torch.Tensor) and c.is_floating_point() and c.dtype != torch.float32:
                c = c.float()
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c, re_latent = self.cond_stage_model.encode(c)
            else:
                c, re_latent = self.cond_stage_model(c)
        if not getattr(self, '_logged_cond_shape', False):
            self._logged_cond_shape = True
        return c, re_latent
