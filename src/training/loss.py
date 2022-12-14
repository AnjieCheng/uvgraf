# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""
from typing import Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from src.torch_utils import training_stats
from src.torch_utils.ops import conv2d_gradfix
from src.torch_utils.ops import upfirdn2d
from src.training.training_utils import sample_patch_params, extract_patches, linear_schedule
from torch.autograd import grad
# from chamferdist import ChamferDistance

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]
    return grad


#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, *args, **kwargs): # to be overridden by subclass
        raise NotImplementedError()

    def progressive_update(self, *args, **kwargs):
        pass

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, cfg, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_batch_shrink=2, pl_decay=0.01):
        super().__init__()
        self.cfg                = cfg
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = cfg.model.loss_kwargs.get('blur_init_sigma', 0)
        self.blur_fade_kimg     = cfg.model.loss_kwargs.get('blur_fade_kimg', 0)
        self.patch_cfg          = OmegaConf.to_container(OmegaConf.create({**self.cfg.training.patch})) # For faster field access

        self.progressive_update(0)

    def progressive_update(self, cur_kimg: int):
        if self.patch_cfg['enabled']:
            if self.patch_cfg['distribution'] in ('uniform', 'discrete_uniform'):
                self.patch_cfg['min_scale'] = linear_schedule(cur_kimg, self.patch_cfg['max_scale'], self.patch_cfg['min_scale_trg'], self.patch_cfg['anneal_kimg'])
            elif self.patch_cfg['distribution'] == 'beta':
                self.patch_cfg['beta'] = linear_schedule(cur_kimg, self.patch_cfg['beta_val_start'], self.patch_cfg['beta_val_end'], self.patch_cfg['anneal_kimg'])
                self.patch_cfg['min_scale'] = self.patch_cfg['min_scale_trg']
            else:
                raise NotImplementedError(f"Uknown patch distribution: {self.patch_cfg['distribution']}")
        self.gpc_spoof_p = linear_schedule(cur_kimg, 1.0, self.cfg.model.generator.camera_cond_spoof_p, 1000)

    def run_G(self, z, c, camera_angles, camera_angles_cond=None, update_emas=False):
        ws = self.G.mapping(z=z, c=c, camera_angles=camera_angles_cond, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(z=torch.randn_like(z), c=c, camera_angles=camera_angles_cond, update_emas=False)[:, cutoff:]
        patch_params = sample_patch_params(len(z), self.patch_cfg, device=z.device) if self.patch_cfg['enabled'] else {}
        patch_kwargs = dict(patch_params=patch_params) if self.patch_cfg['enabled'] else None
        img = self.G.synthesis(ws, camera_angles, update_emas=update_emas, **patch_kwargs)
        return img, ws, patch_params

    def run_D(self, in_img, blur_sigma=0, update_emas=False, **kwargs):
        img = in_img[:,:3,...]
        mask = in_img[:,3:4,...]
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img, num_frames=img.shape[1] // self.G.img_channels) # [batch_size, c * 2, h, w]
        logits, logits_mask = self.D(torch.cat([img,mask], dim=1), update_emas=update_emas, **kwargs)
        return logits, logits_mask

    def extract_patches(self, img: torch.Tensor):
        patch_params = sample_patch_params(len(img), self.patch_cfg, device=img.device)
        img = extract_patches(img, patch_params, resolution=self.patch_cfg['resolution']) # [batch_size, c, h_patch, w_patch]

        return img, patch_params

    def accumulate_gradients(self, phase, real_img, real_c, real_camera_angles, gen_z, gen_c, gen_camera_angles, gen_camera_angles_cond, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg_pl', 'Gall', 'Dmain', 'Dreg', 'Dall']
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dall': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Greg_mvc', 'Gall']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws, patch_params = self.run_G(gen_z, gen_c, gen_camera_angles, camera_angles_cond=gen_camera_angles_cond)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, patch_params=patch_params, camera_angles=gen_camera_angles)


                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg_pl', 'Gall'] and self.cfg.model.loss_kwargs.pl_weight > 0:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws, _patch_params = self.run_G(gen_z[:batch_size], gen_c[:batch_size], gen_camera_angles[:batch_size], camera_angles_cond=gen_camera_angles_cond[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.cfg.model.loss_kwargs.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dall']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                with torch.no_grad():
                    gen_img, _gen_ws, patch_params = self.run_G(gen_z, gen_c, gen_camera_angles, camera_angles_cond=gen_camera_angles_cond, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True, patch_params=patch_params, camera_angles=gen_camera_angles)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dall']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                (real_img, patch_params) = self.extract_patches(real_img) if self.patch_cfg['enabled'] else (real_img, None)
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dall'])
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma, patch_params=patch_params, camera_angles=real_camera_angles)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dall']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dall']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/D/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------

class CanonicalStyleGAN2Loss(StyleGAN2Loss):
    def run_G(self, z, camera_angles, points, camera_angles_cond=None, update_emas=False, verbose=False):
        geo_z = z[...,:self.G.geo_dim]
        tex_z = z[...,-self.G.tex_dim:]
        geo_ws = self.G.geo_mapping(geo_z, camera_angles=camera_angles_cond, update_emas=update_emas)
        tex_ws = self.G.tex_mapping(tex_z, camera_angles=camera_angles_cond, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                geo_cutoff = torch.empty([], dtype=torch.int64, device=geo_ws.device).random_(1, geo_ws.shape[1])
                geo_cutoff = torch.where(torch.rand([], device=geo_ws.device) < self.style_mixing_prob, geo_cutoff, torch.full_like(geo_cutoff, geo_ws.shape[1]))
                geo_ws[:, geo_cutoff:] = self.G.geo_mapping(z=torch.randn_like(geo_z), camera_angles=camera_angles_cond, update_emas=False)[:, geo_cutoff:]
                tex_cutoff = torch.empty([], dtype=torch.int64, device=geo_ws.device).random_(1, tex_ws.shape[1])
                tex_cutoff = torch.where(torch.rand([], device=geo_ws.device) < self.style_mixing_prob, tex_cutoff, torch.full_like(tex_cutoff, tex_ws.shape[1]))
                tex_ws[:, tex_cutoff:] = self.G.tex_mapping(z=torch.randn_like(tex_z), camera_angles=camera_angles_cond, update_emas=False)[:, tex_cutoff:]
        
        patch_params = sample_patch_params(len(z), self.patch_cfg, device=z.device) if self.patch_cfg['enabled'] else {}
        patch_kwargs = dict(patch_params=patch_params) if self.patch_cfg['enabled'] else None

        if verbose:
            img, G_info = self.G.synthesis(geo_ws, tex_ws, camera_angles, points=points, update_emas=update_emas, verbose=True, **patch_kwargs)
            return img, [geo_ws, tex_ws], patch_params, G_info
        else:
            img = self.G.synthesis(geo_ws, tex_ws, camera_angles, points=points, update_emas=update_emas, verbose=False, **patch_kwargs)
            return img, [geo_ws, tex_ws], patch_params

    def accumulate_gradients(self, phase, real_img, real_camera_angles, gen_z, gen_camera_angles, gen_camera_angles_cond, gain, cur_nimg, points):
        with torch.autograd.set_detect_anomaly(True):
            assert phase in ['Gmain', 'Greg_pl', 'Gall', 'Dmain', 'Dreg', 'Dall']
            if self.r1_gamma == 0:
                phase = {'Dreg': 'none', 'Dall': 'Dmain'}.get(phase, phase)
            blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

            # Gmain: Maximize logits for generated images.
            if phase in ['Gmain', 'Greg_mvc', 'Gall']:
                with torch.autograd.profiler.record_function('Gmain_forward'):
                    gen_img, _gen_ws, patch_params, info = self.run_G(gen_z, gen_camera_angles, camera_angles_cond=gen_camera_angles_cond, verbose=True, points=points)
                    gen_logits = self.run_D(gen_img, blur_sigma=blur_sigma, patch_params=patch_params, camera_angles=gen_camera_angles)
                    gen_logits, gen_logits_mask = gen_logits
                    
                    training_stats.report('Loss/scores/fake', gen_logits)
                    training_stats.report('Loss/signs/fake', gen_logits.sign())
                    loss_Grgb = torch.nn.functional.softplus(-gen_logits).mean()
                    training_stats.report('Loss/G/loss_rgb', loss_Grgb)

                    # training_stats.report('Loss/scores/fake_mask', gen_logits_mask)
                    # training_stats.report('Loss/signs/fake_mask', gen_logits_mask.sign())
                    # loss_Gmask = torch.nn.functional.softplus(-gen_logits_mask).mean()
                    # training_stats.report('Loss/G/loss_mask', loss_Gmask)
                    loss_Gmain = loss_Grgb # + loss_Gmask
                    training_stats.report('Loss/G/loss', loss_Gmain)

                    # loss_Ginverse = torch.nn.functional.mse_loss(info['uniform_coords'], info['uniform_coords_b'], reduction='none')
                    # # Todo: (loss_Ginverse * info['uniform_coords_sigmas'].detach()).mean()
                    # # maybe better way is to sample high density points from canonical volume and --> f(x) --> f'(x)
                    # loss_Ginverse = loss_Ginverse.mean() # (info['uniform_coords_sigmas']
                    # training_stats.report('Loss/G/inverse', loss_Ginverse)
                    # training_stats.report('scalar/G/template/beta', self.G.synthesis.template.density.beta)

                    # # sdf_grad = gradient(info['all_sdf'], info['all_points'])
                    # loss_Gsdf = torch.abs(info['all_sdf'].norm(dim=-1) - 1) # ((sdf_grad.norm(2, dim=-1) - 1) ** 2).mean()
                    # training_stats.report('Loss/G/sdf', loss_Gsdf)

                    # chamferDist = ChamferDistance()
                    # loss_Gcoverage = chamferDist(info['uniform_coords'], info['uniform_coords_f'], bidirectional=True).mean()
                    # training_stats.report('Loss/G/coverage', loss_Gcoverage)

                with torch.autograd.profiler.record_function('Gmain_backward'):
                    (loss_Gmain.mean()).mul(gain).backward() #  +loss_Gsdf.mean()+loss_Gcoverage.mean()

        #TODO: Apply path length regularization.

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dall']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                with torch.no_grad():
                    gen_img, _gen_ws, patch_params = self.run_G(gen_z, gen_camera_angles, camera_angles_cond=gen_camera_angles_cond, update_emas=True, points=points)
                gen_logits = self.run_D(gen_img, blur_sigma=blur_sigma, update_emas=True, patch_params=patch_params, camera_angles=gen_camera_angles)
                gen_logits, gen_logits_mask = gen_logits
                
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits).mean()  # -log(1 - sigmoid(gen_logits))
                training_stats.report('Loss/D/loss_genrgb', loss_Dgen)

                # training_stats.report('Loss/scores/fake_mask', gen_logits_mask)
                # training_stats.report('Loss/signs/fake_mask', gen_logits_mask.sign())
                # loss_Dgen_mask = torch.nn.functional.softplus(gen_logits_mask).mean()  # -log(1 - sigmoid(gen_logits))
                # training_stats.report('Loss/D/loss_gen_mask', loss_Dgen_mask)
                # loss_Dgen += loss_Dgen_mask

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        with torch.autograd.set_detect_anomaly(True):
            if phase in ['Dmain', 'Dreg', 'Dall']:
                name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
                with torch.autograd.profiler.record_function(name + '_forward'):
                    (real_img, patch_params) = self.extract_patches(real_img) if self.patch_cfg['enabled'] else (real_img, None)
                    real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dall'])
                    real_logits = self.run_D(real_img_tmp, blur_sigma=blur_sigma, patch_params=patch_params, camera_angles=real_camera_angles)
                    real_logits, real_logits_mask = real_logits
                    
                    training_stats.report('Loss/scores/real', real_logits)
                    training_stats.report('Loss/signs/real', real_logits.sign())

                    # training_stats.report('Loss/scores/real_mask', real_logits_mask)
                    # training_stats.report('Loss/signs/real_mask', real_logits_mask.sign())

                    loss_Dreal = 0
                    if phase in ['Dmain', 'Dall']:
                        loss_Dreal = torch.nn.functional.softplus(-real_logits).mean() # -log(sigmoid(real_logits))
                        training_stats.report('Loss/D/loss_real_rgb', loss_Dreal)
                        
                        # loss_Dreal_mask = torch.nn.functional.softplus(-real_logits_mask).mean()  # -log(sigmoid(real_logits))
                        # training_stats.report('Loss/D/loss_real_mask', loss_Dreal_mask)
                        # loss_Dreal += loss_Dreal_mask
                        training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                    loss_Dr1 = 0
                    if phase in ['Dreg', 'Dall']:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                        r1_penalty = r1_grads.square().sum([1,2,3])
                        loss_Dr1 = r1_penalty.mean() * (self.r1_gamma / 2)
                        training_stats.report('Loss/D/r1_penalty', r1_penalty)
                        training_stats.report('Loss/D/reg', loss_Dr1)

                        # # Compute R1 regularization for discriminator of Mask image
                        # with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        #     r1_grads_mask = \
                        #         torch.autograd.grad(
                        #             outputs=[real_logits_mask.sum()], inputs=[real_img_tmp], create_graph=True,
                        #             only_inputs=True)[0]

                        # r1_penalty_mask = r1_grads_mask.square().sum([1, 2, 3])
                        # loss_Dr1_mask = r1_penalty_mask.mean() * (self.r1_gamma / 2)
                        # training_stats.report('Loss/r1_penalty_mask', r1_penalty_mask)
                        # training_stats.report('Loss/D/reg_mask', loss_Dr1_mask)
                        # loss_Dr1 += loss_Dr1_mask

                with torch.autograd.profiler.record_function(name + '_backward'):
                    (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------