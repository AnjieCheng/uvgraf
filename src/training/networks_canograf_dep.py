import math
from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.torch_utils import misc
from src.torch_utils import persistence
from omegaconf import DictConfig

# import frnn
from pytorch3d.ops import knn_points
from src.training.networks_geometry import FoldSDF
from src.training.networks_stylegan2 import SynthesisBlock
from src.training.networks_stylegan3 import SynthesisNetwork as SG3SynthesisNetwork
from src.training.networks_inr_gan import SynthesisNetwork as INRSynthesisNetwork
from src.training.layers import (
    FullyConnectedLayer,
    MappingNetwork,
    ScalarEncoder1d,
)
from src.training.rendering import (
    fancy_integration,
    get_initial_rays_trig,
    transform_points,
    sample_pdf,
    compute_cam2world_matrix,
)
from src.training.training_utils import *


@misc.profiled_function
def canonical_renderer_pretrain(tex_x: torch.Tensor, geo_x: torch.Tensor, geo_ws: torch.Tensor,
                                coords: torch.Tensor, ray_d_world: torch.Tensor,  sdf_grid: torch.Tensor,
                                folding_grid: torch.Tensor, folding_coords: torch.Tensor, folding_normals: torch.Tensor,
                                mlp_f: Callable, mlp_b: Callable, template: Callable, sanity_mlp: Callable,
                                texture_mlp: Callable, geo_mlp: Callable=None, scale: float=1.0, 
                                rt_sdf: bool=False, rt_radiance: bool=True) -> torch.Tensor:
    # geo
    batch_size, raw_feat_dim, h, w = geo_x.shape
    num_points = coords.shape[1]

    coords_normed = coords / 0.5
    sdfs = F.grid_sample(sdf_grid, coords_normed.view(batch_size, 1, 1, num_points, 3), padding_mode="border").view(batch_size, num_points, 1)
    sigmas = torch.sigmoid(-sdfs / 0.005) / 0.005
    
    K = 16
    dis, indices, _ = knn_points(coords.detach(), folding_coords.detach(), K=K)
    dis = dis.detach()
    indices = indices.detach()
    dis = dis.sqrt()
    weights = 1 / (dis + 1e-7)
    # weights = weights / torch.sum(weights, dim=-1, keepdims=True)  # (1,M,K)
    num_of_grids = folding_grid.shape[1]
    sphere_samples_normed = F.normalize(folding_grid, dim=-1)
    sphere_u = (0.5 + torch.atan2(sphere_samples_normed[..., 0], sphere_samples_normed[..., 2]) / (2 * math.pi)).transpose(0,1)
    sphere_v = (0.5 - torch.asin(sphere_samples_normed[..., 1]) / math.pi).transpose(0,1)
    sphere_uv = torch.cat([sphere_u,sphere_v], dim=1).view(batch_size, 1, num_of_grids, 2)
    sphere_uv = torch.cat([sphere_u,sphere_v], dim=1).view(batch_size, 1, num_of_grids, 2)
    sphere_uv = (sphere_uv * 2) - 1
    sphere_uv_f = F.grid_sample(tex_x, grid=sphere_uv, mode='bilinear', align_corners=True).view(batch_size, 32, num_of_grids) # [batch_size, 3, feat_dim, num_points]
    sphere_uv_f = sphere_uv_f.permute(0, 2, 1) # [batch_size, num_points, feat_dim]

    # sphere_uv_f = get_feat_from_triplane(coords, tex_x, scale=None).mean(dim=1)
    fused_tex_feature = torch.sum(batched_index_select(sphere_uv_f, 1, indices).view(batch_size, num_points, K, 32) * weights[..., None], dim=-2)
    fused_tex_normal = torch.sum(batched_index_select(folding_normals, 1, indices).view(batch_size, num_points, K, 3) * weights[..., None], dim=-2)

    # volume rgbs
    # sphere_tex_feat = get_feat_from_triplane(coords, tex_x, scale=None)
    rgbs = texture_mlp(fused_tex_feature, fused_tex_normal.detach(), sdfs.detach()) # [batch_size, num_points, out_dim]

    return torch.cat([rgbs, sigmas.detach()], dim=-1)

@misc.profiled_function
def canonical_renderer(tex_x: torch.Tensor, geo_x: torch.Tensor, 
                       coords: torch.Tensor, ray_d_world: torch.Tensor, 
                       mlp_f: Callable, mlp_b: Callable, template: Callable, 
                       texture_mlp: Callable, geo_mlp: Callable=None, scale: float=1.0, rt_sdf: bool=False) -> torch.Tensor:
    # geo
    # with torch.no_grad():
    assert geo_x.shape[1] % 3 == 0, f"We use 3 planes: {geo_x.shape}"
    batch_size, raw_feat_dim, h, w = geo_x.shape
    misc.assert_shape(coords, [batch_size, None, 3])

    geo_feat = get_feat_from_triplane(coords, geo_x, scale=scale)
        
    # if geo_x.requires_grad:
    with torch.set_grad_enabled(True):
        coords_org = coords.clone().detach().requires_grad_(True)
        coords = coords_org
        coords_f = F.tanh(mlp_f(geo_feat, coords, ray_d_world)) # [batch_size, num_points, out_dim]
        sigmas, sdfs = template(coords_f, get_sdf=True)
    sdf_grad = gradient(sdfs, coords)

    geo_feat_f = get_feat_from_triplane(coords_f, geo_x, scale=None)
    coords_b = F.tanh(mlp_b(geo_feat_f, coords_f, ray_d_world)) * scale # F.tanh() * scale

    # uv rgbs (starts)
    sphere_samples = sample_sphere_points(radius=0.3, num_points=256)
    batched_sphere_samples = sphere_samples.unsqueeze(0).expand(batch_size, -1, -1).to(coords.device)
    sphere_tex_feat = get_feat_from_triplane(batched_sphere_samples, tex_x, scale=0.5)
    fused_feat = get_nn_fused_feats(coords_f, batched_sphere_samples, sphere_tex_feat, K=4)
    
    rgbs = texture_mlp(fused_feat, sdf_grad.detach(), sdfs.detach()) # [batch_size, num_points, out_dim]

    if geo_x.requires_grad:
        return torch.cat([rgbs, sigmas, coords_f, coords_b, sdf_grad], dim=-1) # , uv.squeeze(1)
    else:
        if rt_sdf:
            return torch.cat([rgbs, sdfs, coords_f, coords_b, coords_b], dim=-1) # , uv.squeeze(1)
        else:
            return torch.cat([rgbs, sigmas, coords_f, coords_b, coords_b], dim=-1) # , uv.squeeze(1)

#----------------------------------------------------------------------------

@persistence.persistent_class
class SanityMLP(nn.Module):
    def __init__(self, out_dim: int=3):
        super().__init__()
        hid_dim = 64
        n_layers = 2
        self.out_dim = out_dim
        backbone_input_dim = 3
        backbone_out_dim = self.out_dim
        self.dims = [backbone_input_dim] + [hid_dim] * (n_layers - 1) + [backbone_out_dim] # (n_hid_layers + 2)
        activations = ['lrelu'] * (len(self.dims) - 2) + ['linear']
        assert len(self.dims) > 2, f"We cant have just a linear layer here: nothing to modulate. Dims: {self.dims}"
        layers = [FullyConnectedLayer(self.dims[i], self.dims[i+1], activation=a) for i, a in enumerate(activations)]
        self.model = nn.Sequential(*layers)

        self.ray_dir_enc = None
        self.color_network = None

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        B, N, _ = xyz.shape
        xyz = xyz.reshape(-1, 3)
        out = self.model(xyz)
        return out.reshape(B, N, 1)

@persistence.persistent_class
class TriPlaneMLP(nn.Module):
    def __init__(self, cfg: DictConfig, out_dim: int):
        super().__init__()
        self.cfg = cfg
        self.out_dim = out_dim

        has_view_cond = self.cfg.tri_plane.view_hid_dim > 0

        if self.cfg.tri_plane.mlp.n_layers == 0:
            assert self.cfg.tri_plane.feat_dim == (self.out_dim + 1), f"Wrong dims: {self.cfg.tri_plane.feat_dim}, {self.out_dim}"
            self.model = nn.Identity()
        else:
            if self.cfg.tri_plane.get('posenc_period_len', 0) > 0:
                self.pos_enc = ScalarEncoder1d(3, x_multiplier=self.cfg.tri_plane.posenc_period_len, const_emb_dim=0)
            else:
                self.pos_enc = None

            backbone_input_dim = self.cfg.tri_plane.feat_dim # 256+self.cfg.tri_plane.feat_dim+3 # self.cfg.tri_plane.feat_dim + 3 + (0 if self.pos_enc is None else self.pos_enc.get_dim())
            backbone_out_dim = (self.cfg.tri_plane.mlp.hid_dim if has_view_cond else self.out_dim)
            self.dims = [backbone_input_dim] + [self.cfg.tri_plane.mlp.hid_dim] * (self.cfg.tri_plane.mlp.n_layers - 1) + [backbone_out_dim] # (n_hid_layers + 2)
            activations = ['lrelu'] * (len(self.dims) - 2) + ['linear']
            assert len(self.dims) > 2, f"We cant have just a linear layer here: nothing to modulate. Dims: {self.dims}"
            layers = [FullyConnectedLayer(self.dims[i], self.dims[i+1], activation=a) for i, a in enumerate(activations)]
            self.model = nn.Sequential(*layers)
            self.model_second = nn.Sequential(*layers)
            self.ray_dir_enc = None
            self.color_network = None

    def forward(self, x: torch.Tensor, coords: torch.Tensor, ray_d_world: torch.Tensor, global_x: torch.Tensor) -> torch.Tensor:
        """
        Params:
            x: [batch_size, 3, num_points, feat_dim] --- volumetric features from tri-planes
            coords: [batch_size, num_points, 3] --- coordinates, assumed to be in [-1, 1]
            ray_d_world: [batch_size, h * w, 3] --- camera ray's view directions
        """
        batch_size, _, num_points, feat_dim = x.shape
        batch_size, _, global_feat_dim = global_x.shape
        x = x.mean(dim=1).reshape(batch_size * num_points, feat_dim) # [batch_size * num_points, feat_dim]
        y_2 = self.model(x)
        y_2 = y_2.view(batch_size, num_points, self.dims[-1]) # [batch_size, num_points, backbone_out_dim]
        """
        if global_x is not None:
            global_x = global_x.expand(-1,num_points,-1)
            x = x.mean(dim=1).reshape(batch_size * num_points, feat_dim) # [batch_size * num_points, feat_dim]
            model_input = torch.cat([coords.reshape(batch_size * num_points, 3), x, global_x.reshape(batch_size * num_points, global_feat_dim)], dim=1)
        else:
            x = x.mean(dim=1).reshape(batch_size * num_points, feat_dim) # [batch_size * num_points, feat_dim]
            model_input = torch.cat([x, coords.reshape(batch_size * num_points, 3)], dim=1)
        y_1 = self.model(model_input) # [batch_size * num_points, out_dim]
        if global_x is not None:
            model_second_input = torch.cat([y_1.reshape(batch_size * num_points, 3), x, global_x.reshape(batch_size * num_points, global_feat_dim)], dim=1)
        else:
            model_second_input = torch.cat([x, coords.reshape(batch_size * num_points, 3)], dim=1)
        y_2 = self.model_second(model_second_input)
        y_2 = y_2.view(batch_size, num_points, self.dims[-1]) # [batch_size, num_points, backbone_out_dim]
        """
        misc.assert_shape(y_2, [batch_size, num_points, self.out_dim])

        return y_2

@persistence.persistent_class
class TextureMLP(nn.Module):
    def __init__(self, cfg: DictConfig, out_dim: int):
        super().__init__()
        self.cfg = cfg
        self.out_dim = out_dim

        has_view_cond = self.cfg.texture.view_hid_dim > 0

        if self.cfg.texture.mlp.n_layers == 0:
            assert self.cfg.texture.feat_dim == (self.out_dim + 1), f"Wrong dims: {self.cfg.texture.feat_dim}, {self.out_dim}"
            self.model = nn.Identity()
        else:
            if self.cfg.texture.get('posenc_period_len', 0) > 0:
                self.pos_enc = ScalarEncoder1d(3, x_multiplier=self.cfg.texture.posenc_period_len, const_emb_dim=0)
            else:
                self.pos_enc = None

            backbone_input_dim = 32 + 3 + 1
            backbone_out_dim = (self.cfg.texture.mlp.hid_dim if has_view_cond else self.out_dim)
            self.dims = [backbone_input_dim] + [self.cfg.texture.mlp.hid_dim] * (self.cfg.texture.mlp.n_layers - 1) + [backbone_out_dim] # (n_hid_layers + 2)
            activations = ['lrelu'] * (len(self.dims) - 2) + ['linear']
            assert len(self.dims) > 2, f"We cant have just a linear layer here: nothing to modulate. Dims: {self.dims}"
            layers = [FullyConnectedLayer(self.dims[i], self.dims[i+1], activation=a) for i, a in enumerate(activations)]
            self.model = nn.Sequential(*layers)

            if self.cfg.texture.view_hid_dim > 0:
                self.ray_dir_enc = ScalarEncoder1d(coord_dim=3, const_emb_dim=0, x_multiplier=64, use_cos=False)
                self.color_network = nn.Sequential(
                    FullyConnectedLayer(self.cfg.texture.view_hid_dim + self.ray_dir_enc.get_dim(), self.cfg.texture.view_hid_dim, activation='lrelu'),
                    FullyConnectedLayer(self.cfg.texture.view_hid_dim, self.out_dim, activation='linear'),
                )
            else:
                self.ray_dir_enc = None
                self.color_network = None

    def forward(self, x: torch.Tensor, grad: torch.Tensor=None, sdf: torch.Tensor=None, ray_d_world: torch.Tensor=None) -> torch.Tensor:
        """
        Params:
            x: [batch_size, 3, num_points, feat_dim] --- volumetric features from tri-planes
            coords: [batch_size, num_points, 3] --- coordinates, assumed to be in [-1, 1]
            ray_d_world: [batch_size, h * w, 3] --- camera ray's view directions
        """
        batch_size, num_points, feat_dim = x.shape
        x = x.reshape(batch_size * num_points, feat_dim) # [batch_size * num_points, feat_dim]
        """
        y = self.model(x)
        y = y.view(batch_size, num_points, self.dims[-1]) # [batch_size, num_points, backbone_out_dim]
        """
        if grad is not None and sdf is not None:
            x = torch.cat([x, grad.reshape(batch_size * num_points, 3), sdf.reshape(batch_size * num_points, 1)], dim=1)
        x = self.model(x) # [batch_size * num_points, out_dim]
        x = x.view(batch_size, num_points, self.dims[-1]) # [batch_size, num_points, backbone_out_dim]
        y = x

        misc.assert_shape(y, [batch_size, num_points, self.out_dim])

        return y

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlocksSequence(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        in_resolution,              # Which resolution do we start with?
        out_resolution,             # Output image resolution.
        in_channels,                # Number of input channels.
        out_channels,               # Number of input channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 4,        # Use FP16 for the N highest resolutions.
        clamp = False,
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert in_resolution == 0 or (in_resolution >= 4 and math.log2(in_resolution).is_integer())
        assert out_resolution >= 4 and math.log2(out_resolution).is_integer()
        assert in_resolution < out_resolution

        super().__init__()

        self.w_dim = w_dim
        self.out_resolution = out_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_fp16_res = num_fp16_res
        self.clamp = clamp

        in_resolution_log2 = 2 if in_resolution == 0 else (int(np.log2(in_resolution)) + 1)
        out_resolution_log2 = int(np.log2(out_resolution))
        self.block_resolutions = [2 ** i for i in range(in_resolution_log2, out_resolution_log2 + 1)]
        out_channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (out_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for block_idx, res in enumerate(self.block_resolutions):
            cur_in_channels = out_channels_dict[res // 2] if block_idx > 0 else in_channels
            cur_out_channels = out_channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.out_resolution)
            block = SynthesisBlock(cur_in_channels, cur_out_channels, w_dim=w_dim, resolution=res,
                img_channels=self.out_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, x: torch.Tensor=None, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        if self.clamp:
            img = img.clamp(min=0,max=1)
        return img

#----------------------------------------------------------------------------
@persistence.persistent_class
class Density(nn.Module):
    def __init__(self, params_init={}):
        super().__init__()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)

@persistence.persistent_class
class LaplaceDensity(Density):
    def __init__(self, params_init={}, beta_min=0.0001, is_hollow=False):
        super().__init__(params_init=params_init)
        self.beta_min = 0.0001
        self.is_hollow= is_hollow

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()
        beta = 0.05
        alpha = 1 / beta
        if self.is_hollow:
            return (1 / (2 * beta)) * torch.exp(-(sdf) / beta)
        else:
            return alpha * (0.5 + 0.5 * (sdf).sign() * torch.expm1(-(sdf).abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta

@persistence.persistent_class
class SphereTemplate(nn.Module):
    def __init__(self, radius=0.5, is_hollow=False):
        super().__init__()
        self.radius = radius
        self.density = LaplaceDensity(params_init={'beta':0.01, 'miu':0.}, beta_min=0.0001, is_hollow=is_hollow)
    
        param = nn.Parameter(torch.tensor(1.0))
        setattr(self, 'gamma', param)

    def forward(self, points, get_sdf=False):
        assert points.size(-1) == 3
        points_flat = points.reshape(-1, 3)
        sdfs_flat = torch.linalg.norm(points_flat, dim=-1) - self.radius
        # return ((-sdfs_flat) * self.gamma).reshape(points.shape[0], points.shape[1])[...,None]

        sigmas_flat = self.density(sdfs_flat)
        sigmas = sigmas_flat.reshape(points.shape[0], points.shape[1])[...,None]
        if get_sdf:
            return sigmas, sdfs_flat.reshape(points.shape[0], points.shape[1])[...,None]
        else:
            return sigmas

#----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        cfg: DictConfig,            # Main config
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        num_fp16_res = 4,           # Number of FP16 res blocks for the upsampler
        **synthesis_seq_kwargs,     # Arguments of SynthesisBlocksSequence
    ):
        super().__init__()
        self.cfg = cfg
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.fold_sdf = FoldSDF(feat_dim=256, 
                                ckpt_path="/data/anjie/Projects/CanonicalVAE/logs/2022-11-16T16-36-17_foldsdf_with_chamfer_scratch_dpsr_spgan_vis/checkpoints/epoch=001679.ckpt",
                                ignore_keys=['dpsr'])

        # sigma
        sigma_decoder_out_channels = self.cfg.tri_plane.feat_dim * 3 + (self.img_channels if self.cfg.bg_model.type == "plane" else 0)

        self.tri_plane_decoder = SynthesisBlocksSequence(
            w_dim=w_dim,
            in_resolution=0,
            out_resolution=self.cfg.tri_plane.res,
            in_channels=0,
            out_channels=sigma_decoder_out_channels,
            architecture='skip',
            num_fp16_res=(0 if self.cfg.tri_plane.fp32 else num_fp16_res),
            use_noise=self.cfg.use_noise,
            **synthesis_seq_kwargs,
        )

        self.tri_plane_mlp_f = TriPlaneMLP(self.cfg, out_dim=3)
        self.tri_plane_mlp_b = TriPlaneMLP(self.cfg, out_dim=3)
        self.template = SphereTemplate(is_hollow=True)
        self.sanity_mlp = SanityMLP(out_dim=1)

        self.texture_mlp = TextureMLP(self.cfg, out_dim=3)

        self.num_ws = self.tri_plane_decoder.num_ws
        self.nerf_noise_std = 0.0
        self.train_resolution = self.cfg.patch.resolution if self.cfg.patch.enabled else self.img_resolution
        self.test_resolution = self.img_resolution

        if self.cfg.bg_model.type in (None, "plane"):
            self.bg_model = None
        elif self.cfg.bg_model.type == "sphere":
            self.bg_model = INRSynthesisNetwork(self.cfg.bg_model, w_dim)
            self.num_ws += self.bg_model.num_ws
        else:
            raise NotImplementedError(f"Uknown BG model type: {self.bg_model}")

        # rgb
        texture_decoder_out_channels = 32
        self.texture_decoder = SynthesisBlocksSequence(
            w_dim=w_dim,
            in_resolution=0,
            out_resolution=self.cfg.texture.res,
            in_channels=0,
            out_channels=texture_decoder_out_channels,
            architecture='skip',
            num_fp16_res=(0 if self.cfg.texture.fp32 else num_fp16_res),
            use_noise=self.cfg.use_noise,
            **synthesis_seq_kwargs,
        )

    def progressive_update(self, cur_kimg: float):
        self.nerf_noise_std = linear_schedule(cur_kimg, self.cfg.nerf_noise_std_init, 0.0, self.cfg.nerf_noise_kimg_growth)

    @torch.no_grad()
    def compute_densities(self, ws: torch.Tensor, coords: torch.Tensor, max_batch_res: int=32, use_bg: bool=False, **block_kwargs) -> torch.Tensor:
        """
        coords: [batch_size, num_points, 3]
        """
        raise NotImplementedError

    def forward(self, geo_ws, tex_ws, camera_angles: torch.Tensor, points: torch.Tensor=None, patch_params: Dict=None, max_batch_res: int=128, return_depth: bool=False, ignore_bg: bool=False, bg_only: bool=False, fov=None, verbose: bool=False, return_tex: bool=False, **block_kwargs):
        """
        geo_ws: [batch_size, num_ws, w_dim] --- latent codes
        tex_ws: [batch_size, num_ws, w_dim] --- latent codes
        camera_angles: [batch_size, 3] --- yaw/pitch/roll angles (roll angles are never used)
        patch_params: Dict {scales: [batch_size, 2], offsets: [batch_size, 2]} --- patch parameters (when we do patchwise training)
        """
        misc.assert_shape(camera_angles, [len(geo_ws), 3])

        geo_global_feat = geo_ws[:, self.tri_plane_decoder.num_ws:self.tri_plane_decoder.num_ws+1]

        if self.cfg.backbone == 'raw_planes':
            geo_feats = self.tri_plane_decoder.repeat(len(geo_ws), 1, 1, 1) + geo_ws.sum() * 0.0 # [batch_size, 3, 256, 256]
            tex_feats = self.texture_decoder.repeat(len(tex_ws), 1, 1, 1) + tex_ws.sum() * 0.0 # [batch_size, 3, 256, 256]
        else:
            geo_feats = self.tri_plane_decoder(geo_ws[:, :self.tri_plane_decoder.num_ws], **block_kwargs) # [batch_size, 3 * feat_dim, tp_h, tp_w]
            tex_feats = self.texture_decoder(tex_ws[:, :self.texture_decoder.num_ws], **block_kwargs) # [batch_size, feat_dim, tp_h, tp_w]

        camera_angles[:, [1]] = torch.clamp(camera_angles[:, [1]], 1e-5, np.pi - 1e-5) # [batch_size, 1]
        batch_size = geo_ws.shape[0]
        h = w = (self.train_resolution if self.training else self.test_resolution)
        fov = self.cfg.dataset.sampling.fov if fov is None else fov # [1] or [batch_size]

        self.fold_sdf.eval()
        with torch.no_grad():
            points = points.to(geo_ws.device)
            batch_p_2d, folding_points, folding_normals, sdf_grid = self.fold_sdf(points)
            # sdf_grid = self.fold_sdf.forward_gdt(points[..., :3], points[..., 3:6])
            sdf_grid = sdf_grid.view(batch_size, 1, *self.fold_sdf.dpsr.res)

        num_steps = self.cfg.num_ray_steps
        rgb_sigma_out_dim = 4
        white_back_end_idx = self.img_channels if self.cfg.dataset.white_back else 0
        nerf_noise_std = self.nerf_noise_std if self.training else 0.0

        z_vals, rays_d_cam = get_initial_rays_trig(
            batch_size, num_steps, resolution=(h, w), device=geo_ws.device, ray_start=self.cfg.dataset.sampling.ray_start,
            ray_end=self.cfg.dataset.sampling.ray_end, fov=fov, patch_params=patch_params)
        c2w = compute_cam2world_matrix(camera_angles, self.cfg.dataset.sampling.radius) # [batch_size, 4, 4]
        points_world, z_vals, ray_d_world, ray_o_world = transform_points(z_vals=z_vals, ray_directions=rays_d_cam, c2w=c2w) # [batch_size, h * w, num_steps, 1], [?]
        points_world = points_world.reshape(batch_size, h * w * num_steps, 3) # [batch_size, h * w * num_steps, 3]

        coarse_output = run_batchwise(
            fn=canonical_renderer_pretrain, data=dict(coords=points_world),
            batch_size=max_batch_res ** 2 * num_steps, dim=1, 
            mlp_f=self.tri_plane_mlp_f, mlp_b=self.tri_plane_mlp_b, template=self.template, texture_mlp=self.texture_mlp,
            geo_x=geo_feats, tex_x=tex_feats, scale=self.cfg.dataset.cube_scale, ray_d_world=ray_d_world,
            geo_ws=geo_global_feat,
            folding_grid=batch_p_2d, folding_coords=folding_points, folding_normals=folding_normals,
            sanity_mlp=self.sanity_mlp,
            sdf_grid=sdf_grid,
        ) # [batch_size, h * w * num_steps, num_feats]
        coarse_output = coarse_output.view(batch_size, h * w, num_steps, rgb_sigma_out_dim) # [batch_size, h * w, num_steps, num_feats] | rgbs, sigmas, f_pts, b_pts
        
        coarse_rgb_sigma = coarse_output[...,:rgb_sigma_out_dim]
        # coarse_f_pts = coarse_output[..., rgb_sigma_out_dim:rgb_sigma_out_dim+3]
        # coarse_b_pts = coarse_output[..., rgb_sigma_out_dim+3:rgb_sigma_out_dim+6]
        # coarse_sdf = coarse_output[..., rgb_sigma_out_dim+6:rgb_sigma_out_dim+9]

        # <==================================================>
        # HIERARCHICAL SAMPLING START
        points_world = points_world.reshape(batch_size, h * w, num_steps, 3) # [batch_size, h * w, num_steps, 3]
        weights = run_batchwise(
            fn=fancy_integration,
            data=dict(rgb_sigma=coarse_rgb_sigma, z_vals=z_vals),
            batch_size=max_batch_res ** 2,
            dim=1,
            clamp_mode=self.cfg.clamp_mode, noise_std=nerf_noise_std, use_inf_depth=self.cfg.bg_model.type is None,
        )['weights'] # [batch_size, h * w, num_steps, 1]
        weights = weights.reshape(batch_size * h * w, num_steps) + 1e-5 # [batch_size * h * w, num_steps]

        # <= Importance sampling START =>
        z_vals = z_vals.reshape(batch_size * h * w, num_steps) # [batch_size * h * w, num_steps]
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:]) # [batch_size * h * w, num_steps - 1]
        z_vals = z_vals.reshape(batch_size, h * w, num_steps, 1) # [batch_size, h * w, num_steps, 1]
        fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps, det=False).detach()
        fine_z_vals = fine_z_vals.reshape(batch_size, h * w, num_steps, 1)

        fine_points = ray_o_world.unsqueeze(2).contiguous() + ray_d_world.unsqueeze(2).contiguous() * fine_z_vals.expand(-1, -1, -1, 3).contiguous()
        fine_points = fine_points.reshape(batch_size, h * w * num_steps, 3)

        # Model prediction on re-sampled find points
        fine_output = run_batchwise(
            fn=canonical_renderer_pretrain, data=dict(coords=fine_points),
            batch_size=max_batch_res ** 2 * num_steps, dim=1, 
            mlp_f=self.tri_plane_mlp_f, mlp_b=self.tri_plane_mlp_b, template=self.template, texture_mlp=self.texture_mlp,
            geo_x=geo_feats, tex_x=tex_feats, scale=self.cfg.dataset.cube_scale, ray_d_world=ray_d_world,
            geo_ws=geo_global_feat,
            folding_grid=batch_p_2d, folding_coords=folding_points, folding_normals=folding_normals,
            sanity_mlp=self.sanity_mlp,
            sdf_grid=sdf_grid,
        ) # [batch_size, h * w * num_steps, num_feats]
        fine_output = fine_output.view(batch_size, h * w, num_steps, rgb_sigma_out_dim) # [batch_size, h * w, num_steps, num_feats]

        fine_rgb_sigma = fine_output[...,:rgb_sigma_out_dim]
        # fine_f_pts = fine_output[..., rgb_sigma_out_dim:rgb_sigma_out_dim+3]
        # fine_b_pts = fine_output[..., rgb_sigma_out_dim+3:rgb_sigma_out_dim+6]
        # fine_sdf = fine_output[..., rgb_sigma_out_dim+6:rgb_sigma_out_dim+9]
        fine_points = fine_points.reshape(batch_size, h * w, num_steps, 3) # [batch_size, h * w, num_steps, 3]

        # Combine coarse and fine points and sort by z_values
        all_rgb_sigma = torch.cat([fine_rgb_sigma, coarse_rgb_sigma], dim=2) # [batch_size, h * w, 2 * num_steps, tri_plane_out_dim + 1]
        # all_sdf = torch.cat([fine_sdf, coarse_sdf], dim=2) # [batch_size, h * w, 2 * num_steps, tri_plane_out_dim + 1]
        all_z_vals = torch.cat([fine_z_vals, z_vals], dim=2) # [batch_size, h * w, 2 * num_steps, 1]
        _, indices = torch.sort(all_z_vals, dim=2) # [batch_size, h * w, 2 * num_steps, 1]
        all_z_vals = torch.gather(all_z_vals, dim=2, index=indices) # [batch_size, h * w, 2 * num_steps, 1]
        all_rgb_sigma = torch.gather(all_rgb_sigma, dim=2, index=indices.expand(-1, -1, -1, rgb_sigma_out_dim)) # [batch_size, h * w, 2 * num_steps, tri_plane_out_dim + 1]
        # all_sdf = torch.gather(all_sdf, dim=2, index=indices.expand(-1, -1, -1, 1))
        # HIERARCHICAL SAMPLING END
        # <==================================================>

        int_out: Dict = run_batchwise(
            fn=fancy_integration,
            data=dict(rgb_sigma=all_rgb_sigma, z_vals=all_z_vals),
            batch_size=max_batch_res ** 2,
            dim=1,
            white_back_end_idx=white_back_end_idx, last_back=self.cfg.dataset.last_back, clamp_mode=self.cfg.clamp_mode,
            noise_std=nerf_noise_std, use_inf_depth=self.cfg.bg_model.type is None)
        misc.assert_shape(int_out['final_transmittance'], [batch_size, h * w])

        assert not bg_only

        img = int_out['depth' if return_depth else 'rendered_feats'] # [batch_size, h * w, 1 | mlp_out_dim]
        img = img.reshape(batch_size, h, w, img.shape[2]) # [batch_size, h, w, 1 | tri_plane_out_dim - 1]
        img = img.permute(0, 3, 1, 2).contiguous() # [batch_size, 1 | mlp_out_dim, h, w]
        mask = int_out['final_transmittance'].reshape(batch_size, h, w, 1).permute(0, 3, 1, 2).contiguous()
        img = torch.cat([img, mask], dim=1)

        if verbose:
            # coords = create_voxel_coords(32, [0.0, 0.0, 0.0], self.cfg.dataset.cube_scale * 2, batch_size=batch_size).to(img)
            # coords_output = run_batchwise(
            #     fn=canonical_renderer_pretrain, data=dict(coords=coords),
            #     batch_size=max_batch_res ** 2 * num_steps, dim=1, 
            #     mlp_f=self.tri_plane_mlp_f, mlp_b=self.tri_plane_mlp_b, template=self.template, texture_mlp=self.texture_mlp,
            #     geo_x=geo_feats, tex_x=tex_feats, scale=self.cfg.dataset.cube_scale, ray_d_world=ray_d_world,
            #     geo_ws=geo_global_feat,
            #     rt_radiance=False,
            # ) # [batch_size, h * w * num_steps, num_feats]
            
            # uniform_coords = coords_output[...,:3]
            # uniform_coords_f = coords_output[...,3:6]
            # uniform_coords_b = coords_output[...,6:9]
            # uniform_coords_sigmas = coords_output[...,9:10]
            # uniform_coords_sdfs = coords_output[...,10:11]

            info = {}
            # info['all_points'] = torch.cat([fine_points, points_world], dim=2) # [batch_size, h * w, 2 * num_steps, 3]
            # info['all_f_pts'] = torch.cat([fine_f_pts, coarse_f_pts], dim=2) # [batch_size, h * w, 2 * num_steps, 3]
            # info['all_b_pts'] = torch.cat([fine_b_pts, coarse_b_pts], dim=2) # [batch_size, h * w, 2 * num_steps, 3]
            # info['all_rgb_sigma'] = all_rgb_sigma
            # info['all_sdf'] = torch.cat([fine_sdf, coarse_sdf], dim=2) # [batch_size, h * w, 2 * num_steps, 3]
            # info['uniform_coords'] = uniform_coords
            # info['uniform_coords_f'] = uniform_coords_f
            # info['uniform_coords_b'] = uniform_coords_b
            # info['uniform_coords_sigmas'] = uniform_coords_sigmas
            # info['uniform_coords_sdfs'] = uniform_coords_sdfs
            return img, info
        else:
            if return_tex:
                return img, None
            else:
                return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        cfg: DictConfig,            # Main config
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.cfg = cfg
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        assert z_dim % 2 == 0
        self.geo_dim = z_dim // 2
        self.tex_dim = z_dim // 2
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(cfg=cfg, w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.geo_num_ws = self.synthesis.num_ws + 1
        self.tex_num_ws = self.synthesis.num_ws
        self.geo_mapping = MappingNetwork(z_dim=self.geo_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.geo_num_ws, **mapping_kwargs)
        self.tex_mapping = MappingNetwork(z_dim=self.tex_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.tex_num_ws, **mapping_kwargs)

    def progressive_update(self, cur_kimg: float):
        self.synthesis.progressive_update(cur_kimg)

    def forward(self, z, c, p, camera_angles, camera_angles_cond=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        geo_z = z[...,:self.geo_dim]
        tex_z = z[...,-self.tex_dim:]
        geo_ws = self.geo_mapping(geo_z, c, camera_angles=camera_angles_cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        tex_ws = self.tex_mapping(tex_z, c, camera_angles=camera_angles_cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(geo_ws, tex_ws, points=p, camera_angles=camera_angles, update_emas=update_emas, **synthesis_kwargs)
        return img

#----------------------------------------------------------------------------

def create_voxel_coords(resolution=256, voxel_origin=[0.0, 0.0, 0.0], cube_size=2.0, batch_size: int=1):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_size / 2
    voxel_size = cube_size / (resolution - 1)

    overall_index = torch.arange(0, resolution ** 3, 1, out=torch.LongTensor())
    coords = torch.zeros(resolution ** 3, 3) # [h, w, d, 3]

    # transform first 3 columns
    # to be the x, y, z index
    coords[:, 2] = overall_index % resolution
    coords[:, 1] = (overall_index.float() / resolution) % resolution
    coords[:, 0] = ((overall_index.float() / resolution) / resolution) % resolution

    # transform first 3 columns
    # to be the x, y, z coordinate
    coords[:, 0] = (coords[:, 0] * voxel_size) + voxel_origin[2] # [voxel_res ** 3]
    coords[:, 1] = (coords[:, 1] * voxel_size) + voxel_origin[1] # [voxel_res ** 3]
    coords[:, 2] = (coords[:, 2] * voxel_size) + voxel_origin[0] # [voxel_res ** 3]

    return coords.repeat(batch_size, 1, 1) # [batch_size, voxel_res ** 3, 3]