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
from src.training.networks_cips import CIPSres
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
from src.training.vis_helper import *

@misc.profiled_function
def canonical_renderer_pretrain(uv_x: torch.Tensor, coords: torch.Tensor, ray_d_world: torch.Tensor, 
                                sdf_grid: torch.Tensor, folding_grid: torch.Tensor, folding_coords: torch.Tensor, folding_normals: torch.Tensor,
                                texture_mlp: Callable, scale: float=1.0,  rt_sdf: bool=False, rt_radiance: bool=True) -> torch.Tensor:
    # geo
    batch_size, num_uv, uv_feat_dim = uv_x.shape
    num_points = coords.shape[1]

    # sdf_grid = sdf_grid.permute(0,1,4,3,2)

    coords_normed = coords / 0.5
    # sdfs = F.grid_sample(sdf_grid, coords_normed.view(batch_size, 1, 1, num_points, 3), padding_mode="border").view(batch_size, num_points, 1)
    sdfs = F.grid_sample(sdf_grid, coords_normed.view(batch_size, 1, 1, num_points, 3), padding_mode="border").view(batch_size, num_points, 1)
    
    # sigmas = torch.sigmoid(-sdfs / 0.005) / 0.005

    beta = 0.005
    alpha = 1 / beta
    sigmas = alpha * (0.5 + 0.5 * (sdfs).sign() * torch.expm1(-(sdfs).abs() / beta))

    K = 8
    dis, indices, _ = knn_points(coords.detach(), folding_coords.detach(), K=K)
    dis = dis.detach()
    indices = indices.detach()
    dis = dis.sqrt()
    weights = 1 / (dis + 1e-7)
    weights = weights / torch.sum(weights, dim=-1, keepdims=True)  # (1,M,K)
    # num_of_grids = folding_grid.shape[1]
    # sphere_samples_normed = F.normalize(folding_grid, dim=-1)
    # sphere_u = (0.5 + torch.atan2(sphere_samples_normed[..., 0], sphere_samples_normed[..., 2]) / (2 * math.pi)).transpose(0,1)
    # sphere_v = (0.5 - torch.asin(sphere_samples_normed[..., 1]) / math.pi).transpose(0,1)
    # sphere_uv = torch.cat([sphere_u,sphere_v], dim=1).view(batch_size, 1, num_of_grids, 2)
    # sphere_uv = torch.cat([sphere_u,sphere_v], dim=1).view(batch_size, 1, num_of_grids, 2)
    # sphere_uv = (sphere_uv * 2) - 1
    # sphere_uv_f = F.grid_sample(tex_x, grid=sphere_uv, mode='bilinear', align_corners=True).view(batch_size, 32, num_of_grids) # [batch_size, 3, feat_dim, num_points]
    # sphere_uv_f = sphere_uv_f.permute(0, 2, 1) # [batch_size, num_points, feat_dim]

    # sphere_uv_f = get_feat_from_triplane(folding_grid, tex_x, scale=0.3).mean(dim=1)

    fused_tex_feature = torch.sum(batched_index_select(uv_x, 1, indices).view(batch_size, num_points, K, 32) * weights[..., None], dim=-2)
    # fused_tex_normal = torch.sum(batched_index_select(folding_normals, 1, indices).view(batch_size, num_points, K, 3) * weights[..., None], dim=-2)

    # volume rgbs
    # fused_tex_feature = get_feat_from_triplane(coords_normed, tex_x, scale=None).mean(dim=1)
    sdf_grid_grad = torch.gradient(sdf_grid.squeeze(1))
    normal_grid = torch.stack([sdf_grid_grad[0], sdf_grid_grad[1], sdf_grid_grad[2]], dim=1)
    normals = F.grid_sample(normal_grid, coords_normed.view(batch_size, 1, 1, num_points, 3), padding_mode="border").view(batch_size, num_points, 3)
    rgbs = texture_mlp(fused_tex_feature, normals.detach(), sdfs.detach()) # [batch_size, num_points, out_dim]
    # import pdb; pdb.set_trace()

    # normed_bp2d = torch.clip((folding_grid+0.5), 0, 1) # normalize color to 0-1
    # rgbs = rgbs * 0 + (batched_index_select(normed_bp2d, 1, indices).view(batch_size, num_points, 3))
    
    # import pdb; pdb.set_trace()
    # ff = plotly_gen_points(folding_coords[0], color=sphere_to_color(normed_bp2d[0].cpu().numpy(), 0.5), rt_html=False, png_path=None)
    # ff = plotly_gen_points(coords[0][(torch.abs(sdfs[0]) < 0.1).squeeze()], color=None, rt_html=False, png_path=None)
    # plotly_gen_points(folding_coords[0], color=None, rt_html=False, png_path=None)

    return torch.cat([rgbs, sigmas.detach()], dim=-1)

#----------------------------------------------------------------------------

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
                                ckpt_path="../../pretrain/foldsdf_car.ckpt",
                                ignore_keys=['dpsr'])

        # rgb
        if self.cfg.texture.type == "cips":
            self.texture_decoder = CIPSres(style_dim=256, out_dim=32)
        elif self.cfg.texture.type == "triplane":
            assert TypeError
            texture_decoder_out_channels = 32 * 3
            self.texture_decoder = SynthesisBlocksSequence(
                w_dim=w_dim,
                in_resolution=0,
                out_resolution=128,
                in_channels=0,
                out_channels=texture_decoder_out_channels,
                architecture='skip',
                num_fp16_res=(0 if self.cfg.texture.fp32 else num_fp16_res),
                use_noise=self.cfg.use_noise,
                **synthesis_seq_kwargs,
            )

        self.texture_mlp = TextureMLP(self.cfg, out_dim=3)

        self.num_ws = self.texture_decoder.num_ws
        self.nerf_noise_std = 0.0
        self.train_resolution = self.cfg.patch.resolution if self.cfg.patch.enabled else self.img_resolution
        self.test_resolution = 64 # self.img_resolution

        if self.cfg.bg_model.type in (None, "plane"):
            self.bg_model = None
        elif self.cfg.bg_model.type == "sphere":
            self.bg_model = INRSynthesisNetwork(self.cfg.bg_model, w_dim)
            self.num_ws += self.bg_model.num_ws
        else:
            raise NotImplementedError(f"Uknown BG model type: {self.bg_model}")

    def progressive_update(self, cur_kimg: float):
        self.nerf_noise_std = linear_schedule(cur_kimg, self.cfg.nerf_noise_std_init, 0.0, self.cfg.nerf_noise_kimg_growth)


    def forward(self, geo_ws, tex_ws, camera_angles: torch.Tensor, points: torch.Tensor=None, patch_params: Dict=None, max_batch_res: int=128, return_depth: bool=False, ignore_bg: bool=False, bg_only: bool=False, fov=None, verbose: bool=False, return_tex: bool=False, **block_kwargs):
        """
        geo_ws: [batch_size, num_ws, w_dim] --- latent codes
        tex_ws: [batch_size, num_ws, w_dim] --- latent codes
        camera_angles: [batch_size, 3] --- yaw/pitch/roll angles (roll angles are never used)
        patch_params: Dict {scales: [batch_size, 2], offsets: [batch_size, 2]} --- patch parameters (when we do patchwise training)
        """
        # misc.assert_shape(camera_angles, [len(geo_ws), 3])
        if not self.training:
            # max_batch_res = 32
            foldsdf_level = 5
        else:
            foldsdf_level = 4

        if camera_angles.size(1) == 3:
            radius = self.cfg.dataset.sampling.radius
        elif camera_angles.size(1) == 5:
            radius = camera_angles[:,3]
            fov = camera_angles[:,4]
            camera_angles = camera_angles[:,:3]

        batch_size = geo_ws.shape[0]
        h = w = (self.train_resolution if self.training else self.test_resolution)
        fov = self.cfg.dataset.sampling.fov if fov is None else fov # [1] or [batch_size]

        self.fold_sdf.eval()
        with torch.no_grad():
            points = points.to(geo_ws.device)

            batch_p_2d, folding_points, folding_normals, sdf_grid = self.fold_sdf(points, level=foldsdf_level)
            # sdf_grid = self.fold_sdf.forward_gdt(points)
            sdf_grid = sdf_grid.view(batch_size, 1, *self.fold_sdf.dpsr.res)

        # import pdb; pdb.set_trace()

        if self.cfg.texture.type == "cips":
            uv_feats = self.texture_decoder(batch_p_2d, tex_ws[:, 0]) # [batch_size, feat_dim, tp_h, tp_w]

        num_steps = self.cfg.num_ray_steps
        rgb_sigma_out_dim = 4
        white_back_end_idx = self.img_channels if self.cfg.dataset.white_back else 0
        nerf_noise_std = self.nerf_noise_std if self.training else 0.0

        z_vals, rays_d_cam = get_initial_rays_trig(
            batch_size, num_steps, resolution=(h, w), device=geo_ws.device, ray_start=self.cfg.dataset.sampling.ray_start,
            ray_end=self.cfg.dataset.sampling.ray_end, fov=fov, patch_params=patch_params, radius=radius)
        c2w = compute_cam2world_matrix(camera_angles, radius) # [batch_size, 4, 4]
        points_world, z_vals, ray_d_world, ray_o_world = transform_points(z_vals=z_vals, ray_directions=rays_d_cam, c2w=c2w) # [batch_size, h * w, num_steps, 1], [?]
        points_world = points_world.reshape(batch_size, h * w * num_steps, 3) # [batch_size, h * w * num_steps, 3]

        coarse_output = run_batchwise(
            fn=canonical_renderer_pretrain, data=dict(coords=points_world),
            batch_size=max_batch_res ** 2 * num_steps, dim=1, 
            texture_mlp=self.texture_mlp, uv_x=uv_feats, 
            scale=self.cfg.dataset.cube_scale, ray_d_world=ray_d_world,
            folding_grid=batch_p_2d, folding_coords=folding_points, folding_normals=folding_normals, sdf_grid=sdf_grid,
        ) # [batch_size, h * w * num_steps, num_feats]
        coarse_output = coarse_output.view(batch_size, h * w, num_steps, rgb_sigma_out_dim) # [batch_size, h * w, num_steps, num_feats] | rgbs, sigmas, f_pts, b_pts
        coarse_rgb_sigma = coarse_output[...,:rgb_sigma_out_dim]


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
            texture_mlp=self.texture_mlp, uv_x=uv_feats, 
            scale=self.cfg.dataset.cube_scale, ray_d_world=ray_d_world,
            folding_grid=batch_p_2d, folding_coords=folding_points, folding_normals=folding_normals, sdf_grid=sdf_grid,
        ) # [batch_size, h * w * num_steps, num_feats]
        fine_output = fine_output.view(batch_size, h * w, num_steps, rgb_sigma_out_dim) # [batch_size, h * w, num_steps, num_feats]

        fine_rgb_sigma = fine_output[...,:rgb_sigma_out_dim]
        fine_points = fine_points.reshape(batch_size, h * w, num_steps, 3) # [batch_size, h * w, num_steps, 3]

        # Combine coarse and fine points and sort by z_values
        all_rgb_sigma = torch.cat([fine_rgb_sigma, coarse_rgb_sigma], dim=2) # [batch_size, h * w, 2 * num_steps, tri_plane_out_dim + 1]
        all_z_vals = torch.cat([fine_z_vals, z_vals], dim=2) # [batch_size, h * w, 2 * num_steps, 1]
        _, indices = torch.sort(all_z_vals, dim=2) # [batch_size, h * w, 2 * num_steps, 1]
        all_z_vals = torch.gather(all_z_vals, dim=2, index=indices) # [batch_size, h * w, 2 * num_steps, 1]
        all_rgb_sigma = torch.gather(all_rgb_sigma, dim=2, index=indices.expand(-1, -1, -1, rgb_sigma_out_dim)) # [batch_size, h * w, 2 * num_steps, tri_plane_out_dim + 1]
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
            info = {}
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

    def forward(self, z, p, camera_angles, c=None, camera_angles_cond=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        geo_z = z[...,:self.geo_dim]
        tex_z = z[...,-self.tex_dim:]
        geo_ws = self.geo_mapping(geo_z, c=c, camera_angles=camera_angles_cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        tex_ws = self.tex_mapping(tex_z, c=c, camera_angles=camera_angles_cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(geo_ws, tex_ws, points=p, camera_angles=camera_angles, update_emas=update_emas, **synthesis_kwargs)
        return img