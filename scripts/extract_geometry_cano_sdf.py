from omegaconf import DictConfig
# from ast import DictComp
# import plyfile
# import argparse
import torch
import numpy as np
# import skimage.measure
# import scipy
import mcubes
import trimesh
import mrcfile
import os
import hydra
# from mayavi import mlab
from tqdm import tqdm

# import plotly.graph_objects as go
# import plotly.express as px

from src.training.training_utils import run_batchwise
from src.training.networks_canograf import canonical_renderer
from scripts.utils import load_generator, set_seed, create_voxel_coords

#----------------------------------------------------------------------------

@hydra.main(config_path="../configs/scripts", config_name="extract_geometry.yaml")
def extract_geometry(cfg: DictConfig):
    device = torch.device('cuda')
    G = load_generator(cfg.ckpt, verbose=cfg.verbose)[0].to(device).eval()

    for seed in tqdm(cfg.seeds, desc='Extracting geometry...'):
        set_seed(seed)
        batch_size = 1
        z = torch.randn(batch_size, G.z_dim, device=device) # [batch_size, z_dim]
        c = torch.zeros(batch_size, G.c_dim, device=device) # [batch_size, c_dim]
        assert G.c_dim == 0
        coords = create_voxel_coords(cfg.voxel_res, cfg.voxel_origin, cfg.cube_size, batch_size) # cfg.voxel_res [batch_size, voxel_res ** 3, 3]
        coords = coords.to(z.device) # [batch_size, voxel_res ** 3, 3]

        geo_z = z[...,:G.geo_dim]
        tex_z = z[...,-G.tex_dim:]
        geo_ws = G.geo_mapping(geo_z, c, truncation_psi=cfg.truncation_psi)
        tex_ws = G.tex_mapping(tex_z, c, truncation_psi=cfg.truncation_psi)

        geo_feats = G.synthesis.tri_plane_decoder(geo_ws[:, :G.synthesis.tri_plane_decoder.num_ws]) # [batch_size, 3 * feat_dim, tp_h, tp_w]
        tex_feats = G.synthesis.texture_decoder(tex_ws[:, :G.synthesis.texture_decoder.num_ws]) # [batch_size, feat_dim, tp_h, tp_w]
        ray_d_world = torch.zeros_like(coords) # [batch_size, num_points, 3]
        
        output = run_batchwise(
            fn=canonical_renderer, data=dict(coords=coords, ray_d_world=ray_d_world),
            batch_size=cfg.max_batch_res ** 3, dim=1, 
            mlp_f=G.synthesis.tri_plane_mlp_f, mlp_b=G.synthesis.tri_plane_mlp_b, template=G.synthesis.template, 
            geo_mlp=G.synthesis.geo_mlp, texture_mlp=G.synthesis.texture_mlp,
            geo_x=geo_feats, tex_x=tex_feats, uv_x=uv_feats, scale=G.synthesis.cfg.dataset.cube_scale,
            rt_sdf=True,
        ) # [batch_size, h * w * num_steps, num_feats]
        # coarse_output = coarse_output.view(batch_size, h * w, num_steps, 3 + 1 + 3 + 3) # [batch_size, h * w, num_steps, num_feats] | rgbs, sigmas, f_pts, b_pts
        
        rgb_sigma_out_dim = 4
        output_rgb_sigma = output[...,:rgb_sigma_out_dim]
        output_f_pts = output[..., rgb_sigma_out_dim:rgb_sigma_out_dim+3]
        output_b_pts = output[..., rgb_sigma_out_dim+3:rgb_sigma_out_dim+6]
        sigma = output_rgb_sigma[:, :, -1:] # [batch_size, num_coords, 1]
        rgb = output_rgb_sigma[:, :, :3]
        uvs = output[...,-2:]

        # filter out zero sigma points
        coords_np = coords[0][sigma.squeeze()!=0].squeeze().cpu().numpy() # 
        output_f_pts_np = output_f_pts[0][sigma.squeeze()!=0].squeeze().cpu().numpy()
        output_b_pts_np = output_b_pts[0][sigma.squeeze()!=0].squeeze().cpu().numpy()

        vcoords_colors = (output_f_pts_np + 0.3) / 0.6 * 255
        vcoords_colors = np.clip(vcoords_colors, 0, 255).astype(int)

        f_dir = output_f_pts_np - coords_np
        b_dir = output_b_pts_np - output_f_pts_np
    
        assert batch_size == 1
        sigma = sigma.reshape(cfg.voxel_res, cfg.voxel_res, cfg.voxel_res).cpu().numpy() # [voxel_res ** 3]

        print('sigma percentiles:', {q: np.percentile(sigma.reshape(-1), q) for q in [50.0, 90.0, 95.0, 97.5, 99.0, 99.5]})
        # vertices, triangles = mcubes.marching_cubes(sigma, np.percentile(sigma, cfg.thresh_percentile))
        vertices, triangles = mcubes.marching_cubes(sigma, 10)
        mesh = trimesh.Trimesh(vertices, triangles)
        os.makedirs('shapes', exist_ok=True)
        mesh.export(f'shapes/shape-{seed}.obj')

        os.makedirs(cfg.output_dir, exist_ok=True)
        with mrcfile.new_mmap(os.path.join(cfg.output_dir, f'{seed}_sdf.mrc'), overwrite=True, shape=sigma.shape, mrc_mode=2) as mrc:
            mrc.data[:] = -sigma

#----------------------------------------------------------------------------

if __name__ == "__main__":
    extract_geometry() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------