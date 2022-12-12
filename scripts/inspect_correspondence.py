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

import plotly.graph_objects as go
# import plotly.express as px

from src.training.training_utils import run_batchwise
from src.training.networks_canograf import canonical_renderer_pretrain
from scripts.utils import load_generator, set_seed, create_voxel_coords

#----------------------------------------------------------------------------

@hydra.main(config_path="../configs/scripts", config_name="extract_geometry.yaml")
def extract_geometry(cfg: DictConfig):
    device = torch.device('cuda')
    G = load_generator(cfg.ckpt, verbose=cfg.verbose)[0].to(device).eval()

    sphere = trimesh.creation.uv_sphere(radius=0.3)
    sphere_v = sphere.vertices
    sphere_c = np.clip(((sphere.vertices + 0.3) / 0.6 * 255), 0, 255).astype(int)
    fig = go.Figure(data=[go.Scatter3d(
        x=sphere_v[:,0],
        y=sphere_v[:,1],
        z=sphere_v[:,2],
        mode='markers',
        marker=dict(
            size=12,
            color=['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(sphere_c[:,0], sphere_c[:,1], sphere_c[:,2])],
            opacity=1
        )
    )])
    html_fpath = os.path.join(cfg.output_dir, 'output_car_fp_all.html')
    with open(html_fpath, 'w') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

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
            fn=canonical_renderer_pretrain, data=dict(coords=coords, ray_d_world=ray_d_world),
            batch_size=cfg.max_batch_res ** 3, dim=1, 
            mlp_f=G.synthesis.tri_plane_mlp_f, mlp_b=G.synthesis.tri_plane_mlp_b, template=G.synthesis.template, 
            geo_mlp=G.synthesis.geo_mlp, texture_mlp=G.synthesis.texture_mlp,
            geo_x=geo_feats, tex_x=tex_feats, scale=G.synthesis.cfg.dataset.cube_scale,
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
        coords_np = coords[0].squeeze().cpu().numpy() # [sigma.squeeze()>=10]
        output_f_pts_np = output_f_pts[0].squeeze().cpu().numpy()
        output_b_pts_np = output_b_pts[0].squeeze().cpu().numpy()

        vcoords_colors = (output_f_pts_np + 0.3) / 0.6 * 255
        vcoords_colors = np.clip(vcoords_colors, 0, 255).astype(int)
    
        assert batch_size == 1
        sigma = sigma.reshape(cfg.voxel_res, cfg.voxel_res, cfg.voxel_res).cpu().numpy() # [voxel_res ** 3]

        print('sigma percentiles:', {q: np.percentile(sigma.reshape(-1), q) for q in [50.0, 90.0, 95.0, 97.5, 99.0, 99.5]})
        # vertices, triangles = mcubes.marching_cubes(sigma, np.percentile(sigma, cfg.thresh_percentile))
        vertices, triangles = mcubes.marching_cubes(sigma, 10)
        mesh = trimesh.Trimesh(vertices, triangles)
        os.makedirs('shapes', exist_ok=True)
        mesh.export(f'shapes/shape-{seed}.obj')

        # fig = go.Figure(data=[go.Scatter3d(
        #     x=coords_np[:,0],
        #     y=coords_np[:,1],
        #     z=coords_np[:,2],
        #     mode='markers',
        #     marker=dict(
        #         size=12,
        #         color=['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(vcoords_colors[:,0], vcoords_colors[:,1], vcoords_colors[:,2])],
        #         opacity=1
        #     )
        # )])

        # with open(html_fpath, 'a') as f:
        #     f.write(fig.to_html(full_html=False))

        f_pts_c = np.clip(((output_f_pts_np + 0.3) / 0.6 * 255), 0, 255).astype(int)
        fig = go.Figure(data=[go.Scatter3d(
            x=output_f_pts_np[:,0],
            y=output_f_pts_np[:,1],
            z=output_f_pts_np[:,2],
            mode='markers',
            marker=dict(
                size=12,
                color=['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(f_pts_c[:,0], f_pts_c[:,1], f_pts_c[:,2])],
                opacity=1
            )
        )])

        with open(html_fpath, 'a') as f:
            f.write(fig.to_html(full_html=False))

        os.makedirs(cfg.output_dir, exist_ok=True)
        with mrcfile.new_mmap(os.path.join(cfg.output_dir, f'{seed}.mrc'), overwrite=True, shape=sigma.shape, mrc_mode=2) as mrc:
            mrc.data[:] = sigma

#----------------------------------------------------------------------------

if __name__ == "__main__":
    extract_geometry() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------