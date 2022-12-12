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
<<<<<<< HEAD
    html_fpath = os.path.join(cfg.output_dir, 'output_car.html')
=======
    html_fpath = os.path.join(cfg.output_dir, 'chair_output.html')
>>>>>>> c86e719c025c476e7ed118c793b654ad99c876e7
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

<<<<<<< HEAD
        """
        # vis feature map
=======
        """ # vis feature map
>>>>>>> c86e719c025c476e7ed118c793b654ad99c876e7
        import matplotlib.pyplot as plt
        tex_feats = geo_feats
        tex_feats = tex_feats.view(3, 32, 64, 64)
        tex_feats = tex_feats.cpu().numpy()
        k = tex_feats.shape[1]  
        size=tex_feats.shape[-1]
        display_grid = np.zeros((size, size * k))
        for i in range(k):
            feature_image = tex_feats[2, i, :, :]
            feature_image-= feature_image.mean()
            feature_image/= feature_image.std ()
            feature_image*=  64
            feature_image+= 128
            feature_image= np.clip(feature_image, 0, 255).astype('uint8')
            display_grid[:, i * size : (i + 1) * size] = feature_image
        scale = 200. / k
        plt.figure(figsize=(scale * k, scale) )
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto')
        plt.tight_layout() 
        plt.savefig('geo_feats_2.png')
<<<<<<< HEAD
        """
        
=======
        import pdb; pdb.set_trace()
        """

>>>>>>> c86e719c025c476e7ed118c793b654ad99c876e7
        output = run_batchwise(
            fn=canonical_renderer_pretrain, data=dict(coords=coords, ray_d_world=ray_d_world),
            batch_size=cfg.max_batch_res ** 3, dim=1, 
            mlp_f=G.synthesis.tri_plane_mlp_f, mlp_b=G.synthesis.tri_plane_mlp_b, template=G.synthesis.template, 
            geo_mlp=G.synthesis.geo_mlp, texture_mlp=G.synthesis.texture_mlp,
            geo_x=geo_feats, tex_x=tex_feats, uv_x=uv_feats, scale=G.synthesis.cfg.dataset.cube_scale,
        ) # [batch_size, h * w * num_steps, num_feats]
        # coarse_output = coarse_output.view(batch_size, h * w, num_steps, 3 + 1 + 3 + 3) # [batch_size, h * w, num_steps, num_feats] | rgbs, sigmas, f_pts, b_pts
        
        rgb_sigma_out_dim = 4
        output_rgb_sigma = output[...,:rgb_sigma_out_dim]
        output_f_pts = output[..., rgb_sigma_out_dim:rgb_sigma_out_dim+3]
        output_b_pts = output[..., rgb_sigma_out_dim+3:rgb_sigma_out_dim+6]
        sigma = output_rgb_sigma[:, :, -1:] # [batch_size, num_coords, 1]
        rgb = output_rgb_sigma[:, :, :3]
        uvs = output[...,-2:]

        from PIL import Image
        data = np.ones((64, 64, 3), dtype=np.uint8) * 255
        for uv,s in zip(uvs[0], sigma[0]):
            if s > 0:
                uv = ((uv * 0.5) + 0.5) * 64
                data[int(uv[0]), int(uv[1])] = [0, 0, 0]
        
        img = Image.fromarray(data, 'RGB')
        img.save('uv_hit.png')
        # import pdb; pdb.set_trace()

        # filter out zero sigma points
        coords_np = coords[0][sigma.squeeze()>=10].squeeze().cpu().numpy() # 
        output_f_pts_np = output_f_pts[0][sigma.squeeze()>=10].squeeze().cpu().numpy()
        output_b_pts_np = output_b_pts[0][sigma.squeeze()>=10].squeeze().cpu().numpy()

        vcoords_colors = (output_f_pts_np + 0.3) / 0.6 * 255
        vcoords_colors = np.clip(vcoords_colors, 0, 255).astype(int)

        f_dir = output_f_pts_np - coords_np
        b_dir = output_b_pts_np - output_f_pts_np

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # draw sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x*0.3, y*0.3, z*0.3, color="r")

        qf = ax.quiver(coords_np[:,0], coords_np[:,1], coords_np[:,2], f_dir[:,0], f_dir[:,1], f_dir[:,2], normalize=False, color="b")
        # qb = ax.quiver(output_f_pts_np[:,0], output_f_pts_np[:,1], output_f_pts_np[:,2], b_dir[:,0], b_dir[:,1], b_dir[:,2], normalize=False, color="y")
        import pdb; pdb.set_trace()
        # plt.savefig('test1.png')
    
        # mlab.quiver3d(coords_np[:,0], coords_np[:,1], coords_np[:,2], output_f_np[:,0], output_f_np[:,1], output_f_np[:,2])
        # # fig = vector_scatter(coords_np[:,0], coords_np[:,1], coords_np[:,2], output_f_np[:,0], output_f_np[:,1], output_f_np[:,2])
        # mlab.savefig(filename='test.png')

        # import pdb; pdb.set_trace()

        # sigma = G.synthesis.compute_densities(geo_ws, tex_ws, coords, max_batch_res=cfg.max_batch_res) # [batch_size, voxel_res ** 3, 1]
        assert batch_size == 1
        sigma = sigma.reshape(cfg.voxel_res, cfg.voxel_res, cfg.voxel_res).cpu().numpy() # [voxel_res ** 3]

        print('sigma percentiles:', {q: np.percentile(sigma.reshape(-1), q) for q in [50.0, 90.0, 95.0, 97.5, 99.0, 99.5]})
        # vertices, triangles = mcubes.marching_cubes(sigma, np.percentile(sigma, cfg.thresh_percentile))
        vertices, triangles = mcubes.marching_cubes(sigma, 10)
        mesh = trimesh.Trimesh(vertices, triangles)
        os.makedirs('shapes', exist_ok=True)
        mesh.export(f'shapes/shape-{seed}.obj')

        """
        surface_pts = torch.tensor(vertices).to(z.device)[None, :, :].float()
        ray_d_world = torch.zeros_like(surface_pts)
        output = run_batchwise(
            fn=canonical_renderer_pretrain, data=dict(coords=surface_pts, ray_d_world=ray_d_world),
            batch_size=cfg.max_batch_res ** 3, dim=1, 
            mlp_f=G.synthesis.tri_plane_mlp_f, mlp_b=G.synthesis.tri_plane_mlp_b, template=G.synthesis.template, 
            geo_mlp=G.synthesis.geo_mlp, texture_mlp=G.synthesis.texture_mlp,
            geo_x=geo_feats, tex_x=tex_feats, scale=G.synthesis.cfg.dataset.cube_scale,
        ) # [batch_size, h * w * num_steps, num_feats]
        # coarse_output = coarse_output.view(batch_size, h * w, num_steps, 3 + 1 + 3 + 3) # [batch_size, h * w, num_steps, num_feats] | rgbs, sigmas, f_pts, b_pts
        surface_pts_rgb_sigma = output[...,:rgb_sigma_out_dim]
        surface_f_pts = output[..., rgb_sigma_out_dim:rgb_sigma_out_dim+3]
        surface_b_pts = output[..., rgb_sigma_out_dim+3:rgb_sigma_out_dim+6]
        surface_pts_sigma = surface_pts_rgb_sigma[:, :, -1:] # [batch_size, num_coords, 1]

        vertex_colors = (surface_f_pts.squeeze().cpu().numpy() + 0.3) / 0.6 * 255
        mesh.visual.vertex_colors[:, :3] = vertex_colors 
        """

        fig = go.Figure(data=[go.Scatter3d(
            x=coords_np[:,0], 
            y=coords_np[:,1],
            z=coords_np[:,2],
            mode='markers',
            marker=dict(
                size=8,
                color=['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(vcoords_colors[:,0], vcoords_colors[:,1], vcoords_colors[:,2])],
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