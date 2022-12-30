from omegaconf import DictConfig
# from ast import DictComp
# import plyfile
# import argparse
import torch
import numpy as np
# import skimage.measure
# import scipy
import trimesh
import os
import hydra
from tqdm import tqdm
from pathlib import Path


from src.training.networks_geometry import FoldSDF

from src.training.training_utils import run_batchwise, sample_sphere_points, get_feat_from_triplane
from src.training.networks_canograf_dep import canonical_renderer_pretrain
from scripts.utils import load_generator, set_seed, create_voxel_coords


# ----------------------------------------------------------------------------
class ShapeDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            path,
            split='train',
            limit_dataset_size=None,
            random_seed=0,
    ):
        self.mesh_path = Path(os.path.join(path, 'manifold_combined'))

        dpsr_car_path = os.path.join(path, 'shapenet_psr', '02958343')
        obj_names = os.listdir(self.mesh_path)
        self.point_cloud_paths = [os.path.join(dpsr_car_path, obj_name,'pointcloud.npz') for obj_name in obj_names]
        self.num_shapes = len(self.point_cloud_paths)

        print('==> use mesh path: %s, num meshes: %d' % (self.mesh_path, len(self.point_cloud_paths)))
        self._raw_shape = [len(self.real_images_dict)] + list(self._load_raw_image(0).shape)

        self._raw_camera_angles = None
        # Apply max_size.
        self._raw_idx = np.arange(self.num_shapes, dtype=np.int64)
        if (limit_dataset_size is not None) and (self._raw_idx.size > limit_dataset_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:limit_dataset_size])

        self.__getitem__(0)

    def __len__(self):
        return self._raw_idx.size

    def _open_file(self, fname):
        return open(fname, 'rb')

    def __getitem__(self, idx):
        # Load point cloud here...
        # sparse_points = np.load(self.sparse_points_list[idx])
        dense_points = np.load(self.point_cloud_paths[idx])

        # surface_points = sparse_points['points'].astype(np.float32)
        # surface_normals = sparse_points['normals'].astype(np.float32)

        surface_points_dense = (dense_points['points'] * 0.95).astype(np.float32)
        surface_normals_dense = dense_points['normals'].astype(np.float32)
        pointcloud = np.concatenate([surface_points_dense, surface_normals_dense], axis=-1)

        return {
            'pointcloud': np.ascontiguousarray(pointcloud).astype(np.float32),
        }

#----------------------------------------------------------------------------

@hydra.main(config_path="../configs/scripts", config_name="extract_correspondence.yaml")
def extract_geometry(cfg: DictConfig):
    device = torch.device('cuda')
    G = load_generator(cfg.ckpt, verbose=cfg.verbose)[0].to(device).eval()


    fig_list = []
    for seed in tqdm(cfg.seeds, desc='Extracting geometry...'):
        set_seed(seed)
        batch_size = 1
        z = torch.randn(batch_size, G.z_dim, device=device) # [batch_size, z_dim]
        c = torch.zeros(batch_size, G.c_dim, device=device) # [batch_size, c_dim]
        assert G.c_dim == 0
        # sphere_mesh = create_sphere_mesh() # [batch_size, voxel_res ** 3, 3]
        # coords = torch.tensor(sphere_mesh.vertices)
        # coords = coords.to(z.device) # [batch_size, voxel_res ** 3, 3]
        coords = create_voxel_coords(cfg.voxel_res, cfg.voxel_origin, cfg.cube_size, batch_size) # [batch_size, voxel_res ** 3, 3]
        coords = coords.to(z.device) # [batch_size, voxel_res ** 3, 3]

        geo_z = z[...,:G.geo_dim]
        tex_z = z[...,-G.tex_dim:]
        geo_ws = G.geo_mapping(geo_z, c, truncation_psi=cfg.truncation_psi)
        tex_ws = G.tex_mapping(tex_z, c, truncation_psi=cfg.truncation_psi)
        geo_global_feat = geo_ws[:, G.synthesis.tri_plane_decoder.num_ws:G.synthesis.tri_plane_decoder.num_ws+1]

        geo_feats = G.synthesis.tri_plane_decoder(geo_ws[:, :G.synthesis.tri_plane_decoder.num_ws]) # [batch_size, 3 * feat_dim, tp_h, tp_w]
        tex_feats = G.synthesis.texture_decoder(tex_ws[:, :G.synthesis.texture_decoder.num_ws]) # [batch_size, feat_dim, tp_h, tp_w]
        ray_d_world = torch.zeros_like(coords) # [batch_size, num_points, 3]


        sphere_samples = sample_sphere_points(radius=0.3, num_points=8192)
        batched_sphere_samples = sphere_samples.unsqueeze(0).expand(batch_size, -1, -1).to(coords.device)
        batched_sphere_samples_np = batched_sphere_samples.squeeze().cpu().numpy()
        geo_feat_f = get_feat_from_triplane(batched_sphere_samples, geo_feats, scale=None)
        sphere_samples_b = torch.tanh(G.synthesis.tri_plane_mlp_b(geo_feat_f, sphere_samples, ray_d_world, geo_ws)) * G.synthesis.cfg.dataset.cube_scale
        sphere_samples_b_np = sphere_samples_b.squeeze().cpu().numpy()
        sphere_samples_colors = (batched_sphere_samples_np + 1) / 2 * 255
        sphere_samples_colors = np.clip(batched_sphere_samples_np, 0, 255).astype(int)

        fig = go.Figure(data=[go.Scatter3d(
            x=sphere_samples_b_np[:,0],
            y=sphere_samples_b_np[:,1],
            z=sphere_samples_b_np[:,2],
            mode='markers',
            marker=dict(
                size=4,
                color=['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(sphere_samples_colors[:,0], sphere_samples_colors[:,1], sphere_samples_colors[:,2])],
                opacity=1
            )
        )])
        fig_list.append(dcc.Graph(id="seed_%d"%(seed), figure=fig))

        
        output = run_batchwise(
            fn=canonical_renderer_pretrain, data=dict(coords=coords, ray_d_world=ray_d_world),
            batch_size=cfg.max_batch_res ** 3, dim=1, 
            mlp_f=G.synthesis.tri_plane_mlp_f, mlp_b=G.synthesis.tri_plane_mlp_b, template=G.synthesis.template, 
            texture_mlp=G.synthesis.texture_mlp, geo_x=geo_feats, tex_x=tex_feats, scale=G.synthesis.cfg.dataset.cube_scale,
            geo_ws=geo_global_feat, tex_ws=None, to_rgb=None)
        output = output.detach()

        # [batch_size, h * w * num_steps, num_feats]
        # coarse_output = coarse_output.view(batch_size, h * w, num_steps, 3 + 1 + 3 + 3) # [batch_size, h * w, num_steps, num_feats] | rgbs, sigmas, f_pts, b_pts
        
        rgb_sigma_out_dim = 4
        output_rgb_sigma = output[...,:rgb_sigma_out_dim]
        output_f_pts = output[..., rgb_sigma_out_dim:rgb_sigma_out_dim+3]
        output_b_pts = output[..., rgb_sigma_out_dim+3:rgb_sigma_out_dim+6]
        sigma = output_rgb_sigma[:, :, -1:] # [batch_size, num_coords, 1]

        # filter out zero sigma points
        coords_np = coords[0][sigma.squeeze()>1].squeeze().cpu().numpy() # 
        output_f_pts_np = output_f_pts[0][sigma.squeeze()>1].squeeze().cpu().numpy()
        output_b_pts_np = output_b_pts[0][sigma.squeeze()>1].squeeze().cpu().numpy()

        vcoords_colors = (output_f_pts_np + 1) / 2 * 255
        vcoords_colors = np.clip(vcoords_colors, 0, 255).astype(int)

        f_dir = output_f_pts_np - coords_np
        b_dir = output_b_pts_np - output_f_pts_np

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
        # fig_list.append(dcc.Graph(id="seed_%d"%(seed), figure=fig))

        # app.layout = html.Div([
        #     html.H4('Visualizing Correspondence'),
        #     html.Pre(
        #         id='structure',
        #         style={
        #             'border': 'thin lightgrey solid', 
        #             'overflowY': 'scroll',
        #             'height': '275px'
        #         }
        #     ),
        # ])

        # # sigma = G.synthesis.compute_densities(geo_ws, tex_ws, coords, max_batch_res=cfg.max_batch_res) # [batch_size, voxel_res ** 3, 1]
        # assert batch_size == 1
        sigma = sigma.reshape(cfg.voxel_res, cfg.voxel_res, cfg.voxel_res).cpu().numpy() # [voxel_res ** 3]

        print('sigma percentiles:', {q: np.percentile(sigma.reshape(-1), q) for q in [50.0, 90.0, 95.0, 97.5, 99.0, 99.5]})
        # vertices, triangles = mcubes.marching_cubes(sigma, np.percentile(sigma, cfg.thresh_percentile))
        # vertices, triangles = mcubes.marching_cubes(sigma, 10)
        # mesh = trimesh.Trimesh(vertices, triangles)
        # os.makedirs('shapes', exist_ok=True)
        # mesh.export(f'shapes/shape-{seed}.obj')

        os.makedirs(cfg.output_dir, exist_ok=True)
        with mrcfile.new_mmap(os.path.join(cfg.output_dir, f'{seed}.mrc'), overwrite=True, shape=sigma.shape, mrc_mode=2) as mrc:
            mrc.data[:] = sigma

        sdf = output[:, :, -1:] # [batch_size, num_coords, 1]
        sdf = sdf.reshape(cfg.voxel_res, cfg.voxel_res, cfg.voxel_res).cpu().numpy()
        os.makedirs(cfg.output_dir, exist_ok=True)
        with mrcfile.new_mmap(os.path.join(cfg.output_dir, f'{seed}_sdf.mrc'), overwrite=True, shape=sdf.shape, mrc_mode=2) as mrc:
            mrc.data[:] = -sdf

        vertices, triangles = mcubes.marching_cubes(sigma, 0.)
        mesh = trimesh.Trimesh(vertices, triangles)
        os.makedirs('shapes', exist_ok=True)
        mesh.export(f'shapes/shape-{seed}.obj')

    app.layout = html.Div(children=fig_list)
    app.run_server(debug=False)
#----------------------------------------------------------------------------

if __name__ == "__main__":
    extract_geometry() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------