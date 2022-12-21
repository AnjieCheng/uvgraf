import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch
from collections import OrderedDict
import plotly.graph_objects as go

def generate_sampled_points(N=128):
    voxel_origin = [-0.5, -0.5, -0.5]
    voxel_size = 1.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3
    return samples

def sphere_to_color(sphere_coords, radius):
    sphere_coords = np.copy(sphere_coords)
    sphere_coords = np.clip(sphere_coords/(radius*2)+0.5,0,1) # normalize color to 0-1
    sphere_coords = np.clip((sphere_coords * 255), 0, 255).astype(int) # normalize color 0-255
    return sphere_coords

def plotly_gen_mesh(verts, faces, color=None, rt_html=True, png_path=None):
    if color is None:
        vertexcolor = None
    else:
        vertexcolor = ['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(color[:,0], color[:,1], color[:,2])]
    fig = go.Figure(data=[go.Mesh3d(
        x=verts[:,0],
        y=verts[:,2],
        z=verts[:,1],
        i=faces[:,0],
        j=faces[:,2],
        k=faces[:,1],
        vertexcolor=vertexcolor,
        opacity=1.0,
    )])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    if png_path:
        fig.write_image(png_path)
    if rt_html:
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    else:
        return fig

def plotly_gen_points(points, color=None, rt_html=True, png_path=None):
    if torch.is_tensor(points):
        points = points.cpu().numpy()

    if color is None:
        pointcolor = None
    else:
        pointcolor = ['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(color[:,0], color[:,1], color[:,2])]

    fig = go.Figure(data=[go.Scatter3d(
        x=points[:,0],
        y=points[:,2],
        z=points[:,1],
        mode='markers',
        marker=dict(
            size=3,
            color=pointcolor,
            opacity=1
        )
    )])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    if png_path:
        fig.write_image(png_path)
    if rt_html:
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    else:
        return fig

def convert_sdf_samples_to_mesh(
    pytorch_3d_sdf_tensor,
    N,
    voxel_origin=[-0.5, -0.5, -0.5],
    offset=None,
    scale=None,
    level=0.0,
):
    voxel_size = (1.0/(N-1))
    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
        )
    except:
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    return mesh_points, faces, normals, values

    # # try writing to the ply file

    # num_verts = verts.shape[0]
    # num_faces = faces.shape[0]

    # verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    # for i in range(0, num_verts):
    #     verts_tuple[i] = tuple(mesh_points[i, :])

    # faces_building = []
    # for i in range(0, num_faces):
    #     faces_building.append(((faces[i, :].tolist(),)))
    # faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    # el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    # el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    # ply_data = plyfile.PlyData([el_verts, el_faces])
    # logging.debug("saving mesh to %s" % (ply_filename_out))
    # ply_data.write(ply_filename_out)

    # logging.debug(
    #     "converting to ply format and writing to file took {} s".format(
    #         time.time() - start_time
    #     )
    # )