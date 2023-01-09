# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import os
from glob import glob
import numpy as np
import zipfile
import math
import random
import torch
import dnnlib
import cv2
# import kaolin as kal
from typing import Tuple
from tqdm import tqdm 
import PIL.Image
from pathlib import Path
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode

try:
    import pyspng
except ImportError:
    pyspng = None

# ----------------------------------------------------------------------------
class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            path,
            resolution=64,  # Ensure specific resolution, None = highest available.
            split='train',
            limit_dataset_size=None,
            random_seed=0,
            **super_kwargs  # Additional arguments for the Dataset base class.
    ):
        self._name = "CompCars"
        self.has_labels = False
        self.label_shape = None
        
        self.image_path = Path(os.path.join(path, 'exemplars_highres'))
        self.mask_path = Path(os.path.join(path, 'exemplars_highres_mask'))
        self.mesh_path = Path(os.path.join(path, 'manifold_combined'))

        self.erode = True

        self.real_images_dict = {x.name.split('.')[0]: x for x in self.image_path.iterdir() if x.name.endswith('.jpg') or x.name.endswith('.png')}
        self.real_images_dict = dict(sorted(self.real_images_dict.items()))
        self.masks_dict = {x: self.mask_path / self.real_images_dict[x].name for x in self.real_images_dict}
        self.masks_dict = dict(sorted(self.masks_dict.items()))
        assert self.real_images_dict.keys() == self.masks_dict.keys()
        self.keys_list = list(self.real_images_dict.keys())
        self.num_images = len(self.keys_list)

        dpsr_car_path = os.path.join(path, 'shapenet_psr', '02958343')
        obj_names = os.listdir(self.mesh_path)
        self.point_cloud_paths = [os.path.join(dpsr_car_path, obj_name,'pointcloud.npz') for obj_name in obj_names]
        self.num_shapes = len(self.point_cloud_paths)

        """
        print("valid img checking...")
        for point_cloud_path in tqdm(point_cloud_paths):
            dense_points = np.load(point_cloud_path)
            surface_points_dense = dense_points['points'].astype(np.float32)
            surface_normals_dense = dense_points['normals'].astype(np.float32)
            pointcloud = np.concatenate([surface_points_dense, surface_normals_dense], axis=-1)

        for i in tqdm(self.real_images_dict.keys()):
            try:
                X = PIL.Image.open(self.real_images_dict[i])
                X.verify()
                X.close()
            except:
                print("skippng: ", self.real_images_dict[i])
        """

        self.img_size = resolution
        self.views_per_sample = 1

        print('==> use image path: %s, num images: %d' % (self.image_path, len(self.real_images_dict)))
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
        # get_image_and_view
        total_selections = len(self.real_images_dict.keys()) // 8
        available_views = get_car_views()
        view_indices = random.sample(list(range(8)), self.views_per_sample)
        # view_indices = [4]
        sampled_view = [available_views[vidx] for vidx in view_indices]
        image_indices = random.sample(list(range(total_selections)), self.views_per_sample)
        # image_indices = [1]
        image_selections = [f'{(iidx * 8 + vidx):05d}' for (iidx, vidx) in zip(image_indices, view_indices)]

        # get camera position
        azimuth = sampled_view[0]['azimuth'] # + (random.random() - 0.5) * self.camera_noise
        elevation = sampled_view[0]['elevation'] # + (random.random() - 0.5) * self.camera_noise
        cam_dist = sampled_view[0]['cam_dist'] # 
        fov = sampled_view[0]['fov'] # 
        angles = np.array([azimuth, elevation, 0, cam_dist, fov]).astype(np.float)

        fname = self.real_images_dict[image_selections[0]]
        fname_mask = self.masks_dict[image_selections[0]]
 
        img = self.process_real_image(fname)
        mask = self.process_real_mask(fname_mask)
        background = np.ones_like(img) * 255
        img = img * (mask == 0).astype(np.float) + background * (1 - (mask == 0).astype(np.float))

        # Load point cloud here...
        # sparse_points = np.load(self.sparse_points_list[idx])
        # idx = 0
        dense_points = np.load(self.point_cloud_paths[idx])

        # surface_points = sparse_points['points'].astype(np.float32)
        # surface_normals = sparse_points['normals'].astype(np.float32)

        surface_points_dense = (dense_points['points']).astype(np.float32)
        surface_normals_dense = dense_points['normals'].astype(np.float32)
        pointcloud = np.concatenate([surface_points_dense, surface_normals_dense], axis=-1)

        # print("///", angles.astype(np.float32), "///")

        return {
            'image': np.ascontiguousarray(img).astype(np.float32), # (3, 64, 64)
            'camera_angles': angles.astype(np.float32), # (5,)
            'mask': np.ascontiguousarray(mask).astype(np.float32), # (1, 64, 64)
            'pointcloud': np.ascontiguousarray(pointcloud).astype(np.float32), # (100000, 6)
        }

    def process_real_image(self, path):
        pad_size = int(self.img_size * 0.1)
        resize = T.Resize(size=(self.img_size - 2 * pad_size, self.img_size - 2 * pad_size), interpolation=InterpolationMode.BICUBIC)
        pad = T.Pad(padding=(pad_size, pad_size), fill=255)
        t_image = pad(torch.from_numpy(np.array(resize(PIL.Image.open(str(path)))).transpose((2, 0, 1))).float())
        return t_image.numpy()

    @staticmethod
    def erode_mask(mask):
        import cv2 as cv
        mask = mask.squeeze(0).numpy().astype(np.uint8)
        kernel_size = 3
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1), (kernel_size, kernel_size))
        mask = cv.erode(mask, element)
        return torch.from_numpy(mask).unsqueeze(0)

    def process_real_mask(self, path):
        pad_size = int(self.img_size * 0.1)
        resize = T.Resize(size=(self.img_size - 2 * pad_size, self.img_size - 2 * pad_size), interpolation=InterpolationMode.NEAREST)
        pad = T.Pad(padding=(pad_size, pad_size), fill=0)
        mask_im = read_image(str(path))[:1, :, :]
        if self.erode:
            eroded_mask = self.erode_mask(mask_im)
        else:
            eroded_mask = mask_im
        t_mask = pad(resize((eroded_mask > 128).float()))
        return (1 - (t_mask[:1, :, :]).float()).numpy()

    def get_camera_angles(self, idx):
        # get_image_and_view
        available_views = get_car_views()
        view_indices = random.sample(list(range(8)), self.views_per_sample)
        # view_indices = [4]
        sampled_view = [available_views[vidx] for vidx in view_indices]

        # get camera position
        azimuth = sampled_view[0]['azimuth'] # + (random.random() - 0.5) * self.camera_noise
        elevation = sampled_view[0]['elevation'] # + (random.random() - 0.5) * self.camera_noise
        cam_dist = sampled_view[0]['cam_dist'] # 
        fov = sampled_view[0]['fov'] # 
        angles = np.array([azimuth, elevation, 0, cam_dist, fov]).astype(np.float)
        return angles

    def _get_raw_camera_angles(self):
        if self._raw_camera_angles is None:
            self._raw_camera_angles = self._load_raw_camera_angles()
            if self._raw_camera_angles is None:
                self._raw_camera_angles = np.zeros([self._raw_shape[0], 3], dtype=np.float32)
            else:
                self._raw_camera_angles = self._raw_camera_angles.astype(np.float32)
            assert isinstance(self._raw_camera_angles, np.ndarray)
            assert self._raw_camera_angles.shape[0] == self._raw_shape[0]
        return self._raw_camera_angles

    def _load_raw_camera_angles(self):
        return None

    def _load_raw_image(self, raw_idx):
        if raw_idx >= len(self.keys_list): 
            raise KeyError(raw_idx)
        
        key = self.keys_list[raw_idx]
        if not os.path.exists(self.real_images_dict[key]):
            raise FileNotFoundError(self.real_images_dict[key])

        img = cv2.imread(str(self.real_images_dict[key]))[..., ::-1]
        img = (img / 255.0).transpose(2, 0, 1)
        return img

    def _load_raw_labels(self):
        return None

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        return self.img_size

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

def get_car_views():
    # front, back, right, left, front_right, front_left, back_right, back_left
    camera_distance = [3.2, 3.2, 1.7, 1.7, 1.5, 1.5, 1.5, 1.5]
    fov = [10, 10, 40, 40, 40, 40, 40, 40]
    azimuth = [3 * math.pi / 2, math.pi / 2,
               0, math.pi,
               math.pi + math.pi / 3, 0 - math.pi / 3,
               math.pi / 2 + math.pi / 6, math.pi / 2 - math.pi / 6]
    azimuth_noise = [0, 0,
                     0, 0,
                     (random.random() - 0.5) * math.pi / 7, (random.random() - 0.5) * math.pi / 7,
                     (random.random() - 0.5) * math.pi / 7, (random.random() - 0.5) * math.pi / 7,]
    elevation = [math.pi / 2, math.pi / 2,
                 math.pi / 2, math.pi / 2,
                 math.pi / 2 - math.pi / 48, math.pi / 2 - math.pi / 48,
                 math.pi / 2 - math.pi / 48, math.pi / 2 - math.pi / 48]
    elevation_noise = [-random.random() * math.pi / 32, -random.random() * math.pi / 32,
                       0, 0,
                       -random.random() * math.pi / 28, -random.random() * math.pi / 28,
                       -random.random() * math.pi / 28, -random.random() * math.pi / 28,]
    # azimuth_noise = [0]*8
    # elevation_noise = [0]*8
    return [{'azimuth': a + an + math.pi, 'elevation': e + en, 'fov': f, 'cam_dist': cd} for a, an, e, en, cd, f in zip(azimuth, azimuth_noise, elevation, elevation_noise, camera_distance, fov)]