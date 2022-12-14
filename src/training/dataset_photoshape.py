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
import torch
import dnnlib
import cv2
# import kaolin as kal
from typing import Tuple
from tqdm import tqdm 
import PIL.Image
from pathlib import Path

try:
    import pyspng
except ImportError:
    pyspng = None

# ----------------------------------------------------------------------------
class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            resolution=64,  # Ensure specific resolution, None = highest available.
            split='train',
            limit_dataset_size=None,
            random_seed=0,
            **super_kwargs  # Additional arguments for the Dataset base class.
    ):
        self._name = "CompCars"
        self.has_labels = False
        self.label_shape = None
        self.image_path = Path("/home/anjie/Downloads/CADTextures/Photoshape/exemplars")
        self.mask_path = Path("/home/anjie/Downloads/CADTextures/Photoshape/exemplars_mask")
        self.mesh_path = Path("/home/anjie/Downloads/CADTextures/Photoshape/shapenet-chairs-manifold-highres")

        self.real_images_dict = {x.name.split('.')[0]: x for x in self.image_path.iterdir() if x.name.endswith('.jpg') or x.name.endswith('.png')}
        self.real_images_dict = dict(sorted(self.real_images_dict.items()))
        self.masks_dict = {x: self.mask_path / self.real_images_dict[x].name for x in self.real_images_dict}
        self.masks_dict = dict(sorted(self.masks_dict.items()))
        assert self.real_images_dict.keys() == self.masks_dict.keys()
        self.keys_list = list(self.real_images_dict.keys())
        self.num_images = len(self.keys_list)

        self.sparse_points_list = [y for x in os.walk(self.mesh_path) for y in glob(os.path.join(x[0], '4096_pointcloud.npz'))]
        self.dense_points_list = [y for x in os.walk(self.mesh_path) for y in glob(os.path.join(x[0], '50000_pointcloud.npz'))]
        assert len(self.sparse_points_list) == len(self.dense_points_list)
        self.num_shapes = len(self.sparse_points_list)

        """
        print("valid img checking...")
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
        print('==> use mesh path: %s, num meshes: %d' % (self.mesh_path, len(self.sparse_points_list)))
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
        sampled_view = [available_views[vidx] for vidx in view_indices]
        image_indices = random.sample(list(range(total_selections)), self.views_per_sample)
        image_selections = [f'{(iidx * 8 + vidx):05d}' for (iidx, vidx) in zip(image_indices, view_indices)]

        # get camera position
        azimuth = sampled_view[0]['azimuth'] # + (random.random() - 0.5) * self.camera_noise
        elevation = sampled_view[0]['elevation'] # + (random.random() - 0.5) * self.camera_noise
        angles = np.array([azimuth, elevation, 0]).astype(np.float)

        fname = self.real_images_dict[image_selections[0]]
        fname_mask = self.masks_dict[image_selections[0]]

        with self._open_file(str(fname)) as f:
            ori_img = np.array(PIL.Image.open(f))
            # if pyspng is not None and self._file_ext(str(fname)) == '.png':
            #     ori_img = pyspng.load(f.read())
            # else:
            #     ori_img = np.array(PIL.Image.open(f))
            # PIL.Image.open(f).show()

        with self._open_file(str(fname_mask)) as f:
            ori_mask = np.array(PIL.Image.open(f))

            # if pyspng is not None and self._file_ext(str(fname_mask)) == '.png':
            #     ori_mask = pyspng.load(f.read())
            # PIL.Image.open(f).show()
            # # else:
            # # ori_mask = np.array(PIL.Image.open(f))

        img = ori_img[:, :, :3]

        if ori_mask.ndim == 3:
            mask = (ori_mask[:, :, 0:1] / 255)
        elif ori_mask.ndim == 2:
            mask = ori_mask[:, :, None]
        else:
            raise ValueError(ori_mask.ndim)

        # print(mask.shape)

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST) ########

        background = np.ones_like(img) * 255
        img = img * (mask[:,:,None] > 0).astype(np.float) + background * (1 - (mask[:,:,None] > 0).astype(np.float))

        img = img.transpose(2, 0, 1) # HWC => CHW

        # Load point cloud here...
        # sparse_points = np.load(self.sparse_points_list[idx])
        dense_points = np.load(self.dense_points_list[idx])

        # surface_points = sparse_points['points'].astype(np.float32)
        # surface_normals = sparse_points['normals'].astype(np.float32)

        surface_points_dense = dense_points['points'].astype(np.float32)
        surface_normals_dense = dense_points['normals'].astype(np.float32)
        pointcloud = np.concatenate([surface_points_dense, surface_normals_dense], axis=-1)

        return {
            'image': np.ascontiguousarray(img).astype(np.float),
            'camera_angles': angles.astype(np.float),
            'mask': np.ascontiguousarray(mask)[None,:,:].astype(np.float),
            'pointcloud': np.ascontiguousarray(pointcloud).astype(np.float),
        }

    def get_camera_angles(self, idx):
        # get view
        available_views = get_car_views()
        view_indices = random.sample(list(range(8)), self.views_per_sample)
        sampled_view = [available_views[vidx] for vidx in view_indices]

        # get camera position
        azimuth = sampled_view[0]['azimuth']
        elevation = sampled_view[0]['elevation']
        angles = np.array([azimuth, elevation, 0]).astype(np.float)
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
