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
import math
import random
import torch
import dnnlib
import cv2, json
from collections import defaultdict
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
        self._name = "PhotoShape"
        self.has_labels = False
        self.label_shape = None
        
        self.image_path = Path(os.path.join(path, 'exemplars'))
        self.mask_path = Path(os.path.join(path, 'exemplars_mask'))
        # self.mesh_path = Path(os.path.join(path, 'manifold_combined'))
        self.pairmeta_path = Path(os.path.join(path, 'metadata', 'pairs.json'))
        self.shapesmeta_path = Path(os.path.join(path, 'metadata', 'shapes.json'))
        self.erode = True
        
        single_mode = False
        limit_dataset_size = None
        if not single_mode:
            self.items = list(x.stem for x in self.image_path.iterdir())[:limit_dataset_size]
        else:
            self.items = ['shape02344_rank02_pair183269']
            if limit_dataset_size is None:
                self.items = self.items * 20000
            else:
                self.items = self.items * limit_dataset_size

        self.real_images_dict = {x.name.split('.')[0]: x for x in self.image_path.iterdir() if x.name.endswith('.jpg') or x.name.endswith('.png')}
        self.real_images_dict = dict(sorted(self.real_images_dict.items()))
        self.masks_dict = {x: self.mask_path / self.real_images_dict[x].name for x in self.real_images_dict}
        self.masks_dict = dict(sorted(self.masks_dict.items()))
        assert self.real_images_dict.keys() == self.masks_dict.keys()
        self.keys_list = list(self.real_images_dict.keys())
        self.num_images = len(self.keys_list)

        self.real_images_preloaded, self.masks_preloaded = {}, {}


        self.pair_meta, self.all_views, self.shape_meta = self.load_pair_meta(self.pairmeta_path, self.shapesmeta_path)
        self.source_id_list = [self.shape_meta[shape_meta_key]['source_id'] for shape_meta_key in self.shape_meta.keys()]

        self.dpsr_chair_path = os.path.join(path, 'shapenet_psr', '03001627')

        # split_file = os.path.join(dpsr_chair_path, 'train' + '.lst')
        # with open(split_file, 'r') as f:
        #     models_c = f.read().split('\n')
        # if '' in models_c:
        #     models_c.remove('')
        # obj_names = os.listdir(dpsr_chair_path)
        # self.point_cloud_paths = [os.path.join(dpsr_chair_path, model, 'pointcloud.npz') for model in models_c]

        # obj_names = os.listdir(self.mesh_path)
        # self.point_cloud_paths = [os.path.join(dpsr_car_path, obj_name,'pointcloud.npz') for obj_name in obj_names]
        self.num_shapes = len(self.source_id_list)

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
        # print('==> use mesh path: %s, num meshes: %d' % (self.mesh_path, len(self.point_cloud_paths)))
        self._raw_shape = [len(self.real_images_dict)] + list(self._load_raw_image(0).shape)

        self._raw_camera_angles = None
        # Apply max_size.
        self._raw_idx = np.arange(self.num_shapes, dtype=np.int64)
        if (limit_dataset_size is not None) and (self._raw_idx.size > limit_dataset_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:limit_dataset_size])

        self.__getitem__(0)

    def __len__(self):
        return len(self.items)

    def _open_file(self, fname):
        return open(fname, 'rb')

    def __getitem__(self, idx):
        # get_image_and_view
        selected_item = self.items[idx]
        
        shape_id = int(selected_item.split('_')[0].split('shape')[1])
        shapenet_id = self.shape_meta[shape_id]['source_id']

        sampled_view = random.sample(self.all_views, self.views_per_sample)
        image_selections = self.get_image_selections(shape_id)
        images, masks, cameras, cam_positions = [], [], [], []
        for c_i, c_v in zip(image_selections, sampled_view):
            import pdb; pdb.set_trace()
            images.append(self.get_real_image(self.meta_to_pair(c_i)))
            masks.append(self.get_real_mask(self.meta_to_pair(c_i)))
            azimuth = c_v['azimuth'] # + (random.random() - 0.5) * self.camera_noise
            elevation = c_v['elevation'] # + (random.random() - 0.5) * self.camera_noise
            fov = c_v['fov']
            radius = c_v['distance']
            import pdb; pdb.set_trace()
            # perspective_cam = spherical_coord_to_cam(c_i['fov'], azimuth, elevation)
            # projection_matrix = intrinsic_to_projection(get_default_perspective_cam()).float()
            # projection_matrix = torch.from_numpy(perspective_cam.projection_mat()).float()
            # view_matrix = torch.from_numpy(np.linalg.inv(generate_camera(np.zeros(3), c['azimuth'], c['elevation']))).float()
            # cam_position = torch.from_numpy(np.linalg.inv(perspective_cam.view_mat())[:3, 3]).float()
            view_matrix = torch.from_numpy(perspective_cam.view_mat()).float()
            cameras.append(torch.matmul(projection_matrix, view_matrix))
            cam_positions.append(cam_position)
        image = torch.cat(images, dim=0)
        mask = torch.cat(masks, dim=0)
        mvp = torch.stack(cameras, dim=0)
        cam_positions = torch.stack(cam_positions, dim=0)
        return image, mask, mvp, cam_positions




        total_selections = len(self.real_images_dict.keys())
        selected_indices = random.sample(list(range(total_selections)), self.views_per_sample)
        image_selections = self.real_images_dict


        available_views = get_car_views()
        view_indices = random.sample(list(range(8)), self.views_per_sample)
        sampled_view = [available_views[vidx] for vidx in view_indices]
        image_indices = random.sample(list(range(total_selections)), self.views_per_sample)
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
        dense_points = np.load(self.point_cloud_paths[idx])

        # surface_points = sparse_points['points'].astype(np.float32)
        # surface_normals = sparse_points['normals'].astype(np.float32)

        surface_points_dense = (dense_points['points'] * 0.95).astype(np.float32)
        surface_normals_dense = dense_points['normals'].astype(np.float32)
        pointcloud = np.concatenate([surface_points_dense, surface_normals_dense], axis=-1)

        return {
            'image': np.ascontiguousarray(img).astype(np.float32),
            'camera_angles': angles.astype(np.float32),
            'mask': np.ascontiguousarray(mask).astype(np.float32),
            'pointcloud': np.ascontiguousarray(pointcloud).astype(np.float32),
        }

    def load_pair_meta(self, pairmeta_path, shapesmeta_path):
        loaded_json = json.loads(Path(pairmeta_path).read_text())
        loaded_json_shape = json.loads(Path(shapesmeta_path).read_text())
        ret_shapedict = {}
        ret_dict = defaultdict(list)
        ret_views = []
        for k in loaded_json.keys():
            if self.meta_to_pair(loaded_json[k]) in self.real_images_dict.keys():
                shape_id = loaded_json[k]['shape_id']
                ret_dict[shape_id].append(loaded_json[k])
                ret_views.append(loaded_json[k])
                ret_shapedict[shape_id] = loaded_json_shape[str(shape_id)]
        return ret_dict, ret_views, ret_shapedict

    def get_image_selections(self, shape_id):
        candidates = self.pair_meta[shape_id]
        if len(candidates) < self.views_per_sample:
            while len(candidates) < self.views_per_sample:
                meta = self.pair_meta[random.choice(list(self.pair_meta.keys()))]
                candidates.extend(meta[:self.views_per_sample - len(candidates)])
        else:
            candidates = random.sample(candidates, self.views_per_sample)
        return candidates

    def get_real_image(self, name):
        if name not in self.real_images_preloaded.keys():
            return self.process_real_image(self.real_images_dict[name])
        else:
            return self.real_images_preloaded[name]

    def get_real_mask(self, name):
        if name not in self.masks_preloaded.keys():
            return self.process_real_mask(self.masks_dict[name])
        else:
            return self.masks_preloaded[name]

    def process_real_image(self, path):
        resize = T.Resize(size=(self.img_size, self.img_size))
        pad = T.Pad(padding=(100, 100), fill=1)
        t_image = resize(pad(read_image(str(path)).float() / 127.5 - 1))
        return (t_image.unsqueeze(0))

    def process_real_mask(self, path):
        resize = T.Resize(size=(self.img_size, self.img_size))
        pad = T.Pad(padding=(100, 100), fill=0)
        if self.erode:
            eroded_mask = self.erode_mask(read_image(str(path)))
        else:
            eroded_mask = read_image(str(path))
        t_mask = resize(pad((eroded_mask > 0).float()))
        return t_mask.unsqueeze(0)

    @staticmethod
    def erode_mask(mask):
        import cv2 as cv
        mask = mask.squeeze(0).numpy().astype(np.uint8)
        kernel_size = 3
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1), (kernel_size, kernel_size))
        mask = cv.erode(mask, element)
        return torch.from_numpy(mask).unsqueeze(0)

    @staticmethod
    def meta_to_pair(c):
        return f'shape{c["shape_id"]:05d}_rank{(c["rank"] - 1):02d}_pair{c["id"]}'


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