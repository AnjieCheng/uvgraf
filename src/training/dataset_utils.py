import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import torchvision
import glob
import PIL
import math
import numpy as np
import zipfile
import time
from scipy.io import loadmat

def read_pose(name,flip=False):
    P = loadmat(name)['angle']
    P_x = -(P[0,0] - 0.1) + math.pi/2 
    if not flip:
        P_y = P[0,1] # + math.pi/2 
    else:
        P_y = -P[0,1] # + math.pi/2 
    P = np.array([P_y, P_x, 0], dtype=np.float32)
    return P

def read_pose_npy(name,flip=False):
    P = np.load(name)
    P_x = P[0] + 0.14 # pitch
    if not flip:
        P_y = P[1] # yaw
    else:
        P_y = -P[1] + math.pi

    P = torch.tensor([P_y, P_x, 0], dtype=np.float32) # rt [yaw, pitch]
    return P

def transform_matrix_to_camera_pos(c2w, flip=False):
    """
    Get camera position with transform matrix
    :param c2w: camera to world transform matrix
    :return: camera position on spherical coord
    """
    c2w[[0,1,2]] = c2w[[1,2,0]]
    pos = c2w[:, -1].squeeze()
    radius = float(np.linalg.norm(pos))
    theta = float(np.arctan2(-pos[0], pos[2]))
    phi = float(np.arctan(-pos[1] / np.linalg.norm(pos[::2])))
    theta = theta + np.pi * 0.5
    phi = phi + np.pi * 0.5
    if flip:
        theta = -theta + math.pi
    P = np.array([theta, phi, 0], dtype=np.float32)
    return P
