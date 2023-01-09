import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import trimesh
import random
from pytorch3d.utils import ico_sphere

from torch.nn import AvgPool2d, Conv1d, Conv2d, Embedding, LeakyReLU, Module

neg = 0.01
neg_2 = 0.2

class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channel, style_dim, use_eql=False):
        super().__init__()
        # Conv = EqualConv1d if use_eql else nn.Conv1d
        Conv = nn.Conv1d

        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = Conv(style_dim, in_channel * 2, 1)

        self.style.weight.data.normal_()
        self.style.bias.data.zero_()

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out

class EdgeBlock(nn.Module):
    """ Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """
    def __init__(self, Fin, Fout, k, attn=True):
        super(EdgeBlock, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.conv_w = nn.Sequential(
            nn.Conv2d(Fin, Fout//2, 1),
            nn.BatchNorm2d(Fout//2),
            nn.LeakyReLU(neg, inplace=True),
            nn.Conv2d(Fout//2, Fout, 1),
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(neg, inplace=True)
        )

        self.conv_x = nn.Sequential(
            nn.Conv2d(2 * Fin, Fout, [1, 1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(neg, inplace=True)
        )

        self.conv_out = nn.Conv2d(Fout, Fout, [1, k], [1, 1])  # Fin, Fout, kernel_size, stride



    def forward(self, x):
        B, C, N = x.shape
        x = get_edge_features(x, self.k) # [B, 2Fin, N, k]
        w = self.conv_w(x[:, C:, :, :])
        w = F.softmax(w, dim=-1)  # [B, Fout, N, k] -> [B, Fout, N, k]

        x = self.conv_x(x)  # Bx2CxNxk
        x = x * w  # Bx2CxNxk

        x = self.conv_out(x)  # [B, 2*Fout, N, 1]

        x = x.squeeze(3)  # BxCxN

        return x

def get_edge_features(x, k, num=-1, idx=None, return_idx=False):
    """
    Args:
        x: point cloud [B, dims, N]
        k: kNN neighbours
    Return:
        [B, 2dims, N, k]    
    """
    B, dims, N = x.shape

    # batched pair-wise distance
    if idx is None:
        xt = x.permute(0, 2, 1)
        xi = -2 * torch.bmm(xt, x)
        xs = torch.sum(xt**2, dim=2, keepdim=True)
        xst = xs.permute(0, 2, 1)
        dist = xi + xs + xst # [B, N, N]

        # get k NN id
        _, idx_o = torch.sort(dist, dim=2)
        idx = idx_o[: ,: ,1:k+1] # [B, N, k]
        idx = idx.contiguous().view(B, N*k)


    # gather
    neighbors = []
    for b in range(B):
        tmp = torch.index_select(x[b], 1, idx[b]) # [d, N*k] <- [d, N], 0, [N*k]
        tmp = tmp.view(dims, N, k).contiguous()
        neighbors.append(tmp)

    neighbors = torch.stack(neighbors) # [B, d, N, k]

    # centralize
    central = x.unsqueeze(3) # [B, d, N, 1]
    central = central.repeat(1, 1, 1, k) # [B, d, N, k]

    ee = torch.cat([central, neighbors-central], dim=1)
    assert ee.shape == (B, 2*dims, N, k)

    if return_idx:
        return ee, idx
    return ee

class SPGANGenerator(nn.Module):
    def __init__(self, z_dim, add_dim=3, use_local=True, use_tanh=False, norm_z=False, use_attn=False):
        super(SPGANGenerator, self).__init__()
        self.np = 2048
        self.nk = 10 # 20//2
        self.nz = z_dim
        self.z_dim = z_dim
        self.off = False
        self.use_attn = False
        self.use_head = False
        self.use_local = use_local
        self.use_tanh = use_tanh
        self.norm_z = norm_z
        self.use_attn = use_attn

        # Conv = EqualConv1d if self.opts.eql else nn.Conv1d
        Conv = nn.Conv1d
        # Linear = EqualLinear if self.opts.eql else nn.Linear
        Linear = nn.Linear

        dim = 128
        self.head = nn.Sequential(
            Conv(add_dim + self.nz, dim, 1),
            #nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
            Conv(dim, dim, 1),
            #nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
        )

        if self.use_attn:
            self.attn = Attention(dim + 512)

        self.global_conv = nn.Sequential(
            Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
            Linear(dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(neg, inplace=True),
        )

        if self.use_tanh:
            self.tail_p = nn.Sequential(
                Conv1d(512+dim, 256, 1),
                nn.LeakyReLU(neg, inplace=True),
                Conv1d(256, 64, 1),
                nn.LeakyReLU(neg, inplace=True),
                Conv1d(64, 3, 1),
                nn.Tanh()
            )
        else:
            self.tail_p = nn.Sequential(
                Conv1d(512+dim, 256, 1),
                nn.LeakyReLU(neg, inplace=True),
                Conv1d(256, 64, 1),
                nn.LeakyReLU(neg, inplace=True),
                Conv1d(64, 3, 1),
            )

        if self.use_head:
            self.pc_head = nn.Sequential(
                Conv(add_dim, dim // 2, 1),
                nn.LeakyReLU(inplace=True),
                Conv(dim // 2, dim, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.EdgeConv1 = EdgeBlock(dim, dim, self.nk)
            self.adain1 = AdaptivePointNorm(dim, dim)
            self.EdgeConv2 = EdgeBlock(dim, dim, self.nk)
            self.adain2 = AdaptivePointNorm(dim, dim)
        else:
            self.EdgeConv1 = EdgeBlock(add_dim, 64, self.nk)
            self.adain1 = AdaptivePointNorm(64, dim)
            self.EdgeConv2 = EdgeBlock(64, dim, self.nk)
            self.adain2 = AdaptivePointNorm(dim, dim)

        self.lrelu1 = nn.LeakyReLU(neg_2)
        self.lrelu2 = nn.LeakyReLU(neg_2)


    def forward(self, z, x):
        B,N,_ = x.size()
        if self.norm_z:
            z = z / (z.norm(p=2, dim=-1, keepdim=True)+1e-8) 
        style = torch.cat([x, z], dim=-1)
        style = style.transpose(2, 1).contiguous()
        style = self.head(style)  # B,C,N

        pc = x.transpose(2, 1).contiguous()
        if self.use_head:
            pc = self.pc_head(pc)

        x1 = self.EdgeConv1(pc)
        x1 = self.lrelu1(x1)
        x1 = self.adain1(x1, style)

        x2 = self.EdgeConv2(x1)
        x2 = self.lrelu2(x2)
        x2 = self.adain2(x2, style)

        feat_global = torch.max(x2, 2, keepdim=True)[0]
        feat_global = feat_global.view(B, -1).contiguous()
        feat_global = self.global_conv(feat_global)
        feat_global = feat_global.view(B, -1, 1).contiguous()
        feat_global = feat_global.repeat(1, 1, N)

        if self.use_local:
            feat_cat = torch.cat((feat_global, x2), dim=1)
        else:
            feat_cat = feat_global

        if self.use_attn:
            feat_cat = self.attn(feat_cat)

        x1_p = self.tail_p(feat_cat).transpose(1,2).contiguous()  # Bx3x256

        if self.use_tanh:
            x1_p = x1_p / 2

        return x1_p

def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]            # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1).contiguous()*num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1).contiguous()*num_points
    idx = idx + idx_base
    idx = idx.view(-1).contiguous()

    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points).contiguous()      # (batch_size, num_dims, num_points)
    if idx is None:
        idx = knn(x, k=k)                       # (batch_size, num_points, k)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()          # (batch_size, num_points, num_dims)
    feature = x.view(batch_size*num_points, -1)[idx, :]                 # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)         # (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (batch_size, num_points, k, num_dims)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)      # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)
  
    return feature                              # (batch_size, 2*num_dims, num_points, k)

class DGCNN(nn.Module):
    def __init__(self, feat_dim):
        super(DGCNN, self).__init__()
        self.k = 20
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(feat_dim)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, feat_dim, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.global_conv = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, x):
        x = x.transpose(2, 1).contiguous()

        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)
        
        local_x = self.conv6(x)                 # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        
        global_x = local_x.max(dim=-1, keepdim=False)[0]     # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)

        global_x = local_x.max(dim=-1, keepdim=True)[0] # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        global_conv_x = self.global_conv(global_x.squeeze(dim=-1))

        # global_x = global_x.unsqueeze(1)          # (batch_size, num_points) -> (batch_size, 1, emb_dims)

        return global_conv_x, local_x.permute(0, 2, 1).contiguous()
        # return feat                             # (batch_size, 1, emb_dims)


class SphereTemplate(nn.Module):
    def __init__(self, radius=0.5):
        super().__init__()
        self.dim = 3
        self.radius = radius
        self.ico_sphere = {}
        for i in range(6):
            points = ico_sphere(i)._verts_list[0] * self.radius 
            rand_indx = torch.randperm(ico_sphere(i)._verts_list[0].size(0))
            self.ico_sphere[i] = points[rand_indx]
            # 4: 2562
            # 5: 10242
            # 6: 40962

    def get_regular_points(self, level=4, batch_size=None):
        if batch_size is not None:
            points = self.ico_sphere[level][None, : , :]
            points = points.expand(batch_size, -1, -1)
            return points
        else:
            return self.ico_sphere[level]

    def get_random_points(self, num_points=2048, batch_size=None):
        if batch_size is not None:
            rnd = torch.randn(batch_size, num_points, 3, dtype=torch.float)
        else:
            rnd = torch.randn(num_points, 3, dtype=torch.float)
        sphere_samples = (rnd / torch.norm(rnd, dim=-1, keepdim=True)) * self.radius
        return sphere_samples

    def forward(self, points):
        assert points.size(-1) == 3
        points_flat = points.reshape(-1, 3)
        sdfs_flat = torch.linalg.norm(points_flat, dim=-1) - self.radius
        sdfs = sdfs_flat.reshape(points.shape[0], points.shape[1])[...,None]
        return sdfs

import os
from pathlib import Path

class FoldSDF(nn.Module):
    def __init__(self, feat_dim, cfg, ckpt_path=None, ignore_keys=[], name=None, preload_data=True):
        super().__init__()
        self.cfg = cfg
        self.feat_dim = feat_dim

        self.template = SphereTemplate()
        self.Encoder = DGCNN(feat_dim=feat_dim)
        self.Fold_P = SPGANGenerator(z_dim=feat_dim, add_dim=3, use_local=True, use_tanh=True)
        self.Fold_N = SPGANGenerator(z_dim=feat_dim, add_dim=3, use_local=True, use_tanh=False)
        self.dpsr = DPSR(res=(128, 128, 128), sig=4)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            self.is_preloaded = False

    def preload(self, batch_size, device):
        if not self.is_preloaded:
            if 'compcars' in self.cfg.dataset.name:
                # load paths
                dpsr_car_path = os.path.join(self.cfg.dataset.path, 'shapenet_psr', '02958343')
                texturify_obj_names = os.listdir(Path(os.path.join(self.cfg.dataset.path, 'manifold_combined')))
                self.point_cloud_paths = [os.path.join(dpsr_car_path, obj_name,'pointcloud.npz') for obj_name in texturify_obj_names]
                self.num_shapes = len(self.point_cloud_paths)

                # load raw points to np
                all_points_normals = []
                for idx in range(self.num_shapes):
                    dense_points = np.load(self.point_cloud_paths[idx])
                    surface_points_dense = (dense_points['points']).astype(np.float32)
                    surface_normals_dense = dense_points['normals'].astype(np.float32)
                    pointcloud = np.concatenate([surface_points_dense, surface_normals_dense], axis=-1)
                    all_points_normals.append(pointcloud)
                all_points_normals = np.stack(all_points_normals, axis=0)
                self.all_points_normals = all_points_normals


                head = 0
                max_batch = 16
                batch_p_2d_all, folding_points_all, folding_normals_all = [], [], []
                while head < self.num_shapes:
                    points_subset = torch.from_numpy(all_points_normals[head : min(head + max_batch, self.num_shapes)]).to(device)
                    preload_batch_size, num_samples_subset = points_subset.shape[0], points_subset.shape[1]

                    with torch.no_grad():
                        batch_p_2d, folding_points, folding_normals = self.forward(points_subset, level=5, fast_forward=True)
                        # sdf_grid_gdt = self.forward_gdt(points_subset)
                        # sdf_grid_gdt = sdf_grid_gdt.view(preload_batch_size, 1, *self.dpsr.res)
                        # sdf_grid_pred = sdf_grid_pred.view(preload_batch_size, 1, *self.dpsr.res)

                    batch_p_2d_all.append(batch_p_2d.cpu().numpy())
                    folding_points_all.append(folding_points.cpu().numpy())
                    folding_normals_all.append(folding_normals.cpu().numpy())
                    del batch_p_2d
                    del folding_points
                    del folding_normals
                    # sdf_grid_gdt_all.append(sdf_grid_gdt.cpu())

                    head += max_batch
                print("Done!")

                self.batch_p_2d_all = np.concatenate(batch_p_2d_all, axis=0) # (1291, 10242, 3)
                self.folding_points_all = np.concatenate(folding_points_all, axis=0) # (1291, 10242, 3)
                self.folding_normals_all = np.concatenate(folding_normals_all, axis=0)
                # sdf_grid_gdt_all = torch.cat(sdf_grid_gdt_all, dim=0) # (1291, 1, 128, 128, 128)
                self.is_preloaded = True
        
        sub_idc = np.random.choice(self.num_shapes, batch_size)
        rt_batch_p_2d = torch.from_numpy(self.batch_p_2d_all[sub_idc]).to(device)
        rt_folding_points = torch.from_numpy(self.folding_points_all[sub_idc]).to(device)
        rt_folding_normals = torch.from_numpy(self.folding_normals_all[sub_idc]).to(device)
        ts_batch_p_2d, ts_folding_points, ts_folding_normals, rt_sdf_grid_pred = self.forward_pred(rt_batch_p_2d, rt_folding_points, rt_folding_normals)
        rt_sdf_grid_gdt = self.forward_gdt(torch.from_numpy(self.all_points_normals[sub_idc]).to(device))

        return ts_batch_p_2d, ts_folding_points, rt_sdf_grid_pred, rt_sdf_grid_gdt

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def forward_train(self, batch):
        surface_points = batch.get('surface_points')
        sampled_points = batch.get('sampled_points')
        # gt_sdf = batch.get('sampled_sdfs')
        # gt_normals = batch.get('sampled_normals')

        sampled_points.requires_grad = True

        batch_size, n_points = sampled_points.size(0), sampled_points.size(1)

        g_latent, l_latent = self.Encoder(surface_points)
        g_latent_stacked = g_latent.view(batch_size, self.feat_dim).contiguous().unsqueeze(1).expand(-1, n_points, -1)

        cano_coords = self.Fold(g_latent_stacked, sampled_points)
        pred_sdf = self.template(cano_coords)

        cano_coords_backward = self.Fold_b(g_latent_stacked, cano_coords)

        gradient_sdf = torch.autograd.grad(pred_sdf, [sampled_points], grad_outputs=torch.ones_like(pred_sdf), create_graph=True)[0] # normal direction in original shape space
        # batch_p_2d = self.template.get_random_points(torch.Size((batch_size, 3, n_points))).permute(1,2,0).contiguous()

        out = {
            # "gt_sdf": gt_sdf,
            "pred_sdf": pred_sdf,
            # "gt_normals": gt_normals,
            "gradient_sdf": gradient_sdf,
            "cano_coords_backward": cano_coords_backward,
        }

        return out

    
    def forward_fold_batch_p_2d(self, batch, batch_p_2d):
        surface_points = batch.get('surface_points')
        batch_size, n_points = batch_p_2d.size(0), batch_p_2d.size(1)
        batch_p_2d = batch_p_2d.to(surface_points.device)
        
        g_latent, l_latent = self.Encoder(surface_points)
        g_latent_stacked = g_latent.view(batch_size, self.feat_dim).contiguous().unsqueeze(1).expand(-1, n_points, -1)
        coords = self.Fold(g_latent_stacked, batch_p_2d)
        normals = self.Fold_N(g_latent_stacked, coords)
        return coords, normals

    def forward_gdt(self, points):
        batch_size, total_n_points = points.size(0), points.size(1)
        surface_points, surface_normals = points[..., 0:3], points[..., 3:6]

        surface_points_dimension = (surface_points.amax(dim=-2, keepdim=True) - surface_points.amin(dim=-2, keepdim=True)).amax(dim=-1, keepdim=True)
        surface_points = surface_points * 0.95 # / surface_points_dimension * 0.7

        # surface_points = surface_points[:,:,torch.LongTensor([2,1,0])]
        # surface_normals = surface_normals [:,:,torch.LongTensor([2,1,0])]

        dpsr_gdt = torch.tanh(self.dpsr(torch.clamp((surface_points + 0.5), 0.0, 0.99), surface_normals))
        return dpsr_gdt.float()


    def forward_pred(self, batch_p_2d, coords, normals):
        sdf_grid = torch.tanh(self.dpsr(torch.clamp((coords+0.5), 0.0, 0.99).detach(), normals))

        batch_p_2d = batch_p_2d[:,:,torch.LongTensor([2,1,0])]
        coords = coords[:,:,torch.LongTensor([2,1,0])]
        normals = normals [:,:,torch.LongTensor([2,1,0])]

        return batch_p_2d, coords, normals, sdf_grid.float()

    def forward(self, points, level=5, batch_p_2d=None, fast_forward=False):
        batch_size, total_n_points = points.size(0), points.size(1)
        surface_points, surface_normals = points[..., 0:3], points[..., 3:6]

        if batch_p_2d is None:
            # batch_p_2d = self.template.get_random_points(num_points=n_points, batch_size=batch_size*multiplication)
            batch_p_2d = self.template.get_regular_points(level=level, batch_size=batch_size)
            if level == 5:
                # hacky way to minimize num_points of batch_p_2d 
                batch_p_2d = batch_p_2d.reshape(batch_size*6, batch_p_2d.size(1)//6, 3)
                n_points = batch_p_2d.size(1)
                multiplication = 6
            elif level == 4:
                n_points = batch_p_2d.size(1)
                multiplication = 1
            
        batch_p_2d = batch_p_2d.to(surface_points.device)
        
        indice = torch.tensor(random.sample(range(total_n_points), n_points)).to(surface_points.device)
        g_latent, l_latent = self.Encoder(surface_points[:,indice,:])
        g_latent_stacked = g_latent.view(batch_size, self.feat_dim).contiguous().unsqueeze(1).expand(-1, n_points, -1)
        if multiplication > 1:
            g_latent_stacked = g_latent.view(batch_size, 1, 1, self.feat_dim).expand(-1, multiplication, n_points, -1)
            g_latent_stacked = g_latent_stacked.reshape(batch_size * multiplication, n_points, -1)
        else:
            g_latent_stacked = g_latent.view(batch_size, self.feat_dim).contiguous().unsqueeze(1).expand(-1, n_points, -1)
        coords = self.Fold_P(g_latent_stacked, batch_p_2d)
        normals = self.Fold_N(g_latent_stacked, coords)

        coords = coords.reshape(batch_size, multiplication*n_points, 3)
        normals = normals.reshape(batch_size, multiplication*n_points, 3)
        batch_p_2d = batch_p_2d.reshape(batch_size, multiplication*n_points, 3)

        coords_dimension = (coords.amax(dim=-2, keepdim=True) - coords.amin(dim=-2, keepdim=True)).amax(dim=-1, keepdim=True)
        coords = coords * 0.95 # / coords_dimension * 0.7

        if fast_forward:
            return batch_p_2d, coords, normals

        sdf_grid = torch.tanh(self.dpsr(torch.clamp((coords+0.5), 0.0, 0.99).detach(), normals))

        batch_p_2d = batch_p_2d[:,:,torch.LongTensor([2,1,0])]
        coords = coords[:,:,torch.LongTensor([2,1,0])]
        normals = normals [:,:,torch.LongTensor([2,1,0])]

        return batch_p_2d, coords, normals, sdf_grid.float()

    def create_mesh(self, surface_points, N=128, max_batch=64 ** 3):
        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)

        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 3, 4)

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
        samples.requires_grad = False

        g_latent, l_latent = self.Encoder(surface_points.unsqueeze(0))

        head = 0
        while head < num_samples:
            sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].to(surface_points.device)[None,...]
            num_samples_subset = sample_subset.shape[1]

            g_latent_stacked = g_latent.view(1, self.Fold.z_dim).contiguous().unsqueeze(1).expand(-1, num_samples_subset, -1)
            cano_coords = self.Fold(g_latent_stacked, sample_subset)
            pred_sdf = self.template(cano_coords)

            samples[head : min(head + max_batch, num_samples), 3] = pred_sdf.squeeze().detach().cpu()
            head += max_batch

        sdf_values = samples[:, 3]
        sdf_values = sdf_values.reshape(N, N, N).data.cpu()

        mesh_points, faces, normals, values =  convert_sdf_samples_to_mesh(
            sdf_values,
            voxel_origin,
            voxel_size,
            offset=None,
            scale=None,
            level=0.
        )

        return {
            "verts": mesh_points,
            "faces": faces,
            "normals": normals,
            "sdf_values": sdf_values,
        }


    def training_step(self, batch, batch_idx, optimizer_idx=0):
        surface_points = batch.get('surface_points')
        surface_normals = batch.get('surface_normals')
        gt_sdf = batch.get('sampled_sdfs')
        gt_normals = batch.get('sampled_normals')
        batch_size = surface_points.shape[0]

        if optimizer_idx == 0:
            # prepare dpsr gdt
            dpsr_gdt = self.dpsr((surface_points + 0.5), surface_normals)
            
            batch_p_2d = self.template.get_random_points(num_points=2048, batch_size=batch_size)
            folding_points, folding_normals = self.forward_fold_batch_p_2d(batch, batch_p_2d)

            surface_loss = CD_loss(folding_points, surface_points)
            self.log("train/surface_loss", surface_loss.clone().detach().mean(), prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

            pred_dpsr = torch.tanh(self.dpsr(torch.clamp((folding_points+0.5), 0.0, 0.99).detach(), folding_normals))
            gdt_dpsr = torch.tanh(self.dpsr((surface_points + 0.5), surface_normals))
            dpsr_loss = F.mse_loss(pred_dpsr, gdt_dpsr)
            self.log("train/dpsr_loss", dpsr_loss.clone().detach().mean(), prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

            # out = self.forward_train(batch)
            # sdf_loss, log_dict_sdf = self.sdf_loss(pred_sdf=out['pred_sdf'], gradient_sdf=out['gradient_sdf'], gt_sdf=gt_sdf, gt_normals=gt_normals)
            # self.log_dict(log_dict_sdf, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
            # inverse_loss = nn.functional.mse_loss(out['cano_coords_backward'], batch.get('sampled_points'))
            # self.log("train/inverse_loss", inverse_loss.clone().detach().mean(), prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

            # surface_points_folded = self.forward_fold_surface_points(batch)
            # batch_p_2d = self.template.get_random_points(num_points=2048, batch_size=batch_size)
            # surface_loss = CD_loss(surface_points_folded, batch_p_2d.to(surface_points_folded.device))
            # self.log("train/surface_loss", surface_loss.clone().detach().mean(), prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

            # if batch_idx % 200 == 0:
            #     self.visualizer.plot_pcd_three_views_color(out['batch_p_2d'][:3], out['self_rec_shape'][:3], ref_pcds=out['surface_points'][:3], 
            #                                                title='train_point_reconstruct', 
            #                                                epoch=self.current_epoch, 
            #                                                logger=self.logger)
            # if batch_idx % 20 == 0 and self.visualizer != None:
            #     self.visualizer.show_pointclouds(points=surface_points[0], title="train/input")
            #     self.visualizer.show_pointclouds(points=corr_out['self_rec_shape'][0], title="train/global_reconstruct")
            #     self.visualizer.show_pointclouds(points=corr_out['batch_p_2d'][0], title="train/batch_p_2d")
            return dpsr_loss + surface_loss

    def test_step(self, batch, batch_idx):
        rt = self(batch)
        return rt

    def test_epoch_end(self, validation_step_outputs):
        all_rec = list()
        all_ref = list()
        others = {}

        for step in range(len(validation_step_outputs)):
            validation_step_output = validation_step_outputs[step]
            for k in validation_step_output.keys():
                if k == "surface_points":
                    all_ref.append(validation_step_output[k])
                elif k == "self_rec_shape":
                    all_rec.append(validation_step_output[k])
                elif "shape" in k:
                    if k in others:
                        others[k].append(validation_step_output[k])
                    else:
                        others[k] = list()
                        others[k].append(validation_step_output[k])

        results = {}
        all_ref = torch.cat(all_ref, 0)
        all_rec = torch.cat(all_rec, 0)

        emd_cd_rt = emd_cd(all_rec, all_ref, 128, accelerated_cd=True)
        emd_cd_rt = {
            'test/'+k: (v.cpu().detach().item() if not isinstance(v, float) else v)
            for k, v in emd_cd_rt.items()}
        
        self.log("test/CD", emd_cd_rt["test/CD"], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/EMD", emd_cd_rt["test/EMD"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)



    def get_cano_coords_and_color(self, surface_points, target_points, max_batch=64 ** 3):
        if torch.is_tensor(target_points):
            target_points = target_points.cpu().numpy()

        num_samples = target_points.shape[0]
        cano_coords = np.zeros_like(target_points)
        g_latent, l_latent = self.Encoder(surface_points.unsqueeze(0))

        head = 0
        while head < num_samples:
            sample_subset = torch.from_numpy(target_points[head : min(head + max_batch, num_samples), 0:3]).float().to(surface_points.device)[None,...]
            num_samples_subset = sample_subset.shape[1]

            g_latent_stacked = g_latent.view(1, self.Fold.z_dim).contiguous().unsqueeze(1).expand(-1, num_samples_subset, -1)
            cano_coords[head : min(head + max_batch, num_samples), 0:3] = self.Fold(g_latent_stacked, sample_subset).squeeze().detach().cpu()
            head += max_batch

        cano_colors = np.copy(cano_coords)
        cano_colors = np.clip(cano_colors/(self.template.radius*2)+0.5,0,1) # normalize color to 0-1
        cano_colors = np.clip((cano_colors * 255), 0, 255).astype(int) # normalize color 0-255
        return cano_coords, cano_colors

    def create_mesh_dpsr(self, surface_points, N=128, max_batch=64 ** 3):
        batch_size = 1
        n_points = 4096
        g_latent, l_latent = self.Encoder(surface_points.unsqueeze(0))

        batch_p_2d = self.template.get_random_points(num_points=n_points, batch_size=batch_size)
        batch_p_2d = batch_p_2d.to(surface_points.device)
        
        g_latent_stacked = g_latent.view(batch_size, self.feat_dim).contiguous().unsqueeze(1).expand(-1, n_points, -1)
        folding_points = self.Fold(g_latent_stacked, batch_p_2d)
        folding_normals = self.Fold_N(g_latent_stacked, folding_points)
        psr_grid = self.dpsr(torch.clamp((folding_points+0.5), 0.0, 0.99), folding_normals)
        v, f, _ = mc_from_psr(psr_grid, zero_level=0)

        if v.size > 0:
            v -= 0.5

        folding_points_np = folding_points.squeeze().cpu().numpy()
        batch_p_2d_np = batch_p_2d.squeeze().cpu().numpy()
        surface_pts_color = sphere_to_color(batch_p_2d_np, self.template.radius)
        _, idx = scipy.spatial.KDTree(folding_points_np).query(v) 
        c = surface_pts_color[idx]

        return {
            'batch_p_2d': batch_p_2d_np,
            'folding_points': folding_points.squeeze().detach().cpu().numpy(),
            'folding_normals': folding_normals.squeeze().detach().cpu().numpy(),
            'verts': v,
            'faces': f,
            'colors': c,
        }


import scipy

def sphere_to_color(sphere_coords, radius):
    sphere_coords = np.copy(sphere_coords)
    sphere_coords = np.clip(sphere_coords/(radius*2)+0.5,0,1) # normalize color to 0-1
    sphere_coords = np.clip((sphere_coords * 255), 0, 255).astype(int) # normalize color 0-255
    return sphere_coords




    
class DPSR(nn.Module):
    def __init__(self, res, sig=10, scale=True, shift=True):
        """
        :param res: tuple of output field resolution. eg., (128,128)
        :param sig: degree of gaussian smoothing
        """
        super(DPSR, self).__init__()
        self.res = res
        self.sig = sig
        self.dim = len(res)
        self.denom = np.prod(res)
        self.G = spec_gaussian_filter(res=res, sig=sig).float()
        self.G.requires_grad = False # True, if we also make sig a learnable parameter
        self.omega = fftfreqs(res, dtype=torch.float32)
        self.scale = scale
        self.shift = shift
        # self.register_buffer("G", G)
        
    def forward(self, V, N):
        """
        :param V: (batch, nv, 2 or 3) tensor for point cloud coordinates
        :param N: (batch, nv, 2 or 3) tensor for point normals
        :return phi: (batch, res, res, ...) tensor of output indicator function field
        """
        assert(V.shape == N.shape) # [b, nv, ndims]
        ras_p = point_rasterize(V, N, self.res)  # [b, n_dim, dim0, dim1, dim2]
        
        ras_s = torch.fft.rfftn(ras_p, dim=(2,3,4))
        ras_s = ras_s.permute(*tuple([0]+list(range(2, self.dim+1))+[self.dim+1, 1]))
        N_ = ras_s[..., None] * self.G.to(ras_s.device) # [b, dim0, dim1, dim2/2+1, n_dim, 1]

        omega = fftfreqs(self.res, dtype=torch.float32).unsqueeze(-1) # [dim0, dim1, dim2/2+1, n_dim, 1]
        omega *= 2 * np.pi  # normalize frequencies
        omega = omega.to(V.device)
        
        DivN = torch.sum(-img(torch.view_as_real(N_[..., 0])) * omega, dim=-2)
        
        Lap = -torch.sum(omega**2, -2) # [dim0, dim1, dim2/2+1, 1]
        Phi = DivN / (Lap+1e-6) # [b, dim0, dim1, dim2/2+1, 2]  
        Phi = Phi.permute(*tuple([list(range(1,self.dim+2)) + [0]]))  # [dim0, dim1, dim2/2+1, 2, b] 
        Phi[tuple([0] * self.dim)] = 0
        Phi = Phi.permute(*tuple([[self.dim+1] + list(range(self.dim+1))]))  # [b, dim0, dim1, dim2/2+1, 2]
        
        phi = torch.fft.irfftn(torch.view_as_complex(Phi), s=self.res, dim=(1,2,3))
        
        if self.shift or self.scale:
            # ensure values at points are zero
            fv = grid_interp(phi.unsqueeze(-1), V, batched=True).squeeze(-1) # [b, nv]
            if self.shift: # offset points to have mean of 0
                offset = torch.mean(fv, dim=-1)  # [b,] 
                phi -= offset.view(*tuple([-1] + [1] * self.dim))
                
            phi = phi.permute(*tuple([list(range(1,self.dim+1)) + [0]]))
            fv0 = phi[tuple([0] * self.dim)]  # [b,]
            phi = phi.permute(*tuple([[self.dim] + list(range(self.dim))]))
            
            if self.scale:
                phi = -phi / torch.abs(fv0.view(*tuple([-1]+[1] * self.dim))) *0.5
        return phi

from skimage import measure

##################################################
# Below are functions for DPSR

def fftfreqs(res, dtype=torch.float32, exact=True):
    """
    Helper function to return frequency tensors
    :param res: n_dims int tuple of number of frequency modes
    :return:
    """

    n_dims = len(res)
    freqs = []
    for dim in range(n_dims - 1):
        r_ = res[dim]
        freq = np.fft.fftfreq(r_, d=1/r_)
        freqs.append(torch.tensor(freq, dtype=dtype))
    r_ = res[-1]
    if exact:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1/r_), dtype=dtype))
    else:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1/r_)[:-1], dtype=dtype))
    omega = torch.meshgrid(freqs)
    omega = list(omega)
    omega = torch.stack(omega, dim=-1)

    return omega

def img(x, deg=1): # imaginary of tensor (assume last dim: real/imag)
    """
    multiply tensor x by i ** deg
    """
    deg %= 4
    if deg == 0:
        res = x
    elif deg == 1:
        res = x[..., [1, 0]]
        res[..., 0] = -res[..., 0]
    elif deg == 2:
        res = -x
    elif deg == 3:
        res = x[..., [1, 0]]
        res[..., 1] = -res[..., 1]
    return res

def spec_gaussian_filter(res, sig):
    omega = fftfreqs(res, dtype=torch.float64) # [dim0, dim1, dim2, d]
    dis = torch.sqrt(torch.sum(omega ** 2, dim=-1))
    filter_ = torch.exp(-0.5*((sig*2*dis/res[0])**2)).unsqueeze(-1).unsqueeze(-1)
    filter_.requires_grad = False

    return filter_

def grid_interp(grid, pts, batched=True):
    """
    :param grid: tensor of shape (batch, *size, in_features)
    :param pts: tensor of shape (batch, num_points, dim) within range (0, 1)
    :return values at query points
    """
    if not batched:
        grid = grid.unsqueeze(0)
        pts = pts.unsqueeze(0)
    dim = pts.shape[-1]
    bs = grid.shape[0]
    size = torch.tensor(grid.shape[1:-1]).to(grid.device).type(pts.dtype)
    cubesize = 1.0 / size
    
    ind0 = torch.floor(pts / cubesize).long()  # (batch, num_points, dim)
    ind1 = torch.fmod(torch.ceil(pts / cubesize), size).long() # periodic wrap-around
    ind01 = torch.stack((ind0, ind1), dim=0) # (2, batch, num_points, dim)
    tmp = torch.tensor([0,1],dtype=torch.long)
    com_ = torch.stack(torch.meshgrid(tuple([tmp] * dim)), dim=-1).view(-1, dim)
    dim_ = torch.arange(dim).repeat(com_.shape[0], 1) # (2**dim, dim)
    ind_ = ind01[com_, ..., dim_]   # (2**dim, dim, batch, num_points)
    ind_n = ind_.permute(2, 3, 0, 1) # (batch, num_points, 2**dim, dim)
    ind_b = torch.arange(bs).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1) # (batch, num_points, 2**dim)
    # latent code on neighbor nodes
    if dim == 2:
        lat = grid.clone()[ind_b, ind_n[..., 0], ind_n[..., 1]] # (batch, num_points, 2**dim, in_features)
    else:
        lat = grid.clone()[ind_b, ind_n[..., 0], ind_n[..., 1], ind_n[..., 2]] # (batch, num_points, 2**dim, in_features)

    # weights of neighboring nodes
    xyz0 = ind0.type(cubesize.dtype) * cubesize        # (batch, num_points, dim)
    xyz1 = (ind0.type(cubesize.dtype) + 1) * cubesize  # (batch, num_points, dim)
    xyz01 = torch.stack((xyz0, xyz1), dim=0) # (2, batch, num_points, dim)
    pos = xyz01[com_, ..., dim_].permute(2,3,0,1)   # (batch, num_points, 2**dim, dim)
    pos_ = xyz01[1-com_, ..., dim_].permute(2,3,0,1)   # (batch, num_points, 2**dim, dim)
    pos_ = pos_.type(pts.dtype)
    dxyz_ = torch.abs(pts.unsqueeze(-2) - pos_) / cubesize # (batch, num_points, 2**dim, dim)
    weights = torch.prod(dxyz_, dim=-1, keepdim=False) # (batch, num_points, 2**dim)
    query_values = torch.sum(lat * weights.unsqueeze(-1), dim=-2)  # (batch, num_points, in_features)
    if not batched:
        query_values = query_values.squeeze(0)
        
    return query_values

def scatter_to_grid(inds, vals, size):
    """
    Scatter update values into empty tensor of size size.
    :param inds: (#values, dims)
    :param vals: (#values)
    :param size: tuple for size. len(size)=dims
    """
    dims = inds.shape[1]
    assert(inds.shape[0] == vals.shape[0])
    assert(len(size) == dims)
    dev = vals.device
    # result = torch.zeros(*size).view(-1).to(dev).type(vals.dtype)  # flatten
    # # flatten inds
    result = torch.zeros(*size, device=dev).view(-1).type(vals.dtype)  # flatten
    # flatten inds
    fac = [np.prod(size[i+1:]) for i in range(len(size)-1)] + [1]
    fac = torch.tensor(fac, device=dev).type(inds.dtype)
    inds_fold = torch.sum(inds*fac, dim=-1)  # [#values,]
    result.scatter_add_(0, inds_fold, vals)
    result = result.view(*size)
    return result

def point_rasterize(pts, vals, size):
    """
    :param pts: point coords, tensor of shape (batch, num_points, dim) within range (0, 1)
    :param vals: point values, tensor of shape (batch, num_points, features)
    :param size: len(size)=dim tuple for grid size
    :return rasterized values (batch, features, res0, res1, res2)
    """
    dim = pts.shape[-1]
    assert(pts.shape[:2] == vals.shape[:2])
    assert(pts.shape[2] == dim)
    size_list = list(size)
    size = torch.tensor(size).to(pts.device).float()
    cubesize = 1.0 / size
    bs = pts.shape[0]
    nf = vals.shape[-1]
    npts = pts.shape[1]
    dev = pts.device
    
    ind0 = torch.floor(pts / cubesize).long()  # (batch, num_points, dim)
    ind1 = torch.fmod(torch.ceil(pts / cubesize), size).long() # periodic wrap-around
    ind01 = torch.stack((ind0, ind1), dim=0) # (2, batch, num_points, dim)
    tmp = torch.tensor([0,1],dtype=torch.long)
    com_ = torch.stack(torch.meshgrid(tuple([tmp] * dim)), dim=-1).view(-1, dim)
    dim_ = torch.arange(dim).repeat(com_.shape[0], 1) # (2**dim, dim)
    ind_ = ind01[com_, ..., dim_]   # (2**dim, dim, batch, num_points)
    ind_n = ind_.permute(2, 3, 0, 1) # (batch, num_points, 2**dim, dim)
    # ind_b = torch.arange(bs).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1) # (batch, num_points, 2**dim)
    ind_b = torch.arange(bs, device=dev).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1) # (batch, num_points, 2**dim)
    
    # weights of neighboring nodes
    xyz0 = ind0.type(cubesize.dtype) * cubesize        # (batch, num_points, dim)
    xyz1 = (ind0.type(cubesize.dtype) + 1) * cubesize  # (batch, num_points, dim)
    xyz01 = torch.stack((xyz0, xyz1), dim=0) # (2, batch, num_points, dim)
    pos = xyz01[com_, ..., dim_].permute(2,3,0,1)   # (batch, num_points, 2**dim, dim)
    pos_ = xyz01[1-com_, ..., dim_].permute(2,3,0,1)   # (batch, num_points, 2**dim, dim)
    pos_ = pos_.type(pts.dtype)
    dxyz_ = torch.abs(pts.unsqueeze(-2) - pos_) / cubesize # (batch, num_points, 2**dim, dim)
    weights = torch.prod(dxyz_, dim=-1, keepdim=False) # (batch, num_points, 2**dim)
    
    ind_b = ind_b.unsqueeze(-1).unsqueeze(-1)      # (batch, num_points, 2**dim, 1, 1)
    ind_n = ind_n.unsqueeze(-2)                    # (batch, num_points, 2**dim, 1, dim)
    ind_f = torch.arange(nf, device=dev).view(1, 1, 1, nf, 1)  # (1, 1, 1, nf, 1)
    # ind_f = torch.arange(nf).view(1, 1, 1, nf, 1)  # (1, 1, 1, nf, 1)
    
    ind_b = ind_b.expand(bs, npts, 2**dim, nf, 1)
    ind_n = ind_n.expand(bs, npts, 2**dim, nf, dim).to(dev)
    ind_f = ind_f.expand(bs, npts, 2**dim, nf, 1)
    inds = torch.cat([ind_b, ind_f, ind_n], dim=-1)  # (batch, num_points, 2**dim, nf, 1+1+dim)
     
    # weighted values
    vals = weights.unsqueeze(-1) * vals.unsqueeze(-2)   # (batch, num_points, 2**dim, nf)
    
    inds = inds.view(-1, dim+2).permute(1, 0).long()  # (1+dim+1, bs*npts*2**dim*nf)
    vals = vals.reshape(-1) # (bs*npts*2**dim*nf)
    tensor_size = [bs, nf] + size_list
    raster = scatter_to_grid(inds.permute(1, 0), vals, [bs, nf] + size_list)
    
    return raster  # [batch, nf, res, res, res]

##################################################
# Below are the utilization functions in general

def mc_from_psr(psr_grid, pytorchify=False, real_scale=False, zero_level=0):
    '''
    Run marching cubes from PSR grid
    '''
    batch_size = psr_grid.shape[0]
    s = psr_grid.shape[-1] # size of psr_grid
    psr_grid_numpy = psr_grid.squeeze().detach().cpu().numpy()
    
    if batch_size>1:
        verts, faces, normals = [], [], []
        for i in range(batch_size):
            verts_cur, faces_cur, normals_cur, values = measure.marching_cubes(psr_grid_numpy[i], level=0)
            verts.append(verts_cur)
            faces.append(faces_cur)
            normals.append(normals_cur)
        verts = np.stack(verts, axis = 0)
        faces = np.stack(faces, axis = 0)
        normals = np.stack(normals, axis = 0)
    else:
        try:
            verts, faces, normals, values = measure.marching_cubes(psr_grid_numpy, level=zero_level)
        except:
            verts, faces, normals, values = measure.marching_cubes(psr_grid_numpy)
    if real_scale:
        verts = verts / (s-1) # scale to range [0, 1]
    else:
        verts = verts / s # scale to range [0, 1)

    if pytorchify:
        device = psr_grid.device
        verts = torch.Tensor(np.ascontiguousarray(verts)).to(device)
        faces = torch.Tensor(np.ascontiguousarray(faces)).to(device)
        normals = torch.Tensor(np.ascontiguousarray(-normals)).to(device)

    return verts, faces, normals