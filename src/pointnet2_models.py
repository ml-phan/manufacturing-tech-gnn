import torch
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
from src.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
# from src.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from src.pointnet2_utils_v2 import *
class get_model(nn.Module):
    def __init__(self, k=2, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


# class PointNet2(nn.Module):
#     def __init__(self,num_class,normal_channel=True):
#         super(PointNet2, self).__init__()
#         in_channel = 6 if normal_channel else 3
#         self.normal_channel = normal_channel
#         self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
#         self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
#         self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
#         self.fc1 = nn.Linear(1024, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.drop1 = nn.Dropout(0.4)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.drop2 = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(256, num_class)
#
#     def forward(self, xyz):
#         B, C, N = xyz.shape
#         xyz = xyz.transpose(1, 2)
#         if self.normal_channel:
#             norm = xyz[:, :, 3:6]
#             xyz = xyz[:, :, :3]
#         else:
#             norm = None
#             xyz  = xyz[:, :, :3]
#         l1_xyz, l1_points = self.sa1(xyz, norm)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         x = l3_points.view(B, 1024)
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         x = self.fc3(x)
#         x = F.log_softmax(x, -1)
#
#
#         return x,l3_points
#
#
# class get_pointnet2_loss(nn.Module):
#     def __init__(self):
#         super(get_pointnet2_loss, self).__init__()
#
#     def forward(self, pred, target, trans_feat):
#         total_loss = F.nll_loss(pred, target)
#
#         return total_loss


class PointCloudDataset(Dataset):
    def __init__(self, dataset):
        """
        Args:
            txt_file_paths (list[str]): list of paths to .txt point cloud files
        """
        self.file_paths = dataset["file"].values
        self.labels = dataset["label"].values

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        pts = np.loadtxt(path, delimiter=",").astype(np.float32)  # shape (2048, 6)
        pts = torch.from_numpy(pts.T)  # shape (6, 2048)

        label = torch.tensor(label, dtype=torch.long)

        return pts, label


class PointCloudDatasetInRAM(torch.utils.data.Dataset):
    def __init__(self, df, preload=True):
        self.labels = df["label"].values
        self.file_paths = df["file"].values

        if preload:
            self.data = []
            for path in tqdm(self.file_paths):
                pts = np.loadtxt(path, delimiter=",").astype(np.float32)
                self.data.append(pts)
            self.labels = df["label"].values
        else:
            self.data = None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if self.data is not None:
            pts = self.data[idx]
        else:
            pts = np.loadtxt(self.file_paths[idx], dtype=np.float32)
        pts = torch.from_numpy(pts.T)
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)
        return pts, label


