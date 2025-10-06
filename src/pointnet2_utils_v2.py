import torch
import torch.nn as nn
import torch.nn.functional as F


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling
    xyz: [B, N, 3] or [B, C, N]
    npoint: number of samples
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query
    xyz: [B, N, 3]
    new_xyz: [B, S, 3]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1,
                                                                  N).repeat(
        [B, S, 1])
    sqrdists = torch.sum(
        (new_xyz.view(B, S, 1, C) - xyz.view(B, 1, N, C)) ** 2, -1)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    xyz: [B, N, 3]
    points: [B, N, C]
    """
    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).repeat(1, 1, C))

    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = torch.gather(xyz.unsqueeze(2).repeat(1, 1, nsample, 1), 1,
                               idx.unsqueeze(-1).repeat(1, 1, 1, C))
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

    if points is not None:
        grouped_points = torch.gather(
            points.unsqueeze(2).repeat(1, 1, nsample, 1), 1,
            idx.unsqueeze(-1).repeat(1, 1, 1, points.shape[-1]))
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp,
                 group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        xyz: [B, 3, N]
        points: [B, C, N]
        """
        xyz = xyz.permute(0, 2, 1)  # [B, N, 3]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, C]

        if self.group_all:
            new_xyz = torch.zeros(xyz.shape[0], 1, 3).to(xyz.device)
            grouped_xyz = xyz.unsqueeze(1)
            if points is not None:
                new_points = torch.cat([grouped_xyz, points.unsqueeze(1)],
                                       dim=-1)
            else:
                new_points = grouped_xyz
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius,
                                                   self.nsample, xyz, points)

        # [B, S, nsample, C] -> [B, C, nsample, S]
        new_points = new_points.permute(0, 3, 2, 1)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]  # [B, C', S]
        new_xyz = new_xyz.permute(0, 2, 1)  # [B, 3, S]

        return new_xyz, new_points


class PointNet2Classification(nn.Module):
    def __init__(self, num_classes, input_channels=6):
        super().__init__()

        # Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32,
                                          in_channel=input_channels + 3,
                                          mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64,
                                          in_channel=128 + 3,
                                          mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None,
                                          nsample=None,
                                          in_channel=256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, data):
        """
        data: [B, C, N] where C=6 (features) and N=2048 (points)
        """
        # Split into xyz (first 3 channels) and features (remaining channels)
        xyz = data[:, :3, :]  # [B, 3, N]
        points = data[:, 3:, :] if data.shape[1] > 3 else None  # [B, 3, N]

        # Hierarchical feature learning
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Global feature
        x = l3_points.view(l3_points.shape[0], -1)  # [B, 1024]

        # Classification head
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x
