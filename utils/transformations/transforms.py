# Copyright 2025 CEA LIST - Samir Abou Haidar
# Modifications based on code from:
# - Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai, 2022
# - Open-MMLab, 2018-2019

# Copyright 2022 - Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai
# Copyright 2018-2019 Open-MMLab

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import numpy as np
from glob import glob
from typing import List


class Compose:
    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, pcloud, labels):
        for t in self.transformations:
            pcloud, labels = t(pcloud, labels)
        return pcloud, labels


class RandomApply:
    def __init__(self, transformation, prob=0.5):
        self.prob = prob
        self.transformation = transformation

    def __call__(self, pcloud, labels):
        if torch.rand(1) < self.prob:
            pcloud, labels = self.transformation(pcloud, labels)
        return pcloud, labels


class Transformation:
    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, pcloud, labels):
        if labels is None:
            return pcloud if self.inplace else np.array(pcloud, copy=True)

        out = (
            (pcloud, labels)
            if self.inplace
            else (np.array(pcloud, copy=True), np.array(labels, copy=True))
        )
        return out


class Identity(Transformation):
    def __init__(self, inplace=False):
        super().__init__(inplace)

    def __call__(self, pcloud, labels):
        return super().__call__(pcloud, labels)


class Rotation(Transformation):
    def __init__(self, dim=2, range=np.pi, inplace=False):
        super().__init__(inplace)
        self.range = range
        self.inplace = inplace
        if dim == 2:
            self.dims = (0, 1)
        elif dim == 1:
            self.dims = (0, 2)
        elif dim == 0:
            self.dims = (1, 2)

    def __call__(self, pcloud, labels):
        # Build rotation matrix
        theta = (2 * torch.rand(1)[0] - 1) * self.range
        # Build rotation matrix
        rot = np.array(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        # Apply rotation
        pcloud, labels = super().__call__(pcloud, labels)
        pcloud[:, self.dims] = pcloud[:, self.dims] @ rot
        return pcloud, labels


class Scale(Transformation):
    def __init__(self, dims=(0, 1), range=0.05, inplace=False):
        super().__init__(inplace)
        self.dims = dims
        self.range = range

    def __call__(self, pcloud, labels):
        pcloud, labels = super().__call__(pcloud, labels)
        scale = 1 + (2 * torch.rand(1).item() - 1) * self.range
        pcloud[:, self.dims] *= scale
        return pcloud, labels


class FlipXY(Transformation):
    def __init__(self, inplace=False):
        super().__init__(inplace=inplace)

    def __call__(self, pcloud, labels):
        pcloud, labels = super().__call__(pcloud, labels)
        id = torch.randint(2, (1,))[0]
        pcloud[:, id] *= -1.0
        return pcloud, labels


class LimitNumPoints(Transformation):
    def __init__(self, dims=(0, 1, 2), max_point=30000, random=False):
        super().__init__(inplace=True)
        self.dims = dims
        self.max_points = max_point
        self.random = random
        assert max_point > 0

    def __call__(self, pcloud, labels):
        pc, labels = super().__call__(pcloud, labels)
        if pc.shape[0] > self.max_points:
            if self.random:
                center = torch.randint(pc.shape[0], (1,))[0]
                center = pc[center : center + 1, self.dims]
            else:
                center = np.zeros((1, len(self.dims)))
            idx = np.argsort(np.square(pc[:, self.dims] - center).sum(axis=1))[
                : self.max_points
            ]
            pc, labels = pc[idx], labels[idx]
        return pc, labels


class Crop(Transformation):
    def __init__(self, dims=(0, 1, 2), fov=((-5, -5, -5), (5, 5, 5)), eps=1e-4):
        super().__init__(inplace=True)
        self.dims = dims
        self.fov = fov
        self.eps = eps
        assert len(fov[0]) == len(fov[1]), "Min and Max FOV must have the same length."
        for i, (min, max) in enumerate(zip(*fov)):
            assert (
                min < max
            ), f"Field of view: min ({min}) < max ({max}) is expected on dimension {i}."

    def __call__(self, pcloud, labels):
        pc, labels = super().__call__(pcloud, labels)

        where = None
        for i, d in enumerate(self.dims):  # Actually a bug below, use d in pc not i!
            temp = (pc[:, d] > self.fov[0][i] + self.eps) & (
                pc[:, d] < self.fov[1][i] - self.eps
            )
            where = temp if where is None else where & temp

        return pc[where], labels[where]


class Voxelize(Transformation):
    def __init__(self, dims=(0, 1, 2), voxel_size=0.1, random=False):
        super().__init__(inplace=True)
        self.dims = dims
        self.voxel_size = voxel_size
        self.random = random
        assert voxel_size >= 0

    def __call__(self, pcloud, labels):
        pc, labels = super().__call__(pcloud, labels)
        if self.voxel_size <= 0:
            return pc, labels

        if self.random:
            permute = torch.randperm(pc.shape[0])
            pc, labels = pc[permute], labels[permute]

        pc_shift = pc[:, self.dims] - pc[:, self.dims].min(0, keepdims=True)

        _, ind = np.unique(
            (pc_shift / self.voxel_size).astype("int"), return_index=True, axis=0
        )

        return pc[ind, :], None if labels is None else labels[ind]
    

class PointSample(Transformation):

    def __init__(self, num_points, sample_range=None, replace=False, inplace=True):
        super().__init__(inplace)
        self.num_points = num_points
        self.sample_range = sample_range
        self.replace = replace

    def _points_random_sampling(self, points, num_samples, sample_range=None, replace=False):
        # Ensure num_samples is an integer if provided as a float
        if isinstance(num_samples, float):
            assert num_samples < 1
            num_samples = int(np.random.uniform(self.num_points, 1.0) * points.shape[0])

        if not replace:
            replace = (points.shape[0] < num_samples)

        # Sampling logic
        point_range = np.arange(len(points))
        if sample_range is not None and not replace:
            dist = np.linalg.norm(points[:, :3], axis=1)
            far_inds = np.where(dist >= sample_range)[0]
            near_inds = np.where(dist < sample_range)[0]

            if len(far_inds) > num_samples:
                far_inds = np.random.choice(far_inds, num_samples, replace=False)
            point_range = near_inds
            num_samples -= len(far_inds)

        choices = np.random.choice(point_range, num_samples, replace=replace)
        if sample_range is not None and not replace:
            choices = np.concatenate((far_inds, choices))
            np.random.shuffle(choices)

        return points[choices], choices

    def __call__(self, pcloud, labels):
        pcloud, labels = super().__call__(pcloud, labels)
        pcloud, choices = self._points_random_sampling(pcloud, self.num_points, self.sample_range, self.replace)
        if labels is not None:
            labels = labels[choices]
        return pcloud, labels

    

class RandomFlip3D(Transformation):
    def __init__(self, flip_ratio_bev_horizontal=0.0, flip_ratio_bev_vertical=0.0, sync_2d=True, inplace=False):
        super().__init__(inplace=inplace)
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        self.sync_2d = sync_2d
        assert isinstance(self.flip_ratio_bev_horizontal, (int, float)) and 0 <= self.flip_ratio_bev_horizontal <= 1
        assert isinstance(self.flip_ratio_bev_vertical, (int, float)) and 0 <= self.flip_ratio_bev_vertical <= 1

    def random_flip_data_3d(self, pcloud, direction):
        """Flip the 3D point cloud in the specified direction."""
        assert direction in ['horizontal', 'vertical']
        
        # Flip point cloud
        pcloud_flipped = np.copy(pcloud)
        if direction == 'horizontal':
            pcloud_flipped[:, 0] = -pcloud_flipped[:, 0]  # Flip x-coordinate
        elif direction == 'vertical':
            pcloud_flipped[:, 1] = -pcloud_flipped[:, 1]  # Flip y-coordinate

        return pcloud_flipped

    def __call__(self, pcloud, labels):
        pcloud, labels = super().__call__(pcloud, labels)
        # Randomly decide whether to flip horizontally or vertically
        flip_horizontal = np.random.rand() < self.flip_ratio_bev_horizontal
        flip_vertical = np.random.rand() < self.flip_ratio_bev_vertical

        # Apply flips
        if flip_horizontal:
            pcloud = self.random_flip_data_3d(pcloud, 'horizontal')
        if flip_vertical:
            pcloud = self.random_flip_data_3d(pcloud, 'vertical')

        return pcloud, labels
  

class GlobalRotScaleTrans(Transformation):
    def __init__(self,
                 rot_range: list = [-0.78539816, 0.78539816],
                 scale_ratio_range: list = [0.95, 1.05],
                 translation_std: list = [0, 0, 0],
                 inplace=False):
        super().__init__(inplace)
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

    def _trans_points(self, pcloud):
        """Translate the point cloud."""
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T
        pcloud[:, :3] += trans_factor  # Apply translation to the first 3 coordinates (x, y, z)
        return pcloud, trans_factor

    def _rot_points(self, pcloud):
        """Rotate the point cloud."""
        rotation = np.random.uniform(self.rot_range[0], self.rot_range[1])
        rotation_matrix = np.array([
            [np.cos(rotation), -np.sin(rotation), 0],
            [np.sin(rotation), np.cos(rotation), 0],
            [0, 0, 1]
        ])  # 3D rotation matrix
        pcloud[:, :3] = pcloud[:, :3] @ rotation_matrix.T  # Apply rotation
        return pcloud, rotation_matrix

    def _scale_points(self, pcloud):
        """Scale the point cloud."""
        scale_factor = np.random.uniform(self.scale_ratio_range[0], self.scale_ratio_range[1])
        pcloud[:, :3] *= scale_factor

        return pcloud


    def __call__(self, pcloud, labels):
        pcloud, labels = super().__call__(pcloud, labels)

        # Apply rotation, scaling and translation

        pcloud, rotation_matrix = self._rot_points(pcloud)

        pcloud = self._scale_points(pcloud)
        
        pcloud, translation_factor = self._trans_points(pcloud)

        return pcloud, labels
    

class ClusterMix(Transformation):
    def __init__(self,
                 H: int,
                 W: int,
                 fov_up: float,
                 fov_down: float,
                 num_areas: List[int],
                 prob: float = 1.0,
                 inplace: bool = True):
        super().__init__(inplace)
        self.H = H
        self.W = W
        self.fov_up = fov_up / 180 * np.pi
        self.fov_down = fov_down / 180 * np.pi
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        self.num_areas = num_areas
        self.prob = prob

    def cluster_vertical_mix_transform(self, points, labels, mix_points, mix_labels):
        depth = np.linalg.norm(points[:, :3], axis=1)
        pitch = np.arcsin(points[:, 2] / depth)
        coors = 1.0 - (pitch + abs(self.fov_down)) / self.fov
        coors *= self.H
        coors = np.floor(coors)
        coors = np.minimum(self.H - 1, coors)
        coors = np.maximum(0, coors).astype(np.int64)

        mix_depth = np.linalg.norm(mix_points[:, :3], axis=1)
        mix_pitch = np.arcsin(mix_points[:, 2] / mix_depth)
        mix_coors = 1.0 - (mix_pitch + abs(self.fov_down)) / self.fov
        mix_coors *= self.H
        mix_coors = np.floor(mix_coors)
        mix_coors = np.minimum(self.H - 1, mix_coors)
        mix_coors = np.maximum(0, mix_coors).astype(np.int64)

        num_areas = np.random.choice(self.num_areas, size=1)[0]
        row_list = np.linspace(0, self.H, num_areas + 1, dtype=int)
        out_points = []
        out_pts_semantic_mask = []

        for i in range(num_areas):
            start_row = row_list[i]
            end_row = row_list[i + 1]
            
            if i % 2 == 0:
                idx = (coors >= start_row) & (coors < end_row)
                out_points.append(points[idx])
                out_pts_semantic_mask.append(labels[idx])
            else:
                idx = (mix_coors >= start_row) & (mix_coors < end_row)
                out_points.append(mix_points[idx])
                out_pts_semantic_mask.append(mix_labels[idx])

        out_points = np.concatenate(out_points, axis=0)
        out_pts_semantic_mask = np.concatenate(out_pts_semantic_mask, axis=0)

        return out_points, out_pts_semantic_mask

    def cluster_horizontal_mix_transform(self, points, labels, mix_points, mix_labels):
        yaw = -np.arctan2(points[:, 1], points[:, 0])
        coors = 0.5 * (yaw / np.pi + 1.0)
        coors *= self.W
        coors = np.floor(coors)
        coors = np.minimum(self.W - 1, coors)
        coors = np.maximum(0, coors).astype(np.int64)

        mix_yaw = -np.arctan2(mix_points[:, 1], mix_points[:, 0])
        mix_coors = 0.5 * (mix_yaw / np.pi + 1.0)
        mix_coors *= self.W
        mix_coors = np.floor(mix_coors)
        mix_coors = np.minimum(self.W - 1, mix_coors)
        mix_coors = np.maximum(0, mix_coors).astype(np.int64)

        start_col = np.random.randint(0, self.W // 2)
        end_col = start_col + self.W // 2

        idx = (coors < start_col) | (coors >= end_col)
        mix_idx = (mix_coors >= start_col) & (mix_coors < end_col)

        out_points = np.concatenate([points[idx], mix_points[mix_idx]], axis=0)
        out_pts_semantic_mask = np.concatenate(
            (labels[idx], mix_labels[mix_idx]), axis=0
        )

        return out_points, out_pts_semantic_mask
    

    def __call__(self, pcloud, labels, mix_points, mix_labels):
        pcloud, labels = super().__call__(pcloud, labels)

        if np.random.rand() > self.prob:
            return pcloud, labels

        if np.random.rand() > 0.5:
            # Vertical mixing
            pcloud, labels = self.cluster_vertical_mix_transform(
                pcloud, labels, mix_points, mix_labels)
        else:
            # Horizontal mixing
            pcloud, labels = self.cluster_horizontal_mix_transform(
                pcloud, labels, mix_points, mix_labels)

        return pcloud, labels
    

class InstanceCopy(Transformation):

    def __init__(self,
                 instance_classes: List[int],
                 prob: float = 1.0,
                 inplace: bool = True):
        super().__init__(inplace)

        self.instance_classes = instance_classes
        self.prob = prob

    def copy_instance(self, points, labels, mix_points, mix_labels):

        concat_points = [points]
        concat_pts_semantic_mask = [labels]
        for instance_class in self.instance_classes:
            mix_idx = mix_labels == instance_class
            concat_points.append(mix_points[mix_idx])
            concat_pts_semantic_mask.append(mix_labels[mix_idx])

        points = np.concatenate(concat_points, axis=0)
        pts_semantic_mask = np.concatenate(concat_pts_semantic_mask, axis=0)

        return points, pts_semantic_mask

    def __call__(self, pcloud, labels, mix_points, mix_labels):
        pcloud, labels = super().__call__(pcloud, labels)
        if np.random.rand() > self.prob:
            return pcloud, labels

        pcloud, labels = self.copy_instance(pcloud, labels, mix_points, mix_labels)
        return pcloud, labels
    

class RangeInterpolation(Transformation):
    def __init__(self, H=64, W=2048, fov_up=3.0, fov_down=-25.0, ignore_index=255, inplace=True):
        super().__init__(inplace)
        self.H = H
        self.W = W
        self.fov_up = fov_up / 180.0 * np.pi
        self.fov_down = fov_down / 180.0 * np.pi
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        self.ignore_index = ignore_index

    def __call__(self, pcloud, labels=None):
        points, labels = super().__call__(pcloud, labels)

        proj_image = np.full((self.H, self.W, 4), -1, dtype=np.float32)
        proj_idx = np.full((self.H, self.W), -1, dtype=np.int64)

        # get depth of all points
        depth = np.linalg.norm(points[:, :3], 2, axis=1)

        # get angles of all points
        yaw = -np.arctan2(points[:, 1], points[:, 0])
        pitch = np.arcsin(points[:, 2] / depth)

        # get projection in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov

        # scale to image size using angular resolution
        proj_x *= self.W
        proj_y *= self.H

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int64)

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int64)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        proj_idx[proj_y[order], proj_x[order]] = indices[order]
        proj_image[proj_y[order], proj_x[order]] = points[order]
        proj_mask = (proj_idx > 0).astype(np.int32)

        
        if labels is not None:
            proj_sem_label = np.full((self.H, self.W),
                                     self.ignore_index,
                                     dtype=np.int64)
            proj_sem_label[proj_y[order],
                           proj_x[order]] = labels[order]

        # Interpolate missing points
        interpolated_points = []
        interpolated_labels = []

        # scan all the pixels
        for y in range(self.H):
            for x in range(self.W):
                # check whether the current pixel is valid
                # if valid, just skip this pixel
                if proj_mask[y, x]:
                    continue

                if (x - 1 >= 0) and (x + 1 < self.W):
                    # only when both of right and left pixels are valid,
                    # the interpolated points will be calculated
                    if proj_mask[y, x - 1] and proj_mask[y, x + 1]:
                        # calculated the potential points
                        mean_points = (proj_image[y, x - 1] +
                                       proj_image[y, x + 1]) / 2
                        # change the current pixel to be valid
                        proj_mask[y, x] = 1
                        proj_image[y, x] = mean_points
                        interpolated_points.append(mean_points)

                        if labels is not None:
                            if proj_sem_label[y,
                                              x - 1] == proj_sem_label[y,
                                                                       x + 1]:
                                # if both pixels share the same semantic label,
                                # then just copy the semantic label
                                cur_label = proj_sem_label[y, x - 1]
                            else:
                                # if they have different labels, we consider it
                                # as boundary and set it as ignored label
                                cur_label = self.ignore_index
                            proj_sem_label[y, x] = cur_label
                            interpolated_labels.append(cur_label)

        # concatenate all the interpolated points and labels
        if len(interpolated_points) > 0:
            interpolated_points = np.array(
                interpolated_points, dtype=np.float32)
            points = np.concatenate((points, interpolated_points),
                                          axis=0)

        if labels is not None:
            interpolated_labels = np.array(interpolated_labels, dtype=np.int64)
            labels = np.concatenate((labels, interpolated_labels), axis=0)
        
        return points, labels


class RangeInterpolationCUDA(Transformation):
    def __init__(self, H=64, W=2048, fov_up=3.0, fov_down=-25.0, ignore_index=255, inplace=True, device='cuda'):
        super().__init__(inplace)

        self.H = H
        self.W = W
        self.fov_up = fov_up / 180.0 * np.pi
        self.fov_down = fov_down / 180.0 * np.pi
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        self.ignore_index = ignore_index
        self.device = device

    def __call__(self, pcloud, labels=None):
        points, labels = super().__call__(pcloud, labels)

        proj_image = torch.full((self.H, self.W, 4), -1, dtype=torch.float32, device=self.device)
        proj_idx = torch.full((self.H, self.W), -1, dtype=torch.int64, device=self.device)
        
        # get depth of all points
        depth = torch.norm(points[:, :3], dim=1)

        # get angles of all points
        yaw = -torch.atan2(points[:, 1], points[:, 0])
        pitch = torch.asin(points[:, 2] / depth)

        # get projection in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov

        # scale to image size using angular resolution
        # then round and clamp for use as index
        proj_x = (proj_x * self.W).floor().long().clamp(0, self.W - 1).to(torch.int64)
        proj_y = (proj_y * self.H).floor().long().clamp(0, self.H - 1).to(torch.int64)

        # order in decreasing depth
        indices = torch.arange(depth.size(0), dtype=torch.int64, device=self.device)
        order = torch.argsort(depth, descending=True)

        proj_idx[proj_y[order], proj_x[order]] = indices[order]
        proj_image[proj_y[order], proj_x[order]] = points[order]
        proj_mask = (proj_idx > 0).int()

        if labels is not None:
            proj_sem_label = torch.full((self.H, self.W), self.ignore_index, dtype=torch.int32, device=self.device)
            proj_sem_label[proj_y[order], proj_x[order]] = labels[order]

        # Interpolate Missing Points (Vectorized)
        left_shift = torch.roll(proj_mask, shifts=-1, dims=1)
        right_shift = torch.roll(proj_mask, shifts=1, dims=1)
        valid_mask = (~proj_mask.bool()) & left_shift.bool() & right_shift.bool()

        # Compute interpolated points
        interpolated_points = (torch.roll(proj_image, shifts=-1, dims=1) +
                               torch.roll(proj_image, shifts=1, dims=1)) / 2
        proj_image[valid_mask] = interpolated_points[valid_mask]
        proj_mask[valid_mask] = 1

        if labels is not None:
            left_labels = torch.roll(proj_sem_label, shifts=-1, dims=1)
            right_labels = torch.roll(proj_sem_label, shifts=1, dims=1)
            interpolated_labels = torch.where(left_labels == right_labels, left_labels, self.ignore_index)
            proj_sem_label[valid_mask] = interpolated_labels[valid_mask]

        # Add interpolated points and labels
        interpolated_points = proj_image[valid_mask]
        points = torch.cat((points, interpolated_points), dim=0)

        if labels is not None:
            interpolated_labels = proj_sem_label[valid_mask]
            labels = torch.cat((labels, interpolated_labels), dim=0)

        return points, labels

