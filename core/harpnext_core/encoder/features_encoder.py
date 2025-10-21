# Copyright 2025 CEA LIST - Samir Abou Haidar
# Modifications based on code from Open-MMLab, 2018-2019

# Copyright 2018-2019 Open-MMLab

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch_scatter
from typing import Optional, Sequence


class FeaturesEncoder(nn.Module):
    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: Sequence[int] = [],
                 with_distance: bool = False,
                 with_cluster_center: bool = False,
                 with_pre_norm: bool = False,
                 feat_compression: Optional[int] = None) -> None:
        super(FeaturesEncoder, self).__init__()
        assert len(feat_channels) > 0

        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # Adjust input channels if we use distance or cluster center
        if with_distance:
            in_channels += 1
        if with_cluster_center:
            in_channels += 3
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center

        feat_channels = [self.in_channels] + list(feat_channels)
        if with_pre_norm:
            self.pre_norm = nn.BatchNorm1d(num_features= self.in_channels, eps=norm_cfg['eps'],  momentum=norm_cfg['momentum'])
        else:
            self.pre_norm = None

        ffe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            norm_layer = nn.BatchNorm1d(num_features= out_filters, eps=norm_cfg['eps'],  momentum=norm_cfg['momentum'])
            if i == len(feat_channels) - 2:
                ffe_layers.append(nn.Linear(in_filters, out_filters))
            else:
                ffe_layers.append(
                    nn.Sequential(
                        nn.Linear(in_filters, out_filters, bias=False),
                        norm_layer,
                        nn.ReLU(inplace=True)
                    )
                )
        self.ffe_layers = nn.ModuleList(ffe_layers)

        # Compression layer if specified
        self.compression_layers = None
        if feat_compression is not None:
            self.compression_layers = nn.Sequential(
                nn.Linear(feat_channels[-1], feat_compression),
                nn.ReLU(inplace=True)
            )

    def forward(self, voxel_dict: dict) -> dict:
        features = voxel_dict['voxels']
        coors = voxel_dict['coors']
        features_ls = [features]

        voxel_coors, inverse_map = torch.unique(
            coors, return_inverse=True, dim=0)

        # Add distance feature if enabled
        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Find distance of x, y, and z from frustum center
        if self._with_cluster_center:
            voxel_mean = torch_scatter.scatter_mean(
                features, inverse_map, dim=0)
            points_mean = voxel_mean[inverse_map]
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Concatenate all features
        features = torch.cat(features_ls, dim=-1)

        # Apply pre-normalization if required
        if self.pre_norm is not None:
            features = self.pre_norm(features)

        # Pass through FFE layers
        point_feats = []
        for ffe in self.ffe_layers:
            features = ffe(features)
            point_feats.append(features)

        # Aggregate features for voxels using torch_scatter
        voxel_feats = torch_scatter.scatter_max(
            features, inverse_map, dim=0)[0]

        # Apply compression layer if specified
        if self.compression_layers is not None:
            voxel_feats = self.compression_layers(voxel_feats)

        voxel_dict['voxel_feats'] = voxel_feats
        voxel_dict['voxel_coors'] = voxel_coors
        voxel_dict['point_feats'] = point_feats

        return voxel_dict
