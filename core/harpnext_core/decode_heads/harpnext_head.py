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

from typing import List, Sequence
import torch.nn as nn
from torch import Tensor

class HARPNeXtHead(nn.Module):

    def __init__(self,
                 in_channels: int,
                 middle_channels: Sequence[int],
                 dropout_ratio: float = 0.5,
                 channels: int = 64,
                 num_classes: int = 19,
                 conv_seg_kernel_size: int =1
                 ) -> None:
        super(HARPNeXtHead, self).__init__()

        self.num_classes = num_classes
        self.channels = channels
        self.conv_seg_kernel_size = conv_seg_kernel_size

        self.mlps = nn.ModuleList()
        for i in range(len(middle_channels)):
            out_channels = middle_channels[i]
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(in_channels, out_channels, bias=False),
                    nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True)))
            in_channels = out_channels

        self.conv_seg = self.build_conv_seg(
            channels=self.channels,
            num_classes=self.num_classes,
            kernel_size=self.conv_seg_kernel_size)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)
        else:
            self.dropout = None

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> nn.Module:
        """Build Convolutional Segmentation Layers."""
        return nn.Linear(channels, num_classes)
    
    def cls_seg(self, feat: Tensor) -> Tensor:
        """Classify each points."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, voxel_dict: dict) -> dict:
        point_feats_backbone = voxel_dict['point_feats_backbone'][0]
        point_feats = voxel_dict['point_feats'][:-1]
        voxel_feats = voxel_dict['voxel_feats'][0]
        voxel_feats = voxel_feats.permute(0, 2, 3, 1)
        pts_coors = voxel_dict['coors']
        map_point_feats = voxel_feats[pts_coors[:, 0], pts_coors[:, 1], pts_coors[:, 2]]

        for i, mlp in enumerate(self.mlps):
            map_point_feats = mlp(map_point_feats)
            if i == 0:
                map_point_feats = map_point_feats + point_feats_backbone
            else:
                map_point_feats = map_point_feats + point_feats[-i]
        
        seg_logit = self.cls_seg(map_point_feats)
        voxel_dict['seg_logit'] = seg_logit
        return voxel_dict


    def predict(self, voxel_dict: dict)-> List[Tensor]:
        seg_logits = voxel_dict['seg_logit']
        return seg_logits

