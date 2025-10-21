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

from typing import List
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class AuxHead(nn.Module):

    def __init__(self,
                channels=128,
                num_classes=20,
                dropout_ratio=0,
                conv_seg_kernel_size=1,
                indices: int = 0,
                ignore_index=255,
                **kwargs) -> None:
        super(AuxHead, self).__init__()

        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_seg_kernel_size = conv_seg_kernel_size
        self.indices = indices
        self.ignore_index = ignore_index

        self.conv_seg = self.build_conv_seg(
            channels=self.channels,
            num_classes=self.num_classes,
            kernel_size=self.conv_seg_kernel_size)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)
        else:
            self.dropout = None
        
    def build_conv_seg(self, channels: int, num_classes: int, kernel_size: int) -> nn.Module:
        return nn.Conv2d(channels, num_classes, kernel_size=kernel_size, padding=kernel_size // 2)
    
    def cls_seg(self, feat: Tensor) -> Tensor:
        """Classify each points."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, voxel_dict: dict) -> dict:
        """Forward function."""
        seg_logit = self.cls_seg(voxel_dict['voxel_feats'][self.indices])
        voxel_dict['seg_logit'] = seg_logit
        return voxel_dict
    

    def predict(self, voxel_dict: dict) -> List[Tensor]:

        seg_logits = voxel_dict['seg_logit']
        seg_logits = seg_logits.permute(0, 2, 3, 1).contiguous()

        coors = voxel_dict['coors']
        batch_size = coors[-1, 0].item() + 1
        seg_pred_list = []
        
        for batch_idx in range(batch_size):
            batch_mask = coors[:, 0] == batch_idx
            res_coors = coors[batch_mask]
            proj_x = res_coors[:, 2]
            proj_y = res_coors[:, 1]
            seg_pred_list.append(seg_logits[batch_idx, proj_y, proj_x])

        return seg_pred_list