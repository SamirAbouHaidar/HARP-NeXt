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

from typing import List, Optional
import torch.nn as nn
from torch import Tensor
from core.harpnext_core.encoder.features_encoder import FeaturesEncoder
from core.harpnext_core.backbone.harpnext_backbone import HARPNeXtBackbone
from core.harpnext_core.decode_heads.harpnext_head import HARPNeXtHead
from core.harpnext_core.decode_heads.aux_head import AuxHead

class HARPNeXt(nn.Module):
    def __init__(self,
                 voxel_encoder: dict,
                 backbone: dict,
                 decode_head: dict,
                 neck: Optional[dict] = None,
                 auxiliary_head: Optional[List[dict]] = None) -> None:
        super(HARPNeXt, self).__init__()

        # Initialize the voxel encoder with configuration from the dictionary
        self.voxel_encoder = FeaturesEncoder(**voxel_encoder)
        
        # Initialize the backbone with configuration from the dictionary
        self.backbone = HARPNeXtBackbone(**backbone)
        
        # Initialize the decode head with configuration from the dictionary
        self.decode_head = HARPNeXtHead(**decode_head)

        # Initialize the auxiliary heads (if provided)
        self.auxiliary_head = nn.ModuleList()
        if auxiliary_head:
            for aux_head_cfg in auxiliary_head:
                self.auxiliary_head.append(AuxHead(**aux_head_cfg))
        
        if neck:
            self.neck = nn.Module(**neck)
        else:
            self.neck = None

    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        """Extract features from points."""
        voxel_dict = batch_inputs_dict['voxels'].copy()
        voxel_dict = self.voxel_encoder(voxel_dict)
        voxel_dict = self.backbone(voxel_dict)
        if self.neck:
            voxel_dict = self.neck(voxel_dict)
        return voxel_dict

    def forward(self,
                 batch_inputs_dict: dict, training = True):

        losses_seg_logits = dict()
        voxel_dict = self.extract_feat(batch_inputs_dict)
        
        if training:
            voxel_dict = self.decode_head.forward(voxel_dict)  
            seg_logits = self.decode_head.predict(voxel_dict) 
            losses_seg_logits.update(self.add_prefix(voxel_dict['seg_logit'], 'HARPNeXtHead.seg_logit'))

            if isinstance(self.auxiliary_head, nn.ModuleList):
                for idx, aux_head in enumerate(self.auxiliary_head):
                    voxel_dict = aux_head.forward(voxel_dict)
                    losses_seg_logits.update(self.add_prefix(voxel_dict['seg_logit'], f'AuxHead_{idx}.seg_logit'))
            else:
                voxel_dict = self.auxiliary_head.forward(voxel_dict)
                losses_seg_logits.update(self.add_prefix(voxel_dict['seg_logit'], 'AuxHead.seg_logit'))

        else:
            voxel_dict = self.decode_head.forward(voxel_dict)
            seg_logits = self.decode_head.predict(voxel_dict)
            losses_seg_logits.update(self.add_prefix(voxel_dict['seg_logit'], 'HARPNeXtHead.seg_logit'))

        out_dict = {
            "seg_logits" : seg_logits,
            "losses_seg_logits" : losses_seg_logits
        }
        return out_dict

    
    def add_prefix(self, input: Tensor, prefix: str) -> dict:
        """Add prefix for a Tensor.

        Args:
            input (Tensor): The input Tensor.
            prefix (str): The prefix to add.

        Returns:

            dict: The dict with keys updated with ``prefix``.
        """
        return {f'{prefix}': input}