# Copyright 2025 CEA LIST - Samir Abou Haidar

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from core.harpnext_core.segmentor.harpnext import HARPNeXt


class Network:
    def __init__(self, net, netconfig):
        self.net = net
        self.netconfig = netconfig
    # Build network
    def build_network(self):

        model_cfg = self.netconfig["model"]
        # Configuration for feature encoder
        voxel_encoder_cfg = model_cfg["voxel_encoder"]
        # Configuration for backbone
        backbone_cfg = model_cfg["backbone"]
        # Configuration for decode head
        decode_head_cfg = model_cfg["decode_head"]
        # Configuration for auxiliary heads
        auxiliary_head_cfg = [
            {'channels': model_cfg["auxiliary_heads"]["channels"], 'num_classes': self.netconfig["classif"]["nb_class"], 'dropout_ratio': model_cfg["auxiliary_heads"]["dropout_ratio"], 'conv_seg_kernel_size': model_cfg["auxiliary_heads"]["conv_seg_kernel_size"], 'ignore_index': self.netconfig["classif"]["ignore_class"]},
            {'channels': model_cfg["auxiliary_heads"]["channels"], 'num_classes': self.netconfig["classif"]["nb_class"], 'dropout_ratio': model_cfg["auxiliary_heads"]["dropout_ratio"], 'conv_seg_kernel_size': model_cfg["auxiliary_heads"]["conv_seg_kernel_size"], 'ignore_index': self.netconfig["classif"]["ignore_class"], 'indices': 2},
            {'channels': model_cfg["auxiliary_heads"]["channels"], 'num_classes': self.netconfig["classif"]["nb_class"], 'dropout_ratio': model_cfg["auxiliary_heads"]["dropout_ratio"], 'conv_seg_kernel_size': model_cfg["auxiliary_heads"]["conv_seg_kernel_size"], 'ignore_index': self.netconfig["classif"]["ignore_class"], 'indices': 3},
            {'channels': model_cfg["auxiliary_heads"]["channels"], 'num_classes': self.netconfig["classif"]["nb_class"], 'dropout_ratio': model_cfg["auxiliary_heads"]["dropout_ratio"], 'conv_seg_kernel_size': model_cfg["auxiliary_heads"]["conv_seg_kernel_size"], 'ignore_index': self.netconfig["classif"]["ignore_class"], 'indices': 4},
        ]

        model = HARPNeXt(voxel_encoder=voxel_encoder_cfg,
                    backbone=backbone_cfg,
                    decode_head=decode_head_cfg,
                    auxiliary_head=auxiliary_head_cfg,
                    neck=None)
        
        return model