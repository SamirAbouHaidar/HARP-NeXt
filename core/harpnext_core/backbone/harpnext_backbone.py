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


from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch import Tensor

class ConvSENeXt(nn.Module):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 reduction: int = 16,
                 norm_cfg=dict(type='BN2d', eps=1e-6, momentum=0.01),
                 act_cfg=dict(type='HSwish', inplace=True),
                 dw_conv_kernel = 7,
                 dw_conv_bias = True) -> None:
        super(ConvSENeXt, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dwconv layer
        self.conv_dw = nn.Conv2d(
            in_channels=inplanes, 
            out_channels=inplanes, 
            kernel_size=dw_conv_kernel,
            stride=stride, 
            padding=dilation,
            groups=inplanes, 
            bias=dw_conv_bias,
            device=device
        )
        self.norm1 = nn.BatchNorm2d(num_features=planes, eps=norm_cfg['eps'], momentum=norm_cfg['momentum'], device=device)

        # pwconv layer
        self.pointwise = nn.Conv2d(
            in_channels=inplanes, 
            out_channels=planes, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=dw_conv_bias,
            device=device
        )
        self.norm2 = nn.BatchNorm2d(num_features=planes, eps=norm_cfg['eps'], momentum=norm_cfg['momentum'], device=device)

         # Activation
        self.activation = nn.Hardswish(inplace=act_cfg['inplace'])

        # Squeeze-and-Excitation (SE) module
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes, planes // reduction, kernel_size=1, device=device),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes // reduction, planes, kernel_size=1, device=device),
            nn.Hardsigmoid(inplace=True)
        )

        # Downsample for skip connections
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # Depthwise convolution
        out = self.conv_dw(x)
        out = self.norm1(out)
        out = self.activation(out)

        # Pointwise convolution
        out = self.pointwise(out)
        out = self.norm2(out)

        # Apply Squeeze-and-Excitation (SE) module
        se_weight = self.se(out)
        out = out * se_weight

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out
 

class EfficientTransformationPipeline(nn.Module):
    def __init__(self, nx, ny):
        super(EfficientTransformationPipeline, self).__init__()
        self.nx = nx
        self.ny = ny
    
    def point2cluster(self, point_features: torch.Tensor, pts_coors: torch.Tensor, stride: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        coors = pts_coors.clone()
        coors[:, 1:3] //= stride

        voxel_coors, inverse_map = torch.unique(coors, return_inverse=True, dim=0, sorted=True)
        
        cluster_features = torch.zeros(voxel_coors.size(0), point_features.size(1), 
                                       dtype=point_features.dtype, device=point_features.device)
        cluster_features = torch_scatter.scatter_max(point_features, inverse_map, dim=0, out=cluster_features)[0]
        
        return voxel_coors, cluster_features
    
    def cluster2pixel(self, cluster_features: torch.Tensor, coors: torch.Tensor, batch_size: int, stride: int = 1) -> torch.Tensor:
        nx = self.nx // stride
        ny = self.ny // stride
        
        indices = coors.t().long()
        values = cluster_features
        size = (batch_size, ny, nx, cluster_features.shape[-1])
        
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size)
        pixel_features = sparse_tensor.to_dense()
        
        return pixel_features.permute(0, 3, 1, 2).contiguous()
    
    def pixel2point(self, pixel_features: torch.Tensor, coors: torch.Tensor, stride: int = 1) -> torch.Tensor:
        batch_indices = coors[:, 0]
        y_indices = coors[:, 1] // stride
        x_indices = coors[:, 2] // stride
        
        return pixel_features[batch_indices, :, y_indices, x_indices].contiguous()


class HARPNeXtBackbone(nn.Module):
    arch_settings = {
    10: (ConvSENeXt, (1, 1, 1, 1))
    }

    def __init__(self,
                 in_channels: int = 16,
                 point_in_channels: int = 384,
                 output_shape: Sequence[int] = [],
                 depth: int = 34,
                 stem_channels: int = 128,
                 num_stages: int = 4,
                 out_channels: Sequence[int] = (128, 128, 128, 128),
                 strides: Sequence[int] = (1, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1),
                 fuse_channels: Sequence[int] = (256, 128),
                 dw_conv_kernel = 7,
                 dw_conv_bias = True,
                 inter_align_corners = True) -> None:
        super(HARPNeXtBackbone, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for HARPNeXtBackbone.')

        self.block, stage_blocks = self.arch_settings[depth]
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        assert len(stage_blocks) == len(out_channels) == len(strides) == len(
        dilations) == num_stages, \
        'The length of stage_blocks, out_channels, strides and ' \
        'dilations should be equal to num_stages.'

        self.stem = self._make_stem_layer(in_channels, stem_channels)
        self.point_stem = self._make_point_layer(point_in_channels, stem_channels)
        self.fusion_stem = self._make_fusion_layer(stem_channels * 2, stem_channels)

        self.etp = EfficientTransformationPipeline(self.nx, self.ny)

        inplanes = stem_channels
        self.res_layers = []
        self.point_fusion_layers = nn.ModuleList()
        self.pixel_fusion_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.strides = []
        self.inter_align_corners = inter_align_corners

        overall_stride = 1
        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]
            overall_stride = stride * overall_stride
            self.strides.append(overall_stride)
            dilation = dilations[i]
            planes = out_channels[i]
            
            res_layer = self._make_res_layer(block=self.block, inplanes=inplanes, planes=planes, num_blocks=num_blocks, stride=stride, dilation=dilation, dw_conv_kernel=dw_conv_kernel, dw_conv_bias=dw_conv_bias)
            self.point_fusion_layers.append(self._make_point_layer(inplanes + planes, planes))
            self.pixel_fusion_layers.append(self._make_fusion_layer(planes * 2, planes))
            self.attention_layers.append(self._make_attention_layer(planes))

            inplanes = planes
            # self.res_layers.append(res_layer)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        in_channels = stem_channels + sum(out_channels)
        self.fuse_layers = []
        self.point_fuse_layers = []

        for i, fuse_channel in enumerate(fuse_channels):
            fuse_layer = self._make_fusion_layer(in_channels, fuse_channel)
            point_fuse_layer = self._make_point_layer(in_channels, fuse_channel)
            in_channels = fuse_channel
            layer_name = f'fuse_layer{i + 1}'
            point_layer_name = f'point_fuse_layer{i + 1}'
            self.add_module(layer_name, fuse_layer)
            self.add_module(point_layer_name, point_fuse_layer)
            self.fuse_layers.append(layer_name)
            self.point_fuse_layers.append(point_layer_name)

    #pixels stem
    def _make_stem_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2, eps=1e-3, momentum=0.01),
            nn.Hardswish(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.Hardswish(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.Hardswish(inplace=True)
        )

    # points stem
    def _make_point_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )

    # fusion layer
    def _make_fusion_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.Hardswish(inplace=True)
        )
    

    # attention layer
    def _make_attention_layer(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01),
            nn.Hardswish(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01),
            nn.Sigmoid()
        )

    # residual ConvSENeXt
    def _make_res_layer(self, block: nn.Module, inplanes, planes, num_blocks, stride, dilation, dw_conv_kernel, dw_conv_bias):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False, device= self.device),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01, device=self.device),
            )

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='HSwish', inplace=True),
                dw_conv_kernel = dw_conv_kernel,
                dw_conv_bias=dw_conv_bias))
        inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    dilation=dilation,
                    norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
                    act_cfg=dict(type='HSwish', inplace=True),
                    dw_conv_kernel = dw_conv_kernel,
                    dw_conv_bias=dw_conv_bias))

        return nn.Sequential(*layers)
    
    def forward(self, voxel_dict):
        point_feats = voxel_dict['point_feats'][-1]
        voxel_feats = voxel_dict['voxel_feats']
        voxel_coors = voxel_dict['voxel_coors']
        pts_coors = voxel_dict['coors']
        batch_size = pts_coors[-1, 0].item() + 1

        x = self.etp.cluster2pixel(voxel_feats, voxel_coors, batch_size, stride=1)
        x = self.stem(x)  
        map_point_feats = self.etp.pixel2point(x, pts_coors, stride=1)
        fusion_point_feats = torch.cat((map_point_feats, point_feats), dim=1)
        point_feats = self.point_stem(fusion_point_feats)

        stride_voxel_coors, cluster_feats = self.etp.point2cluster(point_feats, pts_coors, stride=1)
        pixel_feats = self.etp.cluster2pixel(cluster_feats, stride_voxel_coors, batch_size, stride=1)
        fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)
        x = self.fusion_stem(fusion_pixel_feats)

        outs = [x]
        out_points = [point_feats]
        prev_pixel_feats = pixel_feats

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            # cluster-to-point fusion
            map_point_feats = self.etp.pixel2point(x, pts_coors, stride=self.strides[i])
            fusion_point_feats = torch.cat((map_point_feats, point_feats), dim=1)
            point_feats = self.point_fusion_layers[i](fusion_point_feats)

            # point-to-cluster fusion (skip for the last layer)
            if i < len(self.res_layers) - 1:
                stride_voxel_coors, cluster_feats = self.etp.point2cluster(point_feats, pts_coors, stride=self.strides[i])
                pixel_feats = self.etp.cluster2pixel(cluster_feats, stride_voxel_coors, batch_size, stride=self.strides[i])

            prev_pixel_feats_resized = F.interpolate(prev_pixel_feats, size=x.shape[2:], mode='bilinear', align_corners=self.inter_align_corners)

            # Concatenate the resized tensor
            fusion_pixel_feats = torch.cat((prev_pixel_feats_resized, x), dim=1)
            fuse_out = self.pixel_fusion_layers[i](fusion_pixel_feats)
            attention_map = self.attention_layers[i](fuse_out)
            x = fuse_out * attention_map + x

            outs.append(x)
            out_points.append(point_feats)
            
            if i < len(self.res_layers) - 1:
                prev_pixel_feats = pixel_feats

        for i in range(len(outs)):
            if outs[i].shape != outs[0].shape:
                outs[i] = F.interpolate(outs[i], size=outs[0].size()[2:], mode='bilinear', align_corners=True)

        outs[0] = torch.cat(outs, dim=1)
        out_points[0] = torch.cat(out_points, dim=1)

        for layer_name, point_layer_name in zip(self.fuse_layers, self.point_fuse_layers):
            fuse_layer = getattr(self, layer_name)
            outs[0] = fuse_layer(outs[0])
            point_fuse_layer = getattr(self, point_layer_name)
            out_points[0] = point_fuse_layer(out_points[0])

        voxel_dict['voxel_feats'] = outs
        voxel_dict['point_feats_backbone'] = out_points
        return voxel_dict