# Copyright 2025 CEA LIST - Samir Abou Haidar
# Modifications based on code from Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai

# Copyright 2022 - Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai

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
import numpy as np

import yaml
from glob import glob
from tqdm import tqdm

def getnuScenesProcessor(base):

    class ClassMapper:
        def __init__(self):
            current_folder = os.path.dirname(os.path.realpath(__file__))
            self.mapping = np.load(
                os.path.join(current_folder, "nuscenes_class_mapping.npy")
            )

        def get_index(self, x):
            return self.mapping[x]


    class nuScenesProcessor(base):

        CLASS_NAME = [
            "barrier",
            "bicycle",
            "bus",
            "car",
            "construction_vehicle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "trailer",
            "truck",
            "driveable_surface",
            "other_flat",
            "sidewalk",
            "terrain",
            "manmade",
            "vegetation",
        ]

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # Class mapping
            current_folder = os.path.dirname(os.path.realpath(__file__))
            self.mapper = np.vectorize(ClassMapper().get_index)

            # List all keyframes
            self.list_frames = np.load(
                os.path.join(current_folder, "nuscenes_files.npz")
            )[self.phase]
            if self.phase == "train":
                assert len(self) == 28130
            elif self.phase == "val":
                assert len(self) == 6019
            elif self.phase == "test":
                assert len(self) == 6008
            else:
                raise ValueError(f"Unknown phase {self.phase}.")

            assert not self.instance_cutmix, "Instance CutMix not implemented on nuscenes"

        def __len__(self):
            return len(self.list_frames)

        def load_pc(self, index):
            # Load point cloud
            scan_name = os.path.join(self.rootdir, self.list_frames[index][0])
            pc = np.fromfile(
                scan_name,
                dtype=np.float32,
            )
            pc = pc.reshape((-1, 5))[:, :4]

            # Load segmentation labels
            label_name = os.path.join(self.rootdir, self.list_frames[index][1])
            labels = np.fromfile(
                os.path.join(self.rootdir, self.list_frames[index][1]),
                dtype=np.uint8,
            )
            labels_orig = labels
            labels = self.mapper(labels)
            
            # Label 0 should be ignored
            labels = labels - 1
            labels[labels == -1] = 255
            
            eval_filename = self.list_frames[index][2] # used to evaluate nuscenes with the official api
            
            # returns the point cloud, labels for learning with learning ignore, original labels, filename of scan, and corresponding label name, in addition to official api eval filename
            return pc, labels, labels_orig, scan_name, label_name, eval_filename
        

    return nuScenesProcessor