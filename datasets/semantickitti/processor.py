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
import yaml
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from datasets.semantickitti.instancecutmix import InstanceCutMix, PolarMix


def getSemanticKITTIProcessor(base):

    class SemanticKITTIProcessor(base):
                
        CLASS_NAME = [
            "car",              # 0
            "bicycle",          # 1
            "motorcycle",       # 2
            "truck",            # 3
            "other-vehicle",    # 4
            "person",           # 5
            "bicyclist",        # 6
            "motorcyclist",     # 7
            "road",             # 8
            "parking",          # 9
            "sidewalk",         # 10
            "other-ground",     # 11
            "building",         # 12
            "fence",            # 13
            "vegetation",       # 14
            "trunk",            # 15
            "terrain",          # 16
            "pole",             # 17
            "traffic-sign",     # 18
        ]

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # Config file and class mapping
            current_folder = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(current_folder, "semantic-kitti.yaml")) as stream:
                semkittiyaml = yaml.safe_load(stream)
            self.labels = semkittiyaml["labels"]
            self.color_map = semkittiyaml["color_map"]
            self.content = semkittiyaml["content"]
            self.learning_map = semkittiyaml["learning_map"]
            self.learning_map_inv = semkittiyaml["learning_map_inv"]
            self.learning_ignore = semkittiyaml["learning_ignore"]
            self.split = semkittiyaml["split"]

            # Split
            if self.phase == "train":
                split = semkittiyaml["split"]["train"]
            elif self.phase == "val":
                split = semkittiyaml["split"]["valid"]
            elif self.phase == "test":
                split = semkittiyaml["split"]["test"]
            elif self.phase == "trainval":
                split = semkittiyaml["split"]["train"] + semkittiyaml["split"]["valid"]
            else:
                raise Exception(f"Unknown split {self.phase}")

            # Find all files
            self.im_idx = []
            self.im_idx_label = []
            for i_folder in np.sort(split):
                self.im_idx.extend(
                    glob(
                        os.path.join(
                            self.rootdir,
                            "dataset",
                            "sequences",
                            str(i_folder).zfill(2),
                            "velodyne",
                            "*.bin",
                        )
                    )
                )
                self.im_idx_label.extend(
                    glob(
                        os.path.join(
                            self.rootdir,
                            "dataset",
                            "sequences",
                            str(i_folder).zfill(2),
                            "labels",
                            "*.label",
                        )
                    )
                )
            self.im_idx = np.sort(self.im_idx)
            self.im_idx_label = np.sort(self.im_idx_label)

            # Training with instance cutmix
            if self.instance_cutmix:
                # PolarMix
                self.polarmix = PolarMix(classes=[1, 2, 4, 5, 6])
                # CutMix
                assert (
                    self.phase != "test" and self.phase != "val"
                ), "Instance cutmix should not be applied at test or val time"
                self.cutmix = InstanceCutMix(phase=self.phase)
                if not self.cutmix.test_loaded():
                    print("Instance CutMix is enabled, Extracting instances before training...")
                    for index in tqdm(range(len(self))):
                        x = len(self)
                        self.load_pc(index)
                    print("Done.")
                assert self.cutmix.test_loaded(), "Instances not extracted correctly"

        def __len__(self):
            return len(self.im_idx)
        
        def __load_pc_internal__(self, index):
            # Load point cloud
            pc = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
            # print(f"index is {index}\n")
            # print(f"pointcloud is {pc}\n")
            # Extract Label
            if self.phase == "test":
                labels = np.zeros((pc.shape[0], 1), dtype=np.uint8)
            else:
                labels_inst = np.fromfile(
                    self.im_idx[index].replace("velodyne", "labels")[:-3] + "label",
                    dtype=np.uint32,
                ).reshape((-1, 1))
                # print(f"labels_inst is {labels_inst}\n")
                labels = labels_inst & 0xFFFF  # delete high 16 digits binary
                # print(f"labels is {labels}\n")
                labels = np.vectorize(self.learning_map.__getitem__)(labels).astype(
                    np.int32
                )
            
            # print(f"labels_vect is {labels}\n")

            # Map ignore index (0) to 255
            labels = labels[:, 0] - 1
            labels[labels == -1] = 255

            return pc, labels, labels_inst[:, 0]
        
        def load_pc(self, index):
            # Load the point cloud and labels
            pc, labels, labels_inst = self.__load_pc_internal__(index)

            # Store the original labels before any modifications
            labels_orig = labels_inst.reshape((-1))

            # Instance CutMix and Polarmix
            if self.instance_cutmix:
                # Polarmix
                if self.cutmix.test_loaded():
                    # Randomly select a new index for mixing
                    new_index = torch.randint(len(self), (1,))[0]
                    new_pc, new_label, new_labels_inst = self.__load_pc_internal__(new_index)
                    
                    # Apply polarmix to pc and labels
                    pc, labels = self.polarmix(pc, labels, new_pc, new_label)

                # Cutmix
                pc, labels = self.cutmix(pc, labels, labels_inst)

                # Create reverse mapping
                reverse_learning_map = {v: k for k, v in self.learning_map.items()}

                # get labels_orig back from the new labels after polarmix and instancecutmix
                labels_orig = np.array([reverse_learning_map.get(label + 1, label) for label in labels])
                labels_orig[labels == 255] = 0  # Reset ignore index if needed

            eval_filename = None  # Pass it to comply with nuscenes

            # Return the point cloud, transformed labels, derived original labels, and metadata
            return pc, labels, labels_orig, self.im_idx[index], self.im_idx_label[index], eval_filename

    return SemanticKITTIProcessor