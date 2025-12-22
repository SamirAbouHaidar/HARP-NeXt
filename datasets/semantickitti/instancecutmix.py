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
from glob import glob
import torch
import warnings
import utils.transformations.transforms as tr

class InstanceCutMix:
    def __init__(self, phase="train", temp_dir="/tmp/semantic_kitti_instances/"):

        # Train or Trainval
        self.phase = phase
        assert self.phase in ["train", "trainval"]

        # List of files containing instances for bicycle, motorcycle, person, bicyclist
        # self.bank = {1: [], 2: [], 5: [], 6: []}
        self.bank = {1: [], 2: [], 4: [], 5: [], 6: []}

        # Directory where to store instances
        self.rootdir = os.path.join(temp_dir, self.phase)
        for id_class in self.bank.keys():
            os.makedirs(os.path.join(self.rootdir, f"{id_class}"), exist_ok=True)

        # Load instances
        for key in self.bank.keys():
            self.bank[key] = glob(os.path.join(self.rootdir, f"{key}", "*.bin"))
        self.__loaded__ = self.test_loaded()
        if not self.__loaded__:
            warnings.warn(
                "Instances must be extracted and saved on disk before training"
            )

        # Augmentations applied on Instances
        self.rot = tr.Compose(
            (
                tr.FlipXY(inplace=True),
                tr.Rotation(inplace=True),
                tr.Scale(dims=(0, 1, 2), range=0.1, inplace=True),
            )
        )

        # For each class, maximum number of instance to add
        self.num_to_add = 40

        # Voxelization of 1m to downsample point cloud to ensure that
        # center of the instances are at least 1m away
        self.vox = tr.Voxelize(dims=(0, 1, 2), voxel_size=1.0, random=True)


    def test_loaded(self):
        self.__loaded__ = False
        if self.phase == "train":
            if len(self.bank[1]) != 5083:
                return False
            if len(self.bank[2]) != 3092:
                return False
            if len(self.bank[4]) != 7419:
                return False
            if len(self.bank[5]) != 8084:
                return False
            if len(self.bank[6]) != 1551:
                return False
        elif self.phase == "trainval":
            if len(self.bank[1]) != 8213:
                return False
            if len(self.bank[2]) != 4169:
                return False
            if len(self.bank[4]) != 10516:
                return False
            if len(self.bank[5]) != 12190:
                return False
            if len(self.bank[6]) != 2943:
                return False
        self.__loaded__ = True
        return True


    def cut(self, pc, class_label, instance_label):
        for id_class in self.bank.keys():
            where_class = (class_label == id_class)
            all_instances = np.unique(instance_label[where_class])
            for id_instance in all_instances:
                # Segment instance
                where_ins = (instance_label == id_instance)
                if where_ins.sum() <= 5: continue
                instance = pc[where_ins, :]
                # Center instance
                instance[:, :2] -= instance[:, :2].mean(0, keepdims=True)
                instance[:, 2] -= instance[:, 2].min(0, keepdims=True)
                # Save instance
                pathfile = os.path.join(
                    self.rootdir, f"{id_class}", f"{len(self.bank[id_class]):07d}.bin"
                )
                instance.tofile(pathfile)
                self.bank[id_class].append(pathfile)

    def mix(self, pc, class_label):

        # Find potential location where to add new object (on a surface)
        pc_vox, class_label_vox = self.vox(pc, class_label)
        where_surface = np.where((class_label_vox >= 8) & (class_label_vox <= 10))[0]
        where_surface = where_surface[torch.randperm(len(where_surface))]

        # Add instances of each class in bank
        id_tot = 0
        new_pc, new_label  = [pc], [class_label]
        for id_class in self.bank.keys():
            nb_to_add = torch.randint(self.num_to_add, (1,))[0]
            which_one = torch.randint(len(self.bank[id_class]), (nb_to_add,))
            for ii in range(nb_to_add):
                # Point p where to add the instance
                p = pc_vox[where_surface[min(id_tot, len(where_surface)-1)]]
                # Extract instance
                object = self.bank[id_class][which_one[ii]]
                object = np.fromfile(object, dtype=np.float32).reshape((-1, 4))
                # Augment instance
                label = np.ones((object.shape[0],), dtype=int) * id_class  #was np.int
                object, label = self.rot(object, label)
                # Move instance at point p
                object[:, :3] += p[:3][None]
                # Add instance in the point cloud
                new_pc.append(object)
                # Add corresponding label
                new_label.append(label)
                id_tot += 1

        return np.concatenate(new_pc, 0), np.concatenate(new_label, 0)

    def __call__(self, pc, class_label, instance_label):
        if not self.__loaded__:
            self.cut(pc, class_label, instance_label)

            return pc, class_label
        return self.mix(pc, class_label)

class PolarMix:
    def __init__(self, classes=None):
        self.classes = classes
        self.rot = tr.Rotation(inplace=False)

    def __call__(self, pc1, label1, pc2, label2):
        # --- Scene-level swapping
        if torch.rand(1)[0] < 0.5:
            sector = (2 * float(torch.rand(1)[0]) - 1) * np.pi
            # --- Remove a 180 deg sector in 1
            theta1 = (np.arctan2(pc1[:, 1], pc1[:, 0]) - sector) % (2 * np.pi)
            where1 = (theta1 > 0) & (theta1 < np.pi)
            where1 = ~where1
            pc1, label1 = pc1[where1], label1[where1]
            # --- Replace by corresponding 180 deg sector in 2
            theta2 = (np.arctan2(pc2[:, 1], pc2[:, 0]) - sector) % (2 * np.pi)
            where2 = (theta2 > 0) & (theta2 < np.pi)
            #
            pc = np.concatenate((pc1, pc2[where2]), axis=0)
            label = np.concatenate((label1, label2[where2]), axis=0)
        else:
            pc, label = pc1, label1

        # --- Instance level augmentation
        where_class = label2 == self.classes[0]
        for id_class in self.classes[1:]:
            where_class |= label2 == id_class
        if where_class.sum() > 0:
            pc2, label2 = pc2[where_class], label2[where_class]
            pc22 = self.rot(pc2, label2)[0]
            pc23 = self.rot(pc2, label2)[0]
            pc = np.concatenate((pc, pc2, pc22, pc23), axis=0)
            label = np.concatenate((label, label2, label2, label2), axis=0)

        return pc, label

