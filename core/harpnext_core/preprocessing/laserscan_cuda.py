# Copyright 2025 CEA LIST - Samir Abou Haidar
# Modifications based on code from University of Bonn (MIT License)

# Copyright (c) 2019, University of Bonn

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# limitations under the License.

#!/usr/bin/env python3
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch

class LaserScanCUDA:
  """Class that contains LaserScan with x, y, z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self, dataset=None, project_range=False, range_H=64, range_W=1024, fov_up=10.0, fov_down=-30.0, device = 'cuda', rank = 0):
    self.dataset = dataset
    if self.dataset is None:
      raise Exception(f"Dataset provided {self.dataset} is not semantic_kitti or nuscenes.")
    self.project_range = project_range
    self.range_H = range_H
    self.range_W = range_W
    self.proj_fov_up = fov_up
    self.proj_fov_down = fov_down

    if self.dataset == "nuscenes":
        self.classes_names = [
            "barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle",
            "pedestrian", "traffic_cone", "trailer", "truck", "driveable_surface", 
            "other_flat", "sidewalk", "terrain", "manmade", "vegetation",
        ]
        with open("./datasets/nuscenes/nuscenes.yaml") as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml["learning_map"]

    elif self.dataset == "semantic_kitti":
        self.classes_names = [
            "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", 
            "bicyclist", "motorcyclist", "road", "parking", "sidewalk", "other-ground", 
            "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",
        ]
        with open("./datasets/semantickitti/semantic-kitti.yaml") as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml["learning_map"]
    else:
        raise Exception(f"Dataset {dataset} is not supported. Please use semantic_kitti or nuscenes.")

    self.device = device
    self.rank = rank
    self.reset()

  def reset(self):
    """ Reset scan members. """
    ########################################################################################
    # Point Cloud
    self.points = torch.zeros((0, 3), dtype=torch.float32, device=self.device)         # [m, 3]: x, y, z
    self.remissions = torch.zeros((0, 1), dtype=torch.float32, device=self.device)     # [m, 1]: remission
    ########################################################################################
    # Range Image
    # projected range image - [H, W] range (-1 is no data)
    self.proj_range = torch.full((self.range_H, self.range_W), -1, dtype=torch.float32, device=self.device)

    # unprojected range (list of depths for each point)
    self.unproj_range = torch.zeros((0, 1), dtype=torch.float32, device=self.device)

    # projected point cloud xyz - [H, W, 3] xyz coord (-1 is no data)
    self.proj_range_xyz = torch.full((self.range_H, self.range_W, 3), -1, dtype=torch.float32, device=self.device)

    # projected remission - [H, W] intensity (-1 is no data)
    self.proj_range_remission = torch.full((self.range_H, self.range_W), -1, dtype=torch.float32, device=self.device)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H, W] index (-1 is no data)
    self.proj_range_idx = torch.full((self.range_H, self.range_W), -1, dtype=torch.int32, device=self.device)

    # for each point, where it is in the range image
    self.proj_range_x = torch.zeros((0, 1), dtype=torch.float32, device=self.device)   # [m, 1]: x
    self.proj_range_y = torch.zeros((0, 1), dtype=torch.float32, device=self.device)   # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    self.proj_range_mask = torch.zeros((self.range_H, self.range_W), dtype=torch.int32, device=self.device)  # [H, W] mask
    
    ########################################################################################


  def size(self):
    """ Return the size of the point cloud. """
    return self.points.size(0)

  def __len__(self):
    return self.size()

  def open_scan(self, filename):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()

    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    scan = np.fromfile(filename, dtype=np.float32)
    if self.dataset == "nuscenes":
        scan = scan.reshape((-1, 5))[:, :4]
    elif self.dataset == "semantic_kitti":
        scan = scan.reshape((-1, 4))
    else:
      raise Exception("scan file did not open successfully.")
      

    # put in attribute
    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 3]  # get remission

    # Convert points and remissions to tensors
    points = torch.tensor(points, dtype=torch.float32)
    remissions = torch.tensor(remissions, dtype=torch.float32)

    self.set_points(points, remissions)

  def set_points(self, points, remissions=None):
    """ Set scan attributes (instead of opening from file). """
    # reset just in case there was an open structure
    self.reset()

    # check scan makes sense
    if not isinstance(points, torch.Tensor):
        raise TypeError("Scan should be a PyTorch tensor")

    # check remission makes sense
    if remissions is not None and not isinstance(remissions, torch.Tensor):
        raise TypeError("Remissions should be a PyTorch tensor")

    # put in attribute
    self.points = points  # get xyz
    if remissions is not None:
        self.remissions = remissions  # get remission
    else:
        self.remissions = torch.zeros(points.size(0), dtype=torch.float32)

    # if projection is wanted, then do it and fill in the structure
    if self.project_range:
        self.do_range_projection()


  def do_range_projection(self):
    """ Project a pointcloud into a spherical projection image.
        Function takes no arguments because it can also be called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection).
    """
    # laser parameters
    fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # Get depth of all points
    depth = torch.norm(self.points, p=2, dim=1).cuda(self.rank , non_blocking=True)

    # Get scan components
    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]
    scan_z = self.points[:, 2]

    # Get angles of all points
    yaw = -torch.atan2(scan_y, scan_x)
    pitch = torch.asin(scan_z / (depth + 1e-8))

    # get projections in image coords
    proj_range_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_range_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # Scale to image size using angular resolution
    proj_range_x *= self.range_W  # in [0.0, W]
    proj_range_y *= self.range_H  # in [0.0, H]

    # Round and clamp for use as indices
    proj_range_x = torch.floor(proj_range_x).clamp(0, self.range_W - 1).to(torch.int32)  # in [0, W-1]
    proj_range_y = torch.floor(proj_range_y).clamp(0, self.range_H - 1).to(torch.int32)  # in [0, H-1]

    # Store a copy in original order
    self.proj_range_x = proj_range_x.clone()
    self.proj_range_y = proj_range_y.clone()

    # Copy depth in original order
    self.unproj_range = depth.clone()

    # Order by decreasing depth
    indices = torch.arange(depth.size(0), dtype=torch.int32).cuda(self.rank, non_blocking=True)
    order = torch.argsort(depth, descending=True)
    depth = depth[order]

    indices = indices[order]
    points = self.points[order]
    remission = self.remissions[order]
    proj_range_y = proj_range_y[order]
    proj_range_x = proj_range_x[order]

    # Assign to images
    self.proj_range[proj_range_y, proj_range_x] = depth
    self.proj_range_xyz[proj_range_y, proj_range_x] = points
    self.proj_range_remission[proj_range_y, proj_range_x] = remission
    self.proj_range_idx[proj_range_y, proj_range_x] = indices
    self.proj_range_mask = (self.proj_range_idx >= 0).to(torch.int32)  # Binary mask

class SemLaserScanCUDA(LaserScanCUDA):
  """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
  EXTENSIONS_LABEL = ['.bin', '.label']  #.bin for nuScenes, .label for SemanticKITTI

  def __init__(self, dataset=None, sem_color_dict=None, project_range=False, range_H=64, range_W=1024, fov_up=10.0, fov_down=-30.0, device = 'cuda', rank = 0):
    super(SemLaserScanCUDA, self).__init__(dataset, project_range, range_H, range_W, fov_up, fov_down, device = device, rank = rank)
    self.reset()

    # Make semantic colors
    max_sem_key = 0
    for key, data in sem_color_dict.items():
        if key + 1 > max_sem_key:
            max_sem_key = key + 1

    # Initialize semantic color LUT as a tensor
    self.sem_color_lut = torch.zeros((max_sem_key + 100, 3), dtype=torch.float32, device=self.device)

    # Populate the semantic color LUT
    for key, value in sem_color_dict.items():
        self.sem_color_lut[key] = torch.tensor(value, dtype=torch.float32) / 255.0

    # Make instance colors
    max_inst_id = 100000
    self.inst_color_lut = torch.empty((max_inst_id, 3), dtype=torch.float32, device=self.device)
    self.inst_color_lut.uniform_(0.0, 1.0)    # Random values between 0 and 1

    # Force zero to a gray-ish color
    self.inst_color_lut[0] = torch.full((3,), 0.1, dtype=torch.float32, device=self.device)


  def reset(self):
    """ Reset scan members. """
    super(SemLaserScanCUDA, self).reset()

    # Semantic labels
    self.sem_label = torch.zeros((0, 1), dtype=torch.uint32, device=self.device)         # [m, 1]: label
    self.sem_label_color = torch.zeros((0, 3), dtype=torch.float32, device=self.device)  # [m, 3]: color

    # Instance labels
    self.inst_label = torch.zeros((0, 1), dtype=torch.uint32)         # [m, 1]: label
    self.inst_label_color = torch.zeros((0, 3), dtype=torch.float32, device=self.device)  # [m, 3]: color

    # Range projection color with semantic labels
    self.range_proj_sem_label = torch.zeros((self.range_H, self.range_W),
                                            dtype=torch.int32, device=self.device)  # [H, W]: label
    self.range_proj_sem_color = torch.zeros((self.range_H, self.range_W, 3),
                                            dtype=torch.float32, device=self.device)  # [H, W, 3]: color

    # Range projection color with instance labels
    self.range_proj_inst_label = torch.zeros((self.range_H, self.range_W),
                                             dtype=torch.int32, device=self.device)  # [H, W]: label
    self.range_proj_inst_color = torch.zeros((self.range_H, self.range_W, 3),
                                             dtype=torch.float32, device=self.device)  # [H, W, 3]: color


  def open_label(self, filename):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
      raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    if self.dataset == "nuscenes":
        label = np.fromfile(filename, dtype=np.uint8)
    elif self.dataset == "semantic_kitti":
        label = np.fromfile(filename, dtype=np.uint32)
        label = label.reshape((-1))
    else:
        raise Exception("label did not open successfully.")
        
    # Convert label to tensor
    label = torch.tensor(label, dtype=torch.long)

    # set it
    self.set_label(label)

  def set_label(self, label):
    """ Set points for label not from file but from tensor
    """
    # Check label makes sense
    if not isinstance(label, torch.Tensor):
        raise TypeError("Label should be a torch tensor")

    # Only fill in attribute if the right size
    if label.shape[0] == self.points.shape[0]:
        self.sem_label = label & 0xFFFF  # Semantic label in lower half
        self.inst_label = label >> 16    # Instance id in upper half
    else:
        print("Points shape: ", self.points.shape)
        print("Label shape: ", label.shape)
        raise ValueError("Scan and Label don't contain same number of points")

    # Sanity check
    assert((self.sem_label + (self.inst_label << 16) == label).all())

    if self.project_range:
        self.do_range_label_projection()


  def colorize(self):
    """ Colorize point cloud with the color of each semantic label """
    self.sem_label_color = self.sem_color_lut[self.sem_label]
    self.sem_label_color = self.sem_label_color.view(-1, 3)  # Reshape to (-1, 3) using view for tensors

    self.inst_label_color = self.inst_color_lut[self.inst_label]
    self.inst_label_color = self.inst_label_color.view(-1, 3)  # Reshape to (-1, 3) using view for tensors

  def do_range_label_projection(self):
    # Only map colors to labels that exist
    proj_range_mask = self.proj_range_idx >= 0

    # Semantics
    self.range_proj_sem_label[proj_range_mask] = self.sem_label[self.proj_range_idx[proj_range_mask]]
    self.range_proj_sem_color[proj_range_mask] = self.sem_color_lut[self.sem_label[self.proj_range_idx[proj_range_mask]]]

    # Instances
    self.range_proj_inst_label[proj_range_mask] = self.inst_label[self.proj_range_idx[proj_range_mask]]
    self.range_proj_inst_color[proj_range_mask] = self.inst_color_lut[self.inst_label[self.proj_range_idx[proj_range_mask]]]
