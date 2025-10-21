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


from .semantickitti.processor import getSemanticKITTIProcessor
from .nuscenes.processor import getnuScenesProcessor

##############################################################################################################################
net = "harpnext"
##############################################################################################################################

NetPCProcessor = None
NetCollate_fn = None

if net == "harpnext":
    from datasets.pc_processors.harpnext_pcprocessor import HARPNeXtPCProcessor
    from .pc_processors.harpnext_pcprocessor import Collate

    NetPCProcessor = HARPNeXtPCProcessor
    NetCollate_fn = Collate


SemanticKITTIProcessor = getSemanticKITTIProcessor(NetPCProcessor)
nuScenesProcessor = getnuScenesProcessor(NetPCProcessor)
Collate_fn = NetCollate_fn 

__all__ = [SemanticKITTIProcessor, nuScenesProcessor, Collate_fn]
LIST_DATASETS = {"nuscenes": nuScenesProcessor, "semantic_kitti": SemanticKITTIProcessor}