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
import torch
import torch.nn as nn
import random
import warnings
import argparse
import numpy as np
import utils.transformations.transforms as tr
from utils.metrics.semanticsegmentation import SemSegLoss
from trainer.scheduler import WarmupCosine
from trainer.manager import Manager
from core.network import Network
from datasets import LIST_DATASETS, Collate_fn

from utils.loss.lovasz import Lovasz_softmax
from utils.loss.boundary_loss import BoundaryLoss
from torch.nn import CrossEntropyLoss

def load_configs(mainfile, netfile):
    with open(mainfile, "r") as mf:
        mainconfig = yaml.safe_load(mf)

    with open(netfile, "r") as nf:
        netconfig = yaml.safe_load(nf)

    return mainconfig, netconfig


def get_train_augmentations(args, mainconfig, netconfig):

    list_of_transf = []

    # Optional augmentations
    for aug_name in netconfig["augmentations"].keys():
        if aug_name == "pointsample":
            list_of_transf.append(tr.PointSample(inplace=True, num_points = netconfig["augmentations"]["pointsample"]))
        elif aug_name == "randomflip":
            list_of_transf.append(tr.RandomFlip3D(inplace=True, 
                                                    sync_2d = netconfig["augmentations"]["randomflip"]["sync_2d"],
                                                    flip_ratio_bev_horizontal = netconfig["augmentations"]["randomflip"]["flip_ratio_bev_horizontal"],
                                                    flip_ratio_bev_vertical = netconfig["augmentations"]["randomflip"]["flip_ratio_bev_vertical"]))
        elif aug_name == "GlobalRotScaleTrans":
            rot_range = netconfig["augmentations"]["GlobalRotScaleTrans"]["rot_range"]
            scale_ratio_range = netconfig["augmentations"]["GlobalRotScaleTrans"]["scale_ratio_range"]
            translation_std = netconfig["augmentations"]["GlobalRotScaleTrans"]["translation_std"]
            list_of_transf.append(tr.GlobalRotScaleTrans(inplace=True, rot_range=rot_range, scale_ratio_range=scale_ratio_range, translation_std=translation_std))
        else:
            raise ValueError("Unknown transformation")

        print("List of transformations:", list_of_transf)

        return tr.Compose(list_of_transf)



def get_datasets(netconfig, args):
    kwargs = {
        "dataset": args.dataset,
        "rootdir": args.path_dataset,
        "input_feat": netconfig["input_feat"],
        "range_H": netconfig["range_proj"]["range_H"],
        "range_W": netconfig["range_proj"]["range_W"],
        "fov_up": netconfig["range_proj"]["fov_up"],
        "fov_down": netconfig["range_proj"]["fov_down"],
        "batch_size": mainconfig["dataloader"]["batch_size"],
        "preproc_gpu": netconfig["preproc"]["gpu"],
        "rank": args.gpu
    }

    # Get datatset
    DATASET = LIST_DATASETS.get(args.dataset.lower())
    if DATASET is None:
        raise ValueError(f"Dataset {args.dataset.lower()} not available.")
    
    # Train dataset
    train_dataset = DATASET(
        phase="trainval" if args.trainval else "train",
        train_augmentations=get_train_augmentations(args, mainconfig, netconfig),
        instance_cutmix=mainconfig["augmentations"]["instance_cutmix"],
        **kwargs,
    )

    # Validation dataset
    val_dataset = DATASET(
        phase="val",
        **kwargs,
    )

    return train_dataset, val_dataset


def get_dataloader(train_dataset, val_dataset, args, mainconfig):

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    if Collate_fn is not None:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            collate_fn=Collate_fn(),
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=val_sampler,
            drop_last=False,
            collate_fn=Collate_fn())
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True
        )

        val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False
        )

    return train_loader, val_loader, train_sampler


def get_optimizer(parameters, mainconfig):
    return torch.optim.AdamW(
        parameters,
        lr=mainconfig["optim"]["lr"],
        weight_decay=mainconfig["optim"]["weight_decay"],
        betas=mainconfig["optim"]["betas"],
        eps = mainconfig["optim"]["eps"],
    )


def get_scheduler(optimizer, mainconfig, len_train_loader):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        WarmupCosine(
            mainconfig["scheduler"]["epoch_warmup"] * len_train_loader,
            mainconfig["scheduler"]["max_epoch"] * len_train_loader,
            mainconfig["scheduler"]["min_lr"] / mainconfig["optim"]["lr"],
        ),
    )
    return scheduler


def distributed_training(gpu, ngpus_per_node, args, mainconfig, netconfig):

    # --- Init. distributing training
    args.gpu = gpu
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    net = Network(args.net, netconfig)
    model = net.build_network()
    
    # ---
    args.batch_size = mainconfig["dataloader"]["batch_size"]
    args.workers = mainconfig["dataloader"]["num_workers"]
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        args.batch_size = int(mainconfig["dataloader"]["batch_size"] / ngpus_per_node)
        args.workers = int(
            (mainconfig["dataloader"]["num_workers"] + ngpus_per_node - 1) / ngpus_per_node
        )
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if args.gpu == 0 or args.gpu is not None:
        print(f"Model:\n{model}")
        nb_param = sum([p.numel() for p in model.parameters()]) / 1e6
        print(f"{nb_param} x 10^6 trainable parameters ")

    # --- Optimizer
    optim = get_optimizer(model.parameters(), mainconfig)

    # --- Dataset
    train_dataset, val_dataset = get_datasets(netconfig, args)
    train_loader, val_loader, train_sampler = get_dataloader(
        train_dataset, val_dataset, args, mainconfig
    )

    # --- Loss function
    lovasz = Lovasz_softmax(ignore=netconfig["classif"]["ignore_class"]).cuda(args.gpu)
    bd = BoundaryLoss(ignore_index=netconfig["classif"]["ignore_class"]).cuda(args.gpu)
    ce = CrossEntropyLoss(ignore_index=netconfig["classif"]["ignore_class"]).cuda(args.gpu)
    loss = {
        "lovasz": lovasz,
        "bd": bd,
        "ce": ce
    }

    if(args.eval is False):
        scheduler = get_scheduler(optim, mainconfig, len(train_loader))
    else:
        scheduler = None

    # --- Training
    mng = Manager(
        model,
        loss,
        train_loader,
        val_loader,
        train_sampler,
        optim,
        scheduler,
        mainconfig["scheduler"]["max_epoch"],
        args.log_path,
        args.gpu,
        args.world_size,
        args.fp16,
        args.net,
        LIST_DATASETS.get(args.dataset.lower()).CLASS_NAME,
        tensorboard=(not args.eval),
        checkpoint= args.checkpoint,
        netconfig=netconfig,
        preproc_gpu=netconfig["preproc"]["gpu"],
        perf=args.perf,

    )
    if args.restart:
        mng.load_state(best=True) #True
    if args.eval:
        mng.one_epoch(training=False)
    else:
        mng.train()


def main(args, mainconfig, netconfig):

    args.device = "cuda"
    args.rank = 0
    args.world_size = 1
    args.dist_url = "to-specify"
    args.dist_backend = "nccl"
    args.distributed = args.multiprocessing_distributed

    os.makedirs(args.log_path, exist_ok=True)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)

    if args.gpu is not None:
        args.distributed = False
        args.multiprocessing_distributed = False
        warnings.warn(
            "You chose a specific GPU. Data parallelism is disabled."
        )

    # Extract instances for cutmix
    if mainconfig["augmentations"]["instance_cutmix"]:
        get_datasets(netconfig, args)

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        torch.multiprocessing.spawn(
            distributed_training,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, mainconfig, netconfig),
        )
    else:
        distributed_training(args.gpu, ngpus_per_node, args, mainconfig, netconfig)


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--net",
        type=str,
        help="Network name (harpnext)",
        default="harpnext"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of dataset",
        default="semantic_kitti",
    )
    parser.add_argument(
        "--path_dataset",
        type=str,
        help="Path to dataset",
        default="/path/to/dataset",
    )
    parser.add_argument(
        "--log_path", type=str, required=True, default="./logs/harpnext-experiment", help="Path to log folder"
    )
    parser.add_argument(
        "-r", "--restart", action="store_true", default=False, help="Restart training"
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="Seed for initializing training"
    )
    parser.add_argument(
        "--gpu", default=None, type=int, help="Set to any number to use gpu 0"
    )
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enable autocast for mix precision training",
    )
    parser.add_argument(
        "--mainconfig",
        type=str, 
        required=True, 
        help="Path to main config"
    )
    parser.add_argument(
        "--netconfig",
        type=str, 
        required=True, 
        help="Path to specific network model config"
    )
    parser.add_argument(
        "--trainval",
        action="store_true",
        default=False,
        help="Use train + val as train set",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Run validation only",
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        default=False,
        help="To run in Performance Mode, ensure a batch size of 1",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False, default="./logs/harpnext-experiment/", help="Path to checkpoint directory"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    mainconfig, netconfig = load_configs(args.mainconfig, args.netconfig)
    main(args, mainconfig, netconfig)
