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


import torch
import warnings
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from utils.metrics.semanticsegmentation import overall_accuracy, fast_hist, per_class_iu, per_class_accuracy

import os

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import datetime

class Manager:
    def __init__(
        self,
        net,
        loss,
        loader_train,
        loader_val,
        train_sampler, 
        optim,
        scheduler,
        max_epoch,
        path,
        rank,
        world_size,
        fp16=False,
        network_name="harpnext",
        class_names=None,
        tensorboard=True,
        checkpoint = None,
        netconfig = None,
        preproc_gpu = False,
        perf = False
    ):

        # Optim. methods
        self.optim = optim
        self.fp16 = fp16
        self.scaler = GradScaler() if fp16 else None
        self.scheduler = scheduler

        # Dataloaders
        self.max_epoch = max_epoch
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.train_sampler = train_sampler
        self.class_names = class_names

        # Network
        self.network_name = network_name
        self.net = net
        self.rank = rank
        self.world_size = world_size
        print(f"Trainer on gpu: {self.rank}. World size:{self.world_size}.")
        self.added_rank = rank

        # Loss
        self.loss = loss

        # Checkpoints
        self.best_miou = 0
        self.current_epoch = 0
        self.path_to_ckpt = path

        #For logging
        self.datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.datalog_path = self.path_to_ckpt + f"/datalog/{self.datetime_str}.txt"
        os.makedirs(os.path.dirname(self.datalog_path), exist_ok=True)
        # self.bestmodel_path = self.path_to_ckpt + f"/datalog/best_model.txt"
        # os.makedirs(os.path.dirname(self.bestmodel_path), exist_ok=True)

        #to load checkpoint
        self.checkpoint = checkpoint

        self.netconfig = netconfig
        
        # Monitoring
        if tensorboard and (self.rank == 0 or self.rank is None or self.rank == self.added_rank):
            self.writer_train = SummaryWriter(
                path + "/tensorboard/train/",
                purge_step=self.current_epoch * len(self.loader_train),
                flush_secs=30,
            )
            self.writer_val = SummaryWriter(
                path + "/tensorboard/val/",
                purge_step=self.current_epoch,
                flush_secs=30,
            )
        else:
            self.writer_val = None
            self.writer_train = None

        # Flag for pre_process on GPU:
        self.preproc_gpu = preproc_gpu 

    def print_log(self, running_loss, oAcc, mAcc, mIoU, ious, training):
        if self.rank == 0 or self.rank is None or self.rank == self.added_rank:
            # Global score
            log = (
                f"\nEpoch: {self.current_epoch:d} :\n"
                + f" Loss = {running_loss:.3f}"
                + f" - oAcc = {oAcc:.1f}"
                + f" - mAcc = {mAcc:.1f}"
                + f" - mIoU = {mIoU:.1f}"
            )
            print(log)
            # Log to file only during training
            with open(self.datalog_path, 'a') as f:
                f.write(log + '\n')

            # Per class score
            log = ""
            for i, s in enumerate(ious):
                if self.class_names is None:
                    log += f"Class {i}: {100 * s:.1f} - "
                else:
                    log += f"{self.class_names[i]}: {100 * s:.1f} - "
            print(log[:-3])

            # Log to file only during training
            with open(self.datalog_path, 'a') as f:
                f.write(log[:-3] + '\n')

            # Recall best mIoU
            print(f"Best mIoU was {self.best_miou:.1f}.")

            # Log to file only during training
            with open(self.datalog_path, 'a') as f:
                f.write(f"Best mIoU was {self.best_miou:.1f}." + '\n')

    def gather_scores(self, list_tensors):
        if self.rank == 0 or self.rank == self.added_rank:
            tensor_reduced = [
                [torch.empty_like(t) for _ in range(self.world_size)]
                for t in list_tensors
            ]
            for t, t_reduced in zip(list_tensors, tensor_reduced):
                torch.distributed.gather(t, t_reduced)
            tensor_reduced = [sum(t).cpu() for t in tensor_reduced]
            return tensor_reduced
        else:
            for t in list_tensors:
                torch.distributed.gather(t)

    def one_epoch(self, training=True):
        # Train or eval mode
        if training:
            net = self.net.train()
            loader = self.loader_train
            if self.rank == 0 or self.rank is None or self.rank == self.added_rank:
                print("\nTraining: %d/%d epochs" % (self.current_epoch, self.max_epoch))
                # Log to file
                with open(self.datalog_path, 'a') as f:
                    f.write("\nTraining: %d/%d epochs" % (self.current_epoch, self.max_epoch) + '\n')
            writer = self.writer_train
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self.current_epoch)
        else:
            net = self.net.eval()
            loader = self.loader_val
            if self.rank == 0 or self.rank is None or self.rank == self.added_rank:
                print(
                    "\nValidation: %d/%d epochs" % (self.current_epoch, self.max_epoch)
                )
            # Log to file in validation only when not evaluating performances
                with open(self.datalog_path, 'a') as f:
                    f.write("\nValidation: %d/%d epochs" % (self.current_epoch, self.max_epoch) + '\n')
            writer = self.writer_val
        print_freq = np.max((len(loader) // 10, 1))

        # Stat.
        running_loss = 0.0
        confusion_matrix = 0

        # Loop over mini-batches
        if self.rank == 0 or self.rank is None or self.rank == self.added_rank:
            bar_format = "{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}"
            loader = tqdm(loader, bar_format=bar_format)
        
        enumerator = enumerate(loader)
        dataset = loader.iterable.dataset
            

        for it, batch in enumerator:
            if self.preproc_gpu:
                pc, labels = dataset.load_batch_to_gpu(it)
                batch, pc = dataset.process_batch_gpu(pc, labels)

            if not training and not self.preproc_gpu:  
                batch, pc = dataset.process_batch_cpu(it) 

            # Network inputs
            net_inputs = self.get_network_inputs(batch)
            # Labels
            labels = self.get_labels(batch)

            # Get prediction and loss
            with torch.autocast("cuda", enabled=self.fp16):
                # Logits
                if training:
                    if self.network_name == "harpnext":
                        out = net(net_inputs, training=training)
                    else:
                        raise Exception("not implemented")
                else:
                    with torch.no_grad():
                        if self.network_name == "harpnext":
                            out = net(net_inputs, training=training)
                        else:
                            raise Exception("Not implemented")

                out, labels = self.upsample(net_inputs, batch, out, labels)

                out_losses = out["losses_seg_logits"]
                if training:
                    lamda = self.netconfig["train"]["lamda"]
                    loss_points = self.loss["ce"](out_losses["HARPNeXtHead.seg_logit"], labels["pt_labels"])
                    loss_aux_0 = self.loss["ce"](out_losses["AuxHead_0.seg_logit"], labels["proj_labels"]) + 1.5 * self.loss["lovasz"](out_losses["AuxHead_0.seg_logit"], labels["proj_labels"]) + self.loss["bd"](out_losses["AuxHead_0.seg_logit"], labels["proj_labels"])
                    loss_aux_1 = self.loss["ce"](out_losses["AuxHead_1.seg_logit"], labels["proj_labels"]) + 1.5 * self.loss["lovasz"](out_losses["AuxHead_1.seg_logit"], labels["proj_labels"]) + self.loss["bd"](out_losses["AuxHead_1.seg_logit"], labels["proj_labels"])
                    loss_aux_2 = self.loss["ce"](out_losses["AuxHead_2.seg_logit"], labels["proj_labels"]) + 1.5 * self.loss["lovasz"](out_losses["AuxHead_2.seg_logit"], labels["proj_labels"]) + self.loss["bd"](out_losses["AuxHead_2.seg_logit"], labels["proj_labels"])
                    loss_aux_3 = self.loss["ce"](out_losses["AuxHead_3.seg_logit"], labels["proj_labels"]) + 1.5 * self.loss["lovasz"](out_losses["AuxHead_3.seg_logit"], labels["proj_labels"]) + self.loss["bd"](out_losses["AuxHead_3.seg_logit"], labels["proj_labels"])
                    loss = loss_points + lamda*loss_aux_0 + lamda*loss_aux_1 + lamda*loss_aux_2 + lamda*loss_aux_3
                else:
                    loss = self.loss["ce"](out_losses["HARPNeXtHead.seg_logit"], labels["pt_labels"])

                out  = out["seg_logits"]
                labels = labels["pt_labels"]

            running_loss += loss.detach()

            # # Confusion matrix
            confusion_matrix =  self.get_predictions(confusion_matrix, labels, out, batch, net_inputs)

            # Logs
            if it % print_freq == print_freq - 1 or it == len(loader) - 1:
                # Gather scores
                if self.train_sampler is not None:
                    out = self.gather_scores([running_loss, confusion_matrix])
                else:
                    out = [running_loss.cpu(), confusion_matrix.cpu()]
                if self.rank == 0 or self.rank is None or self.rank == self.added_rank:
                    # Compute scores
                    oAcc = 100 * overall_accuracy(out[1])
                    mAcc = 100 * np.nanmean(per_class_accuracy(out[1]))
                    ious = per_class_iu(out[1])
                    mIoU = 100 * np.nanmean(ious)
                    running_loss_reduced = out[0].item() / self.world_size / (it + 1)
                    # Print score
                    self.print_log(running_loss_reduced, oAcc, mAcc, mIoU, ious, training)
                    # Save in tensorboard
                    if (writer is not None) and (training or it == len(loader) - 1):
                        header = "Train" if training else "Test"
                        step = (
                            self.current_epoch * len(loader) + it
                            if training
                            else self.current_epoch
                        )
                        writer.add_scalar(header + "/loss", running_loss_reduced, step)
                        writer.add_scalar(header + "/oAcc", oAcc, step)
                        writer.add_scalar(header + "/mAcc", mAcc, step)
                        writer.add_scalar(header + "/mIoU", mIoU, step)
                        writer.add_scalar(
                            header + "/lr", self.optim.param_groups[0]["lr"], step
                        )

            # Gradient step
            if training:
                self.optim.zero_grad(set_to_none=True)
                if self.fp16:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optim.step()
                if self.scheduler is not None:
                    self.scheduler.step()

        # Return score
        if self.rank == 0 or self.rank is None or self.rank == self.added_rank:
            return mIoU
        else:
            return None        
        
    def get_network_inputs(self, batch):
        if self.preproc_gpu:
            net_inputs = dict()
            net_inputs['points'] = batch["points"] 

            voxel_dict = dict()
            voxel_dict['voxels'] = batch['voxels']  
            voxel_dict['coors'] = batch['coors']    

            net_inputs['voxels'] = voxel_dict

            return net_inputs
        
        else:
            net_inputs = dict()
            net_inputs['points'] = [
                pt.cuda(self.rank, non_blocking=True) for pt in batch["points"]
                ]

            voxel_dict = dict()
            voxel_dict['voxels'] = batch['voxels'].cuda(self.rank, non_blocking=True)
            voxel_dict['coors'] = batch['coors'].cuda(self.rank, non_blocking=True)

            net_inputs['voxels'] = voxel_dict

            return net_inputs
    
    #returns the original labels per point, at original resolution
    def get_labels(self, batch):
        if self.preproc_gpu:
            proj_range_sem_label = batch["proj_labels"].long().cuda(self.rank, non_blocking=True)
            pt_sem_label = batch["pt_labels"].long().cuda(self.rank, non_blocking=True)
        else:

            proj_range_sem_label = torch.stack(batch["proj_labels"], dim=0).long().cuda(self.rank, non_blocking=True)
            pt_sem_label = torch.cat(batch["pt_labels"], dim=0).long().cuda(self.rank, non_blocking=True)

        labels = {
            'proj_labels': proj_range_sem_label,
            'pt_labels': pt_sem_label
        }
        return labels

    # Upsample to orginal pointcloud resolution, and return out as predictions per class for every point
    def upsample(self, net_inputs, batch, out, labels):
        return out, labels
        
        
    def get_predictions(self, confusion_matrix, labels, out, batch, net_inputs):
        # Confusion matrix
        with torch.no_grad():
            nb_class = out.shape[1]
            pred_label = out.max(1)[1]
            where = labels != self.netconfig["classif"]["ignore_class"]
            confusion_matrix += fast_hist(
                pred_label[where], labels[where], nb_class
            )
        return confusion_matrix   

    def load_state(self, best=False):
        filename = self.path_to_ckpt
        filename += "/ckpt_best.pth" if best else "/ckpt_last.pth"
        rank = 0 if self.rank is None else self.rank
        ckpt = torch.load(
            filename,
            map_location=f"cuda:{rank}",
        )
##############################################################################################################################################################################################################
        state_dict = ckpt["net"]
        try:
            self.net.load_state_dict(state_dict)
        except:
            # If model was trained using DataParallel or DistributedDataParallel
            state_dict = {}
            for key in ckpt["net"].keys():
                state_dict[key[len("module."):]] = ckpt["net"][key]
            self.net.load_state_dict(state_dict) 
##############################################################################################################################################################################################################
        # self.net.load_state_dict(state_dict)        

        if ckpt.get("optim") is None:
            warnings.warn("Optimizer state not available")
        else:
            self.optim.load_state_dict(ckpt["optim"])
        if self.scheduler is not None:
            if ckpt.get("scheduler") is None:
                warnings.warn("Scheduler state not available")
            else:
                self.scheduler.load_state_dict(ckpt["scheduler"])
        if self.fp16:
            if ckpt.get("scaler") is None:
                warnings.warn("Scaler state not available")
            else:
                self.scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("best_miou") is not None:
            self.best_miou = ckpt["best_miou"]
        if ckpt.get("epoch") is not None:
            self.current_epoch = ckpt["epoch"] + 1
        print(
            f"Checkpoint loaded on {torch.device(rank)} (cuda:{rank}): {self.path_to_ckpt}"
        )

    def save_state(self, best=False):
        if self.rank == 0 or self.rank is None or self.rank == self.added_rank:
            dict_to_save = {
                "epoch": self.current_epoch,
                "net": self.net.state_dict(),
                "optim": self.optim.state_dict(),
                "scheduler": self.scheduler.state_dict()
                if self.scheduler is not None
                else None,
                "scaler": self.scaler.state_dict() if self.fp16 else None,
                "best_miou": self.best_miou,
            }
            filename = self.path_to_ckpt
            filename += "/ckpt_best.pth" if best else "/ckpt_last.pth"
            torch.save(dict_to_save, filename)

    def train(self):
        for _ in range(self.current_epoch, self.max_epoch):
            # Train
            self.one_epoch(training=True)
            # Val
            miou = self.one_epoch(training=False)
            # Save best checkpoint
            if miou is not None and miou > self.best_miou:
                self.best_miou = miou
                self.save_state(best=True)
                print(f"\n\n*** New best mIoU: {self.best_miou:.1f}.\n")
            # Save last checkpoint
            self.save_state()
            # Increase epoch number
            self.current_epoch += 1
        if self.rank == 0 or self.rank is None or self.rank == self.added_rank:
            print("Finished Training")
