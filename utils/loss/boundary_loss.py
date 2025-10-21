# This file is from CENet
#https://github.com/huixiancheng/CENet

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    device = label.device
    one_hot_label = torch.eye(
        n_classes, device=device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label

class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5, ignore_index=None):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta
        self.ignore_index = ignore_index

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, W)
        Return:
            - boundary loss, averaged over mini-batch
        """

        n, c, h, w = pred.shape

        # Create mask for ignore_index
        if self.ignore_index is not None:
            mask = (gt != self.ignore_index)  # shape [N, H, W]
            gt = gt * mask  # Apply mask to ground truth
            mask = mask.unsqueeze(1).expand(-1, c, -1, -1)  # Expand mask to shape [N, C, H, W]
            pred = pred * mask  # Apply mask to predictions

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss
