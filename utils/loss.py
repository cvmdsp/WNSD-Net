# Ultralytics YOLO ??, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from .metrics import bbox_iou
from .tal import bbox2dist
from .new_bbox_iou import new_bbox_iou

class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367."""

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).mean(1).sum()
        return loss


# Losses
class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, ):
        super().__init__()

    def forward(self, pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


# class BboxLoss(nn.Module):
#     """Criterion class for computing training losses during training."""
#
#     def __init__(self, reg_max=16, imgsz=640, iou_type='Ciou', Inner_iou=False, Focal=False, Focaler=False, epoch=300,
#                  alpha=1):
#         """Initialize the BboxLoss module with regularization maximum and DFL settings."""
#         super().__init__()
#         self.use_dfl = True
#         self.iou_type = iou_type  # +++
#         self.Inner_iou = Inner_iou  # Inner-IoU
#         self.Focal = Focal  # Focal-IoU
#         self.imgsz = imgsz  # MPDIoU
#         self.Focaler = Focaler  # Focaler-IoU
#         self.epoch = epoch  # Unified-IoU
#         self.alpha = alpha  # AlphaIoU
#         self.reg_max = reg_max
#
#     def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
#         """IoU loss."""
#         weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
#
#         if self.iou_type == "iou":
#             iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, Inner_iou=self.Inner_iou,
#                                Focal=self.Focal, alpha=self.alpha)
#
#         elif self.iou_type == "Giou":
#             iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, GIoU=True,
#                                Inner_iou=self.Inner_iou, Focal=self.Focal, alpha=self.alpha)
#
#         elif self.iou_type == "Diou":
#             iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, DIoU=True,
#                                Inner_iou=self.Inner_iou, Focal=self.Focal, alpha=self.alpha)
#
#         elif self.iou_type == "Siou":
#             iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, SIoU=True,
#                                Inner_iou=self.Inner_iou, Focal=self.Focal, alpha=self.alpha)
#
#         elif self.iou_type == "Eiou":
#             iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, EIoU=True,
#                                Inner_iou=self.Inner_iou, Focal=self.Focal, alpha=self.alpha)
#
#         elif self.iou_type == "Wise-iou":
#             iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, WIoU=True,
#                                Inner_iou=self.Inner_iou, scale=True)
#
#         elif self.iou_type == "MPDiou":
#             # 仅针对正方形image输入
#             iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, MPDIoU=True,
#                                Inner_iou=self.Inner_iou, feat_w=self.imgsz, feat_h=self.imgsz)
#
#         elif self.iou_type == "Shape-iou":
#             iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, ShapeIou=True,
#                                Inner_iou=self.Inner_iou, ShapeIou_scale=0)
#
#         elif self.iou_type == "Powerful-iou":
#             iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, PIouV1=True, PIouV2=False,
#                                PIou_Lambda=1.3)
#
#         elif self.iou_type == "Unified-iou":
#             iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=True, UIoU=True, epoch=self.epoch)
#
#         else:
#             # 默认Ciou
#             iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True,
#                                Inner_iou=self.Inner_iou, Focal=self.Focal, alpha=self.alpha)
#
#         if type(iou) is tuple:
#             if len(iou) == 2:
#                 loss_iou = ((1.0 - iou[0]) * iou[1].detach() * weight).sum() / target_scores_sum
#             else:
#                 loss_iou = (iou[0] * iou[1] * weight).sum() / target_scores_sum
#
#         elif self.iou_type == "Powerful-iou":
#             # 已在new_bbox_iou求得1-iou
#             loss_iou = (iou * weight).sum() / target_scores_sum
#
#         else:
#             if self.Focaler:
#                 # 引入 Focaler-IoU 回归样本 https://arxiv.org/abs/2401.10525
#                 # default d=0.00,u=0.95
#                 d = 0.00
#                 u = 0.95
#                 iou = ((iou - d) / (u - d)).clamp(0, 1)
#
#             loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
#
#
#         if self.use_dfl:
#             target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
#             loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
#             loss_dfl = loss_dfl.sum() / target_scores_sum
#         else:
#             loss_dfl = torch.tensor(0.0).to(pred_dist.device)
#
#         return loss_iou, loss_dfl
#     @staticmethod
#     def _df_loss(pred_dist, target):
#         """Return sum of left and right DFL losses."""
#         # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
#         tl = target.long()  # target left
#         tr = tl + 1  # target right
#         wl = tr - target  # weight left
#         wr = 1 - wl  # weight right
#         return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
#                 F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class KeypointLoss(nn.Module):

    def __init__(self, sigmas) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / (torch.sum(kpt_mask != 0) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return kpt_loss_factor * ((1 - torch.exp(-e)) * kpt_mask).mean()


# Criterion class for computing Detection training losses

import random

import torch
import random
import numpy as np

import torch
import random


class BatchBboxNoiseAugmentation:
    """批量边界框缩放增强（保持中心点不变，只缩放框的大小）"""

    def __init__(self,
                 scale_range=(0.2, 1.0),  # 缩放范围（建议收缩范围稍小，扩大范围稍大）
                 add_ratio=0.5,  # 添加新目标的比例
                 max_new_per_batch=50,  # 每个batch最多添加的新目标数
                 image_size=640,  # 图像尺寸
                 iou_threshold=0.8):  # IoU阈值，避免添加过于重叠的框
        """
        Args:
            scale_range: 缩放倍数范围（<1为缩小，>1为放大）
            add_ratio: 添加新目标的比例（如0.8表示增加80%的新目标）
            max_new_per_batch: 每个batch最多添加的新目标数
            image_size: 图像尺寸
            iou_threshold: IoU阈值，用于避免生成过于重叠的框
        """
        self.scale_range = scale_range
        self.add_ratio = add_ratio
        self.max_new_per_batch = max_new_per_batch
        self.image_size = image_size
        self.iou_threshold = iou_threshold

    def calculate_iou(self, box1, box2):
        """计算两个框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def scale_box_keep_center(self, box, scale_factor):
        """
        保持中心点不变，缩放边界框

        Args:
            box: [x1, y1, x2, y2]
            scale_factor: 缩放因子
        Returns:
            scaled_box: 缩放后的边界框
        """
        x1, y1, x2, y2 = box

        # 计算中心点
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 计算当前宽高
        width = x2 - x1
        height = y2 - y1

        # 应用缩放
        new_width = width * scale_factor
        new_height = height * scale_factor

        # 计算新坐标（保持中心不变）
        new_x1 = center_x - new_width / 2
        new_y1 = center_y - new_height / 2
        new_x2 = center_x + new_width / 2
        new_y2 = center_y + new_height / 2

        # 确保坐标在图像范围内
        new_x1 = max(0, min(new_x1, self.image_size - 1))
        new_y1 = max(0, min(new_y1, self.image_size - 1))
        new_x2 = max(0, min(new_x2, self.image_size - 1))
        new_y2 = max(0, min(new_y2, self.image_size - 1))

        # 确保框有有效尺寸
        if new_x2 - new_x1 < 2:  # 最小宽度为2像素
            new_x2 = new_x1 + 2
        if new_y2 - new_y1 < 2:  # 最小高度为2像素
            new_y2 = new_y1 + 2

        return [new_x1, new_y1, new_x2, new_y2]

    def __call__(self, targets):
        """
        对批量目标应用缩放增强，增加新目标

        Args:
            targets: tensor of shape [B, N, 5]  (class_id, x1, y1, x2, y2)

        Returns:
            augmented_targets: tensor of shape [B, N, 5]
        """
        if targets is None or len(targets) == 0:
            return targets

        batch_size, max_objects, _ = targets.shape
        device = targets.device

        # 创建输出张量，初始化为全0（padding）
        augmented_targets = torch.zeros_like(targets, device=device)

        for b in range(batch_size):
            # 找到该batch中的真实目标
            non_zero_mask = (targets[b, :, 1] != 0) | (targets[b, :, 2] != 0) | \
                            (targets[b, :, 3] != 0) | (targets[b, :, 4] != 0)
            valid_indices = torch.where(non_zero_mask)[0]

            if len(valid_indices) == 0:
                # 如果没有目标，直接跳过
                continue

            # 获取原始目标
            original_count = len(valid_indices)
            original_targets = targets[b, valid_indices].clone()  # [M, 5]

            # 计算要添加的新目标数量
            num_to_add = int(original_count * self.add_ratio)
            num_to_add = min(num_to_add, self.max_new_per_batch)
            num_to_add = min(num_to_add, max_objects - original_count)

            # 复制原始目标
            augmented_targets[b, :original_count] = original_targets

            if num_to_add <= 0:
                continue

            # 生成新目标
            new_targets_list = []
            attempts = 0
            max_attempts = num_to_add * 10  # 防止无限循环

            while len(new_targets_list) < num_to_add and attempts < max_attempts:
                attempts += 1

                # 随机选择一个原始目标作为模板
                source_idx = random.randint(0, original_count - 1)
                original_target = original_targets[source_idx]

                # 获取类别和坐标
                cls_id = original_target[0]
                box = original_target[1:5].cpu().numpy()

                # 随机缩放因子（保持中心不变）
                scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])

                # 缩放框
                new_box = self.scale_box_keep_center(box, scale_factor)

                # 检查与所有已有目标（包括原始和已添加的新目标）的IoU
                valid_new_box = True

                # 检查与原始目标的IoU
                for j in range(original_count):
                    orig_box = original_targets[j, 1:5].cpu().numpy()
                    iou = self.calculate_iou(new_box, orig_box)
                    if iou > self.iou_threshold:
                        valid_new_box = False
                        break

                # 检查与已添加新目标的IoU
                if valid_new_box:
                    for new_target in new_targets_list:
                        existing_box = new_target[0, 1:5].cpu().numpy()
                        iou = self.calculate_iou(new_box, existing_box)
                        if iou > self.iou_threshold:
                            valid_new_box = False
                            break

                if valid_new_box:
                    # 创建新目标
                    new_target = torch.tensor(
                        [[cls_id, new_box[0], new_box[1], new_box[2], new_box[3]]],
                        dtype=targets.dtype,
                        device=device
                    )
                    new_targets_list.append(new_target)

            # 添加新目标
            if new_targets_list:
                new_targets_tensor = torch.cat(new_targets_list, dim=0)
                final_count = min(original_count + len(new_targets_tensor), max_objects)
                augmented_targets[b, original_count:final_count] = \
                    new_targets_tensor[:final_count - original_count]

        return augmented_targets



class v8DetectionLoss:

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        # self.bbox_loss = BboxLoss(m.reg_max - 1, self.hyp.imgsz, self.hyp.iou_type, self.hyp.Inner_iou, self.hyp.Focal,
        #                           self.hyp.Focaler, self.hyp.epochs, self.hyp.alpha).to(device)

        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        # if self.hyp.mode == "train":
        #     targets = BatchBboxNoiseAugmentation()(targets)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


# Criterion class for computing training losses
class v8SegmentationLoss(v8DetectionLoss):

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.nm = model.model[-1].nm  # number of masks
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR ? segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # bbox loss
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)
            # masks loss
            masks = batch['masks'].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            for i in range(batch_size):
                if fg_mask[i].sum():
                    mask_idx = target_gt_idx[i][fg_mask[i]]
                    if self.overlap:
                        gt_mask = torch.where(masks[[i]] == (mask_idx + 1).view(-1, 1, 1), 1.0, 0.0)
                    else:
                        gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                    xyxyn = target_bboxes[i][fg_mask[i]] / imgsz[[1, 0, 1, 0]]
                    marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)
                    mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                    loss[1] += self.single_mask_loss(gt_mask, pred_masks[i][fg_mask[i]], proto[i], mxyxy, marea)  # seg

                # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
                else:
                    loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box / batch_size  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        """Mask loss for one image."""
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n, 32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()


# Criterion class for computing training losses
class v8PoseLoss(v8DetectionLoss):

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            for i in range(batch_size):
                if fg_mask[i].sum():
                    idx = target_gt_idx[i][fg_mask[i]]
                    gt_kpt = keypoints[batch_idx.view(-1) == i][idx]  # (n, 51)
                    gt_kpt[..., 0] /= stride_tensor[fg_mask[i]]
                    gt_kpt[..., 1] /= stride_tensor[fg_mask[i]]
                    area = xyxy2xywh(target_bboxes[i][fg_mask[i]])[:, 2:].prod(1, keepdim=True)
                    pred_kpt = pred_kpts[i][fg_mask[i]]
                    kpt_mask = gt_kpt[..., 2] != 0
                    loss[1] += self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss
                    # kpt_score loss
                    if pred_kpt.shape[-1] == 3:
                        loss[2] += self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose / batch_size  # pose gain
        loss[2] *= self.hyp.kobj / batch_size  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def kpts_decode(self, anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y


class v8ClassificationLoss:

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch['cls'], reduction='sum') / 64
        loss_items = loss.detach()
        return loss, loss_items
