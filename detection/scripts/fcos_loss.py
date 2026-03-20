#!/usr/bin/env python3
"""
FCOS Loss Functions.

FCOS uses three losses:
1. Focal Loss for classification (handles class imbalance)
2. GIoU Loss for box regression (better gradient than L1/L2)
3. BCE Loss for centerness (quality estimation)

Total Loss = L_cls + L_reg + L_ctr

Only positive samples contribute to regression and centerness losses.
Classification loss uses all samples with focal loss to handle imbalance.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for dense classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    - Reduces loss for well-classified examples (high p_t)
    - Focuses training on hard examples
    - alpha balances positive/negative samples
    - gamma controls focusing strength

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: [N, C] logits (before sigmoid)
            target: [N] class labels (0 = background for binary, or class index)
            weight: [N] optional per-sample weights

        Returns:
            loss: scalar focal loss
        """
        # For binary classification (drone vs background)
        if pred.shape[1] == 1:
            pred = pred.squeeze(1)  # [N]
            # Binary focal loss
            p = torch.sigmoid(pred)
            ce_loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')

            p_t = p * target + (1 - p) * (1 - target)
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_weight = alpha_t * (1 - p_t) ** self.gamma

            loss = focal_weight * ce_loss
        else:
            # Multi-class focal loss
            num_classes = pred.shape[1]
            target_onehot = F.one_hot(target, num_classes + 1)[:, 1:]  # Exclude background
            target_onehot = target_onehot.float()

            p = torch.sigmoid(pred)
            ce_loss = F.binary_cross_entropy_with_logits(pred, target_onehot, reduction='none')

            p_t = p * target_onehot + (1 - p) * (1 - target_onehot)
            alpha_t = self.alpha * target_onehot + (1 - self.alpha) * (1 - target_onehot)
            focal_weight = alpha_t * (1 - p_t) ** self.gamma

            loss = (focal_weight * ce_loss).sum(dim=1)

        if weight is not None:
            loss = loss * weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LTRBGIoULoss(nn.Module):
    """
    GIoU Loss for LTRB format boxes (FCOS regression output).

    LTRB format: [left, top, right, bottom] - distances from center point to box edges.
    This is different from xyxy format [x1, y1, x2, y2] - absolute coordinates.

    For LTRB centered at the same point:
    - width = left + right
    - height = top + bottom
    - Intersection uses min of corresponding sides

    Note: Standard GIoU expects xyxy format. Passing LTRB to standard GIoU
    produces wrong gradients because:
    - xyxy: area = (x2 - x1) * (y2 - y1)
    - LTRB: area = (left + right) * (top + bottom)  <-- DIFFERENT!
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: [N, 4] predicted LTRB (left, top, right, bottom)
            target: [N, 4] target LTRB (same format)
            weight: [N] optional per-sample weights

        Returns:
            loss: GIoU loss for LTRB format
        """
        # Extract LTRB components
        pred_l, pred_t, pred_r, pred_b = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        tgt_l, tgt_t, tgt_r, tgt_b = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        # Compute areas (width = l+r, height = t+b)
        pred_area = (pred_l + pred_r).clamp(min=0) * (pred_t + pred_b).clamp(min=0)
        tgt_area = (tgt_l + tgt_r).clamp(min=0) * (tgt_t + tgt_b).clamp(min=0)

        # Intersection: both boxes centered at same point
        # Intersection width = min(pred_l, tgt_l) + min(pred_r, tgt_r)
        inter_l = torch.min(pred_l, tgt_l)
        inter_r = torch.min(pred_r, tgt_r)
        inter_t = torch.min(pred_t, tgt_t)
        inter_b = torch.min(pred_b, tgt_b)

        inter_w = (inter_l + inter_r).clamp(min=0)
        inter_h = (inter_t + inter_b).clamp(min=0)
        inter_area = inter_w * inter_h

        # Union
        union_area = pred_area + tgt_area - inter_area + self.eps

        # IoU
        iou = inter_area / union_area

        # Enclosing box: max of corresponding sides
        enc_l = torch.max(pred_l, tgt_l)
        enc_r = torch.max(pred_r, tgt_r)
        enc_t = torch.max(pred_t, tgt_t)
        enc_b = torch.max(pred_b, tgt_b)

        enc_w = (enc_l + enc_r).clamp(min=0)
        enc_h = (enc_t + enc_b).clamp(min=0)
        enc_area = enc_w * enc_h + self.eps

        # GIoU = IoU - (enclosing - union) / enclosing
        giou = iou - (enc_area - union_area) / enc_area

        # Loss = 1 - GIoU (ranges from 0 to 2)
        loss = 1 - giou

        if weight is not None:
            loss = loss * weight

        if self.reduction == 'mean':
            return loss.mean() if loss.numel() > 0 else loss.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LTRBCIoULoss(nn.Module):
    """
    Complete IoU (CIoU) Loss for LTRB format predictions.

    CIoU = IoU - rho^2(centers) / c^2 - alpha * v

    Where:
    - rho^2(centers) = squared Euclidean distance between predicted and GT box centers
    - c^2 = squared diagonal of smallest enclosing box
    - v = (4/pi^2)(arctan(w_gt/h_gt) - arctan(w_pred/h_pred))^2  (aspect ratio penalty)
    - alpha = v / (1 - IoU + v)  (adaptive weight, detached from gradient)

    For LTRB format (distances from a grid point to box edges):
    - width = left + right
    - height = top + bottom
    - center_offset_x = (right - left) / 2
    - center_offset_y = (bottom - top) / 2

    CIoU provides better convergence than GIoU for small objects because:
    1. The center distance term directly penalizes misalignment
    2. The aspect ratio term encourages correct shape prediction
    3. These terms provide gradients even when boxes don't overlap

    Reference: Zheng et al., "Distance-IoU Loss", AAAI 2020
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: [N, 4] predicted LTRB (left, top, right, bottom distances)
            target: [N, 4] target LTRB (same format)
            weight: [N] optional per-sample weights

        Returns:
            loss: CIoU loss (1 - CIoU), range [0, ~2+]
        """
        # Extract LTRB components
        pred_l, pred_t, pred_r, pred_b = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        tgt_l, tgt_t, tgt_r, tgt_b = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        # Compute widths and heights from LTRB (width = l+r, height = t+b)
        pred_w = (pred_l + pred_r).clamp(min=0)
        pred_h = (pred_t + pred_b).clamp(min=0)
        tgt_w = (tgt_l + tgt_r).clamp(min=0)
        tgt_h = (tgt_t + tgt_b).clamp(min=0)

        # Compute areas
        pred_area = pred_w * pred_h
        tgt_area = tgt_w * tgt_h

        # Intersection: both boxes centered at the same grid point.
        # Intersection width = min(pred_l, tgt_l) + min(pred_r, tgt_r)
        # Intersection height = min(pred_t, tgt_t) + min(pred_b, tgt_b)
        inter_l = torch.min(pred_l, tgt_l)
        inter_r = torch.min(pred_r, tgt_r)
        inter_t = torch.min(pred_t, tgt_t)
        inter_b = torch.min(pred_b, tgt_b)

        inter_w = (inter_l + inter_r).clamp(min=0)
        inter_h = (inter_t + inter_b).clamp(min=0)
        inter_area = inter_w * inter_h

        # Union
        union_area = pred_area + tgt_area - inter_area + self.eps

        # IoU
        iou = inter_area / union_area

        # ---- CIoU-specific terms ----

        # Center distance: centers relative to the grid point
        # pred center offset = ((r - l) / 2, (b - t) / 2)
        pred_cx = (pred_r - pred_l) / 2
        pred_cy = (pred_b - pred_t) / 2
        tgt_cx = (tgt_r - tgt_l) / 2
        tgt_cy = (tgt_b - tgt_t) / 2
        rho2 = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2

        # Enclosing box diagonal squared
        # Enclosing box width = max(pred_l, tgt_l) + max(pred_r, tgt_r)
        # Enclosing box height = max(pred_t, tgt_t) + max(pred_b, tgt_b)
        enc_w = torch.max(pred_l, tgt_l) + torch.max(pred_r, tgt_r)
        enc_h = torch.max(pred_t, tgt_t) + torch.max(pred_b, tgt_b)
        c2 = enc_w ** 2 + enc_h ** 2 + self.eps

        # Aspect ratio penalty
        # v = (4/pi^2)(arctan(w_gt/h_gt) - arctan(w_pred/h_pred))^2
        v = (4.0 / (math.pi ** 2)) * (
            torch.atan(tgt_w / (tgt_h + self.eps)) -
            torch.atan(pred_w / (pred_h + self.eps))
        ) ** 2

        # Adaptive weight (detached from gradient to stabilize training)
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)

        # CIoU = IoU - center_penalty - aspect_ratio_penalty
        ciou = iou - rho2 / c2 - alpha * v

        # Loss = 1 - CIoU
        loss = 1 - ciou

        if weight is not None:
            loss = loss * weight

        if self.reduction == 'mean':
            return loss.mean() if loss.numel() > 0 else loss.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss for box regression (xyxy format).

    WARNING: This expects xyxy format [x1, y1, x2, y2], NOT LTRB format!
    For FCOS regression which outputs LTRB, use LTRBGIoULoss instead.

    GIoU = IoU - |C - U| / |C|

    Where:
    - IoU = intersection / union
    - C = smallest enclosing box
    - U = union

    GIoU ranges from -1 to 1, with 1 being perfect overlap.
    Loss = 1 - GIoU

    Reference: Rezatofighi et al., "Generalized IoU", CVPR 2019
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: [N, 4] predicted boxes (x1, y1, x2, y2) or LTRB
            target: [N, 4] target boxes (same format as pred)
            weight: [N] optional per-sample weights

        Returns:
            loss: scalar GIoU loss
        """
        # Compute areas
        pred_area = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
        target_area = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)

        # Intersection
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Union
        union_area = pred_area + target_area - inter_area + self.eps

        # IoU
        iou = inter_area / union_area

        # Enclosing box
        enc_x1 = torch.min(pred[:, 0], target[:, 0])
        enc_y1 = torch.min(pred[:, 1], target[:, 1])
        enc_x2 = torch.max(pred[:, 2], target[:, 2])
        enc_y2 = torch.max(pred[:, 3], target[:, 3])

        enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0) + self.eps

        # GIoU
        giou = iou - (enc_area - union_area) / enc_area

        # Loss
        loss = 1 - giou

        if weight is not None:
            loss = loss * weight

        if self.reduction == 'mean':
            return loss.mean() if loss.numel() > 0 else loss.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class IoULoss(nn.Module):
    """
    Standard IoU Loss.

    Loss = 1 - IoU
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: [N, 4] predicted boxes (x1, y1, x2, y2) or LTRB
            target: [N, 4] target boxes

        Returns:
            loss: scalar IoU loss
        """
        # For LTRB format, convert to areas directly
        # LTRB: [left, top, right, bottom] distances from center point
        # Area = (left + right) * (top + bottom)

        if (pred >= 0).all() and (target >= 0).all():
            # LTRB format
            pred_area = (pred[:, 0] + pred[:, 2]) * (pred[:, 1] + pred[:, 3])
            target_area = (target[:, 0] + target[:, 2]) * (target[:, 1] + target[:, 3])

            # Intersection dimensions
            inter_l = torch.min(pred[:, 0], target[:, 0])
            inter_t = torch.min(pred[:, 1], target[:, 1])
            inter_r = torch.min(pred[:, 2], target[:, 2])
            inter_b = torch.min(pred[:, 3], target[:, 3])

            inter_area = (inter_l + inter_r).clamp(min=0) * (inter_t + inter_b).clamp(min=0)
        else:
            # xyxy format
            pred_area = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
            target_area = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)

            inter_x1 = torch.max(pred[:, 0], target[:, 0])
            inter_y1 = torch.max(pred[:, 1], target[:, 1])
            inter_x2 = torch.min(pred[:, 2], target[:, 2])
            inter_y2 = torch.min(pred[:, 3], target[:, 3])

            inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        union_area = pred_area + target_area - inter_area + self.eps
        iou = inter_area / union_area

        loss = 1 - iou

        if weight is not None:
            loss = loss * weight

        if self.reduction == 'mean':
            return loss.mean() if loss.numel() > 0 else loss.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FCOSLoss(nn.Module):
    """
    Complete FCOS Loss combining classification, regression, and centerness.

    Total = cls_weight * L_cls + reg_weight * L_reg + ctr_weight * L_ctr

    L_cls: Focal loss over all locations
    L_reg: GIoU loss over positive locations only
    L_ctr: BCE loss over positive locations only
    """

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        ctr_weight: float = 1.0,
        iou_quality_weight: float = 0.0,
        num_classes: int = 1,
        use_giou: bool = True,
        use_ciou: bool = False,
        centerness_weighted_reg: bool = False,
    ):
        super().__init__()

        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.ctr_weight = ctr_weight
        self.iou_quality_weight = iou_quality_weight
        self.num_classes = num_classes

        # Centerness-weighted regression: weight each positive sample's regression
        # loss by its centerness target. This makes the model focus regression quality
        # on high-centerness (well-centered) predictions, which are most likely to
        # survive NMS and become final detections. Edge predictions with low centerness
        # get down-weighted since they contribute less to final detection quality.
        self.centerness_weighted_reg = centerness_weighted_reg

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='sum')

        # Regression loss selection (all operate on LTRB format):
        #   use_ciou=True  -> LTRBCIoULoss (best: center + aspect ratio penalties)
        #   use_giou=True  -> LTRBGIoULoss (good: enclosing box penalty)
        #   neither        -> IoULoss (basic: no gradient when boxes don't overlap)
        # CIoU takes precedence over GIoU when both are True.
        # When centerness_weighted_reg is True, we use 'none' reduction so we can
        # apply per-sample centerness weights before summing.
        reg_reduction = 'none' if centerness_weighted_reg else 'sum'
        if use_ciou:
            self.reg_loss = LTRBCIoULoss(reduction=reg_reduction)
        elif use_giou:
            self.reg_loss = LTRBGIoULoss(reduction=reg_reduction)
        else:
            self.reg_loss = IoULoss(reduction=reg_reduction)

        self.ctr_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.iou_quality_loss = nn.BCEWithLogitsLoss(reduction='sum')

    @staticmethod
    def _ltrb_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Compute IoU for LTRB-format boxes sharing the same anchor point."""
        pred_l, pred_t, pred_r, pred_b = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        tgt_l, tgt_t, tgt_r, tgt_b = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        pred_w = (pred_l + pred_r).clamp(min=0)
        pred_h = (pred_t + pred_b).clamp(min=0)
        tgt_w = (tgt_l + tgt_r).clamp(min=0)
        tgt_h = (tgt_t + tgt_b).clamp(min=0)
        pred_area = pred_w * pred_h
        tgt_area = tgt_w * tgt_h

        inter_l = torch.min(pred_l, tgt_l)
        inter_t = torch.min(pred_t, tgt_t)
        inter_r = torch.min(pred_r, tgt_r)
        inter_b = torch.min(pred_b, tgt_b)
        inter_w = (inter_l + inter_r).clamp(min=0)
        inter_h = (inter_t + inter_b).clamp(min=0)
        inter_area = inter_w * inter_h
        union = pred_area + tgt_area - inter_area + eps
        return (inter_area / union).clamp(min=0.0, max=1.0)

    def forward(
        self,
        cls_preds: List[torch.Tensor],
        reg_preds: List[torch.Tensor],
        ctr_preds: List[torch.Tensor],
        targets: Dict[str, List[torch.Tensor]],
        strides: List[int],
        iou_quality_preds: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute FCOS losses.

        Args:
            cls_preds: List of [B, num_classes, H, W] per level
            reg_preds: List of [B, 4, H, W] LTRB per level
            ctr_preds: List of [B, 1, H, W] per level
            iou_quality_preds: Optional list of [B, 1, H, W] IoU-quality logits
            targets: Dictionary with:
                - labels: List of [B, H*W] per level
                - ltrb_targets: List of [B, H*W, 4] per level
                - centerness_targets: List of [B, H*W] per level
                - pos_masks: List of [B, H*W] per level
            strides: List of strides per level

        Returns:
            Dictionary with loss components and total
        """
        device = cls_preds[0].device

        # Flatten predictions
        all_cls_preds = []
        all_reg_preds = []
        all_ctr_preds = []
        all_iouq_preds = []

        for cls, reg, ctr in zip(cls_preds, reg_preds, ctr_preds):
            B, C, H, W = cls.shape
            # Reshape to [B, H*W, C] then flatten to [B*H*W, C]
            all_cls_preds.append(cls.permute(0, 2, 3, 1).reshape(-1, C))
            all_reg_preds.append(reg.permute(0, 2, 3, 1).reshape(-1, 4))
            all_ctr_preds.append(ctr.permute(0, 2, 3, 1).reshape(-1, 1))
        if iou_quality_preds is not None:
            for iouq in iou_quality_preds:
                all_iouq_preds.append(iouq.permute(0, 2, 3, 1).reshape(-1, 1))

        cls_preds_flat = torch.cat(all_cls_preds, dim=0)  # [N_total, C]
        reg_preds_flat = torch.cat(all_reg_preds, dim=0)  # [N_total, 4]
        ctr_preds_flat = torch.cat(all_ctr_preds, dim=0).squeeze(1)  # [N_total]
        if all_iouq_preds:
            iouq_preds_flat = torch.cat(all_iouq_preds, dim=0).squeeze(1)  # [N_total]
        else:
            iouq_preds_flat = None

        # Flatten targets
        labels_flat = torch.cat([t.flatten() for t in targets['labels']])
        ltrb_flat = torch.cat([t.view(-1, 4) for t in targets['ltrb_targets']])
        centerness_flat = torch.cat([t.flatten() for t in targets['centerness_targets']])
        pos_mask_flat = torch.cat([t.flatten() for t in targets['pos_masks']])

        # Count positives
        num_pos = pos_mask_flat.sum().clamp(min=1)

        # Classification loss (all samples)
        # Convert labels to binary (0 = background, 1 = object)
        cls_targets = (labels_flat > 0).float()
        cls_loss = self.focal_loss(cls_preds_flat, cls_targets)
        cls_loss = cls_loss / num_pos

        # Regression loss (positive samples only)
        if pos_mask_flat.any():
            pos_reg_preds = reg_preds_flat[pos_mask_flat]
            pos_ltrb_targets = ltrb_flat[pos_mask_flat]
            pos_centerness = centerness_flat[pos_mask_flat]

            if self.centerness_weighted_reg:
                # Centerness-weighted regression: weight each sample's regression
                # loss by its centerness target. This focuses regression quality on
                # well-centered predictions that are most likely to survive NMS.
                # Uses 'none' reduction, applies centerness weights, then sums.
                reg_loss_per_sample = self.reg_loss(pos_reg_preds, pos_ltrb_targets)  # [N_pos]
                reg_loss = (reg_loss_per_sample * pos_centerness).sum()
                # Normalize by sum of centerness weights (not num_pos) to keep
                # loss magnitude stable regardless of centerness distribution.
                reg_loss = reg_loss / pos_centerness.sum().clamp(min=1)
            else:
                # Default: unweighted regression loss over positive samples.
                # The original FCOS paper does not weight regression by centerness.
                # Centerness only weights the final confidence during inference.
                reg_loss = self.reg_loss(pos_reg_preds, pos_ltrb_targets)
                # Normalize by num_pos, not centerness.sum()
                # Using centerness.sum() causes unpredictable loss magnitudes
                reg_loss = reg_loss / num_pos

            # Centerness loss (positive samples only)
            pos_ctr_preds = ctr_preds_flat[pos_mask_flat]
            ctr_loss = self.ctr_loss(pos_ctr_preds, pos_centerness)
            ctr_loss = ctr_loss / num_pos

            # Optional IoU-quality branch: regress matched IoU as score quality.
            if iouq_preds_flat is not None and self.iou_quality_weight > 0:
                pos_iouq_preds = iouq_preds_flat[pos_mask_flat]
                with torch.no_grad():
                    pos_iou_targets = self._ltrb_iou(pos_reg_preds.detach(), pos_ltrb_targets)
                iouq_loss = self.iou_quality_loss(pos_iouq_preds, pos_iou_targets) / num_pos
            else:
                iouq_loss = torch.tensor(0.0, device=device)
        else:
            reg_loss = torch.tensor(0.0, device=device)
            ctr_loss = torch.tensor(0.0, device=device)
            iouq_loss = torch.tensor(0.0, device=device)

        # Total loss
        total_loss = (
            self.cls_weight * cls_loss +
            self.reg_weight * reg_loss +
            self.ctr_weight * ctr_loss +
            self.iou_quality_weight * iouq_loss
        )

        return {
            'loss': total_loss,
            'cls_loss': cls_loss.detach(),
            'reg_loss': reg_loss.detach() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            'ctr_loss': ctr_loss.detach() if isinstance(ctr_loss, torch.Tensor) else ctr_loss,
            'iou_quality_loss': iouq_loss.detach() if isinstance(iouq_loss, torch.Tensor) else iouq_loss,
            'num_pos': num_pos.detach()
        }


def test_fcos_loss():
    """Test the FCOS loss functions."""
    print("Testing FCOS losses...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test Focal Loss
    print("\n1. Testing Focal Loss...")
    focal = FocalLoss(alpha=0.25, gamma=2.0)

    pred = torch.randn(100, 1, device=device)
    target = torch.randint(0, 2, (100,), device=device)

    fl = focal(pred, target)
    print(f"   Focal loss: {fl.item():.4f}")

    # Test GIoU Loss
    print("\n2. Testing GIoU Loss...")
    giou = GIoULoss()

    # LTRB format boxes
    pred_ltrb = torch.rand(10, 4, device=device) * 50 + 1  # Positive distances
    target_ltrb = torch.rand(10, 4, device=device) * 50 + 1

    gl = giou(pred_ltrb, target_ltrb)
    print(f"   GIoU loss: {gl.item():.4f}")

    # Test IoU Loss
    print("\n3. Testing IoU Loss...")
    iou = IoULoss()
    il = iou(pred_ltrb, target_ltrb)
    print(f"   IoU loss: {il.item():.4f}")

    # Test complete FCOS Loss
    print("\n4. Testing Complete FCOS Loss...")
    fcos_loss = FCOSLoss(
        focal_alpha=0.25,
        focal_gamma=2.0,
        cls_weight=1.0,
        reg_weight=1.0,
        ctr_weight=1.0,
        num_classes=1
    )

    # Simulate 3 FPN levels, batch size 2
    # Backbone-matching strides [2, 4, 8] with corresponding feature sizes
    batch_size = 2
    cls_preds = [
        torch.randn(batch_size, 1, 320, 320, device=device),  # stride 2
        torch.randn(batch_size, 1, 160, 160, device=device),  # stride 4
        torch.randn(batch_size, 1, 80, 80, device=device),    # stride 8
    ]
    reg_preds = [
        torch.rand(batch_size, 4, 320, 320, device=device) * 16,   # Positive LTRB
        torch.rand(batch_size, 4, 160, 160, device=device) * 32,
        torch.rand(batch_size, 4, 80, 80, device=device) * 64,
    ]
    ctr_preds = [
        torch.randn(batch_size, 1, 320, 320, device=device),
        torch.randn(batch_size, 1, 160, 160, device=device),
        torch.randn(batch_size, 1, 80, 80, device=device),
    ]

    # Simulate targets
    targets = {
        'labels': [
            torch.randint(0, 2, (batch_size, 320*320), device=device),
            torch.randint(0, 2, (batch_size, 160*160), device=device),
            torch.randint(0, 2, (batch_size, 80*80), device=device),
        ],
        'ltrb_targets': [
            torch.rand(batch_size, 320*320, 4, device=device) * 16,
            torch.rand(batch_size, 160*160, 4, device=device) * 32,
            torch.rand(batch_size, 80*80, 4, device=device) * 64,
        ],
        'centerness_targets': [
            torch.rand(batch_size, 320*320, device=device),
            torch.rand(batch_size, 160*160, device=device),
            torch.rand(batch_size, 80*80, device=device),
        ],
        'pos_masks': [
            torch.randint(0, 2, (batch_size, 320*320), device=device).bool(),
            torch.randint(0, 2, (batch_size, 160*160), device=device).bool(),
            torch.randint(0, 2, (batch_size, 80*80), device=device).bool(),
        ],
    }

    strides = [2, 4, 8]
    losses = fcos_loss(cls_preds, reg_preds, ctr_preds, targets, strides)

    print(f"   Total loss: {losses['loss'].item():.4f}")
    print(f"   Cls loss: {losses['cls_loss'].item():.4f}")
    print(f"   Reg loss: {losses['reg_loss']:.4f}")
    print(f"   Ctr loss: {losses['ctr_loss']:.4f}")
    print(f"   Num positives: {losses['num_pos'].item():.0f}")

    # Test with no positives
    print("\n5. Testing with no positive samples...")
    # Feature sizes matching strides [2, 4, 8]
    targets_no_pos = {
        'labels': [torch.zeros(batch_size, h*w, device=device, dtype=torch.long) for h, w in [(320, 320), (160, 160), (80, 80)]],
        'ltrb_targets': [torch.zeros(batch_size, h*w, 4, device=device) for h, w in [(320, 320), (160, 160), (80, 80)]],
        'centerness_targets': [torch.zeros(batch_size, h*w, device=device) for h, w in [(320, 320), (160, 160), (80, 80)]],
        'pos_masks': [torch.zeros(batch_size, h*w, device=device, dtype=torch.bool) for h, w in [(320, 320), (160, 160), (80, 80)]],
    }
    losses_no_pos = fcos_loss(cls_preds, reg_preds, ctr_preds, targets_no_pos, strides)
    print(f"   Total loss (no pos): {losses_no_pos['loss'].item():.4f}")
    assert not torch.isnan(losses_no_pos['loss']), "Loss should not be NaN with no positives"

    print("\nSUCCESS: FCOS losses working!")


if __name__ == '__main__':
    test_fcos_loss()
