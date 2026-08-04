#!/usr/bin/env python3
"""
Loss & Target Assignment for SparseVoxelDet.

Key insight: Unlike FCOS which predicts at ALL grid positions, SparseVoxelDet
predicts ONLY at active voxel positions (where events occurred). This means:

1. Target assignment only considers sparse positions, not a dense grid.
2. A GT box may have NO active voxels near its center — we handle this gracefully.
3. The set of "positive" positions is a subset of active voxels, not all grid cells.

Assignment strategy:
  - For each GT box, find all active voxels whose center falls inside the box.
  - Among those, compute centerness (distance to box center vs edges).
  - If no voxel is inside a GT box, use the nearest voxel within a radius.
  - This is a proximity-based assignment: simple, no chicken-and-egg, no learned components.

Losses:
  - Focal loss for classification (binary: drone vs background)
  - GIoU loss for LTRB box regression (positive samples only)
  - BCE loss for centerness (positive samples only)
  - Optional QFL classification targets and NWD box mixing, config-gated

Total = cls_weight * L_cls + reg_weight * L_reg + ctr_weight * L_ctr
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Target assignment for sparse voxel positions
# ---------------------------------------------------------------------------

def assign_sparse_targets(
    indices_2d: torch.Tensor,       # [M, 3] = (batch, y, x) grid indices
    gt_boxes_batch: List[torch.Tensor],  # List of [Mb, 4] xyxy boxes per batch
    gt_labels_batch: List[torch.Tensor], # List of [Mb] labels per batch
    stride: int = 4,
    center_sampling_radius: float = 1.5,
    num_classes: int = 1,
) -> Dict[str, torch.Tensor]:
    """Assign GT targets to sparse active voxel positions.

    For each active voxel at grid position (y, x), its pixel center is:
        cx = x * stride + stride / 2
        cy = y * stride + stride / 2

    A voxel is positive for a GT box if:
        1. The voxel center falls inside the GT box, AND
        2. The voxel center is within center_sampling_radius * stride of the box center.
    If multiple GT boxes match, assign to the smallest (by area).
    If no voxel matches a GT box, force-match the nearest voxel within max_radius.

    Args:
        indices_2d: [M, 3] sparse position indices (batch, y, x).
        gt_boxes_batch: Per-batch GT boxes in xyxy format.
        gt_labels_batch: Per-batch GT labels.
        stride: Feature stride (pixel distance per grid cell).
        center_sampling_radius: Radius in stride units for center sampling.
        num_classes: Number of classes (1 for binary).

    Returns:
        Dictionary with:
            cls_targets: [M] float, 0=neg, 1=pos for focal loss.
            ltrb_targets: [M, 4] LTRB regression targets (only valid for positives).
            centerness_targets: [M] centerness in [0, 1].
            pos_mask: [M] bool, True for positive voxels.
    """
    M = indices_2d.shape[0]
    device = indices_2d.device

    cls_targets = torch.zeros(M, device=device, dtype=torch.float32)
    ltrb_targets = torch.zeros(M, 4, device=device, dtype=torch.float32)
    centerness_targets = torch.zeros(M, device=device, dtype=torch.float32)
    pos_mask = torch.zeros(M, device=device, dtype=torch.bool)

    # Extract batch indices and pixel centers
    batch_idx = indices_2d[:, 0].int()
    cy = indices_2d[:, 1].float() * stride + stride / 2.0
    cx = indices_2d[:, 2].float() * stride + stride / 2.0

    batch_size = int(batch_idx.max().item()) + 1 if M > 0 else 0
    radius_px = center_sampling_radius * stride

    for b in range(batch_size):
        b_mask = batch_idx == b
        if not b_mask.any():
            continue

        b_indices = torch.where(b_mask)[0]  # indices into M
        b_cx = cx[b_mask]  # [Nb]
        b_cy = cy[b_mask]

        gt_boxes = gt_boxes_batch[b].to(device)  # [Gb, 4]
        if gt_boxes.numel() == 0 or gt_boxes.shape[0] == 0:
            continue

        Gb = gt_boxes.shape[0]
        Nb = b_cx.shape[0]

        # GT box properties
        gt_x1 = gt_boxes[:, 0]  # [Gb]
        gt_y1 = gt_boxes[:, 1]
        gt_x2 = gt_boxes[:, 2]
        gt_y2 = gt_boxes[:, 3]
        gt_cx = (gt_x1 + gt_x2) / 2.0
        gt_cy = (gt_y1 + gt_y2) / 2.0
        gt_area = (gt_x2 - gt_x1).clamp(min=0) * (gt_y2 - gt_y1).clamp(min=0)

        # Compute LTRB for all (voxel, GT) pairs: [Nb, Gb, 4]
        left = b_cx.unsqueeze(1) - gt_x1.unsqueeze(0)    # cx - x1
        top = b_cy.unsqueeze(1) - gt_y1.unsqueeze(0)     # cy - y1
        right = gt_x2.unsqueeze(0) - b_cx.unsqueeze(1)   # x2 - cx
        bottom = gt_y2.unsqueeze(0) - b_cy.unsqueeze(1)  # y2 - cy
        ltrb = torch.stack([left, top, right, bottom], dim=-1)  # [Nb, Gb, 4]

        # Mask 1: voxel center is inside GT box (all LTRB > 0)
        inside = ltrb.min(dim=-1).values > 0  # [Nb, Gb]

        # Mask 2: voxel center is within radius of GT box center
        dist_x = (b_cx.unsqueeze(1) - gt_cx.unsqueeze(0)).abs()
        dist_y = (b_cy.unsqueeze(1) - gt_cy.unsqueeze(0)).abs()
        in_center = (dist_x < radius_px) & (dist_y < radius_px)

        # Combined mask: inside box AND near center
        match_mask = inside & in_center  # [Nb, Gb]

        # Force-match: for GT boxes with no matching voxel, find nearest voxel
        for g in range(Gb):
            if not match_mask[:, g].any():
                # Find nearest voxel to this GT center
                dist = (b_cx - gt_cx[g]) ** 2 + (b_cy - gt_cy[g]) ** 2
                # Only consider voxels within expanded search radius
                max_search = max(radius_px * 3, stride * 8)
                within = dist < max_search ** 2
                if within.any():
                    dist_filtered = dist.clone()
                    dist_filtered[~within] = float('inf')
                    nearest_idx = dist_filtered.argmin()
                    match_mask[nearest_idx, g] = True

        # For each voxel, assign to smallest matching GT box (handles overlaps)
        # If voxel matches no GT, it stays negative.
        assigned_gt = torch.full((Nb,), -1, device=device, dtype=torch.long)
        assigned_area = torch.full((Nb,), float('inf'), device=device)

        for g in range(Gb):
            matched_voxels = match_mask[:, g]
            if not matched_voxels.any():
                continue
            # Only override if this GT is smaller than previously assigned
            smaller = gt_area[g] < assigned_area
            update = matched_voxels & smaller
            if update.any():
                assigned_gt[update] = g
                assigned_area[update] = gt_area[g]

        # Write targets for positive voxels
        pos_local = assigned_gt >= 0
        if pos_local.any():
            pos_global = b_indices[pos_local]
            g_idx = assigned_gt[pos_local]

            # LTRB targets
            pos_ltrb = ltrb[pos_local, g_idx]  # [Npos, 4]
            # Clamp to prevent zero/negative (shouldn't happen due to inside mask, but safety)
            pos_ltrb = pos_ltrb.clamp(min=0.01)

            # Centerness
            l, t, r, bo = pos_ltrb[:, 0], pos_ltrb[:, 1], pos_ltrb[:, 2], pos_ltrb[:, 3]
            lr = torch.min(l, r) / torch.max(l, r).clamp(min=1e-6)
            tb = torch.min(t, bo) / torch.max(t, bo).clamp(min=1e-6)
            ctr = torch.sqrt((lr * tb).clamp(min=0))

            cls_targets[pos_global] = 1.0
            ltrb_targets[pos_global] = pos_ltrb
            centerness_targets[pos_global] = ctr
            pos_mask[pos_global] = True

    return {
        'cls_targets': cls_targets,
        'ltrb_targets': ltrb_targets,
        'centerness_targets': centerness_targets,
        'pos_mask': pos_mask,
    }


# ---------------------------------------------------------------------------
# Box helpers
# ---------------------------------------------------------------------------

def decode_ltrb_to_boxes(
    indices_2d: torch.Tensor,
    ltrb: torch.Tensor,
    stride: int = 4,
) -> torch.Tensor:
    """Decode LTRB distances at sparse voxel centers to xyxy boxes."""
    if ltrb.numel() == 0:
        return ltrb.new_zeros((ltrb.shape[0], 4))

    dtype = ltrb.dtype
    cy = indices_2d[:, 1].to(dtype=dtype) * stride + stride / 2.0
    cx = indices_2d[:, 2].to(dtype=dtype) * stride + stride / 2.0

    left, top, right, bottom = ltrb.unbind(-1)
    return torch.stack([
        cx - left,
        cy - top,
        cx + right,
        cy + bottom,
    ], dim=-1)


def aligned_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Aligned IoU for matching xyxy box tensors of shape [N, 4]."""
    if boxes1.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0],))

    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter + eps
    return (inter / union).clamp(min=0, max=1)


def compute_qfl_targets(
    cls_targets: torch.Tensor,
    indices_2d: torch.Tensor,
    box_ltrb: torch.Tensor,
    ltrb_targets: torch.Tensor,
    pos_mask: torch.Tensor,
    stride: int = 4,
) -> torch.Tensor:
    """Build QFL soft targets with detached predicted-box IoU for positives."""
    qfl_targets = cls_targets.clone()
    if pos_mask.any():
        pred_ltrb = torch.exp(box_ltrb[pos_mask].clamp(max=10.0)).detach()
        pred_boxes = decode_ltrb_to_boxes(indices_2d[pos_mask], pred_ltrb, stride=stride)
        target_boxes = decode_ltrb_to_boxes(indices_2d[pos_mask], ltrb_targets[pos_mask], stride=stride)
        qfl_targets[pos_mask] = aligned_box_iou(pred_boxes, target_boxes).detach()
    return qfl_targets


def normalized_wasserstein_similarity(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    c: float = 12.8,
) -> torch.Tensor:
    """NWD similarity for aligned xyxy boxes."""
    if boxes1.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0],))

    c1x = (boxes1[:, 0] + boxes1[:, 2]) / 2.0
    c1y = (boxes1[:, 1] + boxes1[:, 3]) / 2.0
    c2x = (boxes2[:, 0] + boxes2[:, 2]) / 2.0
    c2y = (boxes2[:, 1] + boxes2[:, 3]) / 2.0

    s1w = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) / 2.0
    s1h = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0) / 2.0
    s2w = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) / 2.0
    s2h = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0) / 2.0

    delta = torch.stack((c1x - c2x, c1y - c2y, s1w - s2w, s1h - s2h), dim=1)
    if delta.dtype in (torch.float16, torch.bfloat16):
        delta = delta.float()
    distance = torch.linalg.vector_norm(delta, ord=2, dim=1)
    return torch.exp(-distance / c)


def nwd_loss_xyxy(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    c: float = 12.8,
) -> torch.Tensor:
    """Per-box NWD loss for aligned xyxy boxes."""
    return 1.0 - normalized_wasserstein_similarity(boxes1, boxes2, c=c)


# ---------------------------------------------------------------------------
# Focal Loss (binary, operating on raw logits)
# ---------------------------------------------------------------------------

class BinaryFocalLoss(nn.Module):
    """Binary focal loss for sparse voxel classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha=0.25 means: 0.25 weight for positives, 0.75 for negatives.
    This DOWN-weights positives and UP-weights negatives — correct for
    FCOS/RetinaNet where positives greatly outnumber... wait, no. In
    event-based detection, we typically have MANY background voxels and
    FEW positive ones. So alpha=0.25 gives less weight to rare positives,
    which is WRONG.

    CRITICAL: For FRED dataset with sparse events, positives are RARE.
    We should use alpha=0.75 to UP-weight positives. But the V3.2 bug
    showed that alpha=0.25 worked better after the fix. Let's keep it
    configurable and default to 0.25 (standard RetinaNet).

    Actually, let me re-derive: in focal loss, alpha_t = alpha for class 1
    (positive), alpha_t = (1-alpha) for class 0 (negative). With alpha=0.25:
    - positives get weight 0.25
    - negatives get weight 0.75
    This is the standard RetinaNet convention where there are far more negatives.
    The focal modulation (1-p_t)^gamma already handles the imbalance by
    down-weighting easy negatives. Alpha slightly corrects for absolute counts.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [M, 1] or [M] raw logits.
            targets: [M] binary targets (0 or 1).

        Returns:
            Scalar focal loss (sum-normalized by num_pos externally).
        """
        if logits.dim() == 2:
            logits = logits.squeeze(1)

        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating = (1 - p_t).pow(self.gamma)

        loss = alpha_t * modulating * ce
        return loss.sum()


class BinaryQualityFocalLoss(nn.Module):
    """Binary quality focal loss for sparse voxel classification."""

    def __init__(self, alpha: float = 0.25, beta: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pos_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [M, 1] or [M] raw logits.
            targets: [M] quality targets in [0, 1].
            pos_mask: [M] optional true-positive mask for negative alpha weighting.

        Returns:
            Scalar QFL loss (sum-normalized by num_pos externally).
        """
        if logits.dim() == 2:
            logits = logits.squeeze(1)

        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        modulating = (targets - p).abs().pow(self.beta)

        if pos_mask is None:
            pos_mask = targets > 0
        alpha_t = torch.ones_like(targets)
        alpha_t = torch.where(pos_mask, alpha_t, alpha_t * (1 - self.alpha))

        loss = alpha_t * modulating * ce
        return loss.sum()


# ---------------------------------------------------------------------------
# LTRB GIoU Loss (for regression)
# ---------------------------------------------------------------------------

class LTRBGIoULoss(nn.Module):
    """GIoU loss for LTRB format box predictions.

    Both pred and target are LTRB distances from the anchor point.
    The anchor point is the same for both, so intersection is computed
    using min of corresponding LTRB components.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: [N, 4] predicted LTRB (already exp'd or raw positive).
            target: [N, 4] target LTRB.
            weight: [N] optional per-sample weights.

        Returns:
            Scalar GIoU loss (sum).
        """
        if pred.numel() == 0:
            return pred.sum()  # 0.0

        pred_l, pred_t, pred_r, pred_b = pred.unbind(-1)
        tgt_l, tgt_t, tgt_r, tgt_b = target.unbind(-1)

        pred_area = (pred_l + pred_r).clamp(min=0) * (pred_t + pred_b).clamp(min=0)
        tgt_area = (tgt_l + tgt_r).clamp(min=0) * (tgt_t + tgt_b).clamp(min=0)

        inter_l = torch.min(pred_l, tgt_l)
        inter_t = torch.min(pred_t, tgt_t)
        inter_r = torch.min(pred_r, tgt_r)
        inter_b = torch.min(pred_b, tgt_b)
        inter_w = (inter_l + inter_r).clamp(min=0)
        inter_h = (inter_t + inter_b).clamp(min=0)
        inter = inter_w * inter_h

        union = pred_area + tgt_area - inter + self.eps
        iou = inter / union

        enc_l = torch.max(pred_l, tgt_l)
        enc_t = torch.max(pred_t, tgt_t)
        enc_r = torch.max(pred_r, tgt_r)
        enc_b = torch.max(pred_b, tgt_b)
        enc_w = (enc_l + enc_r).clamp(min=0)
        enc_h = (enc_t + enc_b).clamp(min=0)
        enc_area = enc_w * enc_h + self.eps

        giou = iou - (enc_area - union) / enc_area
        loss = 1 - giou

        if weight is not None:
            loss = loss * weight

        return loss.sum()


class NWDBoxLoss(nn.Module):
    """NWD loss for aligned xyxy box predictions."""

    def __init__(self, c: float = 12.8) -> None:
        super().__init__()
        self.c = c

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_boxes: [N, 4] predicted xyxy boxes.
            target_boxes: [N, 4] target xyxy boxes.
            weight: [N] optional per-sample weights.

        Returns:
            Scalar NWD loss (sum).
        """
        if pred_boxes.numel() == 0:
            return pred_boxes.sum()

        loss = nwd_loss_xyxy(pred_boxes, target_boxes, c=self.c)

        if weight is not None:
            loss = loss * weight

        return loss.sum()


# ---------------------------------------------------------------------------
# Full SparseVoxelDet Loss
# ---------------------------------------------------------------------------

class SparseVoxelDetLoss(nn.Module):
    """Complete loss for SparseVoxelDet.

    Combines:
        1. Binary focal loss for classification at all active voxels.
        2. GIoU loss for LTRB box regression at positive voxels only.
        3. BCE loss for centerness at positive voxels only.

    Total = cls_weight * (L_cls / num_pos)
          + reg_weight * (L_reg / num_pos)
          + ctr_weight * (L_ctr / num_pos)

    All losses are sum-reduced then divided by num_pos for stable normalization.

    Interface contract: forward() returns dict with at least:
        'loss', 'cls_loss', 'reg_loss', 'ctr_loss', 'num_pos'
    This matches what train_sparse_fcos.py expects at lines 882-905.
    """

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cls_weight: float = 1.0,
        reg_weight: float = 2.0,
        ctr_weight: float = 1.0,
        stride: int = 4,
        center_sampling_radius: float = 1.5,
        num_classes: int = 1,
        use_qfl: bool = False,
        nwd_weight: float = 0.0,
        nwd_c: float = 12.8,
    ) -> None:
        super().__init__()
        if nwd_weight < 0.0 or nwd_weight > 1.0:
            raise ValueError('nwd_weight must be in [0, 1]')
        if nwd_c <= 0.0:
            raise ValueError('nwd_c must be positive')

        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.ctr_weight = ctr_weight
        self.stride = stride
        self.center_sampling_radius = center_sampling_radius
        self.num_classes = num_classes
        self.use_qfl = use_qfl
        self.nwd_weight = nwd_weight
        self.nwd_c = nwd_c

        if use_qfl:
            self.cls_loss_fn = BinaryQualityFocalLoss(alpha=focal_alpha, beta=2.0)
        else:
            self.cls_loss_fn = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.reg_loss_fn = LTRBGIoULoss()
        self.nwd_loss_fn = NWDBoxLoss(c=nwd_c)
        self.ctr_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        gt_boxes_batch: List[torch.Tensor],
        gt_labels_batch: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            outputs: Model outputs from SparseVoxelDet.forward() in training mode:
                cls_logits:  [M, num_cls]
                box_ltrb:    [M, 4] raw LTRB (before exp)
                ctr_logits:  [M, 1]
                indices_2d:  [M, 3] = (batch, y, x)
                spatial_2d:  (H, W)
            gt_boxes_batch: List of [Gb, 4] xyxy boxes per batch element.
            gt_labels_batch: List of [Gb] labels per batch element.

        Returns:
            Dict with 'loss', 'cls_loss', 'reg_loss', 'ctr_loss', 'num_pos'.
        """
        cls_logits = outputs['cls_logits']    # [M, num_cls]
        box_ltrb = outputs['box_ltrb']        # [M, 4]
        ctr_logits = outputs['ctr_logits']    # [M, 1]
        indices_2d = outputs['indices_2d']    # [M, 3]

        device = cls_logits.device

        # --- Target Assignment ---
        targets = assign_sparse_targets(
            indices_2d=indices_2d,
            gt_boxes_batch=gt_boxes_batch,
            gt_labels_batch=gt_labels_batch,
            stride=self.stride,
            center_sampling_radius=self.center_sampling_radius,
            num_classes=self.num_classes,
        )

        cls_targets = targets['cls_targets']        # [M]
        ltrb_targets = targets['ltrb_targets']      # [M, 4]
        ctr_targets = targets['centerness_targets'] # [M]
        pos_mask = targets['pos_mask']              # [M]

        num_pos_raw = pos_mask.sum()                 # True count (may be 0)
        num_pos = num_pos_raw.clamp(min=1).float()   # Safe normalization denominator

        # --- Classification Loss (all voxels) ---
        if self.use_qfl:
            cls_targets = compute_qfl_targets(
                cls_targets=cls_targets,
                indices_2d=indices_2d,
                box_ltrb=box_ltrb,
                ltrb_targets=ltrb_targets,
                pos_mask=pos_mask,
                stride=self.stride,
            )
            cls_loss = self.cls_loss_fn(cls_logits, cls_targets, pos_mask=pos_mask) / num_pos
        else:
            cls_loss = self.cls_loss_fn(cls_logits, cls_targets) / num_pos

        # --- Regression Loss (positive voxels only) ---
        if pos_mask.any():
            pos_box_ltrb = box_ltrb[pos_mask]       # [Npos, 4]
            pos_ltrb_tgt = ltrb_targets[pos_mask]    # [Npos, 4]

            # The model outputs raw LTRB values. Apply exp() to get positive distances.
            # Log-space prediction is standard (FCOS paper Eq. 2): pred = exp(s * ltrb)
            # We use simple exp() — the head init (bias=4.0) ensures reasonable starting boxes.
            # Clamp before exp() to prevent fp16 overflow: exp(10)≈22026, fp16 max≈65504.
            pred_ltrb_decoded = torch.exp(pos_box_ltrb.clamp(max=10.0))

            # Centerness weighting: weight regression by centerness target
            # so well-centered predictions get more regression signal.
            pos_ctr_tgt = ctr_targets[pos_mask]
            giou_loss = self.reg_loss_fn(pred_ltrb_decoded, pos_ltrb_tgt, weight=pos_ctr_tgt)
            if self.nwd_weight > 0.0:
                pos_indices_2d = indices_2d[pos_mask]
                pred_boxes = decode_ltrb_to_boxes(pos_indices_2d, pred_ltrb_decoded, stride=self.stride)
                target_boxes = decode_ltrb_to_boxes(pos_indices_2d, pos_ltrb_tgt, stride=self.stride)
                nwd_loss = self.nwd_loss_fn(pred_boxes, target_boxes, weight=pos_ctr_tgt)
                reg_loss = (1.0 - self.nwd_weight) * giou_loss + self.nwd_weight * nwd_loss
            else:
                reg_loss = giou_loss
            reg_loss = reg_loss / pos_ctr_tgt.sum().clamp(min=1)

            # --- Centerness Loss (positive voxels only) ---
            pos_ctr_logits = ctr_logits[pos_mask].squeeze(1)
            ctr_loss = self.ctr_loss_fn(pos_ctr_logits, pos_ctr_tgt)
            ctr_loss = ctr_loss / num_pos
        else:
            reg_loss = torch.tensor(0.0, device=device)
            ctr_loss = torch.tensor(0.0, device=device)

        total_loss = (
            self.cls_weight * cls_loss
            + self.reg_weight * reg_loss
            + self.ctr_weight * ctr_loss
        )

        return {
            'loss': total_loss,
            'cls_loss': cls_loss.detach(),
            'reg_loss': reg_loss.detach() if isinstance(reg_loss, torch.Tensor) else torch.tensor(reg_loss),
            'ctr_loss': ctr_loss.detach() if isinstance(ctr_loss, torch.Tensor) else torch.tensor(ctr_loss),
            'num_pos': num_pos.detach(),
            'num_pos_raw': num_pos_raw.detach(),  # True count before clamp(min=1)
        }
