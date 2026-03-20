#!/usr/bin/env python3
"""
SparseVoxelDet — Fully Sparse Event-Based Object Detector (6-channel variant).

Architecture:
  Events → Sparse Voxels → SparseSEWResNet → SparseFPN → SparseDetHead → NMS

Zero dense conversions. Every operation is sparse.
Predictions are made ONLY at active voxel positions.

Key design choices:
  - SparseConvTranspose3d for upsampling (SparseInverseConv3d requires indice_key from encoder).
  - Stride-4 fusion: at 640×640, stride-8 yields 80×80 (too coarse for small drones).
  - Temporal dim squeezed at FPN output (3D sparse backbone → 2D detection head).
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from torchvision.ops import nms, batched_nms

from backbone.sparse_sew_resnet import (
    SparseSEWResNet,
    SafeSparseBatchNorm,
    SparseActivation,
    sort_sparse_tensor,
    sparse_add,
)


# ---------------------------------------------------------------------------
# Sparse FPN — Feature Pyramid in Sparse Domain
# ---------------------------------------------------------------------------

class SparseLateralBlock(nn.Module):
    """1×1 sparse conv to unify channel dimensions."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = spconv.SubMConv3d(in_channels, out_channels, 1, bias=False)
        self.bn = SafeSparseBatchNorm(out_channels)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        out = self.conv(x)
        return out.replace_feature(self.bn(out.features))


class SparseUpsampleBlock(nn.Module):
    """Sparse transpose convolution to upsample spatial dimensions 2×.

    Uses SparseConvTranspose3d with stride=(1,2,2) — upsamples only spatial
    dimensions, preserves temporal dimension.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.deconv = spconv.SparseConvTranspose3d(
            in_channels, out_channels, kernel_size=3,
            stride=(1, 2, 2), padding=1,
            bias=False,
        )
        self.bn = SafeSparseBatchNorm(out_channels)
        self.act = SparseActivation()

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        x = sort_sparse_tensor(x)
        out = self.deconv(x)
        out = out.replace_feature(self.bn(out.features))
        return self.act(out)


class SparseOutputBlock(nn.Module):
    """3×3 sparse conv to refine fused features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = spconv.SubMConv3d(channels, channels, 3, bias=False)
        self.bn = SafeSparseBatchNorm(channels)
        self.act = SparseActivation()

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        out = self.conv(x)
        out = out.replace_feature(self.bn(out.features))
        return self.act(out)


class SparseFPN(nn.Module):
    """Feature Pyramid Network entirely in sparse domain.

    Takes backbone outputs at 3 scales (strides 4, 8, 16 with stem_stride=(1,2,2))
    and fuses them top-down to produce a single sparse feature map at stride 4.

    All operations are sparse: lateral 1×1, transpose-conv upsample, sparse_add.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 128,
    ) -> None:
        super().__init__()
        assert len(in_channels_list) == 3, "Expected 3 backbone output levels"

        # Lateral connections: project each level to out_channels
        self.lateral_c2 = SparseLateralBlock(in_channels_list[0], out_channels)
        self.lateral_c3 = SparseLateralBlock(in_channels_list[1], out_channels)
        self.lateral_c4 = SparseLateralBlock(in_channels_list[2], out_channels)

        # Upsample blocks: bring C4→C3 resolution, C3→C2 resolution
        self.up_c4_to_c3 = SparseUpsampleBlock(out_channels, out_channels)
        self.up_c3_to_c2 = SparseUpsampleBlock(out_channels, out_channels)

        # Output refinement
        self.out_block = SparseOutputBlock(out_channels)

    def forward(
        self,
        features: List[spconv.SparseConvTensor],
    ) -> spconv.SparseConvTensor:
        """
        Args:
            features: [c2, c3, c4] sparse feature maps from backbone at strides [4, 8, 16].

        Returns:
            Single sparse feature map at stride 4 with top-down fusion.
        """
        c2, c3, c4 = features

        # Lateral projections
        p4 = self.lateral_c4(c4)
        p3 = self.lateral_c3(c3)
        p2 = self.lateral_c2(c2)

        # Top-down pathway
        p4_up = self.up_c4_to_c3(p4)  # stride 16 → stride 8
        p3 = sparse_add(p3, p4_up)    # Fuse C3 + upsampled C4

        p3_up = self.up_c3_to_c2(p3)  # stride 8 → stride 4
        p2 = sparse_add(p2, p3_up)    # Fuse C2 + upsampled (C3 + C4)

        # Refine
        out = self.out_block(p2)
        return out


# ---------------------------------------------------------------------------
# Temporal Squeeze — Collapse T dimension for 2D detection
# ---------------------------------------------------------------------------

def sparse_temporal_pool(
    x: spconv.SparseConvTensor,
    mode: str = "max",
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Squeeze the temporal dimension out of a 3D sparse tensor.

    Collapses T bins per (batch, y, x) position via max-pool or mean-pool.

    Args:
        x: SparseConvTensor with indices [N, 4] = (batch, t, y, x)
           and spatial_shape = [T, H, W].
        mode: 'max' or 'mean'.

    Returns:
        features_2d: [M, C] pooled features (one per unique (batch, y, x)).
        indices_2d:  [M, 3] = (batch, y, x) integer indices.
        spatial_2d:  [H, W] 2D spatial shape.
    """
    indices = x.indices  # [N, 4] = (b, t, y, x)
    feats = x.features   # [N, C]
    T, H, W = x.spatial_shape

    # Compute unique (batch, y, x) keys — drop the temporal dimension
    b = indices[:, 0].long()
    y = indices[:, 2].long()
    xl = indices[:, 3].long()
    batch_size = x.batch_size

    # Linear key for unique (b, y, x) — ignore t
    key = b * (H * W) + y * W + xl
    unique_keys, inverse = torch.unique(key, return_inverse=True)
    M = unique_keys.shape[0]
    C = feats.shape[1]

    if mode == "max":
        # Scatter max: for each unique spatial position, take the max feature across T
        out = torch.full((M, C), float("-inf"), device=feats.device, dtype=feats.dtype)
        out.scatter_reduce_(0, inverse.unsqueeze(1).expand(-1, C), feats, reduce="amax")
        # Replace only -inf sentinel values (positions that had no input features)
        # Do NOT clamp valid negative maxima — they carry real information
        out = torch.where(out == float("-inf"), torch.zeros_like(out), out)
    elif mode == "mean":
        out = torch.zeros(M, C, device=feats.device, dtype=feats.dtype)
        counts = torch.zeros(M, 1, device=feats.device, dtype=feats.dtype)
        out.scatter_add_(0, inverse.unsqueeze(1).expand(-1, C), feats)
        counts.scatter_add_(0, inverse.unsqueeze(1).expand(-1, 1), torch.ones_like(feats[:, :1]))
        out = out / counts.clamp(min=1.0)
    else:
        raise ValueError(f"Unknown pool mode: {mode}")

    # Recover 2D indices
    indices_2d = torch.zeros(M, 3, device=indices.device, dtype=torch.int32)
    indices_2d[:, 0] = (unique_keys // (H * W)).int()  # batch
    remainder = unique_keys % (H * W)
    indices_2d[:, 1] = (remainder // W).int()           # y
    indices_2d[:, 2] = (remainder % W).int()             # x

    return out, indices_2d, (H, W)


# ---------------------------------------------------------------------------
# Sparse Detection Head — Predict at each active voxel position
# ---------------------------------------------------------------------------

class SparseDetHead(nn.Module):
    """Detection head operating on 2D pooled sparse features.

    Each active (y, x) position predicts:
      - cls:  [num_classes] classification logit
      - ltrb: [4] box regression (left, top, right, bottom from position center)
      - ctr:  [1] centerness logit

    Uses standard 2D convolutions on the dense-collected features.
    Why not sparse 2D conv? Because after temporal pooling, we have a flat
    feature matrix [M, C] where M = num active spatial positions. Applying
    conv2d would require reconstructing a 2D sparse tensor. Instead, we use
    a simple MLP (shared 1×1 convs = linear layers) which is more efficient
    for this count of positions.
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 128,
        num_classes: int = 1,
        num_convs: int = 2,
        prior_prob: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Shared trunk (MLP acting as shared 1×1 convs)
        trunk = []
        ch = in_channels
        for _ in range(num_convs):
            trunk.append(nn.Linear(ch, hidden_channels))
            trunk.append(nn.GroupNorm(min(8, hidden_channels), hidden_channels))
            trunk.append(nn.ReLU(inplace=True))
            ch = hidden_channels
        self.trunk = nn.Sequential(*trunk)

        # Classification branch
        self.cls_head = nn.Linear(hidden_channels, num_classes)
        # Box regression: LTRB (left, top, right, bottom distances)
        self.box_head = nn.Linear(hidden_channels, 4)
        # Centerness
        self.ctr_head = nn.Linear(hidden_channels, 1)

        # Initialize
        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob: float) -> None:
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Cls bias: initialize to give prior_prob output under sigmoid
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.zeros_(self.cls_head.weight)
        nn.init.constant_(self.cls_head.bias, bias_value)

        # Box regression: small positive init
        nn.init.zeros_(self.box_head.weight)
        nn.init.constant_(self.box_head.bias, 4.0)  # Start with ~exp(4)=55px boxes

        # Centerness: zero init
        nn.init.zeros_(self.ctr_head.weight)
        nn.init.zeros_(self.ctr_head.bias)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [M, C] — 2D-pooled sparse features.

        Returns:
            cls_logits: [M, num_classes]
            box_ltrb:   [M, 4] — raw LTRB (apply exp() to get pixel distances)
            ctr_logits: [M, 1]
        """
        h = self.trunk(features)
        cls_logits = self.cls_head(h)
        box_ltrb = self.box_head(h)
        ctr_logits = self.ctr_head(h)
        return cls_logits, box_ltrb, ctr_logits


# ---------------------------------------------------------------------------
# SparseVoxelDet — Full model
# ---------------------------------------------------------------------------

class SparseVoxelDet(nn.Module):
    """Fully sparse event-based object detector.

    Pipeline:
        SparseConvTensor [B, T, H, W]
          → SparseSEWResNet backbone (sparse 3D conv)
          → SparseFPN top-down fusion (sparse 3D conv)
          → Temporal squeeze (max-pool over T) → [M, C] + [M, 3=(b,y,x)]
          → SparseDetHead (MLP) → cls, box, centerness per active position
          → Decode + NMS → detections

    Every convolution until the head is sparse (spconv). The head is an MLP
    on the M active positions. No dense feature maps anywhere.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 1,
        backbone_size: str = "nano_deep",
        fpn_channels: int = 128,
        head_convs: int = 2,
        strides: Optional[List[int]] = None,
        input_size: Tuple[int, int] = (720, 1280),
        time_bins: int = 16,
        prior_prob: float = 0.01,
        # Decode defaults
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        max_detections: int = 100,
        temporal_pool_mode: str = "max",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size  # (H, W)
        self.time_bins = time_bins
        self.stride = 4  # Output stride of fused features
        # Hardcoded to 4 because SparseFPN fuses to stride-4 level.
        # Assert at construction time so config drift is caught immediately.
        assert strides is None or strides[0] == self.stride, (
            f"Model.stride={self.stride} but strides[0]={strides[0] if strides else 'None'}. "
            "SparseFPN fuses to the finest stride. If you change strides, update self.stride."
        )
        self.temporal_pool_mode = temporal_pool_mode

        if strides is None:
            strides = [4, 8, 16]
        self.strides = strides

        # Decode params (can be overridden)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        if max_detections < 1:
            raise ValueError(f"max_detections must be >= 1, got {max_detections}")
        self.max_detections = max_detections

        # Backbone
        self.backbone = SparseSEWResNet(
            in_channels=in_channels,
            size=backbone_size,
            stem_stride=(1, 2, 2),  # Spatial strides [4, 8, 16]
        )

        # Sparse FPN
        self.fpn = SparseFPN(
            in_channels_list=self.backbone.out_channels,
            out_channels=fpn_channels,
        )

        # Detection head
        self.head = SparseDetHead(
            in_channels=fpn_channels,
            hidden_channels=fpn_channels,
            num_classes=num_classes,
            num_convs=head_convs,
            prior_prob=prior_prob,
        )

    def set_decode_params(
        self,
        score_thresh: Optional[float] = None,
        nms_thresh: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> None:
        if score_thresh is not None:
            self.score_thresh = score_thresh
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        if max_detections is not None:
            if max_detections < 1:
                raise ValueError(f"max_detections must be >= 1, got {max_detections}")
            self.max_detections = max_detections

    def get_num_params(self) -> Dict[str, int]:
        """Parameter count breakdown."""
        backbone_p = sum(p.numel() for p in self.backbone.parameters())
        fpn_p = sum(p.numel() for p in self.fpn.parameters())
        head_p = sum(p.numel() for p in self.head.parameters())
        total = backbone_p + fpn_p + head_p
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "backbone": backbone_p,
            "fpn": fpn_p,
            "head": head_p,
            "total": total,
            "trainable": trainable,
        }

    def _decode_detections(
        self,
        cls_logits: torch.Tensor,     # [M, num_cls]
        box_ltrb: torch.Tensor,       # [M, 4]
        ctr_logits: torch.Tensor,     # [M, 1]
        indices_2d: torch.Tensor,     # [M, 3] = (batch, y, x)
        batch_size: int,
        score_thresh: Optional[float] = None,
        nms_thresh: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> torch.Tensor:
        """Decode sparse predictions to boxes. Returns [B, max_det, 6].

        Each detection: [x1, y1, x2, y2, score, class].
        """
        st = score_thresh if score_thresh is not None else self.score_thresh
        nt = nms_thresh if nms_thresh is not None else self.nms_thresh
        md = max_detections if max_detections is not None else self.max_detections
        if md < 1:
            raise ValueError(f"max_detections must be >= 1, got {md}")

        stride = self.stride
        device = cls_logits.device

        # Compute center points from indices
        cx = indices_2d[:, 2].float() * stride + stride / 2.0
        cy = indices_2d[:, 1].float() * stride + stride / 2.0
        batch_idx = indices_2d[:, 0]

        # Decode boxes: LTRB → xyxy
        # Clamp logits before exp() to prevent fp16 overflow (exp(10)≈22026, fp16 max≈65504)
        ltrb = torch.exp(box_ltrb.clamp(max=10.0))  # exp to ensure positive distances
        x1 = cx - ltrb[:, 0]
        y1 = cy - ltrb[:, 1]
        x2 = cx + ltrb[:, 2]
        y2 = cy + ltrb[:, 3]
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [M, 4]

        # Clamp to image bounds
        img_h, img_w = self.input_size
        boxes[:, 0].clamp_(min=0, max=img_w)
        boxes[:, 1].clamp_(min=0, max=img_h)
        boxes[:, 2].clamp_(min=0, max=img_w)
        boxes[:, 3].clamp_(min=0, max=img_h)

        # Score = sigmoid(cls) × sigmoid(centerness)
        cls_scores = torch.sigmoid(cls_logits[:, 0])  # [M] (single class)
        ctr_scores = torch.sigmoid(ctr_logits[:, 0])  # [M]
        scores = cls_scores * ctr_scores

        # Per-batch NMS
        output = torch.zeros(batch_size, md, 6, device=device)
        for b in range(batch_size):
            mask = batch_idx == b
            if not mask.any():
                continue
            b_scores = scores[mask]
            b_boxes = boxes[mask]

            # Score threshold
            keep = b_scores > st
            b_scores = b_scores[keep]
            b_boxes = b_boxes[keep]

            if b_scores.numel() == 0:
                continue

            # NMS
            keep_idx = nms(b_boxes, b_scores, nt)
            keep_idx = keep_idx[:md]

            n = keep_idx.shape[0]
            output[b, :n, :4] = b_boxes[keep_idx]
            output[b, :n, 4] = b_scores[keep_idx]
            output[b, :n, 5] = 0  # class 0

        return output

    def forward(
        self,
        x: spconv.SparseConvTensor,
        batch_size: Optional[int] = None,
        targets: Optional[Dict] = None,
        return_loss_inputs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: SparseConvTensor [N, in_channels] with indices [N, 4]=(b,t,y,x).
            batch_size: Batch size.
            targets: Unused (for interface compat).
            return_loss_inputs: If True, return raw predictions for loss.

        Returns:
            If training or return_loss_inputs:
                {
                    'cls_logits': [M, num_cls],
                    'box_ltrb': [M, 4],
                    'ctr_logits': [M, 1],
                    'indices_2d': [M, 3] = (batch, y, x),
                    'spatial_2d': (H, W),
                }
            Else:
                {
                    'detections': [B, max_det, 6],
                }
        """
        if batch_size is None:
            batch_size = int(x.indices[:, 0].max().item()) + 1

        # 1. Sparse backbone → 3 sparse feature maps
        backbone_features = self.backbone(x)

        # 2. Sparse FPN → single fused sparse map at stride 4
        fused = self.fpn(backbone_features)

        # 3. Temporal squeeze → 2D features + indices
        features_2d, indices_2d, spatial_2d = sparse_temporal_pool(
            fused, mode=self.temporal_pool_mode,
        )

        # 4. Detection head
        cls_logits, box_ltrb, ctr_logits = self.head(features_2d)

        if self.training or return_loss_inputs:
            return {
                "cls_logits": cls_logits,
                "box_ltrb": box_ltrb,
                "ctr_logits": ctr_logits,
                "indices_2d": indices_2d,
                "spatial_2d": spatial_2d,
            }
        else:
            detections = self._decode_detections(
                cls_logits, box_ltrb, ctr_logits,
                indices_2d, batch_size,
            )
            return {"detections": detections}
