#!/usr/bin/env python3
"""
FCOS Detection Head - Anchor-Free Object Detection.

FCOS (Fully Convolutional One-Stage Object Detection) predicts:
1. Classification: per-pixel class probabilities
2. Regression: LTRB distances (left, top, right, bottom) from each pixel to box edges
3. Centerness: quality score penalizing predictions far from object center

Key advantages for drone detection:
- No anchor design needed (drones have varying aspect ratios)
- Per-pixel prediction suits small objects (74% of FRED drones are <32px)
- Centerness suppresses low-quality predictions from object edges

Reference: Tian et al., "FCOS: Fully Convolutional One-Stage Object Detection", ICCV 2019
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class Scale(nn.Module):
    """Learnable scale factor for regression outputs."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Disable autocast: force FP32 for both forward AND backward
        # to prevent AMP FP16 overflow in scale × bbox_pred × stride
        with torch.amp.autocast('cuda', enabled=False):
            return x.float() * self.scale.clamp(min=0.01, max=10.0)


class ConvBlock(nn.Module):
    """Conv + GroupNorm + ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 8
    ):
        super().__init__()
        # Ensure groups divides channels evenly
        groups = min(groups, out_channels)
        while out_channels % groups != 0:
            groups -= 1

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class FCOSHead(nn.Module):
    """
    FCOS Detection Head.

    Architecture:
    - Classification tower: 4 conv layers -> class logits
    - Regression tower: 4 conv layers -> LTRB distances
    - Centerness: from regression tower -> single channel (per FCOS paper Section 3.2)

    Output per scale:
    - cls: [B, num_classes, H, W] - class logits (before sigmoid)
    - reg: [B, 4, H, W] - LTRB distances (positive values)
    - ctr: [B, 1, H, W] - centerness logits (before sigmoid)
    - iouq: [B, 1, H, W] - IoU-quality logits (before sigmoid)
    """

    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 1,
        num_convs: int = 4,
        num_levels: int = 3,
        prior_prob: float = 0.01,
        use_scale: bool = True,
        norm_on_bbox: bool = True,
        gn_groups: int = 8
    ):
        """
        Args:
            in_channels: Input channels from FPN
            num_classes: Number of object classes (1 for drone detection)
            num_convs: Number of conv layers in each tower
            num_levels: Number of FPN levels (default 3 for strides [2, 4, 8])
            prior_prob: Prior probability for focal loss initialization
            use_scale: Use learnable scale for regression
            norm_on_bbox: Normalize regression targets by stride
            gn_groups: Number of groups for GroupNorm in ConvBlock (default 8).
                With 128ch FPN: 8 groups = 16 channels/group (good statistics).
                With 64ch FPN: 8 groups = 8 channels/group (still reasonable).
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_levels = num_levels
        self.use_scale = use_scale
        self.norm_on_bbox = norm_on_bbox

        # Classification tower
        cls_tower = []
        for i in range(num_convs):
            cls_tower.append(ConvBlock(in_channels, in_channels, groups=gn_groups))
        self.cls_tower = nn.Sequential(*cls_tower)

        # Regression tower
        reg_tower = []
        for i in range(num_convs):
            reg_tower.append(ConvBlock(in_channels, in_channels, groups=gn_groups))
        self.reg_tower = nn.Sequential(*reg_tower)

        # Classification output
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)

        # Regression output (LTRB)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, padding=1)

        # Centerness output (shares cls tower features)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)

        # IoU-quality output (shares regression tower features)
        self.iou_quality = nn.Conv2d(in_channels, 1, 3, padding=1)

        # Learnable scales per FPN level (initialized to 1.0)
        if use_scale:
            self.scales = nn.ModuleList([Scale(1.0) for _ in range(num_levels)])

        # Initialize weights
        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob: float):
        """Initialize weights with proper bias for focal loss."""
        for modules in [self.cls_tower, self.reg_tower]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

        # Classification bias for focal loss
        # This ensures initial predictions are close to prior_prob
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

        # Regression and centerness
        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.bbox_pred.bias, 0)

        nn.init.normal_(self.centerness.weight, mean=0, std=0.01)
        nn.init.constant_(self.centerness.bias, 0)

        nn.init.normal_(self.iou_quality.weight, mean=0, std=0.01)
        nn.init.constant_(self.iou_quality.bias, 0)

    def forward(
        self,
        features: List[torch.Tensor],
        strides: Optional[List[int]] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through FCOS head.

        Args:
            features: List of FPN features [P3, P4, P5, ...] with decreasing resolution
            strides: List of strides for each level (for regression normalization)

        Returns:
            cls_outputs: List of [B, num_classes, H, W] classification logits
            reg_outputs: List of [B, 4, H, W] LTRB regression (positive)
            ctr_outputs: List of [B, 1, H, W] centerness logits
            iouq_outputs: List of [B, 1, H, W] IoU-quality logits
        """
        cls_outputs = []
        reg_outputs = []
        ctr_outputs = []
        iouq_outputs = []

        for level, feat in enumerate(features):
            # Classification tower
            cls_feat = self.cls_tower(feat)

            # Regression tower
            reg_feat = self.reg_tower(feat)

            # Classification prediction
            cls_logits = self.cls_logits(cls_feat)

            # Regression prediction (LTRB must be positive)
            bbox_pred = self.bbox_pred(reg_feat)
            if self.use_scale and level < len(self.scales):
                bbox_pred = self.scales[level](bbox_pred)
            bbox_pred = F.relu(bbox_pred)  # LTRB >= 0

            # Optionally normalize by stride (only during inference, not training)
            # Note: This is disabled by default (norm_on_bbox=False) because
            # normalization should be handled in target encoding, not here
            if self.norm_on_bbox and strides is not None and level < len(strides):
                bbox_pred = bbox_pred * strides[level]

            # Centerness from regression tower (FCOS Section 3.2)
            ctr_logits = self.centerness(reg_feat)
            iouq_logits = self.iou_quality(reg_feat)

            cls_outputs.append(cls_logits)
            reg_outputs.append(bbox_pred)
            ctr_outputs.append(ctr_logits)
            iouq_outputs.append(iouq_logits)

        return cls_outputs, reg_outputs, ctr_outputs, iouq_outputs

    def forward_single(
        self,
        feat: torch.Tensor,
        level: int = 0,
        stride: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single FPN level.

        Args:
            feat: Single FPN feature [B, C, H, W]
            level: FPN level index
            stride: Stride for this level

        Returns:
            cls_logits: [B, num_classes, H, W]
            bbox_pred: [B, 4, H, W]
            ctr_logits: [B, 1, H, W]
            iouq_logits: [B, 1, H, W]
        """
        cls_feat = self.cls_tower(feat)
        reg_feat = self.reg_tower(feat)

        cls_logits = self.cls_logits(cls_feat)

        bbox_pred = self.bbox_pred(reg_feat)
        if self.use_scale and level < len(self.scales):
            bbox_pred = self.scales[level](bbox_pred)
        bbox_pred = F.relu(bbox_pred)

        if self.norm_on_bbox and stride is not None:
            bbox_pred = bbox_pred * stride

        # Centerness from regression features
        ctr_logits = self.centerness(reg_feat)
        iouq_logits = self.iou_quality(reg_feat)

        return cls_logits, bbox_pred, ctr_logits, iouq_logits


def ltrb_to_xyxy(
    ltrb: torch.Tensor,
    points: torch.Tensor
) -> torch.Tensor:
    """
    Convert LTRB distances to xyxy boxes.

    Args:
        ltrb: [N, 4] or [B, 4, H, W] LTRB distances
        points: [N, 2] or [H*W, 2] center points (x, y)

    Returns:
        boxes: [N, 4] or [B, H, W, 4] xyxy boxes
    """
    if ltrb.dim() == 2:
        # [N, 4] format
        x1 = points[:, 0] - ltrb[:, 0]
        y1 = points[:, 1] - ltrb[:, 1]
        x2 = points[:, 0] + ltrb[:, 2]
        y2 = points[:, 1] + ltrb[:, 3]
        return torch.stack([x1, y1, x2, y2], dim=1)
    else:
        # [B, 4, H, W] format
        B, _, H, W = ltrb.shape
        # points should be [H*W, 2]
        points = points.view(H, W, 2)
        x1 = points[..., 0] - ltrb[:, 0]
        y1 = points[..., 1] - ltrb[:, 1]
        x2 = points[..., 0] + ltrb[:, 2]
        y2 = points[..., 1] + ltrb[:, 3]
        return torch.stack([x1, y1, x2, y2], dim=1).permute(0, 2, 3, 1)


def generate_points(
    feature_sizes: List[Tuple[int, int]],
    strides: List[int],
    device: torch.device
) -> List[torch.Tensor]:
    """
    Generate grid points for each FPN level.

    Args:
        feature_sizes: List of (H, W) for each level
        strides: List of strides
        device: Target device

    Returns:
        points: List of [H*W, 2] tensors with (x, y) coordinates
    """
    points = []
    for (h, w), stride in zip(feature_sizes, strides):
        # Generate grid
        y = torch.arange(0, h, device=device) * stride + stride // 2
        x = torch.arange(0, w, device=device) * stride + stride // 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        # Flatten to [H*W, 2]
        pts = torch.stack([xx.flatten(), yy.flatten()], dim=1).float()
        points.append(pts)

    return points


def test_fcos_head():
    """Test the FCOS head."""
    print("Testing FCOSHead...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create head
    head = FCOSHead(
        in_channels=128,
        num_classes=1,
        num_convs=4
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in head.parameters())
    print(f"Parameters: {n_params:,}")

    # Create dummy FPN features
    # Backbone-matching strides [2, 4, 8] with corresponding feature sizes
    batch_size = 4
    features = [
        torch.randn(batch_size, 128, 320, 320, device=device),  # P2: stride 2
        torch.randn(batch_size, 128, 160, 160, device=device),  # P3: stride 4
        torch.randn(batch_size, 128, 80, 80, device=device),    # P4: stride 8
    ]
    strides = [2, 4, 8]

    # Forward pass
    cls_outputs, reg_outputs, ctr_outputs, iouq_outputs = head(features, strides)

    print(f"\nOutputs:")
    for i, (cls, reg, ctr, iouq) in enumerate(zip(cls_outputs, reg_outputs, ctr_outputs, iouq_outputs)):
        print(f"  P{i+3}: cls {cls.shape}, reg {reg.shape}, ctr {ctr.shape}, iouq {iouq.shape}")

    # Verify shapes
    for i, (cls, reg, ctr, iouq) in enumerate(zip(cls_outputs, reg_outputs, ctr_outputs, iouq_outputs)):
        H, W = features[i].shape[2:]
        assert cls.shape == (batch_size, 1, H, W), f"cls shape mismatch"
        assert reg.shape == (batch_size, 4, H, W), f"reg shape mismatch"
        assert ctr.shape == (batch_size, 1, H, W), f"ctr shape mismatch"
        assert iouq.shape == (batch_size, 1, H, W), f"iouq shape mismatch"

    # Verify regression is positive
    for reg in reg_outputs:
        assert (reg >= 0).all(), "Regression should be non-negative"

    # Test point generation
    # Feature sizes matching strides [2, 4, 8]
    feature_sizes = [(320, 320), (160, 160), (80, 80)]
    points = generate_points(feature_sizes, strides, device)

    print(f"\nPoints:")
    for i, pts in enumerate(points):
        print(f"  P{i+2}: {pts.shape}, range x=[{pts[:, 0].min():.0f}, {pts[:, 0].max():.0f}], y=[{pts[:, 1].min():.0f}, {pts[:, 1].max():.0f}]")

    # Test LTRB to xyxy conversion
    sample_ltrb = torch.tensor([[10, 20, 30, 40]], device=device, dtype=torch.float32)
    sample_points = torch.tensor([[100, 100]], device=device, dtype=torch.float32)
    boxes = ltrb_to_xyxy(sample_ltrb, sample_points)
    print(f"\nLTRB conversion test:")
    print(f"  LTRB: {sample_ltrb[0].tolist()}")
    print(f"  Point: {sample_points[0].tolist()}")
    print(f"  Box (xyxy): {boxes[0].tolist()}")

    # Expected: [100-10, 100-20, 100+30, 100+40] = [90, 80, 130, 140]
    expected = torch.tensor([[90, 80, 130, 140]], device=device, dtype=torch.float32)
    assert torch.allclose(boxes, expected), f"Expected {expected}, got {boxes}"

    print("\nSUCCESS: FCOSHead working!")


if __name__ == '__main__':
    test_fcos_head()
