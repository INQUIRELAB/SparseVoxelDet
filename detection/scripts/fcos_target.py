#!/usr/bin/env python3
"""
FCOS Target Assignment.

Assigns ground truth boxes to feature map locations based on:
1. Location falls inside a GT box
2. Box size falls within the scale range for this FPN level
3. If multiple boxes, assign to smallest (handles overlapping drones)

Key concepts:
- Positive: location (x, y) is inside a GT box AND box size matches scale
- Centerness: sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))
- LTRB: distances from location to box edges (left, top, right, bottom)

Reference: Tian et al., "FCOS: Fully Convolutional One-Stage Object Detection", ICCV 2019
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math


# Regress ranges per FPN level (min_size, max_size)
# Objects outside this range are not assigned to this level
# Adjusted for backbone strides [4, 8, 16] with stem_stride=(1,2,2)
DEFAULT_REGRESS_RANGES = (
    (-1, 32),      # P3: small objects (stride 4, 74% FRED drones <32px)
    (32, 64),      # P4: medium objects (stride 8)
    (64, 256),     # P5: large objects (stride 16)
)


def generate_points(
    feature_sizes: List[Tuple[int, int]],
    strides: List[int],
    device: torch.device
) -> List[torch.Tensor]:
    """
    Generate grid points for all FPN levels.

    Args:
        feature_sizes: List of (H, W) for each FPN level
        strides: List of strides for each level
        device: Device to create tensors on

    Returns:
        List of [H*W, 2] tensors containing (x, y) coordinates for each level
    """
    points = []
    for (h, w), stride in zip(feature_sizes, strides):
        # Generate grid points at center of each cell
        y = torch.arange(0, h, device=device) * stride + stride // 2
        x = torch.arange(0, w, device=device) * stride + stride // 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        pts = torch.stack([xx.flatten(), yy.flatten()], dim=1).float()
        points.append(pts)
    return points


def compute_centerness(ltrb: torch.Tensor) -> torch.Tensor:
    """
    Compute centerness from LTRB regression targets.

    Centerness = sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))

    This penalizes predictions far from the object center.
    Centerness is 1 at center, decreasing towards edges.

    Args:
        ltrb: [N, 4] tensor of (left, top, right, bottom) distances

    Returns:
        centerness: [N] tensor in [0, 1]
    """
    left = ltrb[:, 0]
    top = ltrb[:, 1]
    right = ltrb[:, 2]
    bottom = ltrb[:, 3]

    # Avoid division by zero
    lr_min = torch.min(left, right)
    lr_max = torch.max(left, right).clamp(min=1e-6)
    tb_min = torch.min(top, bottom)
    tb_max = torch.max(top, bottom).clamp(min=1e-6)

    centerness = torch.sqrt((lr_min / lr_max) * (tb_min / tb_max))
    return centerness


def compute_ltrb_targets(
    points: torch.Tensor,
    gt_boxes: torch.Tensor
) -> torch.Tensor:
    """
    Compute LTRB distances from points to box edges.

    Args:
        points: [N, 2] grid points (x, y)
        gt_boxes: [M, 4] ground truth boxes (x1, y1, x2, y2)

    Returns:
        ltrb: [N, M, 4] distances (left, top, right, bottom) for each point-box pair
    """
    N = points.shape[0]
    M = gt_boxes.shape[0]

    # Expand for broadcasting: [N, 1, 2] and [1, M, 4]
    points = points.unsqueeze(1)  # [N, 1, 2]
    gt_boxes = gt_boxes.unsqueeze(0)  # [1, M, 4]

    # Compute distances
    left = points[..., 0] - gt_boxes[..., 0]    # x - x1
    top = points[..., 1] - gt_boxes[..., 1]     # y - y1
    right = gt_boxes[..., 2] - points[..., 0]   # x2 - x
    bottom = gt_boxes[..., 3] - points[..., 1]  # y2 - y

    ltrb = torch.stack([left, top, right, bottom], dim=-1)  # [N, M, 4]
    return ltrb


def assign_targets_per_level(
    points: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    regress_range: Tuple[float, float],
    stride: int,
    center_sampling: bool = True,
    center_sampling_radius: float = 1.5,
    num_classes: int = 1,
    norm_on_bbox: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assign targets for a single FPN level.

    Args:
        points: [N, 2] grid points (x, y) for this level
        gt_boxes: [M, 4] ground truth boxes (x1, y1, x2, y2)
        gt_labels: [M] ground truth class labels
        regress_range: (min_size, max_size) for this level
        stride: Stride of this level
        center_sampling: Use center sampling (restrict positives to box centers)
        center_sampling_radius: Radius in stride units for center sampling
        num_classes: Number of classes
        norm_on_bbox: If True, normalize LTRB targets by stride (standard FCOS for larger strides)

    Returns:
        labels: [N] class labels (0 = background, 1+ = class)
        ltrb_targets: [N, 4] regression targets
        centerness_targets: [N] centerness values
        pos_mask: [N] boolean mask of positive locations
    """
    N = points.shape[0]
    M = gt_boxes.shape[0]
    device = points.device

    # Handle empty GT case
    if M == 0:
        labels = torch.zeros(N, dtype=torch.long, device=device)
        ltrb_targets = torch.zeros(N, 4, dtype=torch.float32, device=device)
        centerness_targets = torch.zeros(N, dtype=torch.float32, device=device)
        pos_mask = torch.zeros(N, dtype=torch.bool, device=device)
        return labels, ltrb_targets, centerness_targets, pos_mask

    # Compute LTRB for all point-box pairs: [N, M, 4]
    ltrb = compute_ltrb_targets(points, gt_boxes)

    # Check if point is inside box: all LTRB > 0
    inside_box = ltrb.min(dim=-1).values > 0  # [N, M]

    # Center sampling: restrict positives to box centers
    if center_sampling:
        # Compute box centers
        cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2  # [M]
        cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2  # [M]

        # Define center region: point within radius*stride of center
        radius = center_sampling_radius * stride
        x_dist = torch.abs(points[:, 0:1] - cx.unsqueeze(0))  # [N, M]
        y_dist = torch.abs(points[:, 1:2] - cy.unsqueeze(0))  # [N, M]
        in_center = (x_dist <= radius) & (y_dist <= radius)

        # Must be both inside box AND in center region
        inside_box = inside_box & in_center

    # Check if box size is within regress range for this level
    # Size = max(l, t, r, b)
    max_ltrb = ltrb.max(dim=-1).values  # [N, M]
    min_size, max_size = regress_range
    in_range = (max_ltrb >= min_size) & (max_ltrb < max_size)  # [N, M]

    # Valid assignment: inside box AND in size range
    valid = inside_box & in_range  # [N, M]

    # Compute box areas for disambiguation: [M]
    areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    # For each point, find the smallest valid box (or -1 if none)
    # Set invalid boxes to INF area
    areas_expanded = areas.unsqueeze(0).expand(N, M)  # [N, M]
    areas_expanded = torch.where(valid, areas_expanded, torch.tensor(float('inf'), device=device))

    # Find minimum area box per point
    min_areas, assigned_gt_idx = areas_expanded.min(dim=1)  # [N]

    # Mask for positive locations (has valid assignment)
    pos_mask = min_areas < float('inf')

    # Get assigned LTRB targets
    # Gather from [N, M, 4] using indices [N]
    batch_idx = torch.arange(N, device=device)
    ltrb_targets = ltrb[batch_idx, assigned_gt_idx]  # [N, 4]

    # Get assigned labels
    labels = gt_labels[assigned_gt_idx] + 1  # +1 because 0 is background
    labels = torch.where(pos_mask, labels, torch.zeros_like(labels))

    # Compute centerness for positive locations
    centerness_targets = compute_centerness(ltrb_targets)
    centerness_targets = torch.where(pos_mask, centerness_targets, torch.zeros_like(centerness_targets))

    # Set LTRB to 0 for negative locations (for loss masking)
    ltrb_targets = torch.where(pos_mask.unsqueeze(1), ltrb_targets, torch.zeros_like(ltrb_targets))

    # Normalize LTRB targets by stride (standard FCOS for strides [4,8,16])
    # The FCOS head multiplies predictions by stride, so targets must be divided
    if norm_on_bbox and pos_mask.any():
        ltrb_targets = torch.where(
            pos_mask.unsqueeze(1),
            ltrb_targets / stride,
            ltrb_targets
        )

    return labels, ltrb_targets, centerness_targets, pos_mask


def assign_targets_batch(
    points_per_level: List[torch.Tensor],
    gt_boxes_batch: List[torch.Tensor],
    gt_labels_batch: List[torch.Tensor],
    strides: List[int],
    regress_ranges: Tuple[Tuple[float, float], ...] = DEFAULT_REGRESS_RANGES,
    center_sampling: bool = True,
    center_sampling_radius: float = 1.5,
    num_classes: int = 1,
    norm_on_bbox: bool = False
) -> Dict[str, List[torch.Tensor]]:
    """
    Assign targets for a batch of images.

    Args:
        points_per_level: List of [N_l, 2] grid points per FPN level
        gt_boxes_batch: List of [M_b, 4] GT boxes per image
        gt_labels_batch: List of [M_b] GT labels per image
        strides: List of strides per level
        regress_ranges: Tuple of (min, max) per level
        center_sampling: Use center sampling
        center_sampling_radius: Radius for center sampling
        num_classes: Number of classes
        norm_on_bbox: If True, normalize LTRB targets by stride

    Returns:
        Dictionary with:
        - labels: List of [N_l] per level, concatenated across images
        - ltrb_targets: List of [N_l, 4] per level
        - centerness_targets: List of [N_l] per level
        - pos_masks: List of [N_l] per level
    """
    num_levels = len(points_per_level)
    batch_size = len(gt_boxes_batch)

    # Initialize outputs per level
    all_labels = [[] for _ in range(num_levels)]
    all_ltrb = [[] for _ in range(num_levels)]
    all_centerness = [[] for _ in range(num_levels)]
    all_pos_masks = [[] for _ in range(num_levels)]

    # Process each image
    for img_idx in range(batch_size):
        gt_boxes = gt_boxes_batch[img_idx]
        gt_labels = gt_labels_batch[img_idx]

        # Process each level
        for level_idx, (points, stride) in enumerate(zip(points_per_level, strides)):
            regress_range = regress_ranges[level_idx] if level_idx < len(regress_ranges) else regress_ranges[-1]

            labels, ltrb_targets, centerness_targets, pos_mask = assign_targets_per_level(
                points=points,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                regress_range=regress_range,
                stride=stride,
                center_sampling=center_sampling,
                center_sampling_radius=center_sampling_radius,
                num_classes=num_classes,
                norm_on_bbox=norm_on_bbox
            )

            all_labels[level_idx].append(labels)
            all_ltrb[level_idx].append(ltrb_targets)
            all_centerness[level_idx].append(centerness_targets)
            all_pos_masks[level_idx].append(pos_mask)

    # Stack per level
    result = {
        'labels': [torch.stack(ll, dim=0) for ll in all_labels],  # [B, N_l] per level
        'ltrb_targets': [torch.stack(ll, dim=0) for ll in all_ltrb],  # [B, N_l, 4] per level
        'centerness_targets': [torch.stack(ll, dim=0) for ll in all_centerness],  # [B, N_l] per level
        'pos_masks': [torch.stack(ll, dim=0) for ll in all_pos_masks],  # [B, N_l] per level
    }

    return result


def flatten_targets(targets: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Flatten multi-level targets to single tensors.

    Args:
        targets: Dictionary with lists per level

    Returns:
        Dictionary with flattened tensors
    """
    return {
        'labels': torch.cat([t.flatten() for t in targets['labels']]),
        'ltrb_targets': torch.cat([t.view(-1, 4) for t in targets['ltrb_targets']]),
        'centerness_targets': torch.cat([t.flatten() for t in targets['centerness_targets']]),
        'pos_masks': torch.cat([t.flatten() for t in targets['pos_masks']]),
    }


def test_fcos_target():
    """Test target assignment."""
    print("Testing FCOS target assignment...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Simulate 3 FPN levels
    # Backbone-matching strides [2, 4, 8] with corresponding feature sizes
    strides = [2, 4, 8]
    feature_sizes = [(320, 320), (160, 160), (80, 80)]  # For 640x640 input

    # Generate grid points for each level
    points_per_level = []
    for (h, w), stride in zip(feature_sizes, strides):
        y = torch.arange(0, h, device=device) * stride + stride // 2
        x = torch.arange(0, w, device=device) * stride + stride // 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        points = torch.stack([xx.flatten(), yy.flatten()], dim=1).float()
        points_per_level.append(points)
        print(f"Level stride={stride}: {points.shape[0]} points")

    # Create some GT boxes (batch of 2 images)
    # Image 1: 2 drones, Image 2: 1 drone
    gt_boxes_batch = [
        torch.tensor([
            [100, 100, 150, 150],  # 50x50 small drone
            [300, 200, 360, 280],  # 60x80 medium drone
        ], device=device, dtype=torch.float32),
        torch.tensor([
            [400, 300, 480, 400],  # 80x100 medium drone
        ], device=device, dtype=torch.float32),
    ]
    gt_labels_batch = [
        torch.tensor([0, 0], device=device, dtype=torch.long),  # class 0 (drone)
        torch.tensor([0], device=device, dtype=torch.long),
    ]

    # Assign targets
    # Regress ranges matching the finer strides [2, 4, 8]
    regress_ranges = ((-1, 16), (16, 32), (32, 64))
    targets = assign_targets_batch(
        points_per_level=points_per_level,
        gt_boxes_batch=gt_boxes_batch,
        gt_labels_batch=gt_labels_batch,
        strides=strides,
        regress_ranges=regress_ranges,
        center_sampling=True,
        center_sampling_radius=1.5,
        num_classes=1
    )

    # Print statistics
    print(f"\nTarget shapes:")
    for level_idx in range(len(strides)):
        labels = targets['labels'][level_idx]
        pos_mask = targets['pos_masks'][level_idx]
        n_pos = pos_mask.sum().item()
        n_total = pos_mask.numel()
        print(f"  Level {level_idx} (stride={strides[level_idx]}): {n_pos}/{n_total} positive ({100*n_pos/n_total:.2f}%)")

    # Verify positive locations
    for level_idx in range(len(strides)):
        pos_mask = targets['pos_masks'][level_idx]
        labels = targets['labels'][level_idx]
        ltrb = targets['ltrb_targets'][level_idx]
        centerness = targets['centerness_targets'][level_idx]

        if pos_mask.any():
            pos_ltrb = ltrb[pos_mask]
            pos_ctr = centerness[pos_mask]
            print(f"\n  Level {level_idx} positive samples:")
            print(f"    LTRB range: [{pos_ltrb.min():.1f}, {pos_ltrb.max():.1f}]")
            print(f"    Centerness range: [{pos_ctr.min():.3f}, {pos_ctr.max():.3f}]")

            # Verify all LTRB are positive
            assert (pos_ltrb > 0).all(), "LTRB should be positive for positive samples"
            # Verify centerness is in [0, 1]
            assert (pos_ctr >= 0).all() and (pos_ctr <= 1).all(), "Centerness should be in [0, 1]"

    # Test flatten
    flat_targets = flatten_targets(targets)
    print(f"\nFlattened shapes:")
    print(f"  labels: {flat_targets['labels'].shape}")
    print(f"  ltrb_targets: {flat_targets['ltrb_targets'].shape}")
    print(f"  centerness_targets: {flat_targets['centerness_targets'].shape}")
    print(f"  pos_masks: {flat_targets['pos_masks'].shape}")

    total_pos = flat_targets['pos_masks'].sum().item()
    print(f"\nTotal positive samples: {total_pos}")

    print("\nSUCCESS: FCOS target assignment working!")


if __name__ == '__main__':
    test_fcos_target()
