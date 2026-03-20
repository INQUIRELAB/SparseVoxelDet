#!/usr/bin/env python3
"""
Mosaic Augmentation for Sparse Event Data.

Places 4 sparse event samples in quadrants of a single image, merging
coordinates and labels. This is the event-camera equivalent of YOLO's
mosaic augmentation (~2-4% mAP improvement).

Works directly on sparse coordinates (no dense conversion needed).
"""
import numpy as np
from typing import Tuple, List, Dict


def sparse_mosaic(
    samples: List[Dict],
    target_size: Tuple[int, int] = (640, 640),
    time_bins: int = 15,
) -> Dict:
    """
    Create a mosaic from 4 sparse event samples.

    Each sample is placed in one quadrant with coordinates offset accordingly.
    Boxes are clipped to their quadrant boundaries.

    Args:
        samples: List of 4 sample dicts, each with:
            - coords: [N, 3] int32 (t, y, x)
            - feats: [N, 2] float32
            - boxes: [M, 4] float32 (x1, y1, x2, y2)
            - labels: [M] int64
        target_size: Output (H, W)
        time_bins: Number of temporal bins

    Returns:
        Merged sample dict with same format.
    """
    assert len(samples) == 4, f"Mosaic needs exactly 4 samples, got {len(samples)}"

    H, W = target_size
    half_h, half_w = H // 2, W // 2

    # Quadrant offsets: (y_offset, x_offset, y_end, x_end)
    quadrants = [
        (0, 0, half_h, half_w),          # top-left
        (0, half_w, half_h, W),           # top-right
        (half_h, 0, H, half_w),           # bottom-left
        (half_h, half_w, H, W),           # bottom-right
    ]

    all_coords = []
    all_feats = []
    all_boxes = []
    all_labels = []

    for sample, (y_off, x_off, y_end, x_end) in zip(samples, quadrants):
        coords = sample['coords'].copy() if isinstance(sample['coords'], np.ndarray) else sample['coords'].numpy().copy()
        feats = sample['feats'].copy() if isinstance(sample['feats'], np.ndarray) else sample['feats'].numpy().copy()
        boxes = sample['boxes'].copy() if isinstance(sample['boxes'], np.ndarray) else sample['boxes'].numpy().copy()
        labels = sample['labels'].copy() if isinstance(sample['labels'], np.ndarray) else sample['labels'].numpy().copy()

        qh = y_end - y_off
        qw = x_end - x_off

        # Scale coordinates to fit quadrant
        if len(coords) > 0:
            # Scale from full target_size to quadrant size
            coords[:, 1] = (coords[:, 1].astype(np.float64) / H * qh).astype(np.int32) + y_off
            coords[:, 2] = (coords[:, 2].astype(np.float64) / W * qw).astype(np.int32) + x_off

            # Clip to quadrant (shouldn't be needed but safety)
            coords[:, 1] = np.clip(coords[:, 1], y_off, y_end - 1)
            coords[:, 2] = np.clip(coords[:, 2], x_off, x_end - 1)
            # Time stays unchanged
            coords[:, 0] = np.clip(coords[:, 0], 0, time_bins - 1)

            all_coords.append(coords)
            all_feats.append(feats)

        # Scale and clip boxes to quadrant
        if len(boxes) > 0:
            # Scale boxes from full image to quadrant
            boxes[:, 0] = boxes[:, 0] / W * qw + x_off  # x1
            boxes[:, 1] = boxes[:, 1] / H * qh + y_off  # y1
            boxes[:, 2] = boxes[:, 2] / W * qw + x_off  # x2
            boxes[:, 3] = boxes[:, 3] / H * qh + y_off  # y2

            # Clip to quadrant
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], x_off, x_end)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], y_off, y_end)

            # Remove degenerate boxes
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            keep = (widths > 1) & (heights > 1)

            if keep.any():
                all_boxes.append(boxes[keep])
                all_labels.append(labels[keep])

    # Concatenate
    if all_coords:
        merged_coords = np.concatenate(all_coords, axis=0).astype(np.int32)
        merged_feats = np.concatenate(all_feats, axis=0).astype(np.float32)
    else:
        merged_coords = np.zeros((1, 3), dtype=np.int32)
        feat_dim = 2
        for sample in samples:
            sample_feats = sample.get('feats')
            if sample_feats is not None and getattr(sample_feats, 'ndim', 0) == 2:
                feat_dim = int(sample_feats.shape[1])
                break
        merged_feats = np.zeros((1, feat_dim), dtype=np.float32)

    if all_boxes:
        merged_boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
        merged_labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    else:
        merged_boxes = np.zeros((0, 4), dtype=np.float32)
        merged_labels = np.zeros((0,), dtype=np.int64)

    return {
        'coords': merged_coords,
        'feats': merged_feats,
        'boxes': merged_boxes,
        'labels': merged_labels,
    }
