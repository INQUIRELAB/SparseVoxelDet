#!/usr/bin/env python3
"""
Detection metrics: mAP calculation following PASCAL VOC and COCO protocols.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch


@dataclass
class DetectionMetrics:
    """Container for detection evaluation metrics."""
    mAP_50: float          # mAP at IoU=0.50 (PASCAL VOC metric)
    mAP_50_95: float       # mAP at IoU=0.50:0.95:0.05 (COCO metric)
    precision: float       # Precision at best F1 threshold
    recall: float          # Recall at best F1 threshold
    f1: float              # Best F1 score
    ap_per_class: Dict[int, float]  # AP per class at IoU=0.50
    total_predictions: int
    total_ground_truths: int
    true_positives: int
    false_positives: int
    false_negatives: int


def box_iou_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of boxes using numpy (FAST).

    Args:
        boxes1: [N, 4] array (x1, y1, x2, y2)
        boxes2: [M, 4] array (x1, y1, x2, y2)

    Returns:
        [N, M] IoU matrix
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    # Expand dimensions for broadcasting: [N, 1, 4] and [1, M, 4]
    boxes1 = boxes1[:, np.newaxis, :]  # [N, 1, 4]
    boxes2 = boxes2[np.newaxis, :, :]  # [1, M, 4]

    # Intersection
    inter_x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Areas
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # Union
    union_area = area1 + area2 - inter_area

    # IoU
    iou = inter_area / (union_area + 1e-6)
    return iou.squeeze()  # Remove extra dimensions if needed


def yolo_to_xyxy_numpy(labels: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert YOLO format labels to x1y1x2y2 format using numpy.

    Args:
        labels: [M, 5] array (cls, cx, cy, w, h) normalized to [0, 1]
        img_size: (height, width) of the image

    Returns:
        [M, 5] array (x1, y1, x2, y2, cls) in pixel coordinates
    """
    if len(labels) == 0:
        return np.zeros((0, 5))

    h, w = img_size
    cls = labels[:, 0]
    cx = labels[:, 1] * w
    cy = labels[:, 2] * h
    bw = labels[:, 3] * w
    bh = labels[:, 4] * h

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    return np.stack([x1, y1, x2, y2, cls], axis=1)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision using all-point interpolation (COCO style).
    """
    if len(recalls) == 0:
        return 0.0

    # Add sentinel values
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Find points where recall changes
    change_points = np.where(recalls[1:] != recalls[:-1])[0] + 1

    # Sum areas of rectangles
    ap = np.sum((recalls[change_points] - recalls[change_points - 1]) * precisions[change_points])

    return float(ap)


def compute_map_at_iou_range(
    predictions: List[dict],
    ground_truths: List[dict],
    gt_by_image: dict,
    iou_thresholds: np.ndarray = None,
    num_classes: int = 1,
) -> Dict[str, float]:
    """
    Compute AP at multiple IoU thresholds (COCO-style mAP@50:95).

    Args:
        predictions: List of prediction dicts with 'conf', 'box', 'cls', 'img_idx'
        ground_truths: List of GT dicts
        gt_by_image: Pre-grouped GTs by image index
        iou_thresholds: IoU thresholds (default: 0.50:0.95:0.05)
        num_classes: Number of classes

    Returns:
        Dict with:
            - ap_at_X: AP at each threshold
            - mAP_50: mAP at IoU=0.50
            - mAP_75: mAP at IoU=0.75
            - mAP_50_95: Average across all thresholds
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.50, 1.00, 0.05)

    # Sort predictions by confidence
    sorted_preds = sorted(predictions, key=lambda x: x['conf'], reverse=True)

    results = {}

    for iou_thresh in iou_thresholds:
        ap_per_class = {}

        for cls_id in range(num_classes):
            # Filter predictions for this class
            class_preds = [p for p in sorted_preds if p['cls'] == cls_id]

            # Count GTs for this class
            n_gt = sum(1 for g in ground_truths if g['cls'] == cls_id)

            if n_gt == 0:
                ap_per_class[cls_id] = 1.0 if len(class_preds) == 0 else 0.0
                continue
            if len(class_preds) == 0:
                ap_per_class[cls_id] = 0.0
                continue

            # Track matches
            matched_gt = {}
            tp = np.zeros(len(class_preds))
            fp = np.zeros(len(class_preds))

            for pred_idx, pred in enumerate(class_preds):
                img_idx = pred['img_idx']
                pred_box = pred['box'].reshape(1, 4)

                if img_idx not in gt_by_image:
                    fp[pred_idx] = 1
                    continue

                gt_data = gt_by_image[img_idx]
                gt_boxes = gt_data['boxes']
                gt_classes = gt_data['classes']

                class_mask = gt_classes == cls_id
                if not class_mask.any():
                    fp[pred_idx] = 1
                    continue

                gt_boxes_class = gt_boxes[class_mask]
                gt_indices = np.where(class_mask)[0]

                ious = box_iou_numpy(pred_box, gt_boxes_class).flatten()

                if len(ious) > 0:
                    # COCO-style: find best *unmatched* GT above threshold
                    sorted_gt = np.argsort(ious)[::-1]
                    matched = False
                    for gi in sorted_gt:
                        if ious[gi] < iou_thresh:
                            break
                        match_key = (img_idx, gt_indices[gi], iou_thresh)
                        if match_key not in matched_gt:
                            tp[pred_idx] = 1
                            matched_gt[match_key] = True
                            matched = True
                            break
                    if not matched:
                        fp[pred_idx] = 1
                else:
                    fp[pred_idx] = 1

            # Compute PR curve
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recalls = tp_cumsum / n_gt
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

            ap_per_class[cls_id] = compute_ap(recalls, precisions)

        # Mean AP for this threshold
        mean_ap = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0
        results[f'ap_at_{iou_thresh:.2f}'] = mean_ap

    # Summary metrics
    results['mAP_50'] = results.get('ap_at_0.50', 0.0)
    results['mAP_75'] = results.get('ap_at_0.75', 0.0)

    # mAP@50:95 (COCO primary metric)
    ap_values = [results[f'ap_at_{iou:.2f}'] for iou in iou_thresholds]
    results['mAP_50_95'] = float(np.mean(ap_values))

    return results


class MAPCalculator:
    """
    OPTIMIZED mAP calculator.

    Key optimizations:
    1. Store everything as numpy arrays (not torch tensors)
    2. Pre-group ground truths by image during update()
    3. Vectorized IoU computation
    4. Batch processing instead of per-prediction loops
    """

    def __init__(
        self,
        num_classes: int = 1,
        img_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.001,
        max_predictions_per_image: int = 100,  # Limit predictions to avoid slowdown
    ):
        self.num_classes = num_classes
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.max_predictions_per_image = max_predictions_per_image
        self.coco_thresholds = np.array([0.5 + 0.05 * i for i in range(10)])
        self.reset()

    def reset(self):
        """Clear accumulated data."""
        # Store as lists, convert to numpy at compute time
        self.predictions = []  # List of dicts
        self.ground_truths = []  # List of dicts
        self.gt_by_image = {}  # Pre-grouped GTs
        self.img_counter = 0

    def update(
        self,
        predictions: List[torch.Tensor],
        labels: List[torch.Tensor],
    ):
        """
        Add batch predictions and labels.
        Converts to numpy immediately for faster processing later.
        """
        for pred, label in zip(predictions, labels):
            img_idx = self.img_counter

            # Store predictions as numpy (limited to top N by confidence)
            if len(pred) > 0:
                pred_np = pred.cpu().numpy()
                mask = pred_np[:, 4] >= self.conf_threshold
                pred_np = pred_np[mask]

                # Keep only top predictions by confidence to avoid slowdown
                if len(pred_np) > self.max_predictions_per_image:
                    top_indices = np.argsort(pred_np[:, 4])[::-1][:self.max_predictions_per_image]
                    pred_np = pred_np[top_indices]

                for i in range(len(pred_np)):
                    self.predictions.append({
                        'conf': pred_np[i, 4],
                        'box': pred_np[i, :4],
                        'cls': int(pred_np[i, 5]),
                        'img_idx': img_idx,
                    })

            # Store ground truths as numpy and pre-group by image
            if len(label) > 0:
                label_np = label.cpu().numpy()
                gt_xyxy = yolo_to_xyxy_numpy(label_np, self.img_size)

                if img_idx not in self.gt_by_image:
                    self.gt_by_image[img_idx] = {
                        'boxes': [],
                        'classes': [],
                        'matched': [],
                    }

                for i in range(len(gt_xyxy)):
                    self.ground_truths.append({
                        'box': gt_xyxy[i, :4],
                        'cls': int(gt_xyxy[i, 4]),
                        'img_idx': img_idx,
                    })
                    self.gt_by_image[img_idx]['boxes'].append(gt_xyxy[i, :4])
                    self.gt_by_image[img_idx]['classes'].append(int(gt_xyxy[i, 4]))

            self.img_counter += 1

        # Convert GT lists to numpy arrays for faster access
        for img_idx in self.gt_by_image:
            if isinstance(self.gt_by_image[img_idx]['boxes'], list):
                self.gt_by_image[img_idx]['boxes'] = np.array(self.gt_by_image[img_idx]['boxes'])
                self.gt_by_image[img_idx]['classes'] = np.array(self.gt_by_image[img_idx]['classes'])

    def compute(self) -> DetectionMetrics:
        """
        Compute all metrics (OPTIMIZED).
        """
        n_preds = len(self.predictions)
        n_gts = len(self.ground_truths)

        if n_preds == 0 or n_gts == 0:
            return DetectionMetrics(
                mAP_50=0.0, mAP_50_95=0.0,
                precision=0.0, recall=0.0, f1=0.0,
                ap_per_class={},
                total_predictions=n_preds, total_ground_truths=n_gts,
                true_positives=0, false_positives=n_preds, false_negatives=n_gts,
            )

        # Sort predictions by confidence (descending) - do this ONCE
        sorted_preds = sorted(self.predictions, key=lambda x: x['conf'], reverse=True)

        # Compute AP at IoU=0.50
        ap_50_per_class = {}
        for cls_id in range(self.num_classes):
            ap_50_per_class[cls_id] = self._compute_ap_for_class_fast(sorted_preds, cls_id, 0.50)

        mAP_50 = float(np.mean(list(ap_50_per_class.values()))) if ap_50_per_class else 0.0

        # Compute full mAP@50:95 using the optimized function
        iou_range_metrics = compute_map_at_iou_range(
            self.predictions,
            self.ground_truths,
            self.gt_by_image,
            num_classes=self.num_classes,
        )
        mAP_50_95 = iou_range_metrics['mAP_50_95']

        # Compute precision, recall, F1
        tp, fp, fn, precision, recall, f1 = self._compute_pr_f1_fast(sorted_preds, 0.50)

        return DetectionMetrics(
            mAP_50=mAP_50,
            mAP_50_95=mAP_50_95,
            precision=precision,
            recall=recall,
            f1=f1,
            ap_per_class=ap_50_per_class,
            total_predictions=n_preds,
            total_ground_truths=n_gts,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )

    def _compute_ap_for_class_fast(
        self,
        sorted_preds: List[dict],
        class_id: int,
        iou_threshold: float,
    ) -> float:
        """Compute AP for a single class (OPTIMIZED)."""

        # Filter predictions for this class
        class_preds = [p for p in sorted_preds if p['cls'] == class_id]

        # Count ground truths for this class
        n_gt = sum(1 for g in self.ground_truths if g['cls'] == class_id)

        if n_gt == 0:
            return 0.0 if len(class_preds) > 0 else 1.0
        if len(class_preds) == 0:
            return 0.0

        # Track which GTs are matched (reset for each computation)
        matched_gt = {}  # (img_idx, gt_idx) -> True

        # Arrays for TP/FP
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))

        for pred_idx, pred in enumerate(class_preds):
            img_idx = pred['img_idx']
            pred_box = pred['box'].reshape(1, 4)

            # Get ground truths for this image
            if img_idx not in self.gt_by_image:
                fp[pred_idx] = 1
                continue

            gt_data = self.gt_by_image[img_idx]
            gt_boxes = gt_data['boxes']
            gt_classes = gt_data['classes']

            # Filter GTs for this class
            class_mask = gt_classes == class_id
            if not class_mask.any():
                fp[pred_idx] = 1
                continue

            gt_boxes_class = gt_boxes[class_mask]
            gt_indices = np.where(class_mask)[0]

            # Compute IoU with all GTs of this class in this image
            ious = box_iou_numpy(pred_box, gt_boxes_class).flatten()

            # COCO-style: find best *unmatched* GT above threshold
            if len(ious) > 0:
                sorted_gt = np.argsort(ious)[::-1]
                matched = False
                for gi in sorted_gt:
                    if ious[gi] < iou_threshold:
                        break
                    match_key = (img_idx, gt_indices[gi])
                    if match_key not in matched_gt:
                        tp[pred_idx] = 1
                        matched_gt[match_key] = True
                        matched = True
                        break
                if not matched:
                    fp[pred_idx] = 1
            else:
                fp[pred_idx] = 1

        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / n_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        return compute_ap(recalls, precisions)

    def _compute_pr_f1_fast(
        self,
        sorted_preds: List[dict],
        iou_threshold: float,
    ) -> Tuple[int, int, int, float, float, float]:
        """Compute precision, recall, F1 at best threshold (OPTIMIZED)."""

        n_gt = len(self.ground_truths)

        best_f1 = 0
        best_result = (0, len(sorted_preds), n_gt, 0.0, 0.0, 0.0)

        for conf_thresh in [0.1, 0.25, 0.5, 0.75]:
            matched_gt = {}
            tp = 0
            fp = 0

            for pred in sorted_preds:
                if pred['conf'] < conf_thresh:
                    continue

                img_idx = pred['img_idx']
                pred_box = pred['box'].reshape(1, 4)

                if img_idx not in self.gt_by_image:
                    fp += 1
                    continue

                gt_data = self.gt_by_image[img_idx]
                gt_boxes = gt_data['boxes']

                if len(gt_boxes) == 0:
                    fp += 1
                    continue

                # Compute IoU
                ious = box_iou_numpy(pred_box, gt_boxes).flatten()

                # Find best unmatched GT
                matched = False
                sorted_indices = np.argsort(ious)[::-1]
                for gt_idx in sorted_indices:
                    if ious[gt_idx] < iou_threshold:
                        break
                    match_key = (img_idx, gt_idx)
                    if match_key not in matched_gt:
                        tp += 1
                        matched_gt[match_key] = True
                        matched = True
                        break

                if not matched:
                    fp += 1

            fn = n_gt - tp
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)

            if f1 > best_f1:
                best_f1 = f1
                best_result = (tp, fp, fn, precision, recall, f1)

        return best_result
