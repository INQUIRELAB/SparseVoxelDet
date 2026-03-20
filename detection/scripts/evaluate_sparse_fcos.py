#!/usr/bin/env python3
"""
Evaluation script for Sparse FCOS Detector.

Computes:
- mAP@50
- mAP@50:95
- Per-size-bin AP (small/medium/large)
- Comparison to FRED baseline
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
import argparse
import sys
from collections import defaultdict

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader
from detection.scripts.sparse_event_dataset import (
    SparseEventDataset, make_collate_fn, create_sparse_tensor
)
from detection.scripts.metrics import MAPCalculator, compute_map_at_iou_range
from detection.models.sparse_fcos_detector import SparseFCOSDetector
from detection.models.sparse_tqdet import SparseTQDet
from detection.models.sparse_voxel_det import SparseVoxelDet

PARITY_SPLIT_ALLOWLIST = {
    "canonical_train",
    "canonical_test",
    "challenging_train",
    "challenging_test",
}
METRICS_ENGINE_ID = "detection.mapcalc"
METRICS_ENGINE_VERSION = "2026-02-26"


def _resolved(path_str: str) -> Path:
    return Path(path_str).resolve()


def _assert_parity_eval_contract(
    data_dir: str,
    label_dir: str,
    split: str,
    max_samples: Optional[int],
) -> None:
    expected_data = (project_root / "data/datasets/fred_paper_parity/sparse").resolve()
    expected_labels = (project_root / "data/datasets/fred_paper_parity/labels").resolve()
    data_resolved = _resolved(data_dir)
    label_resolved = _resolved(label_dir)

    errors: List[str] = []
    if data_resolved != expected_data:
        errors.append(f"data_dir must be {expected_data}, got {data_resolved}")
    if label_resolved != expected_labels:
        errors.append(f"label_dir must be {expected_labels}, got {label_resolved}")
    if split not in PARITY_SPLIT_ALLOWLIST:
        errors.append(
            f"split '{split}' not in allowlist {sorted(PARITY_SPLIT_ALLOWLIST)}"
        )
    if max_samples is not None:
        errors.append("max_samples is forbidden when --parity-enforced is active")
    if errors:
        raise ValueError(
            "Parity eval contract violation:\n- " + "\n- ".join(errors)
        )


def baseline_reference_for_split(split: str) -> Dict[str, float]:
    """Return paper-referenced YOLO v11 (event) baseline for the requested split."""
    split_l = str(split).lower()
    if "challenging" in split_l:
        return {
            "mAP50": 79.60,
            "mAP5095": 41.63,
        }
    return {
        "mAP50": 87.68,
        "mAP5095": 49.25,
    }


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        box1: [N, 4] boxes (x1, y1, x2, y2)
        box2: [M, 4] boxes (x1, y1, x2, y2)

    Returns:
        iou: [N, M] IoU matrix
    """
    N = box1.shape[0]
    M = box2.shape[0]

    # Expand for broadcasting
    box1 = box1.unsqueeze(1)  # [N, 1, 4]
    box2 = box2.unsqueeze(0)  # [1, M, 4]

    # Intersection
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Areas
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    # Union
    union_area = area1 + area2 - inter_area + 1e-7

    return inter_area / union_area


def compute_ap(
    pred_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    iou_threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Compute Average Precision at a given IoU threshold.

    Args:
        pred_boxes: List of [N_i, 4] predicted boxes per image
        pred_scores: List of [N_i] confidence scores per image
        gt_boxes: List of [M_i, 4] ground truth boxes per image
        iou_threshold: IoU threshold for matching

    Returns:
        ap: Average Precision
        precision: Final precision
        recall: Final recall
    """
    # Collect all predictions with image indices
    all_preds = []
    for img_idx, (boxes, scores) in enumerate(zip(pred_boxes, pred_scores)):
        for box, score in zip(boxes, scores):
            all_preds.append({
                'img_idx': img_idx,
                'box': box,
                'score': score.item() if isinstance(score, torch.Tensor) else score
            })

    # Sort by confidence (descending)
    all_preds.sort(key=lambda x: x['score'], reverse=True)

    # Track which GT boxes have been matched
    gt_matched = [torch.zeros(len(gt), dtype=torch.bool) for gt in gt_boxes]

    # Total number of GT boxes
    n_gt = sum(len(gt) for gt in gt_boxes)
    if n_gt == 0:
        return 0.0, 0.0, 0.0

    # Compute precision-recall curve
    tp = []
    fp = []

    for pred in all_preds:
        img_idx = pred['img_idx']
        pred_box = pred['box'].unsqueeze(0)
        gt = gt_boxes[img_idx]

        if len(gt) == 0:
            fp.append(1)
            tp.append(0)
            continue

        # Compute IoU with all GT boxes in this image
        ious = compute_iou(pred_box, gt).squeeze(0)

        # Find best matching GT
        best_iou, best_idx = ious.max(dim=0)

        if best_iou >= iou_threshold and not gt_matched[img_idx][best_idx]:
            tp.append(1)
            fp.append(0)
            gt_matched[img_idx][best_idx] = True
        else:
            tp.append(0)
            fp.append(1)

    # Compute cumulative TP and FP
    tp = np.array(tp)
    fp = np.array(fp)
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
    recall = tp_cumsum / n_gt

    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max() / 11

    final_precision = precision[-1] if len(precision) > 0 else 0.0
    final_recall = recall[-1] if len(recall) > 0 else 0.0

    return ap, final_precision, final_recall


def compute_map(
    pred_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    iou_thresholds: List[float] = None
) -> Dict[str, float]:
    """
    Compute mAP at multiple IoU thresholds.

    Returns:
        Dictionary with mAP@50, mAP@50:95, and per-threshold APs
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.50 to 0.95

    aps = {}
    for iou_thresh in iou_thresholds:
        ap, prec, rec = compute_ap(pred_boxes, pred_scores, gt_boxes, iou_thresh)
        aps[f'AP@{iou_thresh:.2f}'] = ap

    # Compute summary metrics
    map_50 = aps['AP@0.50']
    map_50_95 = np.mean([aps[f'AP@{t:.2f}'] for t in iou_thresholds])

    return {
        'mAP@50': map_50,
        'mAP@50:95': map_50_95,
        **aps
    }


def compute_size_binned_ap(
    pred_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    size_bins: Dict[str, Tuple[float, float]] = None
) -> Dict[str, float]:
    """
    Compute AP for different object sizes.

    FRED dataset: 74% of drones are <32px (small)
    """
    if size_bins is None:
        size_bins = {
            'small': (0, 32),
            'medium': (32, 96),
            'large': (96, float('inf'))
        }

    results = {}

    for size_name, (min_size, max_size) in size_bins.items():
        # Filter GT boxes by size
        filtered_gt = []
        for gt in gt_boxes:
            if len(gt) == 0:
                filtered_gt.append(gt)
                continue

            # Compute box sizes (max of width, height)
            widths = gt[:, 2] - gt[:, 0]
            heights = gt[:, 3] - gt[:, 1]
            sizes = torch.max(widths, heights)

            mask = (sizes >= min_size) & (sizes < max_size)
            filtered_gt.append(gt[mask])

        # Count GT boxes in this size range
        n_gt = sum(len(gt) for gt in filtered_gt)
        if n_gt == 0:
            results[f'AP@50_{size_name}'] = 0.0
            results[f'n_gt_{size_name}'] = 0
            continue

        # Compute AP
        ap, _, _ = compute_ap(pred_boxes, pred_scores, filtered_gt, iou_threshold=0.5)
        results[f'AP@50_{size_name}'] = ap
        results[f'n_gt_{size_name}'] = n_gt

    return results


def size_bin_name(max_dim: float) -> str:
    if max_dim < 16:
        return "tiny"
    if max_dim < 32:
        return "small"
    if max_dim < 64:
        return "medium"
    return "large"


def temporal_rerank_top1(
    detections: torch.Tensor,
    seq_ids: List[str],
    frame_nums: List[int],
    topk: int,
    weights: Dict[str, float],
    state: Dict[str, Dict[str, object]] | None = None,
) -> torch.Tensor:
    """Apply optional sequence-aware reranking and emit exactly top-1 per frame."""
    batch_size = detections.shape[0]
    out = torch.zeros(batch_size, 1, 6, device=detections.device, dtype=detections.dtype)
    if state is None:
        state = {}
    w_score = float(weights.get("score", 0.55))
    w_motion = float(weights.get("motion_distance", 0.35))
    w_size = float(weights.get("size_continuity", 0.10))

    for b in range(batch_size):
        seq_id = str(seq_ids[b]) if b < len(seq_ids) else "unknown"
        frame_num = int(frame_nums[b]) if b < len(frame_nums) else -1
        det = detections[b]
        valid = det[:, 4] > 0
        det = det[valid]
        if det.shape[0] == 0:
            continue

        # Candidate pool from detector decode.
        k = min(int(topk), det.shape[0])
        candidates = det[:k]

        seq_state = state.get(seq_id)
        if seq_state is None:
            seq_state = {
                "center": None,
                "velocity": (0.0, 0.0),
                "size": None,
                "frames_seen": 0,
                "last_frame_num": -1,
            }
            state[seq_id] = seq_state

        # Reset if there is a large frame gap.
        if int(seq_state["last_frame_num"]) >= 0 and frame_num >= 0 and frame_num - int(seq_state["last_frame_num"]) > 12:
            seq_state["center"] = None
            seq_state["velocity"] = (0.0, 0.0)
            seq_state["size"] = None
            seq_state["frames_seen"] = 0

        best_idx = 0
        best_value = -1e9
        prev_center = seq_state["center"]
        prev_velocity = seq_state["velocity"]
        prev_size = seq_state["size"]
        warm = int(seq_state["frames_seen"]) < 2 or prev_center is None

        for i in range(k):
            box = candidates[i, :4]
            conf = float(candidates[i, 4].item())
            cx = float((box[0] + box[2]).item() * 0.5)
            cy = float((box[1] + box[3]).item() * 0.5)
            w = float((box[2] - box[0]).item())
            h = float((box[3] - box[1]).item())

            if warm:
                score_val = conf
            else:
                px = float(prev_center[0] + prev_velocity[0])
                py = float(prev_center[1] + prev_velocity[1])
                motion_dist = float(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5)
                motion_score = float(np.exp(-motion_dist / 80.0))
                if prev_size is not None and prev_size[0] > 1e-3 and prev_size[1] > 1e-3:
                    area_ratio = (w * h + 1e-6) / (float(prev_size[0] * prev_size[1]) + 1e-6)
                    size_score = float(np.exp(-abs(np.log(area_ratio))))
                else:
                    size_score = 1.0
                score_val = w_score * conf + w_motion * motion_score + w_size * size_score

            if score_val > best_value:
                best_value = score_val
                best_idx = i

        chosen = candidates[best_idx]
        out[b, 0] = chosen

        cbox = chosen[:4]
        new_center = (
            float((cbox[0] + cbox[2]).item() * 0.5),
            float((cbox[1] + cbox[3]).item() * 0.5),
        )
        if prev_center is None:
            new_velocity = (0.0, 0.0)
        else:
            new_velocity = (new_center[0] - float(prev_center[0]), new_center[1] - float(prev_center[1]))
        seq_state["center"] = new_center
        seq_state["velocity"] = new_velocity
        seq_state["size"] = (
            float((cbox[2] - cbox[0]).item()),
            float((cbox[3] - cbox[1]).item()),
        )
        seq_state["frames_seen"] = int(seq_state["frames_seen"]) + 1
        seq_state["last_frame_num"] = frame_num

    return out


def summarize_by_sequence(frame_rows: List[Dict]) -> Dict[str, Dict]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for row in frame_rows:
        groups[str(row["seq_id"])].append(row)

    out: Dict[str, Dict] = {}
    for seq_id, rows in sorted(groups.items()):
        n = len(rows)
        tp = sum(1 for r in rows if r["matched"])
        fp = sum(1 for r in rows if r["num_dets"] > 0 and not r["matched"])
        fn_complete = sum(1 for r in rows if r["error_type"] == "fn_complete_miss")
        fn_loc = sum(1 for r in rows if r["error_type"] == "fn_localization")
        tp_ious = [float(r["best_iou"]) for r in rows if r["matched"]]
        out[seq_id] = {
            "n_frames": n,
            "tp": tp,
            "fp": fp,
            "fn": n - tp,
            "fn_complete_miss": fn_complete,
            "fn_localization": fn_loc,
            "recall": (tp / n) if n else 0.0,
            "precision": (tp / (tp + fp)) if (tp + fp) else 0.0,
            "median_tp_iou": float(np.median(tp_ious)) if tp_ious else 0.0,
            "mean_clip_fraction": float(np.mean([r["clip_fraction"] for r in rows])) if rows else 0.0,
            "clip_rate": float(np.mean([1.0 if r["clipped"] else 0.0 for r in rows])) if rows else 0.0,
        }
    return out


def summarize_clip_stats(frame_rows: List[Dict]) -> Dict:
    if not frame_rows:
        return {
            "n_frames": 0,
            "mean_raw_voxels": 0.0,
            "mean_kept_voxels": 0.0,
            "mean_clip_fraction": 0.0,
            "clip_rate": 0.0,
            "max_clip_fraction": 0.0,
        }
    return {
        "n_frames": len(frame_rows),
        "mean_raw_voxels": float(np.mean([r["raw_n_voxels"] for r in frame_rows])),
        "mean_kept_voxels": float(np.mean([r["kept_n_voxels"] for r in frame_rows])),
        "mean_clip_fraction": float(np.mean([r["clip_fraction"] for r in frame_rows])),
        "clip_rate": float(np.mean([1.0 if r["clipped"] else 0.0 for r in frame_rows])),
        "max_clip_fraction": float(np.max([r["clip_fraction"] for r in frame_rows])),
    }


def summarize_error_mix(frame_rows: List[Dict]) -> Dict:
    counts = defaultdict(int)
    fp_counts = defaultdict(int)
    tp_ious = []
    for row in frame_rows:
        counts[row["error_type"]] += 1
        fp_type = row.get("fp_type")
        if fp_type:
            fp_counts[fp_type] += 1
        if row["matched"]:
            tp_ious.append(float(row["best_iou"]))
    total = len(frame_rows)
    fn_total = counts["fn_complete_miss"] + counts["fn_localization"]
    return {
        "counts": dict(sorted(counts.items())),
        "rates": {k: (v / total if total else 0.0) for k, v in sorted(counts.items())},
        "fp_counts": dict(sorted(fp_counts.items())),
        "fp_rates": {k: (v / total if total else 0.0) for k, v in sorted(fp_counts.items())},
        "fn_complete_miss_rate": (counts["fn_complete_miss"] / fn_total) if fn_total else 0.0,
        "fn_localization_rate": (counts["fn_localization"] / fn_total) if fn_total else 0.0,
        "median_tp_iou": float(np.median(tp_ious)) if tp_ious else 0.0,
        "n_frames": total,
    }


def evaluate(
    checkpoint_path: str,
    data_dir: str,
    label_dir: str,
    split: str = 'val',
    batch_size: int = 4,
    num_workers: int = 4,
    device: str = 'cuda',
    max_samples: int = None,
    output_path: str = None,
    outdir: str = None,
    score_thresh: float = None,
    nms_thresh: float = None,
    max_detections: int = None,
    temporal_rerank_enabled: bool = None,
    temporal_rerank_topk: int = None,
    baseline_reference_name: str = "YOLO v11 (event-only, FRED Table 2)",
    baseline_reference_source: str = "arXiv:2506.05163 Table 2",
    parity_mode: str = "auto",
    parity_coverage: str = "unknown",
    parity_enforced: bool = False,
    metrics_engine: str = "canonical",
) -> Dict[str, float]:
    """
    Evaluate Sparse FCOS detector on FRED dataset.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to sparse data directory
        label_dir: Path to labels directory
        split: Data split ('val' or 'test')
        batch_size: Batch size for evaluation
        num_workers: Number of data loader workers
        device: Device to run on
        max_samples: Maximum samples to evaluate (for quick testing)
        output_path: Path to save results JSON
        outdir: Optional directory for full-val artifact pack
        score_thresh: Optional score threshold override
        nms_thresh: Optional NMS IoU threshold override
        max_detections: Optional max detections/image override
        temporal_rerank_enabled: Optional override for sequence-aware reranking
        temporal_rerank_topk: Optional candidate pool size for reranking
        baseline_reference_name: Baseline label used in printed/JSON gap comparison
        baseline_reference_source: Source string for baseline reference
        parity_mode: Comparison lane label (paper_parity_standard / reliability_strict_top1 / auto)
        parity_coverage: Coverage label such as "47/47 exact" or "58/59 non-exact"

    Returns:
        Dictionary with all metrics
    """
    if parity_enforced:
        _assert_parity_eval_contract(
            data_dir=data_dir,
            label_dir=label_dir,
            split=split,
            max_samples=max_samples,
        )
    elif split not in PARITY_SPLIT_ALLOWLIST and split not in {"train", "val", "test"}:
        raise ValueError(
            f"Unsupported split '{split}'. Allowed: {sorted(PARITY_SPLIT_ALLOWLIST | {'train', 'val', 'test'})}"
        )

    if metrics_engine not in {"canonical", "legacy"}:
        raise ValueError("metrics_engine must be 'canonical' or 'legacy'")
    if parity_enforced and metrics_engine != "canonical":
        raise ValueError(
            "Parity-enforced/reportable evaluation requires metrics_engine='canonical'. "
            "Legacy engine is diagnostic-only."
        )

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device}")

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (DDP models)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Create model from checkpoint config (avoids architecture mismatch)
    # Checkpoints saved by train_sparse_fcos.py include 'config' key
    ckpt_config = checkpoint.get('config', {})
    model_config = ckpt_config.get('model', {})
    fcos_config = ckpt_config.get('fcos', {})
    sparse_config = ckpt_config.get('sparse', {})
    loss_config = ckpt_config.get('loss', {})

    input_size = model_config.get('input_size', [640, 640])
    if isinstance(input_size, int):
        input_size = [input_size, input_size]

    model_type = str(model_config.get('type', 'sparse_fcos')).lower()
    if model_type == 'sparse_tqdet':
        model = SparseTQDet(
            in_channels=2,
            num_classes=model_config.get('num_classes', 1),
            backbone_size=model_config.get('backbone_size', 'nano'),
            fpn_channels=model_config.get('fpn_channels', 128),
            num_refine_layers=model_config.get('num_refine_layers', 4),
            topk=model_config.get('topk', 64),
            strides=fcos_config.get('strides', [4, 8, 16]),
            input_size=tuple(input_size),
            time_bins=sparse_config.get('time_bins', 15),
            bridge_type=model_config.get('bridge_type', 'motion_aware'),
            num_temporal_groups=sparse_config.get('num_temporal_groups', 3),
            use_pan=model_config.get('use_pan', True),
            query_heads=model_config.get('query_heads', 8),
            memory_pool=tuple(model_config.get('memory_pool', [8, 8])),
            use_uncertainty_in_score=float(loss_config.get('uncertainty_weight', 0.0)) > 0.0,
        ).to(device)
    elif model_type == 'sparse_voxel_det':
        det_config = ckpt_config.get('detection', {})
        model = SparseVoxelDet(
            in_channels=2,
            num_classes=model_config.get('num_classes', 1),
            backbone_size=model_config.get('backbone_size', 'nano_deep'),
            fpn_channels=model_config.get('fpn_channels', 128),
            head_convs=model_config.get('head_convs', 2),
            input_size=tuple(input_size),
            time_bins=sparse_config.get('time_bins', 15),
            prior_prob=model_config.get('prior_prob', 0.01),
            score_thresh=float(det_config.get('score_thresh', 0.05)),
            nms_thresh=float(det_config.get('nms_thresh', 0.5)),
            max_detections=int(det_config.get('max_detections', 100)),
            temporal_pool_mode=model_config.get('temporal_pool_mode', 'max'),
        ).to(device)
    else:
        model = SparseFCOSDetector(
            in_channels=2,
            num_classes=model_config.get('num_classes', 1),
            backbone_size=model_config.get('backbone_size', 'nano'),
            fpn_channels=model_config.get('fpn_channels', 64),
            num_head_convs=model_config.get('num_head_convs', 4),
            strides=fcos_config.get('strides', [4, 8, 16]),
            norm_on_bbox=fcos_config.get('norm_on_bbox', True),
            input_size=tuple(input_size),
            time_bins=sparse_config.get('time_bins', 15),
            bridge_type=model_config.get('bridge_type', 'attention'),
            gn_groups=model_config.get('gn_groups', 8),
            use_pan=model_config.get('use_pan', False),
            use_iou_quality=float(loss_config.get('iou_quality_weight', 0.0)) > 0,
        ).to(device)
    load_result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    if missing_keys:
        print(f"WARNING: Missing checkpoint keys: {missing_keys[:8]}{' ...' if len(missing_keys) > 8 else ''}")
    if unexpected_keys:
        print(f"WARNING: Unexpected checkpoint keys: {unexpected_keys[:8]}{' ...' if len(unexpected_keys) > 8 else ''}")
    model.eval()

    # Patch decode thresholds from checkpoint eval config (or CLI overrides).
    eval_cfg = ckpt_config.get('eval', {})
    temporal_cfg = eval_cfg.get('temporal_rerank', {}) or {}
    eval_score_thresh = float(score_thresh) if score_thresh is not None else float(eval_cfg.get('score_thresh', 0.05))
    eval_nms_thresh = float(nms_thresh) if nms_thresh is not None else float(eval_cfg.get('nms_thresh', 0.5))
    eval_max_detections = int(max_detections) if max_detections is not None else int(eval_cfg.get('max_detections', 1))
    temporal_enabled = bool(temporal_cfg.get('enabled', False)) if temporal_rerank_enabled is None else bool(temporal_rerank_enabled)
    temporal_topk = int(temporal_cfg.get('topk', 5)) if temporal_rerank_topk is None else int(temporal_rerank_topk)
    temporal_weights = temporal_cfg.get('weights', {}) if isinstance(temporal_cfg.get('weights', {}), dict) else {}
    decode_max_detections = max(eval_max_detections, temporal_topk) if temporal_enabled else eval_max_detections
    original_decode = None
    if model_type == 'sparse_tqdet' and hasattr(model, 'set_decode_params'):
        model.set_decode_params(
            score_thresh=eval_score_thresh,
            nms_thresh=eval_nms_thresh,
            max_detections=decode_max_detections,
        )
    elif model_type == 'sparse_voxel_det' and hasattr(model, 'set_decode_params'):
        model.set_decode_params(
            score_thresh=eval_score_thresh,
            nms_thresh=eval_nms_thresh,
            max_detections=decode_max_detections,
        )
    else:
        original_decode = model._decode_predictions

        def patched_decode(cls_preds, reg_preds, ctr_preds, device,
                           score_thresh=None, nms_thresh=None, max_detections=None,
                           iou_quality_preds=None):
            return original_decode(
                cls_preds, reg_preds, ctr_preds, device,
                score_thresh=eval_score_thresh if score_thresh is None else score_thresh,
                nms_thresh=eval_nms_thresh if nms_thresh is None else nms_thresh,
                max_detections=decode_max_detections if max_detections is None else max_detections,
                iou_quality_preds=iou_quality_preds,
            )

        model._decode_predictions = patched_decode

    # Create dataset
    print(f"Loading {split} dataset...")
    time_bins = sparse_config.get('time_bins', 15)
    max_voxels_base = int(sparse_config.get('max_voxels', 100000))
    max_voxels_eval = int(sparse_config.get('max_voxels_eval', max_voxels_base))
    dataset = SparseEventDataset(
        sparse_dir=data_dir,
        label_dir=label_dir,
        split=split,
        time_bins=time_bins,
        target_size=(int(input_size[0]), int(input_size[1])),
        augment=False,
        max_voxels=max_voxels_eval,
        voxel_sampling={"mode": "random"},
    )

    if max_samples:
        dataset.samples = dataset.samples[:max_samples]

    collate_fn = make_collate_fn(time_bins=time_bins, base_size=int(input_size[0]))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print(f"Evaluating {len(dataset)} samples...")
    print(
        "Decode/eval policy: "
        f"score={eval_score_thresh} nms={eval_nms_thresh} "
        f"max_det_final={eval_max_detections} max_det_decode={decode_max_detections} "
        f"temporal_rerank={temporal_enabled}"
    )

    # Collect predictions and ground truth
    all_pred_boxes: List[torch.Tensor] = []
    all_pred_scores: List[torch.Tensor] = []
    all_pred_uncertainty: List[torch.Tensor] = []
    all_gt_boxes: List[torch.Tensor] = []
    frame_rows: List[Dict] = []
    temporal_state: Dict[str, Dict[str, object]] = {}
    map_calc = MAPCalculator(
        num_classes=1,
        img_size=tuple(input_size),
        conf_threshold=eval_score_thresh,
        max_predictions_per_image=max(1, int(decode_max_detections if decode_max_detections > 0 else 100)),
    )

    with torch.no_grad():
        for batch in tqdm(dataloader):
            sparse_input = create_sparse_tensor(batch, device)
            output = model(sparse_input, batch['batch_size'])
            detections = output['detections']  # [B, max_det, 6]
            det_unc = output.get('uncertainties', None)

            if temporal_enabled:
                detections = temporal_rerank_top1(
                    detections=detections,
                    seq_ids=[str(x) for x in batch.get("seq_ids", [])],
                    frame_nums=[int(x) for x in batch.get("frame_nums", [])],
                    topk=temporal_topk,
                    weights=temporal_weights,
                    state=temporal_state,
                )

            telem = batch.get("clip_telemetry", {}) or {}
            raw_vox = [int(v) for v in telem.get("raw_voxels", [])]
            kept_vox = [int(v) for v in telem.get("kept_voxels", [])]
            clip_frac = [float(v) for v in telem.get("clip_fraction", [])]
            clipped = [bool(v) for v in telem.get("clipped", [])]
            seq_ids = [str(v) for v in batch.get("seq_ids", [])]
            frame_nums = [int(v) for v in batch.get("frame_nums", [])]
            sample_ids = [str(v) for v in batch.get("sample_ids", [])]

            for b in range(batch['batch_size']):
                det = detections[b]
                valid = det[:, 4] > eval_score_thresh
                det = det[valid]
                if det_unc is not None:
                    unc = det_unc[b][valid]
                else:
                    unc = torch.zeros((det.shape[0],), device=det.device)
                if eval_max_detections > 0 and det.shape[0] > eval_max_detections:
                    det = det[:eval_max_detections]
                    unc = unc[:eval_max_detections]

                gt = batch['gt_boxes'][b].cpu()
                gt_labels = batch['gt_labels'][b].cpu()
                all_gt_boxes.append(gt)
                if len(det) > 0:
                    all_pred_boxes.append(det[:, :4].cpu())
                    all_pred_scores.append(det[:, 4].cpu())
                    all_pred_uncertainty.append(unc.cpu())
                else:
                    all_pred_boxes.append(torch.zeros(0, 4))
                    all_pred_scores.append(torch.zeros(0))
                    all_pred_uncertainty.append(torch.zeros(0))

                seq_id = seq_ids[b] if b < len(seq_ids) else "unknown"
                frame_num = frame_nums[b] if b < len(frame_nums) else -1
                sample_id = sample_ids[b] if b < len(sample_ids) else f"sample_{len(frame_rows):06d}"

                if len(gt) > 0:
                    gt_w_all = (gt[:, 2] - gt[:, 0]).clamp(min=0)
                    gt_h_all = (gt[:, 3] - gt[:, 1]).clamp(min=0)
                    gt_max_dim = float(torch.maximum(gt_w_all, gt_h_all).max().item())
                else:
                    gt_max_dim = 0.0

                if len(det) > 0 and len(gt) > 0:
                    ious = compute_iou(det[:, :4].cpu(), gt)
                    best_iou = float(ious.max().item())
                else:
                    best_iou = 0.0
                matched = bool(best_iou >= 0.5)
                if matched:
                    error_type = "tp"
                elif len(det) == 0:
                    error_type = "fn_complete_miss"
                elif best_iou >= 0.1:
                    error_type = "fn_localization"
                else:
                    error_type = "fn_complete_miss"

                fp_type = None
                if len(det) > 0 and not matched:
                    fp_type = "fp_localization" if best_iou >= 0.1 else "fp_background"
                elif len(det) > 1 and matched:
                    fp_type = "fp_duplicate"

                # Canonical metric engine update (shared with trainer metric path).
                if metrics_engine == "canonical":
                    if len(gt) > 0:
                        H, W = int(input_size[0]), int(input_size[1])
                        x1, y1, x2, y2 = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
                        cx = ((x1 + x2) * 0.5) / float(W)
                        cy = ((y1 + y2) * 0.5) / float(H)
                        bw = (x2 - x1) / float(W)
                        bh = (y2 - y1) / float(H)
                        gt_yolo = torch.stack(
                            [gt_labels.float(), cx.float(), cy.float(), bw.float(), bh.float()], dim=1
                        )
                    else:
                        gt_yolo = torch.zeros((0, 5), dtype=torch.float32)
                    map_calc.update([det.detach().cpu()], [gt_yolo.cpu()])

                frame_rows.append(
                    {
                        "sample_id": sample_id,
                        "seq_id": seq_id,
                        "frame_num": frame_num,
                        "matched": matched,
                        "num_dets": int(det.shape[0]),
                        "mean_uncertainty": float(unc.mean().item()) if unc.numel() > 0 else 0.0,
                        "best_iou": float(best_iou),
                        "error_type": error_type,
                        "fp_type": fp_type,
                        "gt_max_dim": float(gt_max_dim),
                        "size_bin": size_bin_name(gt_max_dim),
                        "raw_n_voxels": raw_vox[b] if b < len(raw_vox) else 0,
                        "kept_n_voxels": kept_vox[b] if b < len(kept_vox) else 0,
                        "clip_fraction": clip_frac[b] if b < len(clip_frac) else 0.0,
                        "clipped": clipped[b] if b < len(clipped) else False,
                    }
                )

    print("Computing metrics...")

    if metrics_engine == "canonical":
        det_metrics = map_calc.compute()
        map_results = {
            "mAP@50": float(det_metrics.mAP_50),
            "mAP@50:95": float(det_metrics.mAP_50_95),
            "precision": float(det_metrics.precision),
            "recall": float(det_metrics.recall),
            "f1": float(det_metrics.f1),
        }
        ap45_pack = compute_map_at_iou_range(
            map_calc.predictions,
            map_calc.ground_truths,
            map_calc.gt_by_image,
            iou_thresholds=np.array([0.45], dtype=np.float32),
            num_classes=1,
        )
        ap_45 = float(ap45_pack.get("ap_at_0.45", 0.0))
    else:
        map_results = compute_map(all_pred_boxes, all_pred_scores, all_gt_boxes)
        ap_50_legacy, prec50, rec50 = compute_ap(
            all_pred_boxes, all_pred_scores, all_gt_boxes, iou_threshold=0.5
        )
        map_results["mAP@50"] = float(ap_50_legacy)
        map_results["precision"] = float(prec50)
        map_results["recall"] = float(rec50)
        map_results["f1"] = float((2.0 * prec50 * rec50) / max(prec50 + rec50, 1e-7))
        ap_45, _, _ = compute_ap(all_pred_boxes, all_pred_scores, all_gt_boxes, iou_threshold=0.45)
    size_results = compute_size_binned_ap(all_pred_boxes, all_pred_scores, all_gt_boxes)

    by_sequence = summarize_by_sequence(frame_rows)
    clip_stats = summarize_clip_stats(frame_rows)
    error_mix = summarize_error_mix(frame_rows)
    uncertainty_vals = torch.cat([u for u in all_pred_uncertainty if u.numel() > 0], dim=0) if any(
        u.numel() > 0 for u in all_pred_uncertainty
    ) else torch.zeros(0)
    if uncertainty_vals.numel() > 0:
        uncertainty_mean = float(uncertainty_vals.mean().item())
        uncertainty_median = float(uncertainty_vals.median().item())
    else:
        uncertainty_mean = 0.0
        uncertainty_median = 0.0

    ranking_error_frames = sum(
        1 for r in frame_rows if (r.get("num_dets", 0) > 0 and not r.get("matched", False))
    )
    ranking_error_rate = (ranking_error_frames / len(frame_rows)) if frame_rows else 0.0

    recall_by_size = {}
    for k in ("tiny", "small", "medium", "large"):
        rows = [r for r in frame_rows if r["size_bin"] == k]
        tp = sum(1 for r in rows if r["matched"])
        recall_by_size[f"recall_{k}"] = (tp / len(rows)) if rows else 0.0
        recall_by_size[f"n_{k}"] = len(rows)

    by_size = {
        **size_results,
        **recall_by_size,
    }

    baseline_ref = baseline_reference_for_split(split)
    if parity_mode == "auto":
        parity_mode = "paper_parity_standard" if eval_max_detections > 1 else "reliability_strict_top1"

    map50_pct = float(map_results.get('mAP@50', 0.0)) * 100.0
    gap_points_mAP50 = map50_pct - float(baseline_ref["mAP50"])
    gap_points_mAP5095 = float(map_results.get('mAP@50:95', 0.0)) * 100.0 - float(baseline_ref["mAP5095"])

    results = {
        **map_results,
        'AP@0.45': ap_45,
        'borderline_gap_AP45_minus_AP50': ap_45 - map_results.get('mAP@50', 0.0),
        **size_results,
        'n_samples': len(dataset),
        'n_predictions': (
            int(det_metrics.total_predictions)
            if metrics_engine == "canonical"
            else int(sum(len(p) for p in all_pred_boxes))
        ),
        'n_gt_total': (
            int(det_metrics.total_ground_truths)
            if metrics_engine == "canonical"
            else int(sum(len(gt) for gt in all_gt_boxes))
        ),
        'score_thresh': eval_score_thresh,
        'nms_thresh': eval_nms_thresh,
        'max_detections': eval_max_detections,
        'decode_max_detections': decode_max_detections,
        'time_bins': int(time_bins),
        'max_voxels_eval': int(max_voxels_eval),
        'temporal_rerank_enabled': temporal_enabled,
        'temporal_rerank_topk': int(temporal_topk) if temporal_enabled else 0,
        'temporal_rerank_weights': temporal_weights if temporal_enabled else {},
        'clip_rate': clip_stats["clip_rate"],
        'mean_clip_fraction': clip_stats["mean_clip_fraction"],
        'median_tp_iou': error_mix["median_tp_iou"],
        'fn_complete_miss_rate': error_mix["fn_complete_miss_rate"],
        'fn_localization_rate': error_mix["fn_localization_rate"],
        'ranking_error_rate': ranking_error_rate,
        'uncertainty_mean': uncertainty_mean,
        'uncertainty_median': uncertainty_median,
        'baseline_reference_name': baseline_reference_name,
        'baseline_reference_source': baseline_reference_source,
        'baseline_reference_mAP50': float(baseline_ref["mAP50"]),
        'baseline_reference_mAP50_95': float(baseline_ref["mAP5095"]),
        'gap_vs_baseline_mAP50_points': gap_points_mAP50,
        'gap_vs_baseline_mAP50_95_points': gap_points_mAP5095,
        'model_type': model_type,
        'parity_mode': str(parity_mode),
        'parity_coverage': str(parity_coverage),
        'parity_enforced': bool(parity_enforced),
        'metrics_engine_id': METRICS_ENGINE_ID if metrics_engine == "canonical" else "legacy.11pt",
        'metrics_version': METRICS_ENGINE_VERSION if metrics_engine == "canonical" else "legacy",
    }

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"mAP@50:     {results['mAP@50'] * 100:.2f}%")
    print(f"mAP@50:95:  {results['mAP@50:95'] * 100:.2f}%")
    print(f"AP@45:      {results['AP@0.45'] * 100:.2f}%")
    print(f"Borderline gap (AP45-AP50): {results['borderline_gap_AP45_minus_AP50'] * 100:.2f}%")
    print(f"Median TP IoU: {results['median_tp_iou']:.3f}")
    print(f"FN complete miss rate: {results['fn_complete_miss_rate'] * 100:.2f}%")
    print(f"FN localization rate:  {results['fn_localization_rate'] * 100:.2f}%")
    print(f"Ranking error rate:    {results['ranking_error_rate'] * 100:.2f}%")
    print(f"Uncertainty mean/med:  {results['uncertainty_mean']:.4f} / {results['uncertainty_median']:.4f}")
    print(f"Clip rate: {results['clip_rate'] * 100:.2f}%")
    print(f"Mean clip fraction: {results['mean_clip_fraction']:.4f}")
    print(f"Total predictions: {results['n_predictions']}")
    print(f"Total GT boxes: {results['n_gt_total']}")
    print("=" * 60)

    baseline_map50 = float(results['baseline_reference_mAP50'])
    gap = baseline_map50 - results['mAP@50'] * 100
    print(f"\nComparison to FRED baseline:")
    print(f"  Baseline ({results['baseline_reference_name']}): {baseline_map50:.2f}%")
    print(f"  Sparse FCOS:           {results['mAP@50'] * 100:.2f}%")
    print(f"  Gap:                   {gap:.2f}%")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    if outdir:
        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        (outdir_path / "fullval_metrics.json").write_text(json.dumps(results, indent=2))
        (outdir_path / "fullval_by_size.json").write_text(json.dumps(by_size, indent=2))
        (outdir_path / "fullval_by_sequence.json").write_text(json.dumps(by_sequence, indent=2))
        (outdir_path / "fullval_clip_stats.json").write_text(json.dumps(clip_stats, indent=2))
        (outdir_path / "fullval_error_mix.json").write_text(json.dumps(error_mix, indent=2))
        with (outdir_path / "fullval_frame_rows.jsonl").open("w") as f:
            for row in frame_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Saved full-val artifact pack to {outdir_path}")

    if original_decode is not None:
        model._decode_predictions = original_decode
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Sparse FCOS Detector')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to sparse data directory')
    parser.add_argument('--label_dir', type=str, default=None,
                        help='Path to labels directory')
    parser.add_argument('--split', type=str, default=None,
                        help='Data split (val or test)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples for quick testing')
    parser.add_argument('--parity-enforced', action='store_true',
                        help='Hard-fail unless parity roots/splits are used and sample caps are disabled')
    parser.add_argument('--metrics-engine', type=str, default='canonical',
                        choices=['canonical', 'legacy'],
                        help='Canonical uses MAPCalculator (same engine as trainer validation)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Optional output directory for fullval_*.json artifact pack')
    parser.add_argument('--score_thresh', type=float, default=None,
                        help='Optional score threshold override')
    parser.add_argument('--nms_thresh', type=float, default=None,
                        help='Optional NMS IoU threshold override')
    parser.add_argument('--max_detections', type=int, default=None,
                        help='Optional max detections/image override')
    parser.add_argument('--temporal-rerank',
                        action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Enable/disable sequence-aware temporal top-1 reranking (default: config value)')
    parser.add_argument('--temporal-rerank-topk', type=int, default=None,
                        help='Candidate pool size for temporal reranking (default from config)')
    parser.add_argument('--baseline-reference-name', type=str,
                        default='YOLO v11 (event-only, FRED Table 2)',
                        help='Display/metadata name for baseline comparison')
    parser.add_argument('--baseline-reference-source', type=str,
                        default='arXiv:2506.05163 Table 2',
                        help='Source string for baseline comparison metadata')
    parser.add_argument('--parity-mode', type=str, default='auto',
                        help='Comparison lane tag: paper_parity_standard | reliability_strict_top1 | auto')
    parser.add_argument('--parity-coverage', type=str, default='unknown',
                        help='Parity coverage tag, e.g. 47/47 exact or 58/59 non-exact')
    args = parser.parse_args()

    if args.parity_enforced and (args.data_dir is None or args.label_dir is None or args.split is None):
        raise ValueError(
            "When --parity-enforced is set, --data_dir, --label_dir, and --split must be provided explicitly."
        )

    data_dir = args.data_dir or 'data/datasets/fred_sparse'
    label_dir = args.label_dir or 'data/datasets/fred_voxel/labels_clean'
    split = args.split or 'val'

    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=data_dir,
        label_dir=label_dir,
        split=split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_samples=args.max_samples,
        output_path=args.output,
        outdir=args.outdir,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        max_detections=args.max_detections,
        temporal_rerank_enabled=args.temporal_rerank,
        temporal_rerank_topk=args.temporal_rerank_topk,
        baseline_reference_name=args.baseline_reference_name,
        baseline_reference_source=args.baseline_reference_source,
        parity_mode=args.parity_mode,
        parity_coverage=args.parity_coverage,
        parity_enforced=bool(args.parity_enforced),
        metrics_engine=str(args.metrics_engine),
    )


if __name__ == '__main__':
    main()
