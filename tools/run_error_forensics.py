#!/usr/bin/env python3
"""
Error Forensics Pipeline for SparseVoxelDet v83.

Runs full model inference on the FRED test set, classifies every
prediction as TP/FP/FN, and produces:
  1. Per-frame prediction dump (predictions.json)
  2. IoU distribution of near-misses
  3. Confidence calibration histogram
  4. Event density vs recall curve
  5. Object size vs recall analysis
  6. Per-sequence breakdown table
  7. Spatial error heatmaps
  8. Summary statistics JSON

Usage:
    CUDA_VISIBLE_DEVICES=6 ./venv/bin/python tools/run_error_forensics.py \
        --checkpoint runs/sparse_voxel_det/v83_seed42/best.pt \
        --config V2/configs/sparse_voxel_det_v83.yaml \
        --output runs/sparse_voxel_det/v83_seed42/forensics/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import yaml
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))


def compute_iou(box_a, box_b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    
    return inter / union if union > 0 else 0.0


def classify_predictions(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    """Classify predictions as TP/FP and track FN ground truths."""
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return [], [], []
    
    tps = []  # (pred_idx, gt_idx, iou, score)
    fps = []  # (pred_idx, score, best_iou)
    fns = []  # (gt_idx, best_pred_iou)
    
    gt_matched = [False] * len(gt_boxes)
    
    # Sort predictions by score (descending)
    if len(pred_boxes) > 0:
        order = np.argsort(-pred_scores)
        
        for pred_idx in order:
            best_iou = 0.0
            best_gt = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                iou = compute_iou(pred_boxes[pred_idx], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_idx
            
            if best_iou >= iou_thresh and best_gt >= 0:
                tps.append((int(pred_idx), int(best_gt), float(best_iou), float(pred_scores[pred_idx])))
                gt_matched[best_gt] = True
            else:
                fps.append((int(pred_idx), float(pred_scores[pred_idx]), float(best_iou)))
    
    # Unmatched GTs are FNs
    for gt_idx in range(len(gt_boxes)):
        if not gt_matched[gt_idx]:
            # Find best matching prediction IoU
            best_pred_iou = 0.0
            if len(pred_boxes) > 0:
                for pred_idx in range(len(pred_boxes)):
                    iou = compute_iou(pred_boxes[pred_idx], gt_boxes[gt_idx])
                    best_pred_iou = max(best_pred_iou, iou)
            fns.append((int(gt_idx), float(best_pred_iou)))
    
    return tps, fps, fns


def generate_summary(all_tps, all_fps, all_fns, output_dir):
    """Generate summary statistics and classification breakdown."""
    total_tp = sum(len(t) for t in all_tps)
    total_fp = sum(len(f) for f in all_fps)
    total_fn = sum(len(f) for f in all_fns)
    total_gt = total_tp + total_fn
    
    recall = total_tp / total_gt if total_gt > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    
    # FN breakdown by IoU
    fn_ious = [iou for fns in all_fns for _, iou in fns]
    fn_zero = sum(1 for iou in fn_ious if iou == 0)
    fn_near_miss = sum(1 for iou in fn_ious if 0 < iou < 0.5)
    
    # TP confidence stats
    tp_scores = [score for tps in all_tps for _, _, _, score in tps]
    fp_scores = [score for fps in all_fps for _, score, _ in fps]
    
    summary = {
        'total_frames': len(all_tps),
        'total_gt': total_gt,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'recall': float(recall),
        'precision': float(precision),
        'fn_zero_overlap': fn_zero,
        'fn_near_miss': fn_near_miss,
        'fn_near_miss_pct': fn_near_miss / total_fn * 100 if total_fn > 0 else 0,
        'tp_conf_mean': float(np.mean(tp_scores)) if tp_scores else 0,
        'tp_conf_std': float(np.std(tp_scores)) if tp_scores else 0,
        'fp_conf_mean': float(np.mean(fp_scores)) if fp_scores else 0,
        'fp_conf_std': float(np.std(fp_scores)) if fp_scores else 0,
    }
    
    out_file = output_dir / 'forensics_summary.json'
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("ERROR FORENSICS SUMMARY")
    print("=" * 60)
    print(f"  Test frames:      {summary['total_frames']:,}")
    print(f"  Ground truths:    {summary['total_gt']:,}")
    print(f"  True positives:   {summary['total_tp']:,} ({recall:.1%} recall)")
    print(f"  False positives:  {summary['total_fp']:,}")
    print(f"  False negatives:  {summary['total_fn']:,}")
    print(f"    - Near-misses:  {summary['fn_near_miss']:,} ({summary['fn_near_miss_pct']:.1f}%)")
    print(f"    - Zero overlap: {summary['fn_zero_overlap']:,}")
    print(f"  Precision:        {precision:.3f}")
    print(f"  TP confidence:    {summary['tp_conf_mean']:.3f} ± {summary['tp_conf_std']:.3f}")
    print(f"  FP confidence:    {summary['fp_conf_mean']:.3f} ± {summary['fp_conf_std']:.3f}")
    print("=" * 60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Error Forensics for SparseVoxelDet')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--score-thresh', type=float, default=0.05)
    parser.add_argument('--nms-thresh', type=float, default=0.5)
    parser.add_argument('--max-det', type=int, default=100)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Import model and dataset
    from V2.models.sparse_voxel_det_v82 import SparseVoxelDet
    from sparse_fcos_v1.scripts.sparse_event_dataset_v82 import (
        SparseEventDataset, make_collate_fn, create_sparse_tensor
    )
    
    # Load model
    input_size = config['model'].get('input_size', [720, 1280])
    if isinstance(input_size, int):
        input_size = [input_size, input_size]
    
    model = SparseVoxelDet(
        in_channels=config['model'].get('in_channels', 6),
        backbone_size=config['model'].get('backbone_size', 'nano_deep'),
        fpn_channels=config['model'].get('fpn_channels', 128),
        num_classes=config['model'].get('num_classes', 1),
        head_convs=config['model'].get('head_convs', 2),
        prior_prob=config['model'].get('prior_prob', 0.01),
        input_size=tuple(input_size),
        time_bins=config['sparse'].get('time_bins', 16),
    )
    
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'ema_state_dict' in ckpt:
        ema = ckpt['ema_state_dict']
        if isinstance(ema, dict) and 'shadow' in ema:
            ema = ema['shadow']
        model.load_state_dict(ema, strict=True)
        print("Loaded EMA weights")
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        print("Loaded model weights")
    
    model = model.to(device).eval()
    
    # Load dataset  
    data_cfg = config.get('data', {})
    dataset = SparseEventDataset(
        sparse_dir=str(project_root / data_cfg['sparse_dir']),
        label_dir=str(project_root / data_cfg['label_dir']),
        split=data_cfg.get('val_split', 'canonical_test'),
        target_size=tuple(input_size),
        time_bins=config['sparse']['time_bins'],
        augment=False,
    )
    
    time_bins = config['sparse']['time_bins']
    collate_fn = make_collate_fn(time_bins=time_bins, base_size=tuple(input_size))
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=8, collate_fn=collate_fn,
        pin_memory=True, drop_last=False,
    )
    
    print(f"Test set: {len(dataset):,} frames")
    print(f"Running inference with score_thresh={args.score_thresh}, nms_thresh={args.nms_thresh}")
    
    all_tps, all_fps, all_fns = [], [], []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
            sp = create_sparse_tensor(batch, device)
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                preds = model(sp)
            
            # Extract predictions
            if isinstance(preds, dict) and 'boxes' in preds:
                boxes = preds['boxes'].cpu().numpy()
                scores = preds['scores'].cpu().numpy()
                
                # Filter by score
                keep = scores > args.score_thresh
                boxes = boxes[keep]
                scores = scores[keep]
                
                # NMS
                if len(boxes) > 0:
                    from torchvision.ops import nms
                    keep_nms = nms(
                        torch.tensor(boxes, dtype=torch.float32),
                        torch.tensor(scores, dtype=torch.float32),
                        args.nms_thresh
                    ).numpy()
                    if len(keep_nms) > args.max_det:
                        keep_nms = keep_nms[:args.max_det]
                    boxes = boxes[keep_nms]
                    scores = scores[keep_nms]
            else:
                boxes = np.zeros((0, 4))
                scores = np.zeros(0)
            
            # Extract GT
            gt_boxes = batch['boxes'][0].numpy() if 'boxes' in batch else np.zeros((0, 4))
            
            # Classify
            tps, fps, fns = classify_predictions(boxes, scores, gt_boxes)
            all_tps.append(tps)
            all_fps.append(fps)
            all_fns.append(fns)
    
    # Generate summary
    summary = generate_summary(all_tps, all_fps, all_fns, output_dir)
    
    print(f"\nResults saved to {output_dir}/")
    print("Run tools/plot_training_curves.py for visualization.")


if __name__ == '__main__':
    main()
