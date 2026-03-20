#!/usr/bin/env python3
"""
FPS / Latency Benchmark for SparseVoxelDet.

Measures per-frame inference time (model forward + NMS), throughput (FPS),
GPU memory usage, and percentile latencies (p50, p95, p99).

Usage:
    CUDA_VISIBLE_DEVICES=0 python V2/scripts/benchmark_fps.py \
        --checkpoint runs/sparse_voxel_det/v83_640_seed42/best.pt \
        --config V2/configs/sparse_voxel_det_v83_640.yaml \
        --num-warmup 50 --num-measure 500
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

import spconv.pytorch as spconv
from detection.scripts.sparse_event_dataset import (
    SparseEventDataset, make_collate_fn, create_sparse_tensor
)
from training.models.sparse_voxel_det import SparseVoxelDet


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load model from checkpoint."""
    input_size = config['model'].get('input_size', [640, 640])
    if isinstance(input_size, int):
        input_size = [input_size, input_size]

    model = SparseVoxelDet(
        in_channels=2,
        backbone_size=config['model'].get('backbone_size', 'nano_deep'),
        fpn_channels=config['model'].get('fpn_channels', 128),
        num_classes=config['model'].get('num_classes', 1),
        head_convs=config['model'].get('head_convs', 2),
        prior_prob=config['model'].get('prior_prob', 0.01),
        input_size=tuple(input_size),
        time_bins=config.get('sparse', {}).get('time_bins', 15),
    )

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Try EMA weights first, then regular model weights
    if 'ema_state_dict' in ckpt:
        ema = ckpt['ema_state_dict']
        if isinstance(ema, dict) and 'shadow' in ema:
            ema = ema['shadow']
        model.load_state_dict(ema, strict=True)
        print(f"Loaded EMA weights from {checkpoint_path}")
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        print(f"Loaded model weights from {checkpoint_path}")
    else:
        model.load_state_dict(ckpt, strict=True)
        print(f"Loaded raw state dict from {checkpoint_path}")

    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='FPS Benchmark for SparseVoxelDet')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num-warmup', type=int, default=50)
    parser.add_argument('--num-measure', type=int, default=500)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Device: {device} ({gpu_name})")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load model
    model = load_model(args.checkpoint, config, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Decode params
    eval_cfg = config.get('eval', {})
    score_thresh = float(eval_cfg.get('score_thresh', 0.05))
    nms_thresh = float(eval_cfg.get('nms_thresh', 0.5))
    max_det = int(eval_cfg.get('max_detections', 100))

    # Load test dataset
    data_cfg = config.get('data', {})
    sparse_dir = str(project_root / data_cfg.get('sparse_dir', 'data/datasets/fred_paper_parity_v82/sparse'))
    label_dir = str(project_root / data_cfg.get('label_dir', 'data/datasets/fred_paper_parity/labels'))
    val_split = data_cfg.get('val_split', 'canonical_test')
    time_bins = config['sparse'].get('time_bins', 15)
    expected_time_bins = config.get('data', {}).get('sparse_contract', {}).get('expected_time_bins', None)

    dataset = SparseEventDataset(
        sparse_dir=sparse_dir,
        label_dir=label_dir,
        split=val_split,
        target_size=(640, 640),
        time_bins=expected_time_bins or time_bins,
        augment=False,
    )

    total_needed = args.num_warmup + args.num_measure
    print(f"Dataset size: {len(dataset)} frames (need {total_needed})")

    # Collate function for batch_size=1
    collate_fn = make_collate_fn(time_bins=time_bins)

    from torch.utils.data import DataLoader, Subset
    if len(dataset) > total_needed:
        dataset = Subset(dataset, list(range(total_needed)))

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # Pre-load all batches into GPU memory
    print("Pre-loading frames into GPU memory...")
    batches = []
    for batch in dataloader:
        sp = create_sparse_tensor(batch, device)
        batches.append(sp)
        if len(batches) >= total_needed:
            break

    if len(batches) < total_needed:
        print(f"WARNING: Only got {len(batches)} frames, using with wrap-around")

    num_warmup = min(args.num_warmup, len(batches) // 2)
    num_measure = min(args.num_measure, len(batches) - num_warmup)
    print(f"Benchmarking: {num_warmup} warmup + {num_measure} measurement frames")

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    all_latencies = []
    from torchvision.ops import nms

    for idx in range(num_warmup + num_measure):
        sp = batches[idx % len(batches)]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            preds = model(sp)

            # Post-process (NMS) — same as real inference
            if isinstance(preds, dict) and 'boxes' in preds:
                boxes = preds['boxes']
                scores = preds['scores']
                if scores.numel() > 0:
                    keep = scores > score_thresh
                    boxes = boxes[keep]
                    scores = scores[keep]
                    if scores.numel() > 0:
                        keep_nms = nms(boxes, scores, nms_thresh)
                        if len(keep_nms) > max_det:
                            keep_nms = keep_nms[:max_det]

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        if idx >= num_warmup:
            all_latencies.append(latency_ms)

    latencies = np.array(all_latencies)

    # GPU memory
    mem_allocated = torch.cuda.max_memory_allocated() / 1024**2
    mem_reserved = torch.cuda.max_memory_reserved() / 1024**2

    results = {
        'gpu': gpu_name,
        'checkpoint': str(args.checkpoint),
        'params': param_count,
        'precision': 'FP16 (AMP)',
        'batch_size': 1,
        'num_frames': len(latencies),
        'mean_latency_ms': float(np.mean(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'median_latency_ms': float(np.median(latencies)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99)),
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'fps_from_mean': float(1000.0 / np.mean(latencies)),
        'fps_from_median': float(1000.0 / np.median(latencies)),
        'gpu_mem_allocated_mb': float(mem_allocated),
        'gpu_mem_reserved_mb': float(mem_reserved),
    }

    # Print results
    print("\n" + "=" * 60)
    print("INFERENCE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"GPU: {gpu_name}")
    print(f"Model: SparseVoxelDet ({param_count:,} params)")
    print(f"Precision: FP16 (AMP)")
    print(f"Batch size: 1 (single-frame)")
    print(f"Frames measured: {results['num_frames']}")
    print(f"")
    print(f"Latency (ms):")
    print(f"  Mean:   {results['mean_latency_ms']:.2f} ± {results['std_latency_ms']:.2f}")
    print(f"  Median: {results['median_latency_ms']:.2f}")
    print(f"  P95:    {results['p95_latency_ms']:.2f}")
    print(f"  P99:    {results['p99_latency_ms']:.2f}")
    print(f"  Min:    {results['min_latency_ms']:.2f}")
    print(f"  Max:    {results['max_latency_ms']:.2f}")
    print(f"")
    print(f"Throughput:")
    print(f"  FPS (from mean):   {results['fps_from_mean']:.1f}")
    print(f"  FPS (from median): {results['fps_from_median']:.1f}")
    print(f"")
    print(f"GPU Memory:")
    print(f"  Peak allocated: {results['gpu_mem_allocated_mb']:.0f} MB")
    print(f"  Peak reserved:  {results['gpu_mem_reserved_mb']:.0f} MB")
    print("=" * 60)

    # Save results
    output_dir = Path(args.checkpoint).parent
    results_file = output_dir / 'benchmark_fps.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
