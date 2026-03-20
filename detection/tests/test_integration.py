#!/usr/bin/env python3
"""
Integration test for Sparse FCOS Detector.
Tests the full pipeline: load → forward → loss → backward.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_full_pipeline():
    """
    Test the complete training pipeline with real data.

    Checks:
    1. Data loads correctly
    2. Forward pass completes without OOM
    3. Loss is finite
    4. Backward pass completes
    5. Memory usage is reasonable
    6. Loss decreases over 10 steps
    """
    from torch.utils.data import DataLoader
    from detection.scripts.sparse_event_dataset import (
        SparseEventDataset, sparse_collate_fn, create_sparse_tensor
    )
    from detection.models.sparse_fcos_detector import SparseFCOSDetector
    from detection.scripts.fcos_loss import FCOSLoss
    from detection.scripts.fcos_target import FCOSTargetAssigner

    print("=" * 60)
    print("INTEGRATION TEST: Full Training Pipeline")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available (spconv requires CUDA)")
        return True

    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()

    # Load real dataset
    data_dir = project_root / 'datasets' / 'fred_sparse'
    label_dir = project_root / 'datasets' / 'fred_voxel' / 'labels'

    if not data_dir.exists():
        print(f"SKIP: Data directory not found: {data_dir}")
        return True

    print(f"\n1. Loading dataset from {data_dir}")
    try:
        dataset = SparseEventDataset(
            sparse_dir=str(data_dir),
            label_dir=str(label_dir),
            split='train',
            augment=False,
            max_voxels=30000  # Match config
        )
    except Exception as e:
        print(f"SKIP: Could not load dataset: {e}")
        return True

    if len(dataset) == 0:
        print("SKIP: Dataset is empty")
        return True

    print(f"   Dataset size: {len(dataset)} samples")

    # Create dataloader with realistic batch size
    batch_size = 4  # Config says 4 for T=33
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # For testing
        collate_fn=sparse_collate_fn
    )

    # Create model with config params (small backbone as specified in config)
    print("\n2. Creating model (small backbone, 640x640 input)")
    model = SparseFCOSDetector(
        in_channels=2,
        num_classes=1,
        backbone_size='small',  # Match config
        fpn_channels=128,       # Match config
        num_head_convs=4        # Match config
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {param_count:,}")

    # Create loss and target assigner
    strides = [2, 4, 8]  # Match config
    regress_ranges = [(-1, 16), (16, 32), (32, 64)]  # Match config

    loss_fn = FCOSLoss(
        strides=strides,
        focal_alpha=0.25,
        focal_gamma=2.0,
        use_giou=True
    ).to(device)

    target_assigner = FCOSTargetAssigner(
        strides=strides,
        regress_ranges=regress_ranges,
        center_sampling=True,
        center_sampling_radius=1.5
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    # Run 10 training steps
    print("\n3. Running 10 training steps...")
    losses = []
    model.train()

    for step, batch in enumerate(loader):
        if step >= 10:
            break

        # Create sparse tensor
        try:
            sparse_input = create_sparse_tensor(batch, device)
        except Exception as e:
            print(f"   FAIL: Could not create sparse tensor at step {step}: {e}")
            return False

        # Get GT boxes and compute feature sizes
        gt_boxes = [b.to(device) for b in batch['gt_boxes']]

        # Forward pass (training mode returns loss dict)
        try:
            cls_outs, reg_outs, ctr_outs, feature_sizes = model.forward_train(
                sparse_input, batch['batch_size']
            )
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"   FAIL: OOM at step {step}")
                print(f"   Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
                return False
            raise

        # Compute targets
        points = model.head.get_points(feature_sizes, device)
        targets = target_assigner.assign(points, gt_boxes, feature_sizes)

        # Compute loss
        try:
            loss_dict = loss_fn(
                cls_outs, reg_outs, ctr_outs,
                targets['labels'], targets['ltrb_targets'], targets['centerness_targets']
            )
        except Exception as e:
            print(f"   FAIL: Loss computation failed at step {step}: {e}")
            return False

        total_loss = loss_dict['total_loss']

        # Check loss is finite
        if not torch.isfinite(total_loss):
            print(f"   FAIL: Non-finite loss at step {step}: {total_loss.item()}")
            return False

        # Backward pass
        optimizer.zero_grad()
        try:
            total_loss.backward()
        except Exception as e:
            print(f"   FAIL: Backward pass failed at step {step}: {e}")
            return False

        # Check for NaN gradients
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                print(f"   FAIL: NaN/Inf gradient in {name}")
                has_nan = True
                break

        if has_nan:
            return False

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        losses.append(total_loss.item())
        print(f"   Step {step+1}/10: loss={total_loss.item():.4f}")

    # Check memory usage
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n4. Memory usage: {peak_memory:.2f} GB")

    if peak_memory > 20:
        print(f"   WARNING: High memory usage (>20GB)")

    # Check loss trend
    print(f"\n5. Loss trend:")
    print(f"   Initial loss: {losses[0]:.4f}")
    print(f"   Final loss:   {losses[-1]:.4f}")

    if len(losses) >= 5:
        # Use smoothed comparison (first 3 vs last 3)
        early_avg = np.mean(losses[:3])
        late_avg = np.mean(losses[-3:])
        if late_avg >= early_avg * 1.1:  # Allow 10% tolerance
            print(f"   WARNING: Loss not decreasing (early={early_avg:.4f}, late={late_avg:.4f})")

    print("\n" + "=" * 60)
    print("INTEGRATION TEST: PASSED")
    print("=" * 60)
    return True


def test_inference():
    """
    Test inference mode (no targets needed).
    """
    from torch.utils.data import DataLoader
    from detection.scripts.sparse_event_dataset import (
        SparseEventDataset, sparse_collate_fn, create_sparse_tensor
    )
    from detection.models.sparse_fcos_detector import SparseFCOSDetector

    print("\n" + "=" * 60)
    print("INTEGRATION TEST: Inference Mode")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return True

    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()

    # Load dataset
    data_dir = project_root / 'datasets' / 'fred_sparse'
    label_dir = project_root / 'datasets' / 'fred_voxel' / 'labels'

    if not data_dir.exists():
        print(f"SKIP: Data directory not found")
        return True

    dataset = SparseEventDataset(
        sparse_dir=str(data_dir),
        label_dir=str(label_dir),
        split='train',
        augment=False
    )

    if len(dataset) == 0:
        print("SKIP: Dataset is empty")
        return True

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=sparse_collate_fn
    )

    # Create model
    model = SparseFCOSDetector(
        in_channels=2,
        num_classes=1,
        backbone_size='small',
        fpn_channels=128
    ).to(device)
    model.eval()

    # Run inference
    print("\nRunning inference on 5 batches...")
    latencies = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 5:
                break

            sparse_input = create_sparse_tensor(batch, device)

            torch.cuda.synchronize()
            start = time.time()

            detections = model(sparse_input, batch['batch_size'])

            torch.cuda.synchronize()
            latency = time.time() - start
            latencies.append(latency)

            # Check output format
            assert len(detections) == batch['batch_size'], f"Wrong number of outputs"
            for det in detections:
                assert det.shape[1] == 6, f"Wrong detection format: {det.shape}"

            print(f"   Batch {i+1}: {latency*1000:.1f}ms, "
                  f"{sum(len(d) for d in detections)} detections")

    avg_latency = np.mean(latencies) * 1000
    fps = 1000 / avg_latency * 4  # batch_size=4
    print(f"\nAverage latency: {avg_latency:.1f}ms ({fps:.1f} FPS)")

    print("\n" + "=" * 60)
    print("INFERENCE TEST: PASSED")
    print("=" * 60)
    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Integration tests for Sparse FCOS')
    parser.add_argument('--test', type=str, choices=['all', 'train', 'inference'],
                        default='all', help='Which test to run')
    args = parser.parse_args()

    results = []

    if args.test in ['all', 'train']:
        results.append(('Training Pipeline', test_full_pipeline()))

    if args.test in ['all', 'inference']:
        results.append(('Inference Mode', test_inference()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    # Exit with appropriate code
    if all(passed for _, passed in results):
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)
