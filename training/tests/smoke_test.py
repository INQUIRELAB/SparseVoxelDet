#!/usr/bin/env python3
"""Smoke test: SparseVoxelDet forward pass + loss computation."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import spconv.pytorch as spconv
from training.models.sparse_voxel_det import SparseVoxelDet
from training.scripts.sparse_voxel_det_loss import SparseVoxelDetLoss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    model = SparseVoxelDet(
        in_channels=2,
        num_classes=1,
        backbone_size="nano_deep",
        fpn_channels=128,
        head_convs=2,
        input_size=(640, 640),
        time_bins=15,
    ).to(device)

    params = model.get_num_params()
    print(f"Parameters: total={params['total']:,}, trainable={params['trainable']:,}")

    # Create loss
    loss_fn = SparseVoxelDetLoss(
        stride=4,
        num_classes=1,
        focal_alpha=0.25,
        focal_gamma=2.0,
        cls_weight=1.0,
        reg_weight=2.0,
        ctr_weight=1.0,
        center_sampling_radius=1.5,
    )

    # Create synthetic sparse input (batch_size=2, T=15, H=640, W=640)
    batch_size = 2
    T, H, W = 15, 640, 640
    N = 5000  # number of active voxels
    spatial_shape = [T, H, W]

    # Random voxel positions
    indices = torch.zeros(N, 4, dtype=torch.int32, device=device)
    indices[:, 0] = torch.randint(0, batch_size, (N,))  # batch
    indices[:, 1] = torch.randint(0, T, (N,))          # time
    indices[:, 2] = torch.randint(0, H, (N,))          # y
    indices[:, 3] = torch.randint(0, W, (N,))          # x
    features = torch.randn(N, 2, device=device)

    sparse_input = spconv.SparseConvTensor(
        features=features,
        indices=indices,
        spatial_shape=spatial_shape,
        batch_size=batch_size,
    )
    print(f"Sparse input: {N} voxels, spatial={spatial_shape}")

    # Forward pass (training mode)
    model.train()
    outputs = model(sparse_input, batch_size, return_loss_inputs=True)
    print(f"Training outputs:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")

    # Create synthetic GT
    gt_boxes = [
        torch.tensor([[100., 100., 200., 200.], [300., 300., 400., 400.]], device=device),
        torch.tensor([[150., 150., 250., 300.]], device=device),
    ]
    gt_labels = [
        torch.tensor([0, 0], device=device),
        torch.tensor([0], device=device),
    ]

    # Compute loss
    losses = loss_fn(outputs, gt_boxes, gt_labels)
    print(f"\nLoss outputs:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.6f}")

    # Forward pass (eval mode)
    model.eval()
    with torch.no_grad():
        # Re-create sparse input (model may have changed algo table state)
        sparse_input2 = spconv.SparseConvTensor(
            features=features.clone(),
            indices=indices.clone(),
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )
        eval_out = model(sparse_input2, batch_size)
    detections = eval_out['detections']
    print(f"\nEval detections: shape={detections.shape}")
    for b in range(batch_size):
        valid = detections[b, :, 4] > 0
        print(f"  Batch {b}: {valid.sum().item()} detections with score > 0")

    # Backward pass test
    model.train()
    sparse_input3 = spconv.SparseConvTensor(
        features=features.clone(),
        indices=indices.clone(),
        spatial_shape=spatial_shape,
        batch_size=batch_size,
    )
    outputs3 = model(sparse_input3, batch_size, return_loss_inputs=True)
    losses3 = loss_fn(outputs3, gt_boxes, gt_labels)
    losses3['loss'].backward()

    # Check gradients
    grad_norms = {}
    total_grad = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            total_grad += gn ** 2
            grad_norms[name] = gn
    total_grad = total_grad ** 0.5
    print(f"\nBackward pass OK. Total grad norm: {total_grad:.4f}")

    # Print top-5 grad norms
    sorted_grads = sorted(grad_norms.items(), key=lambda x: -x[1])[:5]
    for name, gn in sorted_grads:
        print(f"  {name}: {gn:.4f}")

    print("\n✓ Smoke test PASSED")


if __name__ == "__main__":
    main()
