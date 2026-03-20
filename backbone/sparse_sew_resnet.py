#!/usr/bin/env python3
"""
Sparse SEW-ResNet Backbone with spconv.

Sparse convolutions — only non-zero voxels are processed.
Uses spconv library for efficient sparse 3D convolutions.

Architecture:
- Input: spconv.SparseConvTensor [N, 2] (ON/OFF features)
- Backbone: Sparse ResNet blocks
- Output: 3 scale features for FPN

Key insight from research:
- Keep backbone sparse for efficiency
- Convert to dense only at detection head
- "Spiking detection heads reduce performance"

Size variants:
- nano:      base=32, blocks=[1,1,1,1], 3.6M backbone params (original)
- nano_deep: base=32, blocks=[2,2,2,1], SE on L3/L4, 4.8M backbone params (recommended upgrade)
- small:     base=64, blocks=[2,2,2,2], 33.1M backbone params
- medium:    base=96, blocks=[2,3,3,2], 84.5M backbone params
- large:     base=128, blocks=[3,4,6,3], (not recommended - OOM)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from typing import List, Tuple, Optional


class SparseGroupNorm(nn.Module):
    """
    GroupNorm for sparse features - NO running statistics.

    Unlike BatchNorm, GroupNorm:
    - Doesn't accumulate running_mean/running_var
    - Can't be corrupted by NaN batches
    - Works with any batch size (even 1)

    For sparse features [N, C], we use LayerNorm over channels.
    This normalizes each voxel's features independently.
    """

    def __init__(self, num_features: int, num_groups: int = 8):
        super().__init__()
        self.num_features = num_features
        # Use LayerNorm for sparse features (normalizes over channel dim)
        # This is equivalent to GroupNorm with num_groups=num_features
        self.norm = nn.LayerNorm(num_features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # LayerNorm works with any number of elements
        if features.shape[0] == 0:
            return features
        return self.norm(features)


# Alias for backward compatibility
SafeSparseBatchNorm = SparseGroupNorm


def compute_linear_idx(indices: torch.Tensor, spatial_shape: List[int], batch_size: int) -> torch.Tensor:
    """
    Convert 4D indices (batch, t, y, x) to linear indices for sorting/lookup.

    This replaces hardcoded magic numbers with proper dynamic computation
    based on actual spatial dimensions.

    Args:
        indices: [N, 4] tensor of (batch, t, y, x) indices
        spatial_shape: [T, H, W] spatial dimensions
        batch_size: Number of batches

    Returns:
        [N] tensor of linear indices
    """
    T, H, W = spatial_shape
    stride_b = T * H * W
    stride_t = H * W
    stride_h = W

    idx = indices.long()
    return (idx[:, 0] * stride_b + idx[:, 1] * stride_t +
            idx[:, 2] * stride_h + idx[:, 3])


def sort_sparse_tensor(x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
    """
    Sort a SparseConvTensor's indices lexicographically (batch, t, y, x).

    This is required before strided convolutions in spconv to ensure
    consistent index ordering and correct results.

    Args:
        x: Input SparseConvTensor

    Returns:
        SparseConvTensor with sorted indices and corresponding features
    """
    if x.features.shape[0] == 0:
        return x

    linear_idx = compute_linear_idx(x.indices, x.spatial_shape, x.batch_size)
    _, sorted_order = torch.sort(linear_idx)

    return spconv.SparseConvTensor(
        features=x.features[sorted_order],
        indices=x.indices[sorted_order].int(),
        spatial_shape=x.spatial_shape,
        batch_size=x.batch_size
    )


def sparse_add(a: spconv.SparseConvTensor, b: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
    """
    Add two SparseConvTensors WITHOUT converting to dense.

    This is critical for memory efficiency - dense conversion explodes memory:
    [B=8, C=128, T=33, H=320, W=320] = 13.7 GB per tensor!

    Algorithm:
    1. Convert 4D indices to linear indices for fast lookup
    2. Concatenate indices and features from both tensors
    3. Use scatter_add to sum features at duplicate indices
    4. Remove zero-feature voxels to maintain sparsity

    Args:
        a: First SparseConvTensor
        b: Second SparseConvTensor (must have same spatial_shape and batch_size)

    Returns:
        SparseConvTensor with combined features
    """
    # Handle edge cases
    if a.features.shape[0] == 0:
        return b
    if b.features.shape[0] == 0:
        return a

    device = a.features.device
    spatial_shape = a.spatial_shape  # [T, H, W]
    batch_size = a.batch_size

    # Compute strides for linear indexing
    T, H, W = spatial_shape
    stride_b = T * H * W
    stride_t = H * W
    stride_h = W

    # Convert indices to linear form: batch * (T*H*W) + t * (H*W) + h * W + w
    idx_a = a.indices.long()  # [N_a, 4] = (batch, t, h, w)
    idx_b = b.indices.long()  # [N_b, 4]

    linear_a = (idx_a[:, 0] * stride_b + idx_a[:, 1] * stride_t +
                idx_a[:, 2] * stride_h + idx_a[:, 3])
    linear_b = (idx_b[:, 0] * stride_b + idx_b[:, 1] * stride_t +
                idx_b[:, 2] * stride_h + idx_b[:, 3])

    # Concatenate linear indices and features
    all_linear = torch.cat([linear_a, linear_b], dim=0)  # [N_a + N_b]
    all_feats = torch.cat([a.features, b.features], dim=0)  # [N_a + N_b, C]

    # Find unique linear indices and their mapping
    unique_linear, inverse = torch.unique(all_linear, return_inverse=True)

    # Scatter-add features to unique indices
    C = all_feats.shape[1]
    combined_feats = torch.zeros(unique_linear.shape[0], C, device=device, dtype=all_feats.dtype)
    combined_feats.scatter_add_(0, inverse.unsqueeze(1).expand(-1, C), all_feats)

    # Convert unique linear indices back to 4D
    combined_indices = torch.zeros(unique_linear.shape[0], 4, device=device, dtype=torch.int32)
    combined_indices[:, 0] = (unique_linear // stride_b).int()
    remainder = unique_linear % stride_b
    combined_indices[:, 1] = (remainder // stride_t).int()
    remainder = remainder % stride_t
    combined_indices[:, 2] = (remainder // stride_h).int()
    combined_indices[:, 3] = (remainder % stride_h).int()

    # Optional: Remove near-zero voxels to maintain sparsity
    # (activation will handle this, but we can save memory)
    nonzero_mask = combined_feats.abs().sum(dim=1) > 1e-8
    if nonzero_mask.sum() < combined_feats.shape[0]:
        combined_feats = combined_feats[nonzero_mask]
        combined_indices = combined_indices[nonzero_mask]

    return spconv.SparseConvTensor(
        features=combined_feats,
        indices=combined_indices,
        spatial_shape=spatial_shape,
        batch_size=batch_size
    )


class SparseActivation(nn.Module):
    """
    Activation for sparse tensors that MAINTAINS SPARSITY.

    Uses ReLU (or LeakyReLU) which:
    - Maintains zero values as zeros (preserves sparsity)
    - Has well-understood gradients
    - Is what SEW-ResNet uses in practice

    NOTE: Sigmoid outputs non-zero for all inputs, destroying sparsity benefit.
    """

    def __init__(self, negative_slope: float = 0.0):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """Apply ReLU activation to sparse features."""
        feats = x.features

        if self.negative_slope > 0:
            activated = F.leaky_relu(feats, negative_slope=self.negative_slope)
        else:
            activated = F.relu(feats)

        return x.replace_feature(activated)


class SparseSEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for sparse features.

    Operates on sparse feature vectors [N, C]:
    1. Squeeze: global average pooling across all N voxels -> [1, C]
    2. Excitation: FC -> ReLU -> FC -> Sigmoid -> [1, C]
    3. Scale: multiply each voxel's features by channel weights

    This is a lightweight channel attention mechanism (~0.1% params overhead)
    that helps the backbone learn which channels are most informative
    at each spatial scale.

    Note: Unlike dense SE which pools per-image, sparse SE pools across
    ALL voxels in the batch. This is intentional - with sparse data,
    per-image pooling would give unstable statistics due to varying
    voxel counts per image.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, channels)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        feats = x.features  # [N, C]
        if feats.shape[0] == 0:
            return x

        # Squeeze: global average across all voxels
        squeezed = feats.mean(dim=0, keepdim=True)  # [1, C]

        # Excitation: FC -> ReLU -> FC -> Sigmoid
        scale = F.relu(self.fc1(squeezed))
        scale = torch.sigmoid(self.fc2(scale))  # [1, C]

        # Scale features (broadcasts across N voxels)
        return x.replace_feature(feats * scale)


class SparseBasicBlock(nn.Module):
    """
    Sparse ResNet basic block.

    Structure:
    x -> Conv -> BN -> LIF -> Conv -> BN -> + -> LIF
    |___________________________________|

    Uses SubMConv3d (submanifold sparse conv) to preserve sparsity.
    """
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        use_se: bool = False,
        se_reduction: int = 4
    ):
        super().__init__()

        # First conv: SubMConv3d preserves sparsity pattern
        self.conv1 = spconv.SubMConv3d(
            in_channels, out_channels, 3,
            bias=False
        )
        self.bn1 = SafeSparseBatchNorm(out_channels)
        self.lif1 = SparseActivation()

        # Second conv
        if stride > 1:
            # Use SparseConv3d with stride for downsampling
            # Use (1, stride, stride) to only downsample spatial dimensions
            self.conv2 = spconv.SparseConv3d(
                out_channels, out_channels, 3,
                stride=(1, stride, stride), padding=1, bias=False
            )
        else:
            self.conv2 = spconv.SubMConv3d(
                out_channels, out_channels, 3,
                bias=False
            )
        self.bn2 = SafeSparseBatchNorm(out_channels)

        # Optional SE attention (applied after conv2 BN, before residual add)
        self.se = SparseSEBlock(out_channels, se_reduction) if use_se else None

        self.lif2 = SparseActivation()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        identity = x

        # First conv block
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = self.lif1(out)

        # Sort indices before strided conv — spconv requires sorted
        # indices for correct strided convolution output.
        if self.stride > 1:
            out = sort_sparse_tensor(out)

        # Second conv block
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        # Channel attention (if enabled)
        if self.se is not None:
            out = self.se(out)

        # Shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add residual using SPARSE addition (no dense conversion!)
        # Dense conversion would explode memory: [B=8, C=128, T=33, H=320, W=320] = 13.7 GB
        out = sparse_add(out, identity)

        out = self.lif2(out)

        return out


class SparseDownsample(nn.Module):
    """Downsample module for sparse tensors."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        # Use (1, stride, stride) to only downsample spatial dimensions
        self.conv = spconv.SparseConv3d(
            in_channels, out_channels, 1,
            stride=(1, stride, stride), bias=False
        )
        self.bn = SafeSparseBatchNorm(out_channels)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        # Sort indices before strided conv (required by spconv)
        x = sort_sparse_tensor(x)

        out = self.conv(x)
        # GroupNorm works with any batch size (no running stats)
        out = out.replace_feature(self.bn(out.features))
        return out


class SparseStem(nn.Module):
    """
    Sparse stem: Initial conv layer.

    Input: [N, 2] features (ON/OFF counts)
    Output: [N, base_channels] features

    Args:
        in_channels: Input feature channels (2 for ON/OFF)
        out_channels: Output channels (base_channels)
        stride: Conv stride. Use (1, 2, 2) for 2x spatial downsample
                (shifts backbone output strides from [2,4,8] to [4,8,16]).
                Use 1 for no spatial downsample (V1 behavior).
    """

    def __init__(self, in_channels: int = 2, out_channels: int = 64,
                 stride: tuple = 1):
        super().__init__()

        # Initial sparse conv
        self.conv = spconv.SparseConv3d(
            in_channels, out_channels, 3,
            stride=stride, padding=1, bias=False
        )
        self.bn = SafeSparseBatchNorm(out_channels)
        self.lif = SparseActivation()

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        out = self.conv(x)
        out = out.replace_feature(self.bn(out.features))
        out = self.lif(out)
        return out


class SparseSEWResNet(nn.Module):
    """
    Sparse SEW-ResNet backbone.

    Processes sparse event voxels using sparse convolutions.
    Only non-zero voxels are computed - O(N) instead of O(H*W*T).

    Architecture:
    - Stem: 2 -> 64 channels
    - Layer1: 64 -> 64, stride 1 (same resolution)
    - Layer2: 64 -> 128, stride 2 (2x downsample)
    - Layer3: 128 -> 256, stride 2 (4x downsample)
    - Layer4: 256 -> 512, stride 2 (8x downsample)

    Returns features at 3 scales for FPN: [C3, C4, C5]
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        num_blocks: List[int] = [2, 2, 2, 2],
        size: str = 'nano',
        stem_stride: tuple = 1,
        use_se: Optional[List[bool]] = None,
        se_reduction: int = 4
    ):
        """
        Args:
            in_channels: Input event channels (2 = ON/OFF)
            base_channels: Base channel width (overridden by size)
            num_blocks: Blocks per layer (overridden by size)
            size: Size variant ('nano', 'nano_deep', 'small', 'medium', 'large')
            stem_stride: Stride for stem conv. Use (1, 2, 2) for 2x spatial
                         downsample (output strides [4,8,16] instead of [2,4,8]).
            use_se: Per-layer SE attention flags [L1, L2, L3, L4].
                    Overridden by size config if available.
            se_reduction: SE reduction ratio (default 4).
        """
        super().__init__()

        # Size variants
        # 'se' indicates which layers use Squeeze-and-Excitation attention
        size_configs = {
            'nano':      {'base': 32,  'blocks': [1, 1, 1, 1], 'se': [False, False, False, False]},
            'nano_deep': {'base': 32,  'blocks': [2, 2, 2, 1], 'se': [False, False, True, True]},
            'small':     {'base': 64,  'blocks': [2, 2, 2, 2], 'se': [False, False, False, False]},
            'medium':    {'base': 96,  'blocks': [2, 3, 3, 2], 'se': [False, False, True, True]},
            'large':     {'base': 128, 'blocks': [3, 4, 6, 3], 'se': [False, False, True, True]}
        }

        if size in size_configs:
            base_channels = size_configs[size]['base']
            num_blocks = size_configs[size]['blocks']
            use_se = size_configs[size]['se']
        elif use_se is None:
            use_se = [False, False, False, False]

        self.in_channels = in_channels
        self.base_channels = base_channels

        # Stem
        self.stem = SparseStem(in_channels, base_channels, stride=stem_stride)

        # Residual layers
        self.layer1 = self._make_layer(
            base_channels, base_channels, num_blocks[0], stride=1,
            use_se=use_se[0], se_reduction=se_reduction)
        self.layer2 = self._make_layer(
            base_channels, base_channels * 2, num_blocks[1], stride=2,
            use_se=use_se[1], se_reduction=se_reduction)
        self.layer3 = self._make_layer(
            base_channels * 2, base_channels * 4, num_blocks[2], stride=2,
            use_se=use_se[2], se_reduction=se_reduction)
        self.layer4 = self._make_layer(
            base_channels * 4, base_channels * 8, num_blocks[3], stride=2,
            use_se=use_se[3], se_reduction=se_reduction)

        # Output channels for FPN
        self.out_channels = [base_channels * 2, base_channels * 4, base_channels * 8]

        # Initialize weights
        self._init_weights()

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        use_se: bool = False,
        se_reduction: int = 4
    ) -> nn.Sequential:
        """Create a residual layer with optional SE attention."""
        # First block may have stride and channel change
        downsample = None
        if stride != 1 or in_channels != out_channels:
            # Create downsample as a custom module that handles sparse properly
            downsample = SparseDownsample(in_channels, out_channels, stride)

        layers = [SparseBasicBlock(
            in_channels, out_channels, stride, downsample,
            use_se=use_se, se_reduction=se_reduction)]

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(SparseBasicBlock(
                out_channels, out_channels,
                use_se=use_se, se_reduction=se_reduction))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (spconv.SparseConv3d, spconv.SubMConv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SparseGroupNorm):
                # LayerNorm has weight and bias directly
                nn.init.constant_(m.norm.weight, 1)
                nn.init.constant_(m.norm.bias, 0)
            elif isinstance(m, SparseSEBlock):
                # Initialize SE fc2 bias near zero so initial scale ~ 0.5 (sigmoid(0))
                # This means SE starts as near-identity, not adding noise
                nn.init.kaiming_normal_(m.fc1.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.fc1.bias, 0)
                nn.init.constant_(m.fc2.weight, 0)
                nn.init.constant_(m.fc2.bias, 0)  # sigmoid(0) = 0.5 -> scales by 0.5

    def forward(self, x: spconv.SparseConvTensor) -> List[spconv.SparseConvTensor]:
        """
        Forward pass.

        Args:
            x: SparseConvTensor with shape [N, in_channels]

        Returns:
            List of 3 SparseConvTensors at different scales [C3, C4, C5]
        """
        x = self.stem(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)   # 2x downsample
        c3 = self.layer3(c2)   # 4x downsample
        c4 = self.layer4(c3)   # 8x downsample

        # Return 3 scales for FPN
        return [c2, c3, c4]


def test_sparse_backbone():
    """Test the sparse backbone with all size variants."""
    print("Testing SparseSEWResNet...")

    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test all practical size variants
    for size in ['nano', 'nano_deep']:
        print(f"\n{'='*60}")
        print(f"Size variant: {size}")
        print(f"{'='*60}")

        # Create model
        model = SparseSEWResNet(
            in_channels=2,
            size=size,
            stem_stride=(1, 2, 2)
        ).to(device)

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")
        print(f"Output channels: {model.out_channels}")

        # Per-layer breakdown
        for name, module in model.named_children():
            n = sum(p.numel() for p in module.parameters())
            if n > 0:
                print(f"  {name}: {n:,}")

        # Check for SE blocks
        se_count = sum(1 for m in model.modules() if isinstance(m, SparseSEBlock))
        print(f"  SE blocks: {se_count}")

        # Create dummy sparse input
        batch_size = 4
        spatial_shape = [15, 640, 640]  # T, H, W
        n_voxels = 10000

        indices = torch.zeros((n_voxels, 4), dtype=torch.int32, device=device)
        indices[:, 0] = torch.randint(0, batch_size, (n_voxels,))
        indices[:, 1] = torch.randint(0, 15, (n_voxels,))
        indices[:, 2] = torch.randint(0, 640, (n_voxels,))
        indices[:, 3] = torch.randint(0, 640, (n_voxels,))
        features = torch.randn(n_voxels, 2, device=device)

        sparse_input = spconv.SparseConvTensor(
            features=features,
            indices=indices,
            spatial_shape=spatial_shape,
            batch_size=batch_size
        )

        # Forward pass
        with torch.no_grad():
            outputs = model(sparse_input)

        print(f"\nOutputs (3 scales):")
        for i, out in enumerate(outputs):
            print(f"  C{i+3}: features {out.features.shape}, spatial {out.spatial_shape}")

        # Benchmark (only on CUDA)
        if device.type == 'cuda':
            import time

            # Warmup
            for _ in range(5):
                _ = model(sparse_input)
            torch.cuda.synchronize()

            # Time
            start = time.perf_counter()
            for _ in range(20):
                _ = model(sparse_input)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / 20 * 1000

            print(f"\nTiming: {elapsed:.2f} ms per forward pass")
            print(f"Throughput: {1000/elapsed:.0f} FPS")

        print(f"\nSUCCESS: {size} working!")


if __name__ == '__main__':
    test_sparse_backbone()
