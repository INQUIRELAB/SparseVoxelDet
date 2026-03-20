#!/usr/bin/env python3
"""
Sparse FCOS Detector — End-to-End Anchor-Free Event-Based Detector.

Architecture:
1. Sparse SEW-ResNet backbone (sparse 3D convolutions, stem_stride=(1,2,2))
2. TemporalGroupBridge (preserves motion direction via early/mid/late groups)
3. Dense FPN neck (standard feature pyramid, 64ch)
4. FCOS detection head (anchor-free, per-pixel prediction, norm_on_bbox=True)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from typing import List, Tuple, Dict, Optional
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backbone.sparse_sew_resnet import SparseSEWResNet
from detection.models.fcos_head import FCOSHead, generate_points, ltrb_to_xyxy


class SparseToDenseBridge(nn.Module):
    """
    Convert sparse features to dense tensors for FPN.

    Each scale's sparse features are scattered into a dense tensor.
    Temporal dimension is summed (aggregates across time bins).

    This is the key bottleneck where we go from O(N) sparse to O(HW) dense.
    We want to stay sparse as long as possible, then densify for the detection head.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        spatial_shapes: List[Tuple[int, int]]
    ):
        """
        Args:
            in_channels: Channels from backbone at each scale
            out_channels: Target channels for FPN
            spatial_shapes: (H, W) at each scale
        """
        super().__init__()
        self.spatial_shapes = spatial_shapes

        # 1x1 conv adapters for channel adjustment
        self.adapters = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            for in_ch, out_ch in zip(in_channels, out_channels)
        ])

    def _sparse_to_dense(
        self,
        sparse: spconv.SparseConvTensor,
        target_h: int,
        target_w: int,
        batch_size: int,
        level_idx: int = -1
    ) -> torch.Tensor:
        """
        Convert SparseConvTensor to dense [B, C, H, W] tensor.

        Args:
            sparse: SparseConvTensor with indices [N, 4] = (batch, t, y, x)
            target_h: Target height
            target_w: Target width
            batch_size: Batch size
            level_idx: FPN level index for error messages

        Returns:
            Dense tensor [B, C, H, W] with temporal aggregation
        """
        C = sparse.features.shape[1]
        device = sparse.features.device
        dtype = sparse.features.dtype

        # Validate spatial shapes match
        # sparse.spatial_shape is [T, H, W]
        actual_h = sparse.spatial_shape[1]
        actual_w = sparse.spatial_shape[2]
        if actual_h != target_h or actual_w != target_w:
            raise ValueError(
                f"Spatial shape mismatch at level {level_idx}! "
                f"Backbone output: ({actual_h}, {actual_w}), "
                f"Bridge expects: ({target_h}, {target_w}). "
                f"Check that detector strides match backbone output strides."
            )

        # Initialize dense tensor in CHANNELS-LAST layout [B, H, W, C]
        # Using [B,C,H,W].view(-1,C) is WRONG — C is dim 1,
        # so view(-1,C) groups every C consecutive memory elements as a row,
        # mixing spatial dims with channels. Channels-last ensures view(-1,C)
        # gives [B*H*W, C] where each row is a unique (b,y,x) location.
        dense = torch.zeros(
            batch_size, target_h, target_w, C,
            device=device, dtype=dtype
        )

        if sparse.features.shape[0] == 0:
            return dense.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        # Get indices and features
        indices = sparse.indices  # [N, 4] = (batch, t, y, x)
        features = sparse.features  # [N, C]

        # Extract coordinates (ignore temporal, sum over it)
        batch_idx = indices[:, 0].long()
        y_idx = indices[:, 2].long()
        x_idx = indices[:, 3].long()

        # Validate indices are in range (should be guaranteed by shape check above)
        assert y_idx.max() < target_h, f"y_idx out of range: {y_idx.max()} >= {target_h}"
        assert x_idx.max() < target_w, f"x_idx out of range: {x_idx.max()} >= {target_w}"

        # Scatter features to dense (sum over temporal dimension)
        flat_idx = batch_idx * target_h * target_w + y_idx * target_w + x_idx

        # Use scatter_add for efficient accumulation
        flat_dense = dense.view(-1, C)
        flat_dense.scatter_add_(0, flat_idx.unsqueeze(1).expand(-1, C), features)

        return dense.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

    def forward(
        self,
        sparse_features: List[spconv.SparseConvTensor],
        batch_size: int
    ) -> List[torch.Tensor]:
        """
        Convert list of sparse features to dense.

        Args:
            sparse_features: [C2, C3, C4] from backbone
            batch_size: Batch size

        Returns:
            [P2, P3, P4] dense feature maps
        """
        dense_features = []

        for level_idx, (sparse, (h, w), adapter) in enumerate(zip(
            sparse_features, self.spatial_shapes, self.adapters
        )):
            dense = self._sparse_to_dense(sparse, h, w, batch_size, level_idx)
            dense = adapter(dense)
            dense_features.append(dense)

        return dense_features


class TemporalGroupBridge(nn.Module):
    """
    Convert sparse features to dense tensors, preserving temporal information.

    Instead of summing all T time bins (destroying motion direction),
    splits T into groups (early/mid/late), sums within each group,
    concatenates along channels, then reduces with 1x1 conv.

    For T=15, groups=3: [0-4] early, [5-9] mid, [10-14] late.
    This gives the FPN 3x richer temporal context while keeping
    the same output channel count.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        spatial_shapes: List[Tuple[int, int]],
        num_groups: int = 3,
        time_bins: int = 15
    ):
        """
        Args:
            in_channels: Channels from backbone at each scale
            out_channels: Target channels for FPN (same as in_channels typically)
            spatial_shapes: (H, W) at each scale
            num_groups: Number of temporal groups
            time_bins: Total number of time bins
        """
        super().__init__()
        self.spatial_shapes = spatial_shapes
        self.num_groups = num_groups
        self.time_bins = time_bins

        # Group boundaries: T=15, groups=3 -> [0-4], [5-9], [10-14]
        bins_per_group = time_bins // num_groups
        self.group_boundaries = []
        for g in range(num_groups):
            start = g * bins_per_group
            end = (g + 1) * bins_per_group if g < num_groups - 1 else time_bins
            self.group_boundaries.append((start, end))

        # 1x1 conv: C*groups -> out_channels (channel reduction)
        self.channel_reducers = nn.ModuleList([
            nn.Conv2d(in_ch * num_groups, out_ch, 1)
            for in_ch, out_ch in zip(in_channels, out_channels)
        ])

    def _sparse_to_dense_grouped(
        self,
        sparse: spconv.SparseConvTensor,
        target_h: int,
        target_w: int,
        batch_size: int
    ) -> torch.Tensor:
        """
        Convert SparseConvTensor to grouped dense [B, C*groups, H, W] tensor.

        For each temporal group, sums sparse features within that time range,
        then concatenates all groups along the channel dimension.
        """
        C = sparse.features.shape[1]
        device = sparse.features.device
        dtype = sparse.features.dtype

        group_tensors = []
        for g_start, g_end in self.group_boundaries:
            # CHANNELS-LAST: [B, H, W, C] for correct scatter indexing
            dense = torch.zeros(batch_size, target_h, target_w, C,
                                device=device, dtype=dtype)
            if sparse.features.shape[0] > 0:
                t_idx = sparse.indices[:, 1].long()
                mask = (t_idx >= g_start) & (t_idx < g_end)
                if mask.any():
                    idx = sparse.indices[mask]
                    feat = sparse.features[mask]
                    flat = (idx[:, 0].long() * target_h * target_w +
                            idx[:, 2].long() * target_w +
                            idx[:, 3].long())
                    dense.view(-1, C).scatter_add_(
                        0, flat.unsqueeze(1).expand(-1, C), feat)
            group_tensors.append(dense.permute(0, 3, 1, 2).contiguous())

        return torch.cat(group_tensors, dim=1)  # [B, C*groups, H, W]

    def forward(
        self,
        sparse_features: List[spconv.SparseConvTensor],
        batch_size: int
    ) -> List[torch.Tensor]:
        """
        Convert list of sparse features to dense with temporal grouping.

        Args:
            sparse_features: [C2, C3, C4] from backbone
            batch_size: Batch size

        Returns:
            [P2, P3, P4] dense feature maps with temporal info preserved
        """
        dense_features = []
        for sparse, (h, w), reducer in zip(
            sparse_features, self.spatial_shapes, self.channel_reducers
        ):
            # Validate spatial shapes match
            actual_h = sparse.spatial_shape[1]
            actual_w = sparse.spatial_shape[2]
            if actual_h != h or actual_w != w:
                raise ValueError(
                    f"Spatial shape mismatch! Backbone: ({actual_h}, {actual_w}), "
                    f"Bridge expects: ({h}, {w}). "
                    f"Check that strides match backbone output."
                )
            dense = self._sparse_to_dense_grouped(sparse, h, w, batch_size)
            dense = reducer(dense)
            dense_features.append(dense)
        return dense_features


class TemporalAttentionBridge(nn.Module):
    """
    Convert sparse features to dense tensors with learnable temporal aggregation.

    Instead of fixed groups (TemporalGroupBridge), this learns which temporal
    patterns matter using:
    1. Scatter sparse features to [B, C, T, H, W]
    2. Conv1d temporal smoothing (kernel=3) per spatial location
    3. Learned attention weights [B, 1, T, H, W]
    4. Weighted sum -> [B, C, H, W]

    ~10K extra parameters, negligible cost vs fixed grouping.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        spatial_shapes: List[Tuple[int, int]],
        time_bins: int = 15,
    ):
        super().__init__()
        self.spatial_shapes = spatial_shapes
        self.time_bins = time_bins

        # Per-level temporal conv + attention
        self.temporal_convs = nn.ModuleList()
        self.attention_convs = nn.ModuleList()
        self.adapters = nn.ModuleList()

        for in_ch, out_ch in zip(in_channels, out_channels):
            # Temporal smoothing: operates on T dimension
            self.temporal_convs.append(
                nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=1)
            )
            # Attention: produces per-timestep weights
            self.attention_convs.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, 1, kernel_size=1),
                )
            )
            # Channel adapter
            if in_ch != out_ch:
                self.adapters.append(nn.Conv2d(in_ch, out_ch, 1))
            else:
                self.adapters.append(nn.Identity())

    def _sparse_to_5d(
        self,
        sparse: spconv.SparseConvTensor,
        target_h: int,
        target_w: int,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Scatter sparse features to dense [B, C, T, H, W] tensor.
        """
        C = sparse.features.shape[1]
        T = self.time_bins
        device = sparse.features.device
        dtype = sparse.features.dtype

        # CHANNELS-LAST: [B, T, H, W, C] for correct scatter indexing
        # Using [B,C,T,H,W].view(-1,C) is WRONG — same bug as 4D case.
        dense = torch.zeros(
            batch_size, T, target_h, target_w, C,
            device=device, dtype=dtype
        )

        if sparse.features.shape[0] == 0:
            return dense.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, T, H, W]

        indices = sparse.indices  # [N, 4] = (batch, t, y, x)
        features = sparse.features  # [N, C]

        b = indices[:, 0].long()
        t = indices[:, 1].long()
        y = indices[:, 2].long()
        x = indices[:, 3].long()

        # Scatter: dense[b, t, y, x, :] += features
        flat_idx = (b * T * target_h * target_w +
                    t * target_h * target_w +
                    y * target_w + x)

        flat_dense = dense.view(-1, C)
        flat_dense.scatter_add_(0, flat_idx.unsqueeze(1).expand(-1, C), features)

        return dense.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, T, H, W]

    def forward(
        self,
        sparse_features: List[spconv.SparseConvTensor],
        batch_size: int
    ) -> List[torch.Tensor]:
        """
        Convert sparse features to dense with learned temporal attention.

        Returns:
            [P2, P3, P4] dense feature maps
        """
        dense_features = []

        for sparse, (h, w), t_conv, att_conv, adapter in zip(
            sparse_features, self.spatial_shapes,
            self.temporal_convs, self.attention_convs, self.adapters
        ):
            # Validate spatial shapes
            actual_h = sparse.spatial_shape[1]
            actual_w = sparse.spatial_shape[2]
            if actual_h != h or actual_w != w:
                raise ValueError(
                    f"Spatial shape mismatch! Backbone: ({actual_h}, {actual_w}), "
                    f"Bridge expects: ({h}, {w})."
                )

            # Scatter to 5D: [B, C, T, H, W]
            dense_5d = self._sparse_to_5d(sparse, h, w, batch_size)

            B, C, T, H, W = dense_5d.shape

            # Reshape for Conv1d: [B*H*W, C, T]
            x = dense_5d.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)

            # Temporal smoothing
            x = t_conv(x)  # [B*H*W, C, T]

            # Attention weights: [B*H*W, 1, T]
            att = att_conv(x)  # [B*H*W, 1, T]
            att = torch.softmax(att, dim=2)  # Normalize over T

            # Weighted sum over time: [B*H*W, C]
            x = (x * att).sum(dim=2)  # [B*H*W, C]

            # Reshape back: [B, C, H, W]
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)

            # Channel adapter
            x = adapter(x)

            dense_features.append(x)

        return dense_features


class MotionAwareBridge(nn.Module):
    """
    Motion-Aware Bridge: converts sparse 3D features to dense 2D with motion information.

    Three pathways:
    1. Attention-weighted temporal sum (same as TemporalAttentionBridge)
    2. Temporal gradient -> velocity proxy (motion direction/speed)
    3. Temporal variance -> activity consistency (steady vs bursty)

    Combined via gated residuals with conservative initialization (gate~0.018).
    This prevents random gradient/variance features from corrupting backbone features
    at the start of training.

    Parameter overhead vs plain attention: ~2x bottleneck params + 2 gate scalars per level.
    For nano backbone (64/128/256 channels), adds ~10K params total.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        spatial_shapes: List[Tuple[int, int]],
        time_bins: int = 15,
        bottleneck_ratio: int = 4,
    ):
        """
        Args:
            in_channels: Channels from backbone at each scale (e.g. [64, 128, 256])
            out_channels: Target channels for FPN (same as in_channels typically)
            spatial_shapes: (H, W) at each scale
            time_bins: Number of temporal bins (T)
            bottleneck_ratio: Channel reduction ratio for gradient/variance projections
        """
        super().__init__()
        self.spatial_shapes = spatial_shapes
        self.time_bins = time_bins

        # Per-level modules
        self.temporal_attns = nn.ModuleList()
        self.grad_projs = nn.ModuleList()
        self.var_projs = nn.ModuleList()
        self.adapters = nn.ModuleList()

        # Per-level gate parameters (stored as ParameterList for proper registration)
        self.gate_grads = nn.ParameterList()
        self.gate_vars = nn.ParameterList()

        for in_ch, out_ch in zip(in_channels, out_channels):
            # Pathway 1: Attention-weighted temporal sum
            # Global avg pool over spatial dims -> linear to produce T attention weights
            self.temporal_attns.append(nn.Sequential(
                nn.AdaptiveAvgPool3d((time_bins, 1, 1)),
                nn.Flatten(start_dim=1),
                nn.Linear(in_ch * time_bins, time_bins),
            ))

            # Pathway 2: Temporal gradient projection (bottleneck)
            # Reduces channel dim to save compute, then projects back
            mid_ch = max(in_ch // bottleneck_ratio, 8)
            self.grad_projs.append(nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.GroupNorm(min(8, mid_ch), mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, in_ch, 1, bias=False),
            ))

            # Pathway 3: Temporal variance projection (bottleneck)
            # Same architecture as gradient pathway but learns different features
            self.var_projs.append(nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.GroupNorm(min(8, mid_ch), mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, in_ch, 1, bias=False),
            ))

            # Gated residuals - initialized to sigmoid(-4) ~ 0.018 (near-identity at start)
            # -4.0 bias means gradient/variance pathways contribute <2% initially,
            # preventing random features from corrupting attention-weighted backbone features.
            self.gate_grads.append(nn.Parameter(torch.full((1, in_ch, 1, 1), -4.0)))
            self.gate_vars.append(nn.Parameter(torch.full((1, in_ch, 1, 1), -4.0)))

            # Channel adapter (1x1 conv if channel count changes)
            if in_ch != out_ch:
                self.adapters.append(nn.Conv2d(in_ch, out_ch, 1))
            else:
                self.adapters.append(nn.Identity())

    def _sparse_to_5d(
        self,
        sparse: spconv.SparseConvTensor,
        target_h: int,
        target_w: int,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Scatter sparse features to dense [B, C, T, H, W] tensor.

        Uses CHANNELS-LAST scatter pattern to avoid the scatter bug:
        allocate as [B, T, H, W, C], scatter into flat view, then permute.
        This ensures view(-1, C) gives [B*T*H*W, C] where each row is a
        unique (b,t,y,x) location. See SparseToDenseBridge for the full
        explanation of why [B,C,T,H,W].view(-1,C) is WRONG.
        """
        C = sparse.features.shape[1]
        T = self.time_bins
        device = sparse.features.device
        dtype = sparse.features.dtype

        # CHANNELS-LAST: [B, T, H, W, C] for correct scatter indexing
        dense = torch.zeros(
            batch_size, T, target_h, target_w, C,
            device=device, dtype=dtype
        )

        if sparse.features.shape[0] == 0:
            return dense.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, T, H, W]

        indices = sparse.indices  # [N, 4] = (batch, t, y, x)
        features = sparse.features  # [N, C]

        b = indices[:, 0].long()
        t = indices[:, 1].long()
        y = indices[:, 2].long()
        x = indices[:, 3].long()

        # Scatter: dense[b, t, y, x, :] += features
        flat_idx = (b * T * target_h * target_w +
                    t * target_h * target_w +
                    y * target_w + x)

        flat_dense = dense.view(-1, C)
        flat_dense.scatter_add_(0, flat_idx.unsqueeze(1).expand(-1, C), features)

        return dense.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, T, H, W]

    def forward(
        self,
        sparse_features: List[spconv.SparseConvTensor],
        batch_size: int
    ) -> List[torch.Tensor]:
        """
        Convert sparse features to dense with motion-aware temporal aggregation.

        For each FPN level:
        1. Scatter sparse -> dense 5D [B, C, T, H, W]
        2. Pathway 1: attention-weighted temporal sum -> [B, C, H, W]
        3. Pathway 2: temporal gradient (abs diff between consecutive bins) -> projected
        4. Pathway 3: temporal variance -> projected
        5. Combine: attention_out + gate_grad * grad_out + gate_var * var_out

        Args:
            sparse_features: [C2, C3, C4] from backbone
            batch_size: Batch size

        Returns:
            [P2, P3, P4] dense feature maps with motion information
        """
        dense_features = []

        for level_idx, (sparse, (h, w)) in enumerate(zip(
            sparse_features, self.spatial_shapes
        )):
            # Validate spatial shapes match backbone output
            actual_h = sparse.spatial_shape[1]
            actual_w = sparse.spatial_shape[2]
            if actual_h != h or actual_w != w:
                raise ValueError(
                    f"Spatial shape mismatch at level {level_idx}! "
                    f"Backbone: ({actual_h}, {actual_w}), "
                    f"Bridge expects: ({h}, {w})."
                )

            # Scatter to 5D: [B, C, T, H, W]
            dense_5d = self._sparse_to_5d(sparse, h, w, batch_size)
            B, C, T, H, W = dense_5d.shape

            # --- Pathway 1: Attention-weighted temporal sum ---
            # AdaptiveAvgPool3d + Linear produces per-timestep attention weights
            attn_weights = self.temporal_attns[level_idx](dense_5d)  # [B, T]
            attn_weights = F.softmax(attn_weights, dim=1)  # [B, T]
            attn_weights = attn_weights.view(B, 1, T, 1, 1)  # [B, 1, T, 1, 1]
            x_att = (dense_5d * attn_weights).sum(dim=2)  # [B, C, H, W]

            # --- Pathway 2: Temporal gradient (velocity proxy) ---
            # Difference between consecutive time bins captures motion speed/direction.
            # abs().mean() gives average magnitude of change across time.
            if T > 1:
                grad = dense_5d[:, :, 1:, :, :] - dense_5d[:, :, :-1, :, :]  # [B, C, T-1, H, W]
                grad_feat = grad.abs().mean(dim=2)  # [B, C, H, W]
                grad_out = self.grad_projs[level_idx](grad_feat)  # [B, C, H, W]
            else:
                grad_out = torch.zeros_like(x_att)

            # --- Pathway 3: Temporal variance (activity consistency) ---
            # High variance = bursty activity (e.g., fast motion, flicker)
            # Low variance = steady activity (e.g., stationary object, background)
            if T > 1:
                var_feat = dense_5d.var(dim=2)  # [B, C, H, W]
                var_out = self.var_projs[level_idx](var_feat)  # [B, C, H, W]
            else:
                var_out = torch.zeros_like(x_att)

            # --- Combine with gated residuals ---
            # Gates start at sigmoid(-4) ~ 0.018, so output ~ x_att initially.
            # As training progresses, gates open to incorporate motion information.
            gate_g = torch.sigmoid(self.gate_grads[level_idx])  # [1, C, 1, 1]
            gate_v = torch.sigmoid(self.gate_vars[level_idx])   # [1, C, 1, 1]

            output = x_att + gate_g * grad_out + gate_v * var_out

            # Channel adapter (identity if in_ch == out_ch)
            output = self.adapters[level_idx](output)

            dense_features.append(output)

        return dense_features


class SimpleFPN(nn.Module):
    """
    Simple Feature Pyramid Network with top-down pathway.

    Takes multi-scale features and adds top-down connections
    with lateral (skip) connections for better feature fusion.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 128
    ):
        super().__init__()

        # Lateral connections (1x1 conv to match channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels
        ])

        # Output convs (3x3 to smooth after upsampling)
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: [C2, C3, C4] with decreasing resolution

        Returns:
            [P3, P4, P5] with same channels
        """
        # Lateral connections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway (from smallest to largest)
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i-1].shape[2:], mode='nearest')
            laterals[i-1] = laterals[i-1] + up

        # Output convs
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        return outputs


class PANBottomUp(nn.Module):
    """
    Path Aggregation Network (PAN) bottom-up pathway.

    After the FPN top-down pathway (P5 -> P4 -> P3), PAN adds a bottom-up
    pathway (P3 -> P4 -> P5) so that high-resolution P3 features propagate
    back up to deeper levels. This is critical for small object detection
    (e.g., small drones in FRED dataset) because P3 has the best spatial
    resolution but without PAN, P4/P5 don't benefit from it.

    Architecture per level (starting from P4_pan):
        P4_pan = FPN_P4 + Conv3x3(Downsample2x(FPN_P3))
        P5_pan = FPN_P5 + Conv3x3(Downsample2x(P4_pan))

    Each transition uses:
    1. Stride-2 3x3 conv for spatial downsampling (learned, better than maxpool)
    2. 3x3 conv for feature fusion after addition

    Reference: Liu et al., "Path Aggregation Network for Instance Segmentation", CVPR 2018
    """

    def __init__(self, channels: int, num_levels: int = 3):
        """
        Args:
            channels: Number of channels at each FPN level (must be uniform after FPN)
            num_levels: Number of FPN levels (default 3: P3, P4, P5)
        """
        super().__init__()

        # Bottom-up pathway: one downsample + fusion per transition
        # For 3 levels (P3->P4->P5), we need 2 transitions (num_levels - 1)
        self.downsample_convs = nn.ModuleList()
        self.fusion_convs = nn.ModuleList()

        for _ in range(num_levels - 1):
            # Stride-2 conv for 2x spatial downsampling
            # Using conv instead of maxpool preserves more information
            self.downsample_convs.append(
                nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False)
            )
            # 3x3 conv to smooth after element-wise addition
            # Same role as the output convs in FPN top-down
            self.fusion_convs.append(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            )

    def forward(self, fpn_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply bottom-up pathway to FPN outputs.

        Args:
            fpn_features: [P3, P4, P5] from FPN (highest to lowest resolution)

        Returns:
            [P3_pan, P4_pan, P5_pan] with bottom-up feature propagation
        """
        # Start from the highest-resolution level (P3) and go bottom-up
        # P3_pan = P3_fpn (unchanged, it's the starting point)
        pan_features = [fpn_features[0]]

        for i in range(len(self.downsample_convs)):
            # Downsample the previous PAN level
            downsampled = self.downsample_convs[i](pan_features[i])
            # Add to the next FPN level (element-wise)
            fused = fpn_features[i + 1] + downsampled
            # Smooth with 3x3 conv
            fused = self.fusion_convs[i](fused)
            pan_features.append(fused)

        return pan_features


class SparseFCOSDetector(nn.Module):
    """
    Complete Sparse FCOS Detector.

    Pipeline:
    1. Raw sparse events -> SparseSEWResNet backbone
    2. Sparse features -> SparseToDenseBridge -> dense features
    3. Dense features -> FPN neck -> multi-scale features
    4. (Optional) FPN features -> PAN bottom-up -> enhanced features
    5. Multi-scale features -> FCOS head -> predictions

    This is the main model class for training and inference.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 1,
        backbone_size: str = 'nano',
        fpn_channels: int = 64,
        num_head_convs: int = 4,
        strides: List[int] = [4, 8, 16],
        regress_ranges: Tuple[Tuple[float, float], ...] = ((-1, 32), (32, 64), (64, 256)),
        center_sampling: bool = True,
        center_sampling_radius: float = 1.5,
        prior_prob: float = 0.01,
        norm_on_bbox: bool = True,
        input_size: Tuple[int, int] = (640, 640),
        num_temporal_groups: int = 3,
        time_bins: int = 15,
        stem_stride: tuple = (1, 2, 2),
        bridge_type: str = 'attention',
        gn_groups: int = 8,
        use_pan: bool = False,
        use_iou_quality: bool = False,
    ):
        """
        Args:
            in_channels: Input event channels (2 = ON/OFF)
            num_classes: Number of object classes (1 for drone)
            backbone_size: Backbone variant ('nano', 'small', 'medium', 'large')
            fpn_channels: Channels in FPN and head
            num_head_convs: Number of conv layers in FCOS head towers
            strides: Feature map strides ([4,8,16] with stem_stride=(1,2,2))
            regress_ranges: Size ranges for each FPN level
            center_sampling: Use center sampling for target assignment
            center_sampling_radius: Radius for center sampling
            prior_prob: Prior probability for focal loss bias init
            norm_on_bbox: Normalize regression by stride (True for multi-stride)
            input_size: Input spatial size (H, W)
            num_temporal_groups: Number of temporal groups for bridge (3=early/mid/late)
            time_bins: Number of temporal bins
            stem_stride: Stride for backbone stem conv ((1,2,2) for 2x spatial downsample)
            bridge_type: 'attention' (learned temporal), 'group' (fixed groups),
                or 'motion_aware' (attention + gradient + variance with gated residuals)
            gn_groups: Number of groups for GroupNorm in FCOS head ConvBlocks (default 8).
                With 128ch FPN: 8 groups = 16 channels/group (proper group statistics).
                With 64ch FPN: 8 groups = 8 channels/group (still reasonable).
            use_pan: If True, add PAN bottom-up pathway after FPN top-down.
                PAN propagates high-res P3 features back to P4/P5, improving
                small object detection. Adds ~2*fpn_channels*fpn_channels*9*2 params
                (e.g., ~590K for fpn_channels=128). Default False for backward compat.
            use_iou_quality: If True, include IoU-quality logits in inference score
                composition: score = cls * centerness * iou_quality.
        """
        super().__init__()

        self.num_classes = num_classes
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sampling_radius = center_sampling_radius
        self.input_size = input_size
        self.use_iou_quality = bool(use_iou_quality)

        # Sparse backbone with configurable stem stride
        self.backbone = SparseSEWResNet(
            in_channels=in_channels,
            size=backbone_size,
            stem_stride=stem_stride
        )
        backbone_channels = self.backbone.out_channels  # e.g., [64, 128, 256] for nano

        # Compute spatial shapes at each scale
        # With stem_stride=(1,2,2) and backbone strides [2,4,8]:
        #   Effective strides: [4, 8, 16]
        #   Feature sizes for 640x640: [(160, 160), (80, 80), (40, 40)]
        self.feature_sizes = [
            (input_size[0] // s, input_size[1] // s) for s in strides
        ]

        # Sparse-to-dense bridge
        if bridge_type == 'attention':
            self.bridge = TemporalAttentionBridge(
                in_channels=backbone_channels,
                out_channels=backbone_channels,
                spatial_shapes=self.feature_sizes,
                time_bins=time_bins,
            )
        elif bridge_type == 'group':
            self.bridge = TemporalGroupBridge(
                in_channels=backbone_channels,
                out_channels=backbone_channels,
                spatial_shapes=self.feature_sizes,
                num_groups=num_temporal_groups,
                time_bins=time_bins,
            )
        elif bridge_type == 'motion_aware':
            # MotionAwareBridge: attention + temporal gradient + temporal variance
            # with gated residuals (gate init -4.0 -> sigmoid~0.018, near-identity)
            self.bridge = MotionAwareBridge(
                in_channels=backbone_channels,
                out_channels=backbone_channels,
                spatial_shapes=self.feature_sizes,
                time_bins=time_bins,
            )
        else:
            raise ValueError(
                f"Unknown bridge_type: {bridge_type}. "
                f"Use 'attention', 'group', or 'motion_aware'."
            )

        # FPN neck (top-down pathway)
        self.fpn = SimpleFPN(
            in_channels=backbone_channels,
            out_channels=fpn_channels
        )

        # Optional PAN bottom-up pathway after FPN
        # PAN propagates high-resolution P3 features back to P4/P5:
        #   P4_pan = P4_fpn + Conv(Downsample(P3_fpn))
        #   P5_pan = P5_fpn + Conv(Downsample(P4_pan))
        # Critical for small drone detection where P3 has best spatial detail.
        self.use_pan = use_pan
        if use_pan:
            self.pan = PANBottomUp(
                channels=fpn_channels,
                num_levels=len(strides)
            )

        # FCOS detection head
        self.head = FCOSHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_convs=num_head_convs,
            num_levels=len(strides),  # Match number of FPN levels
            prior_prob=prior_prob,
            use_scale=True,
            norm_on_bbox=norm_on_bbox,
            gn_groups=gn_groups
        )

        # Pre-compute grid points for each level (for inference)
        self._init_points()

    def _init_points(self):
        """Initialize grid points for each FPN level."""
        self.register_buffer('_dummy', torch.zeros(1))  # For device tracking

        # Points will be created on first forward pass when we know the device
        self._points_initialized = False
        self._cached_points = None

    def _get_points(self, device: torch.device) -> List[torch.Tensor]:
        """Get or create grid points."""
        if self._cached_points is not None and self._cached_points[0].device == device:
            return self._cached_points

        self._cached_points = generate_points(self.feature_sizes, self.strides, device)
        return self._cached_points

    def forward(
        self,
        x: spconv.SparseConvTensor,
        batch_size: Optional[int] = None,
        targets: Optional[Dict] = None,
        return_loss_inputs: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: SparseConvTensor with event features
                - features: [N, in_channels]
                - indices: [N, 4] = (batch, t, y, x)
            batch_size: Batch size (inferred if not provided)
            targets: Optional targets for training
                - gt_boxes: List of [M, 4] per image
                - gt_labels: List of [M] per image
            return_loss_inputs: If True, return raw predictions even in eval mode
                (useful for validation loss computation with proper BatchNorm behavior)

        Returns:
            Dictionary with:
            - Training (or return_loss_inputs=True): cls_preds, reg_preds, ctr_preds, iou_quality_preds
            - Inference: detections [B, N, 6] = (x1, y1, x2, y2, conf, class)
        """
        # Infer batch size
        if batch_size is None:
            batch_size = x.indices[:, 0].max().item() + 1

        device = x.features.device

        # 1. Sparse backbone
        sparse_features = self.backbone(x)

        # 2. Sparse to dense bridge
        dense_features = self.bridge(sparse_features, batch_size)

        # 3. FPN (top-down pathway)
        fpn_features = self.fpn(dense_features)

        # 3b. PAN (bottom-up pathway, optional)
        if self.use_pan:
            fpn_features = self.pan(fpn_features)

        # 4. FCOS head
        # Training: don't multiply by stride (loss uses normalized targets from norm_on_bbox)
        # Inference: multiply by stride (decode to pixel-space boxes)
        head_strides = None if (self.training or return_loss_inputs) else self.strides
        cls_preds, reg_preds, ctr_preds, iouq_preds = self.head(fpn_features, head_strides)

        if self.training or return_loss_inputs:
            # Return raw predictions for loss computation
            return {
                'cls_preds': cls_preds,
                'reg_preds': reg_preds,
                'ctr_preds': ctr_preds,
                'iou_quality_preds': iouq_preds,
            }
        else:
            # Inference: decode predictions to boxes
            return self._decode_predictions(
                cls_preds, reg_preds, ctr_preds, device,
                iou_quality_preds=(iouq_preds if self.use_iou_quality else None),
            )

    def _decode_predictions(
        self,
        cls_preds: List[torch.Tensor],
        reg_preds: List[torch.Tensor],
        ctr_preds: List[torch.Tensor],
        device: torch.device,
        score_thresh: float = 0.001,
        nms_thresh: float = 0.3,
        max_detections: int = 1,
        iou_quality_preds: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode FCOS predictions to bounding boxes.

        Args:
            cls_preds: List of [B, C, H, W] classification logits
            reg_preds: List of [B, 4, H, W] LTRB distances
            ctr_preds: List of [B, 1, H, W] centerness logits
            device: Target device
            score_thresh: Score threshold for filtering
            nms_thresh: NMS IoU threshold
            max_detections: Maximum detections per image
            iou_quality_preds: Optional IoU-quality logits per FPN level

        Returns:
            Dictionary with 'detections': [B, N, 6] tensor
        """
        batch_size = cls_preds[0].shape[0]
        points = self._get_points(device)

        all_detections = []

        for batch_idx in range(batch_size):
            boxes_list = []
            scores_list = []
            labels_list = []

            for level_idx, (cls, reg, ctr, pts) in enumerate(zip(
                cls_preds, reg_preds, ctr_preds, points
            )):
                # Get predictions for this image
                cls_logits = cls[batch_idx]  # [C, H, W]
                reg_pred = reg[batch_idx]    # [4, H, W]
                ctr_logits = ctr[batch_idx]  # [1, H, W]
                if iou_quality_preds is not None and level_idx < len(iou_quality_preds):
                    iouq_logits = iou_quality_preds[level_idx][batch_idx]  # [1, H, W]
                else:
                    iouq_logits = None

                # Flatten
                H, W = cls_logits.shape[1:]
                cls_logits = cls_logits.permute(1, 2, 0).reshape(-1, self.num_classes)  # [HW, C]
                reg_pred = reg_pred.permute(1, 2, 0).reshape(-1, 4)  # [HW, 4]
                ctr_logits = ctr_logits.permute(1, 2, 0).reshape(-1)  # [HW]
                if iouq_logits is not None:
                    iouq_logits = iouq_logits.permute(1, 2, 0).reshape(-1)  # [HW]

                # Compute scores: cls_prob * centerness * IoU-quality (if enabled)
                cls_prob = torch.sigmoid(cls_logits)  # [HW, C]
                ctr_prob = torch.sigmoid(ctr_logits)  # [HW]
                if iouq_logits is not None:
                    iouq_prob = torch.sigmoid(iouq_logits)
                else:
                    iouq_prob = torch.ones_like(ctr_prob)
                scores = cls_prob * ctr_prob.unsqueeze(1) * iouq_prob.unsqueeze(1)  # [HW, C]

                # Get max scores and labels
                max_scores, labels = scores.max(dim=1)  # [HW]

                # Filter by threshold
                keep = max_scores > score_thresh
                if not keep.any():
                    continue

                max_scores = max_scores[keep]
                labels = labels[keep]
                reg_pred = reg_pred[keep]
                pts_keep = pts[keep]

                # Convert LTRB to xyxy
                boxes = ltrb_to_xyxy(reg_pred, pts_keep)

                # Clip to image bounds
                boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=self.input_size[1])
                boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=self.input_size[0])

                boxes_list.append(boxes)
                scores_list.append(max_scores)
                labels_list.append(labels)

            if not boxes_list:
                # No detections
                all_detections.append(torch.zeros(0, 6, device=device))
                continue

            # Concatenate all levels
            boxes = torch.cat(boxes_list, dim=0)
            scores = torch.cat(scores_list, dim=0)
            labels = torch.cat(labels_list, dim=0)

            # NMS
            from torchvision.ops import batched_nms
            keep = batched_nms(boxes, scores, labels, nms_thresh)
            keep = keep[:max_detections]

            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # Format: [x1, y1, x2, y2, score, label]
            detections = torch.cat([
                boxes,
                scores.unsqueeze(1),
                labels.unsqueeze(1).float()
            ], dim=1)

            all_detections.append(detections)

        # Pad to same size
        max_dets = max(d.shape[0] for d in all_detections) if all_detections else 0
        if max_dets == 0:
            return {'detections': torch.zeros(batch_size, 0, 6, device=device)}

        padded = torch.zeros(batch_size, max_dets, 6, device=device)
        for i, dets in enumerate(all_detections):
            if dets.shape[0] > 0:
                padded[i, :dets.shape[0]] = dets

        return {'detections': padded}

    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        counts = {
            'backbone': sum(p.numel() for p in self.backbone.parameters()),
            'bridge': sum(p.numel() for p in self.bridge.parameters()),
            'fpn': sum(p.numel() for p in self.fpn.parameters()),
        }
        if self.use_pan:
            counts['pan'] = sum(p.numel() for p in self.pan.parameters())
        counts['head'] = sum(p.numel() for p in self.head.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts


def test_sparse_fcos():
    """Test the complete sparse FCOS detector."""
    print("Testing SparseFCOSDetector (TemporalAttentionBridge)...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model with attention bridge
    model = SparseFCOSDetector(
        in_channels=2,
        num_classes=1,
        backbone_size='nano',
        fpn_channels=64,
        strides=[4, 8, 16],
        stem_stride=(1, 2, 2),
        time_bins=15,
        norm_on_bbox=True,
        bridge_type='attention',
    ).to(device)

    # Parameter counts
    params = model.get_num_params()
    print(f"\nParameters:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")

    # Create dummy sparse input
    batch_size = 2
    spatial_shape = [15, 640, 640]  # T=15, H=640, W=640
    n_voxels = 10000

    indices = torch.zeros((n_voxels, 4), dtype=torch.int32, device=device)
    indices[:, 0] = torch.randint(0, batch_size, (n_voxels,))
    indices[:, 1] = torch.randint(0, 15, (n_voxels,))  # Match spatial_shape T=15
    indices[:, 2] = torch.randint(0, 640, (n_voxels,))
    indices[:, 3] = torch.randint(0, 640, (n_voxels,))
    features = torch.randn(n_voxels, 2, device=device)

    sparse_input = spconv.SparseConvTensor(
        features=features,
        indices=indices,
        spatial_shape=spatial_shape,
        batch_size=batch_size
    )

    print(f"\nInput:")
    print(f"  Voxels: {n_voxels}")
    print(f"  Spatial shape: {spatial_shape}")
    print(f"  Sparsity: {1 - n_voxels / (batch_size * 15 * 640 * 640):.2%}")

    # Training forward pass
    model.train()
    outputs = model(sparse_input, batch_size)

    print(f"\nTraining outputs:")
    for key, val in outputs.items():
        if isinstance(val, list):
            print(f"  {key}: {[v.shape for v in val]}")
        else:
            print(f"  {key}: {val.shape}")

    # Verify shapes
    for level_idx, (cls, reg, ctr, iouq) in enumerate(zip(
        outputs['cls_preds'], outputs['reg_preds'], outputs['ctr_preds'], outputs['iou_quality_preds']
    )):
        H, W = model.feature_sizes[level_idx]
        assert cls.shape == (batch_size, 1, H, W), f"cls shape mismatch at level {level_idx}"
        assert reg.shape == (batch_size, 4, H, W), f"reg shape mismatch at level {level_idx}"
        assert ctr.shape == (batch_size, 1, H, W), f"ctr shape mismatch at level {level_idx}"
        assert iouq.shape == (batch_size, 1, H, W), f"iouq shape mismatch at level {level_idx}"

    # Inference forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(sparse_input, batch_size)

    print(f"\nInference outputs:")
    print(f"  detections: {outputs['detections'].shape}")

    # Benchmark
    import time

    model.train()
    for _ in range(3):
        _ = model(sparse_input, batch_size)
    torch.cuda.synchronize()

    start = time.perf_counter()
    n_iters = 10
    for _ in range(n_iters):
        _ = model(sparse_input, batch_size)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1000

    print(f"\nTiming (training mode):")
    print(f"  Forward: {elapsed:.1f} ms")
    print(f"  Throughput: {1000/elapsed:.0f} FPS")

    print("\nSUCCESS: SparseFCOSDetector working!")


if __name__ == '__main__':
    test_sparse_fcos()
