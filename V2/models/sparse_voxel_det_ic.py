#!/usr/bin/env python3
"""
SparseVoxelDet-IC (v85_ic) — inverse-conv FPN variant of the v82/v84 model.

Replaces the FPN's SparseConvTranspose3d upsampling (which DILATES the active
set ~4.4-5.0x per stage, exploding a ~4.3K-voxel input to ~345K active head
positions) with spconv.SparseInverseConv3d, which restores EXACTLY the active
set that existed before the paired downsampling conv (shared indice_key +
kernel_size). Result: FPN output active set == backbone c2 active set.

Mechanism (verified on spconv 2.3.8, GPU probes 2026-07-02):
  * SparseInverseConv3d(in, out, kernel_size, indice_key) requires the SAME
    kernel_size and indice_key as the paired forward strided conv, and its
    input rows must be aligned to that conv's OUTPUT row order (the stored
    indice pairs reference row positions).
  * The stock backbone's downsampling convs carry no indice_key, and
    sparse_add / sort_sparse_tensor build fresh SparseConvTensors that DROP
    indice_dict and reorder rows. So this file:
      1. SparseSEWResNetIC — replaces layer3[0].conv2 / layer4[0].conv2 with
         indice_key-carrying copies (same kernel/stride/init; state_dict keys
         unchanged), captures each conv's output (indices + IndiceData) via a
         module-local forward hook (EMA-deepcopy-safe), and rebuilds the c3/c4
         stage outputs onto the captured row order (zero-fill for rows pruned
         by sparse_add's epsilon threshold).
      2. SparseFPNIC — laterals as before (SubMConv3d preserves row order and
         indice_dict), upsampling via SparseInverseConv3d (NO sorting), and
         exact-match sparse addition that aligns both operands to the captured
         row order by linear coordinate. Alignment integrity is hard-checked
         every forward (count + coordinate equality) — crashes loudly on any
         active-set mismatch instead of silently corrupting features.
  * Numerical exactness of transport+realign verified: max abs diff 0.0 vs
    running the inverse conv directly on the forward conv's output.

Head unchanged. Backbone weights/architecture unchanged (only indice_key
metadata added to two convs).
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import spconv.pytorch as spconv

from models.snn.sparse_sew_resnet import (
    SparseSEWResNet,
    SafeSparseBatchNorm,
    SparseActivation,
    compute_linear_idx,
)
from V2.models.sparse_voxel_det_v82 import (
    SparseVoxelDet,
    SparseLateralBlock,
    SparseOutputBlock,
    sparse_temporal_pool,
)

# indice_keys pairing FPN inverse convs with backbone downsampling convs
DOWN_KEY_C3 = "ic_down_c3"  # layer3[0].conv2 : c2 (stride 4) -> c3 (stride 8)
DOWN_KEY_C4 = "ic_down_c4"  # layer4[0].conv2 : c3 (stride 8) -> c4 (stride 16)


def _row_map_to_target(
    src_indices: torch.Tensor,
    target_indices: torch.Tensor,
    spatial_shape: List[int],
    batch_size: int,
    context: str,
) -> torch.Tensor:
    """For each src row, its row position in target (matching by coordinate).

    Requires every src coordinate to exist in target. Crashes with context on
    any mismatch — a mismatch means the inverse-conv pairing is broken.
    """
    if src_indices.shape[0] > 0 and target_indices.shape[0] == 0:
        raise RuntimeError(
            f"IC-FPN [{context}]: target active set empty but source has "
            f"{src_indices.shape[0]} positions — indice_key pairing broken.")
    lin_src = compute_linear_idx(src_indices, spatial_shape, batch_size)
    lin_tgt = compute_linear_idx(target_indices, spatial_shape, batch_size)
    order = torch.argsort(lin_tgt)
    pos = torch.searchsorted(lin_tgt[order], lin_src)
    pos = pos.clamp(max=max(lin_tgt.shape[0] - 1, 0))
    row_map = order[pos] if order.shape[0] > 0 else pos
    if not torch.equal(lin_tgt[row_map], lin_src):
        n_missing = int((lin_tgt[row_map] != lin_src).sum().item())
        raise RuntimeError(
            f"IC-FPN [{context}]: {n_missing}/{src_indices.shape[0]} source "
            f"positions not found in target active set "
            f"({target_indices.shape[0]} positions) — indice_key pairing broken.")
    return row_map


def _align_features_to_target(
    x: spconv.SparseConvTensor,
    target_indices: torch.Tensor,
    context: str,
) -> torch.Tensor:
    """Reorder x.features to target_indices row order (exact set match required)."""
    if x.indices.shape[0] != target_indices.shape[0]:
        raise RuntimeError(
            f"IC-FPN [{context}]: active-set size mismatch "
            f"{x.indices.shape[0]} vs {target_indices.shape[0]} — "
            f"inverse conv did not restore the expected active set.")
    # target->src map, then gather: aligned[i] = features[row of target coord i]
    row_map = _row_map_to_target(target_indices, x.indices, x.spatial_shape,
                                 x.batch_size, context)
    return x.features[row_map]


class SparseSEWResNetIC(SparseSEWResNet):
    """SparseSEWResNet whose stage-transition downsampling convs carry
    indice_keys, with c3/c4 outputs rebuilt onto those convs' output row order
    so FPN inverse convs can consume them directly.

    Architecture and state_dict keys identical to SparseSEWResNet.
    """

    DOWN_KEYS = {"layer3": DOWN_KEY_C3, "layer4": DOWN_KEY_C4}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for lname, key in self.DOWN_KEYS.items():
            block = getattr(self, lname)[0]
            old = block.conv2
            out_ch = self.base_channels * (4 if lname == "layer3" else 8)
            new = spconv.SparseConv3d(
                out_ch, out_ch, 3, stride=(1, 2, 2), padding=1,
                bias=False, indice_key=key,
            )
            with torch.no_grad():
                new.weight.copy_(old.weight)
            block.conv2 = new
            # Module-local capture (NOT a closure over the backbone): survives
            # ModelEMA's deepcopy — the copied conv's hook writes to the copy.
            block.conv2.register_forward_hook(self._make_capture_hook(key))

    @staticmethod
    def _make_capture_hook(key: str):
        def hook(module, inputs, output):
            module._ic_last_capture = (
                output.indices,
                output.indice_dict[key],
                output.spatial_shape,
            )
        return hook

    def _restore_downsample_set(
        self, t: spconv.SparseConvTensor, lname: str, key: str,
    ) -> spconv.SparseConvTensor:
        """Rebuild stage output onto the downsample conv's output row order.

        The stage's residual sparse_add may prune near-zero rows, so the final
        stage active set can be a SUBSET of the conv output set; pruned rows
        are restored with zero features (scatter). Injects the conv's
        IndiceData so the paired SparseInverseConv3d can run.
        """
        conv = getattr(self, lname)[0].conv2
        saved_indices, indice_data, _ = conv._ic_last_capture
        row_map = _row_map_to_target(
            t.indices, saved_indices, t.spatial_shape, t.batch_size,
            f"restore_{key}")
        feats = torch.zeros(
            saved_indices.shape[0], t.features.shape[1],
            device=t.features.device, dtype=t.features.dtype,
        ).index_copy(0, row_map, t.features)
        out = spconv.SparseConvTensor(
            feats, saved_indices, t.spatial_shape, t.batch_size)
        out.indice_dict[key] = indice_data
        return out

    def forward(self, x: spconv.SparseConvTensor) -> List[spconv.SparseConvTensor]:
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self._restore_downsample_set(self.layer3(c2), "layer3", DOWN_KEY_C3)
        c4 = self._restore_downsample_set(self.layer4(c3), "layer4", DOWN_KEY_C4)
        return [c2, c3, c4]

    def ic_capture(self, key: str):
        """(indices, IndiceData, spatial_shape) of the keyed downsample conv."""
        lname = {DOWN_KEY_C3: "layer3", DOWN_KEY_C4: "layer4"}[key]
        return getattr(self, lname)[0].conv2._ic_last_capture


class SparseInverseUpsampleBlock(nn.Module):
    """Upsample via SparseInverseConv3d — restores EXACTLY the active set that
    existed before the paired downsampling conv (shared indice_key, kernel 3).

    No sort_sparse_tensor here: the input's row order must stay aligned to the
    paired conv's output order (the stored indice pairs reference row positions).
    """

    def __init__(self, in_channels: int, out_channels: int, indice_key: str) -> None:
        super().__init__()
        self.deconv = spconv.SparseInverseConv3d(
            in_channels, out_channels, 3, indice_key=indice_key, bias=False)
        self.bn = SafeSparseBatchNorm(out_channels)
        self.act = SparseActivation()

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        out = self.deconv(x)
        out = out.replace_feature(self.bn(out.features))
        return self.act(out)


class SparseFPNIC(nn.Module):
    """SparseFPN with inverse-conv upsampling and exact-match sparse addition.

    Same lateral/out blocks and channel plan as SparseFPN; only the upsample
    path changed. Output active set == c2 active set (hard-checked).
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 128) -> None:
        super().__init__()
        assert len(in_channels_list) == 3, "Expected 3 backbone output levels"
        self.lateral_c2 = SparseLateralBlock(in_channels_list[0], out_channels)
        self.lateral_c3 = SparseLateralBlock(in_channels_list[1], out_channels)
        self.lateral_c4 = SparseLateralBlock(in_channels_list[2], out_channels)
        self.up_c4_to_c3 = SparseInverseUpsampleBlock(out_channels, out_channels, DOWN_KEY_C4)
        self.up_c3_to_c2 = SparseInverseUpsampleBlock(out_channels, out_channels, DOWN_KEY_C3)
        self.out_block = SparseOutputBlock(out_channels)

    def forward(
        self,
        features: List[spconv.SparseConvTensor],
        backbone: SparseSEWResNetIC,
    ) -> spconv.SparseConvTensor:
        c2, c3, c4 = features

        # Top: lateral (SubM: preserves row order + indice_dict incl. DOWN_KEY_C4)
        p4 = self.lateral_c4(c4)
        p4_up = self.up_c4_to_c3(p4)  # active set == c3's set, exactly

        # Fuse at c3 level, rebuilt onto the layer3-downsample output order so
        # up_c3_to_c2's inverse conv can consume it.
        c3_indices, c3_indice_data, _ = backbone.ic_capture(DOWN_KEY_C3)
        p3 = self.lateral_c3(c3)
        fused_feats = (
            _align_features_to_target(p3, c3_indices, "fuse_c3.lateral")
            + _align_features_to_target(p4_up, c3_indices, "fuse_c3.upsampled")
        )
        p3f = spconv.SparseConvTensor(
            fused_feats, c3_indices, c3.spatial_shape, c3.batch_size)
        p3f.indice_dict[DOWN_KEY_C3] = c3_indice_data

        p3_up = self.up_c3_to_c2(p3f)  # active set == c2's set, exactly

        # Fuse at c2 level (exact-match addition in c2's row order)
        p2 = self.lateral_c2(c2)
        p2 = p2.replace_feature(
            p2.features
            + _align_features_to_target(p3_up, p2.indices, "fuse_c2.upsampled")
        )

        return self.out_block(p2)


class SparseVoxelDetIC(SparseVoxelDet):
    """SparseVoxelDet with inverse-conv FPN. Head and loss interface unchanged."""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 1,
        backbone_size: str = "nano_deep",
        fpn_channels: int = 128,
        head_convs: int = 2,
        strides: Optional[List[int]] = None,
        input_size: Tuple[int, int] = (720, 1280),
        time_bins: int = 16,
        prior_prob: float = 0.01,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        max_detections: int = 100,
        temporal_pool_mode: str = "max",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            backbone_size=backbone_size,
            fpn_channels=fpn_channels,
            head_convs=head_convs,
            strides=strides,
            input_size=input_size,
            time_bins=time_bins,
            prior_prob=prior_prob,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            max_detections=max_detections,
            temporal_pool_mode=temporal_pool_mode,
        )
        # Swap in the IC backbone/FPN (same channel plan; head untouched)
        self.backbone = SparseSEWResNetIC(
            in_channels=in_channels,
            size=backbone_size,
            stem_stride=(1, 2, 2),
        )
        self.fpn = SparseFPNIC(
            in_channels_list=self.backbone.out_channels,
            out_channels=fpn_channels,
        )

    def forward(
        self,
        x: spconv.SparseConvTensor,
        batch_size: Optional[int] = None,
        targets: Optional[Dict] = None,
        return_loss_inputs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if batch_size is None:
            batch_size = int(x.indices[:, 0].max().item()) + 1

        backbone_features = self.backbone(x)
        fused = self.fpn(backbone_features, self.backbone)

        features_2d, indices_2d, spatial_2d = sparse_temporal_pool(
            fused, mode=self.temporal_pool_mode,
        )
        cls_logits, box_ltrb, ctr_logits = self.head(features_2d)

        if self.training or return_loss_inputs:
            return {
                "cls_logits": cls_logits,
                "box_ltrb": box_ltrb,
                "ctr_logits": ctr_logits,
                "indices_2d": indices_2d,
                "spatial_2d": spatial_2d,
            }
        detections = self._decode_detections(
            cls_logits, box_ltrb, ctr_logits, indices_2d, batch_size,
        )
        return {"detections": detections}
