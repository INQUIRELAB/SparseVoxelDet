"""Dense cuDNN execution for a loaded SparseVoxelDet-IC module tree."""

from __future__ import annotations

import copy
import types
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

from models.snn.sparse_sew_resnet import SparseBasicBlock, SparseDownsample
from V2.models.sparse_voxel_det_ic import SparseFPNIC, SparseSEWResNetIC
from V2.models.sparse_voxel_det_v82 import sparse_temporal_pool


Triple = Tuple[int, int, int]


def _triple(value: Any) -> Triple:
    if isinstance(value, Sequence):
        if len(value) != 3:
            raise ValueError(f"expected a 3-vector, got {value!r}")
        return tuple(int(v) for v in value)  # type: ignore[return-value]
    return (int(value),) * 3


@dataclass(frozen=True)
class StoredSupport:
    input_support: torch.Tensor
    output_support: torch.Tensor
    input_shape: Triple
    output_shape: Triple


@dataclass
class DenseForwardResult:
    outputs: Dict[str, torch.Tensor]
    timings_ms: Dict[str, Dict[str, float]]


class TimingLedger:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._events: List[Tuple[str, str, torch.cuda.Event, torch.cuda.Event]] = []

    def measure(
        self,
        layer: str,
        phase: str,
        fn: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        if not self.enabled:
            return fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        self._events.append((layer, phase, start, end))
        return result

    def materialize(self) -> Dict[str, Dict[str, float]]:
        if not self.enabled:
            return {}
        torch.cuda.synchronize()
        result: Dict[str, Dict[str, float]] = {}
        for layer, phase, start, end in self._events:
            per_layer = result.setdefault(layer, {})
            per_layer[f"{phase}_ms"] = per_layer.get(f"{phase}_ms", 0.0) + start.elapsed_time(end)
        return result


class SupportContext:
    def __init__(self, timings: TimingLedger) -> None:
        self.timings = timings
        self.keyed: Dict[str, StoredSupport] = {}


class DenseSparseTensor:
    """Dense values plus the coordinate support on which sparse feature ops act."""

    def __init__(
        self,
        values: torch.Tensor,
        support: torch.Tensor,
        context: SupportContext,
    ) -> None:
        if values.ndim != 5 or support.shape != (values.shape[0], 1, *values.shape[2:]):
            raise ValueError(
                f"invalid dense/support shapes: {tuple(values.shape)}, {tuple(support.shape)}"
            )
        if support.dtype is not torch.bool:
            raise TypeError("support must be boolean")
        self.values = values
        self.support = support
        self.context = context

    @classmethod
    def from_sparse(
        cls,
        x: spconv.SparseConvTensor,
        context: SupportContext,
    ) -> "DenseSparseTensor":
        indices = x.indices.long()
        if indices.ndim != 2 or indices.shape[1] != 4:
            raise ValueError(f"expected [N,4] sparse indices, got {tuple(indices.shape)}")
        if torch.unique(indices, dim=0).shape[0] != indices.shape[0]:
            raise ValueError("input contains duplicate coordinates; dense scatter would be ambiguous")
        shape = (int(x.batch_size), int(x.features.shape[1]), *map(int, x.spatial_shape))
        values = torch.zeros(shape, device=x.features.device, dtype=x.features.dtype)
        support = torch.zeros(
            (shape[0], 1, *shape[2:]), device=x.features.device, dtype=torch.bool
        )
        b, d, h, w = indices.unbind(1)
        values[b, :, d, h, w] = x.features
        support[b, 0, d, h, w] = True
        return cls(values, support, context)

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def spatial_shape(self) -> List[int]:
        return list(map(int, self.values.shape[2:]))

    @property
    def indices(self) -> torch.Tensor:
        return self.support[:, 0].nonzero(as_tuple=False).to(torch.int32)

    @property
    def features(self) -> torch.Tensor:
        return self.values.permute(0, 2, 3, 4, 1)[self.support[:, 0]]

    def replace_feature(self, features: torch.Tensor) -> "DenseSparseTensor":
        if features.ndim != 2 or features.shape[0] != int(self.support.sum().item()):
            raise ValueError(
                f"replacement features {tuple(features.shape)} do not match active support"
            )
        shape = (self.batch_size, int(features.shape[1]), *self.spatial_shape)
        values = torch.zeros(shape, device=features.device, dtype=features.dtype)
        channel_last = values.permute(0, 2, 3, 4, 1)
        channel_last[self.support[:, 0]] = features
        return DenseSparseTensor(values, self.support, self.context)

    def restore_keyed_output(self, key: str) -> "DenseSparseTensor":
        if key not in self.context.keyed:
            raise RuntimeError(f"missing stored support for indice_key={key!r}")
        stored = self.context.keyed[key]
        if tuple(self.spatial_shape) != stored.output_shape:
            raise RuntimeError(
                f"{key}: stage shape {tuple(self.spatial_shape)} != captured "
                f"downsample output {stored.output_shape}"
            )
        values = self.values.masked_fill(~stored.output_support, 0)
        return DenseSparseTensor(values, stored.output_support, self.context)


def _masked(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    return values.masked_fill(~support, 0)


def _same_support(a: DenseSparseTensor, b: DenseSparseTensor, context: str) -> None:
    if tuple(a.values.shape) != tuple(b.values.shape):
        raise RuntimeError(f"{context}: dense shapes differ")
    if not torch.equal(a.support, b.support):
        mismatch = int(torch.logical_xor(a.support, b.support).sum().item())
        raise RuntimeError(f"{context}: support differs at {mismatch} coordinates")


def _sparse_add_dense(a: DenseSparseTensor, b: DenseSparseTensor) -> DenseSparseTensor:
    if a.context is not b.context:
        raise RuntimeError("cannot add tensors from different support contexts")
    if tuple(a.values.shape) != tuple(b.values.shape):
        raise RuntimeError("sparse_add operands have different dense shapes")
    support = torch.logical_or(a.support, b.support)
    values = a.values + b.values
    nonzero = values.abs().sum(dim=1, keepdim=True) > 1e-8
    support = torch.logical_and(support, nonzero)
    return DenseSparseTensor(_masked(values, support), support, a.context)


def _exact_add_dense(
    a: DenseSparseTensor,
    b: DenseSparseTensor,
    context: str,
) -> DenseSparseTensor:
    _same_support(a, b, context)
    return DenseSparseTensor(_masked(a.values + b.values, a.support), a.support, a.context)


def _weight_rule_label(kernel_size: Triple, stride: Triple) -> str:
    if kernel_size == (1, 1, 1) and stride == (1, 1, 1):
        return "pointwise fast path [I/G,O] -> OIDHW: reshape(I,O).t()"
    return "KRSC [O,D,H,W,I/G] -> OIDHW: permute(0,4,1,2,3)"


def _krsc_to_conv3d(
    weight: torch.Tensor,
    out_channels: int,
    in_channels: int,
    groups: int,
    kernel_size: Triple,
    stride: Triple,
) -> torch.Tensor:
    expected = (out_channels, *kernel_size, in_channels // groups)
    if tuple(weight.shape) != expected:
        raise RuntimeError(
            f"spconv KRSC weight expected {expected}, found {tuple(weight.shape)}"
        )
    if kernel_size == (1, 1, 1) and stride == (1, 1, 1):
        # Measured, not assumed. A 1x1x1 stride-1 convolution takes a pointwise
        # fast path in this spconv build, and that path reads the stored buffer
        # as [I/G, O] - the transpose of the KRSC reading that is correct
        # everywhere else. Recovered by least squares against the op's own
        # output, per class and per stride (probe_k1_types.py, probe_k1_stride.py):
        #                                    KRSC      transposed
        #   SubMConv3d   k=1 stride 1,1,1   6.67e-01   4.75e-09   <- transposed
        #   SparseConv3d k=1 stride 1,1,1   6.43e-01   5.37e-09   <- transposed
        #   SparseConv3d k=1 stride 1,2,2   9.89e-09   6.34e-01   <- KRSC
        #   SparseConv3d k=1 stride 2,2,2   1.17e-08   6.16e-01   <- KRSC
        # with the k=3 control matching permute(0,4,1,2,3) at 1.788139e-07.
        # The gate is stride, not operator class: the model's three
        # backbone.layer{2,3,4}.0.downsample.conv are k=1 SparseConv3d at stride
        # [1,2,2] and take the ordinary path, while the three fpn.lateral_c*.conv
        # are k=1 SubMConv3d at stride 1 and take the transposed one. Keying this
        # branch on kernel size alone repaired the laterals and broke the
        # backbone, moving c2's output deviation from 4.291534e-06 to
        # 3.745914e+00.
        if groups != 1:
            raise RuntimeError(
                "grouped 1x1x1 spconv weight layout is unverified on this build"
            )
        return (
            weight.reshape(in_channels, out_channels)
            .t()
            .reshape(out_channels, in_channels, 1, 1, 1)
            .contiguous()
        )
    return weight.permute(0, 4, 1, 2, 3).contiguous()


def _krsc_to_conv_transpose3d(
    weight: torch.Tensor,
    out_channels: int,
    in_channels: int,
    groups: int,
    kernel_size: Triple,
) -> torch.Tensor:
    expected = (out_channels, *kernel_size, in_channels // groups)
    if tuple(weight.shape) != expected:
        raise RuntimeError(
            f"spconv inverse KRSC weight expected {expected}, found {tuple(weight.shape)}"
        )
    out_per_group = out_channels // groups
    in_per_group = in_channels // groups
    grouped = weight.reshape(groups, out_per_group, *kernel_size, in_per_group)
    return grouped.permute(0, 5, 1, 2, 3, 4).reshape(
        in_channels, out_per_group, *kernel_size
    ).contiguous()


def _conv_output_shape(
    shape: Triple,
    kernel: Triple,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
) -> Triple:
    return tuple(
        (shape[i] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1)
        // stride[i]
        + 1
        for i in range(3)
    )  # type: ignore[return-value]


def _output_padding(
    input_shape: Triple,
    desired_shape: Triple,
    kernel: Triple,
    stride: Triple,
    padding: Triple,
    dilation: Triple,
) -> Triple:
    base = tuple(
        (input_shape[i] - 1) * stride[i]
        - 2 * padding[i]
        + dilation[i] * (kernel[i] - 1)
        + 1
        for i in range(3)
    )
    result = tuple(desired_shape[i] - base[i] for i in range(3))
    if any(v < 0 or v >= stride[i] for i, v in enumerate(result)):
        raise RuntimeError(
            f"cannot recover shape {desired_shape} from {input_shape}; "
            f"computed output_padding={result}"
        )
    return result  # type: ignore[return-value]


class DenseConvAdapter(nn.Module):
    def __init__(
        self,
        layer_name: str,
        sparse_module: nn.Module,
        kind: str,
        paired: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.layer_name = layer_name
        self.kind = kind
        self.in_channels = int(sparse_module.in_channels)
        self.out_channels = int(sparse_module.out_channels)
        self.kernel_size = _triple(sparse_module.kernel_size)
        self.groups = int(sparse_module.groups)
        self.indice_key = getattr(sparse_module, "indice_key", None)
        sparse_padding = _triple(sparse_module.padding)
        sparse_stride = _triple(sparse_module.stride)
        sparse_dilation = _triple(sparse_module.dilation)
        has_bias = sparse_module.bias is not None

        if kind == "submanifold":
            if sparse_stride != (1, 1, 1):
                raise RuntimeError(f"{layer_name}: SubMConv3d stride must be one")
            centered = tuple(
                sparse_dilation[i] * (self.kernel_size[i] - 1) // 2 for i in range(3)
            )
            if any(
                sparse_dilation[i] * (self.kernel_size[i] - 1) % 2 for i in range(3)
            ):
                raise RuntimeError(f"{layer_name}: even submanifold kernel extent is unsupported")
            self.stride, self.padding, self.dilation = sparse_stride, centered, sparse_dilation
            self.op = nn.Conv3d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=has_bias,
            )
            copied = _krsc_to_conv3d(
                sparse_module.weight,
                self.out_channels,
                self.in_channels,
                self.groups,
                self.kernel_size,
                self.stride,
            )
            permutation = _weight_rule_label(self.kernel_size, self.stride)
        elif kind == "regular":
            self.stride, self.padding, self.dilation = (
                sparse_stride,
                sparse_padding,
                sparse_dilation,
            )
            self.op = nn.Conv3d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=has_bias,
            )
            copied = _krsc_to_conv3d(
                sparse_module.weight,
                self.out_channels,
                self.in_channels,
                self.groups,
                self.kernel_size,
                self.stride,
            )
            permutation = _weight_rule_label(self.kernel_size, self.stride)
        elif kind == "inverse":
            if paired is None:
                raise RuntimeError(f"{layer_name}: inverse conv has no keyed forward conv")
            self.stride = paired["stride"]
            self.padding = paired["padding"]
            self.dilation = paired["dilation"]
            if self.kernel_size != paired["kernel_size"] or self.groups != paired["groups"]:
                raise RuntimeError(f"{layer_name}: inverse and keyed forward parameters differ")
            self.op = nn.ConvTranspose3d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=has_bias,
            )
            copied = _krsc_to_conv_transpose3d(
                sparse_module.weight,
                self.out_channels,
                self.in_channels,
                self.groups,
                self.kernel_size,
            )
            permutation = (
                "KRSC [O,D,H,W,I/G] -> grouped I(O/G)DHW: "
                "reshape(G,O/G,D,H,W,I/G), permute(0,5,1,2,3,4), reshape"
            )
        else:
            raise ValueError(f"unknown sparse convolution kind: {kind}")

        with torch.no_grad():
            self.op.weight.copy_(copied)
            if has_bias:
                self.op.bias.copy_(sparse_module.bias)
        support_kernel = torch.ones((1, 1, *self.kernel_size), dtype=torch.float32)
        self.register_buffer("_support_kernel", support_kernel, persistent=False)
        self.map_entry = {
            "layer": layer_name,
            "sparse_op": type(sparse_module).__name__,
            "dense_op": type(self.op).__name__,
            "kernel_size": list(self.kernel_size),
            "stride": list(self.stride),
            "sparse_padding": list(sparse_padding),
            "dense_padding": list(self.padding),
            "dilation": list(self.dilation),
            "groups": self.groups,
            "bias": has_bias,
            "indice_key": self.indice_key,
            "support_restoration": {
                "submanifold": "reuse input support",
                "regular": "kernel reachability from input support",
                "inverse": "stored input support of keyed forward downsample",
            }[kind],
            "mask_timed_separately": True,
            "weight_layout": "spconv 2.x KRSC",
            "weight_permutation": permutation,
        }

    def _regular_support(self, x: DenseSparseTensor) -> torch.Tensor:
        counts = F.conv3d(
            x.support.to(dtype=x.values.dtype),
            self._support_kernel.to(dtype=x.values.dtype),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        return counts > 0

    def forward(self, x: DenseSparseTensor) -> DenseSparseTensor:
        ledger = x.context.timings
        if self.kind == "inverse":
            if self.indice_key is None or self.indice_key not in x.context.keyed:
                raise RuntimeError(
                    f"{self.layer_name}: no stored support for indice_key={self.indice_key!r}"
                )
            stored = x.context.keyed[self.indice_key]
            if tuple(x.spatial_shape) != stored.output_shape:
                raise RuntimeError(
                    f"{self.layer_name}: inverse input shape does not match keyed output"
                )
            _output_padding(
                tuple(x.spatial_shape),
                stored.input_shape,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
            )
            values = ledger.measure(
                self.layer_name,
                "conv",
                lambda: self.op(
                    x.values,
                    output_size=(
                        x.batch_size,
                        self.out_channels,
                        *stored.input_shape,
                    ),
                ),
            )
            support = stored.input_support
        else:
            values = ledger.measure(self.layer_name, "conv", lambda: self.op(x.values))
            if self.kind == "submanifold":
                if tuple(values.shape[2:]) != tuple(x.spatial_shape):
                    raise RuntimeError(
                        f"{self.layer_name}: centered SubMConv3d changed spatial shape"
                    )
                support = x.support
            else:
                support = ledger.measure(
                    self.layer_name, "support", lambda: self._regular_support(x)
                )
                expected = _conv_output_shape(
                    tuple(x.spatial_shape),
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                )
                if tuple(values.shape[2:]) != expected:
                    raise RuntimeError(f"{self.layer_name}: Conv3d output-shape mismatch")
                if self.indice_key is not None:
                    if self.indice_key in x.context.keyed:
                        raise RuntimeError(f"duplicate indice_key={self.indice_key!r} in one forward")
                    x.context.keyed[self.indice_key] = StoredSupport(
                        x.support,
                        support,
                        tuple(x.spatial_shape),
                        tuple(map(int, support.shape[2:])),
                    )
        values = ledger.measure(
            self.layer_name, "mask", lambda: _masked(values, support)
        )
        return DenseSparseTensor(values, support, x.context)


def _dense_basic_block_forward(
    self: SparseBasicBlock, x: DenseSparseTensor
) -> DenseSparseTensor:
    identity = x
    out = self.conv1(x)
    out = out.replace_feature(self.bn1(out.features))
    out = self.lif1(out)
    out = self.conv2(out)
    out = out.replace_feature(self.bn2(out.features))
    if self.se is not None:
        out = self.se(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    return self.lif2(_sparse_add_dense(out, identity))


def _dense_downsample_forward(
    self: SparseDownsample, x: DenseSparseTensor
) -> DenseSparseTensor:
    out = self.conv(x)
    return out.replace_feature(self.bn(out.features))


def _dense_backbone_forward(
    self: SparseSEWResNetIC, x: DenseSparseTensor
) -> List[DenseSparseTensor]:
    x = self.stem(x)
    c1 = self.layer1(x)
    c2 = self.layer2(c1)
    key3 = self.DOWN_KEYS["layer3"]
    key4 = self.DOWN_KEYS["layer4"]
    c3 = self.layer3(c2).restore_keyed_output(key3)
    c4 = self.layer4(c3).restore_keyed_output(key4)
    return [c2, c3, c4]


def _dense_fpn_forward(
    self: SparseFPNIC,
    features: List[DenseSparseTensor],
    backbone: SparseSEWResNetIC,
) -> DenseSparseTensor:
    del backbone
    c2, c3, c4 = features
    p4 = self.lateral_c4(c4)
    p4_up = self.up_c4_to_c3(p4)
    p3 = self.lateral_c3(c3)
    p3f = _exact_add_dense(p3, p4_up, "IC-FPN fuse_c3")
    p3_up = self.up_c3_to_c2(p3f)
    p2 = self.lateral_c2(c2)
    p2 = _exact_add_dense(p2, p3_up, "IC-FPN fuse_c2")
    return self.out_block(p2)


def _kind(module: nn.Module) -> Optional[str]:
    if isinstance(module, spconv.SubMConv3d):
        return "submanifold"
    if isinstance(module, spconv.SparseInverseConv3d):
        return "inverse"
    if isinstance(module, spconv.SparseConv3d):
        return "regular"
    return None


def _forward_specs(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}
    for name, module in model.named_modules():
        if _kind(module) != "regular":
            continue
        key = getattr(module, "indice_key", None)
        if key is None:
            continue
        if key in specs:
            raise RuntimeError(f"indice_key={key!r} is assigned to more than one forward conv")
        specs[key] = {
            "layer": name,
            "kernel_size": _triple(module.kernel_size),
            "stride": _triple(module.stride),
            "padding": _triple(module.padding),
            "dilation": _triple(module.dilation),
            "groups": int(module.groups),
        }
    return specs


def _replace_sparse_convs(
    root: nn.Module,
    forward_specs: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    transplant_map: List[Dict[str, Any]] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            kind = _kind(child)
            if kind is None:
                visit(child, full_name)
                continue
            paired = None
            if kind == "inverse":
                key = getattr(child, "indice_key", None)
                paired = forward_specs.get(key)
            adapter = DenseConvAdapter(full_name, child, kind, paired)
            setattr(parent, child_name, adapter)
            transplant_map.append(adapter.map_entry)

    visit(root, "")
    sparse_types = tuple(
        sparse_type
        for sparse_type in (
            getattr(spconv, "SubMConv3d", None),
            getattr(spconv, "SparseConv3d", None),
            getattr(spconv, "SparseInverseConv3d", None),
            getattr(spconv, "SparseConvTranspose3d", None),
        )
        if sparse_type is not None
    )
    remaining = [
        (name, type(module).__name__)
        for name, module in root.named_modules()
        if isinstance(module, sparse_types)
    ]
    if remaining:
        raise RuntimeError(f"untransplanted sparse convolutions remain: {remaining}")
    return transplant_map


class DenseExecutionModel(nn.Module):
    def __init__(self, model: nn.Module, transplant_map: List[Dict[str, Any]]) -> None:
        super().__init__()
        self.model = model
        self.transplant_map = transplant_map

    def forward(
        self,
        x: spconv.SparseConvTensor,
        batch_size: Optional[int] = None,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        return_loss_inputs: bool = False,
        collect_timings: bool = False,
    ) -> DenseForwardResult:
        del targets
        if x.features.dtype is not torch.float32:
            raise TypeError(f"dense gate requires FP32 input, got {x.features.dtype}")
        if collect_timings and not x.features.is_cuda:
            raise RuntimeError("CUDA-event timing requires a CUDA input")
        context = SupportContext(TimingLedger(collect_timings))
        dense = DenseSparseTensor.from_sparse(x, context)
        if batch_size is None:
            batch_size = dense.batch_size
        backbone_features = self.model.backbone(dense)
        fused = self.model.fpn(backbone_features, self.model.backbone)
        features_2d, indices_2d, spatial_2d = sparse_temporal_pool(
            fused, mode=self.model.temporal_pool_mode
        )
        cls_logits, box_ltrb, ctr_logits = self.model.head(features_2d)
        if self.model.training or return_loss_inputs:
            outputs = {
                "cls_logits": cls_logits,
                "box_ltrb": box_ltrb,
                "ctr_logits": ctr_logits,
                "indices_2d": indices_2d,
                "spatial_2d": spatial_2d,
            }
        else:
            outputs = {
                "detections": self.model._decode_detections(
                    cls_logits, box_ltrb, ctr_logits, indices_2d, batch_size
                )
            }
        return DenseForwardResult(outputs, context.timings.materialize())


def transplant_sparse_model(sparse_model: nn.Module) -> DenseExecutionModel:
    """Deep-copy a loaded SparseVoxelDetIC and replace every sparse convolution."""
    dense_model = copy.deepcopy(sparse_model).float()
    if not isinstance(dense_model.backbone, SparseSEWResNetIC):
        raise TypeError("expected SparseVoxelDetIC with SparseSEWResNetIC backbone")
    if not isinstance(dense_model.fpn, SparseFPNIC):
        raise TypeError("expected SparseVoxelDetIC with SparseFPNIC")
    specs = _forward_specs(dense_model)
    transplant_map = _replace_sparse_convs(dense_model, specs)
    for module in dense_model.modules():
        if isinstance(module, SparseBasicBlock):
            module.forward = types.MethodType(_dense_basic_block_forward, module)
        elif isinstance(module, SparseDownsample):
            module.forward = types.MethodType(_dense_downsample_forward, module)
    dense_model.backbone.forward = types.MethodType(
        _dense_backbone_forward, dense_model.backbone
    )
    dense_model.fpn.forward = types.MethodType(_dense_fpn_forward, dense_model.fpn)
    non_fp32 = [
        name for name, parameter in dense_model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype is not torch.float32
    ]
    if non_fp32:
        raise RuntimeError(f"non-FP32 parameters after transplant: {non_fp32}")
    return DenseExecutionModel(dense_model, transplant_map)


__all__ = [
    "DenseExecutionModel",
    "DenseForwardResult",
    "transplant_sparse_model",
]
