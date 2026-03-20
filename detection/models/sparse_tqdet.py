#!/usr/bin/env python3
"""Sparse Temporal Query Detector (SparseTQDet).

Full-rewrite detector path for event-based drone detection:
1) Sparse encoder + temporal bridge + FPN/PAN backbone features.
2) Proposal head emits coarse candidate logits + boxes.
3) Query refinement stack applies self/cross attention for top-K candidates.
4) Output heads predict class logit, box deltas, IoU-quality, uncertainty.

Inference supports both lanes:
- paper parity (typically max_detections=100)
- reliability strict top-1 (max_detections=1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from torchvision.ops import batched_nms

from detection.models.fcos_head import Scale, generate_points
from detection.models.sparse_fcos_detector import (
    MotionAwareBridge,
    PANBottomUp,
    SimpleFPN,
    TemporalAttentionBridge,
    TemporalGroupBridge,
)
from backbone.sparse_sew_resnet import SparseSEWResNet


@dataclass
class DecodeParams:
    score_thresh: float = 0.001
    nms_thresh: float = 0.3
    max_detections: int = 1


class ProposalHead(nn.Module):
    """Dense proposal head over multi-scale features."""

    def __init__(self, in_channels: int, hidden_channels: int = 128, num_levels: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8 if hidden_channels % 8 == 0 else 4, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8 if hidden_channels % 8 == 0 else 4, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.obj_logits = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        self.box_ltrb = nn.Conv2d(hidden_channels, 4, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(num_levels)])

        nn.init.normal_(self.obj_logits.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.obj_logits.bias, -2.2)
        nn.init.normal_(self.box_ltrb.weight, mean=0.0, std=0.01)
        # Start from non-degenerate coarse boxes (~16x16 around feature points).
        nn.init.constant_(self.box_ltrb.bias, 8.0)

    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        obj_out: List[torch.Tensor] = []
        box_out: List[torch.Tensor] = []
        feat_out: List[torch.Tensor] = []
        for lvl, feat in enumerate(features):
            h = self.stem(feat)
            obj_out.append(self.obj_logits(h))
            box_out.append(F.relu(self.scales[min(lvl, len(self.scales) - 1)](self.box_ltrb(h))))
            feat_out.append(h)
        return obj_out, box_out, feat_out


class BoxPositionalEncoding(nn.Module):
    """Learnable box-based positional encoding for attention queries."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(4, dim)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)

    def forward(self, boxes: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        cx = 0.5 * (boxes[..., 0] + boxes[..., 2]) / float(max(img_w, 1))
        cy = 0.5 * (boxes[..., 1] + boxes[..., 3]) / float(max(img_h, 1))
        w = (boxes[..., 2] - boxes[..., 0]).clamp(min=0.0) / float(max(img_w, 1))
        h = (boxes[..., 3] - boxes[..., 1]).clamp(min=0.0) / float(max(img_h, 1))
        return self.proj(torch.stack([cx, cy, w, h], dim=-1))


class QueryRefinementBlock(nn.Module):
    """Lightweight transformer-style query refinement."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        q: torch.Tensor,
        memory: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_attn = q if query_pos is None else (q + query_pos)
        m_attn = memory if memory_pos is None else (memory + memory_pos)

        q_res, _ = self.self_attn(q_attn, q_attn, q_attn, need_weights=False)
        q = self.norm1(q + q_res)

        q_cross = q if query_pos is None else (q + query_pos)
        q_res, _ = self.cross_attn(q_cross, m_attn, m_attn, need_weights=False)
        q = self.norm2(q + q_res)

        q_res = self.ffn(q)
        q = self.norm3(q + q_res)
        return q


class SparseTQDet(nn.Module):
    """Sparse Temporal Query Detector."""

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 1,
        backbone_size: str = "nano",
        fpn_channels: int = 128,
        num_refine_layers: int = 4,
        topk: int = 64,
        strides: List[int] = [4, 8, 16],
        input_size: Tuple[int, int] = (640, 640),
        time_bins: int = 15,
        bridge_type: str = "motion_aware",
        num_temporal_groups: int = 3,
        use_pan: bool = True,
        query_heads: int = 8,
        memory_pool: Tuple[int, int] = (8, 8),
        use_uncertainty_in_score: bool = True,
        use_centerness_in_score: bool = True,
        ranking_score_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.topk = int(topk)
        self.strides = list(strides)
        self.input_size = tuple(input_size)
        self.time_bins = int(time_bins)
        self.memory_pool = tuple(memory_pool)
        self.use_uncertainty_in_score = bool(use_uncertainty_in_score)
        self.use_centerness_in_score = bool(use_centerness_in_score)
        self.ranking_score_weight = float(ranking_score_weight)

        self.backbone = SparseSEWResNet(
            in_channels=in_channels,
            size=backbone_size,
            stem_stride=(1, 2, 2),
        )
        backbone_channels = self.backbone.out_channels

        self.feature_sizes = [(self.input_size[0] // s, self.input_size[1] // s) for s in self.strides]

        if bridge_type == "attention":
            self.bridge = TemporalAttentionBridge(
                in_channels=backbone_channels,
                out_channels=backbone_channels,
                spatial_shapes=self.feature_sizes,
                time_bins=time_bins,
            )
        elif bridge_type == "group":
            self.bridge = TemporalGroupBridge(
                in_channels=backbone_channels,
                out_channels=backbone_channels,
                spatial_shapes=self.feature_sizes,
                num_groups=num_temporal_groups,
                time_bins=time_bins,
            )
        elif bridge_type == "motion_aware":
            self.bridge = MotionAwareBridge(
                in_channels=backbone_channels,
                out_channels=backbone_channels,
                spatial_shapes=self.feature_sizes,
                time_bins=time_bins,
            )
        else:
            raise ValueError(f"Unknown bridge_type: {bridge_type}")

        self.fpn = SimpleFPN(in_channels=backbone_channels, out_channels=fpn_channels)
        self.use_pan = bool(use_pan)
        if self.use_pan:
            self.pan = PANBottomUp(channels=fpn_channels, num_levels=len(self.strides))

        self.proposal_head = ProposalHead(
            in_channels=fpn_channels,
            hidden_channels=fpn_channels,
            num_levels=len(self.strides),
        )

        self.refine_blocks = nn.ModuleList(
            [QueryRefinementBlock(dim=fpn_channels, num_heads=query_heads) for _ in range(num_refine_layers)]
        )
        self.query_pos_enc = BoxPositionalEncoding(dim=fpn_channels)
        self.memory_pos_enc = nn.Linear(4, fpn_channels)
        nn.init.xavier_uniform_(self.memory_pos_enc.weight, gain=0.1)
        nn.init.zeros_(self.memory_pos_enc.bias)

        self.cls_head = nn.Linear(fpn_channels, 1)
        self.ctr_head = nn.Linear(fpn_channels, 1)
        self.rank_head = nn.Linear(fpn_channels, 1)
        self.box_delta_heads = nn.ModuleList([nn.Linear(fpn_channels, 4) for _ in range(num_refine_layers)])
        self.box_delta_head = nn.Linear(fpn_channels, 4)
        self.iou_quality_head = nn.Linear(fpn_channels, 1)
        self.uncertainty_head = nn.Linear(fpn_channels, 1)

        self._decode_params = DecodeParams()
        self._cached_points: Optional[List[torch.Tensor]] = None

    def set_decode_params(
        self,
        score_thresh: Optional[float] = None,
        nms_thresh: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> None:
        if score_thresh is not None:
            self._decode_params.score_thresh = float(score_thresh)
        if nms_thresh is not None:
            self._decode_params.nms_thresh = float(nms_thresh)
        if max_detections is not None:
            self._decode_params.max_detections = int(max_detections)

    def _get_points(self, device: torch.device) -> List[torch.Tensor]:
        if self._cached_points is not None and self._cached_points[0].device == device:
            return self._cached_points
        self._cached_points = generate_points(self.feature_sizes, self.strides, device)
        return self._cached_points

    @staticmethod
    def _decode_ltrb(pts: torch.Tensor, ltrb: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        x1 = pts[..., 0] - ltrb[..., 0]
        y1 = pts[..., 1] - ltrb[..., 1]
        x2 = pts[..., 0] + ltrb[..., 2]
        y2 = pts[..., 1] + ltrb[..., 3]
        # Avoid in-place updates here to keep autograd graph stable.
        x1 = x1.clamp(min=0, max=img_w)
        y1 = y1.clamp(min=0, max=img_h)
        x2 = x2.clamp(min=0, max=img_w)
        y2 = y2.clamp(min=0, max=img_h)
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _gather_batched(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # values: [B, N, C], idx: [B, K]
        b, _, c = values.shape
        idx_exp = idx.unsqueeze(-1).expand(b, idx.shape[1], c)
        return torch.gather(values, 1, idx_exp)

    @staticmethod
    def _apply_deltas(boxes: torch.Tensor, deltas: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        # boxes/deltas: [B, K, 4], xyxy + normalized deltas
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        w = (x2 - x1).clamp(min=1.0)
        h = (y2 - y1).clamp(min=1.0)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        dx = torch.tanh(deltas[..., 0]) * 0.5
        dy = torch.tanh(deltas[..., 1]) * 0.5
        dw = torch.tanh(deltas[..., 2]) * 0.4
        dh = torch.tanh(deltas[..., 3]) * 0.4

        cx2 = cx + dx * w
        cy2 = cy + dy * h
        w2 = w * torch.exp(dw)
        h2 = h * torch.exp(dh)

        x1n = (cx2 - 0.5 * w2).clamp(min=0, max=img_w)
        y1n = (cy2 - 0.5 * h2).clamp(min=0, max=img_h)
        x2n = (cx2 + 0.5 * w2).clamp(min=0, max=img_w)
        y2n = (cy2 + 0.5 * h2).clamp(min=0, max=img_h)

        return torch.stack([x1n, y1n, x2n, y2n], dim=-1)

    def _build_memory_tokens(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens: List[torch.Tensor] = []
        positions: List[torch.Tensor] = []
        ph, pw = self.memory_pool
        n_levels = max(len(features), 1)
        for lvl, feat in enumerate(features):
            pooled = F.adaptive_avg_pool2d(feat, (ph, pw))  # [B, C, ph, pw]
            tokens.append(pooled.flatten(2).transpose(1, 2))  # [B, ph*pw, C]

            bsz = feat.shape[0]
            gy, gx = torch.meshgrid(
                torch.linspace(0.0, 1.0, ph, device=feat.device),
                torch.linspace(0.0, 1.0, pw, device=feat.device),
                indexing="ij",
            )
            level = torch.full_like(gx, float(lvl) / float(max(n_levels - 1, 1)))
            one = torch.ones_like(gx)
            pos = torch.stack([gx, gy, level, one], dim=-1).reshape(1, ph * pw, 4).expand(bsz, -1, -1)
            positions.append(pos)
        tok = torch.cat(tokens, dim=1)
        pos = torch.cat(positions, dim=1)
        return tok, self.memory_pos_enc(pos)

    def _decode_predictions(
        self,
        pred_boxes: torch.Tensor,
        pred_logits: torch.Tensor,
        ctr_logits: torch.Tensor,
        iou_quality_logits: torch.Tensor,
        ranking_logits: torch.Tensor,
        uncertainty: torch.Tensor,
        score_thresh: Optional[float] = None,
        nms_thresh: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        score_thresh = self._decode_params.score_thresh if score_thresh is None else float(score_thresh)
        nms_thresh = self._decode_params.nms_thresh if nms_thresh is None else float(nms_thresh)
        max_detections = self._decode_params.max_detections if max_detections is None else int(max_detections)

        bsz, _, _ = pred_boxes.shape
        det_list: List[torch.Tensor] = []
        unc_list: List[torch.Tensor] = []

        for b in range(bsz):
            boxes = pred_boxes[b]
            cls_prob = torch.sigmoid(pred_logits[b])
            ctr_prob = torch.sigmoid(ctr_logits[b])
            iouq_prob = torch.sigmoid(iou_quality_logits[b])
            if self.use_uncertainty_in_score:
                score = cls_prob * iouq_prob * torch.exp(-uncertainty[b].clamp(min=0.0, max=8.0))
            else:
                score = cls_prob * iouq_prob
            if self.use_centerness_in_score:
                score = score * ctr_prob
            if self.ranking_score_weight > 0.0:
                rank_prob = torch.sigmoid(ranking_logits[b])
                score = score * rank_prob.pow(self.ranking_score_weight)

            keep = score > score_thresh
            if keep.any():
                boxes_k = boxes[keep]
                score_k = score[keep]
                unc_k = uncertainty[b][keep]
                labels = torch.zeros_like(score_k, dtype=torch.long)
                keep_idx = batched_nms(boxes_k, score_k, labels, nms_thresh)
                if max_detections > 0:
                    keep_idx = keep_idx[:max_detections]
                boxes_k = boxes_k[keep_idx]
                score_k = score_k[keep_idx]
                unc_k = unc_k[keep_idx]
                det = torch.cat([boxes_k, score_k.unsqueeze(1), torch.zeros(score_k.shape[0], 1, device=score_k.device)], dim=1)
                det_list.append(det)
                unc_list.append(unc_k)
            else:
                det_list.append(torch.zeros((0, 6), device=pred_boxes.device))
                unc_list.append(torch.zeros((0,), device=pred_boxes.device))

        max_len = max((d.shape[0] for d in det_list), default=0)
        if max_len == 0:
            return {
                "detections": torch.zeros((bsz, 0, 6), device=pred_boxes.device),
                "uncertainties": torch.zeros((bsz, 0), device=pred_boxes.device),
            }

        det_pad = torch.zeros((bsz, max_len, 6), device=pred_boxes.device)
        unc_pad = torch.zeros((bsz, max_len), device=pred_boxes.device)
        for i, (d, u) in enumerate(zip(det_list, unc_list)):
            if d.numel() > 0:
                det_pad[i, : d.shape[0]] = d
                unc_pad[i, : u.shape[0]] = u
        return {"detections": det_pad, "uncertainties": unc_pad}

    def get_num_params(self) -> Dict[str, int]:
        counts = {
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
            "bridge": sum(p.numel() for p in self.bridge.parameters()),
            "fpn": sum(p.numel() for p in self.fpn.parameters()),
            "proposal_head": sum(p.numel() for p in self.proposal_head.parameters()),
            "query_refine": sum(p.numel() for p in self.refine_blocks.parameters()),
            "output_heads": (
                sum(p.numel() for p in self.cls_head.parameters())
                + sum(p.numel() for p in self.ctr_head.parameters())
                + sum(p.numel() for p in self.rank_head.parameters())
                + sum(p.numel() for p in self.box_delta_heads.parameters())
                + sum(p.numel() for p in self.box_delta_head.parameters())
                + sum(p.numel() for p in self.iou_quality_head.parameters())
                + sum(p.numel() for p in self.uncertainty_head.parameters())
                + sum(p.numel() for p in self.query_pos_enc.parameters())
                + sum(p.numel() for p in self.memory_pos_enc.parameters())
            ),
        }
        if self.use_pan:
            counts["pan"] = sum(p.numel() for p in self.pan.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    def forward(
        self,
        x: spconv.SparseConvTensor,
        batch_size: Optional[int] = None,
        targets: Optional[Dict] = None,
        return_loss_inputs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if batch_size is None:
            batch_size = int(x.indices[:, 0].max().item()) + 1

        device = x.features.device
        sparse_features = self.backbone(x)
        dense_features = self.bridge(sparse_features, batch_size)
        fpn_features = self.fpn(dense_features)
        if self.use_pan:
            fpn_features = self.pan(fpn_features)

        obj_levels, ltrb_levels, feat_levels = self.proposal_head(fpn_features)
        points = self._get_points(device)

        all_logits: List[torch.Tensor] = []
        all_boxes: List[torch.Tensor] = []
        all_feats: List[torch.Tensor] = []
        img_h, img_w = self.input_size

        for lvl, (obj, ltrb, feat, pts) in enumerate(zip(obj_levels, ltrb_levels, feat_levels, points)):
            bsz, _, h, w = obj.shape
            obj_flat = obj.permute(0, 2, 3, 1).reshape(bsz, h * w)  # logits
            ltrb_flat = ltrb.permute(0, 2, 3, 1).reshape(bsz, h * w, 4)
            feat_flat = feat.permute(0, 2, 3, 1).reshape(bsz, h * w, feat.shape[1])

            pts_expand = pts.unsqueeze(0).expand(bsz, -1, -1)
            boxes = self._decode_ltrb(pts_expand, ltrb_flat, img_h, img_w)

            all_logits.append(obj_flat)
            all_boxes.append(boxes)
            all_feats.append(feat_flat)

        cand_logits = torch.cat(all_logits, dim=1)  # [B, N]
        cand_boxes = torch.cat(all_boxes, dim=1)    # [B, N, 4]
        cand_feats = torch.cat(all_feats, dim=1)    # [B, N, C]

        k = min(self.topk, cand_logits.shape[1])
        topk_logits, topk_idx = torch.topk(cand_logits, k=k, dim=1)
        coarse_boxes = self._gather_batched(cand_boxes, topk_idx)
        query_feats = self._gather_batched(cand_feats, topk_idx)

        memory, memory_pos = self._build_memory_tokens(fpn_features)
        q = query_feats
        refined_boxes = coarse_boxes
        for i, blk in enumerate(self.refine_blocks):
            q_pos = self.query_pos_enc(refined_boxes, img_h, img_w)
            q = blk(q, memory, query_pos=q_pos, memory_pos=memory_pos)
            delta_i = self.box_delta_heads[i](q)
            refined_boxes = self._apply_deltas(refined_boxes, delta_i, img_h, img_w)

        pred_logits = self.cls_head(q).squeeze(-1)
        ctr_logits = self.ctr_head(q).squeeze(-1)
        ranking_logits = self.rank_head(q).squeeze(-1)
        iouq_logits = self.iou_quality_head(q).squeeze(-1)
        uncertainty = F.softplus(self.uncertainty_head(q).squeeze(-1))
        if len(self.refine_blocks) == 0:
            box_delta = self.box_delta_head(q)
            pred_boxes = self._apply_deltas(coarse_boxes, box_delta, img_h, img_w)
        else:
            pred_boxes = refined_boxes

        if self.training or return_loss_inputs:
            return {
                "proposal_logits": topk_logits,
                "proposal_boxes": coarse_boxes,
                "pred_logits": pred_logits,
                "ctr_logits": ctr_logits,
                "pred_boxes": pred_boxes,
                "iou_quality_logits": iouq_logits,
                "ranking_logits": ranking_logits,
                "uncertainty": uncertainty,
                "query_indices": topk_idx,
            }

        return self._decode_predictions(
            pred_boxes=pred_boxes,
            pred_logits=pred_logits,
            ctr_logits=ctr_logits,
            iou_quality_logits=iouq_logits,
            ranking_logits=ranking_logits,
            uncertainty=uncertainty,
        )
