#!/usr/bin/env python3
"""Query assignment for SparseTQDet.

Assigns query candidates to GT boxes for training losses.
Supports both single-object and multi-object frames.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


def pairwise_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute IoU matrix for xyxy boxes.

    Args:
        boxes1: [N, 4]
        boxes2: [M, 4]
    Returns:
        [N, M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    x11, y11, x12, y12 = boxes1.unbind(dim=-1)
    x21, y21, x22, y22 = boxes2.unbind(dim=-1)

    inter_x1 = torch.maximum(x11[:, None], x21[None, :])
    inter_y1 = torch.maximum(y11[:, None], y21[None, :])
    inter_x2 = torch.minimum(x12[:, None], x22[None, :])
    inter_y2 = torch.minimum(y12[:, None], y22[None, :])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter + eps

    return inter / union


def assign_queries_batch(
    pred_boxes: torch.Tensor,
    gt_boxes_batch: List[torch.Tensor],
    gt_labels_batch: List[torch.Tensor],
    query_scores: Optional[torch.Tensor] = None,
    positive_iou: float = 0.5,
    force_match: bool = True,
    force_match_min_iou: float = 0.10,
    soft_force_match: bool = True,
    mode: str = "max_iou",
    soft_labels: bool = True,
    max_pos_per_gt: Optional[int] = None,
    task_aligned_topk: int = 8,
    task_aligned_alpha: float = 1.0,
    task_aligned_beta: float = 6.0,
    task_aligned_min_iou: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Assign query targets for a batch.

    Args:
        pred_boxes: [B, K, 4] predicted query boxes.
        gt_boxes_batch: list of [M_b, 4].
        gt_labels_batch: list of [M_b] (unused for binary class now).
        query_scores: optional [B, K] confidence used by task-aligned assignment.
        positive_iou: IoU threshold for positive query label (max-iou mode).
        force_match: if True, ensure at least one positive when GT exists.
        mode: "max_iou" or "task_aligned".
        soft_labels: if True, positive labels use IoU soft targets.
        max_pos_per_gt: cap positive queries per GT.
        task_aligned_topk/alpha/beta/min_iou: task-aligned assignment controls.

    Returns:
        Dict with labels/targets/masks and matching metadata.
    """
    device = pred_boxes.device
    bsz, k, _ = pred_boxes.shape

    labels = torch.zeros((bsz, k), device=device, dtype=torch.float32)
    box_targets = torch.zeros((bsz, k, 4), device=device, dtype=torch.float32)
    iou_targets = torch.zeros((bsz, k), device=device, dtype=torch.float32)
    pos_mask = torch.zeros((bsz, k), device=device, dtype=torch.bool)
    matched_query_idx = torch.full((bsz,), -1, device=device, dtype=torch.long)
    matched_iou = torch.zeros((bsz,), device=device, dtype=torch.float32)
    proposal_recall_at_k = {
        "proposal_recall@16": 0.0,
        "proposal_recall@32": 0.0,
        "proposal_recall@64": 0.0,
        "proposal_recall@128": 0.0,
    }
    valid_gt_frames = 0

    mode = str(mode).lower()
    if mode not in {"max_iou", "task_aligned"}:
        raise ValueError(f"Unsupported query assignment mode: {mode}")
    max_pos_per_gt_int = int(max_pos_per_gt) if max_pos_per_gt is not None else None
    max_pos_per_gt_int = max(1, max_pos_per_gt_int) if max_pos_per_gt_int is not None else None
    task_topk = max(1, int(task_aligned_topk))
    task_alpha = float(task_aligned_alpha)
    task_beta = float(task_aligned_beta)
    task_min_iou = float(task_aligned_min_iou)

    for b in range(bsz):
        gt_boxes = gt_boxes_batch[b].to(device)
        _ = gt_labels_batch[b]  # kept for interface symmetry

        if gt_boxes.numel() == 0:
            continue
        valid_gt_frames += 1

        iou_mat = pairwise_iou_xyxy(pred_boxes[b], gt_boxes)  # [K, M]
        best_iou_per_query, best_gt_idx = iou_mat.max(dim=1)  # [K]
        assigned_gt_idx = best_gt_idx.clone()

        if mode == "task_aligned":
            if query_scores is None:
                cls_score = torch.ones_like(best_iou_per_query)
            else:
                cls_score = query_scores[b].to(device).detach().clamp(min=0.0, max=1.0)
                if cls_score.shape[0] != best_iou_per_query.shape[0]:
                    raise ValueError(
                        f"query_scores shape mismatch: got {tuple(cls_score.shape)}, "
                        f"expected ({best_iou_per_query.shape[0]},)"
                    )

            alignment_best = torch.full_like(best_iou_per_query, -1.0)
            assigned_gt_idx = torch.full_like(best_gt_idx, -1)
            num_gt = int(gt_boxes.shape[0])
            for g in range(num_gt):
                iou_col = iou_mat[:, g].clamp(min=0.0)
                align = cls_score.clamp(min=1e-6).pow(task_alpha) * iou_col.pow(task_beta)
                k_sel = min(task_topk, int(align.shape[0]))
                top_idx = torch.topk(align, k=k_sel, largest=True).indices
                if task_min_iou > 0.0:
                    top_idx = top_idx[iou_col[top_idx] >= task_min_iou]
                if max_pos_per_gt_int is not None and top_idx.numel() > max_pos_per_gt_int:
                    top_idx = top_idx[:max_pos_per_gt_int]
                if top_idx.numel() == 0:
                    continue

                cand_scores = align[top_idx]
                prev_scores = alignment_best[top_idx]
                replace = cand_scores > prev_scores
                if replace.any():
                    q_idx = top_idx[replace]
                    alignment_best[q_idx] = cand_scores[replace]
                    assigned_gt_idx[q_idx] = g

            query_pos = assigned_gt_idx >= 0
        else:
            query_pos = best_iou_per_query >= float(positive_iou)
            if max_pos_per_gt_int is not None and query_pos.any():
                capped = torch.zeros_like(query_pos)
                for g in range(int(gt_boxes.shape[0])):
                    idx = torch.where(query_pos & (best_gt_idx == g))[0]
                    if idx.numel() == 0:
                        continue
                    if idx.numel() > max_pos_per_gt_int:
                        ious = best_iou_per_query[idx]
                        sel = torch.topk(ious, k=max_pos_per_gt_int, largest=True).indices
                        idx = idx[sel]
                    capped[idx] = True
                query_pos = capped

        forced_soft_idx = -1
        forced_soft_val = 0.0
        if force_match and not query_pos.any():
            q = int(torch.argmax(best_iou_per_query).item())
            q_iou = float(best_iou_per_query[q].item())
            if q_iou >= float(force_match_min_iou):
                query_pos[q] = True
                assigned_gt_idx[q] = best_gt_idx[q]
                if soft_force_match:
                    forced_soft_idx = q
                    forced_soft_val = max(min(q_iou, 1.0), 0.05)

        pos_mask[b] = query_pos
        if query_pos.any():
            pos_iou = best_iou_per_query[query_pos]
            if soft_labels:
                labels[b, query_pos] = pos_iou.clamp(min=0.05, max=1.0)
            else:
                labels[b, query_pos] = 1.0
        if forced_soft_idx >= 0 and soft_force_match:
            labels[b, forced_soft_idx] = max(float(labels[b, forced_soft_idx].item()), forced_soft_val)
        iou_targets[b] = best_iou_per_query

        if query_pos.any():
            gt_idx = assigned_gt_idx[query_pos].clamp(min=0)
            box_targets[b, query_pos] = gt_boxes[gt_idx]
            q_best = int(torch.argmax(best_iou_per_query).item())
            matched_query_idx[b] = q_best
            matched_iou[b] = best_iou_per_query[q_best]

        # Proposal recall@K proxy: queries are already top-k by coarse proposal score.
        for k in (16, 32, 64, 128):
            kk = min(k, int(best_iou_per_query.shape[0]))
            if kk <= 0:
                continue
            recall_hit = bool((best_iou_per_query[:kk] >= 0.5).any().item())
            proposal_recall_at_k[f"proposal_recall@{k}"] += 1.0 if recall_hit else 0.0

    if valid_gt_frames > 0:
        for k in proposal_recall_at_k:
            proposal_recall_at_k[k] /= float(valid_gt_frames)

    positive_query_ratio = float(pos_mask.float().mean().item())

    return {
        "labels": labels,
        "box_targets": box_targets,
        "iou_targets": iou_targets,
        "pos_mask": pos_mask,
        "matched_query_idx": matched_query_idx,
        "matched_iou": matched_iou,
        "positive_query_ratio": positive_query_ratio,
        "proposal_recall_at_k": proposal_recall_at_k,
    }
