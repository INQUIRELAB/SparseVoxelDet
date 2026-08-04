#!/usr/bin/env python3
"""Quality-aligned loss arm with an exact strict-loss disabled path."""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import strict_loss
except ImportError:
    import strict_loss


_DIAGNOSTIC_KEYS = (
    "num_gt",
    "num_gt_with_candidates",
    "gt_zero_candidates",
    "dynamic_k_sum",
    "num_pos_raw",
    "quota_fill_ratio",
    "quota_deficit",
    "conflict_sites",
    "gt_zero_after_conflict",
    "multi_gt_samples",
    "multi_gt_gt_zero_assigned",
    "candidate_count_mean",
    "candidate_count_max",
    "classification_quality_target_mean",
    "classification_quality_target_max",
    "decoded_iou_target_mean",
    "decoded_iou_target_max",
)


def _pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0))[:, None]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0))[None, :]
    return (inter / (area1 + area2 - inter + eps)).clamp(0, 1)


def _aligned_giou_loss(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
    union = area1 + area2 - inter
    iou = inter / (union + eps)
    enc_lt = torch.minimum(boxes1[:, :2], boxes2[:, :2])
    enc_rb = torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
    enc_wh = (enc_rb - enc_lt).clamp_min(0)
    enc_area = enc_wh[:, 0] * enc_wh[:, 1]
    giou = iou - (enc_area - union) / (enc_area + eps)
    return 1.0 - giou


def _deferred_assign(
    candidate_mask: torch.Tensor,
    log_alignment: torch.Tensor,
    ious: torch.Tensor,
    cls_confidence: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    quotas: List[int],
) -> Tuple[List[int], List[List[int]]]:
    n_sites, n_gt = candidate_mask.shape
    mask_cpu = candidate_mask.detach().cpu()
    align_cpu = log_alignment.detach().cpu()
    iou_cpu = ious.detach().cpu()
    cls_cpu = cls_confidence.detach().cpu()
    boxes_cpu = gt_boxes.detach().cpu()
    labels_cpu = gt_labels.detach().cpu()

    canonical = []
    for g in range(n_gt):
        box_key = tuple(float(v) for v in boxes_cpu[g].tolist())
        label_key = int(labels_cpu[g].item()) if labels_cpu.numel() else 0
        area = max(0.0, box_key[2] - box_key[0]) * max(0.0, box_key[3] - box_key[1])
        canonical.append((area, box_key, label_key, g))

    preferences: List[List[int]] = []
    for g in range(n_gt):
        sites = [s for s in range(n_sites) if bool(mask_cpu[s, g])]
        sites.sort(key=lambda s: (-float(align_cpu[s, g]), -float(iou_cpu[s, g]), -float(cls_cpu[s]), s))
        preferences.append(sites)

    gt_order = sorted(range(n_gt), key=lambda g: (canonical[g][1], canonical[g][2], canonical[g][3]))
    owners = [-1] * n_sites
    assigned = [[] for _ in range(n_gt)]
    pointers = [0] * n_gt

    def priority(site: int, gt: int) -> tuple:
        area, box_key, label_key, original = canonical[gt]
        return (
            -float(align_cpu[site, gt]),
            -float(iou_cpu[site, gt]),
            -float(cls_cpu[site]),
            area,
            box_key,
            label_key,
            original,
        )

    while True:
        proposed = False
        for g in gt_order:
            if len(assigned[g]) >= quotas[g] or pointers[g] >= len(preferences[g]):
                continue
            site = preferences[g][pointers[g]]
            pointers[g] += 1
            proposed = True
            incumbent = owners[site]
            if incumbent < 0:
                owners[site] = g
                assigned[g].append(site)
            elif priority(site, g) < priority(site, incumbent):
                assigned[incumbent].remove(site)
                owners[site] = g
                assigned[g].append(site)
        if not proposed:
            break
    for sites in assigned:
        sites.sort()
    return owners, assigned


def assign_quality_targets(
    indices_2d: torch.Tensor,
    cls_logits: torch.Tensor,
    box_ltrb: torch.Tensor,
    gt_boxes_batch: List[torch.Tensor],
    gt_labels_batch: List[torch.Tensor],
    stride: int = 4,
    task_aligned_alpha: float = 1.0,
    task_aligned_beta: float = 6.0,
    dynamic_k_topq: int = 10,
) -> Dict[str, torch.Tensor]:
    device = cls_logits.device
    n_sites = indices_2d.shape[0]
    cls_quality = torch.zeros(n_sites, device=device, dtype=torch.float32)
    iou_targets = torch.zeros(n_sites, device=device, dtype=torch.float32)
    ltrb_targets = torch.zeros((n_sites, 4), device=device, dtype=torch.float32)
    assigned_gt = torch.full((n_sites,), -1, device=device, dtype=torch.long)
    target_boxes = torch.zeros((n_sites, 4), device=device, dtype=torch.float32)
    pos_mask = torch.zeros(n_sites, device=device, dtype=torch.bool)
    totals = {key: 0.0 for key in _DIAGNOSTIC_KEYS}
    candidate_counts: List[int] = []
    total_quota = 0
    total_assigned = 0

    device_type = device.type if device.type in {"cpu", "cuda"} else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        indices = indices_2d.detach()
        logits = cls_logits.detach().float()
        if logits.ndim != 2 or logits.shape != (n_sites, 1):
            raise ValueError(f"Quality-aligned binary loss requires [N, 1] logits, got {tuple(cls_logits.shape)}")
        confidence = torch.sigmoid(logits[:, 0])
        decoded_ltrb = torch.exp(box_ltrb.detach().float().clamp(max=10.0))
        decoded_boxes = strict_loss.decode_ltrb_to_boxes(indices, decoded_ltrb, stride=stride)
        batch_ids = indices[:, 0].long() if n_sites else indices.new_zeros((0,), dtype=torch.long)
        cy_all = indices[:, 1].float() * stride + stride / 2.0 if n_sites else decoded_boxes.new_zeros((0,))
        cx_all = indices[:, 2].float() * stride + stride / 2.0 if n_sites else decoded_boxes.new_zeros((0,))

        for b, (boxes_raw, labels_raw) in enumerate(zip(gt_boxes_batch, gt_labels_batch)):
            boxes = boxes_raw.to(device=device, dtype=torch.float32)
            labels = labels_raw.to(device=device)
            n_gt = int(boxes.shape[0])
            totals["num_gt"] += n_gt
            if n_gt > 1:
                totals["multi_gt_samples"] += 1
            if n_gt == 0:
                continue
            site_global = torch.where(batch_ids == b)[0]
            n_local = int(site_global.numel())
            if n_local == 0:
                candidate_counts.extend([0] * n_gt)
                totals["gt_zero_candidates"] += n_gt
                if n_gt > 1:
                    totals["multi_gt_gt_zero_assigned"] += n_gt
                continue

            cx = cx_all[site_global]
            cy = cy_all[site_global]
            left = cx[:, None] - boxes[None, :, 0]
            top = cy[:, None] - boxes[None, :, 1]
            right = boxes[None, :, 2] - cx[:, None]
            bottom = boxes[None, :, 3] - cy[:, None]
            ltrb = torch.stack((left, top, right, bottom), dim=-1)
            candidates = ltrb.amin(dim=-1) > 0
            pair_iou = _pairwise_iou(decoded_boxes[site_global], boxes)
            local_conf = confidence[site_global]
            log_alignment = (
                task_aligned_alpha * torch.log(local_conf[:, None].clamp_min(1e-8))
                + task_aligned_beta * torch.log(pair_iou.clamp_min(1e-8))
            )
            counts = candidates.sum(dim=0).tolist()
            candidate_counts.extend(int(v) for v in counts)
            totals["num_gt_with_candidates"] += sum(int(v > 0) for v in counts)
            totals["gt_zero_candidates"] += sum(int(v == 0) for v in counts)
            totals["conflict_sites"] += int((candidates.sum(dim=1) > 1).sum().item())

            quotas: List[int] = []
            for g, count_raw in enumerate(counts):
                count = int(count_raw)
                if count == 0:
                    quotas.append(0)
                    continue
                candidate_ious = pair_iou[candidates[:, g], g]
                top_count = min(dynamic_k_topq, count)
                quota = int(math.floor(float(torch.topk(candidate_ious, k=top_count).values.sum().item())))
                quotas.append(max(1, min(count, quota)))
            total_quota += sum(quotas)
            owners, assigned_lists = _deferred_assign(
                candidates, log_alignment, pair_iou, local_conf, boxes, labels, quotas
            )

            for g, sites in enumerate(assigned_lists):
                if counts[g] > 0 and not sites:
                    totals["gt_zero_after_conflict"] += 1
                if not sites:
                    continue
                local_sites = torch.tensor(sites, device=device, dtype=torch.long)
                global_sites = site_global[local_sites]
                selected_log = log_alignment[local_sites, g]
                normalized = torch.exp(selected_log - selected_log.max())
                selected_iou = pair_iou[local_sites, g]
                quality = (normalized * selected_iou.max()).clamp(0, 1)
                cls_quality[global_sites] = quality
                iou_targets[global_sites] = selected_iou.clamp(0, 1)
                ltrb_targets[global_sites] = ltrb[local_sites, g]
                target_boxes[global_sites] = boxes[g]
                assigned_gt[global_sites] = g
                pos_mask[global_sites] = True
                total_assigned += len(sites)
            if n_gt > 1:
                totals["multi_gt_gt_zero_assigned"] += sum(not sites for sites in assigned_lists)

    totals["dynamic_k_sum"] = float(total_quota)
    totals["num_pos_raw"] = float(total_assigned)
    totals["quota_deficit"] = float(max(total_quota - total_assigned, 0))
    totals["quota_fill_ratio"] = float(total_assigned / total_quota) if total_quota else 1.0
    totals["candidate_count_mean"] = float(sum(candidate_counts) / len(candidate_counts)) if candidate_counts else 0.0
    totals["candidate_count_max"] = float(max(candidate_counts)) if candidate_counts else 0.0
    if total_assigned:
        totals["classification_quality_target_mean"] = float(cls_quality[pos_mask].mean().item())
        totals["classification_quality_target_max"] = float(cls_quality[pos_mask].max().item())
        totals["decoded_iou_target_mean"] = float(iou_targets[pos_mask].mean().item())
        totals["decoded_iou_target_max"] = float(iou_targets[pos_mask].max().item())
    diagnostics = {key: torch.tensor(value, device=device, dtype=torch.float32) for key, value in totals.items()}
    return {
        "pos_mask": pos_mask,
        "assigned_gt": assigned_gt,
        "ltrb_targets": ltrb_targets,
        "target_boxes": target_boxes,
        "cls_quality": cls_quality.detach(),
        "iou_targets": iou_targets.detach(),
        **diagnostics,
    }


class SparseVoxelDetLoss(nn.Module):
    """Strict-compatible loss with a config-gated quality-aligned mode."""

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cls_weight: float = 1.0,
        reg_weight: float = 2.0,
        ctr_weight: float = 1.0,
        stride: int = 4,
        center_sampling_radius: float = 1.5,
        num_classes: int = 1,
        use_qfl: bool = False,
        nwd_weight: float = 0.0,
        nwd_c: float = 12.8,
        task_aligned_enabled: bool = False,
        task_aligned_alpha: float = 1.0,
        task_aligned_beta: float = 6.0,
        dynamic_k_topq: int = 10,
        quality_bootstrap_epochs: int = 2,
    ) -> None:
        super().__init__()
        self.strict = strict_loss.SparseVoxelDetLoss(
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            cls_weight=cls_weight,
            reg_weight=reg_weight,
            ctr_weight=ctr_weight,
            stride=stride,
            center_sampling_radius=center_sampling_radius,
            num_classes=num_classes,
            use_qfl=use_qfl,
            nwd_weight=nwd_weight,
            nwd_c=nwd_c,
        )
        self.task_aligned_enabled = bool(task_aligned_enabled)
        self.task_aligned_alpha = float(task_aligned_alpha)
        self.task_aligned_beta = float(task_aligned_beta)
        self.dynamic_k_topq = int(dynamic_k_topq)
        self.quality_bootstrap_epochs = int(quality_bootstrap_epochs)
        self.cls_weight = float(cls_weight)
        self.reg_weight = float(reg_weight)
        self.ctr_weight = float(ctr_weight)
        self.stride = int(stride)
        self.nwd_weight = float(nwd_weight)
        self.nwd_c = float(nwd_c)
        self.epoch = 0
        self.cls_loss_fn = strict_loss.BinaryQualityFocalLoss(alpha=focal_alpha, beta=2.0)
        if self.task_aligned_enabled:
            expected = (self.task_aligned_alpha, self.task_aligned_beta, self.dynamic_k_topq, self.quality_bootstrap_epochs)
            if expected != (1.0, 6.0, 10, 2):
                raise ValueError(f"Rung-1 assignment contract mismatch: {expected}")
            if self.nwd_weight != 0.5 or self.nwd_c != 12.8:
                raise ValueError(f"Rung-1 regression contract requires nwd_weight=0.5 and nwd_c=12.8, got {self.nwd_weight}, {self.nwd_c}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        gt_boxes_batch: List[torch.Tensor],
        gt_labels_batch: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not self.task_aligned_enabled:
            return self.strict(outputs, gt_boxes_batch, gt_labels_batch)

        cls_logits = outputs["cls_logits"]
        box_ltrb = outputs["box_ltrb"]
        ctr_logits = outputs["ctr_logits"]
        indices_2d = outputs["indices_2d"]
        device_type = cls_logits.device.type if cls_logits.device.type in {"cpu", "cuda"} else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            targets = assign_quality_targets(
                indices_2d=indices_2d,
                cls_logits=cls_logits,
                box_ltrb=box_ltrb,
                gt_boxes_batch=gt_boxes_batch,
                gt_labels_batch=gt_labels_batch,
                stride=self.stride,
                task_aligned_alpha=self.task_aligned_alpha,
                task_aligned_beta=self.task_aligned_beta,
                dynamic_k_topq=self.dynamic_k_topq,
            )
            pos_mask = targets["pos_mask"]
            num_pos_raw = pos_mask.sum()
            num_pos = num_pos_raw.clamp(min=1).float()
            blend = min(max(self.epoch / max(self.quality_bootstrap_epochs, 1), 0.0), 1.0)
            cls_targets = torch.zeros_like(targets["cls_quality"])
            cls_targets[pos_mask] = (1.0 - blend) + blend * targets["cls_quality"][pos_mask]
            cls_loss = self.cls_loss_fn(cls_logits.float(), cls_targets, pos_mask=pos_mask) / num_pos

            if pos_mask.any():
                decoded_ltrb = torch.exp(box_ltrb[pos_mask].float().clamp(max=10.0))
                pred_boxes = strict_loss.decode_ltrb_to_boxes(indices_2d[pos_mask], decoded_ltrb, stride=self.stride)
                target_boxes = targets["target_boxes"][pos_mask]
                giou = _aligned_giou_loss(pred_boxes, target_boxes)
                nwd = strict_loss.nwd_loss_xyxy(pred_boxes, target_boxes, c=self.nwd_c)
                reg_loss = ((1.0 - self.nwd_weight) * giou + self.nwd_weight * nwd).mean()
                quality_loss = F.binary_cross_entropy_with_logits(
                    ctr_logits[pos_mask].float().squeeze(1),
                    targets["iou_targets"][pos_mask],
                    reduction="mean",
                )
            else:
                reg_loss = box_ltrb.float().sum() * 0.0
                quality_loss = ctr_logits.float().sum() * 0.0

            total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss + self.ctr_weight * quality_loss

        result = {
            "loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "reg_loss": reg_loss.detach(),
            "ctr_loss": quality_loss.detach(),
            "quality_loss": quality_loss.detach(),
            "num_pos": num_pos.detach(),
            "num_pos_raw": num_pos_raw.detach(),
            "bootstrap_blend": torch.tensor(blend, device=cls_logits.device, dtype=torch.float32),
        }
        result.update({key: targets[key].detach() for key in _DIAGNOSTIC_KEYS})
        return result
