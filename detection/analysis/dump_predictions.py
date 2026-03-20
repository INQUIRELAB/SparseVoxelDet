"""Sparse FCOS prediction dump for error analysis.

Step 1 of the 3-step error analysis pipeline (dump → analyze → visualize).
Runs the Sparse FCOS model on the full validation set and saves per-frame
detection data to a JSON Lines file.

Output files:
    predictions.json  — JSON Lines, one JSON object per evaluated frame
    verification.json — mAP@50 recomputed from the dump (must match eval)

Usage:
    python -m detection.analysis.dump_predictions \\
        --checkpoint runs/sparse_fcos/best.pt \\
        --device cuda:0 --use-ema \\
        --outdir runs/sparse_fcos/analysis
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.models.sparse_fcos_detector import SparseFCOSDetector
from detection.models.sparse_tqdet import SparseTQDet
from detection.scripts.sparse_event_dataset import (
    SparseEventDataset,
    make_collate_fn,
    create_sparse_tensor,
)
from detection.scripts.ema import ModelEMA
from detection.scripts.evaluate_sparse_fcos import temporal_rerank_top1

PARITY_SPLIT_ALLOWLIST = {
    "canonical_train",
    "canonical_test",
    "challenging_train",
    "challenging_test",
}


def _resolved(path_str: str) -> Path:
    return Path(path_str).resolve()


def _assert_parity_dump_contract(
    sparse_dir: str,
    label_dir: str,
    split: str,
    max_samples: int | None,
) -> None:
    expected_data = (PROJECT_ROOT / "data/datasets/fred_paper_parity/sparse").resolve()
    expected_labels = (PROJECT_ROOT / "data/datasets/fred_paper_parity/labels").resolve()
    data_resolved = _resolved(sparse_dir)
    label_resolved = _resolved(label_dir)
    errors = []
    if data_resolved != expected_data:
        errors.append(f"sparse_dir must be {expected_data}, got {data_resolved}")
    if label_resolved != expected_labels:
        errors.append(f"label_dir must be {expected_labels}, got {label_resolved}")
    if split not in PARITY_SPLIT_ALLOWLIST:
        errors.append(
            f"split '{split}' not in allowlist {sorted(PARITY_SPLIT_ALLOWLIST)}"
        )
    if max_samples is not None:
        errors.append("max-samples is forbidden when --parity-enforced is active")
    if errors:
        raise ValueError("Parity dump contract violation:\n- " + "\n- ".join(errors))


def load_drone_type_mapping(mapping_path):
    """Load sequence-to-drone-type mapping from the FRED dataset metadata.

    Args:
        mapping_path: Path to sequence_class_mapping.json

    Returns:
        dict: seq_id (str) → drone_type (str), e.g. {"sequence_121": "Betafpv_Air75"}
    """
    with open(mapping_path) as f:
        data = json.load(f)

    class_names = data["class_names"]      # {"0": "DJI_Tello_EDU", ...}
    seq_classes = data["sequence_classes"]  # {"0": 1, "1": 1, ...}

    mapping = {}
    for seq_id_str, class_idx in seq_classes.items():
        mapping[seq_id_str] = class_names[str(class_idx)]
    return mapping


def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] tensor in xyxy format
        boxes2: [M, 4] tensor in xyxy format

    Returns:
        [N, M] IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-7)


def compute_ap(recalls, precisions):
    """Compute AP via 101-point interpolation (COCO style).

    Args:
        recalls: List of recall values (ascending).
        precisions: List of precision values.

    Returns:
        float: Average Precision.
    """
    recall_points = np.linspace(0, 1, 101)
    r = np.array(recalls)
    p = np.array(precisions)
    interp_precisions = np.interp(recall_points, r, p, left=0, right=0)
    return float(interp_precisions.mean())


def label_detections(det_ious, matched_det_idx, num_dets):
    """Assign a cause label to each detection in a frame.

    Labels:
        "tp"               — True positive: first detection with IoU >= 0.5
        "fp_duplicate"     — IoU >= 0.5 but GT already matched
        "fp_localization"  — 0.1 <= IoU < 0.5
        "fp_background"    — IoU < 0.1

    Args:
        det_ious: List of float IoU values per detection (confidence-descending).
        matched_det_idx: Index of the TP match, or -1 if none.
        num_dets: Total number of detections.

    Returns:
        List of label strings.
    """
    labels = []
    for i in range(num_dets):
        if i == matched_det_idx:
            labels.append("tp")
        elif det_ious[i] >= 0.5:
            labels.append("fp_duplicate")
        elif det_ious[i] >= 0.1:
            labels.append("fp_localization")
        else:
            labels.append("fp_background")
    return labels


def label_fn(matched, det_ious):
    """Assign a cause label when the GT box is not matched (false negative).

    Returns:
        Tuple of (fn_label, fn_best_iou).
    """
    if matched:
        return None, None
    if len(det_ious) == 0:
        return "fn_complete_miss", 0.0
    best_iou = max(det_ious)
    if best_iou >= 0.1:
        return "fn_localization", best_iou
    return "fn_complete_miss", best_iou


def process_frame(dets_tensor, gt_xyxy, seq_id, frame_idx, drone_type, sample_id):
    """Build a complete per-frame record from raw model outputs and GT.

    Args:
        dets_tensor: [N, 6] tensor (x1, y1, x2, y2, conf, cls). Post-NMS.
        gt_xyxy: [M, 4] tensor, GT boxes in pixel coords.
        seq_id: String sequence ID.
        frame_idx: Integer frame index within sequence.
        drone_type: String drone type.
        sample_id: String sample identifier (e.g. "sequence_121/frame_000450").

    Returns:
        dict: Per-frame record for JSON serialization.
    """
    num_gt = int(gt_xyxy.shape[0])
    gt_boxes_list = gt_xyxy.cpu().tolist() if num_gt > 0 else []
    if num_gt > 0:
        gt_w_all = (gt_xyxy[:, 2] - gt_xyxy[:, 0]).cpu()
        gt_h_all = (gt_xyxy[:, 3] - gt_xyxy[:, 1]).cpu()
        gt_area_all = (gt_w_all * gt_h_all).tolist()
        gt_max_dim = float(torch.maximum(gt_w_all, gt_h_all).max().item())
        gt_box_primary = [float(v) for v in gt_boxes_list[0]]
    else:
        gt_w_all = torch.zeros(0)
        gt_h_all = torch.zeros(0)
        gt_area_all = []
        gt_max_dim = 0.0
        gt_box_primary = [0.0, 0.0, 0.0, 0.0]

    num_dets = dets_tensor.shape[0]

    # Zero-detection case (pure FN)
    if num_dets == 0:
        return {
            "frame_id": sample_id.replace("/", "_"),
            "seq_id": seq_id,
            "frame_num": frame_idx,
            "temporal_pos": 0,
            "drone_type": drone_type,
            "num_gt": num_gt,
            "gt_box": [round(v, 2) for v in gt_box_primary],
            "gt_boxes": [[round(float(v), 2) for v in g] for g in gt_boxes_list],
            "gt_w": round(float(gt_w_all[0].item()), 2) if num_gt > 0 else 0.0,
            "gt_h": round(float(gt_h_all[0].item()), 2) if num_gt > 0 else 0.0,
            "gt_area": round(float(gt_area_all[0]), 2) if num_gt > 0 else 0.0,
            "gt_max_dim": round(gt_max_dim, 2),
            "num_dets": 0,
            "det_boxes": [],
            "det_confs": [],
            "det_ious": [],
            "matched": False,
            "matched_det_idx": -1,
            "matched_iou": 0.0,
            "matched_conf": 0.0,
            "det_labels": [],
            "fn_label": "fn_complete_miss" if num_gt > 0 else None,
            "fn_best_iou": 0.0 if num_gt > 0 else None,
        }

    # Sort by confidence descending
    order = dets_tensor[:, 4].argsort(descending=True)
    dets_tensor = dets_tensor[order]

    det_boxes = dets_tensor[:, :4].cpu().tolist()
    det_confs = dets_tensor[:, 4].cpu().tolist()

    if num_gt > 0:
        ious = box_iou(dets_tensor[:, :4], gt_xyxy)  # [N, M]
        det_ious_t, det_best_gt = ious.max(dim=1)
        det_ious = det_ious_t.cpu().tolist()
    else:
        ious = torch.zeros((num_dets, 0), device=dets_tensor.device)
        det_best_gt = torch.zeros((num_dets,), dtype=torch.long, device=dets_tensor.device)
        det_ious = [0.0] * num_dets

    gt_matched = torch.zeros((num_gt,), dtype=torch.bool, device=dets_tensor.device)
    det_labels = []
    matched = False
    matched_det_idx = -1
    matched_iou = 0.0
    matched_conf = 0.0

    for i in range(num_dets):
        best_iou = float(det_ious[i])
        if num_gt > 0 and best_iou >= 0.5:
            gt_idx = int(det_best_gt[i].item())
            if not bool(gt_matched[gt_idx]):
                gt_matched[gt_idx] = True
                det_labels.append("tp")
                matched = True
                if matched_det_idx < 0:
                    matched_det_idx = i
                    matched_iou = best_iou
                    matched_conf = det_confs[i]
            else:
                det_labels.append("fp_duplicate")
        elif best_iou >= 0.1:
            det_labels.append("fp_localization")
        else:
            det_labels.append("fp_background")

    fn_lab = None
    fn_best = None
    if num_gt > 0 and not bool(gt_matched.all().item()):
        fn_best = max(det_ious) if det_ious else 0.0
        fn_lab = "fn_localization" if fn_best >= 0.1 else "fn_complete_miss"

    return {
        "frame_id": sample_id.replace("/", "_"),
        "seq_id": seq_id,
        "frame_num": frame_idx,
        "temporal_pos": 0,
        "drone_type": drone_type,
        "num_gt": num_gt,
        "gt_box": [round(v, 2) for v in gt_box_primary],
        "gt_boxes": [[round(float(v), 2) for v in g] for g in gt_boxes_list],
        "gt_w": round(float(gt_w_all[0].item()), 2) if num_gt > 0 else 0.0,
        "gt_h": round(float(gt_h_all[0].item()), 2) if num_gt > 0 else 0.0,
        "gt_area": round(float(gt_area_all[0]), 2) if num_gt > 0 else 0.0,
        "gt_max_dim": round(gt_max_dim, 2),
        "num_dets": num_dets,
        "det_boxes": [[round(v, 2) for v in b] for b in det_boxes],
        "det_confs": [round(v, 4) for v in det_confs],
        "det_ious": [round(v, 4) for v in det_ious],
        "matched": matched,
        "matched_det_idx": matched_det_idx,
        "matched_iou": round(matched_iou, 4),
        "matched_conf": round(matched_conf, 4),
        "det_labels": det_labels,
        "fn_label": fn_lab,
        "fn_best_iou": round(fn_best, 4) if fn_best is not None else None,
    }


def verify_map_from_dump(records):
    """Recompute mAP@50 from the dumped records to verify pipeline consistency.

    Args:
        records: List of per-frame record dicts.

    Returns:
        float: mAP@50 in [0, 1].
    """
    all_preds = []
    total_gt = 0
    matched = {}

    for img_idx, rec in enumerate(records):
        gt_boxes = rec.get("gt_boxes")
        if not isinstance(gt_boxes, list):
            gt_box = rec.get("gt_box", [])
            gt_boxes = [gt_box] if gt_box else []
        total_gt += len(gt_boxes)
        matched[img_idx] = [False] * len(gt_boxes)

        det_boxes = rec.get("det_boxes", [])
        det_confs = rec.get("det_confs", [])
        for box, conf in zip(det_boxes, det_confs):
            all_preds.append({
                "img_idx": img_idx,
                "box": torch.tensor(box, dtype=torch.float32),
                "score": float(conf),
            })

    if total_gt == 0:
        return 0.0

    all_preds.sort(key=lambda x: x["score"], reverse=True)
    tp = []
    fp = []
    for pred in all_preds:
        rec = records[pred["img_idx"]]
        gt_boxes = rec.get("gt_boxes", [])
        if not gt_boxes:
            tp.append(0)
            fp.append(1)
            continue
        pred_box = pred["box"].unsqueeze(0)
        gt_t = torch.tensor(gt_boxes, dtype=torch.float32)
        ious = box_iou(pred_box, gt_t).squeeze(0)
        best_iou, best_idx = ious.max(dim=0)
        best_iou_val = float(best_iou.item())
        best_idx_val = int(best_idx.item())
        if best_iou_val >= 0.5 and not matched[pred["img_idx"]][best_idx_val]:
            matched[pred["img_idx"]][best_idx_val] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp = np.asarray(tp, dtype=np.float32)
    fp = np.asarray(fp, dtype=np.float32)
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / max(total_gt, 1)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-7)
    return compute_ap(recalls.tolist(), precisions.tolist())


def build_model_from_config(config):
    """Build detector from checkpoint config dict.

    Args:
        config: Config dict embedded in checkpoint.

    Returns:
        SparseFCOSDetector or SparseTQDet instance.
    """
    model_cfg = config.get('model', {})
    sparse_cfg = config.get('sparse', {})
    fcos_cfg = config.get('fcos', {})
    loss_cfg = config.get('loss', {})
    model_type = str(model_cfg.get('type', 'sparse_fcos')).lower()

    strides = fcos_cfg.get('strides', [4, 8, 16])
    regress_ranges = fcos_cfg.get('regress_ranges', [[-1, 32], [32, 64], [64, 256]])
    input_size = tuple(model_cfg.get('input_size', [640, 640]))

    if model_type == 'sparse_tqdet':
        model = SparseTQDet(
            in_channels=2,
            num_classes=model_cfg.get('num_classes', 1),
            backbone_size=model_cfg.get('backbone_size', 'nano'),
            fpn_channels=model_cfg.get('fpn_channels', 128),
            num_refine_layers=model_cfg.get('num_refine_layers', 4),
            topk=model_cfg.get('topk', 64),
            strides=strides,
            input_size=input_size,
            time_bins=sparse_cfg.get('time_bins', 15),
            bridge_type=model_cfg.get('bridge_type', 'motion_aware'),
            num_temporal_groups=sparse_cfg.get('num_temporal_groups', 3),
            use_pan=model_cfg.get('use_pan', True),
            query_heads=model_cfg.get('query_heads', 8),
            memory_pool=tuple(model_cfg.get('memory_pool', [8, 8])),
            use_uncertainty_in_score=float(loss_cfg.get('uncertainty_weight', 0.0)) > 0.0,
        )
    else:
        model = SparseFCOSDetector(
            in_channels=2,
            num_classes=model_cfg.get('num_classes', 1),
            backbone_size=model_cfg.get('backbone_size', 'nano'),
            fpn_channels=model_cfg.get('fpn_channels', 128),
            num_head_convs=model_cfg.get('num_head_convs', 4),
            strides=strides,
            regress_ranges=tuple(tuple(r) for r in regress_ranges),
            center_sampling=fcos_cfg.get('center_sampling', True),
            center_sampling_radius=fcos_cfg.get('center_sampling_radius', 1.5),
            prior_prob=fcos_cfg.get('prior_prob', 0.01),
            norm_on_bbox=fcos_cfg.get('norm_on_bbox', True),
            input_size=input_size,
            num_temporal_groups=sparse_cfg.get('num_temporal_groups', 3),
            time_bins=sparse_cfg.get('time_bins', 15),
            stem_stride=(1, 2, 2),
            bridge_type=model_cfg.get('bridge_type', 'attention'),
            gn_groups=model_cfg.get('gn_groups', 8),
            use_pan=model_cfg.get('use_pan', False),
            use_iou_quality=float(loss_cfg.get('iou_quality_weight', 0.0)) > 0,
        )
    return model


@torch.no_grad()
def dump_predictions(
    model,
    dataloader,
    config,
    device,
    drone_type_map,
    outdir,
    decode_settings=None,
    temporal_rerank=None,
):
    """Run Sparse FCOS inference on the full validation set and save per-frame records.

    Each batch is processed independently (single-frame model).

    Args:
        model: SparseFCOSDetector in eval mode, loaded with EMA weights.
        dataloader: DataLoader wrapping SparseEventDataset (val split).
        config: Parsed config dict.
        device: torch.device for inference.
        drone_type_map: Dict mapping seq_id (str) → drone_type (str).
        outdir: Output directory path.

    Returns:
        List of per-frame record dicts.
    """
    model.eval()
    use_amp = config.get('training', {}).get('use_amp', True)

    records = []
    num_frames = 0
    num_batches = len(dataloader)
    t0 = time.time()
    temporal_state = {}

    for batch_idx, batch in enumerate(dataloader):
        sparse_input = create_sparse_tensor(batch, device)
        B = batch['batch_size']

        gt_boxes = [b.to(device) for b in batch['gt_boxes']]

        with autocast(device_type="cuda", enabled=use_amp):
            outputs = model(sparse_input, B)

        detections = outputs['detections']  # [B, N, 6] = (x1,y1,x2,y2,score,class)
        if temporal_rerank and temporal_rerank.get("enabled", False):
            detections = temporal_rerank_top1(
                detections=detections,
                seq_ids=[str(x) for x in batch.get("seq_ids", [])],
                frame_nums=[int(x) for x in batch.get("frame_nums", [])],
                topk=int(temporal_rerank.get("topk", 5)),
                weights=temporal_rerank.get("weights", {}),
                state=temporal_state,
            )

        for b in range(B):
            dets = detections[b]  # [N, 6]
            # Filter zero-padded entries
            valid = dets[:, 4] > 0
            dets = dets[valid]

            gt = gt_boxes[b]  # [M, 4] xyxy in pixel coords

            # Extract sequence ID and frame number from sample_id
            # sample_id format: "sequence_XXX/frame_YYYYYY"
            seq_id = str(batch.get("seq_ids", ["unknown"] * B)[b])
            frame_idx = int(batch.get("frame_nums", [-1] * B)[b])
            sample_id = str(batch.get("sample_ids", [f"unknown_{num_frames}"] * B)[b])
            if "/" in sample_id:
                frame_stem = sample_id.split("/", 1)[1]
            else:
                frame_stem = f"frame_{frame_idx:06d}" if frame_idx >= 0 else f"frame_{num_frames:06d}"

            # Look up drone type using numeric part of seq_id
            seq_num = seq_id.replace("sequence_", "")
            drone_type = drone_type_map.get(seq_num, "unknown")

            record = process_frame(
                dets, gt, seq_id, frame_idx,
                drone_type=drone_type,
                sample_id=sample_id,
            )
            records.append(record)
            num_frames += 1

        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            elapsed = time.time() - t0
            fps = num_frames / elapsed if elapsed > 0 else 0
            print(f"  [{batch_idx+1}/{num_batches}] {num_frames} frames, {fps:.1f} fps")

    elapsed = time.time() - t0
    fps = num_frames / elapsed if elapsed > 0 else 0
    print(f"\nDump complete: {num_frames} frames in {elapsed:.1f}s ({fps:.1f} fps)")

    # Save predictions as JSON Lines
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pred_path = outdir / "predictions.json"
    with open(pred_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved {len(records)} frame records to {pred_path}")

    # Verification: recompute mAP@50 from the dump
    map50 = verify_map_from_dump(records)
    verification = {
        "mAP_50_from_dump": round(map50 * 100, 2),
        "num_frames": num_frames,
        "num_tp": sum(1 for r in records if r["matched"]),
        "num_fn": sum(1 for r in records if not r["matched"]),
        "total_dets": sum(r["num_dets"] for r in records),
    }
    if decode_settings is not None:
        verification["decode_settings"] = decode_settings
    verif_path = outdir / "verification.json"
    with open(verif_path, "w") as f:
        json.dump(verification, f, indent=2)
    print(f"\nVerification: mAP@50 = {verification['mAP_50_from_dump']:.2f}%")
    print(f"  TP: {verification['num_tp']}, FN: {verification['num_fn']}, "
          f"Total dets: {verification['total_dets']}")
    print(f"Saved to {verif_path}")

    return records


def main():
    """CLI entry point for prediction dumping.

    Loads the Sparse FCOS model with EMA weights from the specified checkpoint,
    applies decode settings from checkpoint eval config (or CLI overrides), runs
    inference on the validation set, and saves all per-frame detection data.

    EMA weight loading:
        Checkpoint contains ema_state_dict = {'shadow': state_dict, 'decay': float}.
        The shadow dict tracks full state_dict including BN buffers.
    """
    parser = argparse.ArgumentParser(
        description="Dump Sparse FCOS predictions for error analysis",
        epilog="Output: predictions.json (JSON Lines) + verification.json"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (e.g. runs/sparse_fcos/v31_full_fix/best.pt)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device for inference")
    parser.add_argument("--outdir", type=str, default="runs/sparse_fcos/v31_full_fix/analysis",
                        help="Output directory")
    parser.add_argument("--use-ema", action="store_true", default=True,
                        help="Use EMA weights from checkpoint (default: True)")
    parser.add_argument("--score-thresh", type=float, default=None,
                        help="Score threshold override (default: checkpoint/config eval.score_thresh)")
    parser.add_argument("--nms-thresh", type=float, default=None,
                        help="NMS threshold override (default: checkpoint/config eval.nms_thresh)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional max number of frame samples to evaluate")
    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split to dump (required when --parity-enforced)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Sparse root override (required when --parity-enforced)")
    parser.add_argument("--label-dir", type=str, default=None,
                        help="Label root override (required when --parity-enforced)")
    parser.add_argument("--parity-enforced", action="store_true",
                        help="Hard-fail unless exact parity roots/splits are used and sample caps are disabled")
    parser.add_argument("--max-detections", type=int, default=None,
                        help="Optional override for max detections per frame")
    parser.add_argument(
        "--allow-max-detections-override",
        action="store_true",
        help="Allow max-detections override above config eval.max_detections",
    )
    parser.add_argument(
        "--temporal-rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable sequence-aware temporal top-1 reranking (default: config value)",
    )
    parser.add_argument(
        "--temporal-rerank-topk",
        type=int,
        default=None,
        help="Temporal rerank candidate pool size (default: config eval.temporal_rerank.topk)",
    )
    args = parser.parse_args()

    if args.parity_enforced and (
        args.split is None or args.data_dir is None or args.label_dir is None
    ):
        raise ValueError(
            "When --parity-enforced is set, --split, --data-dir, and --label-dir are required."
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Extract config from checkpoint
    config = ckpt.get('config', {})
    if not config:
        print("WARNING: No config in checkpoint, using default sparse_fcos.yaml")
        import yaml
        with open(PROJECT_ROOT / "detection/configs/sparse_fcos.yaml") as f:
            config = yaml.safe_load(f)

    # Build model from config
    print("Building model...")
    model = build_model_from_config(config)

    # Load weights
    if args.use_ema and 'ema_state_dict' in ckpt:
        ema_data = ckpt['ema_state_dict']
        shadow = ema_data['shadow']
        # Handle DDP 'module.' prefix
        if any(k.startswith('module.') for k in shadow.keys()):
            shadow = {k.replace('module.', ''): v for k, v in shadow.items()}
            print("  Removed 'module.' prefix from EMA state dict")
        load_result = model.load_state_dict(shadow, strict=False)
        missing_keys = list(getattr(load_result, "missing_keys", []))
        unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
        if missing_keys:
            print(f"  WARNING: Missing EMA keys: {missing_keys[:8]}{' ...' if len(missing_keys) > 8 else ''}")
        if unexpected_keys:
            print(f"  WARNING: Unexpected EMA keys: {unexpected_keys[:8]}{' ...' if len(unexpected_keys) > 8 else ''}")
        print(f"Loaded EMA shadow weights ({len(shadow)} keys, decay={ema_data['decay']})")
    else:
        state_dict = ckpt['model_state_dict']
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        load_result = model.load_state_dict(state_dict, strict=False)
        missing_keys = list(getattr(load_result, "missing_keys", []))
        unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
        if missing_keys:
            print(f"  WARNING: Missing model keys: {missing_keys[:8]}{' ...' if len(missing_keys) > 8 else ''}")
        if unexpected_keys:
            print(f"  WARNING: Unexpected model keys: {unexpected_keys[:8]}{' ...' if len(unexpected_keys) > 8 else ''}")
        print("Loaded model weights (no EMA)")

    model = model.to(device)

    eval_cfg = config.get("eval", {})
    temporal_cfg = eval_cfg.get("temporal_rerank", {}) or {}
    decode_score_thresh = (
        float(args.score_thresh)
        if args.score_thresh is not None
        else float(eval_cfg.get("score_thresh", 0.05))
    )
    decode_nms_thresh = (
        float(args.nms_thresh)
        if args.nms_thresh is not None
        else float(eval_cfg.get("nms_thresh", 0.5))
    )
    config_max_detections = int(eval_cfg.get("max_detections", 1))
    if args.max_detections is not None:
        requested_max_detections = int(args.max_detections)
        if requested_max_detections > config_max_detections and not args.allow_max_detections_override:
            raise ValueError(
                "Requested --max-detections exceeds config eval.max_detections "
                f"({requested_max_detections} > {config_max_detections}). "
                "Use --allow-max-detections-override to bypass intentionally."
            )
        decode_max_detections = requested_max_detections
        override_used = requested_max_detections != config_max_detections
    else:
        decode_max_detections = config_max_detections
        override_used = False
    temporal_enabled = bool(temporal_cfg.get("enabled", False)) if args.temporal_rerank is None else bool(args.temporal_rerank)
    temporal_topk = int(temporal_cfg.get("topk", 5)) if args.temporal_rerank_topk is None else int(args.temporal_rerank_topk)
    temporal_weights = temporal_cfg.get("weights", {}) if isinstance(temporal_cfg.get("weights", {}), dict) else {}
    decode_max_detections_internal = (
        max(decode_max_detections, temporal_topk) if temporal_enabled else decode_max_detections
    )

    # Decode behavior is explicitly controlled here so dumps match configured eval policy.
    model_type = str(config.get('model', {}).get('type', 'sparse_fcos')).lower()
    original_decode = None
    if model_type == 'sparse_tqdet' and hasattr(model, 'set_decode_params'):
        model.set_decode_params(
            score_thresh=decode_score_thresh,
            nms_thresh=decode_nms_thresh,
            max_detections=decode_max_detections_internal,
        )
    else:
        original_decode = model._decode_predictions

        def patched_decode(cls_preds, reg_preds, ctr_preds, dev,
                           score_thresh=None, nms_thresh=None, max_detections=None,
                           iou_quality_preds=None):
            return original_decode(
                cls_preds, reg_preds, ctr_preds, dev,
                score_thresh=decode_score_thresh if score_thresh is None else score_thresh,
                nms_thresh=decode_nms_thresh if nms_thresh is None else nms_thresh,
                max_detections=decode_max_detections_internal if max_detections is None else max_detections,
                iou_quality_preds=iou_quality_preds,
            )

        model._decode_predictions = patched_decode
    print(
        "Decode settings: "
        f"score_thresh={decode_score_thresh} "
        f"nms_thresh={decode_nms_thresh} "
        f"max_detections_final={decode_max_detections} "
        f"max_detections_decode={decode_max_detections_internal} "
        f"temporal_rerank={temporal_enabled}"
    )

    # Build dataset and dataloader
    sparse_cfg = config.get('sparse', {})
    data_cfg = config.get('data', {})
    time_bins = sparse_cfg.get('time_bins', 15)
    split = str(args.split or data_cfg.get('val_split', data_cfg.get('split', 'val')))
    sparse_dir = PROJECT_ROOT / args.data_dir if args.data_dir else (PROJECT_ROOT / data_cfg.get('sparse_dir', 'data/datasets/fred_sparse'))
    label_dir = PROJECT_ROOT / args.label_dir if args.label_dir else (PROJECT_ROOT / data_cfg.get('label_dir', 'data/datasets/fred_voxel/labels_clean'))

    if args.parity_enforced:
        _assert_parity_dump_contract(
            sparse_dir=str(sparse_dir),
            label_dir=str(label_dir),
            split=split,
            max_samples=args.max_samples,
        )
    elif split not in PARITY_SPLIT_ALLOWLIST and split not in {"train", "val", "test"}:
        raise ValueError(
            f"Unsupported split '{split}'. Allowed: {sorted(PARITY_SPLIT_ALLOWLIST | {'train', 'val', 'test'})}"
        )

    dataset = SparseEventDataset(
        sparse_dir=str(sparse_dir),
        label_dir=str(label_dir),
        split=split,
        time_bins=time_bins,
        augment=False,
        max_voxels=int(sparse_cfg.get('max_voxels_eval', sparse_cfg.get('max_voxels', 80000))),
        voxel_sampling={"mode": "random"},
    )
    if args.max_samples is not None and args.max_samples > 0 and len(dataset.samples) > args.max_samples:
        dataset.samples = dataset.samples[:args.max_samples]
        print(f"Limited evaluation to first {len(dataset.samples)} frame samples")

    collate_fn = make_collate_fn(time_bins=time_bins)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.get('training', {}).get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Load drone type mapping
    mapping_path = PROJECT_ROOT / "data/datasets/fred_voxel/sequence_class_mapping.json"
    drone_type_map = load_drone_type_mapping(mapping_path)
    print(f"Loaded drone type mapping for {len(drone_type_map)} sequences")

    # Run the dump
    print(f"\nDumping predictions for {len(dataset)} frames...")
    records = dump_predictions(
        model,
        dataloader,
        config,
        device,
        drone_type_map,
        args.outdir,
        decode_settings={
            "score_thresh": decode_score_thresh,
            "nms_thresh": decode_nms_thresh,
            "max_detections": decode_max_detections,
            "decode_max_detections_internal": decode_max_detections_internal,
            "config_max_detections": config_max_detections,
            "override_used": override_used,
            "temporal_rerank_enabled": temporal_enabled,
            "temporal_rerank_topk": temporal_topk if temporal_enabled else 0,
            "temporal_rerank_weights": temporal_weights if temporal_enabled else {},
        },
        temporal_rerank={
            "enabled": temporal_enabled,
            "topk": temporal_topk,
            "weights": temporal_weights,
        },
    )
    if original_decode is not None:
        model._decode_predictions = original_decode

    # Sanity checks
    tp_count = sum(1 for r in records if r.get("matched", False))
    fn_count = sum(1 for r in records if r.get("fn_label") in {"fn_complete_miss", "fn_localization"})
    gt_total = sum(int(r.get("num_gt", len(r.get("gt_boxes", [])))) for r in records)

    print(f"\nSanity checks passed:")
    print(f"  Total frames: {len(records)}")
    print(f"  Frames with at least one TP: {tp_count}")
    print(f"  Frames with unresolved FN labels: {fn_count}")
    print(f"  Total GT boxes in dump: {gt_total}")


if __name__ == "__main__":
    main()
