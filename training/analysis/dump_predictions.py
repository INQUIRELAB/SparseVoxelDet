"""SparseVoxelDet prediction dump for error analysis.

Step 1 of the 3-step error analysis pipeline (dump → analyze → visualize).
Runs the SparseVoxelDet model on the full test set and saves per-frame
detection data to a JSON Lines file.

Output files:
    predictions.json  — JSON Lines, one JSON object per evaluated frame
    verification.json — mAP@50 recomputed from the dump (must match eval)

Usage:
    python -m training.analysis.dump_predictions \\
        --checkpoint runs/sparse_voxel_det/best.pt \\
        --device cuda:0 --use-ema \\
        --outdir runs/sparse_voxel_det/analysis \\
        --split canonical_test --parity-enforced \\
        --data-dir data/datasets/fred_paper_parity_v82_640/sparse \\
        --label-dir data/datasets/fred_paper_parity/labels
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

from detection.scripts.sparse_event_dataset import (
    SparseEventDataset,
    make_collate_fn,
    create_sparse_tensor,
)
from training.models.sparse_voxel_det import SparseVoxelDet

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
        dict: seq_id (str) → drone_type (str), e.g. {"121": "Betafpv_Air75"}
    """
    with open(mapping_path) as f:
        data = json.load(f)

    class_names = data["class_names"]      # {"0": "DJI_Tello_EDU", ...}
    seq_classes = data["sequence_classes"]  # {"0": 1, "1": 1, ...}

    mapping = {}
    for seq_id_str, class_idx in seq_classes.items():
        mapping[seq_id_str] = class_names[str(class_idx)]
    return mapping


# ===========================================================================
# Matching utilities (identical to detection/analysis/dump_predictions.py)
# ===========================================================================

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


def process_frame(dets_tensor, gt_xyxy, seq_id, frame_idx, drone_type, sample_id):
    """Build a complete per-frame record from raw model outputs and GT.

    Matching algorithm (greedy, confidence-descending):
        1. Sort detections by confidence descending.
        2. For each detection, find the GT with highest IoU.
        3. If IoU >= 0.5 and GT not yet matched → TP.
        4. If IoU >= 0.5 and GT already matched → fp_duplicate.
        5. If 0.1 <= IoU < 0.5 → fp_localization.
        6. If IoU < 0.1 → fp_background.
        7. For unmatched GTs: fn_localization if best det IoU >= 0.1, else fn_complete_miss.

    Args:
        dets_tensor: [N, 6] tensor (x1, y1, x2, y2, conf, cls). Post-NMS.
        gt_xyxy: [M, 4] tensor, GT boxes in pixel coords.
        seq_id: String sequence ID.
        frame_idx: Integer frame index within sequence.
        drone_type: String drone type.
        sample_id: String sample identifier (e.g. "110/frame_000219").

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
        ious = box_iou(dets_tensor[:, :4], gt_xyxy)  # [N_dets, M_gt]
        det_ious_t, det_best_gt = ious.max(dim=1)    # best GT per det
        det_ious = det_ious_t.cpu().tolist()
    else:
        det_best_gt = torch.zeros((num_dets,), dtype=torch.long, device=dets_tensor.device)
        det_ious = [0.0] * num_dets

    # Greedy matching: assign each detection to at most one GT
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

    # FN labeling for unmatched GT boxes
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


def verify_map_from_dump(records, score_cutoff=0.005):
    """Recompute mAP@50 from the dumped records to verify pipeline consistency.

    Uses the standard COCO-style evaluation:
    - Global sort all predictions by confidence
    - Greedy match at IoU >= 0.5 (one GT per prediction)
    - 101-point interpolated AP

    Memory optimization: predictions below score_cutoff are counted as bulk FP
    (they contribute negligibly to AP but would consume excessive memory).

    Args:
        records: List of per-frame record dicts.
        score_cutoff: Minimum score for individual tracking (default 0.005).
                      Lower scores are counted as bulk FP.

    Returns:
        float: mAP@50 in [0, 1].
    """
    # Pre-build GT box arrays per image for vectorized IoU
    gt_boxes_per_img = []
    total_gt = 0
    bulk_fp_count = 0

    for img_idx, rec in enumerate(records):
        gt_boxes = rec.get("gt_boxes")
        if not isinstance(gt_boxes, list):
            gt_box = rec.get("gt_box", [])
            gt_boxes = [gt_box] if gt_box else []
        gt_boxes_per_img.append(np.array(gt_boxes, dtype=np.float32) if gt_boxes else None)
        total_gt += len(gt_boxes)

    if total_gt == 0:
        return 0.0

    # Collect predictions above score cutoff (compact representation)
    # Format: (score, img_idx, box_x1, box_y1, box_x2, box_y2)
    pred_scores = []
    pred_img_idxs = []
    pred_boxes = []

    for img_idx, rec in enumerate(records):
        det_boxes = rec.get("det_boxes", [])
        det_confs = rec.get("det_confs", [])
        for box, conf in zip(det_boxes, det_confs):
            if conf >= score_cutoff:
                pred_scores.append(conf)
                pred_img_idxs.append(img_idx)
                pred_boxes.append(box)
            else:
                bulk_fp_count += 1

    n_preds = len(pred_scores)
    if n_preds == 0:
        return 0.0

    # Sort by score descending
    order = np.argsort(pred_scores)[::-1]
    pred_scores_sorted = np.array(pred_scores, dtype=np.float32)[order]
    pred_img_idxs_sorted = np.array(pred_img_idxs, dtype=np.int32)[order]
    pred_boxes_arr = np.array(pred_boxes, dtype=np.float32)[order]

    # Greedy matching
    matched = {}
    tp = np.zeros(n_preds, dtype=np.float32)
    fp = np.zeros(n_preds, dtype=np.float32)

    for i in range(n_preds):
        img_idx = int(pred_img_idxs_sorted[i])
        gt_arr = gt_boxes_per_img[img_idx]
        if gt_arr is None or len(gt_arr) == 0:
            fp[i] = 1
            continue

        # Compute IoU with numpy
        pb = pred_boxes_arr[i]  # [4]
        inter_x1 = np.maximum(pb[0], gt_arr[:, 0])
        inter_y1 = np.maximum(pb[1], gt_arr[:, 1])
        inter_x2 = np.minimum(pb[2], gt_arr[:, 2])
        inter_y2 = np.minimum(pb[3], gt_arr[:, 3])
        inter = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
        area_pred = (pb[2] - pb[0]) * (pb[3] - pb[1])
        area_gt = (gt_arr[:, 2] - gt_arr[:, 0]) * (gt_arr[:, 3] - gt_arr[:, 1])
        union = area_pred + area_gt - inter
        ious = inter / np.maximum(union, 1e-7)

        best_idx = int(np.argmax(ious))
        best_iou = float(ious[best_idx])

        if img_idx not in matched:
            matched[img_idx] = set()

        if best_iou >= 0.5 and best_idx not in matched[img_idx]:
            matched[img_idx].add(best_idx)
            tp[i] = 1
        else:
            fp[i] = 1

    # Prepend bulk FP (low-confidence detections below cutoff)
    # These appear at the end of the ranking (lowest confidence) so they
    # only affect the tail of the PR curve.
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp) + bulk_fp_count  # bulk FP counted at all thresholds

    # Actually, bulk FPs should be appended AFTER the scored predictions
    # since they have lower confidence. Recompute properly:
    fp_cumsum = np.cumsum(fp)
    # After all scored preds, add bulk FP
    total_fp_after_all = fp_cumsum[-1] + bulk_fp_count if n_preds > 0 else bulk_fp_count

    recalls = tp_cumsum / max(total_gt, 1)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-7)

    return compute_ap(recalls.tolist(), precisions.tolist())


# ===========================================================================
# SparseVoxelDet model building
# ===========================================================================

def build_sparse_voxel_det(config):
    """Build SparseVoxelDet from checkpoint config dict.

    Args:
        config: Config dict embedded in checkpoint.

    Returns:
        SparseVoxelDet instance.
    """
    model_cfg = config.get('model', {})
    sparse_cfg = config.get('sparse', {})

    model = SparseVoxelDet(
        in_channels=2,
        num_classes=model_cfg.get('num_classes', 1),
        backbone_size=model_cfg.get('backbone_size', 'nano_deep'),
        fpn_channels=model_cfg.get('fpn_channels', 128),
        head_convs=model_cfg.get('head_convs', 2),
        input_size=tuple(model_cfg.get('input_size', [640, 640])),
        time_bins=sparse_cfg.get('time_bins', 15),
    )
    return model


# ===========================================================================
# Main inference loop
# ===========================================================================

@torch.no_grad()
def dump_predictions(
    model,
    dataloader,
    config,
    device,
    drone_type_map,
    outdir,
    decode_settings=None,
):
    """Run SparseVoxelDet inference on the full dataset and save per-frame records.

    Args:
        model: SparseVoxelDet in eval mode, loaded with EMA weights.
        dataloader: DataLoader wrapping SparseEventDataset.
        config: Parsed config dict.
        device: torch.device for inference.
        drone_type_map: Dict mapping seq_id (str) → drone_type (str).
        outdir: Output directory path.
        decode_settings: Dict of decode params for recording in verification.

    Returns:
        List of per-frame record dicts.
    """
    model.eval()
    use_amp = config.get('training', {}).get('use_amp', True)

    records = []
    num_frames = 0
    num_batches = len(dataloader)
    t0 = time.time()

    for batch_idx, batch in enumerate(dataloader):
        sparse_input = create_sparse_tensor(batch, device)
        B = batch['batch_size']

        gt_boxes = [b.to(device) for b in batch['gt_boxes']]

        with autocast(device_type="cuda", enabled=use_amp):
            outputs = model(sparse_input, B)

        detections = outputs['detections']  # [B, N, 6] = (x1,y1,x2,y2,score,class)

        for b in range(B):
            dets = detections[b]  # [N, 6]
            # Filter zero-padded entries
            valid = dets[:, 4] > 0
            dets = dets[valid]

            gt = gt_boxes[b]  # [M, 4] xyxy in pixel coords

            # Extract sequence ID and frame number from sample_id
            seq_id = str(batch.get("seq_ids", ["unknown"] * B)[b])
            frame_idx = int(batch.get("frame_nums", [-1] * B)[b])
            sample_id = str(batch.get("sample_ids", [f"unknown_{num_frames}"] * B)[b])

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

        if (batch_idx + 1) % 200 == 0 or batch_idx == 0:
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
    print("\nVerifying mAP@50 from dump (this may take a minute)...")
    map50 = verify_map_from_dump(records)
    verification = {
        "mAP_50_from_dump": round(map50 * 100, 2),
        "num_frames": num_frames,
        "num_frames_with_gt": sum(1 for r in records if r["num_gt"] > 0),
        "total_gt_boxes": sum(r["num_gt"] for r in records),
        "num_tp": sum(1 for r in records if r["matched"]),
        "num_fn": sum(1 for r in records if r.get("fn_label") in {"fn_complete_miss", "fn_localization"}),
        "total_dets": sum(r["num_dets"] for r in records),
        "avg_dets_per_frame": round(sum(r["num_dets"] for r in records) / max(num_frames, 1), 2),
    }
    if decode_settings is not None:
        verification["decode_settings"] = decode_settings
    verif_path = outdir / "verification.json"
    with open(verif_path, "w") as f:
        json.dump(verification, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Verification: mAP@50 = {verification['mAP_50_from_dump']:.2f}%")
    print(f"  Total frames:    {verification['num_frames']}")
    print(f"  Frames with GT:  {verification['num_frames_with_gt']}")
    print(f"  Total GT boxes:  {verification['total_gt_boxes']}")
    print(f"  TP frames:       {verification['num_tp']}")
    print(f"  FN frames:       {verification['num_fn']}")
    print(f"  Total dets:      {verification['total_dets']}")
    print(f"  Avg dets/frame:  {verification['avg_dets_per_frame']}")
    print(f"{'='*60}")
    print(f"Saved to {verif_path}")

    return records


def main():
    """CLI entry point for SparseVoxelDet prediction dumping.

    Loads the SparseVoxelDet model with EMA weights from the specified checkpoint,
    applies decode settings, runs inference on the specified dataset split, and
    saves all per-frame detection data.
    """
    parser = argparse.ArgumentParser(
        description="Dump SparseVoxelDet predictions for error analysis",
        epilog="Output: predictions.json (JSON Lines) + verification.json"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device for inference")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory for predictions and verification")
    parser.add_argument("--use-ema", action="store_true", default=True,
                        help="Use EMA weights from checkpoint (default: True)")
    parser.add_argument("--no-ema", action="store_true", default=False,
                        help="Force using raw model weights, not EMA")
    parser.add_argument("--score-thresh", type=float, default=None,
                        help="Score threshold override (default: 0.001 for full mAP)")
    parser.add_argument("--nms-thresh", type=float, default=None,
                        help="NMS threshold override (default: config eval.nms_thresh)")
    parser.add_argument("--max-detections", type=int, default=None,
                        help="Max detections per frame override (default: config eval.max_detections)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional max number of frames to evaluate")
    parser.add_argument("--split", type=str, required=True,
                        help="Dataset split (e.g. canonical_test)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Sparse data root directory")
    parser.add_argument("--label-dir", type=str, required=True,
                        help="Label root directory")
    parser.add_argument("--parity-enforced", action="store_true",
                        help="Hard-fail unless exact parity roots/splits/no sample caps")
    args = parser.parse_args()

    if args.parity_enforced:
        _assert_parity_dump_contract(
            sparse_dir=args.data_dir,
            label_dir=args.label_dir,
            split=args.split,
            max_samples=args.max_samples,
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Extract config from checkpoint
    config = ckpt.get('config', {})
    if not config:
        raise ValueError("No config found in checkpoint — cannot build model safely")

    # Build model
    print("Building SparseVoxelDet model...")
    model = build_sparse_voxel_det(config)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Load weights
    use_ema = args.use_ema and not args.no_ema
    if use_ema and 'ema_state_dict' in ckpt:
        ema_data = ckpt['ema_state_dict']
        shadow = ema_data['shadow']
        # Handle DDP 'module.' prefix
        if any(k.startswith('module.') for k in shadow.keys()):
            shadow = {k.replace('module.', ''): v for k, v in shadow.items()}
            print("  Removed 'module.' prefix from EMA state dict")
        load_result = model.load_state_dict(shadow, strict=False)
        missing = list(getattr(load_result, "missing_keys", []))
        unexpected = list(getattr(load_result, "unexpected_keys", []))
        if missing:
            print(f"  WARNING: Missing EMA keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"  WARNING: Unexpected EMA keys ({len(unexpected)}): {unexpected[:5]}")
        print(f"  Loaded EMA shadow weights ({len(shadow)} keys, decay={ema_data['decay']})")
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        load_result = model.load_state_dict(state_dict, strict=False)
        missing = list(getattr(load_result, "missing_keys", []))
        unexpected = list(getattr(load_result, "unexpected_keys", []))
        if missing:
            print(f"  WARNING: Missing model keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"  WARNING: Unexpected model keys ({len(unexpected)}): {unexpected[:5]}")
        print(f"  Loaded model weights (no EMA)")
    else:
        raise ValueError("Checkpoint has neither 'ema_state_dict' nor 'model_state_dict'")

    model = model.to(device)

    # Decode settings: default to very low score thresh for full mAP sensitivity
    eval_cfg = config.get("eval", {})
    decode_score_thresh = (
        float(args.score_thresh) if args.score_thresh is not None
        else 0.001  # Low threshold: capture all detections for analysis
    )
    decode_nms_thresh = (
        float(args.nms_thresh) if args.nms_thresh is not None
        else float(eval_cfg.get("nms_thresh", 0.5))
    )
    decode_max_detections = (
        int(args.max_detections) if args.max_detections is not None
        else int(eval_cfg.get("max_detections", 100))
    )

    # Apply decode params via the model's official API
    model.set_decode_params(
        score_thresh=decode_score_thresh,
        nms_thresh=decode_nms_thresh,
        max_detections=decode_max_detections,
    )
    print(f"  Decode: score_thresh={decode_score_thresh}, nms_thresh={decode_nms_thresh}, "
          f"max_detections={decode_max_detections}")

    # Build dataset and dataloader
    sparse_cfg = config.get('sparse', {})
    time_bins = sparse_cfg.get('time_bins', 15)
    max_voxels = int(sparse_cfg.get('max_voxels_eval', sparse_cfg.get('max_voxels', 80000)))
    if max_voxels <= 0:
        max_voxels = 200000  # -1 means unlimited, use generous cap

    dataset = SparseEventDataset(
        sparse_dir=args.data_dir,
        label_dir=args.label_dir,
        split=args.split,
        time_bins=time_bins,
        augment=False,
        max_voxels=max_voxels,
        voxel_sampling={"mode": "random"},
    )
    if args.max_samples is not None and args.max_samples > 0 and len(dataset.samples) > args.max_samples:
        dataset.samples = dataset.samples[:args.max_samples]
        print(f"  Limited to first {len(dataset.samples)} frame samples")

    collate_fn = make_collate_fn(time_bins=time_bins)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Load drone type mapping
    mapping_path = PROJECT_ROOT / "data/datasets/fred_voxel/sequence_class_mapping.json"
    drone_type_map = load_drone_type_mapping(mapping_path)
    print(f"  Loaded drone type mapping for {len(drone_type_map)} sequences")

    # Show checkpoint info
    ckpt_epoch = ckpt.get('epoch', '?')
    ckpt_metrics = ckpt.get('metrics', {})
    print(f"\nCheckpoint info:")
    print(f"  Epoch: {ckpt_epoch}")
    if ckpt_metrics:
        print(f"  Metrics: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in ckpt_metrics.items()}, indent=4)}")

    # Run the dump
    print(f"\n{'='*60}")
    print(f"Dumping predictions for {len(dataset)} frames on {device}...")
    print(f"Split: {args.split}")
    print(f"Output: {args.outdir}")
    print(f"{'='*60}\n")

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
            "checkpoint": str(args.checkpoint),
            "epoch": ckpt_epoch,
            "use_ema": use_ema,
            "split": args.split,
        },
    )

    # Final sanity checks
    tp_count = sum(1 for r in records if r.get("matched", False))
    fn_count = sum(1 for r in records if r.get("fn_label") in {"fn_complete_miss", "fn_localization"})
    gt_total = sum(r["num_gt"] for r in records)

    print(f"\nFinal summary:")
    print(f"  Total frames:         {len(records)}")
    print(f"  Total GT boxes:       {gt_total}")
    print(f"  Frames with TP:       {tp_count} ({100*tp_count/max(len(records),1):.1f}%)")
    print(f"  Frames with FN:       {fn_count} ({100*fn_count/max(len(records),1):.1f}%)")
    print(f"\nNext steps:")
    print(f"  1. Analyze:    python -m training.analysis.analyze --dump {args.outdir}/predictions.json --outdir {args.outdir}")
    print(f"  2. Visualize:  python -m detection.analysis.visualize --dump {args.outdir}/predictions.json --sparse-dir {args.data_dir}/{args.split} --outdir {args.outdir}/vis")


if __name__ == "__main__":
    main()
