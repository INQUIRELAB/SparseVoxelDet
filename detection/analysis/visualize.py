"""Sparse event frame visualization with GT and detection boxes.

Step 3 of the error analysis pipeline (dump → analyze → visualize).
Renders worst false positives, worst false negatives, and sample true positives.

Box color coding:
    - Lime green (solid): Ground truth box
    - Cyan blue (solid): True positive detection
    - Orange (dashed): Duplicate FP (IoU >= 0.5 but GT already matched)
    - Red (dash-dot): Localization FP (0.1 <= IoU < 0.5)
    - Red (dotted): Background FP (IoU < 0.1)

Usage:
    python -m detection.analysis.visualize \\
        --dump runs/analysis/predictions.json \\
        --sparse-dir data/datasets/fred_sparse/val \\
        --outdir runs/analysis
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

DET_STYLES = {
    "tp": ("#00bfff", "-", "TP"),
    "fp_duplicate": ("#ff6600", "--", "FP-dup"),
    "fp_localization": ("#ff0000", "-.", "FP-loc"),
    "fp_background": ("#ff0000", ":", "FP-bg"),
}


# ============================================================================
# Data Loading
# ============================================================================

def load_records(dump_path):
    """Load JSON Lines predictions file produced by dump_predictions.py.

    Args:
        dump_path: Path to predictions.json file.

    Returns:
        List of per-frame record dicts.
    """
    records = []
    with open(dump_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_sparse_event_frame(sparse_dir, frame_id, target_size=(640, 640)):
    """Load a sparse .npz file and render it as an RGB event visualization.

    The sparse data format contains:
        - coords: [N, 3] int (t, y, x) — voxel coordinates
        - feats: [N, 2] float (on_count, off_count) — event polarities

    Rendering pipeline:
        1. Accumulate ON/OFF counts per pixel (sum over temporal dimension)
        2. Normalize each polarity to [0, 1] independently
        3. Map to RGB: R=ON, B=OFF, G=min(ON,OFF) for overlap

    The frame_id format from dump is "sequence_XXX_frame_YYYYYY".
    We need to reconstruct the path as sparse_dir/sequence_XXX/frame_YYYYYY.npz.

    Args:
        sparse_dir: Root directory containing sequence subdirectories with .npz files.
        frame_id: Frame identifier string (e.g. "sequence_121_frame_000450").
        target_size: (H, W) spatial dimensions.

    Returns:
        np.ndarray: [H, W, 3] float32 RGB image in [0, 1].
    """
    H, W = target_size

    # Parse frame_id back to path: "sequence_XXX_frame_YYYYYY" -> sequence_XXX/frame_YYYYYY.npz
    parts = frame_id.split("_frame_")
    if len(parts) == 2:
        seq_dir = parts[0]
        frame_stem = f"frame_{parts[1]}"
        npz_path = Path(sparse_dir) / seq_dir / f"{frame_stem}.npz"
    else:
        # Fallback: try direct match
        npz_path = Path(sparse_dir) / f"{frame_id}.npz"

    if not npz_path.exists():
        return np.zeros((H, W, 3), dtype=np.float32)

    data = np.load(npz_path)

    if 'coords' in data:
        coords = data['coords']  # [N, 3] = (t, y, x)
        feats = data['feats']    # [N, 2] = (on, off)
    elif 'spikes' in data:
        # Dense format fallback
        spikes = data['spikes']  # [T, 2, H, W]
        on_events = spikes[:, 0, :, :].sum(axis=0).astype(np.float32)
        off_events = spikes[:, 1, :, :].sum(axis=0).astype(np.float32)
        on_max = on_events.max() if on_events.max() > 0 else 1
        off_max = off_events.max() if off_events.max() > 0 else 1
        image = np.zeros((H, W, 3), dtype=np.float32)
        image[:, :, 0] = on_events / on_max
        image[:, :, 2] = off_events / off_max
        image[:, :, 1] = np.minimum(image[:, :, 0], image[:, :, 2])
        return image
    else:
        return np.zeros((H, W, 3), dtype=np.float32)

    # Accumulate ON/OFF per pixel (sum over time)
    on_image = np.zeros((H, W), dtype=np.float32)
    off_image = np.zeros((H, W), dtype=np.float32)

    if len(coords) > 0:
        y = coords[:, 1].astype(np.int32)
        x = coords[:, 2].astype(np.int32)

        # Clip to valid range
        y = np.clip(y, 0, H - 1)
        x = np.clip(x, 0, W - 1)

        on_vals = feats[:, 0].astype(np.float32)
        off_vals = feats[:, 1].astype(np.float32)

        np.add.at(on_image, (y, x), on_vals)
        np.add.at(off_image, (y, x), off_vals)

    # Normalize
    on_max = on_image.max() if on_image.max() > 0 else 1
    off_max = off_image.max() if off_image.max() > 0 else 1
    on_norm = on_image / on_max
    off_norm = off_image / off_max

    # Map to RGB
    image = np.zeros((H, W, 3), dtype=np.float32)
    image[:, :, 0] = on_norm   # Red = ON
    image[:, :, 2] = off_norm  # Blue = OFF
    image[:, :, 1] = np.minimum(on_norm, off_norm)  # Green = overlap

    return image


# ============================================================================
# Drawing
# ============================================================================

def draw_frame(image, record, title_extra="", show_all_dets=True, max_labels=8):
    """Draw an event frame with annotated GT and detection bounding boxes.

    Args:
        image: [H, W, 3] float32 RGB array.
        record: Per-frame record dict from the prediction dump.
        title_extra: Additional text for the figure title.
        show_all_dets: If True, draw all detections.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, origin="upper")

    # Ground truth box (lime green)
    gt = record["gt_box"]
    gt_w = gt[2] - gt[0]
    gt_h = gt[3] - gt[1]
    gt_rect = patches.Rectangle(
        (gt[0], gt[1]), gt_w, gt_h,
        linewidth=2, edgecolor="lime", facecolor="none", linestyle="-",
    )
    ax.add_patch(gt_rect)
    gt_text_y = max(gt[1] - 4, 6)
    ax.text(
        gt[0],
        gt_text_y,
        f"GT {record['gt_max_dim']:.0f}px",
        color="lime",
        fontsize=8,
        fontweight="bold",
        clip_on=True,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6),
    )

    # Detection boxes (color-coded by error type)
    summary_lines = []
    if show_all_dets:
        for i in range(record["num_dets"]):
            box = record["det_boxes"][i]
            conf = record["det_confs"][i]
            iou = record["det_ious"][i]
            label = record["det_labels"][i]

            color, ls, short_label = DET_STYLES.get(label, ("#ff0000", ":", "FP"))

            bw = box[2] - box[0]
            bh = box[3] - box[1]
            rect = patches.Rectangle(
                (box[0], box[1]), bw, bh,
                linewidth=1.5, edgecolor=color, facecolor="none", linestyle=ls,
            )
            ax.add_patch(rect)

            if i < max_labels:
                summary_lines.append(f"{i:02d} {short_label} c={conf:.2f} iou={iou:.2f}")

    if show_all_dets and summary_lines:
        hidden = max(record["num_dets"] - max_labels, 0)
        if hidden > 0:
            summary_lines.append(f"... +{hidden} more")
        summary_text = "\n".join(summary_lines)
        ax.text(
            0.01,
            0.99,
            summary_text,
            transform=ax.transAxes,
            color="white",
            fontsize=7,
            va="top",
            ha="left",
            clip_on=True,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.55),
        )

    # Title
    status = "TP" if record["matched"] else "FN"
    title = f"{record['frame_id']} | {record['drone_type']} | {status}"
    if record["matched"]:
        title += f" | IoU={record['matched_iou']:.3f} conf={record['matched_conf']:.3f}"
    if title_extra:
        title += f" | {title_extra}"
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 640)
    ax.set_ylim(640, 0)
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig


# ============================================================================
# Rendering Functions
# ============================================================================

def render_worst_fp(records, sparse_dir, outdir, num_fp=50, max_labels=8):
    """Render top-N highest-confidence false positive detections.

    Args:
        records: List of per-frame record dicts.
        sparse_dir: Path to sparse .npz files directory.
        outdir: Base output directory.
        num_fp: Number of worst FP images to render.
    """
    fp_dir = Path(outdir) / "worst_fp"
    fp_dir.mkdir(parents=True, exist_ok=True)

    fp_entries = []
    for r in records:
        for i, label in enumerate(r["det_labels"]):
            if label != "tp":
                fp_entries.append((r["det_confs"][i], r, i, label))

    fp_entries.sort(key=lambda x: -x[0])

    rendered = 0
    for rank, (conf, record, det_idx, label) in enumerate(fp_entries[:num_fp]):
        image = load_sparse_event_frame(sparse_dir, record["frame_id"])
        fig = draw_frame(image, record, title_extra=f"rank={rank} {label}", max_labels=max_labels)
        fname = f"{rank:03d}_{conf:.3f}_{record['frame_id']}.png"
        fig.savefig(fp_dir / fname, dpi=120)
        plt.close(fig)
        rendered += 1

    print(f"Rendered {rendered} worst FP images to {fp_dir}")


def render_worst_fn(records, sparse_dir, outdir, num_fn=50, max_labels=8):
    """Render top-N missed detections sorted by GT size descending.

    Args:
        records: List of per-frame record dicts.
        sparse_dir: Path to sparse .npz files directory.
        outdir: Base output directory.
        num_fn: Number of worst FN images to render.
    """
    fn_dir = Path(outdir) / "worst_fn"
    fn_dir.mkdir(parents=True, exist_ok=True)

    fn_records = [r for r in records if not r["matched"]]
    fn_records.sort(key=lambda r: -r["gt_max_dim"])

    rendered = 0
    for rank, record in enumerate(fn_records[:num_fn]):
        image = load_sparse_event_frame(sparse_dir, record["frame_id"])
        extra = f"rank={rank} {record['fn_label']}"
        if record["fn_best_iou"] is not None:
            extra += f" best_iou={record['fn_best_iou']:.3f}"
        fig = draw_frame(image, record, title_extra=extra, max_labels=max_labels)
        fname = f"{rank:03d}_{record['gt_max_dim']:.0f}px_{record['frame_id']}.png"
        fig.savefig(fn_dir / fname, dpi=120)
        plt.close(fig)
        rendered += 1

    print(f"Rendered {rendered} worst FN images to {fn_dir}")


def render_sample_tp(records, sparse_dir, outdir, num_tp=30, max_labels=8):
    """Render a random sample of true positive frames.

    Args:
        records: List of per-frame record dicts.
        sparse_dir: Path to sparse .npz files directory.
        outdir: Base output directory.
        num_tp: Number of TP samples to render.
    """
    tp_dir = Path(outdir) / "sample_tp"
    tp_dir.mkdir(parents=True, exist_ok=True)

    tp_records = [r for r in records if r["matched"]]
    if len(tp_records) > num_tp:
        tp_records = random.sample(tp_records, num_tp)

    rendered = 0
    for i, record in enumerate(tp_records):
        image = load_sparse_event_frame(sparse_dir, record["frame_id"])
        fig = draw_frame(image, record, show_all_dets=True, max_labels=max_labels)
        fname = f"{i:03d}_iou{record['matched_iou']:.2f}_conf{record['matched_conf']:.2f}_{record['frame_id']}.png"
        fig.savefig(tp_dir / fname, dpi=120)
        plt.close(fig)
        rendered += 1

    print(f"Rendered {rendered} sample TP images to {tp_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    """CLI entry point for prediction visualization.

    Loads the prediction dump, renders three sets of images:
        1. worst_fp/ — highest-confidence false positives
        2. worst_fn/ — largest missed ground-truth boxes
        3. sample_tp/ — random true positive samples

    All rendering is CPU-only (no GPU needed).
    """
    parser = argparse.ArgumentParser(
        description="Visualize Sparse FCOS predictions with event frame rendering",
        epilog="Output: worst_fp/, worst_fn/, sample_tp/ subdirectories"
    )
    parser.add_argument("--dump", type=str, required=True,
                        help="Path to predictions.json")
    parser.add_argument("--sparse-dir", type=str, required=True,
                        help="Path to sparse .npz directory (e.g. data/datasets/fred_sparse/val)")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--num-fp", type=int, default=50,
                        help="Number of worst false positives to render")
    parser.add_argument("--num-fn", type=int, default=50,
                        help="Number of worst false negatives to render")
    parser.add_argument("--num-tp", type=int, default=30,
                        help="Number of sample true positives to render")
    parser.add_argument("--max-labels", type=int, default=8,
                        help="Maximum number of per-frame detection labels shown in legend block")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for TP sampling")
    args = parser.parse_args()

    random.seed(args.seed)

    records = load_records(args.dump)
    print(f"Loaded {len(records)} frame records")

    tp_count = sum(1 for r in records if r["matched"])
    fn_count = len(records) - tp_count
    print(f"  TP: {tp_count}, FN: {fn_count}")

    print("\nRendering worst false positives...")
    render_worst_fp(records, args.sparse_dir, args.outdir, args.num_fp, max_labels=args.max_labels)

    print("\nRendering worst false negatives...")
    render_worst_fn(records, args.sparse_dir, args.outdir, args.num_fn, max_labels=args.max_labels)

    print("\nRendering sample true positives...")
    render_sample_tp(records, args.sparse_dir, args.outdir, args.num_tp, max_labels=args.max_labels)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
