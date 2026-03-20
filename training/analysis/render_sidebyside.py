"""Side-by-side Event + RGB visualization for error analysis.

Renders pairs of (sparse event frame, RGB frame) with GT and detection boxes.
Designed for investigating catastrophic sequences and complete miss frames.

Modes:
    1. sequence_overview: Every Nth frame of a full sequence, side-by-side event+RGB
    2. fn_gallery: All FN frames from a given sequence, event+RGB side-by-side
    3. complete_miss_gallery: All complete-miss frames across dataset

Usage:
    # Render sequence overview (every 20th frame)
    python -m training.analysis.render_sidebyside \\
        --dump runs/sparse_voxel_det/analysis/predictions.json \\
        --sparse-dir data/datasets/fred_paper_parity_v82_640/sparse/canonical_test \\
        --rgb-root data/processed/FRED/test \\
        --outdir runs/sparse_voxel_det/analysis/sidebyside \\
        --mode sequence_overview --seq-id 111 --step 20

    # Render all FN frames from a sequence
    python -m training.analysis.render_sidebyside \\
        --dump runs/sparse_voxel_det/analysis/predictions.json \\
        --sparse-dir data/datasets/fred_paper_parity_v82_640/sparse/canonical_test \\
        --rgb-root data/processed/FRED/test \\
        --outdir runs/sparse_voxel_det/analysis/sidebyside \\
        --mode fn_gallery --seq-id 111

    # Render all complete miss frames (across all sequences)
    python -m training.analysis.render_sidebyside \\
        --dump runs/sparse_voxel_det/analysis/predictions.json \\
        --sparse-dir data/datasets/fred_paper_parity_v82_640/sparse/canonical_test \\
        --rgb-root data/processed/FRED/test \\
        --outdir runs/sparse_voxel_det/analysis/sidebyside \\
        --mode complete_miss_gallery
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


# ============================================================================
# Style constants
# ============================================================================

DET_STYLES = {
    "tp":               ("#00bfff", "-",  2.0, "TP"),
    "fp_duplicate":     ("#ff6600", "--", 1.5, "FP-dup"),
    "fp_localization":  ("#ff0000", "-.", 1.5, "FP-loc"),
    "fp_background":    ("#ff0000", ":",  1.0, "FP-bg"),
}
GT_COLOR = "#00ff00"


# ============================================================================
# Data loading
# ============================================================================

def load_records(dump_path, seq_filter=None, fn_only=False, complete_miss_only=False):
    """Load JSONL records, optionally filtering."""
    records = []
    with open(dump_path) as f:
        for line in f:
            rec = json.loads(line)
            if seq_filter and rec["seq_id"] != str(seq_filter):
                continue
            if fn_only and rec["matched"]:
                continue
            if complete_miss_only and (rec["matched"] or rec.get("fn_label") != "fn_complete_miss"):
                continue
            records.append(rec)
    return records


def load_sparse_event_frame(sparse_dir, seq_id, frame_idx, target_size=(640, 640)):
    """Load sparse .npz and render as RGB event image."""
    H, W = target_size
    npz_path = Path(sparse_dir) / str(seq_id) / f"frame_{frame_idx:06d}.npz"

    if not npz_path.exists():
        return np.zeros((H, W, 3), dtype=np.float32)

    data = np.load(npz_path)
    if "coords" not in data:
        return np.zeros((H, W, 3), dtype=np.float32)

    coords = data["coords"]  # [N, 3] = (t, y, x)
    feats = data["feats"]    # [N, 2] = (on, off)

    on_acc = np.zeros((H, W), dtype=np.float32)
    off_acc = np.zeros((H, W), dtype=np.float32)

    y = coords[:, 1].clip(0, H - 1)
    x = coords[:, 2].clip(0, W - 1)
    np.add.at(on_acc, (y, x), feats[:, 0])
    np.add.at(off_acc, (y, x), feats[:, 1])

    on_max = on_acc.max() if on_acc.max() > 0 else 1
    off_max = off_acc.max() if off_acc.max() > 0 else 1
    on_norm = on_acc / on_max
    off_norm = off_acc / off_max

    image = np.zeros((H, W, 3), dtype=np.float32)
    image[:, :, 0] = on_norm   # Red = ON
    image[:, :, 2] = off_norm  # Blue = OFF
    image[:, :, 1] = np.minimum(on_norm, off_norm)  # Green = overlap
    return image


def load_rgb_frame(rgb_root, seq_id, frame_idx, target_size=(640, 640)):
    """Load RGB frame by index, resize to target_size for alignment."""
    rgb_dir = Path(rgb_root) / str(seq_id) / "RGB"
    if not rgb_dir.exists():
        return np.zeros((*target_size, 3), dtype=np.float32)

    files = sorted(os.listdir(rgb_dir))
    if frame_idx >= len(files) or frame_idx < 0:
        return np.zeros((*target_size, 3), dtype=np.float32)

    img_path = rgb_dir / files[frame_idx]
    img = Image.open(img_path).convert("RGB")
    img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def parse_frame_idx(frame_id):
    """Extract sequence ID and frame index from frame_id like '111_frame_000500'."""
    parts = frame_id.split("_frame_")
    if len(parts) == 2:
        return parts[0], int(parts[1])
    return None, None


# ============================================================================
# Drawing
# ============================================================================

def draw_boxes_on_ax(ax, record, show_dets=True, max_dets=10):
    """Draw GT and detection boxes on a matplotlib axis."""
    # GT box (green)
    if record.get("gt_box"):
        x1, y1, x2, y2 = record["gt_box"]
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2.0,
                                  edgecolor=GT_COLOR, facecolor="none",
                                  linestyle="-")
        ax.add_patch(rect)
        ax.text(x1, y1 - 3, "GT", fontsize=7, color=GT_COLOR,
                fontweight="bold", va="bottom")

    if not show_dets:
        return

    # Detection boxes (top N by confidence)
    det_boxes = record.get("det_boxes", [])
    det_confs = record.get("det_confs", [])
    det_labels = record.get("det_labels", [])

    n_show = min(len(det_boxes), max_dets)
    for i in range(n_show):
        box = det_boxes[i]
        conf = det_confs[i] if i < len(det_confs) else 0
        label = det_labels[i] if i < len(det_labels) else "?"

        style = DET_STYLES.get(label, ("#888888", "-", 1.0, label))
        color, ls, lw, display = style

        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=lw,
                                  edgecolor=color, facecolor="none",
                                  linestyle=ls)
        ax.add_patch(rect)
        if conf > 0.05:  # only label significant dets
            ax.text(x2 + 2, y1, f"{conf:.2f}", fontsize=5, color=color, va="top")


def render_sidebyside(event_img, rgb_img, record, title="", show_dets=True,
                      max_dets=8, figsize=(14, 5)):
    """Render event + RGB side-by-side with boxes."""
    fig, (ax_event, ax_rgb) = plt.subplots(1, 2, figsize=figsize)

    # Event frame
    ax_event.imshow(event_img)
    draw_boxes_on_ax(ax_event, record, show_dets=show_dets, max_dets=max_dets)
    ax_event.set_title("Sparse Events", fontsize=10)
    ax_event.axis("off")

    # RGB frame (boxes in original coords need scaling: 1280->640 x, 720->640 y)
    # But our GT boxes are already in 640x640 space (squashed).
    # RGB is resized to 640x640 for alignment, so boxes apply directly.
    ax_rgb.imshow(rgb_img)
    draw_boxes_on_ax(ax_rgb, record, show_dets=False, max_dets=0)  # GT only on RGB
    ax_rgb.set_title("RGB Frame", fontsize=10)
    ax_rgb.axis("off")

    # Title
    status = "TP" if record["matched"] else f"FN ({record.get('fn_label', '?')})"
    gt_dim = record.get("gt_max_dim", 0)
    conf_str = f"conf={record.get('matched_conf', 0):.2f}" if record["matched"] else ""
    iou_str = f"IoU={record.get('matched_iou', record.get('fn_best_iou', 0)):.3f}"
    fig.suptitle(f"{title} | {status} | GT {gt_dim:.0f}px | {iou_str} {conf_str}",
                 fontsize=11, fontweight="bold")

    plt.tight_layout()
    return fig


# ============================================================================
# Rendering modes
# ============================================================================

def render_sequence_overview(records, sparse_dir, rgb_root, outdir, seq_id, step=20):
    """Render every Nth frame of a sequence side-by-side."""
    seq_records = [r for r in records if r["seq_id"] == str(seq_id)]
    seq_records.sort(key=lambda r: r["frame_num"])

    out_path = Path(outdir) / f"seq_{seq_id}_overview"
    out_path.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, rec in enumerate(seq_records):
        if i % step != 0:
            continue
        _, frame_idx = parse_frame_idx(rec["frame_id"])
        if frame_idx is None:
            continue

        event_img = load_sparse_event_frame(sparse_dir, seq_id, frame_idx)
        rgb_img = load_rgb_frame(rgb_root, seq_id, frame_idx)

        fig = render_sidebyside(event_img, rgb_img, rec,
                                title=f"seq{seq_id} frame {frame_idx}")
        fname = f"{count:04d}_frame{frame_idx:06d}.png"
        fig.savefig(out_path / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        count += 1

    print(f"Rendered {count} overview frames to {out_path}")
    return count


def render_fn_gallery(records, sparse_dir, rgb_root, outdir, seq_id,
                      max_frames=500):
    """Render all FN frames from a specific sequence."""
    fn_recs = [r for r in records
               if r["seq_id"] == str(seq_id) and not r["matched"]]
    fn_recs.sort(key=lambda r: r["frame_num"])

    out_path = Path(outdir) / f"seq_{seq_id}_fn"
    out_path.mkdir(parents=True, exist_ok=True)

    count = 0
    for rec in fn_recs[:max_frames]:
        _, frame_idx = parse_frame_idx(rec["frame_id"])
        if frame_idx is None:
            continue

        event_img = load_sparse_event_frame(sparse_dir, seq_id, frame_idx)
        rgb_img = load_rgb_frame(rgb_root, seq_id, frame_idx)

        fig = render_sidebyside(event_img, rgb_img, rec,
                                title=f"seq{seq_id} FN frame {frame_idx}")
        label = rec.get("fn_label", "fn")
        fname = f"{count:04d}_{label}_frame{frame_idx:06d}.png"
        fig.savefig(out_path / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        count += 1

    print(f"Rendered {count} FN frames from seq {seq_id} to {out_path}")
    return count


def render_complete_miss_gallery(records, sparse_dir, rgb_root, outdir,
                                 max_per_seq=30, max_total=300):
    """Render complete miss frames grouped by sequence."""
    miss_recs = [r for r in records
                 if not r["matched"] and r.get("fn_label") == "fn_complete_miss"]

    # Group by sequence
    from collections import defaultdict
    by_seq = defaultdict(list)
    for r in miss_recs:
        by_seq[r["seq_id"]].append(r)

    out_path = Path(outdir) / "complete_misses"
    out_path.mkdir(parents=True, exist_ok=True)

    total = 0
    for seq_id in sorted(by_seq.keys(), key=lambda s: -len(by_seq[s])):
        seq_recs = sorted(by_seq[seq_id], key=lambda r: r["frame_num"])
        seq_out = out_path / f"seq_{seq_id}"
        seq_out.mkdir(exist_ok=True)

        count = 0
        for rec in seq_recs[:max_per_seq]:
            _, frame_idx = parse_frame_idx(rec["frame_id"])
            if frame_idx is None:
                continue

            event_img = load_sparse_event_frame(sparse_dir, seq_id, frame_idx)
            rgb_img = load_rgb_frame(rgb_root, seq_id, frame_idx)

            fig = render_sidebyside(event_img, rgb_img, rec,
                                    title=f"COMPLETE MISS seq{seq_id} f{frame_idx}")
            fname = f"{count:04d}_frame{frame_idx:06d}.png"
            fig.savefig(seq_out / fname, dpi=120, bbox_inches="tight")
            plt.close(fig)
            count += 1
            total += 1

            if total >= max_total:
                print(f"Hit max_total={max_total}, stopping")
                return total

        if count > 0:
            print(f"  seq {seq_id}: {count} complete miss frames rendered "
                  f"({len(seq_recs)} total)")

    print(f"Rendered {total} complete miss frames to {out_path}")
    return total


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side Event + RGB visualization for error analysis")
    parser.add_argument("--dump", required=True,
                        help="Path to predictions.json (JSONL)")
    parser.add_argument("--sparse-dir", required=True,
                        help="Root sparse dir (canonical_test/)")
    parser.add_argument("--rgb-root", required=True,
                        help="Root RGB dir (data/processed/FRED/test/)")
    parser.add_argument("--outdir", required=True,
                        help="Output directory")
    parser.add_argument("--mode", required=True,
                        choices=["sequence_overview", "fn_gallery",
                                 "complete_miss_gallery"],
                        help="Visualization mode")
    parser.add_argument("--seq-id", type=str, default=None,
                        help="Sequence ID (required for sequence_overview/fn_gallery)")
    parser.add_argument("--step", type=int, default=20,
                        help="Frame step for sequence_overview mode")
    parser.add_argument("--max-per-seq", type=int, default=30,
                        help="Max frames per sequence (complete_miss_gallery)")
    parser.add_argument("--max-total", type=int, default=300,
                        help="Max total frames (complete_miss_gallery)")
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Max frames (fn_gallery)")
    args = parser.parse_args()

    print(f"Loading records from {args.dump}...")
    records = load_records(args.dump)
    print(f"  Loaded {len(records)} records")

    if args.mode == "sequence_overview":
        if not args.seq_id:
            parser.error("--seq-id required for sequence_overview mode")
        render_sequence_overview(records, args.sparse_dir, args.rgb_root,
                                 args.outdir, args.seq_id, step=args.step)

    elif args.mode == "fn_gallery":
        if not args.seq_id:
            parser.error("--seq-id required for fn_gallery mode")
        render_fn_gallery(records, args.sparse_dir, args.rgb_root,
                          args.outdir, args.seq_id, max_frames=args.max_frames)

    elif args.mode == "complete_miss_gallery":
        render_complete_miss_gallery(records, args.sparse_dir, args.rgb_root,
                                     args.outdir,
                                     max_per_seq=args.max_per_seq,
                                     max_total=args.max_total)


if __name__ == "__main__":
    main()
