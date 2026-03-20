#!/usr/bin/env python3
"""
Regenerate sparse data — 640x640 ablation (6ch features, downscaled from native).

Output per frame:
- coords: [N, 3] int16 (t_bin, y, x) at 640x640
- feats:  [N, 6] float16 (log1p_on, log1p_off, recency_on, recency_off, std_on, std_off)
- time_bins: int32 scalar metadata (=16)
- n_events: int64 scalar (raw event count for this frame)

Usage:
    python tools/regenerate_parity_sparse_coords_v82_640.py --dry-run
    python tools/regenerate_parity_sparse_coords_v82_640.py --workers 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

# ============================================================================
# Constants — V82: native resolution, 16 time bins, 6-channel features
# ============================================================================
NUM_TIME_BINS = 16          # power-of-2 temporal bins (2.08 ms each)
FRAME_DURATION_US = 33333   # microseconds per frame
SENSOR_SIZE = (720, 1280)   # original sensor (H, W)
TARGET_SIZE = (640, 640)    # ABLATION: downscale to v80-style square resolution
MAX_EVENTS_PER_FRAME = 500000
HOT_PIXEL_THRESHOLD_MULT = 3  # suppress pixels with > MULT events per time bin


def events_to_sparse_voxels(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    frame_duration_us: int = FRAME_DURATION_US,
    num_time_bins: int = NUM_TIME_BINS,
    sensor_size: Tuple[int, int] = SENSOR_SIZE,
    target_size: Tuple[int, int] = TARGET_SIZE,
    max_events: int = MAX_EVENTS_PER_FRAME,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Convert raw events to sparse voxel coordinates and 6-channel features.

    V82 6-channel temporal surface per voxel:
      ch0: log1p(on_count)   - event density, ON polarity
      ch1: log1p(off_count)  - event density, OFF polarity
      ch2: recency_on        - exponential decay of most recent ON timestamp
      ch3: recency_off       - exponential decay of most recent OFF timestamp
      ch4: std_on            - temporal std of ON events within voxel
      ch5: std_off           - temporal std of OFF events within voxel

    Also applies hot pixel filtering: suppresses voxels where event rate
    exceeds HOT_PIXEL_THRESHOLD_MULT times the average per-bin rate.

    Returns:
        coords: [N, 3] int16  (t_bin, y, x)
        feats:  [N, 6] float16 (6-channel temporal surface)
        n_events: int  (original event count before subsampling)
    """
    n_events_raw = len(x)

    if n_events_raw == 0:
        return (
            np.zeros((0, 3), dtype=np.int16),
            np.zeros((0, 6), dtype=np.float16),
            0,
        )

    # Subsample if too many events
    if n_events_raw > max_events:
        idx = np.random.choice(n_events_raw, max_events, replace=False)
        x, y, t, p = x[idx], y[idx], t[idx], p[idx]

    n_events = len(x)

    # Scale spatial coordinates: sensor_size -> target_size (identity for native)
    scale_y = target_size[0] / sensor_size[0]
    scale_x = target_size[1] / sensor_size[1]
    x_scaled = np.clip((x * scale_x).astype(np.int32), 0, target_size[1] - 1)
    y_scaled = np.clip((y * scale_y).astype(np.int32), 0, target_size[0] - 1)

    # Quantize time to bins
    t_bin = np.clip(
        (t * num_time_bins // frame_duration_us).astype(np.int32),
        0,
        num_time_bins - 1,
    )

    # Vectorized voxelization via linear indexing
    T, H, W = num_time_bins, target_size[0], target_size[1]
    linear_idx = t_bin * (H * W) + y_scaled * W + x_scaled
    unique_idx, inverse = np.unique(linear_idx, return_inverse=True)
    n_voxels = len(unique_idx)

    # Decode linear index back to (t, y, x)
    coords = np.zeros((n_voxels, 3), dtype=np.int16)
    coords[:, 0] = unique_idx // (H * W)
    coords[:, 1] = (unique_idx % (H * W)) // W
    coords[:, 2] = unique_idx % W

    # --- Polarity masks ---
    on_mask = (p == 1)
    off_mask = (p == 0)

    # --- Channel 0-1: log1p event counts per polarity ---
    on_counts = np.bincount(inverse, weights=on_mask.astype(np.float32), minlength=n_voxels)
    off_counts = np.bincount(inverse, weights=off_mask.astype(np.float32), minlength=n_voxels)

    # --- Hot pixel filter ---
    # Suppress voxels with event count > MULT * average events per voxel
    total_counts = on_counts + off_counts
    avg_events_per_voxel = n_events / max(n_voxels, 1)
    hot_threshold = HOT_PIXEL_THRESHOLD_MULT * max(avg_events_per_voxel, 1.0)
    hot_mask = total_counts > hot_threshold
    if hot_mask.any():
        keep = ~hot_mask
        coords = coords[keep]
        on_counts = on_counts[keep]
        off_counts = off_counts[keep]
        # Rebuild inverse mapping for kept voxels (needed for recency/std)
        old_to_new = np.full(n_voxels, -1, dtype=np.int64)
        new_indices = np.where(keep)[0]
        old_to_new[new_indices] = np.arange(len(new_indices))
        new_inverse = old_to_new[inverse]
        # Events mapping to removed voxels get -1; filter them out
        event_keep = new_inverse >= 0
        inverse = new_inverse[event_keep]
        t_local = t[event_keep]
        on_mask_local = on_mask[event_keep]
        off_mask_local = off_mask[event_keep]
        n_voxels = len(new_indices)
    else:
        t_local = t
        on_mask_local = on_mask
        off_mask_local = off_mask

    if n_voxels == 0:
        return (
            np.zeros((0, 3), dtype=np.int16),
            np.zeros((0, 6), dtype=np.float16),
            n_events_raw,
        )

    # --- Channel 2-3: recency (exponential decay of most recent timestamp) ---
    # For each voxel, find the most recent timestamp per polarity
    # Use scatter-max via np.maximum.at
    t_float = t_local.astype(np.float64)
    t_max_global = t_float.max() if len(t_float) > 0 else 1.0
    t_range = max(t_max_global - t_float.min(), 1.0)
    LAMBDA = 5.0 / t_range  # decay constant: e^(-5) ~ 0.007 at oldest event

    recency_on = np.zeros(n_voxels, dtype=np.float64)
    recency_off = np.zeros(n_voxels, dtype=np.float64)

    # Last timestamp per voxel per polarity
    last_t_on = np.full(n_voxels, -1e18, dtype=np.float64)
    last_t_off = np.full(n_voxels, -1e18, dtype=np.float64)

    on_event_idx = np.where(on_mask_local)[0]
    off_event_idx = np.where(off_mask_local)[0]

    if len(on_event_idx) > 0:
        np.maximum.at(last_t_on, inverse[on_event_idx], t_float[on_event_idx])
    if len(off_event_idx) > 0:
        np.maximum.at(last_t_off, inverse[off_event_idx], t_float[off_event_idx])

    # Exponential decay: recency = exp(-lambda * (t_max - t_last))
    valid_on = last_t_on > -1e17
    valid_off = last_t_off > -1e17
    recency_on[valid_on] = np.exp(-LAMBDA * (t_max_global - last_t_on[valid_on]))
    recency_off[valid_off] = np.exp(-LAMBDA * (t_max_global - last_t_off[valid_off]))

    # --- Channel 4-5: temporal std per polarity ---
    # std = sqrt(E[t^2] - E[t]^2) normalized by time range
    std_on = np.zeros(n_voxels, dtype=np.float64)
    std_off = np.zeros(n_voxels, dtype=np.float64)

    if len(on_event_idx) > 0:
        inv_on = inverse[on_event_idx]
        t_on = t_float[on_event_idx]
        sum_t = np.bincount(inv_on, weights=t_on, minlength=n_voxels)
        sum_t2 = np.bincount(inv_on, weights=t_on ** 2, minlength=n_voxels)
        cnt_on = np.bincount(inv_on, minlength=n_voxels).astype(np.float64)
        valid = cnt_on > 1
        mean_t = np.divide(sum_t, cnt_on, where=valid, out=np.zeros_like(sum_t))
        var_t = np.divide(sum_t2, cnt_on, where=valid, out=np.zeros_like(sum_t2)) - mean_t ** 2
        var_t = np.maximum(var_t, 0.0)  # numerical safety
        std_on[valid] = np.sqrt(var_t[valid]) / t_range  # normalize to [0, ~1]

    if len(off_event_idx) > 0:
        inv_off = inverse[off_event_idx]
        t_off = t_float[off_event_idx]
        sum_t = np.bincount(inv_off, weights=t_off, minlength=n_voxels)
        sum_t2 = np.bincount(inv_off, weights=t_off ** 2, minlength=n_voxels)
        cnt_off = np.bincount(inv_off, minlength=n_voxels).astype(np.float64)
        valid = cnt_off > 1
        mean_t = np.divide(sum_t, cnt_off, where=valid, out=np.zeros_like(sum_t))
        var_t = np.divide(sum_t2, cnt_off, where=valid, out=np.zeros_like(sum_t2)) - mean_t ** 2
        var_t = np.maximum(var_t, 0.0)
        std_off[valid] = np.sqrt(var_t[valid]) / t_range

    # --- Stack 6 channels ---
    feats = np.stack([
        np.log1p(on_counts),    # ch0: log1p ON count
        np.log1p(off_counts),   # ch1: log1p OFF count
        recency_on,             # ch2: recency ON
        recency_off,            # ch3: recency OFF
        std_on,                 # ch4: temporal std ON
        std_off,                # ch5: temporal std OFF
    ], axis=1).astype(np.float16)

    return coords, feats, n_events_raw


def process_sequence(
    seq_id: str,
    raw_split: str,
    label_files: List[Path],
    output_dir: Path,
    num_time_bins: int,
    dry_run: bool,
) -> Dict:
    """Process all frames for one sequence from raw events."""
    stats = {
        "seq_id": seq_id,
        "n_frames": 0,
        "n_voxels_total": 0,
        "n_events_total": 0,
        "errors": [],
    }

    event_file = Path(f"data/processed/FRED/{raw_split}/{seq_id}/Event/events.hdf5")
    if not event_file.exists():
        stats["errors"].append(f"Missing HDF5: {event_file}")
        return stats

    try:
        with h5py.File(event_file, "r") as f:
            events_data = f["CD"]["events"][:]
        ev_x = events_data["x"].astype(np.int32)
        ev_y = events_data["y"].astype(np.int32)
        ev_t = events_data["t"].astype(np.int64)
        ev_p = events_data["p"].astype(np.int32)
        t_start = int(ev_t[0]) if len(ev_t) > 0 else 0
    except Exception as e:
        stats["errors"].append(f"HDF5 read error: {e}")
        return stats

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Sort label files by frame index for efficient sequential access
    frame_indices = []
    for label_file in label_files:
        name = label_file.stem
        parts = name.split("_frame_")
        if len(parts) != 2:
            continue
        frame_indices.append(int(parts[1]))
    frame_indices.sort()

    # Use binary search (O(log N)) instead of boolean masking (O(N)) per frame.
    # Events are already sorted by timestamp in HDF5, so searchsorted works.
    for frame_idx in frame_indices:
        frame_start = t_start + frame_idx * FRAME_DURATION_US
        frame_end = frame_start + FRAME_DURATION_US

        i0 = np.searchsorted(ev_t, frame_start, side="left")
        i1 = np.searchsorted(ev_t, frame_end, side="left")

        x = ev_x[i0:i1]
        y = ev_y[i0:i1]
        t = ev_t[i0:i1] - frame_start
        p = ev_p[i0:i1]

        coords, feats, n_events = events_to_sparse_voxels(
            x, y, t, p, num_time_bins=num_time_bins
        )

        stats["n_frames"] += 1
        stats["n_voxels_total"] += len(coords)
        stats["n_events_total"] += n_events

        if not dry_run:
            frame_file = output_dir / f"frame_{frame_idx:06d}.npz"
            np.savez_compressed(
                frame_file,
                coords=coords,
                feats=feats,
                n_events=np.int64(n_events),
                time_bins=np.int32(num_time_bins),
            )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate ALL parity sparse data from raw HDF5 in coords format"
    )
    parser.add_argument(
        "--parity-root",
        type=str,
        default="data/datasets/fred_paper_parity",
        help="Root of parity dataset (labels/ subdirs)",
    )
    parser.add_argument(
        "--output-sparse-root",
        type=str,
        default="data/datasets/fred_paper_parity_v82_640/sparse",
        help="Output root for regenerated v82-640 ablation sparse data.",
    )
    parser.add_argument(
        "--time-bins",
        type=int,
        default=NUM_TIME_BINS,
        help=f"Number of temporal bins (default: {NUM_TIME_BINS})",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["canonical_train", "canonical_test", "challenging_train", "challenging_test"],
    )
    args = parser.parse_args()

    parity_root = Path(args.parity_root)
    label_root = parity_root / "labels"
    output_root = Path(args.output_sparse_root)

    if not label_root.exists():
        print(f"ERROR: Label root not found: {label_root}")
        sys.exit(1)

    # Discover which raw split (train/test) each sequence lives in
    raw_train_seqs = set(os.listdir("data/processed/FRED/train"))
    raw_test_seqs = set(os.listdir("data/processed/FRED/test"))

    def raw_split_for(seq_id: str) -> Optional[str]:
        if seq_id in raw_train_seqs:
            return "train"
        elif seq_id in raw_test_seqs:
            return "test"
        return None

    # Build work items: (seq_id, raw_split, label_files, output_dir)
    work_items = []
    total_labels = 0
    for split in args.splits:
        split_label_dir = label_root / split
        if not split_label_dir.exists():
            print(f"WARNING: Skipping split {split} — no labels at {split_label_dir}")
            continue

        # Group label files by sequence (os.listdir is much faster than Path.iterdir for 500K+ entries)
        seq_labels: Dict[str, List[Path]] = defaultdict(list)
        print(f"  Scanning labels in {split}...", end=" ", flush=True)
        all_names = os.listdir(split_label_dir)
        print(f"{len(all_names):,} files")
        for name in all_names:
            if not name.endswith(".txt"):
                continue
            parts = name.split("_frame_")
            if len(parts) != 2:
                continue
            seq_id = parts[0]
            seq_labels[seq_id].append(split_label_dir / name)

        for seq_id, labels in sorted(seq_labels.items()):
            rs = raw_split_for(seq_id)
            if rs is None:
                print(f"ERROR: No raw HDF5 for seq {seq_id} in split {split}")
                continue
            out_dir = output_root / split / seq_id
            work_items.append((seq_id, rs, labels, out_dir, split))
            total_labels += len(labels)

    print(f"{'DRY RUN: ' if args.dry_run else ''}Regenerating {len(work_items)} sequences, "
          f"{total_labels:,} frames across splits: {args.splits}")
    print(f"  Time bins: {args.time_bins}")
    print(f"  Output: {output_root}")
    print(f"  Workers: {args.workers}")

    if not args.dry_run:
        # Create split dirs (caller should clean old data first!)
        for split in args.splits:
            split_dir = output_root / split
            split_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    all_stats = []
    errors = []
    split_stats = Counter()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for seq_id, rs, labels, out_dir, split in work_items:
            fut = executor.submit(
                process_sequence,
                seq_id, rs, labels, out_dir, args.time_bins, args.dry_run,
            )
            futures[fut] = (seq_id, split)

        done = 0
        for future in as_completed(futures):
            seq_id, split = futures[future]
            done += 1
            try:
                stats = future.result()
                all_stats.append(stats)
                split_stats[split] += stats["n_frames"]
                if stats["errors"]:
                    errors.extend(stats["errors"])
                if done % 20 == 0 or done == len(work_items):
                    elapsed = time.time() - start_time
                    frames_done = sum(s["n_frames"] for s in all_stats)
                    print(f"  [{done}/{len(work_items)}] seqs done, "
                          f"{frames_done:,} frames, {elapsed:.0f}s elapsed")
            except Exception as e:
                errors.append(f"Fatal error on seq {seq_id}: {e}")

    elapsed = time.time() - start_time
    total_frames = sum(s["n_frames"] for s in all_stats)
    total_voxels = sum(s["n_voxels_total"] for s in all_stats)
    total_events = sum(s["n_events_total"] for s in all_stats)

    print(f"\n{'DRY RUN ' if args.dry_run else ''}COMPLETE in {elapsed:.0f}s")
    print(f"  Sequences: {len(all_stats)}")
    print(f"  Frames:    {total_frames:,}")
    print(f"  Voxels:    {total_voxels:,}")
    print(f"  Events:    {total_events:,}")
    print(f"  Errors:    {len(errors)}")
    if total_frames > 0:
        print(f"  Avg voxels/frame: {total_voxels / total_frames:.0f}")

    for split, count in sorted(split_stats.items()):
        print(f"  {split}: {count:,} frames")

    if errors:
        print(f"\nFirst 10 errors:")
        for e in errors[:10]:
            print(f"  {e}")

    if not args.dry_run:
        # Spot-check: verify 10 random files
        print(f"\nVerification (10 random files):")
        import random
        check_files = []
        for split in args.splits:
            split_dir = output_root / split
            for seq_dir in split_dir.iterdir():
                if not seq_dir.is_dir():
                    continue
                for f in seq_dir.iterdir():
                    if f.suffix == ".npz":
                        check_files.append(f)
        if check_files:
            samples = random.sample(check_files, min(10, len(check_files)))
            all_ok = True
            for fp in samples:
                d = np.load(fp)
                k = sorted(d.keys())
                if "coords" not in k or "feats" not in k or "time_bins" not in k:
                    print(f"  FAIL {fp.name}: keys={k}")
                    all_ok = False
                else:
                    c, f = d["coords"], d["feats"]
                    tb = int(d["time_bins"])
                    tmax = int(c[:, 0].max()) + 1 if len(c) > 0 else 0
                    print(
                        f"  OK {fp.name}: voxels={len(c)}, feats=[{f.min():.4f},{f.max():.4f}], "
                        f"tmax={tmax}, time_bins={tb}, dtype_c={c.dtype} dtype_f={f.dtype}"
                    )
            if all_ok:
                print("  [OK] All verification checks passed!")

        # Write manifest
        manifest = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "format": "coords",
            "time_bins": args.time_bins,
            "feature_encoding": "6ch_temporal_surface",
            "feature_channels": ["log1p_on", "log1p_off", "recency_on", "recency_off", "std_on", "std_off"],
            "sensor_size": list(SENSOR_SIZE),
            "target_size": list(TARGET_SIZE),
            "hot_pixel_threshold_mult": HOT_PIXEL_THRESHOLD_MULT,
            "frame_duration_us": FRAME_DURATION_US,
            "total_sequences": len(all_stats),
            "total_frames": total_frames,
            "total_voxels": total_voxels,
            "total_events": total_events,
            "splits": dict(sorted(split_stats.items())),
            "errors": len(errors),
        }
        manifest_path = output_root / "manifest.json"
        with open(manifest_path, "w") as mf:
            json.dump(manifest, mf, indent=2)
        print(f"\nManifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
