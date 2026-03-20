#!/usr/bin/env python3
"""
Sparse Event Dataset — 6-channel variant with rectangular resolution support.

Loads pre-processed sparse event tensors (.npz) and YOLO-format labels.
Supports augmentation (flip, crop, dropout, mosaic) and batched sparse collation.
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import spconv.pytorch as spconv
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json
import random
import time

from detection.scripts.event_mosaic import sparse_mosaic

LOG1P_500 = float(np.log1p(500.0))


class SparseEventDataset(Dataset):
    """
    Dataset for sparse event tensors with YOLO labels.

    Data structure expected:
    sparse_dir/
        sequence_001/
            frame_000000.npz  # coords, feats, n_events
            frame_000001.npz
            ...
        manifest.json  # metadata

    label_dir/
        sequence_001_frame_000000.txt  # YOLO format
        sequence_001_frame_000001.txt
        ...
    """

    def __init__(
        self,
        sparse_dir: str,
        label_dir: str,
        split: str = 'train',
        time_bins: int = 33,
        target_size: Tuple[int, int] = (640, 640),
        augment: bool = True,
        horizontal_flip_prob: float = 0.5,
        event_dropout_prob: float = 0.1,
        temporal_flip_prob: float = 0.0,
        polarity_flip_prob: float = 0.0,
        scale_range: Tuple[float, float] = (1.0, 1.0),
        mosaic_prob: float = 0.0,
        max_voxels: int = 100000,
        voxel_sampling: Optional[Dict] = None,
        cache_labels: bool = True,
        feature_channels: Optional[int] = None
    ):
        """
        Args:
            sparse_dir: Directory with sparse tensors
            label_dir: Directory with YOLO format labels
            split: Data split ('train', 'val', 'test')
            time_bins: Number of temporal bins
            target_size: (H, W) spatial size
            augment: Enable augmentation
            horizontal_flip_prob: Probability of horizontal flip
            event_dropout_prob: Probability of dropping voxels
            temporal_flip_prob: Probability of reversing time dimension
            polarity_flip_prob: Probability of swapping ON/OFF channels
            scale_range: (min_scale, max_scale) for spatial affine augmentation
            mosaic_prob: Probability of 4-image mosaic augmentation
            max_voxels: Maximum voxels per sample
            voxel_sampling: Optional voxel retention policy when clipping.
                Supported:
                    mode: "random" | "weighted"
                    gt_expand_px: float (default 8)
                    temporal_radius_bins: int (default 2)
                    spatial_radius_px: float (default 56)
                    weights:
                        gt_window: float (default 0.4)
                        temporal_window: float (default 0.4)
                        global: float (default 0.2)
            cache_labels: Cache labels in memory
            feature_channels: If set, slice features to first N channels (for ablation).
                E.g. feature_channels=2 with 6ch data uses only log1p counts.
        """
        self.sparse_dir = Path(sparse_dir) / split
        self.label_dir = Path(label_dir) / split
        self.feature_channels = feature_channels
        self.time_bins = time_bins
        self.target_size = target_size
        self.spatial_shape = [time_bins, target_size[0], target_size[1]]  # [T, H, W]
        self.augment = augment and split == 'train'
        self.horizontal_flip_prob = horizontal_flip_prob
        self.event_dropout_prob = event_dropout_prob
        self.temporal_flip_prob = temporal_flip_prob
        self.polarity_flip_prob = polarity_flip_prob
        self.scale_range = scale_range
        self.mosaic_prob = mosaic_prob
        self.max_voxels = int(max_voxels)
        self.cache_labels = cache_labels
        self.voxel_sampling = dict(voxel_sampling or {})
        self.voxel_sampling_mode = str(self.voxel_sampling.get("mode", "random")).lower()
        self.voxel_sampling_gt_expand = float(self.voxel_sampling.get("gt_expand_px", 8.0))
        self.voxel_sampling_temporal_radius = int(self.voxel_sampling.get("temporal_radius_bins", 2))
        self.voxel_sampling_spatial_radius = float(self.voxel_sampling.get("spatial_radius_px", 56.0))
        weights_cfg = self.voxel_sampling.get("weights", {}) or {}
        w_gt = float(weights_cfg.get("gt_window", 0.4))
        w_temporal = float(weights_cfg.get("temporal_window", 0.4))
        w_global = float(weights_cfg.get("global", 0.2))
        total_w = max(w_gt + w_temporal + w_global, 1e-6)
        self.voxel_sampling_weights = {
            "gt_window": max(w_gt / total_w, 0.0),
            "temporal_window": max(w_temporal / total_w, 0.0),
            "global": max(w_global / total_w, 0.0),
        }

        # Load manifest
        manifest_path = self.sparse_dir / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}

        # Find all samples
        self.samples = self._find_samples()
        print(f"Found {len(self.samples)} samples in {split}")

        # Label cache
        self._label_cache = {} if cache_labels else None

        # In-memory sparse data cache (preloads all .npz -> coords/feats into RAM)
        self._sparse_cache: Optional[Dict[Path, Tuple[np.ndarray, np.ndarray]]] = None

        # Format telemetry
        self._load_count: int = 0

    def preload_to_ram(self) -> None:
        """Preload all sparse .npz files into RAM for zero-I/O training.

        Call this after construction to eliminate disk reads during training.
        Typical memory: ~17KB/sample × 510K samples ≈ 9GB.
        """
        import sys
        cache = {}
        print(f"Preloading {len(self.samples)} samples into RAM...", flush=True)
        t0 = time.time()
        for i, (sparse_path, _) in enumerate(self.samples):
            cache[sparse_path] = self._load_sparse(sparse_path)
            if (i + 1) % 50000 == 0:
                mb = sum(c.nbytes + f.nbytes for c, f in cache.values()) / 1e6
                print(f"  Preloaded {i+1}/{len(self.samples)} ({mb:.0f} MB)", flush=True)
        elapsed = time.time() - t0
        total_mb = sum(c.nbytes + f.nbytes for c, f in cache.values()) / 1e6
        print(f"  Preload complete: {len(cache)} samples, {total_mb:.0f} MB, {elapsed:.1f}s", flush=True)
        self._sparse_cache = cache

    def _find_samples(self) -> List[Tuple[Path, Path]]:
        """Find all valid (sparse_file, label_file) pairs.

        Every sparse file MUST have a matching label.
        Missing labels are counted and reported.
        """
        samples = []
        missing_labels = []

        # Iterate through sequence directories
        for seq_dir in sorted(self.sparse_dir.iterdir()):
            if not seq_dir.is_dir():
                continue

            seq_id = seq_dir.name

            # Find all frame files
            for frame_file in sorted(seq_dir.glob('frame_*.npz')):
                frame_idx = int(frame_file.stem.split('_')[1])

                # Find corresponding label
                label_file = self.label_dir / f'{seq_id}_frame_{frame_idx:06d}.txt'
                if not label_file.exists():
                    # Try without zero padding
                    label_file = self.label_dir / f'{seq_id}_frame_{frame_idx}.txt'

                if label_file.exists():
                    samples.append((frame_file, label_file))
                else:
                    missing_labels.append(str(frame_file))

        if missing_labels:
            print(f"  WARNING: {len(missing_labels)} sparse files have NO matching label!")
            print(f"  First 5: {missing_labels[:5]}")
            raise RuntimeError(
                f"Data integrity violation: {len(missing_labels)}/{len(samples) + len(missing_labels)} "
                f"sparse files missing labels in {self.sparse_dir}. "
                f"No silent fallbacks — fix labels before proceeding."
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sparse(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load sparse coordinates and features from .npz file.

        Expected file contents (from regenerate_parity_sparse_coords.py):
            coords:    [N, 3] int16   (t_bin, y, x)
            feats:     [N, 2] float16 (log(1+on_count), log(1+off_count))
            time_bins: int32 scalar   (=33)
            n_events:  int64 scalar   (raw event count)

        Temporal rebinning (e.g. 33→15) is applied when data_T != target T.
        Duplicates from rebinning are coalesced by taking the max feature.
        """
        # Fast path: serve from RAM cache
        if self._sparse_cache is not None and path in self._sparse_cache:
            c, f = self._sparse_cache[path]
            return c.copy(), f.copy()

        data = np.load(path)

        if 'coords' in data and 'feats' in data:
            coords = data['coords'].astype(np.int32)   # [N, 3] = (t, y, x)
            feats = data['feats'].astype(np.float32)   # [N, C] = feature channels (2ch or 6ch)
        elif 'spikes' in data:
            # Legacy dense format: [T, 2, H, W] normalized by log1p(500).
            spikes = data['spikes'].astype(np.float32)
            if spikes.ndim != 4 or spikes.shape[1] != 2:
                raise ValueError(
                    f"Invalid spikes tensor in {path}: shape={spikes.shape}"
                )

            mask = np.any(spikes != 0, axis=1)  # [T, H, W]
            t_idx, y_idx, x_idx = np.where(mask)
            if len(t_idx) == 0:
                coords = np.zeros((0, 3), dtype=np.int32)
                feats = np.zeros((0, 2), dtype=np.float32)
            else:
                coords = np.stack([t_idx, y_idx, x_idx], axis=1).astype(np.int32)
                feats = spikes[t_idx, :, y_idx, x_idx].astype(np.float32)
                # Convert normalized spikes scale back to log1p-count scale.
                feats *= LOG1P_500
        else:
            raise ValueError(
                f"Expected coords+feats or spikes format in {path}, "
                f"got keys={list(data.keys())}."
            )

        self._load_count += 1

        # Input validation
        if len(coords) > 0:
            T, H, W = self.spatial_shape

            # Temporal rebinning: map data time bins to target T.
            # Prefer explicit time_bins metadata from npz; fall back to active-max.
            if 'time_bins' in data:
                data_T = int(data['time_bins'])
            else:
                data_T = int(coords[:, 0].max()) + 1
            if data_T != T:
                coords[:, 0] = (coords[:, 0] * T) // data_T

                # Coalesce duplicates created by rebinning: take max per voxel.
                # E.g. T=33→15 maps ~2.2 source bins per target bin.
                # spconv does NOT coalesce — duplicates corrupt feature magnitude.
                keys = (coords[:, 0].astype(np.int64) * (H * W)
                        + coords[:, 1].astype(np.int64) * W
                        + coords[:, 2].astype(np.int64))
                unique_keys, inverse = np.unique(keys, return_inverse=True)
                if len(unique_keys) < len(coords):
                    n_unique = len(unique_keys)
                    new_feats = np.zeros((n_unique, feats.shape[1]), dtype=feats.dtype)
                    np.maximum.at(new_feats, inverse, feats)
                    # Reconstruct coords from unique keys
                    new_coords = np.zeros((n_unique, 3), dtype=coords.dtype)
                    new_coords[:, 0] = (unique_keys // (H * W)).astype(coords.dtype)
                    new_coords[:, 1] = ((unique_keys % (H * W)) // W).astype(coords.dtype)
                    new_coords[:, 2] = (unique_keys % W).astype(coords.dtype)
                    coords = new_coords
                    feats = new_feats

            # Clip coordinates to valid range (safety net)
            t_oob = np.sum((coords[:, 0] < 0) | (coords[:, 0] >= T))
            y_oob = np.sum((coords[:, 1] < 0) | (coords[:, 1] >= H))
            x_oob = np.sum((coords[:, 2] < 0) | (coords[:, 2] >= W))

            if t_oob + y_oob + x_oob > 0:
                coords[:, 0] = np.clip(coords[:, 0], 0, T - 1)
                coords[:, 1] = np.clip(coords[:, 1], 0, H - 1)
                coords[:, 2] = np.clip(coords[:, 2], 0, W - 1)

            # Check for NaN/Inf in features
            if not np.all(np.isfinite(feats)):
                feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Ablation support: slice to first N channels if feature_channels is set
        if self.feature_channels is not None and feats.shape[1] > self.feature_channels:
            feats = feats[:, :self.feature_channels].copy()

        return coords, feats

    def _load_labels(self, path: Path) -> np.ndarray:
        """Load YOLO format labels."""
        if self._label_cache is not None and path in self._label_cache:
            return self._label_cache[path].copy()

        labels = []
        if path.exists():
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        labels.append([cls, cx, cy, w, h])

        labels = np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

        if self._label_cache is not None:
            self._label_cache[path] = labels.copy()

        return labels

    def _yolo_to_xyxy(self, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert YOLO format labels to xyxy boxes.

        YOLO format: [cls, cx, cy, w, h] where values are normalized [0, 1]
        Returns: boxes [N, 4] (x1, y1, x2, y2) in pixels, labels [N]
        """
        if len(labels) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        H, W = self.target_size
        classes = labels[:, 0].astype(np.int64)
        cx = labels[:, 1] * W
        cy = labels[:, 2] * H
        w = labels[:, 3] * W
        h = labels[:, 4] * H

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        return boxes, classes

    def _augment_horizontal_flip(
        self,
        coords: np.ndarray,
        feats: np.ndarray,
        boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply horizontal flip augmentation."""
        W = self.target_size[1]

        # Flip x coordinates
        coords = coords.copy()
        coords[:, 2] = W - 1 - coords[:, 2]  # x = W - 1 - x

        # Flip boxes
        if len(boxes) > 0:
            boxes = boxes.copy()
            x1, x2 = boxes[:, 0].copy(), boxes[:, 2].copy()
            boxes[:, 0] = W - x2
            boxes[:, 2] = W - x1

        return coords, feats, boxes

    def _augment_event_dropout(
        self,
        coords: np.ndarray,
        feats: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Randomly drop some voxels."""
        n_voxels = len(coords)
        keep_mask = np.random.random(n_voxels) > self.event_dropout_prob
        return coords[keep_mask], feats[keep_mask]

    def _augment_temporal_flip(
        self,
        coords: np.ndarray,
    ) -> np.ndarray:
        """Reverse temporal dimension. Doesn't affect boxes (same spatial locations)."""
        coords = coords.copy()
        T = self.time_bins
        coords[:, 0] = T - 1 - coords[:, 0]
        return coords

    def _augment_polarity_flip(
        self,
        feats: np.ndarray,
    ) -> np.ndarray:
        """Swap ON/OFF channels. Handles both 2ch and 6ch features.

        2ch: [on, off] → [off, on]
        6ch: [on_cnt, off_cnt, rec_on, rec_off, std_on, std_off]
             → [off_cnt, on_cnt, rec_off, rec_on, std_off, std_on]
        """
        feats = feats.copy()
        n_ch = feats.shape[1] if feats.ndim == 2 else 0
        if n_ch == 6:
            feats = feats[:, [1, 0, 3, 2, 5, 4]].copy()
        elif n_ch >= 2:
            feats = feats[:, ::-1].copy()
        return feats

    def _augment_spatial_affine(
        self,
        coords: np.ndarray,
        feats: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Random scale + translate on sparse coordinates and boxes."""
        H, W = self.target_size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])

        # Scale around image center
        cy, cx = H / 2.0, W / 2.0
        coords = coords.copy()
        new_y = (coords[:, 1].astype(np.float64) - cy) * scale + cy
        new_x = (coords[:, 2].astype(np.float64) - cx) * scale + cx

        # Random translate (up to 10% of image size)
        ty = random.uniform(-0.1, 0.1) * H
        tx = random.uniform(-0.1, 0.1) * W
        new_y += ty
        new_x += tx

        # Clip and filter out-of-bounds
        new_y = np.round(new_y).astype(np.int32)
        new_x = np.round(new_x).astype(np.int32)

        valid = (new_y >= 0) & (new_y < H) & (new_x >= 0) & (new_x < W)
        coords = coords[valid]
        coords[:, 1] = new_y[valid]
        coords[:, 2] = new_x[valid]
        feats = feats[valid]

        # Scale and translate boxes
        if len(boxes) > 0:
            boxes = boxes.copy()
            # Scale from center
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - cx) * scale + cx + tx
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - cy) * scale + cy + ty
            # Clip boxes
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H)
            # Remove degenerate boxes
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            keep = (widths > 2) & (heights > 2)
            boxes = boxes[keep]
            labels = labels[keep]

        return coords, feats, boxes, labels

    def _load_raw_sample(self, idx: int) -> Dict:
        """Load a sample without augmentation (for mosaic composition)."""
        sparse_path, label_path = self.samples[idx]
        coords, feats = self._load_sparse(sparse_path)
        yolo_labels = self._load_labels(label_path)
        boxes, labels = self._yolo_to_xyxy(yolo_labels)
        return {
            'coords': coords,
            'feats': feats,
            'boxes': boxes,
            'labels': labels,
        }

    def _sample_without_replacement(self, candidates: np.ndarray, k: int) -> np.ndarray:
        """Sample up to k candidate indices without replacement."""
        if k <= 0 or candidates.size == 0:
            return np.zeros((0,), dtype=np.int64)
        if candidates.size <= k:
            return candidates.astype(np.int64, copy=False)
        return np.random.choice(candidates, size=k, replace=False).astype(np.int64)

    def _weighted_subsample_indices(self, coords: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Weighted retention when voxel count exceeds cap.

        Mixes three pools:
        1) GT-inflated spatial window
        2) Temporal neighborhood around GT center proxy
        3) Global background
        """
        n_voxels = len(coords)
        max_voxels = self.max_voxels
        all_idx = np.arange(n_voxels, dtype=np.int64)
        if max_voxels <= 0 or n_voxels <= max_voxels:
            return all_idx

        if boxes is None or len(boxes) == 0:
            return self._sample_without_replacement(all_idx, max_voxels)

        # Pool 1: GT-inflated spatial window
        gt_expand = self.voxel_sampling_gt_expand
        gt_spatial_mask = np.zeros((n_voxels,), dtype=bool)
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            gt_spatial_mask |= (
                (coords[:, 1] >= (y1 - gt_expand)) &
                (coords[:, 1] <= (y2 + gt_expand)) &
                (coords[:, 2] >= (x1 - gt_expand)) &
                (coords[:, 2] <= (x2 + gt_expand))
            )
        gt_idx = all_idx[gt_spatial_mask]

        # GT center proxy (single-target dataset but kept generic for >1 boxes)
        centers_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
        centers_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
        cx = float(centers_x.mean())
        cy = float(centers_y.mean())
        if gt_idx.size > 0:
            t_center = int(round(coords[gt_idx, 0].mean()))
        else:
            t_center = int(round(coords[:, 0].mean()))

        # Pool 2: temporal neighborhood around center proxy
        temporal_mask = np.abs(coords[:, 0] - t_center) <= self.voxel_sampling_temporal_radius
        spatial_dist2 = (coords[:, 1] - cy) ** 2 + (coords[:, 2] - cx) ** 2
        spatial_mask = spatial_dist2 <= (self.voxel_sampling_spatial_radius ** 2)
        temporal_idx = all_idx[temporal_mask & spatial_mask]

        q_gt = int(round(max_voxels * self.voxel_sampling_weights["gt_window"]))
        q_temporal = int(round(max_voxels * self.voxel_sampling_weights["temporal_window"]))
        q_global = max_voxels - q_gt - q_temporal

        selected: List[np.ndarray] = []
        available = np.ones((n_voxels,), dtype=bool)

        def take_from_pool(pool: np.ndarray, quota: int) -> int:
            if quota <= 0:
                return 0
            pool = pool[available[pool]]
            taken = self._sample_without_replacement(pool, quota)
            if taken.size > 0:
                selected.append(taken)
                available[taken] = False
            return int(taken.size)

        got_gt = take_from_pool(gt_idx, q_gt)
        got_temporal = take_from_pool(temporal_idx, q_temporal)

        deficit = (q_gt - got_gt) + (q_temporal - got_temporal)
        got_global = take_from_pool(all_idx, q_global + max(deficit, 0))

        total_selected = got_gt + got_temporal + got_global
        if total_selected < max_voxels:
            remaining = all_idx[available]
            fill = self._sample_without_replacement(remaining, max_voxels - total_selected)
            if fill.size > 0:
                selected.append(fill)

        if not selected:
            return self._sample_without_replacement(all_idx, max_voxels)

        return np.concatenate(selected, axis=0)[:max_voxels]

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Returns:
            Dictionary with:
            - coords: [N, 3] int32 (t, y, x)
            - feats: [N, 2] float32 (on_count, off_count)
            - boxes: [M, 4] float32 (x1, y1, x2, y2)
            - labels: [M] int64 class labels
            - n_voxels: int
        """
        sparse_path, label_path = self.samples[idx]
        seq_id = sparse_path.parent.name
        try:
            frame_num = int(sparse_path.stem.split('_')[1])
        except Exception:
            frame_num = -1

        # Mosaic augmentation: merge 4 samples before other augmentations
        if self.augment and self.mosaic_prob > 0 and random.random() < self.mosaic_prob:
            # Load this sample + 3 random others (raw, no augmentation)
            indices = [idx] + random.choices(range(len(self)), k=3)
            mosaic_samples = [self._load_raw_sample(i) for i in indices]
            merged = sparse_mosaic(
                mosaic_samples,
                target_size=self.target_size,
                time_bins=self.time_bins,
            )
            coords, feats = merged['coords'], merged['feats']
            boxes, labels = merged['boxes'], merged['labels']
        else:
            # Load sparse data
            coords, feats = self._load_sparse(sparse_path)

            # Load labels
            yolo_labels = self._load_labels(label_path)
            boxes, labels = self._yolo_to_xyxy(yolo_labels)

        # Augmentation
        if self.augment:
            # Horizontal flip
            if random.random() < self.horizontal_flip_prob:
                coords, feats, boxes = self._augment_horizontal_flip(coords, feats, boxes)

            # Temporal flip (reverse time dimension)
            if self.temporal_flip_prob > 0 and random.random() < self.temporal_flip_prob:
                coords = self._augment_temporal_flip(coords)

            # Polarity flip (swap ON/OFF channels)
            if self.polarity_flip_prob > 0 and random.random() < self.polarity_flip_prob:
                feats = self._augment_polarity_flip(feats)

            # Spatial affine (scale + translate)
            if self.scale_range != (1.0, 1.0):
                coords, feats, boxes, labels = self._augment_spatial_affine(coords, feats, boxes, labels)

            # Event dropout
            if self.event_dropout_prob > 0:
                coords, feats = self._augment_event_dropout(coords, feats)

        # Coalesce duplicate coordinates (augmentation can create them via rounding).
        # spconv does NOT coalesce — duplicates double feature magnitude at that cell.
        if len(coords) > 1:
            # Lexicographic hash for fast duplicate detection
            T, H, W = self.spatial_shape
            keys = coords[:, 0].astype(np.int64) * (H * W) + coords[:, 1].astype(np.int64) * W + coords[:, 2].astype(np.int64)
            _, unique_idx = np.unique(keys, return_index=True)
            if len(unique_idx) < len(coords):
                coords = coords[unique_idx]
                feats = feats[unique_idx]

        raw_n_voxels = int(len(coords))
        clipped = False
        if self.max_voxels > 0 and len(coords) > self.max_voxels:
            clipped = True
            if self.voxel_sampling_mode == "weighted":
                indices = self._weighted_subsample_indices(coords, boxes)
            else:
                indices = self._sample_without_replacement(
                    np.arange(len(coords), dtype=np.int64),
                    self.max_voxels
                )
            coords = coords[indices]
            feats = feats[indices]
        kept_n_voxels = int(len(coords))
        clip_fraction = 0.0
        if raw_n_voxels > 0 and clipped:
            clip_fraction = 1.0 - (kept_n_voxels / float(raw_n_voxels))

        # Handle empty case — clear GT too to prevent hallucinated targets
        if len(coords) == 0:
            n_feat_ch = self.feature_channels if self.feature_channels else feats.shape[1] if len(feats) > 0 else 6
            coords = np.zeros((1, 3), dtype=np.int32)
            feats = np.zeros((1, n_feat_ch), dtype=np.float32)
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        return {
            'coords': torch.from_numpy(coords),
            'feats': torch.from_numpy(feats),
            'boxes': torch.from_numpy(boxes),
            'labels': torch.from_numpy(labels),
            'n_voxels': len(coords),
            'sample_id': f"{seq_id}/{sparse_path.stem}",
            'seq_id': seq_id,
            'frame_num': frame_num,
            'raw_n_voxels': raw_n_voxels,
            'kept_n_voxels': kept_n_voxels,
            'clip_fraction': float(clip_fraction),
            'clipped': bool(clipped),
            'voxel_sampling_mode': self.voxel_sampling_mode,
        }


def _collate_batch(
    batch: List[Dict],
    time_bins: Optional[int] = None,
    target_size: Tuple[int, int] = (720, 1280),
    base_size: Tuple[int, int] = (720, 1280),
) -> Dict:
    """
    Core collation logic for sparse event batches.

    V82: supports rectangular (H, W) target_size and base_size.

    Args:
        batch: List of sample dictionaries
        time_bins: Explicit time_bins. If None, inferred from data.
        target_size: (H, W) spatial size for this batch.
        base_size: (H, W) original spatial size the data was generated at.

    Returns:
        Dictionary with coords, feats, spatial_shape, gt_boxes, gt_labels, batch_size
    """
    batch_size = len(batch)
    # Ensure tuples
    if isinstance(target_size, (int, float)):
        target_size = (int(target_size), int(target_size))
    if isinstance(base_size, (int, float)):
        base_size = (int(base_size), int(base_size))
    target_H, target_W = target_size
    base_H, base_W = base_size
    scale_y = target_H / base_H
    scale_x = target_W / base_W
    needs_scale = abs(scale_y - 1.0) > 1e-6 or abs(scale_x - 1.0) > 1e-6

    # Collect all coordinates and features
    all_coords = []
    all_feats = []
    gt_boxes = []
    gt_labels = []
    sample_ids: List[str] = []
    raw_voxels: List[int] = []
    kept_voxels: List[int] = []
    clip_fractions: List[float] = []
    clipped_flags: List[bool] = []
    sampling_modes: List[str] = []
    seq_ids: List[str] = []
    frame_nums: List[int] = []

    for batch_idx, sample in enumerate(batch):
        coords = sample['coords'].clone()  # [N_i, 3] = (t, y, x)
        feats = sample['feats']            # [N_i, C] (C=2 or 6)
        boxes = sample['boxes'].clone()    # [M, 4]

        # Scale spatial coordinates if multi-scale
        if needs_scale:
            # Scale y and x independently
            coords[:, 1] = (coords[:, 1].float() * scale_y).int()
            coords[:, 2] = (coords[:, 2].float() * scale_x).int()
            coords[:, 1] = coords[:, 1].clamp(0, target_H - 1)
            coords[:, 2] = coords[:, 2].clamp(0, target_W - 1)
            # Scale boxes (x1, y1, x2, y2) — x scales with scale_x, y with scale_y
            if len(boxes) > 0:
                boxes[:, 0] *= scale_x  # x1
                boxes[:, 1] *= scale_y  # y1
                boxes[:, 2] *= scale_x  # x2
                boxes[:, 3] *= scale_y  # y2

        # Add batch index as first column
        batch_col = torch.full((coords.shape[0], 1), batch_idx, dtype=torch.int32)
        coords_with_batch = torch.cat([batch_col, coords], dim=1)  # [N_i, 4] = (b, t, y, x)

        all_coords.append(coords_with_batch)
        all_feats.append(feats)

        gt_boxes.append(boxes)
        gt_labels.append(sample['labels'])
        sample_ids.append(str(sample.get('sample_id', f"sample_{batch_idx:06d}")))
        raw_voxels.append(int(sample.get('raw_n_voxels', int(coords.shape[0]))))
        kept_voxels.append(int(sample.get('kept_n_voxels', int(coords.shape[0]))))
        clip_fractions.append(float(sample.get('clip_fraction', 0.0)))
        clipped_flags.append(bool(sample.get('clipped', False)))
        sampling_modes.append(str(sample.get('voxel_sampling_mode', 'unknown')))
        seq_ids.append(str(sample.get('seq_id', 'unknown')))
        frame_nums.append(int(sample.get('frame_num', -1)))

    # Concatenate
    coords = torch.cat(all_coords, dim=0)  # [N_total, 4]
    feats = torch.cat(all_feats, dim=0)    # [N_total, C]

    # Determine time_bins
    if time_bins is None:
        if coords.shape[0] > 0:
            time_bins = coords[:, 1].max().item() + 1
        else:
            time_bins = 16

    # Spatial shape: [T, H, W] — supports rectangular
    spatial_shape = [time_bins, target_H, target_W]

    return {
        'coords': coords,
        'feats': feats,
        'spatial_shape': spatial_shape,
        'gt_boxes': gt_boxes,
        'gt_labels': gt_labels,
        'sample_ids': sample_ids,
        'seq_ids': seq_ids,
        'frame_nums': frame_nums,
        'clip_telemetry': {
            'raw_voxels': raw_voxels,
            'kept_voxels': kept_voxels,
            'clip_fraction': clip_fractions,
            'clipped': clipped_flags,
            'sampling_mode': sampling_modes,
            'seq_id': seq_ids,
            'frame_num': frame_nums,
        },
        'batch_size': batch_size,
        'input_size': target_size,
    }


def sparse_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for sparse event batches (backward-compatible, infers time_bins).

    Use make_collate_fn() for explicit time_bins control.
    """
    return _collate_batch(batch, time_bins=None)


def make_collate_fn(
    time_bins: int = 16,
    multi_scale_sizes: Optional[List[int]] = None,
    base_size: Tuple[int, int] = (720, 1280),
):
    """
    Create collate function with explicit time_bins and optional multi-scale.

    V83: supports rectangular multi-scale. multi_scale_sizes is interpreted as
    a list of HEIGHT values. Width is computed proportionally to maintain the
    base aspect ratio.  For square base_size, behaves as before (sizes = H = W).

    Args:
        time_bins: Number of temporal bins (must match dataset config)
        multi_scale_sizes: List of height values for multi-scale training.
            Width is auto-computed to maintain aspect ratio.  None = fixed size.
        base_size: (H, W) tuple. For backward compat, also accepts int (square).

    Returns:
        Collate function for DataLoader
    """
    if isinstance(base_size, (int, float)):
        base_size = (int(base_size), int(base_size))

    base_H, base_W = base_size
    is_square = (base_H == base_W)
    aspect = base_W / base_H  # e.g. 1280/720 ≈ 1.778

    def collate_fn(batch: List[Dict]) -> Dict:
        if multi_scale_sizes is not None and len(multi_scale_sizes) > 0:
            chosen_h = random.choice(multi_scale_sizes)
            if is_square:
                target_size = (chosen_h, chosen_h)
            else:
                # Proportional width, maintain aspect ratio
                chosen_w = int(round(chosen_h * aspect))
                target_size = (chosen_h, chosen_w)
        else:
            target_size = base_size
        return _collate_batch(batch, time_bins=time_bins,
                              target_size=target_size, base_size=base_size)
    return collate_fn


def create_sparse_tensor(batch: Dict, device: torch.device) -> spconv.SparseConvTensor:
    """
    Create SparseConvTensor from collated batch.

    Args:
        batch: Output from sparse_collate_fn
        device: Target device

    Returns:
        SparseConvTensor ready for model
    """
    return spconv.SparseConvTensor(
        features=batch['feats'].to(device),
        indices=batch['coords'].to(device),
        spatial_shape=batch['spatial_shape'],
        batch_size=batch['batch_size']
    )


def test_dataset():
    """Test the sparse event dataset."""
    print("Testing SparseEventDataset...")

    # These paths are examples - adjust for your setup
    project_root = Path(__file__).parent.parent.parent
    sparse_dir = project_root / 'datasets' / 'fred_sparse'
    label_dir = project_root / 'datasets' / 'fred_voxel' / 'labels'

    # Check if data exists
    if not sparse_dir.exists():
        print(f"Sparse dir not found: {sparse_dir}")
        print("Creating synthetic test data...")

        # Create synthetic test data
        test_dir = Path('/tmp/test_sparse_dataset')
        sparse_test = test_dir / 'sparse' / 'train' / 'seq_001'
        label_test = test_dir / 'labels' / 'train'
        sparse_test.mkdir(parents=True, exist_ok=True)
        label_test.mkdir(parents=True, exist_ok=True)

        # Create synthetic samples
        for i in range(5):
            # Sparse data
            n_voxels = np.random.randint(1000, 5000)
            coords = np.random.randint(0, [33, 640, 640], size=(n_voxels, 3)).astype(np.int16)
            feats = np.random.rand(n_voxels, 2).astype(np.float16)
            np.savez_compressed(sparse_test / f'frame_{i:06d}.npz', coords=coords, feats=feats)

            # Labels (YOLO format)
            n_boxes = np.random.randint(1, 4)
            with open(label_test / f'seq_001_frame_{i:06d}.txt', 'w') as f:
                for _ in range(n_boxes):
                    cx, cy = np.random.rand(2)
                    w, h = np.random.rand(2) * 0.1 + 0.02
                    f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        # Create manifest
        manifest = {'split': 'train', 'n_frames': 5}
        with open(test_dir / 'sparse' / 'train' / 'manifest.json', 'w') as f:
            json.dump(manifest, f)

        sparse_dir = test_dir / 'sparse'
        label_dir = test_dir / 'labels'

    # Create dataset
    dataset = SparseEventDataset(
        sparse_dir=str(sparse_dir),
        label_dir=str(label_dir),
        split='train',
        augment=True
    )

    print(f"Dataset size: {len(dataset)}")

    # Test single sample
    sample = dataset[0]
    print(f"\nSingle sample:")
    print(f"  coords: {sample['coords'].shape}")
    print(f"  feats: {sample['feats'].shape}")
    print(f"  boxes: {sample['boxes'].shape}")
    print(f"  labels: {sample['labels'].shape}")
    print(f"  n_voxels: {sample['n_voxels']}")

    # Test DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=sparse_collate_fn
    )

    batch = next(iter(dataloader))
    print(f"\nBatch:")
    print(f"  coords: {batch['coords'].shape}")
    print(f"  feats: {batch['feats'].shape}")
    print(f"  gt_boxes: {[b.shape for b in batch['gt_boxes']]}")
    print(f"  batch_size: {batch['batch_size']}")

    # Test sparse tensor creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sparse_tensor = create_sparse_tensor(batch, device)
    print(f"\nSparseConvTensor:")
    print(f"  features: {sparse_tensor.features.shape}")
    print(f"  indices: {sparse_tensor.indices.shape}")
    print(f"  spatial_shape: {sparse_tensor.spatial_shape}")

    print("\nSUCCESS: SparseEventDataset working!")


if __name__ == '__main__':
    test_dataset()
