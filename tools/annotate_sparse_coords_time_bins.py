#!/usr/bin/env python3
"""Annotate coords-format sparse npz files with explicit `time_bins` metadata.

This is intended to support strict temporal-bin contract checks for coords files.
It rewrites only files that contain both `coords` and `feats`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


DEFAULT_SPLITS = (
    "canonical_train",
    "canonical_test",
    "challenging_train",
    "challenging_test",
)


def iter_npz_files(root: Path, splits: Iterable[str]) -> Iterable[Path]:
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for p in sorted(split_dir.glob("*/*.npz")):
            yield p


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate coords sparse npz files with time_bins metadata.")
    parser.add_argument(
        "--sparse-root",
        type=Path,
        default=Path("data/datasets/fred_paper_parity/sparse"),
        help="Root containing split/seq/frame_XXXXXX.npz.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=list(DEFAULT_SPLITS),
        help="Splits to process.",
    )
    parser.add_argument(
        "--time-bins",
        type=int,
        required=True,
        help="Expected source time bin count to write into each coords file.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing time_bins value if present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report changes without rewriting files.",
    )
    args = parser.parse_args()

    expected_tb = int(args.time_bins)
    if expected_tb <= 0:
        raise ValueError("--time-bins must be > 0")

    stats: Dict[str, int] = {
        "total_npz_seen": 0,
        "coords_files": 0,
        "updated": 0,
        "already_ok": 0,
        "skipped_existing": 0,
        "non_coords_files": 0,
        "errors": 0,
    }

    for npz_path in iter_npz_files(args.sparse_root, args.splits):
        stats["total_npz_seen"] += 1
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                keys = list(data.files)
                payload = {k: data[k] for k in keys}
        except Exception:
            stats["errors"] += 1
            continue

        if not (("coords" in payload) and ("feats" in payload)):
            stats["non_coords_files"] += 1
            continue

        stats["coords_files"] += 1
        existing = payload.get("time_bins", None)
        existing_tb = None
        if existing is not None:
            try:
                existing_tb = int(np.asarray(existing).item())
            except Exception:
                existing_tb = None

        if existing_tb == expected_tb:
            stats["already_ok"] += 1
            continue

        if existing is not None and not args.overwrite_existing:
            stats["skipped_existing"] += 1
            continue

        payload["time_bins"] = np.asarray(expected_tb, dtype=np.int32)
        stats["updated"] += 1
        if args.dry_run:
            continue

        tmp_path = npz_path.with_suffix(npz_path.suffix + ".tmp")
        np.savez_compressed(tmp_path, **payload)
        tmp_path.replace(npz_path)

    print("annotate_sparse_coords_time_bins summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  sparse_root: {args.sparse_root}")
    print(f"  splits: {list(args.splits)}")
    print(f"  target_time_bins: {expected_tb}")
    print(f"  dry_run: {bool(args.dry_run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
