#!/usr/bin/env python3
"""Build official FRED parity roots with multi-object labels from coordinates.txt.

This script materializes:
  - labels per split from official coordinates + official split manifests
  - sparse links from existing sparse roots
  - optional sparse generation for missing frames from raw events.hdf5

Output layout:
  out_root/
    labels/{canonical_train,canonical_test,challenging_train,challenging_test}/*.txt
    sparse/{canonical_train,canonical_test,challenging_train,challenging_test}/{seq}/frame_XXXXXX.npz
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

# Reuse the existing encoder to keep preprocessing behavior aligned.
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str((PROJECT_ROOT / "archive" / "scripts_v9" / "snn").resolve()))
from event_spike_encoder import EventSpikeEncoder  # type: ignore


FRAME_RE = re.compile(r"frame_(?P<ts>\d+)\.png$")


@dataclass(frozen=True)
class SplitSpec:
    name: str
    txt_path: Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_timestamp(frame_idx: int) -> str:
    # Match official FRED repo logic exactly.
    return str(float(f"{(frame_idx + 1) * 0.033333:.6f}"))


def load_split_ids(path: Path) -> List[str]:
    rows: List[str] = []
    for line in path.read_text().splitlines():
        tok = line.strip().strip("/")
        if tok:
            rows.append(tok)
    return sorted(set(rows), key=lambda x: int(x))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        for p in sorted(path.rglob("*"), reverse=True):
            if p.is_symlink() or p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        if path.exists():
            path.rmdir()
    path.mkdir(parents=True, exist_ok=True)


def replace_symlink(dst: Path, src: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def resolve_seq_dir(seq_id: str, train_root: Path, test_root: Path) -> Optional[Path]:
    tr = train_root / seq_id
    if tr.exists():
        return tr
    te = test_root / seq_id
    if te.exists():
        return te
    return None


def load_coordinates(coord_path: Path) -> Dict[str, List[Tuple[float, float, float, float]]]:
    out: Dict[str, List[Tuple[float, float, float, float]]] = {}
    if not coord_path.exists():
        return out
    for line in coord_path.read_text().splitlines():
        if ": " not in line:
            continue
        ts, rest = line.strip().split(": ", 1)
        vals = [v.strip() for v in rest.split(", ")]
        if len(vals) < 6:
            continue
        x1, y1, x2, y2 = map(float, vals[:4])
        out.setdefault(ts, []).append((x1, y1, x2, y2))
    return out


def count_frames(seq_dir: Path) -> int:
    frames_dir = seq_dir / "Event" / "Frames"
    if not frames_dir.exists():
        return 0
    ts_values: List[int] = []
    for p in frames_dir.glob("*.png"):
        m = FRAME_RE.search(p.name)
        if not m:
            continue
        ts_values.append(int(m.group("ts")))
    if not ts_values:
        return 0
    return len(ts_values)


def find_sparse_source(
    seq_id: str,
    frame_idx: int,
    flat_roots: Sequence[Path],
    tree_roots: Sequence[Path],
) -> Optional[Path]:
    fname = f"{seq_id}_frame_{frame_idx:06d}.npz"
    frame_name = f"frame_{frame_idx:06d}.npz"
    for root in flat_roots:
        for split in ("train", "val", "test"):
            p = root / split / fname
            if p.exists():
                return p
    for root in tree_roots:
        for split in ("train", "val", "test"):
            p = root / split / seq_id / frame_name
            if p.exists():
                return p
    return None


def load_events(seq_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    h5_path = seq_dir / "Event" / "events.hdf5"
    if not h5_path.exists():
        return None
    with h5py.File(h5_path, "r") as f:
        if "CD" in f and "events" in f["CD"]:
            events = f["CD"]["events"][:]
        elif "events" in f:
            events = f["events"][:]
        else:
            return None
    x = events["x"].astype(np.int32)
    y = events["y"].astype(np.int32)
    t = events["t"].astype(np.int64)
    p = events["p"].astype(np.int32)
    return x, y, t, p


def build_labels_for_sequence(
    seq_id: str,
    seq_dir: Path,
    label_split_dir: Path,
    sensor_w: int,
    sensor_h: int,
    clip_boxes: bool,
    write: bool,
) -> Tuple[List[int], int, int]:
    """Return (frame_idxs_with_labels, dropped_boxes, raw_box_count)."""
    coords = load_coordinates(seq_dir / "coordinates.txt")
    n_frames = count_frames(seq_dir)
    ensure_dir(label_split_dir)

    labeled_idxs: List[int] = []
    dropped = 0
    raw_boxes = 0

    for frame_idx in range(n_frames):
        ts = to_timestamp(frame_idx)
        boxes = coords.get(ts, [])
        if not boxes:
            continue

        lines: List[str] = []
        for x1, y1, x2, y2 in boxes:
            raw_boxes += 1
            if clip_boxes:
                x1 = max(0.0, min(x1, float(sensor_w)))
                x2 = max(0.0, min(x2, float(sensor_w)))
                y1 = max(0.0, min(y1, float(sensor_h)))
                y2 = max(0.0, min(y2, float(sensor_h)))
            if x2 <= x1 or y2 <= y1:
                dropped += 1
                continue
            cx = ((x1 + x2) / 2.0) / float(sensor_w)
            cy = ((y1 + y2) / 2.0) / float(sensor_h)
            w = (x2 - x1) / float(sensor_w)
            h = (y2 - y1) / float(sensor_h)
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not lines:
            continue

        labeled_idxs.append(frame_idx)
        if write:
            out = label_split_dir / f"{seq_id}_frame_{frame_idx:06d}.txt"
            out.write_text("\n".join(lines) + "\n")

    return labeled_idxs, dropped, raw_boxes


def maybe_generate_sparse(
    seq_id: str,
    seq_dir: Path,
    missing_idxs: Iterable[int],
    sparse_seq_dir: Path,
    sensor_w: int,
    sensor_h: int,
    target_w: int,
    target_h: int,
    time_bins: int,
    window_ms: float,
    fixed_scale_max: float,
    binary: bool,
    write: bool,
) -> int:
    idxs = sorted(set(int(x) for x in missing_idxs))
    if not idxs:
        return 0
    events = load_events(seq_dir)
    if events is None:
        return 0
    x, y, t, p = events
    if len(t) == 0:
        return 0

    # Keep arrays time-sorted and use searchsorted windows for per-frame slicing.
    order = np.argsort(t, kind="stable")
    t = t[order]
    x = x[order]
    y = y[order]
    p = p[order]
    t_min = int(t[0])
    frame_duration_us = 33333
    window_us = int(window_ms * 1000.0)

    enc = EventSpikeEncoder(
        height=target_h,
        width=target_w,
        time_window_ms=window_ms,
        num_bins=time_bins,
        binary_mode=binary,
        fixed_scale_max=fixed_scale_max,
    )
    sx = float(target_w) / float(sensor_w)
    sy = float(target_h) / float(sensor_h)
    ensure_dir(sparse_seq_dir)

    done = 0
    for frame_idx in idxs:
        frame_end = t_min + (frame_idx + 1) * frame_duration_us
        frame_start = frame_end - window_us
        i0 = int(np.searchsorted(t, frame_start, side="left"))
        i1 = int(np.searchsorted(t, frame_end, side="left"))
        xw = x[i0:i1]
        yw = y[i0:i1]
        tw = t[i0:i1]
        pw = p[i0:i1]
        spikes = enc.encode(xw, yw, tw, pw, t_start=frame_start, t_end=frame_end, scale_x=sx, scale_y=sy)
        if write:
            out = sparse_seq_dir / f"frame_{frame_idx:06d}.npz"
            np.savez_compressed(out, spikes=spikes.numpy().astype(np.float16))
        done += 1
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description="Build official FRED multi-object parity roots.")
    parser.add_argument("--official-splits-root", type=Path, default=Path("paper/artifacts/splits"))
    parser.add_argument("--raw-train-root", type=Path, default=Path("data/processed/FRED/train"))
    parser.add_argument("--raw-test-root", type=Path, default=Path("data/processed/FRED/test"))
    parser.add_argument(
        "--flat-sparse-root",
        type=Path,
        nargs="*",
        default=[Path("data/processed/FRED/spikes_v6_clean")],
    )
    parser.add_argument(
        "--tree-sparse-root",
        type=Path,
        nargs="*",
        default=[
            Path("data/datasets/fred_sparse"),
            Path("data/datasets/fred_sparse_missing"),
            Path("data/datasets/fred_sparse_missing_train_rest"),
        ],
    )
    parser.add_argument("--out-root", type=Path, default=Path("data/datasets/fred_paper_parity_official"))
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("paper/presentation_advisor/snapshot/split_parity_official_build.json"),
    )
    parser.add_argument("--clip-boxes", action="store_true", default=True)
    parser.add_argument("--no-clip-boxes", dest="clip_boxes", action="store_false")
    parser.add_argument("--sensor-h", type=int, default=720)
    parser.add_argument("--sensor-w", type=int, default=1280)
    parser.add_argument("--target-h", type=int, default=640)
    parser.add_argument("--target-w", type=int, default=640)
    parser.add_argument("--generate-missing-sparse", action="store_true")
    parser.add_argument("--generate-only-seq", type=str, nargs="*", default=[])
    parser.add_argument("--gen-time-bins", type=int, default=10)
    parser.add_argument("--gen-window-ms", type=float, default=33.33)
    parser.add_argument("--gen-fixed-scale-max", type=float, default=500.0)
    parser.add_argument("--gen-binary", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    split_specs = [
        SplitSpec("canonical_train", args.official_splits_root / "canonical" / "train_split.txt"),
        SplitSpec("canonical_test", args.official_splits_root / "canonical" / "test_split.txt"),
        SplitSpec("challenging_train", args.official_splits_root / "challenging" / "challenging_train_split.txt"),
        SplitSpec("challenging_test", args.official_splits_root / "challenging" / "challenging_test_split.txt"),
    ]
    for spec in split_specs:
        if not spec.txt_path.exists():
            raise FileNotFoundError(f"Missing split file: {spec.txt_path}")

    out_sparse = args.out_root / "sparse"
    out_labels = args.out_root / "labels"
    if args.clean and not args.dry_run:
        reset_dir(out_sparse)
        reset_dir(out_labels)
    elif not args.dry_run:
        ensure_dir(out_sparse)
        ensure_dir(out_labels)

    only_seq = set(args.generate_only_seq)
    report: Dict[str, object] = {
        "generated_at": now_iso(),
        "out_root": str(args.out_root),
        "clip_boxes": bool(args.clip_boxes),
        "dry_run": bool(args.dry_run),
        "generate_missing_sparse": bool(args.generate_missing_sparse),
        "split_stats": {},
    }

    for spec in split_specs:
        ids = load_split_ids(spec.txt_path)
        split_stat = {
            "official_ids": len(ids),
            "seq_found": 0,
            "seq_missing_raw": [],
            "label_files": 0,
            "label_boxes_raw": 0,
            "label_boxes_dropped": 0,
            "sparse_linked": 0,
            "sparse_generated": 0,
            "sparse_missing": 0,
            "missing_sparse_by_seq": {},
        }
        if not args.dry_run:
            ensure_dir(out_sparse / spec.name)
            ensure_dir(out_labels / spec.name)

        for seq_id in ids:
            seq_dir = resolve_seq_dir(seq_id, args.raw_train_root, args.raw_test_root)
            if seq_dir is None:
                split_stat["seq_missing_raw"].append(seq_id)
                continue
            split_stat["seq_found"] += 1

            label_split_dir = out_labels / spec.name
            labeled_idxs, dropped_boxes, raw_box_count = build_labels_for_sequence(
                seq_id=seq_id,
                seq_dir=seq_dir,
                label_split_dir=label_split_dir,
                sensor_w=args.sensor_w,
                sensor_h=args.sensor_h,
                clip_boxes=bool(args.clip_boxes),
                write=not args.dry_run,
            )
            split_stat["label_files"] += len(labeled_idxs)
            split_stat["label_boxes_raw"] += raw_box_count
            split_stat["label_boxes_dropped"] += dropped_boxes

            sparse_seq_dir = out_sparse / spec.name / seq_id
            missing_idxs: List[int] = []
            for frame_idx in labeled_idxs:
                src = find_sparse_source(seq_id, frame_idx, args.flat_sparse_root, args.tree_sparse_root)
                if src is None:
                    missing_idxs.append(frame_idx)
                    continue
                split_stat["sparse_linked"] += 1
                if not args.dry_run:
                    ensure_dir(sparse_seq_dir)
                    replace_symlink(sparse_seq_dir / f"frame_{frame_idx:06d}.npz", src)

            if missing_idxs:
                split_stat["sparse_missing"] += len(missing_idxs)
                split_stat["missing_sparse_by_seq"][seq_id] = len(missing_idxs)
                if args.generate_missing_sparse and (not only_seq or seq_id in only_seq):
                    generated = maybe_generate_sparse(
                        seq_id=seq_id,
                        seq_dir=seq_dir,
                        missing_idxs=missing_idxs,
                        sparse_seq_dir=sparse_seq_dir,
                        sensor_w=args.sensor_w,
                        sensor_h=args.sensor_h,
                        target_w=args.target_w,
                        target_h=args.target_h,
                        time_bins=args.gen_time_bins,
                        window_ms=args.gen_window_ms,
                        fixed_scale_max=args.gen_fixed_scale_max,
                        binary=bool(args.gen_binary),
                        write=not args.dry_run,
                    )
                    split_stat["sparse_generated"] += generated
                    split_stat["sparse_missing"] -= generated
                    if generated >= len(missing_idxs):
                        split_stat["missing_sparse_by_seq"].pop(seq_id, None)
                    else:
                        split_stat["missing_sparse_by_seq"][seq_id] = len(missing_idxs) - generated

        report["split_stats"][spec.name] = split_stat

    ensure_dir(args.report_json.parent)
    args.report_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
