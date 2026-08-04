#!/usr/bin/env python3
"""Build and exhaustively verify the canonical-val YOLO event-frame view."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset"
SPLIT_ROOT = PROJECT / "paper/artifacts/splits/canonical_val"
OFFICIAL_TRAIN = PROJECT / "paper/artifacts/splits/canonical/train_split.txt"
SPARSE_ROOT = PROJECT / "data/datasets/fred_paper_parity_v82_640/sparse"
LABEL_ROOT = PROJECT / "data/datasets/fred_paper_parity/labels"
EVENT_ROOT = PROJECT / "data/processed/FRED"
EXPECTED = {
    "train": {"sparse_split": "canonical_val_train", "split_file": "train_split.txt", "split_sha256": "f50b6335860c06c062fe0f295bd42b449f97ab92da71f8448fda9842f706490c", "sequences": 147, "frames": 406701},
    "val": {"sparse_split": "canonical_val", "split_file": "val_split.txt", "split_sha256": "1f0f1ee72b07b55be5f5c5a561a0e3d649fb5c859d56a10eeb414ea134d53bb9", "sequences": 37, "frames": 103672},
}
EVENT_TS = re.compile(r"_(\d+)\.png$")
SPARSE_FRAME = re.compile(r"frame_(\d+)\.npz$")
PNG_TYPES = {0: "grayscale", 2: "RGB", 3: "palette", 4: "grayscale_alpha", 6: "RGBA"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_ids(path: Path) -> list[str]:
    ids = [line.strip().strip("/") for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)) or any(not x.isdigit() for x in ids):
        raise ValueError(f"Invalid or duplicate sequence ID in {path}")
    return sorted(ids, key=int)


def event_frames(seq: str) -> tuple[Path, list[Path]]:
    candidates = [EVENT_ROOT / part / seq / "Event/Frames" for part in ("train", "test")]
    found = [p for p in candidates if p.is_dir()]
    if len(found) != 1:
        raise RuntimeError(f"Expected exactly one event source for seq {seq}, found {found}")
    if found[0].parts[-4] != "train":
        raise RuntimeError(f"canonical-val sequence {seq} resolved outside FRED train: {found[0]}")
    rows = []
    for entry in os.scandir(found[0]):
        if not entry.is_file() or not entry.name.endswith(".png"):
            continue
        match = EVENT_TS.search(entry.name)
        if match is None:
            raise RuntimeError(f"Unparseable event filename: {entry.path}")
        rows.append((int(match.group(1)), Path(entry.path)))
    rows.sort(key=lambda row: row[0])
    if not rows or len({ts for ts, _ in rows}) != len(rows):
        raise RuntimeError(f"Empty or duplicate event timestamps in {found[0]}")
    return found[0], [path for _, path in rows]


def sparse_frames(split: str, seq: str) -> list[tuple[int, Path]]:
    seq_dir = SPARSE_ROOT / split / seq
    rows = []
    for path in seq_dir.glob("frame_*.npz"):
        match = SPARSE_FRAME.fullmatch(path.name)
        if match:
            rows.append((int(match.group(1)), path))
    rows.sort(key=lambda row: row[0])
    if not rows or len({idx for idx, _ in rows}) != len(rows):
        raise RuntimeError(f"Empty or duplicate sparse indices in {seq_dir}")
    return rows


def png_contract(path: Path) -> tuple[int, int, int, int]:
    with path.open("rb") as f:
        header = f.read(29)
    if len(header) != 29 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"Invalid PNG header: {path}")
    width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", header[16:29])
    return width, height, bit_depth, color_type


def parse_label(path: Path) -> tuple[int, tuple[float, float, float, float]]:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"Empty label file: {path}")
    extrema = [math.inf, math.inf, -math.inf, -math.inf]
    for line_no, row in enumerate(rows, 1):
        if len(row) != 5:
            raise ValueError(f"Expected 5 YOLO fields at {path}:{line_no}, got {len(row)}")
        cls, vals = int(row[0]), [float(x) for x in row[1:]]
        if cls != 0 or not all(math.isfinite(x) for x in vals):
            raise ValueError(f"Invalid class/coordinate at {path}:{line_no}: {row}")
        cx, cy, width, height = vals
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Out-of-range YOLO box at {path}:{line_no}: {row}")
        extrema[0], extrema[1] = min(extrema[0], cx), min(extrema[1], cy)
        extrema[2], extrema[3] = max(extrema[2], width), max(extrema[3], height)
    return len(rows), tuple(extrema)


def link_exact(source: Path, destination: Path, check_only: bool) -> None:
    source = source.resolve()
    if os.path.lexists(destination):
        if not destination.is_symlink() or destination.resolve() != source:
            raise RuntimeError(f"Refusing to replace non-matching path: {destination}")
        return
    if check_only:
        raise FileNotFoundError(f"Missing prepared link: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def write_exact(path: Path, content: str, check_only: bool) -> None:
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"Refusing to overwrite non-matching generated file: {path}")
        return
    if check_only:
        raise FileNotFoundError(f"Missing prepared file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-png-header-scan", action="store_true", help="Development only; full audit scans every selected PNG.")
    args = parser.parse_args()
    official_ids = set(read_ids(OFFICIAL_TRAIN))
    split_ids = {name: read_ids(SPLIT_ROOT / spec["split_file"]) for name, spec in EXPECTED.items()}
    if set(split_ids["train"]) & set(split_ids["val"]):
        raise RuntimeError("Train/val sequence leakage")
    if set(split_ids["train"]) | set(split_ids["val"]) != official_ids:
        raise RuntimeError("Train/val union does not equal official canonical train")
    audit: dict[str, object] = {
        "status": "ok",
        "protocol": "FRED official canonical-train only; random.Random(42) sequence carve-out",
        "source_roots": {"event_png": str(EVENT_ROOT / "train/<seq>/Event/Frames"), "sparse_frame_index": str(SPARSE_ROOT), "labels": str(LABEL_ROOT)},
        "official_canonical_train_sequences": len(official_ids),
        "train_val_sequence_overlap": 0,
        "canonical_test_used": False,
        "splits": {},
    }
    selected_sources: dict[str, set[str]] = {"train": set(), "val": set()}
    for output_split, spec in EXPECTED.items():
        split_path = SPLIT_ROOT / spec["split_file"]
        if sha256_file(split_path) != spec["split_sha256"]:
            raise RuntimeError(f"Split hash mismatch: {split_path}")
        ids = split_ids[output_split]
        if len(ids) != spec["sequences"]:
            raise RuntimeError(f"Sequence-count mismatch for {output_split}: {len(ids)}")
        image_lines: list[str] = []
        mapping_hash = hashlib.sha256()
        frame_count = object_count = multi_object_count = raw_event_count = 0
        image_contracts: dict[str, int] = {}
        sequence_frame_counts: dict[str, int] = {}
        cx_min = cy_min = math.inf
        width_max = height_max = -math.inf
        for seq in ids:
            sparse_split = str(spec["sparse_split"])
            indexed_sparse = sparse_frames(sparse_split, seq)
            _, ordered_events = event_frames(seq)
            raw_event_count += len(ordered_events)
            sequence_frame_counts[seq] = len(indexed_sparse)
            label_source_dir = LABEL_ROOT / sparse_split
            link_exact(label_source_dir, DATASET / "labels" / output_split / seq, args.check_only)
            for frame_idx, sparse_path in indexed_sparse:
                if frame_idx >= len(ordered_events):
                    raise IndexError(f"{seq}/frame_{frame_idx:06d} exceeds {len(ordered_events)} event PNGs")
                event_source = ordered_events[frame_idx]
                label_source = label_source_dir / f"{seq}_frame_{frame_idx:06d}.txt"
                if not label_source.exists():
                    raise FileNotFoundError(f"Missing paired label: {label_source}")
                n_objects, extrema = parse_label(label_source)
                object_count += n_objects
                multi_object_count += int(n_objects > 1)
                cx_min, cy_min = min(cx_min, extrema[0]), min(cy_min, extrema[1])
                width_max, height_max = max(width_max, extrema[2]), max(height_max, extrema[3])
                if not args.skip_png_header_scan:
                    width, height, depth, color_type = png_contract(event_source)
                    contract = f"{width}x{height}|bit{depth}|{PNG_TYPES.get(color_type, f'type{color_type}')}"
                    image_contracts[contract] = image_contracts.get(contract, 0) + 1
                    if (width, height, depth, color_type) != (1280, 720, 8, 2):
                        raise RuntimeError(f"Unexpected event PNG contract {contract}: {event_source}")
                image_destination = DATASET / "images" / output_split / seq / f"{seq}_frame_{frame_idx:06d}.png"
                link_exact(event_source, image_destination, args.check_only)
                mapped_label = DATASET / "labels" / output_split / seq / f"{seq}_frame_{frame_idx:06d}.txt"
                if not mapped_label.exists() or mapped_label.resolve() != label_source.resolve():
                    raise RuntimeError(f"Ultralytics image-to-label mapping failed: {mapped_label}")
                image_lines.append(f"./images/{output_split}/{seq}/{image_destination.name}")
                source_key = str(event_source.resolve())
                selected_sources[output_split].add(source_key)
                mapping_hash.update(f"{output_split}\t{seq}\t{frame_idx}\t{sparse_path.resolve()}\t{source_key}\t{label_source.resolve()}\n".encode())
                frame_count += 1
        if frame_count != spec["frames"] or len(image_lines) != frame_count:
            raise RuntimeError(f"Frame-count mismatch for {output_split}: {frame_count} != {spec['frames']}")
        if len(selected_sources[output_split]) != frame_count:
            raise RuntimeError(f"Duplicate event sources in {output_split}")
        list_content = "\n".join(image_lines) + "\n"
        list_path = DATASET / f"{output_split}_images.txt"
        write_exact(list_path, list_content, args.check_only)
        audit["splits"][output_split] = {
            "split_file": str(split_path), "split_sha256": spec["split_sha256"], "sequence_count": len(ids),
            "paired_frame_count": frame_count, "object_count": object_count, "multi_object_frame_count": multi_object_count,
            "raw_event_png_count_across_sequences": raw_event_count,
            "image_contract_counts": image_contracts if not args.skip_png_header_scan else "not_scanned",
            "label_extrema": {"min_cx": cx_min, "min_cy": cy_min, "max_width": width_max, "max_height": height_max},
            "image_list": str(list_path), "image_list_sha256": hashlib.sha256(list_content.encode()).hexdigest(),
            "source_mapping_sha256": mapping_hash.hexdigest(), "sequence_frame_counts": sequence_frame_counts,
        }
    overlap = selected_sources["train"] & selected_sources["val"]
    if overlap:
        raise RuntimeError(f"Train/val source-frame leakage: {len(overlap)} paths")
    audit["train_val_source_frame_overlap"] = 0
    audit["total_paired_frames"] = sum(row["paired_frame_count"] for row in audit["splits"].values())
    audit_text = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    write_exact(BASE / "dataset_audit.json", audit_text, args.check_only)
    print(audit_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())