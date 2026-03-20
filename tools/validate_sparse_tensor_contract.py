#!/usr/bin/env python3
"""Validate sparse tensor format contract for parity roots.

This checker is intentionally strict and fail-fast. It is designed to catch:
1) Mixed sparse encodings (`coords+feats` vs dense `spikes`) in one training lane
2) Unexpected sparse format relative to an explicit expected format
3) Optional time-bin inconsistencies in sampled files
4) Basic malformed payloads (shape/key mismatches)

By default it samples `files_per_seq=1` file per sequence for speed.
Increase `--files-per-seq` when auditing intra-sequence drift.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


PARITY_SPLITS = (
    "canonical_train",
    "canonical_test",
    "challenging_train",
    "challenging_test",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pick_sample_indices(n: int, k: int) -> List[int]:
    """Pick up to k deterministic spread indices in [0, n)."""
    if n <= 0 or k <= 0:
        return []
    if n <= k:
        return list(range(n))
    if k == 1:
        return [0]
    out: List[int] = []
    for i in range(k):
        idx = int(round(i * (n - 1) / float(k - 1)))
        if not out or idx != out[-1]:
            out.append(idx)
    return out


def percentile_dict(values: List[float], percentiles: List[Tuple[str, float]]) -> Dict[str, Optional[float]]:
    if not values:
        return {name: None for name, _ in percentiles}
    arr = np.asarray(values, dtype=np.float64)
    out: Dict[str, Optional[float]] = {}
    for name, p in percentiles:
        out[name] = float(np.percentile(arr, p))
    return out


def _scalar_positive_int(value: Any) -> Optional[int]:
    """Return int(value) if it is a positive scalar integer-like, else None."""
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.size != 1:
        return None
    try:
        iv = int(arr.item())
    except Exception:
        return None
    return iv if iv > 0 else None


def inspect_npz(path: Path, expected_feat_channels: int = 2) -> Dict[str, Any]:
    """Inspect one sparse npz payload and return compact metadata."""
    result: Dict[str, Any] = {
        "path": str(path),
        "format": "unknown",
        "time_bins": None,
        "time_bins_source": "unknown",
        "issues": [],
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            if {"coords", "feats"}.issubset(keys):
                coords = data["coords"]
                feats = data["feats"]
                result["format"] = "coords"
                result["coords_dtype"] = str(coords.dtype)
                result["feats_dtype"] = str(feats.dtype)

                if coords.ndim != 2 or coords.shape[1] != 3:
                    result["issues"].append(
                        f"coords_shape_invalid:{tuple(coords.shape)} expected [N,3]"
                    )
                expected_feat_ch = expected_feat_channels
                if feats.ndim != 2 or feats.shape[1] != expected_feat_ch:
                    result["issues"].append(
                        f"feats_shape_invalid:{tuple(feats.shape)} expected [N,{expected_feat_ch}]"
                    )
                if coords.ndim == 2 and feats.ndim == 2 and coords.shape[0] != feats.shape[0]:
                    result["issues"].append(
                        f"coords_feats_count_mismatch:{coords.shape[0]}!={feats.shape[0]}"
                    )

                if feats.ndim == 2 and feats.shape[1] >= 2:
                    finite = bool(np.all(np.isfinite(feats)))
                    has_negative = bool(np.any(feats < 0))
                    result["coords_feats_finite"] = finite
                    result["coords_feats_has_negative"] = has_negative
                    if feats.size > 0:
                        f32 = feats.astype(np.float32, copy=False)
                        result["coords_feats_min"] = float(np.min(f32))
                        result["coords_feats_max"] = float(np.max(f32))
                        result["coords_feats_mean"] = float(np.mean(f32))
                        result["coords_feats_file_max"] = float(np.max(f32))
                        # Keep report compact: deterministic prefix sample only.
                        flat = f32.reshape(-1)
                        sample_n = min(512, flat.size)
                        result["coords_feats_sample_values"] = [float(x) for x in flat[:sample_n]]
                    else:
                        result["coords_feats_min"] = 0.0
                        result["coords_feats_max"] = 0.0
                        result["coords_feats_mean"] = 0.0
                        result["coords_feats_file_max"] = 0.0
                        result["coords_feats_sample_values"] = []
                # Prefer explicit metadata when available; fallback to active-max estimate.
                time_bins_meta = None
                for k in ("time_bins", "num_bins", "T"):
                    if k in keys:
                        time_bins_meta = _scalar_positive_int(data[k])
                        if time_bins_meta is not None:
                            break

                if time_bins_meta is not None:
                    result["time_bins"] = int(time_bins_meta)
                    result["time_bins_source"] = "metadata"
                elif coords.size > 0 and coords.ndim == 2 and coords.shape[1] >= 1:
                    tmax = int(np.max(coords[:, 0]))
                    result["time_bins"] = int(tmax + 1)
                    result["time_bins_source"] = "active_max_plus_1"
                else:
                    result["time_bins"] = None
                    result["time_bins_source"] = "none"
                return result

            if "spikes" in keys:
                spikes = data["spikes"]
                result["format"] = "spikes"
                result["spikes_dtype"] = str(spikes.dtype)
                if spikes.ndim != 4:
                    result["issues"].append(
                        f"spikes_shape_invalid:{tuple(spikes.shape)} expected [T,C,H,W]"
                    )
                else:
                    result["time_bins"] = int(spikes.shape[0])
                    result["time_bins_source"] = "spikes_shape"
                    result["spikes_channels"] = int(spikes.shape[1])
                return result

            result["issues"].append(f"unknown_keys:{sorted(keys)}")
            return result
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        result["issues"].append(f"read_error:{type(exc).__name__}:{exc}")
        return result


def validate_split(
    split_dir: Path,
    files_per_seq: int,
    expected_feat_channels: int = 2,
) -> Dict[str, Any]:
    seq_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()]) if split_dir.exists() else []

    format_counts: Counter[str] = Counter()
    time_bins_counts: Counter[int] = Counter()
    time_bins_source_counts: Counter[str] = Counter()
    time_bins_exact_counts: Counter[int] = Counter()
    time_bins_inferred_counts: Counter[int] = Counter()
    malformed_files: List[str] = []
    seq_formats: Dict[str, List[str]] = defaultdict(list)
    seq_time_bins_exact: Dict[str, set[int]] = defaultdict(set)
    coords_with_time_bins_metadata = 0
    coords_without_time_bins_metadata = 0
    coords_feature_file_count = 0
    coords_feature_dtype_counts: Counter[str] = Counter()
    coords_feature_nonfinite_count = 0
    coords_feature_negative_count = 0
    coords_feature_file_max_values: List[float] = []
    coords_feature_value_samples: List[float] = []
    coords_feature_min_seen: Optional[float] = None
    coords_feature_max_seen: Optional[float] = None
    sampled_files = 0

    for seq_dir in seq_dirs:
        frame_files = sorted(seq_dir.glob("frame_*.npz"))
        if not frame_files:
            continue
        sample_idx = pick_sample_indices(len(frame_files), files_per_seq)
        for idx in sample_idx:
            meta = inspect_npz(frame_files[idx], expected_feat_channels=expected_feat_channels)
            sampled_files += 1
            fmt = str(meta.get("format", "unknown"))
            format_counts[fmt] += 1
            seq_formats[seq_dir.name].append(fmt)

            tbin = meta.get("time_bins", None)
            tbin_source = str(meta.get("time_bins_source", "unknown"))
            time_bins_source_counts[tbin_source] += 1
            if isinstance(tbin, int):
                time_bins_counts[int(tbin)] += 1
                if tbin_source in {"metadata", "spikes_shape"}:
                    time_bins_exact_counts[int(tbin)] += 1
                    seq_time_bins_exact[seq_dir.name].add(int(tbin))
                elif tbin_source == "active_max_plus_1":
                    time_bins_inferred_counts[int(tbin)] += 1

            if fmt == "coords":
                if tbin_source == "metadata":
                    coords_with_time_bins_metadata += 1
                else:
                    coords_without_time_bins_metadata += 1

                coords_feature_file_count += 1
                feats_dtype = str(meta.get("feats_dtype", "unknown"))
                coords_feature_dtype_counts[feats_dtype] += 1
                if not bool(meta.get("coords_feats_finite", True)):
                    coords_feature_nonfinite_count += 1
                if bool(meta.get("coords_feats_has_negative", False)):
                    coords_feature_negative_count += 1
                file_max = meta.get("coords_feats_file_max", None)
                if isinstance(file_max, (int, float)):
                    coords_feature_file_max_values.append(float(file_max))
                fmin = meta.get("coords_feats_min", None)
                fmax = meta.get("coords_feats_max", None)
                if isinstance(fmin, (int, float)):
                    fv = float(fmin)
                    coords_feature_min_seen = fv if coords_feature_min_seen is None else min(coords_feature_min_seen, fv)
                if isinstance(fmax, (int, float)):
                    fv = float(fmax)
                    coords_feature_max_seen = fv if coords_feature_max_seen is None else max(coords_feature_max_seen, fv)
                sample_vals = meta.get("coords_feats_sample_values", [])
                if isinstance(sample_vals, list) and sample_vals:
                    # Keep memory bounded.
                    take_n = min(256, len(sample_vals))
                    coords_feature_value_samples.extend(float(v) for v in sample_vals[:take_n])

            issues = list(meta.get("issues", []))
            if issues:
                malformed_files.append(f"{meta.get('path')} :: {'; '.join(issues)}")

    non_uniform_exact = {
        seq: sorted(list(tb_set))
        for seq, tb_set in seq_time_bins_exact.items()
        if len(tb_set) > 1
    }

    file_max_pct = percentile_dict(
        coords_feature_file_max_values,
        [("p50", 50.0), ("p90", 90.0), ("p99", 99.0), ("max", 100.0)],
    )
    value_sample_pct = percentile_dict(
        coords_feature_value_samples,
        [("p50", 50.0), ("p90", 90.0), ("p99", 99.0), ("p999", 99.9)],
    )
    file_max_ratio_p99_p50 = None
    if file_max_pct.get("p50") is not None and file_max_pct.get("p99") is not None:
        p50 = float(file_max_pct["p50"])
        p99 = float(file_max_pct["p99"])
        file_max_ratio_p99_p50 = float(p99 / max(p50, 1e-6))

    return {
        "split_dir": str(split_dir),
        "sequence_count": len(seq_dirs),
        "sampled_files": int(sampled_files),
        "format_counts": dict(sorted(format_counts.items())),
        "time_bins_counts": {str(k): int(v) for k, v in sorted(time_bins_counts.items())},
        "time_bins_source_counts": dict(sorted(time_bins_source_counts.items())),
        "time_bins_exact_counts": {str(k): int(v) for k, v in sorted(time_bins_exact_counts.items())},
        "time_bins_inferred_counts": {str(k): int(v) for k, v in sorted(time_bins_inferred_counts.items())},
        "coords_with_time_bins_metadata": int(coords_with_time_bins_metadata),
        "coords_without_time_bins_metadata": int(coords_without_time_bins_metadata),
        "coords_feature_file_count": int(coords_feature_file_count),
        "coords_feature_dtype_counts": dict(sorted(coords_feature_dtype_counts.items())),
        "coords_feature_nonfinite_count": int(coords_feature_nonfinite_count),
        "coords_feature_negative_count": int(coords_feature_negative_count),
        "coords_feature_min_seen": coords_feature_min_seen,
        "coords_feature_max_seen": coords_feature_max_seen,
        "coords_feature_file_max_percentiles": file_max_pct,
        "coords_feature_sample_value_percentiles": value_sample_pct,
        "coords_feature_file_max_p99_p50_ratio": file_max_ratio_p99_p50,
        "non_uniform_time_bins_exact_seq_count": int(len(non_uniform_exact)),
        "non_uniform_time_bins_exact_seq_examples": [
            {"seq": seq, "time_bins": bins}
            for seq, bins in sorted(non_uniform_exact.items(), key=lambda kv: int(kv[0]))[:20]
        ],
        "malformed_count": len(malformed_files),
        "malformed_examples": malformed_files[:20],
        "seqs_by_format": {
            "coords": sorted([s for s, f in seq_formats.items() if "coords" in f]),
            "spikes": sorted([s for s, f in seq_formats.items() if "spikes" in f]),
            "unknown": sorted([s for s, f in seq_formats.items() if "unknown" in f]),
        },
        "seq_format_examples": {
            "coords": sorted([s for s, f in seq_formats.items() if "coords" in f])[:20],
            "spikes": sorted([s for s, f in seq_formats.items() if "spikes" in f])[:20],
            "unknown": sorted([s for s, f in seq_formats.items() if "unknown" in f])[:20],
        },
    }


def build_remediation(violations: List[str]) -> List[str]:
    actions: List[str] = []
    if any("mixed_formats" in v for v in violations):
        actions.append(
            "Rebuild sparse parity roots with a single format. Do not mix `coords+feats` and `spikes` in one run."
        )
        actions.append(
            "Choose one canonical format and enforce it via `data.sparse_contract.expected_format`."
        )
    if any("unexpected_format" in v for v in violations):
        actions.append(
            "Convert offending files to the expected format or regenerate sparse roots from one source pipeline."
        )
    if any("time_bins_mismatch" in v for v in violations):
        actions.append(
            "Regenerate sparse tensors with one temporal bin config and keep `expected_time_bins` aligned with model config."
        )
    if any("malformed" in v for v in violations):
        actions.append(
            "Quarantine malformed npz files and regenerate those sequences before training."
        )
    if any("missing_split_dir" in v for v in violations):
        actions.append(
            "Fix sparse root path or split naming; all configured splits must exist under sparse root."
        )
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate sparse tensor format contract.")
    parser.add_argument(
        "--sparse-root",
        type=Path,
        default=Path("data/datasets/fred_paper_parity/sparse"),
        help="Root containing split/sequence/frame_XXXXXX.npz trees.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=list(PARITY_SPLITS),
        help="Splits to validate (default: parity splits).",
    )
    parser.add_argument(
        "--files-per-seq",
        type=int,
        default=1,
        help="How many files to sample per sequence (deterministic spread).",
    )
    parser.add_argument(
        "--allow-mixed-formats",
        action="store_true",
        help="Allow both coords and spikes in the same validation run.",
    )
    parser.add_argument(
        "--expected-format",
        type=str,
        default=None,
        choices=["coords", "spikes"],
        help="Optional expected format. Any mismatch fails validation.",
    )
    parser.add_argument(
        "--expected-time-bins",
        type=int,
        default=None,
        help="Optional expected time bin count (sampled files).",
    )
    parser.add_argument(
        "--enforce-uniform-time-bins",
        action="store_true",
        help="Fail if sampled files contain more than one time bin count.",
    )
    parser.add_argument(
        "--enforce-per-seq-uniform-time-bins",
        action="store_true",
        help="Fail if any sequence has >1 exact time-bin value in sampled files.",
    )
    parser.add_argument(
        "--require-coords-time-bins-metadata",
        action="store_true",
        help="Fail if sampled coords files do not carry explicit time-bin metadata.",
    )
    parser.add_argument(
        "--require-coords-feats-finite",
        action="store_true",
        help="Fail if any sampled coords file has non-finite feature values.",
    )
    parser.add_argument(
        "--require-coords-feats-nonnegative",
        action="store_true",
        help="Fail if any sampled coords file has negative feature values.",
    )
    parser.add_argument(
        "--allowed-coords-feat-dtypes",
        type=str,
        nargs="*",
        default=None,
        help="Allowed dtype names for coords feats (e.g., float16 float32).",
    )
    parser.add_argument(
        "--coords-feat-max-value",
        type=float,
        default=None,
        help="Fail if sampled coords feature global max exceeds this value.",
    )
    parser.add_argument(
        "--coords-feat-value-p99-max",
        type=float,
        default=None,
        help="Fail if sampled coords feature value p99 exceeds this value.",
    )
    parser.add_argument(
        "--coords-feat-file-max-p99-max",
        type=float,
        default=None,
        help="Fail if p99 of per-file feature maxima exceeds this value.",
    )
    parser.add_argument(
        "--coords-feat-file-max-p99-p50-max-ratio",
        type=float,
        default=None,
        help="Fail if (per-file-max p99 / p50) exceeds this ratio.",
    )
    parser.add_argument(
        "--expected-feat-channels",
        type=int,
        default=2,
        help="Expected number of feature channels per voxel (default 2, use 6 for v82).",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path to write detailed validation report JSON.",
    )
    args = parser.parse_args()

    splits = [str(s) for s in args.splits]
    files_per_seq = max(1, int(args.files_per_seq))

    split_reports: Dict[str, Any] = {}
    violations: List[str] = []
    notices: List[str] = []
    global_formats: Counter[str] = Counter()
    global_time_bins: Counter[int] = Counter()
    global_time_bins_exact: Counter[int] = Counter()
    global_time_bins_inferred: Counter[int] = Counter()
    global_coords_missing_time_bins_metadata = 0
    global_coords_with_time_bins_metadata = 0
    global_coords_feat_nonfinite = 0
    global_coords_feat_negative = 0
    global_coords_feat_dtype_counts: Counter[str] = Counter()
    global_coords_feat_max_seen: Optional[float] = None
    global_coords_feat_value_p99_samples: List[float] = []
    global_coords_feat_file_max_values: List[float] = []

    for split in splits:
        split_dir = args.sparse_root / split
        if not split_dir.exists():
            violations.append(f"{split}:missing_split_dir:{split_dir}")
            split_reports[split] = {
                "split_dir": str(split_dir),
                "missing": True,
            }
            continue

        rep = validate_split(split_dir=split_dir, files_per_seq=files_per_seq, expected_feat_channels=args.expected_feat_channels)
        split_reports[split] = rep

        fmt_counts = rep.get("format_counts", {})
        for k, v in fmt_counts.items():
            global_formats[str(k)] += int(v)

        for k, v in rep.get("time_bins_counts", {}).items():
            try:
                global_time_bins[int(k)] += int(v)
            except Exception:
                continue

        for k, v in rep.get("time_bins_exact_counts", {}).items():
            try:
                global_time_bins_exact[int(k)] += int(v)
            except Exception:
                continue
        for k, v in rep.get("time_bins_inferred_counts", {}).items():
            try:
                global_time_bins_inferred[int(k)] += int(v)
            except Exception:
                continue

        global_coords_with_time_bins_metadata += int(rep.get("coords_with_time_bins_metadata", 0))
        global_coords_missing_time_bins_metadata += int(rep.get("coords_without_time_bins_metadata", 0))
        global_coords_feat_nonfinite += int(rep.get("coords_feature_nonfinite_count", 0))
        global_coords_feat_negative += int(rep.get("coords_feature_negative_count", 0))
        for k, v in rep.get("coords_feature_dtype_counts", {}).items():
            global_coords_feat_dtype_counts[str(k)] += int(v)
        split_feat_max_seen = rep.get("coords_feature_max_seen", None)
        if isinstance(split_feat_max_seen, (int, float)):
            fv = float(split_feat_max_seen)
            global_coords_feat_max_seen = fv if global_coords_feat_max_seen is None else max(global_coords_feat_max_seen, fv)

        split_feat_pcts = rep.get("coords_feature_sample_value_percentiles", {}) or {}
        split_p99 = split_feat_pcts.get("p99", None)
        if isinstance(split_p99, (int, float)):
            global_coords_feat_value_p99_samples.append(float(split_p99))
        split_file_max_pcts = rep.get("coords_feature_file_max_percentiles", {}) or {}
        split_file_max_p99 = split_file_max_pcts.get("p99", None)
        if isinstance(split_file_max_p99, (int, float)):
            global_coords_feat_file_max_values.append(float(split_file_max_p99))

        malformed_count = int(rep.get("malformed_count", 0))
        if malformed_count > 0:
            violations.append(f"{split}:malformed:{malformed_count}")
        if args.enforce_per_seq_uniform_time_bins:
            non_uniform_seq_count = int(rep.get("non_uniform_time_bins_exact_seq_count", 0))
            if non_uniform_seq_count > 0:
                violations.append(f"{split}:non_uniform_time_bins_exact_seq:{non_uniform_seq_count}")
        if args.require_coords_time_bins_metadata:
            missing_meta = int(rep.get("coords_without_time_bins_metadata", 0))
            if missing_meta > 0:
                violations.append(f"{split}:coords_missing_time_bins_metadata:{missing_meta}")
        if args.require_coords_feats_finite:
            bad_nonfinite = int(rep.get("coords_feature_nonfinite_count", 0))
            if bad_nonfinite > 0:
                violations.append(f"{split}:coords_feats_nonfinite:{bad_nonfinite}")
        if args.require_coords_feats_nonnegative:
            bad_negative = int(rep.get("coords_feature_negative_count", 0))
            if bad_negative > 0:
                violations.append(f"{split}:coords_feats_negative:{bad_negative}")
        if args.allowed_coords_feat_dtypes:
            allowed = set(str(x) for x in args.allowed_coords_feat_dtypes)
            seen = set(str(x) for x in (rep.get("coords_feature_dtype_counts", {}) or {}).keys())
            disallowed = sorted([x for x in seen if x not in allowed])
            if disallowed:
                violations.append(f"{split}:coords_feat_dtype_disallowed:{disallowed}")
        if args.coords_feat_max_value is not None:
            split_max = rep.get("coords_feature_max_seen", None)
            if isinstance(split_max, (int, float)) and float(split_max) > float(args.coords_feat_max_value):
                violations.append(
                    f"{split}:coords_feat_max_value_exceeded:{float(split_max):.6f}>{float(args.coords_feat_max_value):.6f}"
                )
        if args.coords_feat_value_p99_max is not None:
            p99_val = (rep.get("coords_feature_sample_value_percentiles", {}) or {}).get("p99", None)
            if isinstance(p99_val, (int, float)) and float(p99_val) > float(args.coords_feat_value_p99_max):
                violations.append(
                    f"{split}:coords_feat_value_p99_exceeded:{float(p99_val):.6f}>{float(args.coords_feat_value_p99_max):.6f}"
                )
        if args.coords_feat_file_max_p99_max is not None:
            p99_filemax = (rep.get("coords_feature_file_max_percentiles", {}) or {}).get("p99", None)
            if isinstance(p99_filemax, (int, float)) and float(p99_filemax) > float(args.coords_feat_file_max_p99_max):
                violations.append(
                    f"{split}:coords_feat_file_max_p99_exceeded:{float(p99_filemax):.6f}>{float(args.coords_feat_file_max_p99_max):.6f}"
                )
        if args.coords_feat_file_max_p99_p50_max_ratio is not None:
            ratio = rep.get("coords_feature_file_max_p99_p50_ratio", None)
            if isinstance(ratio, (int, float)) and float(ratio) > float(args.coords_feat_file_max_p99_p50_max_ratio):
                violations.append(
                    f"{split}:coords_feat_file_max_p99_p50_ratio_exceeded:{float(ratio):.6f}>{float(args.coords_feat_file_max_p99_p50_max_ratio):.6f}"
                )

    present_formats = sorted([f for f, n in global_formats.items() if n > 0])
    if not args.allow_mixed_formats and len(present_formats) > 1:
        violations.append(f"global:mixed_formats:{present_formats}")

    if args.expected_format is not None:
        unexpected = [f for f in present_formats if f != args.expected_format]
        if unexpected:
            violations.append(
                f"global:unexpected_format:expected={args.expected_format}:found={unexpected}"
            )

    if args.expected_time_bins is not None:
        expected_tb = int(args.expected_time_bins)
        exact_mismatched = sorted([tb for tb in global_time_bins_exact.keys() if tb != expected_tb])
        if exact_mismatched:
            violations.append(
                f"global:time_bins_exact_mismatch:expected={expected_tb}:found={exact_mismatched}"
            )
        inferred_over = sorted([tb for tb in global_time_bins_inferred.keys() if tb > expected_tb])
        if inferred_over:
            violations.append(
                f"global:time_bins_inferred_gt_expected:expected={expected_tb}:found={inferred_over}"
            )
        inferred_below = sorted([tb for tb in global_time_bins_inferred.keys() if tb < expected_tb])
        if inferred_below:
            notices.append(
                f"global:time_bins_inferred_below_expected:{inferred_below} (may be due to trailing empty bins in coords)"
            )

    if args.enforce_uniform_time_bins:
        present_exact_tb = sorted(global_time_bins_exact.keys())
        if len(present_exact_tb) > 1:
            violations.append(f"global:non_uniform_time_bins_exact:{present_exact_tb}")
        elif len(present_exact_tb) == 0:
            present_inferred_tb = sorted(global_time_bins_inferred.keys())
            if len(present_inferred_tb) > 1:
                notices.append(
                    f"global:non_uniform_time_bins_inferred:{present_inferred_tb} (exact metadata unavailable)"
                )

    if args.require_coords_feats_finite and global_coords_feat_nonfinite > 0:
        violations.append(f"global:coords_feats_nonfinite:{global_coords_feat_nonfinite}")
    if args.require_coords_feats_nonnegative and global_coords_feat_negative > 0:
        violations.append(f"global:coords_feats_negative:{global_coords_feat_negative}")
    if args.allowed_coords_feat_dtypes:
        allowed = set(str(x) for x in args.allowed_coords_feat_dtypes)
        seen = set(global_coords_feat_dtype_counts.keys())
        disallowed = sorted([x for x in seen if x not in allowed])
        if disallowed:
            violations.append(f"global:coords_feat_dtype_disallowed:{disallowed}")
    if args.coords_feat_max_value is not None and global_coords_feat_max_seen is not None:
        if float(global_coords_feat_max_seen) > float(args.coords_feat_max_value):
            violations.append(
                f"global:coords_feat_max_value_exceeded:{float(global_coords_feat_max_seen):.6f}>{float(args.coords_feat_max_value):.6f}"
            )

    global_coords_feat_value_p99_of_split_p99 = None
    if global_coords_feat_value_p99_samples:
        global_coords_feat_value_p99_of_split_p99 = float(np.percentile(np.asarray(global_coords_feat_value_p99_samples), 99.0))
    if args.coords_feat_value_p99_max is not None and global_coords_feat_value_p99_of_split_p99 is not None:
        if float(global_coords_feat_value_p99_of_split_p99) > float(args.coords_feat_value_p99_max):
            violations.append(
                f"global:coords_feat_value_p99_exceeded:{float(global_coords_feat_value_p99_of_split_p99):.6f}>{float(args.coords_feat_value_p99_max):.6f}"
            )

    global_coords_feat_file_max_p99_of_split_p99 = None
    if global_coords_feat_file_max_values:
        global_coords_feat_file_max_p99_of_split_p99 = float(np.percentile(np.asarray(global_coords_feat_file_max_values), 99.0))
    if args.coords_feat_file_max_p99_max is not None and global_coords_feat_file_max_p99_of_split_p99 is not None:
        if float(global_coords_feat_file_max_p99_of_split_p99) > float(args.coords_feat_file_max_p99_max):
            violations.append(
                f"global:coords_feat_file_max_p99_exceeded:{float(global_coords_feat_file_max_p99_of_split_p99):.6f}>{float(args.coords_feat_file_max_p99_max):.6f}"
            )

    status = "pass" if not violations else "fail"
    report = {
        "generated_at": now_iso(),
        "status": status,
        "sparse_root": str(args.sparse_root),
        "splits": splits,
        "files_per_seq": files_per_seq,
        "allow_mixed_formats": bool(args.allow_mixed_formats),
        "expected_format": args.expected_format,
        "expected_time_bins": args.expected_time_bins,
        "enforce_uniform_time_bins": bool(args.enforce_uniform_time_bins),
        "enforce_per_seq_uniform_time_bins": bool(args.enforce_per_seq_uniform_time_bins),
        "require_coords_time_bins_metadata": bool(args.require_coords_time_bins_metadata),
        "require_coords_feats_finite": bool(args.require_coords_feats_finite),
        "require_coords_feats_nonnegative": bool(args.require_coords_feats_nonnegative),
        "allowed_coords_feat_dtypes": list(args.allowed_coords_feat_dtypes) if args.allowed_coords_feat_dtypes else None,
        "coords_feat_max_value": args.coords_feat_max_value,
        "coords_feat_value_p99_max": args.coords_feat_value_p99_max,
        "coords_feat_file_max_p99_max": args.coords_feat_file_max_p99_max,
        "coords_feat_file_max_p99_p50_max_ratio": args.coords_feat_file_max_p99_p50_max_ratio,
        "global_summary": {
            "format_counts": dict(sorted(global_formats.items())),
            "time_bins_counts": {str(k): int(v) for k, v in sorted(global_time_bins.items())},
            "time_bins_exact_counts": {str(k): int(v) for k, v in sorted(global_time_bins_exact.items())},
            "time_bins_inferred_counts": {str(k): int(v) for k, v in sorted(global_time_bins_inferred.items())},
            "present_formats": present_formats,
            "coords_with_time_bins_metadata": int(global_coords_with_time_bins_metadata),
            "coords_without_time_bins_metadata": int(global_coords_missing_time_bins_metadata),
            "coords_feat_dtype_counts": dict(sorted(global_coords_feat_dtype_counts.items())),
            "coords_feat_nonfinite_count": int(global_coords_feat_nonfinite),
            "coords_feat_negative_count": int(global_coords_feat_negative),
            "coords_feat_max_seen": global_coords_feat_max_seen,
            "coords_feat_value_p99_of_split_p99": global_coords_feat_value_p99_of_split_p99,
            "coords_feat_file_max_p99_of_split_p99": global_coords_feat_file_max_p99_of_split_p99,
        },
        "split_reports": split_reports,
        "violations": violations,
        "notices": notices,
        "remediation": build_remediation(violations),
    }

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
