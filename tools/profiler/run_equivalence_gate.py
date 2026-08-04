#!/usr/bin/env python3
"""Run only the ten-frame FP32 dense/sparse equivalence gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch
import spconv.pytorch as spconv

from V2.models.sparse_voxel_det_ic import SparseVoxelDetIC
from dense_transplant import transplant_sparse_model


EXPECTED_CHECKPOINT_NAME = "epoch_005.pt"
EXPECTED_CHECKPOINT_SHA256 = (
    "b25c62a09190a09c3659854f985c477cc4e052ea089e3f9bb438fb56d05c08ba"
)
ATOL = 1e-3
# The three cards carrying the seed-123 confirmation run. Refused unconditionally so
# the gate can never take a training GPU, whatever CUDA_VISIBLE_DEVICES says.
DDP3_PINNED_UUID_PREFIXES = ("1d11b997", "2a7554bd", "48d3a2b0")
OUTPUT_PATH = Path(__file__).resolve().parent / "equivalence_gate.json"
HEAD_OUTPUTS = ("cls_logits", "box_ltrb", "ctr_logits")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-state-key",
        help="Exact checkpoint key containing the state_dict; omit only when it is uniquely identifiable.",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frames-manifest", type=Path, required=True)
    parser.add_argument("--in-channels", type=int, default=6)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--backbone-size", default="nano_deep")
    parser.add_argument("--fpn-channels", type=int, default=128)
    parser.add_argument("--head-convs", type=int, default=2)
    parser.add_argument("--input-height", type=int, default=720)
    parser.add_argument("--input-width", type=int, default=1280)
    parser.add_argument("--time-bins", type=int, default=16)
    parser.add_argument("--temporal-pool-mode", choices=("max", "mean"), default="max")
    parser.add_argument(
        "--allow-non-5090",
        action="store_true",
        help=(
            "Run the numerical check on a non-5090 card. Equivalence is a property of "
            "the weights and the arithmetic, not of the board, so this does not weaken "
            "the gate; the card is recorded in the receipt either way. The forbidden "
            "baseline GPU and the three DDP3-pinned cards stay refused regardless."
        ),
    )
    return parser.parse_args()


def _has_sealed_component(path: Path) -> bool:
    # A bare "test" component covers data/processed/FRED/test and raw_data/FRED/test,
    # which the canonical_test / challenging_test tokens do NOT match. Check all three
    # forms plus any *_test component so no sealed partition can be reached by any name.
    for part in path.parts:
        folded = part.casefold()
        if folded == "test" or folded.endswith("_test"):
            return True
    return False


def _resolve_under(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"frame payload escapes the declared data root: {relative}") from exc
    return candidate


def _preflight(
    data_root_arg: Path,
    manifest_arg: Path,
) -> Tuple[Path, Path, List[Dict[str, Any]]]:
    data_root = data_root_arg.resolve(strict=True)
    manifest_path = manifest_arg.resolve(strict=True)
    if data_root.name != "canonical_val":
        raise ValueError(
            f"--data-root must resolve to a directory named canonical_val, got {data_root.name!r}"
        )
    if _has_sealed_component(data_root) or _has_sealed_component(manifest_path):
        raise ValueError("resolved data path contains a sealed split component")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = document.get("frames")
    if not isinstance(frames, list) or len(frames) < 10:
        raise ValueError("frames manifest must contain at least ten entries")
    resolved: List[Dict[str, Any]] = []
    seen_ids = set()
    for entry in frames:
        if not isinstance(entry, dict):
            raise TypeError("each frame manifest entry must be an object")
        frame_id = entry.get("frame_id")
        relative_path = entry.get("path")
        coordinate_count = entry.get("coordinate_count")
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("each frame requires a non-empty string frame_id")
        if frame_id in seen_ids:
            raise ValueError(f"duplicate frame_id: {frame_id}")
        seen_ids.add(frame_id)
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"{frame_id}: missing relative payload path")
        if not isinstance(coordinate_count, int) or coordinate_count < 0:
            raise ValueError(f"{frame_id}: coordinate_count must be a non-negative integer")
        payload_path = _resolve_under(data_root, relative_path)
        if _has_sealed_component(payload_path):
            raise ValueError("resolved frame payload path contains a sealed split component")
        resolved.append(
            {
                "frame_id": frame_id,
                "payload_path": payload_path,
                "coordinate_count": coordinate_count,
            }
        )
    print("sealed_split_preflight: PASS", flush=True)
    return data_root, manifest_path, resolved


def _even_positions(length: int, count: int) -> List[int]:
    if length < count:
        raise ValueError(f"activity bin has {length} frames; {count} are required")
    if count == 1:
        return [length // 2]
    return [(i * (length - 1)) // (count - 1) for i in range(count)]


def _select_gate_frames(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(frames, key=lambda item: (item["coordinate_count"], item["frame_id"]))
    n = len(ranked)
    edge = n // 3
    bins = (
        ("low", ranked[:edge], 3, 0),
        ("median", ranked[edge : n - edge], 4, edge),
        ("high", ranked[n - edge :], 3, n - edge),
    )
    selected: List[Dict[str, Any]] = []
    for label, population, count, offset in bins:
        for local_rank in _even_positions(len(population), count):
            item = dict(population[local_rank])
            global_rank = offset + local_rank
            item["activity_bin"] = label
            item["selection_reason"] = (
                f"coordinate-count rank {global_rank + 1}/{n}; "
                f"evenly spaced within the {label} activity third"
            )
            selected.append(item)
    if len({item["frame_id"] for item in selected}) != 10:
        raise RuntimeError("activity selection did not produce ten unique frames")
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_state(
    checkpoint_path: Path,
    state_key: str | None,
) -> Tuple[Mapping[str, torch.Tensor], str]:
    resolved = checkpoint_path.resolve(strict=True)
    if resolved.name != EXPECTED_CHECKPOINT_NAME:
        raise ValueError(f"checkpoint filename must be {EXPECTED_CHECKPOINT_NAME}")
    digest = _sha256(resolved)
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected {EXPECTED_CHECKPOINT_SHA256}, got {digest}"
        )
    loaded = torch.load(resolved, map_location="cpu", weights_only=True)
    def is_state_dict(candidate: Any) -> bool:
        return (
            isinstance(candidate, Mapping)
            and bool(candidate)
            and all(
                isinstance(key, str) and torch.is_tensor(value)
                for key, value in candidate.items()
            )
        )

    if state_key == "ROOT":
        state = loaded
    elif state_key is not None:
        if not isinstance(loaded, Mapping) or state_key not in loaded:
            raise KeyError(f"checkpoint does not contain the explicit key {state_key!r}")
        state = loaded[state_key]
    elif is_state_dict(loaded):
        state = loaded
    elif isinstance(loaded, Mapping):
        candidates = [
            (key, value) for key, value in loaded.items() if is_state_dict(value)
        ]
        if len(candidates) != 1:
            names = [str(key) for key, _ in candidates]
            raise RuntimeError(
                "checkpoint state_dict is not unique; pass --checkpoint-state-key "
                f"explicitly (candidates: {names})"
            )
        state = candidates[0][1]
    else:
        raise TypeError("checkpoint does not contain an identifiable state_dict")
    if not is_state_dict(state):
        raise TypeError("selected checkpoint object is not a non-empty tensor state_dict")
    return state, digest


def _build_model(args: argparse.Namespace, state: Mapping[str, torch.Tensor]) -> SparseVoxelDetIC:
    model = SparseVoxelDetIC(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        backbone_size=args.backbone_size,
        fpn_channels=args.fpn_channels,
        head_convs=args.head_convs,
        input_size=(args.input_height, args.input_width),
        time_bins=args.time_bins,
        temporal_pool_mode=args.temporal_pool_mode,
    ).float()
    model.load_state_dict(state, strict=True)
    return model.eval()


def _gpu_uuid(allow_non_5090: bool = False) -> tuple[str, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"gate requires exactly one visible GPU, found {torch.cuda.device_count()}"
        )
    properties = torch.cuda.get_device_properties(0)
    if "5090" not in properties.name and not allow_non_5090:
        raise RuntimeError(f"expected RTX 5090, found {properties.name}")
    uuid = getattr(properties, "uuid", None)
    if uuid is None:
        raise RuntimeError("this PyTorch build does not expose the CUDA device UUID")
    if isinstance(uuid, bytes):
        uuid = uuid.decode("ascii")
    uuid = str(uuid)
    folded = uuid.casefold()
    if "b279b278" in folded:
        raise RuntimeError("the forbidden baseline GPU is visible")
    for pinned in DDP3_PINNED_UUID_PREFIXES:
        if pinned in folded:
            raise RuntimeError(
                f"GPU {pinned} is pinned to the DDP3 confirmation run and must not be used"
            )
    return uuid, properties.name


def _load_frame(
    selected: Mapping[str, Any],
    args: argparse.Namespace,
) -> spconv.SparseConvTensor:
    payload = torch.load(selected["payload_path"], map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{selected['frame_id']}: payload must be a mapping")
    required = ("features", "indices", "spatial_shape")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"{selected['frame_id']}: payload missing {missing}")
    features = payload["features"]
    indices = payload["indices"]
    spatial_shape = tuple(int(value) for value in payload["spatial_shape"])
    if not torch.is_tensor(features) or features.ndim != 2:
        raise TypeError(f"{selected['frame_id']}: features must be [N,C]")
    if not torch.is_tensor(indices) or indices.ndim != 2 or indices.shape[1] != 4:
        raise TypeError(f"{selected['frame_id']}: indices must be [N,4]")
    if features.shape[0] != indices.shape[0]:
        raise ValueError(f"{selected['frame_id']}: feature/index row counts differ")
    if int(indices.shape[0]) != selected["coordinate_count"]:
        raise ValueError(
            f"{selected['frame_id']}: manifest coordinate_count "
            f"{selected['coordinate_count']} != payload count {indices.shape[0]}"
        )
    expected_shape = (args.time_bins, args.input_height, args.input_width)
    if spatial_shape != expected_shape:
        raise ValueError(
            f"{selected['frame_id']}: spatial_shape {spatial_shape} != {expected_shape}"
        )
    if features.shape[1] != args.in_channels:
        raise ValueError(
            f"{selected['frame_id']}: input channels {features.shape[1]} "
            f"!= {args.in_channels}"
        )
    indices = indices.to(device="cuda:0", dtype=torch.int32)
    features = features.to(device="cuda:0", dtype=torch.float32)
    if indices.numel() and (
        int(indices[:, 0].min().item()) != 0 or int(indices[:, 0].max().item()) != 0
    ):
        raise ValueError(f"{selected['frame_id']}: each gate payload must contain one batch")
    return spconv.SparseConvTensor(
        features=features,
        indices=indices,
        spatial_shape=list(spatial_shape),
        batch_size=1,
    )


def _sorted_outputs(
    outputs: Mapping[str, Any],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Tuple[int, int]]:
    indices = outputs["indices_2d"]
    spatial = tuple(int(value) for value in outputs["spatial_2d"])
    if not torch.is_tensor(indices) or indices.ndim != 2 or indices.shape[1] != 3:
        raise TypeError("pre-NMS indices_2d must be [M,3]")
    if len(spatial) != 2:
        raise ValueError("spatial_2d must contain H,W")
    height, width = spatial
    idx = indices.long()
    keys = idx[:, 0] * (height * width) + idx[:, 1] * width + idx[:, 2]
    if torch.unique(keys).numel() != keys.numel():
        raise RuntimeError("pre-NMS coordinates are not unique")
    order = torch.argsort(keys)
    tensors = {}
    for name in HEAD_OUTPUTS:
        value = outputs[name]
        if not torch.is_tensor(value) or value.shape[0] != indices.shape[0]:
            raise ValueError(f"{name} is not aligned with indices_2d")
        tensors[name] = value[order]
    return keys[order], tensors, spatial


def _compare_frame(
    sparse_outputs: Mapping[str, Any],
    dense_outputs: Mapping[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    sparse_keys, sparse_tensors, sparse_spatial = _sorted_outputs(sparse_outputs)
    dense_keys, dense_tensors, dense_spatial = _sorted_outputs(dense_outputs)
    coordinates_match = (
        sparse_spatial == dense_spatial
        and sparse_keys.shape == dense_keys.shape
        and torch.equal(sparse_keys, dense_keys)
    )
    deviations: Dict[str, Any] = {}
    passed = coordinates_match
    for name in HEAD_OUTPUTS:
        if not coordinates_match or sparse_tensors[name].shape != dense_tensors[name].shape:
            deviations[name] = None
            passed = False
            continue
        if sparse_tensors[name].numel() == 0:
            deviation = 0.0
        else:
            deviation = float(
                (sparse_tensors[name] - dense_tensors[name]).abs().max().item()
            )
        deviations[name] = deviation
        passed = passed and deviation <= ATOL
    return passed, {
        "coordinates_match": coordinates_match,
        "max_abs_deviation": deviations,
    }


def _write_result(result: Dict[str, Any]) -> None:
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(OUTPUT_PATH)


def main() -> int:
    args = _parse_args()
    _, manifest_path, frames = _preflight(args.data_root, args.frames_manifest)
    selected = _select_gate_frames(frames)
    result: Dict[str, Any] = {
        "gate": "FAIL",
        "atol": ATOL,
        "precision": "FP32",
        "checkpoint_sha256": None,
        "gpu_uuid": None,
        "frames_manifest": str(manifest_path),
        "transplant_map": [],
        "frames": [],
    }
    phase = "checkpoint"
    try:
        state, checkpoint_sha = _checkpoint_state(
            args.checkpoint, args.checkpoint_state_key
        )
        result["checkpoint_sha256"] = checkpoint_sha
        phase = "model_construction"
        sparse_model = _build_model(args, state)
        dense_model = transplant_sparse_model(sparse_model)
        result["transplant_map"] = dense_model.transplant_map
        phase = "gpu_preflight"
        gpu_uuid, gpu_name = _gpu_uuid(args.allow_non_5090)
        result["gpu_uuid"] = gpu_uuid
        result["gpu_name"] = gpu_name
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        sparse_model = sparse_model.to("cuda:0").eval()
        dense_model = dense_model.to("cuda:0").eval()
        all_passed = True
        with torch.inference_mode():
            for frame in selected:
                phase = f"sparse_forward:{frame['frame_id']}"
                sparse_input = _load_frame(frame, args)
                sparse_outputs = sparse_model(
                    sparse_input, batch_size=1, return_loss_inputs=True
                )
                phase = f"dense_forward:{frame['frame_id']}"
                dense_result = dense_model(
                    sparse_input,
                    batch_size=1,
                    return_loss_inputs=True,
                    collect_timings=False,
                )
                if dense_result.timings_ms:
                    raise RuntimeError("equivalence gate unexpectedly collected timing results")
                phase = f"comparison:{frame['frame_id']}"
                frame_passed, comparison = _compare_frame(
                    sparse_outputs, dense_result.outputs
                )
                all_passed = all_passed and frame_passed
                result["frames"].append(
                    {
                        "frame_id": frame["frame_id"],
                        "activity_bin": frame["activity_bin"],
                        "input_coordinate_count": frame["coordinate_count"],
                        "selection_reason": frame["selection_reason"],
                        "gate": "PASS" if frame_passed else "FAIL",
                        **comparison,
                    }
                )
        result["gate"] = "PASS" if all_passed else "FAIL"
    except torch.cuda.OutOfMemoryError as exc:
        result["error"] = {
            "phase": phase,
            "type": "CUDA_OUT_OF_MEMORY",
            "message": str(exc),
        }
    except Exception as exc:
        result["error"] = {
            "phase": phase,
            "type": type(exc).__name__,
            "message": str(exc),
        }
    _write_result(result)
    print(f"equivalence_gate: {result['gate']}")
    return 0 if result["gate"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
