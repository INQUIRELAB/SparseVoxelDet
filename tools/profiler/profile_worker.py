#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
from pathlib import Path
from typing import Any

import torch

from common import (
    DenseConvCapture,
    LinearCapture,
    RulebookCapture,
    assert_paths_unsealed,
    build_arm,
    load_contract,
    load_records,
    make_sparse_input,
    process_identity,
    require,
    settle_gpu,
    sha256_file,
    target_gpu_snapshot,
    verify_cuda_placement,
    verify_no_cotenant,
    write_json_new,
    write_jsonl_new,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("compute", "latency", "memory"), required=True)
    parser.add_argument("--arm", choices=("sparse", "dense"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--oom-output", type=Path, required=True)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--pass-position", type=int)
    parser.add_argument("--ordinal", type=int)
    return parser.parse_args()


def output_shape(arm: str, output: Any) -> list[int]:
    mapping = output.outputs if arm == "dense" else output
    return list(mapping["detections"].shape)


def forward(arm: str, model: Any, input_value: Any, collect_timings: bool) -> Any:
    if arm == "dense":
        return model(input_value, batch_size=1, collect_timings=collect_timings)
    require(not collect_timings, "sparse arm has no dense timing ledger")
    return model(input_value, batch_size=1)


def base_metadata(args: argparse.Namespace, contract: dict[str, Any], model_meta: dict[str, Any], placement: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "arm": args.arm,
        "mode": args.mode,
        "process": process_identity(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "precision": contract["precision"],
        "gpu": target_gpu_snapshot(contract),
        "cuda_placement": placement,
        "model": model_meta,
    }


def run_compute(args: argparse.Namespace, contract: dict[str, Any], records: list[dict[str, Any]], model: Any, device: torch.device, metadata: dict[str, Any]) -> None:
    linear = LinearCapture(model)
    conv = DenseConvCapture(model) if args.arm == "dense" else None
    rulebooks = RulebookCapture(model) if args.arm == "sparse" else None
    rows: list[dict[str, Any]] = []

    def frame_loop() -> None:
        with torch.inference_mode():
            for ordinal, record in enumerate(records):
                input_value = make_sparse_input(record, device)
                if rulebooks is not None:
                    rulebooks.begin_frame(ordinal)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    detector_output = forward(args.arm, model, input_value, collect_timings=False)
                torch.cuda.synchronize()
                linear_rows = linear.finalize()
                conv_rows = conv.finalize() if conv is not None else rulebooks.finalize_frame()
                convolution_macs = int(sum(row["mac"] for row in conv_rows))
                linear_macs = int(sum(row["mac"] for row in linear_rows))
                realized_pairs = int(sum(row["realized_kernel_map_pairs"] if args.arm == "dense" else row["edge_count"] for row in conv_rows))
                rows.append({
                    "ordinal": ordinal,
                    "frame_id": record["frame_id"],
                    "sequence": record["frame_id"].split("/", 1)[0],
                    "arm": args.arm,
                    "input_coordinate_count": int(input_value.features.shape[0]),
                    "realized_kernel_map_pair_count": realized_pairs,
                    "convolution_macs": convolution_macs,
                    "linear_macs": linear_macs,
                    "logical_macs": convolution_macs + linear_macs,
                    "convolution_layers": conv_rows,
                    "linear_layers": linear_rows,
                    "output_shape": output_shape(args.arm, detector_output),
                })
                del detector_output, input_value
                if (ordinal + 1) % 100 == 0:
                    print(f"COMPUTE {args.arm} {ordinal + 1}/{len(records)}", flush=True)

    if rulebooks is None:
        frame_loop()
    else:
        with rulebooks:
            frame_loop()
    linear.close()
    if conv is not None:
        conv.close()
    require(len(rows) == contract["sample_manifest"]["frame_count"], "compute output count drift")
    metadata.update({
        "logical_mac_definition": {
            "sparse": "realized rulebook edges times input channels per group times output channels; 1x1 fast paths use realized input rows",
            "dense": "realized dense output positions times kernel volume times input channels per group times output channels",
            "linear": "realized rows times input features times output features",
            "one_mac_flop_equivalents": 2,
            "exclusions": contract["logical_mac_exclusions"],
        },
        "frame_count": len(rows),
    })
    verify_no_cotenant(contract, allow_self=True)
    write_jsonl_new(args.output, rows)
    require(args.metadata_output is not None, "compute metadata output missing")
    write_json_new(args.metadata_output, metadata)


def timing_split(arm: str, output: Any) -> dict[str, float | None]:
    if arm == "sparse":
        return {"conv_latency_ms": None, "support_latency_ms": None, "mask_latency_ms": None}
    totals = {"conv": 0.0, "support": 0.0, "mask": 0.0}
    for phases in output.timings_ms.values():
        for phase in totals:
            totals[phase] += float(phases.get(f"{phase}_ms", 0.0))
    return {
        "conv_latency_ms": totals["conv"],
        "support_latency_ms": totals["support"],
        "mask_latency_ms": totals["mask"],
    }


def run_latency(args: argparse.Namespace, contract: dict[str, Any], records: list[dict[str, Any]], panels: dict[str, Any], model: Any, device: torch.device, metadata: dict[str, Any]) -> None:
    require(args.repeat is not None and args.pass_position is not None, "latency repeat metadata missing")
    require(contract["repeat_orders"][args.repeat][args.pass_position] == args.arm, "repeat order drift")
    settle = settle_gpu(contract)
    warmup = panels["warmup_ordinals"]
    require(len(warmup) == 200, "warmup panel drift")
    with torch.inference_mode():
        for index, ordinal in enumerate(warmup):
            input_value = make_sparse_input(records[ordinal], device)
            torch.cuda.synchronize()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                value = forward(args.arm, model, input_value, collect_timings=False)
            torch.cuda.synchronize()
            del value, input_value
            if (index + 1) % 50 == 0:
                print(f"WARMUP r{args.repeat + 1} {args.arm} {index + 1}/200", flush=True)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for ordinal, record in enumerate(records):
            input_value = make_sparse_input(record, device)
            torch.cuda.synchronize()
            start.record()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                value = forward(args.arm, model, input_value, collect_timings=args.arm == "dense")
            end.record()
            torch.cuda.synchronize()
            full_ms = float(start.elapsed_time(end))
            split = timing_split(args.arm, value)
            if args.arm == "dense":
                subtotal = sum(float(split[key]) for key in split)
                require(subtotal <= full_ms + 0.25, f"dense timing ledger exceeds full forward: {subtotal} > {full_ms}")
            rows.append({
                "repeat": args.repeat,
                "pass_position": args.pass_position,
                "arm": args.arm,
                "ordinal": ordinal,
                "frame_id": record["frame_id"],
                "sequence": record["frame_id"].split("/", 1)[0],
                "full_latency_ms": full_ms,
                **split,
                "output_shape": output_shape(args.arm, value),
            })
            del value, input_value
            if (ordinal + 1) % 100 == 0:
                print(f"LATENCY r{args.repeat + 1} {args.arm} {ordinal + 1}/{len(records)}", flush=True)
    metadata.update({
        "repeat": args.repeat,
        "pass_position": args.pass_position,
        "repeat_order": contract["repeat_orders"][args.repeat],
        "warmup_ordinals": warmup,
        "settle": settle,
        "timing_scope": {
            "included": "full detector forward including temporal pooling, decode, NMS, dense support generation, and dense masking",
            "excluded": ["file loading", "preprocessing", "host-to-device copy", "SparseConvTensor construction"],
            "timer": "CUDA events; synchronize after each observation",
            "dense_split": "nested CUDA events for transplanted convolution, support generation, and masking",
        },
        "frame_count": len(rows),
    })
    verify_no_cotenant(contract, allow_self=True)
    write_jsonl_new(args.output, rows)
    require(args.metadata_output is not None, "latency metadata output missing")
    write_json_new(args.metadata_output, metadata)


def run_memory(args: argparse.Namespace, contract: dict[str, Any], records: list[dict[str, Any]], panels: dict[str, Any], model: Any, device: torch.device, metadata: dict[str, Any]) -> None:
    require(args.ordinal is not None and args.ordinal in panels["memory_ordinals"], "memory ordinal outside locked panel")
    record = records[args.ordinal]
    input_value = make_sparse_input(record, device)
    settle = settle_gpu(contract)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        warmup_output = forward(args.arm, model, input_value, collect_timings=False)
    torch.cuda.synchronize()
    del warmup_output
    gc.collect()
    torch.cuda.synchronize()
    baseline = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        detector_output = forward(args.arm, model, input_value, collect_timings=False)
    torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_allocated(device))
    require(peak >= baseline, "allocated peak below baseline")
    metadata.update({
        "batch_size": 1,
        "fresh_process": True,
        "ordinal": args.ordinal,
        "frame_id": record["frame_id"],
        "sequence": record["frame_id"].split("/", 1)[0],
        "input_coordinate_count": int(input_value.features.shape[0]),
        "warmup_forwards": 1,
        "input_built_before_peak_reset": True,
        "warmup_output_released_before_peak_reset": True,
        "post_input_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "output_shape": output_shape(args.arm, detector_output),
        "settle": settle,
    })
    verify_no_cotenant(contract, allow_self=True)
    write_json_new(args.output, metadata)


def main() -> int:
    args = parse_args()
    resolved = [args.repo.resolve(strict=True), args.contract.resolve(strict=True), args.manifest.resolve(strict=True), args.panels.resolve(strict=True), args.output.resolve(strict=False), args.oom_output.resolve(strict=False)]
    if args.metadata_output is not None:
        resolved.append(args.metadata_output.resolve(strict=False))
    assert_paths_unsealed(resolved)
    print("sealed_split_preflight: PASS", flush=True)
    contract = load_contract(args.contract)
    profile_root = (args.repo / contract["repo_relative_profile_dir"]).resolve(strict=True)
    for candidate in resolved[4:]:
        try:
            candidate.relative_to(profile_root)
        except ValueError as error:
            raise RuntimeError(f"worker output must stay inside {profile_root}: {candidate}") from error
    checkpoint = (args.repo / contract["checkpoint"]["path"]).resolve(strict=True)
    require(sha256_file(checkpoint) == contract["checkpoint"]["sha256"], "checkpoint SHA-256 mismatch before CUDA")
    require(sha256_file(args.manifest) == contract["sample_manifest"]["sha256"], "sample manifest SHA-256 mismatch before CUDA")
    require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER drift")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == contract["gpu"]["uuid"], "CUDA_VISIBLE_DEVICES drift")
    records, _ = load_records(args.manifest, args.repo, contract, inspect_arrays=False)
    panels = json.loads(args.panels.read_text(encoding="utf-8"))
    require(not args.output.exists(), f"output exists: {args.output}")
    if args.metadata_output is not None:
        require(not args.metadata_output.exists(), f"metadata output exists: {args.metadata_output}")
    target_gpu_snapshot(contract)
    verify_no_cotenant(contract, allow_self=False)
    device = torch.device("cuda:0")
    placement = verify_cuda_placement(contract)
    model, model_meta = build_arm(args.arm, args.repo, checkpoint, contract)
    model = model.to(device).eval()
    verify_no_cotenant(contract, allow_self=True)
    metadata = base_metadata(args, contract, model_meta, placement)
    if args.mode == "compute":
        run_compute(args, contract, records, model, device, metadata)
    elif args.mode == "latency":
        run_latency(args, contract, records, panels, model, device, metadata)
    else:
        run_memory(args, contract, records, panels, model, device, metadata)
    print(f"WORKER PASS {args.mode} {args.arm}", flush=True)
    return 0


if __name__ == "__main__":
    arguments = None
    try:
        raise SystemExit(main())
    except torch.cuda.OutOfMemoryError as error:
        arguments = parse_args()
        if not arguments.oom_output.exists():
            write_json_new(arguments.oom_output, {
                "schema_version": 1,
                "status": "OOM",
                "arm": arguments.arm,
                "phase": arguments.mode,
                "repeat": arguments.repeat,
                "ordinal": arguments.ordinal,
                "message": str(error),
                "no_crop_or_retry": True,
            })
        print(f"WORKER OOM {arguments.mode} {arguments.arm}", flush=True)
        raise SystemExit(42)
