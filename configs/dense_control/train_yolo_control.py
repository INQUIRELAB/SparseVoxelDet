#!/usr/bin/env python3
"""Pinned single-GPU YOLO11 dense control; refuses any GPU except the assigned UUID."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA_YAML = BASE / "fred_canonical_val.yaml"
AUDIT = BASE / "dataset_audit.json"
ASSIGNED = {
    "n": "GPU-48d3a2b0-fc78-8bc8-fdce-5a246fdc4989",
    "s": "GPU-e0f8ae94-d53d-e1dd-63d5-730a80d0b6a4",
}
FORBIDDEN = {
    "GPU-b279b278-d3e7-eb16-73d2-f6f4b002276c",
    "GPU-1d11b997-90a9-ece7-9ce6-44ad85346817",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_row(uuid: str) -> dict[str, object]:
    query = "index,uuid,name,memory.total,memory.used,utilization.gpu"
    text = subprocess.check_output(["nvidia-smi", "-i", uuid, f"--query-gpu={query}", "--format=csv,noheader,nounits"], text=True).strip()
    idx, found_uuid, name, total, used, util = [x.strip() for x in text.split(",")]
    return {"physical_index": int(idx), "uuid": found_uuid, "name": name, "memory_total_mib": int(total), "memory_used_mib": int(used), "utilization_percent": int(util)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("n", "s"), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    expected_uuid = ASSIGNED[args.variant]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != expected_uuid or expected_uuid in FORBIDDEN:
        raise RuntimeError(f"GPU contract violation: expected CUDA_VISIBLE_DEVICES={expected_uuid}, got {visible!r}")
    audit = json.loads(AUDIT.read_text())
    if audit.get("status") != "ok" or audit.get("canonical_test_used") is not False:
        raise RuntimeError(f"Dataset audit failed or touched canonical_test: {AUDIT}")
    if audit["splits"]["train"]["paired_frame_count"] != 406701 or audit["splits"]["val"]["paired_frame_count"] != 103672:
        raise RuntimeError("Dataset frame counts changed")
    if args.variant == "n" and args.batch == 128:
        autobatch = json.loads((BASE / "autobatch_preflight.json").read_text())
        if not autobatch.get("requested_supported_by_autobatch") or autobatch.get("recommended_batch", 0) < 128:
            raise RuntimeError(f"Batch-128 not supported by recorded autobatch preflight: {autobatch}")
    if args.variant == "s":
        preflight_path = BASE / "autobatch_preflight_yolo11s.json"
        preflight = json.loads(preflight_path.read_text())
        if preflight.get("status") != "passed" or preflight.get("gpu", {}).get("uuid") != expected_uuid:
            raise RuntimeError(f"YOLO11s preflight contract failed: {preflight}")
        if args.batch != preflight.get("selected_fixed_batch") or args.batch > preflight.get("recommended_batch", 0):
            raise RuntimeError(f"YOLO11s batch does not match immutable preflight: requested={args.batch}, preflight={preflight}")
    gpu_before = gpu_row(expected_uuid)
    if gpu_before["memory_used_mib"] > 1024:
        raise RuntimeError(f"Assigned GPU is no longer free: {gpu_before}")

    os.chdir(BASE)
    import torch
    import ultralytics
    from ultralytics import YOLO

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != gpu_before["name"]:
        raise RuntimeError(f"CUDA mask mismatch: count={torch.cuda.device_count()} name={torch.cuda.get_device_name(0)} row={gpu_before}")
    weight_name = f"yolo11{args.variant}.pt"
    model = YOLO(weight_name)
    weight_path = BASE / weight_name
    run_name = f"yolo11{args.variant}_seed{args.seed}_e{args.epochs}_b{args.batch}"
    run_dir = BASE / "runs" / run_name
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite or auto-increment existing run: {run_dir}")
    train_args = {
        "data": str(DATA_YAML), "epochs": args.epochs, "imgsz": 640, "batch": args.batch,
        "device": 0, "workers": args.workers, "seed": args.seed, "deterministic": True,
        "single_cls": True, "pretrained": True, "optimizer": "auto", "amp": True,
        "patience": args.epochs, "close_mosaic": 10, "cache": False, "fraction": 1.0,
        "rect": False, "val": True, "save": True, "save_period": 1, "plots": False,
        "project": str(BASE / "runs"), "name": run_name, "exist_ok": False, "verbose": True,
    }
    contract = {
        "status": "launching", "pid": os.getpid(), "variant": f"yolo11{args.variant}",
        "pretrained_weight": str(weight_path), "pretrained_weight_sha256": sha256(weight_path),
        "dataset_audit": str(AUDIT), "dataset_audit_sha256": sha256(AUDIT),
        "data_yaml": str(DATA_YAML), "data_yaml_sha256": sha256(DATA_YAML),
        "gpu": gpu_before, "cuda_visible_devices": visible,
        "torch": torch.__version__, "cuda": torch.version.cuda, "ultralytics": ultralytics.__version__,
        "train_args": train_args,
    }
    contract_path = BASE / f"launch_contract_{run_name}.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(json.dumps(contract, indent=2), flush=True)
    results = model.train(**train_args)
    contract["status"] = "complete"
    contract["results_dict"] = getattr(results, "results_dict", {})
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
