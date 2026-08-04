#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_frame_list_sha256(frame_ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(frame_ids) + "\n").encode("utf-8")).hexdigest()


def write_json_new(path: Path, value: Any) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value["schema_version"] == 1, "contract schema drift")
    return value


def has_sealed_component(path: Path) -> bool:
    sealed = {"test", "canonical_test", "challenging_test"}
    return any(part.casefold() in sealed for part in path.parts)


def assert_paths_unsealed(paths: list[Path]) -> None:
    for path in paths:
        require(not has_sealed_component(path), f"sealed split path refused: {path}")


def resolve_repo_path(repo: Path, relative: str, must_exist: bool = True) -> Path:
    source = Path(relative)
    path = source.resolve(strict=must_exist) if source.is_absolute() else (repo / source).resolve(strict=must_exist)
    assert_paths_unsealed([path])
    return path


def load_records(manifest: Path, repo: Path, contract: dict[str, Any], inspect_arrays: bool) -> tuple[list[dict[str, Any]], list[int]]:
    rows = read_jsonl(manifest)
    require(len(rows) == contract["sample_manifest"]["frame_count"], "manifest row-count drift")
    seen: set[str] = set()
    counts: list[int] = []
    sequences: set[str] = set()
    for ordinal, row in enumerate(rows):
        require(row.get("ordinal") == ordinal, f"manifest ordinal drift at {ordinal}")
        require(row.get("val_list_index_zero_based") == ordinal * 20, f"manifest selection drift at {ordinal}")
        frame_id = row.get("frame_id")
        require(isinstance(frame_id, str) and frame_id.count("/") == 1, f"invalid frame_id at {ordinal}")
        sequence, frame = frame_id.split("/", 1)
        require(sequence and frame and frame_id not in seen, f"duplicate or invalid frame_id: {frame_id}")
        seen.add(frame_id)
        sequences.add(sequence)
        sparse_path = resolve_repo_path(repo, row["sparse_source"])
        row["resolved_sparse_source"] = str(sparse_path)
        if inspect_arrays:
            coords, _ = load_sparse_arrays(row)
            counts.append(int(coords.shape[0]))
    require(len(sequences) == contract["sample_manifest"]["represented_sequence_count"], "represented sequence-count drift")
    require(all(int(value) != int(contract["sample_manifest"]["omitted_sequence"]) for value in sequences), "omitted sequence is present")
    return rows, counts


def load_sparse_arrays(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = Path(record["resolved_sparse_source"])
    with np.load(path, allow_pickle=False) as payload:
        require({"coords", "feats"}.issubset(payload.files), f"missing sparse arrays: {path}")
        coords = payload["coords"].astype(np.int32, copy=False)
        feats = payload["feats"].astype(np.float32, copy=False)
        if "time_bins" in payload.files:
            require(int(payload["time_bins"]) == 16, f"time-bin drift: {path}")
    require(coords.ndim == 2 and coords.shape[1] == 3, f"coordinate shape drift: {path}")
    require(feats.shape == (coords.shape[0], 6), f"feature shape drift: {path}")
    require(coords.shape[0] > 0, f"empty input: {path}")
    require((coords[:, 0] >= 0).all() and (coords[:, 0] < 16).all(), f"time bounds: {path}")
    require((coords[:, 1:] >= 0).all() and (coords[:, 1:] < 640).all(), f"spatial bounds: {path}")
    return np.ascontiguousarray(coords), np.ascontiguousarray(feats)


def stable_activity_panels(records: list[dict[str, Any]], activities: list[int], contract: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(activities, dtype=np.int64)
    edges = np.quantile(values, np.linspace(0, 1, 11), method="linear")
    require(len(np.unique(edges)) == 11, f"activity decile edges collapsed: {edges.tolist()}")
    bins = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, 9)
    warmup: list[int] = []
    memory: list[int] = []
    for decile in range(10):
        members = np.flatnonzero(bins == decile).tolist()
        members.sort(key=lambda index: (hashlib.sha256(records[index]["frame_id"].encode()).hexdigest(), index))
        require(len(members) >= contract["warmup_per_activity_decile"], f"short activity decile {decile}")
        warmup.extend(members[: contract["warmup_per_activity_decile"]])
        memory.extend(members[: contract["memory_frames_per_activity_decile"]])
    return {
        "selection": "NumPy linear input-coordinate deciles; stable SHA256(frame_id), then ordinal",
        "input_coordinate_edges": edges.tolist(),
        "warmup_ordinals": warmup,
        "memory_ordinals": memory,
        "rows": [
            {
                "ordinal": index,
                "frame_id": records[index]["frame_id"],
                "input_coordinate_count": int(values[index]),
                "input_coordinate_activity_bin": int(bins[index]),
            }
            for index in range(len(records))
        ],
    }


def command_output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def normalize_uuid(value: str) -> str:
    return value.casefold().removeprefix("gpu-").replace("-", "")


def gpu_rows() -> list[dict[str, str]]:
    fields = ["index", "uuid", "name", "memory.used", "memory.free", "utilization.gpu", "pstate", "power.draw", "power.limit"]
    output = command_output(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    return [dict(zip(fields, [part.strip() for part in line.split(",")])) for line in output.splitlines()]


def target_gpu_snapshot(contract: dict[str, Any]) -> dict[str, str]:
    expected = contract["gpu"]
    matches = [row for row in gpu_rows() if normalize_uuid(row["uuid"]) == normalize_uuid(expected["uuid"])]
    require(len(matches) == 1, f"target GPU UUID inventory mismatch: {matches}")
    row = matches[0]
    require(row["name"] == expected["name"], f"GPU name mismatch: {row}")
    require(abs(float(row["power.limit"]) - expected["power_limit_watts"]) <= 0.01, f"GPU power-limit mismatch: {row}")
    return row


def compute_apps() -> list[dict[str, Any]]:
    output = command_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    rows = []
    for line in output.splitlines():
        if line.strip():
            uuid, pid, name, used = [part.strip() for part in line.split(",", 3)]
            rows.append({"gpu_uuid": uuid, "pid": int(pid), "process_name": name, "used_memory_mib": int(used)})
    return rows


def verify_no_cotenant(contract: dict[str, Any], allow_self: bool) -> dict[str, Any]:
    target = [row for row in compute_apps() if normalize_uuid(row["gpu_uuid"]) == normalize_uuid(contract["gpu"]["uuid"])]
    allowed = {os.getpid()} if allow_self else set()
    require(not [row for row in target if row["pid"] not in allowed], f"GPU co-tenant detected: {target}")
    if allow_self:
        require(any(row["pid"] == os.getpid() for row in target), f"own CUDA PID missing: {target}")
    else:
        require(not target, f"target GPU occupied before worker: {target}")
    return {"target_processes": target, "all_compute_processes": compute_apps()}


def verify_cuda_placement(contract: dict[str, Any]) -> dict[str, Any]:
    import torch

    require(torch.cuda.device_count() == 1, f"visible CUDA device count: {torch.cuda.device_count()}")
    properties = torch.cuda.get_device_properties(0)
    observed = properties.uuid.decode("ascii") if isinstance(properties.uuid, bytes) else str(properties.uuid)
    require(normalize_uuid(observed) == normalize_uuid(contract["gpu"]["uuid"]), f"CUDA UUID mismatch: {observed}")
    require(properties.name == contract["gpu"]["name"], f"CUDA GPU name mismatch: {properties.name}")
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    placement = verify_no_cotenant(contract, allow_self=True)
    return {"logical_device_count": 1, "logical_uuid": observed, "logical_name": properties.name, "placement": placement}


def settle_gpu(contract: dict[str, Any]) -> dict[str, Any]:
    import torch

    consecutive = 0
    previous_pstate: str | None = None
    snapshots = []
    for _ in range(40):
        torch.cuda.synchronize()
        verify_no_cotenant(contract, allow_self=True)
        row = target_gpu_snapshot(contract)
        snapshots.append(row)
        if int(row["utilization.gpu"]) <= 5 and row["pstate"] == previous_pstate:
            consecutive += 1
            if consecutive == 2:
                return {"status": "SETTLED", "snapshots": snapshots[-4:]}
        else:
            consecutive = 0
        previous_pstate = row["pstate"]
        time.sleep(0.5)
    raise ContractError(f"GPU failed to settle: {snapshots[-4:]}")


def make_sparse_input(record: dict[str, Any], device: Any) -> Any:
    import torch
    import spconv.pytorch as spconv

    coords, feats = load_sparse_arrays(record)
    indices = np.concatenate((np.zeros((coords.shape[0], 1), dtype=np.int32), coords), axis=1)
    return spconv.SparseConvTensor(
        features=torch.from_numpy(feats).to(device=device, dtype=torch.float32),
        indices=torch.from_numpy(indices).to(device=device, dtype=torch.int32),
        spatial_shape=[16, 640, 640],
        batch_size=1,
    )


def build_sparse_model(repo: Path, checkpoint_path: Path, contract: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import torch

    require(sha256_file(checkpoint_path) == contract["checkpoint"]["sha256"], "checkpoint SHA-256 mismatch before model construction")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from V2.models.sparse_voxel_det_ic import SparseVoxelDetIC

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(int(checkpoint.get("epoch", -1)) == contract["checkpoint"]["epoch"], "checkpoint epoch drift")
    embedded = checkpoint.get("config")
    require(isinstance(embedded, dict), "checkpoint embedded config missing")
    model_cfg = embedded.get("model", {})
    sparse_cfg = embedded.get("sparse", {})
    eval_cfg = embedded.get("eval", {})
    expected = contract["model"]
    for key in ("type", "in_channels", "num_classes", "backbone_size", "fpn_channels", "head_convs", "input_size", "prior_prob", "temporal_pool_mode"):
        observed = model_cfg.get(key)
        expected_value = expected[key]
        require(observed == expected_value, f"checkpoint model config drift: {key}={observed!r}")
    require(int(sparse_cfg.get("time_bins", -1)) == expected["time_bins"], "checkpoint time_bins drift")
    for key in ("score_thresh", "nms_thresh", "max_detections"):
        require(key in eval_cfg, f"checkpoint eval config missing: {key}")
    model = SparseVoxelDetIC(
        in_channels=expected["in_channels"],
        num_classes=expected["num_classes"],
        backbone_size=expected["backbone_size"],
        fpn_channels=expected["fpn_channels"],
        head_convs=expected["head_convs"],
        input_size=tuple(expected["input_size"]),
        time_bins=expected["time_bins"],
        prior_prob=expected["prior_prob"],
        score_thresh=float(eval_cfg["score_thresh"]),
        nms_thresh=float(eval_cfg["nms_thresh"]),
        max_detections=int(eval_cfg["max_detections"]),
        temporal_pool_mode=expected["temporal_pool_mode"],
    ).float().eval()
    state = checkpoint.get("ema_state_dict", {}).get("shadow")
    require(isinstance(state, dict) and state, "checkpoint EMA shadow missing")
    model.load_state_dict(state, strict=True)
    metadata = {"checkpoint_epoch": int(checkpoint["epoch"]), "weights": "ema_state_dict.shadow", "parameter_count": int(sum(value.numel() for value in model.parameters()))}
    return model, metadata


def build_arm(arm: str, repo: Path, checkpoint_path: Path, contract: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    sparse_model, metadata = build_sparse_model(repo, checkpoint_path, contract)
    if arm == "sparse":
        return sparse_model, metadata
    require(arm == "dense", f"unknown arm: {arm}")
    transplant_dir = resolve_repo_path(repo, "_mp1_2026-07-26", must_exist=True)
    if str(transplant_dir) not in sys.path:
        sys.path.insert(0, str(transplant_dir))
    import dense_transplant

    require(hasattr(dense_transplant, "_weight_rule_label"), "dense transplant lacks the measured stride-aware pointwise rule")
    parameters = inspect.signature(dense_transplant._krsc_to_conv3d).parameters
    require("stride" in parameters, "dense transplant is the archived pre-fix implementation; stride-aware pointwise layout is required")
    dense_model = dense_transplant.transplant_sparse_model(sparse_model)
    metadata["dense_transplant_sha256"] = sha256_file(Path(dense_transplant.__file__).resolve())
    metadata["dense_transplant_map"] = dense_model.transplant_map
    del sparse_model
    gc.collect()
    return dense_model, metadata


def popcount_int32_tensors(tensors: list[Any]) -> Any:
    import torch

    require(bool(tensors), "empty mask tensor list")
    total = torch.zeros((), dtype=torch.int64, device=tensors[0].device)
    for tensor in tensors:
        value = tensor.to(torch.int64) & 0xFFFFFFFF
        value = value - ((value >> 1) & 0x55555555)
        value = (value & 0x33333333) + ((value >> 2) & 0x33333333)
        value = (value + (value >> 4)) & 0x0F0F0F0F
        value = value + (value >> 8)
        value = value + (value >> 16)
        total = total + (value & 0x3F).sum(dtype=torch.int64)
    return total


class RulebookCapture:
    def __init__(self, model: Any) -> None:
        from spconv.pytorch.conv import SparseConvolution

        self.sparse_type = SparseConvolution
        self.expected_names = [name for name, module in model.named_modules() if isinstance(module, SparseConvolution)]
        self.name_by_id = {id(module): name for name, module in model.named_modules()}
        self.records: list[dict[str, Any]] = []
        self.current: list[dict[str, Any]] = []
        self.frame_ordinal: int | None = None

    def __enter__(self) -> "RulebookCapture":
        import torch
        from spconv.pytorch import ops as spconv_ops

        self.spconv_ops = spconv_ops
        self.original_forward = self.sparse_type.forward
        self.original_native = spconv_ops.get_indice_pairs
        self.original_implicit = spconv_ops.get_indice_pairs_implicit_gemm
        capture = self

        def native_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = capture.original_native(*args, **kwargs)
            require(capture.current and isinstance(result, (tuple, list)) and len(result) > 2 and torch.is_tensor(result[2]), "native pair API drift")
            capture.current[-1]["pair_calls"].append({"api": "Native", "edge_tensor": result[2].sum(dtype=torch.int64)})
            return result

        def implicit_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = capture.original_implicit(*args, **kwargs)
            require(capture.current and isinstance(result, (tuple, list)) and len(result) > 4, "implicit pair API drift")
            pair_fwd, masks = result[2], result[4]
            require(torch.is_tensor(pair_fwd) and isinstance(masks, list) and masks, "implicit pair payload drift")
            edge = (pair_fwd >= 0).sum(dtype=torch.int64)
            if capture.frame_ordinal == 0:
                require(int(edge.item()) == int(popcount_int32_tensors(masks).item()), f"ordinary pair-mask mismatch: {capture.current[-1]['name']}")
            capture.current[-1]["pair_calls"].append({"api": "MaskImplicitGemm", "edge_tensor": edge})
            return result

        def sparse_forward(module: Any, sparse_input: Any, *args: Any, **kwargs: Any) -> Any:
            name = capture.name_by_id.get(id(module))
            require(name is not None, "unmapped sparse convolution")
            row = {
                "name": name,
                "class_name": type(module).__name__,
                "in_channels": int(module.in_channels),
                "out_channels": int(module.out_channels),
                "groups": int(getattr(module, "groups", 1)),
                "kernel": [int(value) for value in module.kernel_size],
                "inverse": bool(module.inverse),
                "transposed": bool(module.transposed),
                "conv1x1_fast_path": bool(module.conv1x1),
                "n_in": int(sparse_input.features.shape[0]),
                "pair_calls": [],
            }
            inverse_data = None
            if module.inverse:
                inverse_data = sparse_input.find_indice_pair(module.indice_key)
                require(inverse_data is not None, f"missing inverse rulebook: {name}")
                edge_count = int((inverse_data.pair_bwd >= 0).sum().item())
                require(edge_count == int(popcount_int32_tensors(inverse_data.pair_mask_bwd_splits).item()), f"inverse pair-mask mismatch: {name}")
                row.update({"edge_count": edge_count, "edge_source": "reused_inverse_pair_bwd_valid_entries"})
            capture.current.append(row)
            try:
                output = capture.original_forward(module, sparse_input, *args, **kwargs)
            finally:
                require(capture.current and capture.current[-1] is row, "rulebook stack corruption")
                capture.current.pop()
            row["n_out"] = int(output.features.shape[0])
            if module.inverse:
                require(not row["pair_calls"] and torch.equal(output.indices, inverse_data.indices), f"inverse coordinate or pair-generation drift: {name}")
                row["inverse_pair_mask_popcount"] = int(row["edge_count"])
                row["inverse_output_equals_paired_downsample_input"] = True
            elif row["conv1x1_fast_path"]:
                require(not row["pair_calls"], f"unexpected 1x1 rulebook: {name}")
                row.update({"edge_count": row["n_in"], "edge_source": "one_to_one_torch_mm_fast_path"})
            else:
                require(len(row["pair_calls"]) == 1, f"pair-call count drift: {name}")
                call = row["pair_calls"][0]
                row.update({"edge_count": int(call["edge_tensor"].item()), "edge_source": f"{call['api']}_pair_fwd_valid_entries"})
            capture.records.append(row)
            return output

        spconv_ops.get_indice_pairs = native_wrapper
        spconv_ops.get_indice_pairs_implicit_gemm = implicit_wrapper
        self.sparse_type.forward = sparse_forward
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.sparse_type.forward = self.original_forward
        self.spconv_ops.get_indice_pairs = self.original_native
        self.spconv_ops.get_indice_pairs_implicit_gemm = self.original_implicit

    def begin_frame(self, ordinal: int) -> None:
        require(not self.records and not self.current, "unfinalized sparse frame")
        self.frame_ordinal = ordinal

    def finalize_frame(self) -> list[dict[str, Any]]:
        require(Counter(row["name"] for row in self.records) == Counter(self.expected_names), "sparse execution inventory drift")
        result = []
        for source in self.records:
            row = {key: value for key, value in source.items() if key != "pair_calls"}
            row["mac"] = int(row["edge_count"]) * (int(row["in_channels"]) // int(row["groups"])) * int(row["out_channels"])
            result.append(row)
        self.records = []
        self.frame_ordinal = None
        return result


class LinearCapture:
    def __init__(self, model: Any) -> None:
        import torch

        self.name_by_id = {id(module): name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
        self.expected: Counter[str] | None = None
        self.rows: list[dict[str, Any]] = []
        self.handles = [module.register_forward_hook(self._hook) for module in model.modules() if isinstance(module, torch.nn.Linear)]

    def _hook(self, module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        count = int(inputs[0].numel() // module.in_features)
        self.rows.append({"name": self.name_by_id[id(module)], "rows": count, "mac": count * int(module.in_features) * int(module.out_features)})

    def finalize(self) -> list[dict[str, Any]]:
        counter = Counter(row["name"] for row in self.rows)
        if self.expected is None:
            self.expected = counter
        require(counter == self.expected, "linear execution inventory drift")
        result, self.rows = self.rows, []
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class DenseConvCapture:
    def __init__(self, model: Any) -> None:
        import torch

        kinds = (torch.nn.Conv3d, torch.nn.ConvTranspose3d)
        self.name_by_id = {id(module): name for name, module in model.named_modules() if isinstance(module, kinds)}
        self.expected = Counter(self.name_by_id.values())
        self.rows: list[dict[str, Any]] = []
        self.handles = [module.register_forward_hook(self._hook) for module in model.modules() if isinstance(module, kinds)]

    def _hook(self, module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        name = self.name_by_id[id(module)]
        kernel_volume = int(np.prod(module.kernel_size))
        output_positions = int(output.numel() // module.out_channels)
        pairs = output_positions * kernel_volume
        self.rows.append({
            "name": name,
            "class_name": type(module).__name__,
            "output_shape": list(output.shape),
            "output_positions": output_positions,
            "kernel_volume": kernel_volume,
            "realized_kernel_map_pairs": pairs,
            "mac": pairs * (int(module.in_channels) // int(module.groups)) * int(module.out_channels),
        })

    def finalize(self) -> list[dict[str, Any]]:
        require(Counter(row["name"] for row in self.rows) == self.expected, "dense convolution execution inventory drift")
        result, self.rows = self.rows, []
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def process_identity() -> dict[str, Any]:
    fields = Path("/proc/self/stat").read_text(encoding="utf-8").split()
    return {"pid": os.getpid(), "process_start_ticks": int(fields[21])}


def release_cuda(*values: Any) -> None:
    import torch

    del values
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
