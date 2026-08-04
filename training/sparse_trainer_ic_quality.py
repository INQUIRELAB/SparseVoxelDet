#!/usr/bin/env python3
"""
Stage B SparseVoxelDet trainer derived from the checkpoint-safe V82 trainer.

V82 changes vs original:
  1. Imports from v82 model and dataset modules
  2. Reads in_channels from config (default 6)
  3. Rectangular (H, W) collation support
  4. Native 1280×720 resolution

Usage:
    python V2/scripts/train_sparse_voxel_det_v82.py --config V2/configs/sparse_voxel_det_v82.yaml
"""
import argparse
import copy
import hashlib
import os
import sys
import time
import json
import math
import random
import subprocess
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, Optional, Any, List, Tuple

if __name__ == "__main__":
    raise SystemExit("Direct trainer entry is forbidden; use the protected DDP3 controller")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.amp import autocast, GradScaler
import spconv.pytorch as spconv
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from V2.models.sparse_voxel_det_v82 import SparseVoxelDet
from sparse_fcos_v1.scripts.sparse_event_dataset_v82 import (
    SparseEventDataset, sparse_collate_fn, make_collate_fn, create_sparse_tensor
)
from quality_aligned_loss import SparseVoxelDetLoss
from sparse_fcos_v1.scripts.ema import ModelEMA
from sparse_fcos_v1.scripts.metrics import MAPCalculator
from sparse_fcos_v1.scripts.evaluate_sparse_fcos import temporal_rerank_top1

QUALITY_DIAGNOSTIC_KEYS = (
    "num_gt", "num_gt_with_candidates", "gt_zero_candidates", "dynamic_k_sum",
    "num_pos_raw", "quota_fill_ratio", "quota_deficit", "conflict_sites",
    "gt_zero_after_conflict", "multi_gt_samples", "multi_gt_gt_zero_assigned",
    "candidate_count_mean", "candidate_count_max", "classification_quality_target_mean",
    "classification_quality_target_max", "decoded_iou_target_mean", "decoded_iou_target_max",
)

CHECKPOINT_COMMIT_HOOK = None
COLLECTIVE_CONSENSUS = None
GLOBAL_REDUCE_SCALARS = None
BROADCAST_RANK0_PAYLOAD = None
RUN_RANK0_STAGE = None
GATHER_RANK_ERRORS = None
ABORT_DISTRIBUTED_JOB = None
COLLECTIVE_STOP_REQUESTED = None
WORKER_CONTEXT = None

PARITY_SPLIT_ALLOWLIST = {
    "canonical_train",
    "canonical_test",
    "challenging_train",
    "challenging_test",
}

CORRECTED_LABEL_ROOT = (
    project_root / "data/datasets/fred_paper_parity/labels_rawcomplete"
).resolve()
CORRECTED_MANIFEST_SHA256 = "6a973831e215c733e77f4ba2553ae0e138a20cf01f1c5e30387292f52b2c56ee"
EXPECTED_LABEL_SPLITS = {
    "canonical_val_train": {
        "files": 406701,
        "bytes": 16090527,
        "boxes": 433424,
        "two_box_files": 26723,
        "sha256": "6ef5b023a0bbd91b02bc7b59d7e2493e5c2b61eab6d736ffefe5fa8cbd1fc4c2",
    },
    "canonical_val": {
        "files": 103672,
        "bytes": 4549954,
        "boxes": 121982,
        "two_box_files": 18310,
        "sha256": "e2a664a3fe8027743c4aab9ecf70126b5c5e232a0587cd9b433e020cf50bf1ad",
    },
}
QUALITY_RUNTIME_POLICY_VERSION = "ic-quality-ddp3-seed42-v2"
QUALITY_CHECKPOINT_LINEAGE = "ic-quality-aligned-ddp3-seed42"
QUALITY_RESUME_POLICY = "full-state; archived DDP3 commit; one verified load per rank"
QUALITY_BUILD_DIR = Path(__file__).resolve().parent
QUALITY_CONFIG_PATH = (QUALITY_BUILD_DIR / "ic_quality_ddp3_e20.yaml").resolve()
QUALITY_CONFIG_SHA256 = "1067e925c3ffe753c4bdbf5816e30a38d863b48d184d9638720c374cc86bbf42"
QUALITY_SPARSE_VALIDATOR_SHA256 = "375b55513952fe03190313089466b20ccbfad5558cd003f2027454c0728672ad"
QUALITY_ACTIVE_CLAIM = QUALITY_BUILD_DIR / "contracts_ddp3/ic_quality_ddp3_seed42.active"
QUALITY_OUTPUT_MARKER = ".quality_ddp3_writer_claim.json"
QUALITY_ORDERED_UUIDS = (
    "GPU-1d11b997-90a9-ece7-9ce6-44ad85346817",
    "GPU-2a7554bd-5a91-25ab-3338-e2308ecb2a27",
    "GPU-48d3a2b0-fc78-8bc8-fdce-5a246fdc4989",
)
QUALITY_FORBIDDEN_UUIDS = (
    "GPU-b279b278-d3e7-eb16-73d2-f6f4b002276c",
)
QUALITY_WORLD_SIZE = 3
QUALITY_SAMPLER_SEED = 42
QUALITY_PER_RANK_BATCH = 2
QUALITY_GLOBAL_BATCH = 6
QUALITY_ACCUMULATION_STEPS = 1
QUALITY_TRAIN_ROSTER_SAMPLES = 406701
QUALITY_SAMPLER_SAMPLES = 406701
QUALITY_OPTIMIZED_SAMPLES = 406698
QUALITY_OPTIMIZER_STEPS_PER_EPOCH = 67783
QUALITY_WARMUP_STEPS = 5000
QUALITY_EPOCHS = 20
QUALITY_TOTAL_OPTIMIZER_STEPS = 1355660
QUALITY_RUNTIME_SOURCES = {
    "strict_loss": QUALITY_BUILD_DIR / "strict_loss.py",
    "quality_aligned_loss": QUALITY_BUILD_DIR / "quality_aligned_loss.py",
    "models.snn.sparse_sew_resnet": project_root / "models/snn/sparse_sew_resnet.py",
    "se_per_sample_patch": project_root / "tools/sol_forensics/investigators/se_per_sample_patch.py",
    "V2": project_root / "V2/__init__.py",
    "V2.models": project_root / "V2/models/__init__.py",
    "V2.models.sparse_voxel_det_v82": project_root / "V2/models/sparse_voxel_det_v82.py",
    "V2.models.sparse_voxel_det_ic": project_root / "V2/models/sparse_voxel_det_ic.py",
    "sparse_fcos_v1": project_root / "sparse_fcos_v1/__init__.py",
    "sparse_fcos_v1.scripts": project_root / "sparse_fcos_v1/scripts/__init__.py",
    "sparse_fcos_v1.scripts.event_mosaic": project_root / "sparse_fcos_v1/scripts/event_mosaic.py",
    "sparse_fcos_v1.scripts.sparse_event_dataset_v82": project_root / "sparse_fcos_v1/scripts/sparse_event_dataset_v82.py",
    "sparse_fcos_v1.scripts.ema": project_root / "sparse_fcos_v1/scripts/ema.py",
    "sparse_fcos_v1.scripts.metrics": project_root / "sparse_fcos_v1/scripts/metrics.py",
    "sparse_fcos_v1.scripts.evaluate_sparse_fcos": project_root / "sparse_fcos_v1/scripts/evaluate_sparse_fcos.py",
    "sparse_tensor_contract_validator": project_root / "tools/validate_sparse_tensor_contract.py",
    "sparse_trainer_ic_quality": QUALITY_BUILD_DIR / "sparse_trainer_ic_quality.py",
}
QUALITY_BUILD_ARTIFACTS = {
    "strict_loss": QUALITY_BUILD_DIR / "strict_loss.py",
    "quality_loss": QUALITY_BUILD_DIR / "quality_aligned_loss.py",
    "trainer": QUALITY_BUILD_DIR / "sparse_trainer_ic_quality.py",
    "config": QUALITY_CONFIG_PATH,
    "launcher": QUALITY_BUILD_DIR / "train_ic_quality.py",
    "preflight": QUALITY_BUILD_DIR / "preflight_quality.py",
    "quality_tests": QUALITY_BUILD_DIR / "test_quality_aligned_loss.py",
    "contract_tests": QUALITY_BUILD_DIR / "test_ddp3_contracts.py",
}
QUALITY_REMOTE_ARTIFACTS = {
    "per_sample_se": project_root / "tools/sol_forensics/investigators/se_per_sample_patch.py",
    "sparse_sew": project_root / "models/snn/sparse_sew_resnet.py",
    "v2_package": project_root / "V2/__init__.py",
    "v2_models_package": project_root / "V2/models/__init__.py",
    "base_model": project_root / "V2/models/sparse_voxel_det_v82.py",
    "ic_model": project_root / "V2/models/sparse_voxel_det_ic.py",
    "sparse_fcos_package": project_root / "sparse_fcos_v1/__init__.py",
    "sparse_fcos_scripts_package": project_root / "sparse_fcos_v1/scripts/__init__.py",
    "event_mosaic": project_root / "sparse_fcos_v1/scripts/event_mosaic.py",
    "dataset": project_root / "sparse_fcos_v1/scripts/sparse_event_dataset_v82.py",
    "ema": project_root / "sparse_fcos_v1/scripts/ema.py",
    "metrics": project_root / "sparse_fcos_v1/scripts/metrics.py",
    "evaluator": project_root / "sparse_fcos_v1/scripts/evaluate_sparse_fcos.py",
    "sparse_contract_validator": project_root / "tools/validate_sparse_tensor_contract.py",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified_quality_config(config_path: str) -> Dict:
    resolved = Path(config_path).resolve()
    if resolved != QUALITY_CONFIG_PATH:
        raise RuntimeError(f"Unauthorized quality trainer config: {resolved}")
    source = resolved.read_bytes()
    actual_hash = hashlib.sha256(source).hexdigest()
    if actual_hash != QUALITY_CONFIG_SHA256:
        raise RuntimeError(
            f"Quality trainer config source drift: expected {QUALITY_CONFIG_SHA256}, got {actual_hash}"
        )
    config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise RuntimeError("Quality trainer config is malformed")
    return config


def expected_quality_runtime_contract() -> Dict[str, Any]:
    source_hashes = {
        "build": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in QUALITY_BUILD_ARTIFACTS.items()
        },
        "remote": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in QUALITY_REMOTE_ARTIFACTS.items()
        },
    }
    return {
        "experiment_name": "ic_quality_aligned_ddp3_seed42_e20",
        "seed": 42,
        "runtime_policy_version": QUALITY_RUNTIME_POLICY_VERSION,
        "checkpoint_lineage": QUALITY_CHECKPOINT_LINEAGE,
        "source_hashes": source_hashes,
        "runtime_source_sha256": {
            module_name: sha256_file(source_path)
            for module_name, source_path in QUALITY_RUNTIME_SOURCES.items()
        },
        "ordered_physical_uuids": list(QUALITY_ORDERED_UUIDS),
        "forbidden_physical_uuids": list(QUALITY_FORBIDDEN_UUIDS),
        "world_size": QUALITY_WORLD_SIZE,
        "local_world_size": QUALITY_WORLD_SIZE,
        "rank_mapping_policy": [
            {
                "rank": rank,
                "local_rank": rank,
                "uuid": QUALITY_ORDERED_UUIDS[rank],
            }
            for rank in range(QUALITY_WORLD_SIZE)
        ],
        "per_rank_batch_size": QUALITY_PER_RANK_BATCH,
        "global_effective_batch_size": QUALITY_GLOBAL_BATCH,
        "gradient_accumulation_steps": QUALITY_ACCUMULATION_STEPS,
        "sampler": {"seed": QUALITY_SAMPLER_SEED, "shuffle": True, "drop_last": True},
        "train_roster_samples": QUALITY_TRAIN_ROSTER_SAMPLES,
        "sampler_samples_per_epoch": QUALITY_SAMPLER_SAMPLES,
        "optimized_samples_per_epoch": QUALITY_OPTIMIZED_SAMPLES,
        "optimizer_steps_per_epoch": QUALITY_OPTIMIZER_STEPS_PER_EPOCH,
        "warmup_optimizer_steps": QUALITY_WARMUP_STEPS,
        "epochs": QUALITY_EPOCHS,
        "total_optimizer_steps": QUALITY_TOTAL_OPTIMIZER_STEPS,
        "scheduler": "full_cosine",
        "validation": {"owner_rank": 0, "sharded": False, "roster_samples": 103672},
        "resume_policy": QUALITY_RESUME_POLICY,
    }


def validate_quality_runtime_contract(contract: object) -> None:
    expected = expected_quality_runtime_contract()
    if contract != expected:
        raise RuntimeError("Quality trainer runtime contract does not match the verified build")


def fingerprint_label_split(split_dir: Path) -> Dict[str, Any]:
    digest = hashlib.sha256(b"sparsevoxeldet-label-split-v1\0")
    files = total_bytes = boxes = two_box_files = 0
    for entry in sorted(os.scandir(split_dir), key=lambda item: item.name):
        if not entry.is_file(follow_symlinks=False):
            raise RuntimeError(f"Unexpected corrected-label entry: {entry.path}")
        name = entry.name.encode("utf-8")
        source = Path(entry.path).read_bytes()
        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError(f"Non-UTF-8 corrected-label file: {entry.path}") from error
        rows = [line.split() for line in text.splitlines() if line.strip()]
        if len(rows) not in (1, 2):
            raise RuntimeError(f"Unexpected label cardinality at {entry.path}: {len(rows)}")
        for line_no, row in enumerate(rows, 1):
            if len(row) != 5 or row[0] != "0":
                raise RuntimeError(f"Invalid corrected-label row at {entry.path}:{line_no}: {row}")
            cx, cy, width, height = [float(value) for value in row[1:]]
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise RuntimeError(f"Out-of-range corrected-label row at {entry.path}:{line_no}: {row}")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(source).to_bytes(8, "big"))
        digest.update(source)
        files += 1
        total_bytes += len(source)
        boxes += len(rows)
        two_box_files += int(len(rows) == 2)
    return {
        "files": files,
        "bytes": total_bytes,
        "boxes": boxes,
        "two_box_files": two_box_files,
        "sha256": digest.hexdigest(),
    }


def assert_corrected_label_contract(
    label_dir: Path,
    train_split: str,
    val_split: str,
) -> None:
    resolved = label_dir.resolve()
    if resolved != CORRECTED_LABEL_ROOT:
        raise RuntimeError(
            f"Stage B requires label_dir={CORRECTED_LABEL_ROOT}, got {resolved}"
        )
    manifest = resolved / "MANIFEST.md"
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if manifest_sha != CORRECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Corrected-label MANIFEST hash mismatch: {manifest_sha}")
    if train_split != "canonical_val_train" or val_split != "canonical_val":
        raise RuntimeError(
            f"Corrected-label split mismatch: train={train_split}, val={val_split}"
        )
    measured = {
        split_name: fingerprint_label_split(resolved / split_name)
        for split_name in (train_split, val_split)
    }
    if measured != EXPECTED_LABEL_SPLITS:
        raise RuntimeError(
            f"Corrected-label split content mismatch: {measured} != {EXPECTED_LABEL_SPLITS}"
        )
    print(
        "Stage B corrected-label preflight: "
        f"root={resolved} manifest_sha256={manifest_sha} split_fingerprints={measured}",
        flush=True,
    )


def load_json_object(path: Path) -> Dict[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Malformed JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root is not an object in {path}")
    return payload


def require_worker_context() -> Dict[str, Any]:
    if not isinstance(WORKER_CONTEXT, dict):
        raise RuntimeError(
            "Direct trainer entry is forbidden; use the protected DDP3 controller"
        )
    required = {
        "rank", "local_rank", "world_size", "device", "output_dir", "claim_path",
        "stop_request_path", "runtime_contract", "launch_origin", "resume_receipt",
    }
    if set(WORKER_CONTEXT) != required:
        raise RuntimeError(
            f"Injected DDP3 worker context fields changed: "
            f"{sorted(set(WORKER_CONTEXT) ^ required)}"
        )
    helpers = {
        "COLLECTIVE_CONSENSUS": COLLECTIVE_CONSENSUS,
        "GLOBAL_REDUCE_SCALARS": GLOBAL_REDUCE_SCALARS,
        "BROADCAST_RANK0_PAYLOAD": BROADCAST_RANK0_PAYLOAD,
        "RUN_RANK0_STAGE": RUN_RANK0_STAGE,
        "GATHER_RANK_ERRORS": GATHER_RANK_ERRORS,
        "ABORT_DISTRIBUTED_JOB": ABORT_DISTRIBUTED_JOB,
        "COLLECTIVE_STOP_REQUESTED": COLLECTIVE_STOP_REQUESTED,
    }
    missing = sorted(name for name, helper in helpers.items() if not callable(helper))
    if missing:
        raise RuntimeError(f"DDP3 collective helpers were not injected: {missing}")
    return WORKER_CONTEXT


def validate_quality_output_claim(output_dir: Path, resume_path: Optional[Path]) -> None:
    context = require_worker_context()
    resolved_output = output_dir.resolve()
    claim_path = Path(context["claim_path"]).resolve()
    if claim_path != QUALITY_ACTIVE_CLAIM.resolve():
        raise RuntimeError(f"DDP3 trainer received a foreign controller claim: {claim_path}")
    if resolved_output != Path(context["output_dir"]).resolve():
        raise RuntimeError("DDP3 trainer output differs from injected output reservation")
    owner_path = claim_path / "owner.json"
    auth_path = claim_path / "authorization.json"
    registration_path = claim_path / "workers" / f"rank_{context['rank']}.json"
    for path in (owner_path, auth_path, registration_path):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"DDP3 authorization record is missing or invalid: {path}")
    owner = load_json_object(owner_path)
    registration = load_json_object(registration_path)
    expected_mode = "resume" if resume_path is not None else "fresh"
    if (
        owner.get("status") != "active"
        or owner.get("mode") != expected_mode
        or Path(str(owner.get("output_dir", ""))).resolve() != resolved_output
    ):
        raise RuntimeError(f"DDP3 controller claim does not own this trainer: {owner}")
    if (
        registration.get("status") != "registered"
        or registration.get("rank") != context["rank"]
        or registration.get("local_rank") != context["local_rank"]
        or registration.get("world_size") != QUALITY_WORLD_SIZE
        or registration.get("pid") != os.getpid()
        or registration.get("controller_id") != owner.get("controller_id")
        or registration.get("reservation_id") != owner.get("reservation_id")
    ):
        raise RuntimeError(f"DDP3 rank registration does not authorize this trainer: {registration}")
    marker_path = resolved_output / QUALITY_OUTPUT_MARKER
    if not resolved_output.is_dir() or marker_path.is_symlink() or not marker_path.is_file():
        raise RuntimeError(f"DDP3 output reservation is missing or invalid: {marker_path}")
    marker = load_json_object(marker_path)
    required_marker = {
        "status": "reserved",
        "mode": "fresh",
        "runtime_policy_version": QUALITY_RUNTIME_POLICY_VERSION,
        "seed": 42,
        "output_dir": str(resolved_output),
        "claim_path": str(QUALITY_ACTIVE_CLAIM.resolve()),
        "reservation_id": owner.get("reservation_id"),
    }
    for key, expected_value in required_marker.items():
        if marker.get(key) != expected_value:
            raise RuntimeError(f"DDP3 output reservation mismatch for {key}: {marker.get(key)!r}")
    expected_keys = set(required_marker) | {"claim_owner_sha256", "reserved_central"}
    if set(marker) != expected_keys:
        raise RuntimeError(
            f"DDP3 output reservation fields changed: {sorted(set(marker) ^ expected_keys)}"
        )
    if resume_path is not None:
        resolved_resume = resume_path.resolve()
        if resolved_resume.parent != resolved_output or not resolved_resume.is_file():
            raise RuntimeError(f"DDP3 resume checkpoint is outside owned output: {resolved_resume}")
        receipt = context.get("resume_receipt")
        if not isinstance(receipt, dict) or receipt.get("path") != str(resolved_resume):
            raise RuntimeError("DDP3 trainer resume path lacks the per-rank verified receipt")


class EpochSubsetSampler(torch.utils.data.Sampler):
    """Random subset of dataset, reshuffled each epoch.

    Limits iterations per epoch to max_samples / batch_size, making
    training feasible on large datasets (e.g., 351K -> 80K samples/epoch).
    """

    def __init__(self, dataset_size: int, max_samples: int, seed: int = 42):
        self.dataset_size = dataset_size
        self.max_samples = min(max_samples, dataset_size)
        self.epoch = 0
        self.seed = seed

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=g)[:self.max_samples]
        return iter(indices.tolist())

    def __len__(self):
        return self.max_samples


class EpochRandomSampler(torch.utils.data.Sampler):
    """Full-dataset permutation determined only by seed and epoch."""

    def __init__(self, dataset_size: int, seed: int):
        self.dataset_size = dataset_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.dataset_size, generator=generator).tolist())

    def __len__(self):
        return self.dataset_size


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python/NumPy/PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_worker_init_fn(base_seed: int):
    """Create deterministic DataLoader worker seed initializer."""
    def _worker_init_fn(worker_id: int) -> None:
        seed = int(base_seed) + int(worker_id)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
    return _worker_init_fn


def run_parity_preflight(require_exact: bool = True) -> None:
    """Run the hard parity contract checker and fail fast on mismatch."""
    cmd = [
        sys.executable,
        str(project_root / "tools" / "validate_paper_data_contract.py"),
    ]
    if require_exact:
        cmd.append("--require-exact")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Parity preflight failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def run_sparse_tensor_contract_preflight(
    sparse_root: Path,
    splits: List[str],
    allow_mixed_formats: bool = False,
    expected_format: Optional[str] = None,
    files_per_seq: int = 1,
    expected_time_bins: Optional[int] = None,
    enforce_uniform_time_bins: bool = False,
    enforce_per_seq_uniform_time_bins: bool = False,
    require_coords_time_bins_metadata: bool = False,
    report_json: Optional[Path] = None,
    expected_feat_channels: int = 2,
) -> None:
    """Run sparse tensor contract validator and fail fast on violations."""
    uniq_splits = sorted(set(str(s) for s in splits))
    validator_path = (project_root / "tools" / "validate_sparse_tensor_contract.py").resolve()
    validator_sha = sha256_file(validator_path)
    if validator_sha != QUALITY_SPARSE_VALIDATOR_SHA256:
        raise RuntimeError(
            f"Sparse tensor validator source drift: expected "
            f"{QUALITY_SPARSE_VALIDATOR_SHA256}, got {validator_sha}"
        )
    cmd = [
        sys.executable,
        str(validator_path),
        "--sparse-root",
        str(sparse_root),
        "--splits",
        *uniq_splits,
        "--files-per-seq",
        str(max(1, int(files_per_seq))),
    ]
    if allow_mixed_formats:
        cmd.append("--allow-mixed-formats")
    if expected_format:
        cmd.extend(["--expected-format", str(expected_format)])
    if expected_time_bins is not None:
        cmd.extend(["--expected-time-bins", str(int(expected_time_bins))])
    if enforce_uniform_time_bins:
        cmd.append("--enforce-uniform-time-bins")
    if enforce_per_seq_uniform_time_bins:
        cmd.append("--enforce-per-seq-uniform-time-bins")
    if require_coords_time_bins_metadata:
        cmd.append("--require-coords-time-bins-metadata")
    if expected_feat_channels != 2:
        cmd.extend(["--expected-feat-channels", str(int(expected_feat_channels))])
    if report_json is not None:
        cmd.extend(["--report-json", str(report_json)])

    environment = os.environ.copy()
    for key in (
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
        "MASTER_ADDR", "MASTER_PORT", "ROLE_RANK", "ROLE_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
        "SPARSEVOXELDET_DDP3_MODE", "SPARSEVOXELDET_DDP3_CLAIM",
        "SPARSEVOXELDET_DDP3_TOKEN", "SPARSEVOXELDET_DDP3_CONTROLLER_ID",
        "NCCL_ASYNC_ERROR_HANDLING", "TORCH_NCCL_ASYNC_ERROR_HANDLING", "NCCL_BLOCKING_WAIT",
    ):
        environment.pop(key, None)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=environment,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Sparse tensor contract preflight failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def assert_split_allowed_for_parity(split_name: str) -> None:
    """Reject split names outside paper parity allowlist."""
    if split_name not in PARITY_SPLIT_ALLOWLIST:
        raise ValueError(
            f"Parity mode requires split in {sorted(PARITY_SPLIT_ALLOWLIST)}, got '{split_name}'"
        )


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def append_jsonl(path: Optional[Path], payload: Dict[str, Any]) -> None:
    """Append a JSON line payload if path is configured."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def summarize_clip_telemetry(batch: Dict[str, Any]) -> Dict[str, float]:
    """Compute clipping telemetry summary for one collated batch."""
    telem = batch.get("clip_telemetry", {}) or {}
    raw = [int(v) for v in telem.get("raw_voxels", [])]
    kept = [int(v) for v in telem.get("kept_voxels", [])]
    frac = [float(v) for v in telem.get("clip_fraction", [])]
    clipped = [bool(v) for v in telem.get("clipped", [])]

    n = len(raw)
    if n == 0:
        return {
            "n_samples": 0.0,
            "raw_sum": 0.0,
            "kept_sum": 0.0,
            "clip_fraction_sum": 0.0,
            "clipped_count": 0.0,
        }

    return {
        "n_samples": float(n),
        "raw_sum": float(sum(raw)),
        "kept_sum": float(sum(kept)),
        "clip_fraction_sum": float(sum(frac)),
        "clipped_count": float(sum(1 for v in clipped if v)),
    }


def tensors_finite(obj: Any) -> bool:
    """Return True if all tensors in nested object are finite."""
    if isinstance(obj, torch.Tensor):
        return bool(torch.isfinite(obj).all().item())
    if isinstance(obj, dict):
        return all(tensors_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(tensors_finite(v) for v in obj)
    return True


# (resolve_query_assignment_mode removed — not used by SparseVoxelDet)


def _grad_l2_norm(grads: List[Optional[torch.Tensor]]) -> float:
    total = 0.0
    for g in grads:
        if g is None:
            continue
        total += float(torch.sum(g.detach().float() * g.detach().float()).item())
    return math.sqrt(total) if total > 0.0 else 0.0


def compute_per_loss_grad_norms(
    losses: Dict[str, Any],
    model: nn.Module,
    component_keys: List[str],
) -> Dict[str, float]:
    """Compute per-loss gradient norms for diagnostics (expensive; interval-gated)."""
    params = [p for p in model.parameters() if p.requires_grad]
    out: Dict[str, float] = {}
    if not params:
        return out

    for key in component_keys:
        raw = losses.get(key)
        if not isinstance(raw, torch.Tensor) or not raw.requires_grad:
            continue
        grads = torch.autograd.grad(raw, params, retain_graph=True, allow_unused=True)
        out[key] = _grad_l2_norm(grads)
    return out


def _build_weight_decay_param_groups(model: nn.Module, weight_decay: float) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Split parameters into decay/no-decay groups.

    Do not apply weight decay to:
    - bias parameters
    - norm layer parameters
    - 1D scale/shift parameters
    """
    norm_modules = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.GroupNorm,
        nn.LayerNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LocalResponseNorm,
    )

    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    seen: set[int] = set()

    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)

            if name.endswith("bias") or isinstance(module, norm_modules) or param.ndim <= 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    # Safety net: include any trainable params not visited above.
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        if param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    stats = {
        "decay_tensors": len(decay_params),
        "no_decay_tensors": len(no_decay_params),
        "decay_params": sum(p.numel() for p in decay_params),
        "no_decay_params": sum(p.numel() for p in no_decay_params),
    }
    return groups, stats


def create_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    """Create optimizer from config."""
    opt_config = config.get('training', {})
    lr = opt_config.get('lr', 0.001)
    weight_decay = opt_config.get('weight_decay', 0.0001)
    optimizer_type = opt_config.get('optimizer', 'AdamW')

    param_groups, group_stats = _build_weight_decay_param_groups(model, weight_decay)

    if optimizer_type == 'AdamW':
        optimizer = optim.AdamW(param_groups, lr=lr)
    elif optimizer_type == 'SGD':
        momentum = opt_config.get('momentum', 0.9)
        optimizer = optim.SGD(param_groups, lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # Lightweight metadata for startup logging/debug.
    setattr(optimizer, "_wd_group_stats", group_stats)
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict, steps_per_epoch: int) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler.

    `steps_per_epoch` should be the number of optimizer steps, not dataloader
    iterations, so schedules stay aligned when gradient accumulation is used.
    """
    train_config = config.get('training', {})
    epochs = train_config.get('epochs', 100)
    scheduler_type = train_config.get('scheduler', 'cosine')

    # Warmup: prefer explicit warmup_steps; fall back to warmup_epochs * steps_per_epoch.
    # CRITICAL FIX: warmup_epochs=5 with 63K steps/epoch = 318K warmup steps (37h!).
    # Default to 3000 steps (~3 minutes) if neither key is set.
    if 'warmup_steps' in train_config and int(train_config['warmup_steps']) > 0:
        warmup_steps = int(train_config['warmup_steps'])
    else:
        warmup_epochs = train_config.get('warmup_epochs', 0)
        warmup_steps = max(0, warmup_epochs * steps_per_epoch)
    if warmup_steps > 10000:
        import warnings
        warnings.warn(
            f"warmup_steps={warmup_steps} is very large (>{warmup_steps/steps_per_epoch:.1f} epochs, "
            f"~{warmup_steps * 2 / 3600:.1f}h at 19 samples/s). Did you mean warmup_steps, not warmup_epochs?"
        )
    total_steps = max(1, epochs * steps_per_epoch)

    if scheduler_type == 'cosine':
        # Warmup + cosine decay
        decay_steps = max(total_steps - warmup_steps, 1)

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / decay_steps
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == 'step':
        step_size = max(1, train_config.get('step_size', 30) * steps_per_epoch)
        gamma = train_config.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        return optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.0)


def generate_points_for_levels(feature_sizes, strides, device):
    """Generate grid points for all FPN levels."""
    points = []
    for (h, w), stride in zip(feature_sizes, strides):
        y = torch.arange(0, h, device=device) * stride + stride // 2
        x = torch.arange(0, w, device=device) * stride + stride // 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        pts = torch.stack([xx.flatten(), yy.flatten()], dim=1).float()
        points.append(pts)
    return points


class CollectiveBatchSkip(Exception):
    def __init__(self, reason: str, stage: str):
        super().__init__(reason)
        self.reason = reason
        self.stage = stage


def prepare_ddp_batch(
    model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    default_input_size,
    strict_finite_checks: bool,
):
    sparse_input = create_sparse_tensor(batch, device)
    if strict_finite_checks and not tensors_finite(batch.get("feats")):
        raise CollectiveBatchSkip("non-finite input features", "input_features")
    if strict_finite_checks and not tensors_finite(sparse_input.features):
        raise CollectiveBatchSkip("non-finite sparse tensor features", "sparse_tensor")
    cur_size = batch.get("input_size", tuple(default_input_size))
    if isinstance(cur_size, (int, float)):
        cur_input_size = [int(cur_size), int(cur_size)]
    elif isinstance(cur_size, (list, tuple)):
        cur_input_size = list(cur_size)
    else:
        cur_input_size = list(default_input_size)
    raw = model.module if hasattr(model, "module") else model
    size_key = (cur_input_size[0], cur_input_size[1])
    if size_key != (raw.input_size[0], raw.input_size[1]):
        raw.input_size = tuple(cur_input_size)
    return (
        sparse_input,
        int(batch["batch_size"]),
        [boxes.to(device) for boxes in batch["gt_boxes"]],
        [labels.to(device) for labels in batch["gt_labels"]],
    )


def forward_ddp_loss(
    model: nn.Module,
    sparse_input,
    batch_size: int,
    gt_boxes,
    gt_labels,
    loss_fn: nn.Module,
    use_amp: bool,
    amp_dtype,
    strict_finite_checks: bool,
):
    if use_amp:
        with autocast("cuda", dtype=amp_dtype):
            outputs = model(sparse_input, batch_size, return_loss_inputs=True)
            losses = loss_fn(outputs, gt_boxes, gt_labels)
    else:
        outputs = model(sparse_input, batch_size, return_loss_inputs=True)
        losses = loss_fn(outputs, gt_boxes, gt_labels)
    loss = losses["loss"]
    if strict_finite_checks and not tensors_finite(outputs):
        raise CollectiveBatchSkip("non-finite model outputs", "forward_outputs")
    if strict_finite_checks and not tensors_finite(losses):
        raise CollectiveBatchSkip("non-finite loss component", "loss_terms")
    if not torch.isfinite(loss):
        raise CollectiveBatchSkip("non-finite total loss", "loss_total")
    return losses, loss


def collective_next_batch(iterator, device: torch.device):
    import torch.distributed as dist

    batch = None
    state: Dict[str, Optional[str]]
    try:
        batch = next(iterator)
        state = {"status": "batch", "error": None}
    except StopIteration:
        state = {"status": "end", "error": None}
    except BaseException as error:
        state = {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }
    gathered: List[object] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, state)
    statuses = [item.get("status") if isinstance(item, dict) else None for item in gathered]
    if statuses == ["end"] * dist.get_world_size():
        return False, None
    if statuses != ["batch"] * dist.get_world_size():
        ABORT_DISTRIBUTED_JOB(
            f"DDP3 DataLoader states diverged before batch collectives: {gathered}"
        )
    return True, batch


def resolve_batch_consensus(
    local_code: int,
    local_error: Optional[str],
    device: torch.device,
    phase: str,
) -> int:
    decision = COLLECTIVE_CONSENSUS(local_code, device)
    if decision == 3:
        errors = GATHER_RANK_ERRORS(local_error)
        ABORT_DISTRIBUTED_JOB(
            f"DDP3 {phase} failed before backward: "
            + ("; ".join(errors) if errors else "unknown rank-local failure")
        )
    if decision not in (0, 1, 2):
        ABORT_DISTRIBUTED_JOB(f"DDP3 {phase} produced invalid consensus code {decision}")
    return decision


def synchronized_scaler_scale(scaler: GradScaler, device: torch.device) -> float:
    scale = float(scaler.get_scale())
    reduced = GLOBAL_REDUCE_SCALARS(
        {"scale_min": scale, "scale_max": scale},
        {"scale_min": "min", "scale_max": "max"},
        device,
    )
    if (
        not math.isfinite(reduced["scale_min"])
        or reduced["scale_min"] <= 0
        or reduced["scale_min"] != reduced["scale_max"]
    ):
        ABORT_DISTRIBUTED_JOB(f"DDP3 GradScaler state diverged: {reduced}")
    return reduced["scale_min"]


def collective_scaler_backoff(scaler: GradScaler, device: torch.device) -> float:
    current = synchronized_scaler_scale(scaler, device)
    next_scale = current * float(scaler.get_backoff_factor())
    if not math.isfinite(next_scale) or next_scale <= 0:
        ABORT_DISTRIBUTED_JOB(f"DDP3 GradScaler backoff is invalid: {next_scale}")
    scaler.update(new_scale=next_scale)
    return synchronized_scaler_scale(scaler, device)


def scaler_found_inf(scaler: GradScaler, optimizer: optim.Optimizer) -> bool:
    found_inf = scaler._found_inf_per_device(optimizer)
    if not isinstance(found_inf, dict) or not found_inf:
        raise RuntimeError("DDP3 GradScaler did not record unscale overflow state")
    total = sum(float(value.item()) for value in found_inf.values())
    if not math.isfinite(total) or total < 0:
        raise RuntimeError(f"DDP3 GradScaler overflow state is invalid: {total}")
    return total > 0


def reduce_ddp_epoch_state(
    local_state: Dict[str, float],
    device: torch.device,
    world_size: int = QUALITY_WORLD_SIZE,
) -> Dict[str, float]:
    sum_keys = {
        "loss_sum", "cls_loss_sum", "reg_loss_sum", "ctr_loss_sum",
        "iou_quality_loss_sum", "proposal_loss_sum", "ranking_loss_sum",
        "uncertainty_loss_sum", "positive_query_ratio_sum", "ranking_gap_sum",
        "near_boundary_mass_sum", "proposal_recall16_sum", "proposal_recall32_sum",
        "proposal_recall64_sum", "proposal_recall128_sum", "sample_count", "positive_count",
        "quality_num_gt", "quality_num_gt_with_candidates", "quality_gt_zero_candidates",
        "quality_dynamic_k_sum", "quality_num_pos_raw", "quality_quota_deficit",
        "quality_conflict_sites", "quality_gt_zero_after_conflict", "quality_multi_gt_samples",
        "quality_multi_gt_gt_zero_assigned", "quality_candidate_total", "quality_cls_total",
        "quality_iou_total", "clip_sample_count", "clip_raw_sum", "clip_kept_sum",
        "clip_fraction_sum", "clip_clipped_count", "sanitized_grad_steps",
    }
    max_keys = {
        "quality_candidate_count_max", "quality_classification_target_max",
        "quality_decoded_iou_target_max", "max_consecutive_nonfinite", "elapsed_seconds",
    }
    synchronized_keys = {
        "optimizer_steps", "optimizer_steps_completed", "skipped_non_finite",
        "skipped_non_finite_grad", "skipped_oom", "processed_batches",
        "successful_batches", "nonfinite_events", "aborted_early", "stopped_by_controller",
        "learning_rate",
    }
    special_keys = {"first_nonfinite_batch"}
    expected = sum_keys | max_keys | synchronized_keys | special_keys
    if set(local_state) != expected:
        raise RuntimeError(
            "DDP3 epoch reduction fields changed: "
            f"{sorted(set(local_state) ^ expected)}"
        )
    if world_size != QUALITY_WORLD_SIZE:
        raise RuntimeError(f"DDP3 epoch reduction requires world size 3, got {world_size}")

    values: Dict[str, float] = {}
    operations: Dict[str, str] = {}
    for key in sorted(sum_keys):
        values[key] = float(local_state[key])
        operations[key] = "sum"
    for key in sorted(max_keys):
        values[key] = float(local_state[key])
        operations[key] = "max"
    for key in sorted(synchronized_keys):
        values[f"{key}__min"] = float(local_state[key])
        operations[f"{key}__min"] = "min"
        values[f"{key}__max"] = float(local_state[key])
        operations[f"{key}__max"] = "max"
    values["first_nonfinite_batch"] = float(local_state["first_nonfinite_batch"])
    operations["first_nonfinite_batch"] = "min"

    reduced = GLOBAL_REDUCE_SCALARS(values, operations, device)
    result = {key: reduced[key] for key in sum_keys | max_keys | special_keys}
    for key in synchronized_keys:
        minimum = reduced[f"{key}__min"]
        maximum = reduced[f"{key}__max"]
        if minimum != maximum:
            raise RuntimeError(
                f"DDP3 synchronized epoch counter diverged for {key}: "
                f"min={minimum}, max={maximum}"
            )
        result[key] = maximum
    return result


def one_uuid_child_environment(uuid: str) -> Dict[str, str]:
    if uuid not in QUALITY_ORDERED_UUIDS:
        raise RuntimeError(f"Unauthorized DDP3 child GPU UUID: {uuid}")
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = uuid
    return environment


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    config: Dict,
    scaler: Optional[GradScaler] = None,
    logger = None,
    ema: Optional[ModelEMA] = None,
    output_dir: Optional[Path] = None,
    start_batch_idx: int = 0,
    start_optimizer_steps_in_epoch: int = 0,
    start_optimizer_steps_completed: int = 0,
) -> Dict[str, float]:
    """Train for one epoch.

    Supports mid-epoch checkpointing and heartbeat writes (controlled by
    training.checkpoint_interval_minutes in config, default 30 min).

    If start_batch_idx > 0, skip batches [0..start_batch_idx) — used when
    resuming from a mid-epoch checkpoint to avoid repeating work.
    """
    model.train()

    # SparseVoxelDet: no TQDet/FCOS dispatch needed
    grad_clip = config.get('training', {}).get('grad_clip', 10.0)
    use_amp = config.get('training', {}).get('use_amp', True) and device.type == "cuda"
    accumulation_steps = max(1, config.get('training', {}).get('gradient_accumulation_steps', 1))
    per_loss_grad_diag_interval = int(config.get('training', {}).get('per_loss_grad_diag_interval', 0))
    flush_partial_accumulation = config.get('training', {}).get('flush_partial_accumulation', False)
    if accumulation_steps != QUALITY_ACCUMULATION_STEPS or flush_partial_accumulation:
        raise RuntimeError("DDP3 requires accumulation=1 with no partial-window flush")
    nan_grad_action = str(config.get('training', {}).get('nan_grad_action', 'skip')).lower()
    if nan_grad_action != 'skip':
        raise RuntimeError("DDP3 requires collective gradient skip; sanitization is forbidden")
    mem_cleanup_interval = int(config.get('training', {}).get('mem_cleanup_interval', 2000))
    runtime_cfg = config.get('_runtime', {})
    strict_finite_checks = bool(runtime_cfg.get('strict_finite_checks', True))
    abort_on_skip_rate = float(runtime_cfg.get('abort_on_skip_rate', 0.10))
    abort_on_consecutive_nonfinite = int(runtime_cfg.get('abort_on_consecutive_nonfinite', 200))
    is_main_process = bool(runtime_cfg.get('is_main_process', True))
    worker_context = require_worker_context()
    rank = int(worker_context["rank"])
    if is_main_process != (rank == 0):
        raise RuntimeError("DDP3 rank-0 writer identity drifted")
    stop_request_path = Path(worker_context["stop_request_path"])
    amp_mode = str(runtime_cfg.get('amp_mode', 'fp16')).lower()
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    finite_trace_dir = Path(runtime_cfg['finite_trace_dir']) if runtime_cfg.get('finite_trace_dir') else None
    max_train_batches = runtime_cfg.get('max_train_batches')
    if max_train_batches is not None:
        max_train_batches = int(max_train_batches)
    nonfinite_trace_path = (finite_trace_dir / "nonfinite_batches.jsonl") if (finite_trace_dir and is_main_process) else None
    batch_health_path = (finite_trace_dir / "batch_health.jsonl") if (finite_trace_dir and is_main_process) else None

    # Mid-epoch checkpoint + heartbeat interval (minutes)
    ckpt_interval_min = float(config.get('training', {}).get('checkpoint_interval_minutes', 30))
    ckpt_interval_sec = ckpt_interval_min * 60.0
    last_ckpt_time = time.time()  # Track wall-time since last mid-epoch save

    per_loss_keys = [
        "cls_loss_raw",
        "ctr_loss_raw",
        "reg_loss_raw",
    ]

    # Default feature sizes
    default_input_size = config.get('model', {}).get('input_size', [640, 640])

    # Metrics
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_ctr_loss = 0.0
    total_iouq_loss = 0.0
    total_proposal_loss = 0.0
    total_ranking_loss = 0.0
    total_uncertainty_loss = 0.0
    total_positive_query_ratio = 0.0
    total_ranking_gap = 0.0
    total_near_boundary_mass = 0.0
    total_proposal_recall16 = 0.0
    total_proposal_recall32 = 0.0
    total_proposal_recall64 = 0.0
    total_proposal_recall128 = 0.0
    total_samples = 0
    total_pos = 0
    quality_diag_sums = {key: 0.0 for key in QUALITY_DIAGNOSTIC_KEYS}
    quality_candidate_total = 0.0
    quality_cls_total = 0.0
    quality_iou_total = 0.0
    clip_n_samples = 0.0
    clip_raw_sum = 0.0
    clip_kept_sum = 0.0
    clip_fraction_sum = 0.0
    clip_clipped_count = 0.0
    optimizer_steps = int(start_optimizer_steps_in_epoch)
    optimizer_steps_completed = int(start_optimizer_steps_completed)
    if optimizer_steps < 0 or optimizer_steps_completed < 0:
        raise RuntimeError("DDP3 resume optimizer lineage is negative")
    skipped_non_finite = 0
    skipped_non_finite_grad = 0
    skipped_oom = 0
    micro_batches_in_window = 0
    sanitized_grad_steps = 0
    processed_batches = 0
    successful_batches = 0
    nonfinite_events = 0
    consecutive_nonfinite = 0
    max_consecutive_nonfinite = 0
    first_nonfinite_batch = None
    rolling_skip = deque(maxlen=500)
    aborted_early = False
    abort_reason = ""
    stopped_by_controller = False
    emergency_checkpoint = ""
    latest_loss_grad_norms: Dict[str, float] = {}

    if hasattr(loss_fn, "set_epoch"):
        loss_fn.set_epoch(epoch)

    start_time = time.time()

    def commit_emergency(next_batch_idx: int) -> str:
        if output_dir is None:
            raise RuntimeError("DDP3 emergency checkpoint requires an owned output directory")
        checkpoint_path = output_dir / "emergency_stop.pt"

        def writer():
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                {
                    "status": "controller_power_stop",
                    "optimizer_steps": optimizer_steps,
                    "optimizer_steps_completed": optimizer_steps_completed,
                    "next_batch_idx": next_batch_idx,
                },
                checkpoint_path,
                config,
                ema=ema,
                batch_idx=next_batch_idx - 1,
                optimizer_steps_in_epoch=optimizer_steps,
                optimizer_steps_completed=optimizer_steps_completed,
                next_batch_idx=next_batch_idx,
            )
            return str(checkpoint_path.resolve())

        return str(run_rank0_stage(rank, "emergency full-state checkpoint", writer))

    data_iterator = iter(dataloader)
    for batch_idx in range(len(dataloader)):
        has_batch, batch = collective_next_batch(data_iterator, device)
        if not has_batch:
            ABORT_DISTRIBUTED_JOB(
                f"DDP3 DataLoader ended at batch {batch_idx} before expected {len(dataloader)}"
            )
        # Mid-epoch resume: skip already-processed batches
        # NOTE: Do NOT call scheduler.step() here — scheduler state was
        # already restored from checkpoint. Stepping again would double-count
        # and permanently misalign LR for all remaining epochs.
        if batch_idx < start_batch_idx:
            continue
        if max_train_batches is not None and batch_idx >= max_train_batches:
            if is_main_process:
                print(f"  Reached max_train_batches={max_train_batches}; ending epoch early.")
            break
        if aborted_early:
            break
        if COLLECTIVE_STOP_REQUESTED(stop_request_path, device):
            emergency_checkpoint = commit_emergency(batch_idx)
            stopped_by_controller = True
            abort_reason = "controller watchdog requested a power stop"
            break
        processed_batches += 1
        sample_ids = list(batch.get('sample_ids', []))
        clip_stats = summarize_clip_telemetry(batch)
        clip_n_samples += clip_stats["n_samples"]
        clip_raw_sum += clip_stats["raw_sum"]
        clip_kept_sum += clip_stats["kept_sum"]
        clip_fraction_sum += clip_stats["clip_fraction_sum"]
        clip_clipped_count += clip_stats["clipped_count"]
        skip_reason = ""
        skip_stage = ""

        def register_skip(reason: str, stage: str, bucket: str, details: Optional[Dict[str, Any]] = None, reset_scaler: bool = False) -> None:
            nonlocal skipped_non_finite, skipped_non_finite_grad, skipped_oom
            nonlocal micro_batches_in_window, consecutive_nonfinite, max_consecutive_nonfinite
            nonlocal aborted_early, abort_reason, first_nonfinite_batch, nonfinite_events
            nonlocal skip_reason, skip_stage
            skip_reason = reason
            skip_stage = stage

            if bucket == "oom":
                skipped_oom += 1
                consecutive_nonfinite = 0
            elif bucket == "non_finite_grad":
                skipped_non_finite_grad += 1
                consecutive_nonfinite += 1
                nonfinite_events += 1
            else:
                skipped_non_finite += 1
                consecutive_nonfinite += 1
                nonfinite_events += 1

            max_consecutive_nonfinite = max(max_consecutive_nonfinite, consecutive_nonfinite)
            if bucket != "oom" and first_nonfinite_batch is None:
                first_nonfinite_batch = batch_idx

            if bucket != "oom":
                append_jsonl(
                    nonfinite_trace_path,
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "stage": stage,
                        "reason": reason,
                        "sample_ids": sample_ids[:16],
                        "details": details or {},
                    },
                )

            rolling_skip.append(1)
            window_skip_rate = sum(rolling_skip) / max(len(rolling_skip), 1)

            # Hard gates
            if strict_finite_checks and bucket != "oom" and batch_idx < 200:
                aborted_early = True
                abort_reason = f"Non-finite detected in warmup window at batch {batch_idx}"
            elif strict_finite_checks and consecutive_nonfinite > abort_on_consecutive_nonfinite:
                aborted_early = True
                abort_reason = (
                    f"Consecutive non-finite batches exceeded threshold: "
                    f"{consecutive_nonfinite} > {abort_on_consecutive_nonfinite}"
                )
            elif strict_finite_checks and len(rolling_skip) == rolling_skip.maxlen and window_skip_rate > abort_on_skip_rate:
                aborted_early = True
                abort_reason = (
                    f"Rolling skip rate exceeded threshold over {rolling_skip.maxlen} batches: "
                    f"{window_skip_rate:.3f} > {abort_on_skip_rate:.3f}"
                )

            append_jsonl(
                batch_health_path,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "status": "skipped",
                    "stage": stage,
                    "reason": reason,
                    "bucket": bucket,
                    "sample_ids": sample_ids[:16],
                    "rolling_skip_rate": window_skip_rate,
                    "consecutive_nonfinite": consecutive_nonfinite,
                    "aborted_early": aborted_early,
                    "abort_reason": abort_reason,
                },
            )

            optimizer.zero_grad(set_to_none=True)
            micro_batches_in_window = 0
            if reset_scaler and scaler is not None:
                collective_scaler_backoff(scaler, device)

        # Periodic memory cleanup to prevent CUDA fragmentation
        if mem_cleanup_interval > 0 and batch_idx > 0 and batch_idx % mem_cleanup_interval == 0:
            import gc; gc.collect()
            torch.cuda.empty_cache()

        try:
            local_code = 0
            local_error = None
            local_skip = None
            try:
                sparse_input, batch_size, gt_boxes, gt_labels = prepare_ddp_batch(
                    model,
                    batch,
                    device,
                    default_input_size,
                    strict_finite_checks,
                )
            except CollectiveBatchSkip as error:
                local_code = 1
                local_skip = error
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower():
                    local_code = 2
                else:
                    local_code = 3
                    local_error = f"{type(error).__name__}: {error}"
            except BaseException as error:
                local_code = 3
                local_error = f"{type(error).__name__}: {error}"
            preparation_decision = resolve_batch_consensus(
                local_code,
                local_error,
                device,
                "batch preparation",
            )
            if preparation_decision == 2:
                register_skip("collective batch-preparation OOM", "batch_preparation", "oom")
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                continue
            if preparation_decision == 1:
                reason = local_skip.reason if local_skip is not None else "peer rank non-finite input"
                stage = local_skip.stage if local_skip is not None else "peer_input"
                register_skip(reason, stage, "non_finite")
                if aborted_early:
                    break
                continue

            if micro_batches_in_window == 0:
                optimizer.zero_grad(set_to_none=True)

            local_code = 0
            local_error = None
            local_skip = None
            try:
                losses, loss = forward_ddp_loss(
                    model,
                    sparse_input,
                    batch_size,
                    gt_boxes,
                    gt_labels,
                    loss_fn,
                    use_amp,
                    amp_dtype,
                    strict_finite_checks,
                )
            except CollectiveBatchSkip as error:
                local_code = 1
                local_skip = error
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower():
                    local_code = 2
                else:
                    local_code = 3
                    local_error = f"{type(error).__name__}: {error}"
            except BaseException as error:
                local_code = 3
                local_error = f"{type(error).__name__}: {error}"
            forward_decision = resolve_batch_consensus(
                local_code,
                local_error,
                device,
                "forward/loss",
            )
            if forward_decision == 2:
                register_skip("collective forward/loss OOM", "forward_loss", "oom")
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                continue
            if forward_decision == 1:
                reason = local_skip.reason if local_skip is not None else "peer rank non-finite forward/loss"
                stage = local_skip.stage if local_skip is not None else "peer_forward_loss"
                register_skip(reason, stage, "non_finite")
                if aborted_early:
                    break
                continue

            latest_loss_grad_norms = {}
            if (
                per_loss_grad_diag_interval > 0
                and batch_idx % per_loss_grad_diag_interval == 0
                and not use_amp
            ):
                try:
                    latest_loss_grad_norms = compute_per_loss_grad_norms(losses, model, per_loss_keys)
                except RuntimeError as grad_diag_err:
                    print(f"  WARNING: per-loss grad diagnostics failed at batch {batch_idx}: {grad_diag_err}")

            scaled_loss = loss / accumulation_steps
            if use_amp and scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            micro_batches_in_window += 1

            if micro_batches_in_window >= accumulation_steps:
                local_scaled_gradients_finite = check_gradients_finite(model)
                if COLLECTIVE_CONSENSUS(int(not local_scaled_gradients_finite), device):
                    register_skip(
                        "collective non-finite scaled gradients before optimizer step",
                        "scaled_gradient_check",
                        "non_finite_grad",
                        reset_scaler=(use_amp and scaler is not None),
                    )
                    if aborted_early:
                        break
                    continue

                if use_amp and scaler is not None:
                    synchronized_scaler_scale(scaler, device)
                    scaler.unscale_(optimizer)
                    if COLLECTIVE_CONSENSUS(int(scaler_found_inf(scaler, optimizer)), device):
                        register_skip(
                            "collective GradScaler overflow before optimizer step",
                            "amp_overflow_check",
                            "non_finite_grad",
                            reset_scaler=True,
                        )
                        if aborted_early:
                            break
                        continue

                local_unscaled_gradients_finite = check_gradients_finite(model)
                if COLLECTIVE_CONSENSUS(int(not local_unscaled_gradients_finite), device):
                    register_skip(
                        "collective non-finite unscaled gradients before optimizer step",
                        "unscaled_gradient_check",
                        "non_finite_grad",
                        reset_scaler=(use_amp and scaler is not None),
                    )
                    if aborted_early:
                        break
                    continue

                grad_norm = compute_gradient_norm(model)
                if is_main_process and grad_norm > 100:
                    print(f"  WARNING: Large gradient norm: {grad_norm:.1f}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                    synchronized_scaler_scale(scaler, device)
                else:
                    optimizer.step()
                scheduler.step()
                optimizer_steps += 1
                optimizer_steps_completed += 1
                if ema is not None:
                    ema.update(model.module if hasattr(model, 'module') else model)
                optimizer.zero_grad(set_to_none=True)
                micro_batches_in_window = 0

            # Update metrics (inside try block, runs for all successful batches)
            successful_batches += 1
            total_loss += loss.item() * batch_size
            total_cls_loss += losses['cls_loss'].item() * batch_size
            total_reg_loss += (losses['reg_loss'].item() if isinstance(losses['reg_loss'], torch.Tensor) else losses['reg_loss']) * batch_size
            total_ctr_loss += (losses['ctr_loss'].item() if isinstance(losses['ctr_loss'], torch.Tensor) else losses['ctr_loss']) * batch_size
            total_samples += batch_size
            total_pos += losses.get('num_pos_raw', losses['num_pos']).item()
            batch_quality = {
                key: float(losses[key].item())
                for key in QUALITY_DIAGNOSTIC_KEYS
                if key in losses
            }
            for key in (
                "num_gt", "num_gt_with_candidates", "gt_zero_candidates", "dynamic_k_sum",
                "num_pos_raw", "quota_deficit", "conflict_sites", "gt_zero_after_conflict",
                "multi_gt_samples", "multi_gt_gt_zero_assigned",
            ):
                quality_diag_sums[key] += batch_quality.get(key, 0.0)
            batch_num_gt = batch_quality.get("num_gt", 0.0)
            batch_num_pos_raw = batch_quality.get("num_pos_raw", 0.0)
            quality_candidate_total += batch_quality.get("candidate_count_mean", 0.0) * batch_num_gt
            quality_diag_sums["candidate_count_max"] = max(
                quality_diag_sums["candidate_count_max"], batch_quality.get("candidate_count_max", 0.0)
            )
            quality_cls_total += batch_quality.get("classification_quality_target_mean", 0.0) * batch_num_pos_raw
            quality_iou_total += batch_quality.get("decoded_iou_target_mean", 0.0) * batch_num_pos_raw
            quality_diag_sums["classification_quality_target_max"] = max(
                quality_diag_sums["classification_quality_target_max"], batch_quality.get("classification_quality_target_max", 0.0)
            )
            quality_diag_sums["decoded_iou_target_max"] = max(
                quality_diag_sums["decoded_iou_target_max"], batch_quality.get("decoded_iou_target_max", 0.0)
            )
            consecutive_nonfinite = 0
            rolling_skip.append(0)
            rolling_skip_rate = sum(rolling_skip) / max(len(rolling_skip), 1)
            append_jsonl(
                batch_health_path,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "status": "ok",
                    "loss": float(loss.item()),
                    "cls_loss": float(losses['cls_loss'].item()),
                    "reg_loss": float(losses['reg_loss'].item() if isinstance(losses['reg_loss'], torch.Tensor) else losses['reg_loss']),
                    "ctr_loss": float(losses['ctr_loss'].item() if isinstance(losses['ctr_loss'], torch.Tensor) else losses['ctr_loss']),
                    "num_pos": float(losses['num_pos'].item()),
                    "num_pos_raw": int(losses.get('num_pos_raw', losses['num_pos']).item()),
                    "quality_diagnostics": batch_quality,
                    "per_loss_grad_norms": latest_loss_grad_norms,
                    "rolling_skip_rate": rolling_skip_rate,
                    "clip_fraction_mean": (
                        float(clip_stats["clip_fraction_sum"] / max(clip_stats["n_samples"], 1.0))
                    ),
                    "sample_ids": sample_ids[:16],
                },
            )

            # Logging
            log_interval = config.get('logging', {}).get('log_interval', 10)
            if is_main_process and batch_idx % log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - start_time
                samples_per_sec = total_samples / elapsed if elapsed > 0 else 0

                print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                      f"loss={loss.item():.4f} cls={losses['cls_loss'].item():.4f} "
                      f"reg={losses['reg_loss']:.4f} ctr={losses['ctr_loss']:.4f} "
                      f"pos={int(losses.get('num_pos_raw', losses['num_pos']).item())} lr={lr:.2e} "
                      f"samples/s={samples_per_sec:.1f}")
                if batch_quality:
                    print(
                        "    quality_diag "
                        f"gt={int(batch_quality.get('num_gt', 0))} "
                        f"zero={int(batch_quality.get('gt_zero_candidates', 0))} "
                        f"quota={int(batch_quality.get('dynamic_k_sum', 0))} "
                        f"fill={batch_quality.get('quota_fill_ratio', 0.0):.3f} "
                        f"conflicts={int(batch_quality.get('conflict_sites', 0))} "
                        f"unassigned={int(batch_quality.get('gt_zero_after_conflict', 0))}"
                    )
                if latest_loss_grad_norms:
                    grad_diag_msg = " ".join(
                        f"{k.replace('_loss_raw', '')}={v:.2f}" for k, v in sorted(latest_loss_grad_norms.items())
                    )
                    print(f"    grad_diag {grad_diag_msg}")

            now = time.time()
            checkpoint_due = int(
                is_main_process
                and output_dir is not None
                and now - last_ckpt_time >= ckpt_interval_sec
                and micro_batches_in_window == 0
            )
            if COLLECTIVE_CONSENSUS(checkpoint_due, device):
                def write_mid_epoch_checkpoint():
                    nonlocal last_ckpt_time
                    last_ckpt_time = now
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, epoch,
                        {
                            'batch_idx': batch_idx,
                            'loss': loss.item(),
                            'mAP_50': float(config.get('_runtime', {}).get('best_map', 0.0)),
                            'mAP_50_95': float(config.get('_runtime', {}).get('best_map_50_95', 0.0)),
                            'selection_best_mAP_50': float(config.get('_runtime', {}).get('best_map', 0.0)),
                            'selection_best_mAP_50_95': float(config.get('_runtime', {}).get('best_map_50_95', 0.0)),
                            'selection_best_epoch': int(config.get('_runtime', {}).get('best_epoch', -1)),
                            'val_loss': float(config.get('_runtime', {}).get('best_loss', float('inf'))),
                        },
                        output_dir / 'latest.pt', config, ema=ema,
                        batch_idx=batch_idx,
                        optimizer_steps_in_epoch=optimizer_steps,
                        optimizer_steps_completed=optimizer_steps_completed,
                        next_batch_idx=batch_idx + 1,
                    )
                    print(f"  [mid-epoch ckpt] Saved at batch {batch_idx}/{len(dataloader)} "
                          f"({batch_idx/len(dataloader)*100:.1f}%)")
                    heartbeat = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "batches_total": len(dataloader),
                        "progress_pct": round(batch_idx / len(dataloader) * 100, 1),
                        "status": "training",
                        "loss": float(total_loss / max(total_samples, 1)),
                        "samples_per_sec": float(total_samples / max(now - start_time, 1)),
                        "lr": float(optimizer.param_groups[0]['lr']),
                        "skip_rate": float(sum(rolling_skip) / max(len(rolling_skip), 1)),
                    }
                    (output_dir / "run_health.json").write_text(
                        json.dumps(heartbeat, indent=2)
                    )
                    return str((output_dir / "latest.pt").resolve())

                run_rank0_stage(rank, "mid-epoch checkpoint and heartbeat", write_mid_epoch_checkpoint)

        except BaseException as error:
            ABORT_DISTRIBUTED_JOB(
                f"DDP3 rank {rank} failed during backward/optimizer/batch handling "
                f"at epoch {epoch}, batch {batch_idx}: {type(error).__name__}: {error}"
            )

    if micro_batches_in_window != 0:
        ABORT_DISTRIBUTED_JOB("DDP3 left a partial optimizer window with accumulation=1")
    optimizer.zero_grad(set_to_none=True)
    if not stopped_by_controller and not aborted_early:
        if COLLECTIVE_STOP_REQUESTED(stop_request_path, device):
            next_batch_idx = (
                min(int(max_train_batches), len(dataloader))
                if max_train_batches is not None
                else len(dataloader)
            )
            emergency_checkpoint = commit_emergency(next_batch_idx)
            stopped_by_controller = True
            abort_reason = "controller watchdog requested a power stop"

    local_epoch_state = {
        "loss_sum": total_loss,
        "cls_loss_sum": total_cls_loss,
        "reg_loss_sum": total_reg_loss,
        "ctr_loss_sum": total_ctr_loss,
        "iou_quality_loss_sum": total_iouq_loss,
        "proposal_loss_sum": total_proposal_loss,
        "ranking_loss_sum": total_ranking_loss,
        "uncertainty_loss_sum": total_uncertainty_loss,
        "positive_query_ratio_sum": total_positive_query_ratio,
        "ranking_gap_sum": total_ranking_gap,
        "near_boundary_mass_sum": total_near_boundary_mass,
        "proposal_recall16_sum": total_proposal_recall16,
        "proposal_recall32_sum": total_proposal_recall32,
        "proposal_recall64_sum": total_proposal_recall64,
        "proposal_recall128_sum": total_proposal_recall128,
        "sample_count": total_samples,
        "positive_count": total_pos,
        "quality_num_gt": quality_diag_sums["num_gt"],
        "quality_num_gt_with_candidates": quality_diag_sums["num_gt_with_candidates"],
        "quality_gt_zero_candidates": quality_diag_sums["gt_zero_candidates"],
        "quality_dynamic_k_sum": quality_diag_sums["dynamic_k_sum"],
        "quality_num_pos_raw": quality_diag_sums["num_pos_raw"],
        "quality_quota_deficit": quality_diag_sums["quota_deficit"],
        "quality_conflict_sites": quality_diag_sums["conflict_sites"],
        "quality_gt_zero_after_conflict": quality_diag_sums["gt_zero_after_conflict"],
        "quality_multi_gt_samples": quality_diag_sums["multi_gt_samples"],
        "quality_multi_gt_gt_zero_assigned": quality_diag_sums["multi_gt_gt_zero_assigned"],
        "quality_candidate_total": quality_candidate_total,
        "quality_cls_total": quality_cls_total,
        "quality_iou_total": quality_iou_total,
        "quality_candidate_count_max": quality_diag_sums["candidate_count_max"],
        "quality_classification_target_max": quality_diag_sums["classification_quality_target_max"],
        "quality_decoded_iou_target_max": quality_diag_sums["decoded_iou_target_max"],
        "clip_sample_count": clip_n_samples,
        "clip_raw_sum": clip_raw_sum,
        "clip_kept_sum": clip_kept_sum,
        "clip_fraction_sum": clip_fraction_sum,
        "clip_clipped_count": clip_clipped_count,
        "optimizer_steps": optimizer_steps,
        "optimizer_steps_completed": optimizer_steps_completed,
        "skipped_non_finite": skipped_non_finite,
        "skipped_non_finite_grad": skipped_non_finite_grad,
        "skipped_oom": skipped_oom,
        "sanitized_grad_steps": sanitized_grad_steps,
        "processed_batches": processed_batches,
        "successful_batches": successful_batches,
        "nonfinite_events": nonfinite_events,
        "first_nonfinite_batch": (
            first_nonfinite_batch
            if first_nonfinite_batch is not None
            else QUALITY_OPTIMIZER_STEPS_PER_EPOCH + 1
        ),
        "max_consecutive_nonfinite": max_consecutive_nonfinite,
        "aborted_early": int(aborted_early),
        "stopped_by_controller": int(stopped_by_controller),
        "learning_rate": optimizer.param_groups[0]["lr"],
        "elapsed_seconds": time.time() - start_time,
    }
    global_state = reduce_ddp_epoch_state(local_epoch_state, device)
    global_abort_reason = ""
    if bool(global_state["aborted_early"]) or bool(global_state["stopped_by_controller"]):
        reasons = GATHER_RANK_ERRORS(abort_reason or None)
        global_abort_reason = "; ".join(reasons) if reasons else "collective stop without a reason"

    sample_count = global_state["sample_count"]
    if sample_count == 0 and is_main_process:
        print("  WARNING: No globally optimized samples were recorded this epoch")
    skipped_total = int(
        global_state["skipped_non_finite"]
        + global_state["skipped_non_finite_grad"]
        + global_state["skipped_oom"]
    )
    processed_batches_global = int(global_state["processed_batches"])
    skip_rate = skipped_total / max(processed_batches_global, 1)
    if skipped_total > 0 and is_main_process:
        print(
            f"  Skip summary: nonfinite_loss={int(global_state['skipped_non_finite'])} "
            f"nonfinite_grad={int(global_state['skipped_non_finite_grad'])} "
            f"oom={int(global_state['skipped_oom'])} "
            f"total={skipped_total}/{processed_batches_global} ({skip_rate*100:.1f}%)"
        )
    if bool(global_state["aborted_early"]) and is_main_process:
        print(f"  ABORT GATE TRIGGERED: {global_abort_reason}")

    def divided(numerator: str, denominator: float) -> float:
        return global_state[numerator] / denominator if denominator else 0.0

    epoch_num_gt = global_state["quality_num_gt"]
    epoch_quota = global_state["quality_dynamic_k_sum"]
    epoch_num_pos_raw = global_state["quality_num_pos_raw"]
    quality_epoch_metrics = {
        "num_gt": epoch_num_gt,
        "num_gt_with_candidates": global_state["quality_num_gt_with_candidates"],
        "gt_zero_candidates": global_state["quality_gt_zero_candidates"],
        "dynamic_k_sum": epoch_quota,
        "num_pos_raw": epoch_num_pos_raw,
        "quota_fill_ratio": epoch_num_pos_raw / epoch_quota if epoch_quota else 1.0,
        "quota_deficit": global_state["quality_quota_deficit"],
        "conflict_sites": global_state["quality_conflict_sites"],
        "gt_zero_after_conflict": global_state["quality_gt_zero_after_conflict"],
        "multi_gt_samples": global_state["quality_multi_gt_samples"],
        "multi_gt_gt_zero_assigned": global_state["quality_multi_gt_gt_zero_assigned"],
        "candidate_count_mean": divided("quality_candidate_total", epoch_num_gt),
        "candidate_count_max": global_state["quality_candidate_count_max"],
        "classification_quality_target_mean": divided("quality_cls_total", epoch_num_pos_raw),
        "classification_quality_target_max": global_state["quality_classification_target_max"],
        "decoded_iou_target_mean": divided("quality_iou_total", epoch_num_pos_raw),
        "decoded_iou_target_max": global_state["quality_decoded_iou_target_max"],
    }

    first_failure = int(global_state["first_nonfinite_batch"])
    if first_failure > QUALITY_OPTIMIZER_STEPS_PER_EPOCH:
        first_failure = -1
    metrics = {
        "loss": divided("loss_sum", sample_count),
        "cls_loss": divided("cls_loss_sum", sample_count),
        "reg_loss": divided("reg_loss_sum", sample_count),
        "ctr_loss": divided("ctr_loss_sum", sample_count),
        "iou_quality_loss": divided("iou_quality_loss_sum", sample_count),
        "proposal_loss": divided("proposal_loss_sum", sample_count),
        "ranking_loss": divided("ranking_loss_sum", sample_count),
        "uncertainty_loss": divided("uncertainty_loss_sum", sample_count),
        "positive_query_ratio": divided("positive_query_ratio_sum", sample_count),
        "ranking_gap_mean": divided("ranking_gap_sum", sample_count),
        "near_boundary_mass_045_055": divided("near_boundary_mass_sum", sample_count),
        "proposal_recall_at_16": divided("proposal_recall16_sum", sample_count),
        "proposal_recall_at_32": divided("proposal_recall32_sum", sample_count),
        "proposal_recall_at_64": divided("proposal_recall64_sum", sample_count),
        "proposal_recall_at_128": divided("proposal_recall128_sum", sample_count),
        "avg_pos": divided("positive_count", global_state["successful_batches"]),
        "optimized_samples": int(sample_count),
        "lr": global_state["learning_rate"],
        "time": global_state["elapsed_seconds"],
        "optimizer_steps": int(global_state["optimizer_steps"]),
        "optimizer_steps_completed": int(global_state["optimizer_steps_completed"]),
        "skipped_non_finite": int(global_state["skipped_non_finite"]),
        "skipped_non_finite_grad": int(global_state["skipped_non_finite_grad"]),
        "skipped_oom": int(global_state["skipped_oom"]),
        "sanitized_grad_steps": int(global_state["sanitized_grad_steps"]),
        "skipped_total": skipped_total,
        "skip_rate": skip_rate,
        "processed_batches": processed_batches_global,
        "successful_batches": int(global_state["successful_batches"]),
        "nonfinite_events": int(global_state["nonfinite_events"]),
        "first_nonfinite_batch": first_failure,
        "max_consecutive_nonfinite": int(global_state["max_consecutive_nonfinite"]),
        "aborted_early": bool(global_state["aborted_early"]),
        "stopped_by_controller": bool(global_state["stopped_by_controller"]),
        "emergency_checkpoint": emergency_checkpoint,
        "abort_reason": global_abort_reason,
        "clip_fraction_mean": divided("clip_fraction_sum", global_state["clip_sample_count"]),
        "clip_rate": divided("clip_clipped_count", global_state["clip_sample_count"]),
        "raw_voxels_mean": divided("clip_raw_sum", global_state["clip_sample_count"]),
        "kept_voxels_mean": divided("clip_kept_sum", global_state["clip_sample_count"]),
    }
    metrics.update({f"quality_{key}": value for key, value in quality_epoch_metrics.items()})
    return metrics


class ValidationStopRequested(RuntimeError):
    pass


def validation_stop_requested(stop_request_path: Optional[Path]) -> bool:
    return stop_request_path is not None and stop_request_path.is_file()


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    config: Dict,
    compute_map: bool = True,
    epoch: Optional[int] = None,
    stop_request_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Run validation with loss computation and optional mAP evaluation."""
    model.eval()

    input_size = config.get('model', {}).get('input_size', [640, 640])

    # Restore model to default input_size (may have been changed by multi-scale training)
    raw = model.module if hasattr(model, 'module') else model
    raw.input_size = tuple(input_size)

    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_ctr_loss = 0.0
    total_samples = 0
    clip_n_samples = 0.0
    clip_raw_sum = 0.0
    clip_kept_sum = 0.0
    clip_fraction_sum = 0.0
    clip_clipped_count = 0.0

    # mAP calculator
    eval_config = config.get('eval', {})
    temporal_cfg = eval_config.get('temporal_rerank', {}) or {}
    eval_score_thresh = float(eval_config.get('score_thresh', 0.05))
    eval_nms_thresh = float(eval_config.get('nms_thresh', 0.5))
    eval_max_detections = int(eval_config.get('max_detections', 100))
    temporal_enabled = bool(temporal_cfg.get('enabled', False))
    temporal_topk = int(temporal_cfg.get('topk', 5))
    temporal_weights = temporal_cfg.get('weights', {}) if isinstance(temporal_cfg.get('weights', {}), dict) else {}
    decode_max_detections = max(eval_max_detections, temporal_topk) if temporal_enabled else eval_max_detections
    map_calc = MAPCalculator(
        num_classes=1,
        img_size=tuple(input_size),
        conf_threshold=eval_score_thresh,
    ) if compute_map else None

    raw.set_decode_params(
        score_thresh=eval_score_thresh,
        nms_thresh=eval_nms_thresh,
        max_detections=decode_max_detections,
    )

    try:
        temporal_state: Dict[str, Dict[str, Any]] = {}
        for batch in dataloader:
            if validation_stop_requested(stop_request_path):
                raise ValidationStopRequested(
                    "controller watchdog requested a power stop during validation"
                )
            clip_stats = summarize_clip_telemetry(batch)
            clip_n_samples += clip_stats["n_samples"]
            clip_raw_sum += clip_stats["raw_sum"]
            clip_kept_sum += clip_stats["kept_sum"]
            clip_fraction_sum += clip_stats["clip_fraction_sum"]
            clip_clipped_count += clip_stats["clipped_count"]

            sparse_input = create_sparse_tensor(batch, device)
            batch_size = batch['batch_size']

            gt_boxes = [b.to(device) for b in batch['gt_boxes']]
            gt_labels = [l.to(device) for l in batch['gt_labels']]

            # Forward pass with loss inputs
            outputs = model(sparse_input, batch_size, return_loss_inputs=True)
            losses = loss_fn(outputs, gt_boxes, gt_labels)

            total_loss += losses['loss'].item() * batch_size
            total_cls_loss += losses['cls_loss'].item() * batch_size
            total_reg_loss += (losses['reg_loss'].item() if isinstance(losses['reg_loss'], torch.Tensor) else losses['reg_loss']) * batch_size
            total_ctr_loss += (losses['ctr_loss'].item() if isinstance(losses['ctr_loss'], torch.Tensor) else losses['ctr_loss']) * batch_size
            total_samples += batch_size

            # mAP: run inference to get decoded detections
            if map_calc is not None:
                # Decode from raw predictions
                detections = raw._decode_detections(
                    cls_logits=outputs["cls_logits"],
                    box_ltrb=outputs["box_ltrb"],
                    ctr_logits=outputs["ctr_logits"],
                    indices_2d=outputs["indices_2d"],
                    batch_size=batch_size,
                    score_thresh=eval_score_thresh,
                    nms_thresh=eval_nms_thresh,
                    max_detections=decode_max_detections,
                )
                if temporal_enabled:
                    detections = temporal_rerank_top1(
                        detections=detections,
                        seq_ids=[str(x) for x in batch.get("seq_ids", [])],
                        frame_nums=[int(x) for x in batch.get("frame_nums", [])],
                        topk=temporal_topk,
                        weights=temporal_weights,
                        state=temporal_state,
                    )

                # Build per-image prediction tensors for MAPCalculator
                preds_list = []
                for b in range(batch_size):
                    dets = detections[b]  # [N, 6]
                    # Filter zero-padded entries and align with eval score threshold.
                    valid = dets[:, 4] > eval_score_thresh
                    dets = dets[valid]
                    if eval_max_detections > 0 and dets.shape[0] > eval_max_detections:
                        dets = dets[:eval_max_detections]
                    preds_list.append(dets)

                # GT labels in YOLO format [cls, cx, cy, w, h] for MAPCalculator
                # Convert gt_boxes (xyxy) back to YOLO format for MAPCalculator.update()
                H, W = input_size
                gt_yolo_list = []
                for boxes_i, labels_i in zip(gt_boxes, gt_labels):
                    if len(boxes_i) > 0:
                        x1, y1, x2, y2 = boxes_i[:, 0], boxes_i[:, 1], boxes_i[:, 2], boxes_i[:, 3]
                        cx = ((x1 + x2) / 2) / W
                        cy = ((y1 + y2) / 2) / H
                        w = (x2 - x1) / W
                        h = (y2 - y1) / H
                        yolo = torch.stack([labels_i.float(), cx, cy, w, h], dim=1)
                        gt_yolo_list.append(yolo)
                    else:
                        gt_yolo_list.append(torch.zeros(0, 5, device=device))

                map_calc.update(preds_list, gt_yolo_list)
    finally:
        pass  # No patched decode to restore

    metrics = {
        'val_loss': total_loss / max(total_samples, 1),
        'val_cls_loss': total_cls_loss / max(total_samples, 1),
        'val_reg_loss': total_reg_loss / max(total_samples, 1),
        'val_ctr_loss': total_ctr_loss / max(total_samples, 1),
        'val_clip_fraction_mean': clip_fraction_sum / max(clip_n_samples, 1.0),
        'val_clip_rate': clip_clipped_count / max(clip_n_samples, 1.0),
        'val_raw_voxels_mean': clip_raw_sum / max(clip_n_samples, 1.0),
        'val_kept_voxels_mean': clip_kept_sum / max(clip_n_samples, 1.0),
    }

    if map_calc is not None:
        det_metrics = map_calc.compute()
        metrics['mAP_50'] = det_metrics.mAP_50
        metrics['mAP_50_95'] = det_metrics.mAP_50_95
        metrics['precision'] = det_metrics.precision
        metrics['recall'] = det_metrics.recall
        metrics['f1'] = det_metrics.f1
        metrics['metrics_engine_id'] = "sparse_voxel_det.mapcalc"
        metrics['metrics_version'] = "2026-02-26"

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: Optional[GradScaler],
    epoch: int,
    metrics: Dict,
    path: Path,
    config: Dict,
    ema: Optional[ModelEMA] = None,
    batch_idx: Optional[int] = None,
    optimizer_steps_in_epoch: Optional[int] = None,
    optimizer_steps_completed: Optional[int] = None,
    next_batch_idx: Optional[int] = None,
):
    """
    Save checkpoint with DDP-safe state dict handling.

    Saves model without 'module.' prefix for portability.
    Mid-epoch checkpoints bind the next sampler batch, rank-local sample
    offset, and cumulative optimizer-step lineage.
    """
    context = require_worker_context()
    if int(context["rank"]) != 0:
        raise RuntimeError("Only DDP3 rank 0 may write checkpoints")
    if not callable(CHECKPOINT_COMMIT_HOOK):
        raise RuntimeError("DDP3 checkpoint commit hook is missing")
    path = Path(path)
    if path.parent.resolve() != Path(context["output_dir"]).resolve():
        raise RuntimeError(f"DDP3 checkpoint path is outside the owned output: {path}")
    if (
        isinstance(optimizer_steps_in_epoch, bool)
        or not isinstance(optimizer_steps_in_epoch, int)
        or not 0 <= optimizer_steps_in_epoch <= QUALITY_OPTIMIZER_STEPS_PER_EPOCH
    ):
        raise RuntimeError("DDP3 checkpoint requires a valid epoch optimizer-step count")
    if (
        isinstance(optimizer_steps_completed, bool)
        or not isinstance(optimizer_steps_completed, int)
        or optimizer_steps_completed < optimizer_steps_in_epoch
    ):
        raise RuntimeError("DDP3 checkpoint requires a valid cumulative optimizer-step count")
    if next_batch_idx is not None and optimizer_steps_in_epoch > next_batch_idx:
        raise RuntimeError("DDP3 checkpoint optimizer steps exceed the consumed batch boundary")
    if batch_idx is None and next_batch_idx is not None:
        raise RuntimeError("Full-epoch DDP3 checkpoint cannot carry next_batch_idx")
    if batch_idx is not None and next_batch_idx != batch_idx + 1:
        raise RuntimeError("Mid-epoch DDP3 checkpoint batch lineage is inconsistent")

    # Handle DDP - save without 'module.' prefix for portability
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'ema_state_dict': ema.state_dict() if ema is not None else None,
        'metrics': metrics,
        'config': config,
        'checkpoint_lineage': QUALITY_CHECKPOINT_LINEAGE,
        'runtime_contract': copy.deepcopy(context["runtime_contract"]),
        'launch_origin': copy.deepcopy(context["launch_origin"]),
        'optimizer_boundary': {
            'micro_batches_in_window': 0,
            'optimizer_steps_in_epoch': optimizer_steps_in_epoch,
            'optimizer_steps_completed': optimizer_steps_completed,
            'sampler_epoch': epoch,
            'next_batch_idx': next_batch_idx,
            'next_rank_local_sample_offset': (
                QUALITY_OPTIMIZED_SAMPLES // QUALITY_WORLD_SIZE
                if next_batch_idx is None
                else next_batch_idx * QUALITY_PER_RANK_BATCH
            ),
        },
    }
    # Mid-epoch marker: batch_idx != None means this is a partial-epoch save.
    # Resume will continue from this batch instead of jumping to epoch+1.
    if batch_idx is not None:
        checkpoint['batch_idx'] = batch_idx
    temporary_path = path.with_name(
        f".{path.name}.rank0-{os.getpid()}-{time.time_ns()}.pending"
    )
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)
    CHECKPOINT_COMMIT_HOOK(path, checkpoint)
    tag = f" (mid-epoch batch {batch_idx})" if batch_idx is not None else ""
    print(f"Saved checkpoint to {path}{tag}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    checkpoint_data: Optional[Dict] = None,
) -> Dict:
    """
    Load checkpoint with DDP-safe state dict handling.

    From FAILURES.md: DDP checkpoints have 'module.' prefix that must be handled.
    """
    checkpoint = checkpoint_data if checkpoint_data is not None else torch.load(path, map_location='cpu')

    # Handle DDP 'module.' prefix - from FAILURES.md lesson
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        # Remove 'module.' prefix from DDP checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        print("  Removed 'module.' prefix from DDP checkpoint")

    # Full-state quality resume must match the exact model schema.
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
    return checkpoint


def load_weights_only(
    path: Path,
    model: nn.Module,
    ema: Optional[ModelEMA],
    device: torch.device,
) -> Dict:
    """Load model (+ optional EMA) weights without optimizer/scheduler/scaler state."""
    checkpoint = torch.load(path, map_location='cpu')

    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        print("  Removed 'module.' prefix from weights-only checkpoint")

    if hasattr(model, 'module'):
        load_result = model.module.load_state_dict(state_dict, strict=False)
    else:
        load_result = model.load_state_dict(state_dict, strict=False)

    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    if missing_keys:
        print(f"  WARNING: Missing keys while loading weights-only checkpoint: {missing_keys[:8]}{' ...' if len(missing_keys) > 8 else ''}")
    if unexpected_keys:
        print(f"  WARNING: Unexpected keys while loading weights-only checkpoint: {unexpected_keys[:8]}{' ...' if len(unexpected_keys) > 8 else ''}")

    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'], device=device)

    print(f"Loaded weights-only checkpoint from {path} (epoch {checkpoint.get('epoch', 'unknown')})")
    return checkpoint


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def check_gradients_finite(model: nn.Module) -> bool:
    """Check if all gradients are finite (no NaN/Inf)."""
    for name, p in model.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                print(f"  WARNING: Non-finite gradient in {name}")
                return False
    return True


def sanitize_gradients(model: nn.Module, clip_value: float = 100.0) -> int:
    """Replace NaN/Inf gradients in-place and clamp extreme values.

    Returns:
        Number of gradient elements that were non-finite before sanitization.
    """
    repaired = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        grad = p.grad.data
        nonfinite = ~torch.isfinite(grad)
        if nonfinite.any():
            repaired += int(nonfinite.sum().item())
            torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0, out=grad)
        if clip_value > 0:
            grad.clamp_(min=-clip_value, max=clip_value)
    return repaired


def setup_ddp():
    """Validate the process group already established by the protected worker."""
    context = require_worker_context()
    if not dist.is_initialized():
        raise RuntimeError("DDP3 trainer requires the controller-initialized process group")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(context["local_rank"])
    if (
        rank != context["rank"]
        or world_size != QUALITY_WORLD_SIZE
        or context["world_size"] != QUALITY_WORLD_SIZE
        or local_rank != rank
    ):
        raise RuntimeError("DDP3 trainer process-group topology mismatch")
    if torch.cuda.current_device() != local_rank:
        raise RuntimeError("DDP3 trainer current CUDA device differs from protected local rank")
    return rank, local_rank, world_size


def cleanup_ddp():
    """The protected launcher owns process-group teardown."""
    return None


def run_rank0_stage(rank: int, label: str, function):
    return RUN_RANK0_STAGE(rank, label, function)


def main() -> int:
    worker_context = require_worker_context()
    parser = argparse.ArgumentParser(description='Train SparseVoxelDet (Option C)')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--weights-only', type=str, default=None,
                        help='Load model+EMA weights only (fresh optimizer/scheduler/scaler, start epoch 0)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--strict-finite-checks',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable strict finite checks with trace logs and hard abort gates',
    )
    parser.add_argument(
        '--finite-trace-dir',
        type=str,
        default=None,
        help='Directory for finite-trace artifacts (default: <output_dir>/finite_trace)',
    )
    parser.add_argument(
        '--abort-on-skip-rate',
        type=float,
        default=0.10,
        help='Abort if rolling skip rate over 500 batches exceeds this value',
    )
    parser.add_argument(
        '--abort-on-consecutive-nonfinite',
        type=int,
        default=200,
        help='Abort if consecutive non-finite batches exceed this threshold',
    )
    parser.add_argument(
        '--amp-mode',
        choices=['off', 'fp16', 'bf16'],
        default='fp16',
        help='AMP mode override for this run (default: fp16 to match config)',
    )
    parser.add_argument(
        '--epochs-override',
        type=int,
        default=None,
        help='Optional override for total epochs (useful for dry-runs)',
    )
    parser.add_argument(
        '--max-train-batches',
        type=int,
        default=None,
        help='Optional max batches per epoch (useful for dry-runs)',
    )
    parser.add_argument(
        '--skip-validation',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Skip validation/mAP/audits (useful for short dry-run stability checks)',
    )
    args = parser.parse_args()

    config = load_verified_quality_config(args.config)
    experiment_config = config.get("experiment", {})
    if not isinstance(experiment_config, dict):
        raise RuntimeError("Quality trainer experiment config is malformed")
    validate_quality_runtime_contract(experiment_config.get("runtime_contract"))
    if not args.output_dir:
        raise RuntimeError("Quality trainer requires the protected output directory")
    authorized_output_dir = Path(args.output_dir)
    authorized_resume_path = Path(args.resume) if args.resume else None
    validate_quality_output_claim(authorized_output_dir, authorized_resume_path)

    # Validate the controller-owned DDP process group.
    rank, local_rank, world_size = setup_ddp()
    is_main_process = (rank == 0)
    if is_main_process:
        print(f"Loaded config from {args.config}")

    deterministic = bool(config.get("training", {}).get("deterministic", False))
    seed_everything(args.seed + rank, deterministic=deterministic)
    if is_main_process:
        print(f"Seed setup: base_seed={args.seed + rank} deterministic={deterministic}")

    # Setup device
    device = worker_context["device"]
    if device != torch.device(f'cuda:{local_rank}'):
        raise RuntimeError("DDP3 trainer received an unauthorized device")

    if is_main_process:
        print(f"Using device: {device} (world_size={world_size})")

    # Setup output directory
    output_dir = authorized_output_dir
    if is_main_process:
        print(f"Output directory: {output_dir}")

        # Save config
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)

    finite_trace_dir = Path(args.finite_trace_dir) if args.finite_trace_dir else (output_dir / "finite_trace")
    if is_main_process:
        finite_trace_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    data_config = config.get('data', {})
    sparse_dir = project_root / data_config.get('sparse_dir', 'datasets/fred_paper_parity_v82/sparse')
    label_dir = project_root / data_config.get('label_dir', 'datasets/fred_paper_parity/labels')
    train_split = str(data_config.get('train_split', 'train'))
    val_split = str(data_config.get('val_split', 'val'))
    run_rank0_stage(
        rank,
        "corrected-label byte fingerprints",
        lambda: assert_corrected_label_contract(label_dir, train_split, val_split),
    )

    parity_enforced = bool(data_config.get('parity_enforced', False))

    if parity_enforced:
        run_parity_preflight(require_exact=True)
        assert_split_allowed_for_parity(train_split)
        assert_split_allowed_for_parity(val_split)
        expected_sparse = (project_root / "data/datasets/fred_paper_parity_v82/sparse").resolve()
        expected_labels = (project_root / "data/datasets/fred_paper_parity/labels").resolve()
        if sparse_dir.resolve() != expected_sparse:
            raise ValueError(
                f"Parity mode requires sparse_dir={expected_sparse}, got {sparse_dir.resolve()}"
            )
        if label_dir.resolve() != expected_labels:
            raise ValueError(
                f"Parity mode requires label_dir={expected_labels}, got {label_dir.resolve()}"
            )

    model_config = config.get('model', {})
    sparse_config = config.get('sparse', {})
    base_max_voxels = int(sparse_config.get('max_voxels', 30000))
    max_voxels_train = int(sparse_config.get('max_voxels_train', base_max_voxels))
    max_voxels_eval = int(sparse_config.get('max_voxels_eval', base_max_voxels))
    voxel_sampling_cfg = sparse_config.get('voxel_sampling', {}) or {}
    sparse_time_bins = int(sparse_config.get('time_bins', 33))

    sparse_contract_cfg = data_config.get('sparse_contract', {}) or {}
    sparse_contract_enabled = bool(sparse_contract_cfg.get('enabled', parity_enforced))
    sparse_contract_allow_mixed = bool(sparse_contract_cfg.get('allow_mixed_formats', False))
    sparse_contract_expected_format = sparse_contract_cfg.get('expected_format', None)
    sparse_contract_files_per_seq = int(sparse_contract_cfg.get('files_per_seq', 1))
    sparse_contract_expected_tb = sparse_contract_cfg.get('expected_time_bins', None)
    sparse_contract_uniform_tb = bool(sparse_contract_cfg.get('enforce_uniform_time_bins', False))
    sparse_contract_per_seq_uniform_tb = bool(sparse_contract_cfg.get('enforce_per_seq_uniform_time_bins', False))
    sparse_contract_require_coords_tb_meta = bool(
        sparse_contract_cfg.get('require_coords_time_bins_metadata', False)
    )
    if sparse_contract_expected_tb is not None:
        sparse_contract_expected_tb = int(sparse_contract_expected_tb)

    if sparse_contract_enabled:
        contract_report_json = output_dir / "preflight_sparse_tensor_contract.json"
        run_rank0_stage(
            rank,
            "sparse tensor contract",
            lambda: run_sparse_tensor_contract_preflight(
                sparse_root=sparse_dir,
                splits=[train_split, val_split],
                allow_mixed_formats=sparse_contract_allow_mixed,
                expected_format=sparse_contract_expected_format,
                files_per_seq=sparse_contract_files_per_seq,
                expected_time_bins=sparse_contract_expected_tb,
                enforce_uniform_time_bins=sparse_contract_uniform_tb,
                enforce_per_seq_uniform_time_bins=sparse_contract_per_seq_uniform_tb,
                require_coords_time_bins_metadata=sparse_contract_require_coords_tb_meta,
                report_json=contract_report_json,
                expected_feat_channels=model_config.get('in_channels', 2),
            ),
        )

    aug_config = config.get('augmentation', {})

    # Thread input_size from config into dataset/collate to prevent silent geometry mismatch
    model_input_size = tuple(model_config.get('input_size', [720, 1280]))  # (H, W)
    dataset_target_size = model_input_size  # Used by dataset for spatial bounds
    collate_base_size = model_input_size  # V82: pass (H, W) tuple, not just H
    feature_channels = model_config.get('feature_channels', None)  # Ablation: slice to N ch

    train_dataset = SparseEventDataset(
        sparse_dir=str(sparse_dir),
        label_dir=str(label_dir),
        split=train_split,
        time_bins=sparse_time_bins,
        target_size=dataset_target_size,
        augment=True,
        horizontal_flip_prob=aug_config.get('horizontal_flip', 0.5),
        event_dropout_prob=aug_config.get('event_dropout', 0.1),
        temporal_flip_prob=aug_config.get('temporal_flip', 0.0),
        polarity_flip_prob=aug_config.get('polarity_flip', 0.0),
        scale_range=tuple(aug_config.get('scale_range', [1.0, 1.0])),
        mosaic_prob=aug_config.get('mosaic_prob', 0.0),
        max_voxels=max_voxels_train,
        voxel_sampling=voxel_sampling_cfg,
        feature_channels=feature_channels,
    )

    val_dataset = None
    if is_main_process:
        val_dataset = SparseEventDataset(
            sparse_dir=str(sparse_dir),
            label_dir=str(label_dir),
            split=val_split,
            time_bins=sparse_time_bins,
            target_size=dataset_target_size,
            augment=False,
            max_voxels=max_voxels_eval,
            voxel_sampling={"mode": "random"},
            feature_channels=feature_channels,
        )

    train_config = config.get('training', {})
    batch_size = train_config.get('batch_size', 8)
    num_workers = train_config.get('num_workers', 4)

    # Preload datasets into RAM if configured (eliminates disk I/O during training)
    cache_dataset = train_config.get('cache_dataset', False)
    if cache_dataset:
        if is_main_process:
            print("Caching datasets in RAM...")
        train_dataset.preload_to_ram()
        if val_dataset is not None:
            val_dataset.preload_to_ram()

    # Use make_collate_fn with explicit time_bins (fixes V1 bug)
    time_bins = sparse_time_bins
    multi_scale_sizes = aug_config.get('multi_scale_sizes', None)
    train_collate_fn = make_collate_fn(
        time_bins=time_bins,
        multi_scale_sizes=multi_scale_sizes,
        base_size=collate_base_size,
    )
    # Validation always uses fixed model input size
    val_collate_fn = make_collate_fn(time_bins=time_bins, base_size=collate_base_size)

    # Exact three-rank deterministic sampling; no subset or padding.
    max_samples = train_config.get('max_samples_per_epoch', None)
    if max_samples is not None:
        raise ValueError("DDP3 quality training forbids max_samples_per_epoch caps")
    if world_size != QUALITY_WORLD_SIZE:
        raise RuntimeError(f"DDP3 quality training requires world size 3, got {world_size}")
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=QUALITY_WORLD_SIZE,
        rank=rank,
        shuffle=True,
        seed=QUALITY_SAMPLER_SEED,
        drop_last=True,
    )

    loader_seed = int(args.seed + rank * 100000)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(loader_seed)
    worker_init_fn = make_worker_init_fn(loader_seed)
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=loader_generator,
        **loader_kwargs,
    )

    val_loader = None
    if is_main_process:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=None,
            num_workers=num_workers,
            collate_fn=val_collate_fn,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            generator=loader_generator,
            **loader_kwargs,
        )

    if len(train_dataset) != QUALITY_TRAIN_ROSTER_SAMPLES:
        raise RuntimeError(
            f"DDP3 train roster has {len(train_dataset)} samples, "
            f"expected {QUALITY_TRAIN_ROSTER_SAMPLES}"
        )
    if len(train_loader) != QUALITY_OPTIMIZER_STEPS_PER_EPOCH:
        raise RuntimeError(
            f"DDP3 loader has {len(train_loader)} steps, "
            f"expected {QUALITY_OPTIMIZER_STEPS_PER_EPOCH}"
        )
    optimized_samples = (
        len(train_loader) * batch_size * QUALITY_WORLD_SIZE
    )
    if optimized_samples != QUALITY_OPTIMIZED_SAMPLES:
        raise RuntimeError(
            f"DDP3 optimized sample schedule is {optimized_samples}, "
            f"expected {QUALITY_OPTIMIZED_SAMPLES}"
        )
    run_rank0_stage(
        rank,
        "complete unsharded validation roster",
        lambda: (
            None
            if val_dataset is not None
            and len(val_dataset) == EXPECTED_LABEL_SPLITS["canonical_val"]["files"]
            and val_loader is not None
            and val_loader.sampler.__class__.__name__ == "SequentialSampler"
            else (_ for _ in ()).throw(
                RuntimeError("Rank 0 validation roster is incomplete or sharded")
            )
        ),
    )

    if is_main_process:
        print(f"Train split: {train_split}")
        print(f"Val split: {val_split}")
        print(f"Parity enforced: {parity_enforced}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)} (rank 0 complete roster, unsharded)")
        print(
            "Voxel caps: "
            f"train={max_voxels_train} eval={max_voxels_eval} "
            f"(base={base_max_voxels}, sampling={voxel_sampling_cfg.get('mode', 'random')})"
        )

    # Create model
    loss_config = config.get('loss', {})

    model = SparseVoxelDet(
        in_channels=model_config.get('in_channels', 6),
        num_classes=model_config.get('num_classes', 1),
        backbone_size=model_config.get('backbone_size', 'nano_deep'),
        fpn_channels=model_config.get('fpn_channels', 128),
        head_convs=model_config.get('head_convs', 2),
        input_size=tuple(model_config.get('input_size', [640, 640])),
        time_bins=sparse_config.get('time_bins', 15),
        prior_prob=model_config.get('prior_prob', 0.01),
        score_thresh=float(config.get('eval', {}).get('score_thresh', 0.05)),
        nms_thresh=float(config.get('eval', {}).get('nms_thresh', 0.5)),
        max_detections=int(config.get('eval', {}).get('max_detections', 10)),
        temporal_pool_mode=model_config.get('temporal_pool_mode', 'max'),
    ).to(device)

    # Exact DDP3 wrapping with synchronized normalization state.
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=True,
    )

    params = model.module.get_num_params() if hasattr(model, 'module') else model.get_num_params()
    if is_main_process:
        print(f"Model parameters: {params['total']:,}")

    # Create optimizer, scheduler, loss
    accum_steps = max(1, train_config.get('gradient_accumulation_steps', 1))
    flush_partial_for_schedule = bool(train_config.get('flush_partial_accumulation', False))
    if flush_partial_for_schedule:
        optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / accum_steps))
    else:
        optimizer_steps_per_epoch = max(1, len(train_loader) // accum_steps)
    if accum_steps != QUALITY_ACCUMULATION_STEPS:
        raise RuntimeError(
            f"DDP3 accumulation is {accum_steps}, expected {QUALITY_ACCUMULATION_STEPS}"
        )
    if optimizer_steps_per_epoch != QUALITY_OPTIMIZER_STEPS_PER_EPOCH:
        raise RuntimeError(
            f"DDP3 optimizer schedule is {optimizer_steps_per_epoch} steps/epoch, "
            f"expected {QUALITY_OPTIMIZER_STEPS_PER_EPOCH}"
        )
    if (
        int(train_config.get("warmup_steps", -1)) != QUALITY_WARMUP_STEPS
        or int(train_config.get("epochs", -1)) * optimizer_steps_per_epoch
        != QUALITY_TOTAL_OPTIMIZER_STEPS
    ):
        raise RuntimeError("DDP3 warmup or full cosine step schedule drifted")
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, optimizer_steps_per_epoch)
    if is_main_process:
        wd_stats = getattr(optimizer, "_wd_group_stats", None)
        if isinstance(wd_stats, dict):
            print(
                "Optimizer param groups: "
                f"decay_tensors={wd_stats.get('decay_tensors', 0)} "
                f"no_decay_tensors={wd_stats.get('no_decay_tensors', 0)} "
                f"decay_params={wd_stats.get('decay_params', 0):,} "
                f"no_decay_params={wd_stats.get('no_decay_params', 0):,}"
            )

    # Read detection config (used by loss and model, not duplicated)
    detection_config = config.get('detection', {})

    loss_fn = SparseVoxelDetLoss(
        stride=int(detection_config.get('stride', 4)),
        num_classes=model_config.get('num_classes', 1),
        focal_alpha=loss_config.get('focal_alpha', 0.25),
        focal_gamma=loss_config.get('focal_gamma', 2.0),
        cls_weight=loss_config.get('cls_weight', 1.0),
        reg_weight=loss_config.get('reg_weight', 2.0),
        ctr_weight=loss_config.get('ctr_weight', 1.0),
        center_sampling_radius=float(detection_config.get('center_sampling_radius',
                                     loss_config.get('center_sampling_radius', 1.5))),
        use_qfl=bool(loss_config.get('use_qfl', False)),
        nwd_weight=float(loss_config.get('nwd_weight', 0.0)),
        nwd_c=float(loss_config.get('nwd_c', 12.8)),
        task_aligned_enabled=bool(loss_config.get('task_aligned_enabled', False)),
        task_aligned_alpha=float(loss_config.get('task_aligned_alpha', 1.0)),
        task_aligned_beta=float(loss_config.get('task_aligned_beta', 6.0)),
        dynamic_k_topq=int(loss_config.get('dynamic_k_topq', 10)),
        quality_bootstrap_epochs=int(loss_config.get('quality_bootstrap_epochs', 2)),
    )

    # AMP mode override
    amp_mode = str(args.amp_mode).lower()
    cfg_use_amp = bool(train_config.get('use_amp', True))
    if amp_mode == "off":
        use_amp = False
    else:
        use_amp = cfg_use_amp and device.type == 'cuda'
    train_config['use_amp'] = use_amp

    if use_amp and amp_mode == "fp16":
        scaler = GradScaler(
            "cuda",
            init_scale=float(train_config.get('amp_init_scale', 2048.0)),
            growth_factor=float(train_config.get('amp_growth_factor', 2.0)),
            backoff_factor=float(train_config.get('amp_backoff_factor', 0.5)),
            growth_interval=int(train_config.get('amp_growth_interval', 2000)),
        )
    else:
        scaler = None

    # EMA
    use_ema = train_config.get('use_ema', True)
    ema_decay = train_config.get('ema_decay', 0.9999)
    raw_model = model.module if hasattr(model, 'module') else model
    ema = ModelEMA(raw_model, decay=ema_decay) if use_ema else None
    if is_main_process and ema is not None:
        print(f"Using EMA with decay={ema_decay}")

    # Resume / weights-only loading
    start_epoch = 0
    start_batch_idx = 0
    resume_optimizer_steps_in_epoch = 0
    optimizer_steps_completed = 0
    best_map = 0.0
    best_map_50_95 = 0.0
    best_epoch = -1
    best_loss = float('inf')
    if args.weights_only:
        checkpoint = load_weights_only(
            Path(args.weights_only), model, ema, device
        )
        if 'metrics' in checkpoint:
            if 'mAP_50' in checkpoint['metrics']:
                best_map = checkpoint['metrics'].get('selection_best_mAP_50', checkpoint['metrics']['mAP_50'])
                best_map_50_95 = checkpoint['metrics'].get('selection_best_mAP_50_95', checkpoint['metrics'].get('mAP_50_95', 0.0))
                best_epoch = int(checkpoint['metrics'].get('selection_best_epoch', checkpoint.get('epoch', -1)))
            if 'val_loss' in checkpoint['metrics']:
                best_loss = checkpoint['metrics']['val_loss']
        if is_main_process:
            print("Starting from epoch 0 with fresh optimizer/scheduler/scaler (weights-only mode)")
    elif args.resume:
        checkpoint = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, scaler
        )
        boundary = checkpoint["optimizer_boundary"]
        optimizer_steps_completed = int(boundary["optimizer_steps_completed"])
        resume_batch_idx = checkpoint.get('batch_idx', None)
        if resume_batch_idx is not None:
            start_epoch = int(boundary["sampler_epoch"])
            start_batch_idx = int(boundary["next_batch_idx"])
            resume_optimizer_steps_in_epoch = int(boundary["optimizer_steps_in_epoch"])
            if start_batch_idx * QUALITY_PER_RANK_BATCH != int(boundary["next_rank_local_sample_offset"]):
                raise RuntimeError("DDP3 resume rank-local sample offset drifted")
            if is_main_process:
                print(
                    f"Mid-epoch resume: epoch {start_epoch}, batch {start_batch_idx}, "
                    f"optimizer_steps_in_epoch={resume_optimizer_steps_in_epoch}, "
                    f"optimizer_steps_completed={optimizer_steps_completed}"
                )
        else:
            start_epoch = checkpoint['epoch'] + 1
            start_batch_idx = 0
            resume_optimizer_steps_in_epoch = 0
        if 'metrics' in checkpoint:
            if 'mAP_50' in checkpoint['metrics']:
                best_map = checkpoint['metrics'].get('selection_best_mAP_50', checkpoint['metrics']['mAP_50'])
                best_map_50_95 = checkpoint['metrics'].get('selection_best_mAP_50_95', checkpoint['metrics'].get('mAP_50_95', 0.0))
                best_epoch = int(checkpoint['metrics'].get('selection_best_epoch', checkpoint.get('epoch', -1)))
            if 'val_loss' in checkpoint['metrics']:
                best_loss = checkpoint['metrics']['val_loss']
        # Load EMA state if available
        if ema is not None and 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'], device=device)

    # Training loop
    epochs = int(args.epochs_override) if args.epochs_override is not None else int(train_config.get('epochs', 100))
    val_interval = int(train_config.get('val_interval', 5))
    val_interval_after_epoch = train_config.get('val_interval_after_epoch', None)
    val_interval_after = int(train_config.get('val_interval_after', val_interval))
    eval_config = config.get('eval', {})
    audit_enabled = bool(eval_config.get('audit_enabled', False))
    audit_interval = int(eval_config.get('audit_interval', val_interval))
    audit_max_samples = int(eval_config.get('audit_max_samples', 1024))
    audit_num_fp = int(eval_config.get('audit_num_fp', 12))
    audit_num_fn = int(eval_config.get('audit_num_fn', 12))
    audit_num_tp = int(eval_config.get('audit_num_tp', 12))
    audit_batch_size = int(eval_config.get('audit_batch_size', 8))
    audit_max_detections = eval_config.get('audit_max_detections', None)
    if audit_max_detections is not None:
        audit_max_detections = int(audit_max_detections)
    audit_timeout_sec = int(eval_config.get('audit_timeout_sec', 0))
    fullval_forensics_enabled = bool(eval_config.get('fullval_forensics_enabled', False))
    fullval_forensics_interval = int(eval_config.get('fullval_forensics_interval', 5))
    fullval_forensics_batch_size = int(eval_config.get('fullval_forensics_batch_size', 2))
    fullval_forensics_timeout_sec = int(eval_config.get('fullval_forensics_timeout_sec', 7200))
    eval_max_detections = int(eval_config.get('max_detections', 100))
    fullval_forensics_script = project_root / "tools" / "run_fullval_forensics.py"
    if fullval_forensics_enabled and not fullval_forensics_script.exists():
        raise RuntimeError(
            "fullval_forensics_enabled=true but tools/run_fullval_forensics.py is missing. "
            "Disable fullval_forensics or add the script."
        )

    # Close-mosaic: disable mosaic augmentation for the last N epochs.
    # Final epochs should train on clean, un-mosaiced images so the model
    # calibrates on the real data distribution (from V3.1 audit).
    close_mosaic_epoch = train_config.get('close_mosaic_epoch', None)

    # Gradient accumulation config for logging
    effective_batch_size = batch_size * accum_steps * world_size
    if effective_batch_size != QUALITY_GLOBAL_BATCH:
        raise RuntimeError(
            f"DDP3 effective global batch is {effective_batch_size}, "
            f"expected {QUALITY_GLOBAL_BATCH}"
        )
    flush_partial_accumulation = train_config.get('flush_partial_accumulation', False)
    nan_grad_action_cfg = str(train_config.get('nan_grad_action', 'skip')).lower()

    # Runtime recovery controls (not part of model architecture config)
    config['_runtime'] = {
        'strict_finite_checks': bool(args.strict_finite_checks),
        'finite_trace_dir': str(finite_trace_dir),
        'abort_on_skip_rate': float(args.abort_on_skip_rate),
        'abort_on_consecutive_nonfinite': int(args.abort_on_consecutive_nonfinite),
        'is_main_process': bool(is_main_process),
        'amp_mode': amp_mode,
        'max_train_batches': args.max_train_batches,
        'rank0_writer': bool(is_main_process),
        'launch_origin': copy.deepcopy(worker_context["launch_origin"]),
    }

    # Determine loss type string for logging
    loss_type_str = (
        'SparseVoxelDet (quality focal + decoded GIoU/NWD + decoded-IoU quality)'
        if loss_config.get('task_aligned_enabled', False)
        else 'SparseVoxelDet (strict focal + GIoU/NWD + centerness)'
    )

    if is_main_process:
        print(f"\n{'='*60}")
        print(f"Training Configuration Summary")
        print(f"{'='*60}")
        print(f"Epochs: {start_epoch} -> {epochs-1} ({epochs - start_epoch} total)")
        print(
            f"Batch size: {batch_size}/rank x {world_size} ranks x "
            f"{accum_steps} accum = {effective_batch_size} global effective"
        )
        print(f"Optimizer steps/epoch: {optimizer_steps_per_epoch} (from {len(train_loader)} loader batches)")
        print(f"Flush partial accumulation: {flush_partial_accumulation}")
        print(f"Non-finite grad action: {nan_grad_action_cfg}")
        print(f"AMP mode: {amp_mode} (scaler={'on' if scaler is not None else 'off'})")
        print(f"Strict finite checks: {args.strict_finite_checks}")
        print(f"Abort on skip rate: {args.abort_on_skip_rate:.3f} (rolling 500)")
        print(f"Abort on consecutive non-finite: {args.abort_on_consecutive_nonfinite}")
        if args.max_train_batches is not None:
            print(f"Max train batches/epoch override: {args.max_train_batches}")
        print(f"Skip validation: {args.skip_validation}")
        print(f"Model type: SparseVoxelDet (fully sparse)")
        print(f"Loss: {loss_type_str}")
        print(
            f"Full-val forensics: {fullval_forensics_enabled} "
            f"(interval={fullval_forensics_interval}, batch={fullval_forensics_batch_size}, "
            f"timeout={fullval_forensics_timeout_sec}s)"
        )
        if close_mosaic_epoch is not None:
            print(f"Close-mosaic epoch: {close_mosaic_epoch} "
                  f"(mosaic disabled for last {epochs - close_mosaic_epoch} epochs)")
        else:
            print(f"Close-mosaic: disabled (mosaic active for all epochs)")

    # Load existing history on resume so we never lose previous epoch records
    history = []
    history_path = output_dir / 'history.json'
    if history_path.exists() and start_epoch > 0:
        try:
            with open(history_path) as f:
                history = json.load(f)
            if is_main_process:
                print(f"Loaded {len(history)} existing history entries from {history_path}")
        except (json.JSONDecodeError, Exception) as e:
            if is_main_process:
                print(f"WARNING: Could not load history.json ({e}), starting fresh")
            history = []
    run_health_path = output_dir / "run_health.json"

    training_aborted = False
    training_stopped = False
    try:
        for epoch in range(start_epoch, epochs):
            # Set epoch for proper shuffling (DDP or EpochSubsetSampler)
            if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
                train_sampler.set_epoch(epoch)

            # Close-mosaic: disable mosaic augmentation at the specified epoch.
            # This modifies the dataset's mosaic_prob in-place so all subsequent
            # epochs also have mosaic disabled (prob stays at 0.0).
            if close_mosaic_epoch is not None and epoch == close_mosaic_epoch:
                if hasattr(train_dataset, 'mosaic_prob'):
                    old_prob = train_dataset.mosaic_prob
                    train_dataset.mosaic_prob = 0.0
                    if is_main_process:
                        print(f"\n  [Close-Mosaic] Epoch {epoch}: "
                              f"disabled mosaic (was {old_prob:.2f}, now 0.0)")

            if is_main_process:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch}/{epochs-1}")
                print(f"{'='*60}")

            # Train
            epoch_start_batch = start_batch_idx if epoch == start_epoch else 0
            epoch_start_optimizer_steps = (
                resume_optimizer_steps_in_epoch if epoch == start_epoch else 0
            )
            config['_runtime']['best_map'] = float(best_map)
            config['_runtime']['best_map_50_95'] = float(best_map_50_95)
            config['_runtime']['best_epoch'] = int(best_epoch)
            config['_runtime']['best_loss'] = float(best_loss)
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scheduler, loss_fn,
                device, epoch, config, scaler, ema=ema,
                output_dir=output_dir,
                start_batch_idx=epoch_start_batch,
                start_optimizer_steps_in_epoch=epoch_start_optimizer_steps,
                start_optimizer_steps_completed=optimizer_steps_completed,
            )
            optimizer_steps_completed = int(train_metrics["optimizer_steps_completed"])

            if is_main_process:
                print(f"\nTrain: loss={train_metrics['loss']:.4f} "
                      f"cls={train_metrics['cls_loss']:.4f} "
                      f"reg={train_metrics['reg_loss']:.4f} "
                      f"ctr={train_metrics['ctr_loss']:.4f} "
                      f"steps={int(train_metrics.get('optimizer_steps', 0))} "
                      f"skips={int(train_metrics.get('skipped_total', 0))} "
                      f"sanitized={int(train_metrics.get('sanitized_grad_steps', 0))}"
                      f" ({train_metrics.get('skip_rate', 0.0)*100:.1f}%) "
                      f"clip_rate={train_metrics.get('clip_rate', 0.0)*100:.1f}% "
                      f"clip_frac={train_metrics.get('clip_fraction_mean', 0.0):.3f}")
                print(
                    "Quality epoch: "
                    f"gt={int(train_metrics.get('quality_num_gt', 0))} "
                    f"zero={int(train_metrics.get('quality_gt_zero_candidates', 0))} "
                    f"quota={int(train_metrics.get('quality_dynamic_k_sum', 0))} "
                    f"pos={int(train_metrics.get('quality_num_pos_raw', 0))} "
                    f"fill={train_metrics.get('quality_quota_fill_ratio', 0.0):.3f} "
                    f"conflicts={int(train_metrics.get('quality_conflict_sites', 0))} "
                    f"unassigned={int(train_metrics.get('quality_gt_zero_after_conflict', 0))}"
                )
                if train_metrics.get("aborted_early", False):
                    print(f"ABORTED EARLY: {train_metrics.get('abort_reason', 'unknown')}")

            if train_metrics.get("stopped_by_controller", False) or train_metrics.get("aborted_early", False):
                training_stopped = bool(train_metrics.get("stopped_by_controller", False))
                training_aborted = bool(train_metrics.get("aborted_early", False))
                terminal_status = "controller_stop" if training_stopped else "aborted"
                metrics = train_metrics.copy()
                metrics["epoch"] = epoch
                history.append(metrics)

                def write_terminal_state():
                    with open(output_dir / 'history.json', 'w') as f:
                        json.dump(history, f, indent=2)
                    run_health = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "epoch": epoch,
                        "status": terminal_status,
                        "reason": train_metrics.get("abort_reason", ""),
                        "emergency_checkpoint": train_metrics.get("emergency_checkpoint", ""),
                        "optimizer_steps_completed": int(train_metrics.get("optimizer_steps_completed", 0)),
                        "skip_rate": train_metrics.get("skip_rate", 0.0),
                        "skipped_total": int(train_metrics.get("skipped_total", 0)),
                        "max_consecutive_nonfinite": int(train_metrics.get("max_consecutive_nonfinite", 0)),
                        "nonfinite_events": int(train_metrics.get("nonfinite_events", 0)),
                        "clip_rate": float(train_metrics.get("clip_rate", 0.0)),
                        "clip_fraction_mean": float(train_metrics.get("clip_fraction_mean", 0.0)),
                        "raw_voxels_mean": float(train_metrics.get("raw_voxels_mean", 0.0)),
                        "kept_voxels_mean": float(train_metrics.get("kept_voxels_mean", 0.0)),
                    }
                    run_health_path.write_text(json.dumps(run_health, indent=2))
                    return terminal_status

                run_rank0_stage(rank, "terminal run state", write_terminal_state)
                break

            # Validate
            metrics = train_metrics.copy()
            did_validate = False
            current_val_interval = val_interval
            if val_interval_after_epoch is not None and epoch >= int(val_interval_after_epoch):
                current_val_interval = max(1, val_interval_after)

            def commit_epoch_boundary_stop(reason: str) -> str:
                def write_emergency_checkpoint():
                    checkpoint_path = output_dir / "emergency_stop.pt"
                    checkpoint_metrics = dict(metrics)
                    checkpoint_metrics.update({
                        "status": "controller_power_stop",
                        "reason": reason,
                    })
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, epoch, checkpoint_metrics,
                        checkpoint_path, config, ema=ema,
                        optimizer_steps_in_epoch=int(train_metrics["optimizer_steps"]),
                        optimizer_steps_completed=optimizer_steps_completed,
                    )
                    return str(checkpoint_path.resolve())

                emergency_path = str(run_rank0_stage(
                    rank,
                    "epoch-boundary emergency full-state checkpoint",
                    write_emergency_checkpoint,
                ))
                terminal_metrics = dict(metrics)
                terminal_metrics.update({
                    "epoch": epoch,
                    "status": "controller_stop",
                    "reason": reason,
                    "stopped_by_controller": True,
                    "emergency_checkpoint": emergency_path,
                })
                history.append(terminal_metrics)

                def write_validation_stop_state():
                    with open(output_dir / "history.json", "w") as handle:
                        json.dump(history, handle, indent=2)
                    run_health = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "epoch": epoch,
                        "status": "controller_stop",
                        "reason": reason,
                        "emergency_checkpoint": emergency_path,
                        "optimizer_steps_completed": optimizer_steps_completed,
                    }
                    run_health_path.write_text(json.dumps(run_health, indent=2))
                    return emergency_path

                run_rank0_stage(rank, "validation-stop run state", write_validation_stop_state)
                return emergency_path

            if (not args.skip_validation) and ((epoch + 1) % current_val_interval == 0 or epoch == epochs - 1):
                did_validate = True

                def write_pre_validation_boundary():
                    checkpoint_metrics = dict(metrics)
                    checkpoint_metrics["status"] = "pre_validation_optimizer_boundary"
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, epoch, checkpoint_metrics,
                        output_dir / "latest.pt", config, ema=ema,
                        optimizer_steps_in_epoch=int(train_metrics["optimizer_steps"]),
                        optimizer_steps_completed=optimizer_steps_completed,
                    )
                    return str((output_dir / "latest.pt").resolve())

                run_rank0_stage(
                    rank,
                    "pre-validation optimizer-boundary checkpoint",
                    write_pre_validation_boundary,
                )
                if COLLECTIVE_STOP_REQUESTED(Path(worker_context["stop_request_path"]), device):
                    commit_epoch_boundary_stop(
                        "controller watchdog requested a power stop before validation"
                    )
                    training_stopped = True
                    break

                def validate_rank_zero():
                    if val_loader is None:
                        raise RuntimeError("Rank 0 validation loader is missing")
                    torch.cuda.empty_cache()
                    ema_applied = False
                    try:
                        if ema is not None:
                            ema.apply_shadow(raw_model)
                            ema_applied = True
                        try:
                            return validate(
                                raw_model,
                                val_loader,
                                loss_fn,
                                device,
                                config,
                                epoch=epoch,
                                stop_request_path=Path(worker_context["stop_request_path"]),
                            )
                        except ValidationStopRequested as error:
                            return {
                                "_controller_stop_requested": True,
                                "_controller_stop_reason": str(error),
                            }
                    finally:
                        if ema_applied:
                            ema.restore(raw_model)

                val_metrics = run_rank0_stage(rank, "unsharded rank-zero validation", validate_rank_zero)
                if not isinstance(val_metrics, dict):
                    raise RuntimeError("DDP3 validation broadcast returned malformed metrics")
                validation_stop = val_metrics.get("_controller_stop_requested") is True
                if validation_stop or COLLECTIVE_STOP_REQUESTED(
                    Path(worker_context["stop_request_path"]), device
                ):
                    reason = str(val_metrics.get(
                        "_controller_stop_reason",
                        "controller watchdog requested a power stop after validation",
                    ))
                    commit_epoch_boundary_stop(reason)
                    training_stopped = True
                    break
                metrics.update(val_metrics)

                def select_and_save_best():
                    selected_map = float(best_map)
                    selected_map_50_95 = float(best_map_50_95)
                    selected_epoch = int(best_epoch)
                    selected_loss = float(best_loss)
                    save_best = False
                    if 'mAP_50' in val_metrics:
                        current_map = float(val_metrics['mAP_50'])
                        current_map_50_95 = float(val_metrics['mAP_50_95'])
                        if current_map > selected_map or (
                            current_map == selected_map and current_map_50_95 > selected_map_50_95
                        ):
                            selected_map = current_map
                            selected_map_50_95 = current_map_50_95
                            selected_epoch = epoch
                            save_best = True
                    elif float(val_metrics['val_loss']) < selected_loss:
                        selected_loss = float(val_metrics['val_loss'])
                        selected_epoch = epoch
                        save_best = True
                    selection = {
                        "best_map": selected_map,
                        "best_map_50_95": selected_map_50_95,
                        "best_epoch": selected_epoch,
                        "best_loss": selected_loss,
                        "save_best": save_best,
                    }
                    if save_best:
                        checkpoint_metrics = dict(metrics)
                        checkpoint_metrics['selection_best_mAP_50'] = selected_map
                        checkpoint_metrics['selection_best_mAP_50_95'] = selected_map_50_95
                        checkpoint_metrics['selection_best_epoch'] = selected_epoch
                        checkpoint_metrics['selection_best_loss'] = selected_loss
                        save_checkpoint(
                            model, optimizer, scheduler, scaler, epoch, checkpoint_metrics,
                            output_dir / 'best.pt', config, ema=ema,
                            optimizer_steps_in_epoch=int(train_metrics["optimizer_steps"]),
                            optimizer_steps_completed=optimizer_steps_completed,
                        )
                    return selection

                selection = run_rank0_stage(rank, "validation selection and best checkpoint", select_and_save_best)
                if not isinstance(selection, dict) or set(selection) != {
                    "best_map", "best_map_50_95", "best_epoch", "best_loss", "save_best",
                }:
                    raise RuntimeError("DDP3 checkpoint-selection broadcast is malformed")
                best_map = float(selection["best_map"])
                best_map_50_95 = float(selection["best_map_50_95"])
                best_epoch = int(selection["best_epoch"])
                best_loss = float(selection["best_loss"])
                if COLLECTIVE_STOP_REQUESTED(
                    Path(worker_context["stop_request_path"]), device
                ):
                    commit_epoch_boundary_stop(
                        "controller watchdog requested a power stop during checkpoint selection"
                    )
                    training_stopped = True
                    break
                if is_main_process:
                    map_str = ""
                    if 'mAP_50' in val_metrics:
                        map_str = (f" mAP@50={val_metrics['mAP_50']:.4f}"
                                   f" mAP@50:95={val_metrics['mAP_50_95']:.4f}"
                                   f" P={val_metrics['precision']:.3f}"
                                   f" R={val_metrics['recall']:.3f}")
                    print(f"Val: loss={val_metrics['val_loss']:.4f} "
                          f"cls={val_metrics['val_cls_loss']:.4f} "
                          f"reg={val_metrics['val_reg_loss']:.4f} "
                          f"ctr={val_metrics['val_ctr_loss']:.4f}"
                          f"{map_str}")

            metrics['selection_best_mAP_50'] = float(best_map)
            metrics['selection_best_mAP_50_95'] = float(best_map_50_95)
            metrics['selection_best_epoch'] = int(best_epoch)
            metrics['selection_best_loss'] = float(best_loss)

            def write_epoch_checkpoints():
                checkpoint_kwargs = {
                    "optimizer_steps_in_epoch": int(train_metrics["optimizer_steps"]),
                    "optimizer_steps_completed": optimizer_steps_completed,
                }
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, metrics,
                    output_dir / 'latest.pt', config, ema=ema, **checkpoint_kwargs,
                )
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, metrics,
                    output_dir / f'epoch_{epoch:03d}.pt', config, ema=ema, **checkpoint_kwargs,
                )
                return {
                    "latest": str((output_dir / 'latest.pt').resolve()),
                    "epoch": str((output_dir / f'epoch_{epoch:03d}.pt').resolve()),
                }

            run_rank0_stage(rank, "latest and epoch checkpoints", write_epoch_checkpoints)

            if did_validate and audit_enabled and (epoch + 1) % audit_interval == 0:
                audit_dir = output_dir / f"audit_epoch_{epoch:03d}"
                audit_cmd = [
                    sys.executable,
                    str(project_root / "tools" / "run_compact_audit.py"),
                    "--model", "sparse_voxel_det",
                    "--checkpoint", str(output_dir / "latest.pt"),
                    "--outdir", str(audit_dir),
                    "--device", args.device,
                    "--reportability", "diagnostic",
                    "--split", str(val_split),
                    "--data-dir", str(sparse_dir),
                    "--label-dir", str(label_dir),
                    "--num-fp", str(audit_num_fp),
                    "--num-fn", str(audit_num_fn),
                    "--num-tp", str(audit_num_tp),
                    "--batch-size", str(audit_batch_size),
                ]
                if parity_enforced and audit_max_samples <= 0:
                    audit_cmd += ["--parity-enforced"]
                if audit_max_samples is not None and int(audit_max_samples) > 0:
                    audit_cmd += ["--max-samples", str(audit_max_samples)]
                if audit_max_detections is not None:
                    audit_cmd += ["--max-detections", str(audit_max_detections)]

                def run_compact_audit():
                    print(f"  Running compact visual audit: {audit_dir}")
                    try:
                        run_kwargs = {
                            "check": True,
                            "env": one_uuid_child_environment(QUALITY_ORDERED_UUIDS[0]),
                        }
                        if audit_timeout_sec > 0:
                            run_kwargs["timeout"] = audit_timeout_sec
                        subprocess.run(audit_cmd, **run_kwargs)
                        print("  Compact visual audit completed")
                        return {"status": "success"}
                    except Exception as error:
                        print(f"  WARNING: Compact visual audit failed: {error}")
                        return {"status": "warning", "error": f"{type(error).__name__}: {error}"}

                run_rank0_stage(rank, "compact audit", run_compact_audit)

            if did_validate and fullval_forensics_enabled and (epoch + 1) % fullval_forensics_interval == 0:
                forensic_dir = output_dir / "fullval_forensics" / f"epoch_{epoch:03d}"
                forensic_cmd = [
                    sys.executable,
                    str(fullval_forensics_script),
                    "--run-dir", str(output_dir),
                    "--checkpoint", str(output_dir / "latest.pt"),
                    "--device", args.device,
                    "--epoch", str(epoch),
                    "--batch-size", str(fullval_forensics_batch_size),
                    "--dump-batch-size", str(fullval_forensics_batch_size),
                    "--max-detections", str(eval_max_detections),
                    "--split", str(val_split),
                    "--data-dir", str(sparse_dir),
                    "--label-dir", str(label_dir),
                    "--timeout-sec", str(fullval_forensics_timeout_sec),
                    "--outdir", str(forensic_dir),
                ]
                if parity_enforced:
                    forensic_cmd += ["--parity-enforced"]

                def run_fullval_forensics():
                    print(f"  Running full-val forensic bundle: {forensic_dir}")
                    child_env = one_uuid_child_environment(QUALITY_ORDERED_UUIDS[0])
                    try:
                        subprocess.run(forensic_cmd, check=True, env=child_env)
                        casebook_cmd = [
                            sys.executable,
                            str(project_root / "tools" / "build_failure_casebook.py"),
                            "--run-dir", str(output_dir),
                            "--epoch", str(epoch),
                            "--forensic-dir", str(forensic_dir),
                            "--outdir", str(output_dir / "forensics"),
                        ]
                        subprocess.run(casebook_cmd, check=True, env=child_env)
                        print("  Full-val forensic bundle completed")
                        return {"status": "success"}
                    except Exception as error:
                        print(f"  WARNING: Full-val forensic bundle failed: {error}")
                        return {"status": "warning", "error": f"{type(error).__name__}: {error}"}

                run_rank0_stage(rank, "full-validation forensics", run_fullval_forensics)

            metrics['epoch'] = epoch
            history.append(metrics)

            def write_epoch_state():
                with open(output_dir / 'history.json', 'w') as f:
                    json.dump(history, f, indent=2)
                run_health = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "epoch": epoch,
                    "status": "running",
                    "optimizer_steps_completed": optimizer_steps_completed,
                    "skip_rate": float(metrics.get("skip_rate", 0.0)),
                    "skipped_total": int(metrics.get("skipped_total", 0)),
                    "max_consecutive_nonfinite": int(metrics.get("max_consecutive_nonfinite", 0)),
                    "nonfinite_events": int(metrics.get("nonfinite_events", 0)),
                    "clip_rate": float(metrics.get("clip_rate", 0.0)),
                    "clip_fraction_mean": float(metrics.get("clip_fraction_mean", 0.0)),
                    "abort_on_skip_rate": float(args.abort_on_skip_rate),
                    "abort_on_consecutive_nonfinite": int(args.abort_on_consecutive_nonfinite),
                    "strict_finite_checks": bool(args.strict_finite_checks),
                }
                if "mAP_50" in metrics:
                    run_health["mAP_50"] = float(metrics["mAP_50"])
                    run_health["precision"] = float(metrics.get("precision", 0.0))
                    run_health["recall"] = float(metrics.get("recall", 0.0))
                    run_health["metrics_engine_id"] = str(metrics.get("metrics_engine_id", "sparse_voxel_det.mapcalc"))
                    run_health["metrics_version"] = str(metrics.get("metrics_version", "2026-02-26"))
                run_health_path.write_text(json.dumps(run_health, indent=2))
                return str(run_health_path.resolve())

            run_rank0_stage(rank, "history and run health", write_epoch_state)

        if is_main_process:
            if training_stopped:
                print("\nTraining stopped by the controller watchdog after an emergency checkpoint.")
            elif training_aborted:
                print("\nTraining stopped by recovery abort gates.")
            else:
                print("\nTraining complete!")
            if best_map > 0:
                print(f"Best mAP@50: {best_map:.4f}")
            print(f"Best validation loss: {best_loss:.4f}")
            print(f"Checkpoints saved to: {output_dir}")

    finally:
        cleanup_ddp()

    if training_stopped:
        return 3
    return 2 if training_aborted else 0


if __name__ == '__main__':
    raise SystemExit(main())
