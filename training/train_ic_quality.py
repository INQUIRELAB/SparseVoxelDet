#!/usr/bin/env python3
"""Fail-closed controller and protected three-rank launcher for DDP3 quality IC."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


BASE = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[3]
INVESTIGATORS = PROJECT / "tools/sol_forensics/investigators"
CONFIG = BASE / "ic_quality_ddp3_e20.yaml"
OUTPUT_DIR = PROJECT / "tools/sol_forensics/ic_quality_ddp3_arm_2026-07-20/runs/ic_quality_ddp3_seed42"
CONTRACTS_DIR = BASE / "contracts_ddp3"
ACTIVE_CLAIM = CONTRACTS_DIR / "ic_quality_ddp3_seed42.active"
STRICT_CONFIG = PROJECT / "tools/sol_forensics/stage_b_ic_arm_2026-07-16/ic_nwd_e20.yaml"
LABEL_ROOT = PROJECT / "data/datasets/fred_paper_parity/labels_rawcomplete"
RUNTIME_POLICY_VERSION = "ic-quality-ddp3-seed42-v2"
CHECKPOINT_LINEAGE = "ic-quality-aligned-ddp3-seed42"
OUTPUT_RESERVATION_FILE = ".quality_ddp3_writer_claim.json"
PREFLIGHT_RECEIPT = BASE / "preflight.json"
PREFLIGHT_REPLAY = BASE / "preflight_replay.json"
WORLD_SIZE = 3
SAMPLER_SEED = 42
PER_RANK_BATCH = 2
ACCUMULATION_STEPS = 1
GLOBAL_BATCH = 6
TRAIN_ROSTER_SAMPLES = 406701
SAMPLER_SAMPLES = 406701
OPTIMIZED_SAMPLES = 406698
OPTIMIZER_STEPS_PER_EPOCH = 67783
WARMUP_STEPS = 5000
EPOCHS = 20
TOTAL_OPTIMIZER_STEPS = 1355660
POWER_LIMIT_WATTS = 400.0
AGGREGATE_POWER_LIMIT_WATTS = 2000.0
PROTECTED_PHYSICAL_INDICES = {7, 8}
WATCHDOG_INTERVAL_SECONDS = 5.0
GPU_QUERY_TIMEOUT_SECONDS = 15.0
COORDINATED_STOP_GRACE_SECONDS = 45.0
FORCED_STOP_GRACE_SECONDS = 15.0
ORDERED_UUIDS = (
    "GPU-1d11b997-90a9-ece7-9ce6-44ad85346817",
    "GPU-2a7554bd-5a91-25ab-3338-e2308ecb2a27",
    "GPU-48d3a2b0-fc78-8bc8-fdce-5a246fdc4989",
)
FORBIDDEN_UUIDS = frozenset({
    "GPU-b279b278-d3e7-eb16-73d2-f6f4b002276c",
})
EXPECTED_LABEL_MANIFEST_SHA = "6a973831e215c733e77f4ba2553ae0e138a20cf01f1c5e30387292f52b2c56ee"
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
EXPECTED_VAL = {
    key: EXPECTED_LABEL_SPLITS["canonical_val"][key]
    for key in ("files", "boxes", "two_box_files")
}


EXPECTED_BUILD = {
    BASE / "strict_loss.py": "92957d72221c72a656bd10aef3fdc1f74e3f5c35e6c118be8ef8c6e6cc3c4526",
    BASE / "quality_aligned_loss.py": "08b7030fbab85449ef4a17ce4a373e73fa2c8392fb12c28119dc23d6c06d6290",
    BASE / "sparse_trainer_ic_quality.py": "45c4ded584a6e60b0b24100762487f8c6f26a97a330c1618dc0921a3fbabd442",
    CONFIG: "1067e925c3ffe753c4bdbf5816e30a38d863b48d184d9638720c374cc86bbf42",
    BASE / "test_quality_aligned_loss.py": "41320d1f75ee34af60ec1bc5922ecd334f8e81082c91ad778b417de3534f700e",
    BASE / "test_ddp3_contracts.py": "e02239fdbb0acd546cc456328f6b83578ec33f0088d354f5a80045e412990dae",
}
EXPECTED_REMOTE = {
    INVESTIGATORS / "se_per_sample_patch.py": "d4a48432eeec12c2cb0dd4267081e56ef13771eaf1e7357466ecf995c4ae291e",
    PROJECT / "models/snn/sparse_sew_resnet.py": "d2bdd011954126d6011409269b8db26c86cf77cb5605a492c308ceceb3e3b4a4",
    PROJECT / "V2/__init__.py": "d53555a2665ba026873e8bc4ae32d278503b0a32c612e7bf672f2af86376525a",
    PROJECT / "V2/models/__init__.py": "1da219e44a219daa274a6810a6bdd23fd495dc5292f7f894af2bf0c05ad26f24",
    PROJECT / "V2/models/sparse_voxel_det_v82.py": "c86cc66f98e1049ef868c6aa9c6de13071134ae79e74ba9e3d79a141927fd39a",
    PROJECT / "V2/models/sparse_voxel_det_ic.py": "b298de64849793eac84fb5adc3ffac9a4e76ff9f06ca464cde7c4b82e1df2a60",
    PROJECT / "sparse_fcos_v1/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    PROJECT / "sparse_fcos_v1/scripts/__init__.py": "b3fcb851f21af67cf87f13b7ac710f859d28761a6afebc20d1b8907bb8fbf425",
    PROJECT / "sparse_fcos_v1/scripts/event_mosaic.py": "031fe548dc8a9de4ec5ecb1184b5402e89a272b5df63ecbfda8225d6cdd80684",
    PROJECT / "sparse_fcos_v1/scripts/sparse_event_dataset_v82.py": "eda3f6647684882b097a64a60682f6c9b8922ad1abfd63931c1570b1fa433576",
    PROJECT / "sparse_fcos_v1/scripts/ema.py": "194dabd20c5e5afa4d83da03e144dc5a110a6343a814364b32c8a68a1e79ce8f",
    PROJECT / "sparse_fcos_v1/scripts/metrics.py": "23c8632412719e3e917459b40ad601d4315cab67ced789ba385982c04c6cf6ba",
    PROJECT / "sparse_fcos_v1/scripts/evaluate_sparse_fcos.py": "73fab955126fb9a10d1ccb8c4ce467fd1b4282509479333da632c85c383717d1",
    PROJECT / "tools/validate_sparse_tensor_contract.py": "375b55513952fe03190313089466b20ccbfad5558cd003f2027454c0728672ad",
}
RUNTIME_SOURCES = {
    "strict_loss": BASE / "strict_loss.py",
    "quality_aligned_loss": BASE / "quality_aligned_loss.py",
    "models.snn.sparse_sew_resnet": PROJECT / "models/snn/sparse_sew_resnet.py",
    "se_per_sample_patch": INVESTIGATORS / "se_per_sample_patch.py",
    "V2": PROJECT / "V2/__init__.py",
    "V2.models": PROJECT / "V2/models/__init__.py",
    "V2.models.sparse_voxel_det_v82": PROJECT / "V2/models/sparse_voxel_det_v82.py",
    "V2.models.sparse_voxel_det_ic": PROJECT / "V2/models/sparse_voxel_det_ic.py",
    "sparse_fcos_v1": PROJECT / "sparse_fcos_v1/__init__.py",
    "sparse_fcos_v1.scripts": PROJECT / "sparse_fcos_v1/scripts/__init__.py",
    "sparse_fcos_v1.scripts.event_mosaic": PROJECT / "sparse_fcos_v1/scripts/event_mosaic.py",
    "sparse_fcos_v1.scripts.sparse_event_dataset_v82": PROJECT / "sparse_fcos_v1/scripts/sparse_event_dataset_v82.py",
    "sparse_fcos_v1.scripts.ema": PROJECT / "sparse_fcos_v1/scripts/ema.py",
    "sparse_fcos_v1.scripts.metrics": PROJECT / "sparse_fcos_v1/scripts/metrics.py",
    "sparse_fcos_v1.scripts.evaluate_sparse_fcos": PROJECT / "sparse_fcos_v1/scripts/evaluate_sparse_fcos.py",
    "sparse_tensor_contract_validator": PROJECT / "tools/validate_sparse_tensor_contract.py",
    "sparse_trainer_ic_quality": BASE / "sparse_trainer_ic_quality.py",
}
DDP_ENV_KEYS = {
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR",
    "MASTER_PORT", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
}
INTERNAL_ENV_KEYS = {
    "SPARSEVOXELDET_DDP3_MODE", "SPARSEVOXELDET_DDP3_CLAIM",
    "SPARSEVOXELDET_DDP3_TOKEN", "SPARSEVOXELDET_DDP3_CONTROLLER_ID",
}
CONTROLLER_PROTECTED_ENV_KEYS = DDP_ENV_KEYS | INTERNAL_ENV_KEYS | {
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
}


class StoreOnce(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        seen = getattr(namespace, "_seen_quality_flags", set())
        if self.dest in seen:
            parser.error(f"duplicate protected argument: {option_string}")
        setattr(namespace, "_seen_quality_flags", seen | {self.dest})
        setattr(namespace, self.dest, values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_open_handle(handle) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def stat_identity(stat: os.stat_result) -> dict[str, int]:
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def load_json_object(source: bytes, label: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"Duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(source, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Malformed JSON in {label}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root is not an object in {label}")
    return payload


def write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl_fsync(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def central_now() -> datetime:
    return datetime.now(ZoneInfo("America/Chicago"))


def parse_launch_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True, action=StoreOnce)
    parser.add_argument("--output_dir", required=True, action=StoreOnce)
    parser.add_argument("--seed", required=True, type=int, action=StoreOnce)
    parser.add_argument("--resume", default=None, action=StoreOnce)
    args = parser.parse_args(argv)
    if hasattr(args, "_seen_quality_flags"):
        delattr(args, "_seen_quality_flags")
    return args


def validate_identity_args(args: argparse.Namespace) -> None:
    if args.seed != 42:
        raise RuntimeError(f"DDP3 quality arm permits seed 42 only, got {args.seed}")
    if Path(args.config).resolve() != CONFIG.resolve():
        raise RuntimeError(f"Unauthorized DDP3 quality config: {args.config}")
    if Path(args.output_dir).resolve() != OUTPUT_DIR.resolve():
        raise RuntimeError(f"Unauthorized DDP3 quality output: {args.output_dir}")


def validate_controller_args(args: argparse.Namespace) -> None:
    validate_identity_args(args)
    if args.resume is None and OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite DDP3 quality run: {OUTPUT_DIR}")
    if args.resume is not None and not OUTPUT_DIR.is_dir():
        raise RuntimeError(f"DDP3 resume output directory is missing: {OUTPUT_DIR}")


def reject_controller_environment(environment: dict[str, str] | None = None) -> None:
    source = os.environ if environment is None else environment
    present = sorted(key for key in CONTROLLER_PROTECTED_ENV_KEYS if key in source)
    if present:
        raise RuntimeError(
            f"DDP3 controller forbids user-supplied protected environment variables: {present}"
        )


def load_verified_yaml(path: Path, expected_hash: str) -> dict[str, object]:
    source = path.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual != expected_hash:
        raise RuntimeError(f"DDP3 source drift for {path}: expected {expected_hash}, got {actual}")
    payload = yaml.safe_load(source)
    if not isinstance(payload, dict):
        raise RuntimeError(f"DDP3 YAML is malformed: {path}")
    return payload


def validate_protocol_config(config: dict[str, object]) -> None:
    training = config.get("training")
    ddp = config.get("ddp")
    if not isinstance(training, dict) or not isinstance(ddp, dict):
        raise RuntimeError("DDP3 quality config is missing training or ddp policy")
    expected_training = {
        "epochs": EPOCHS,
        "batch_size": PER_RANK_BATCH,
        "num_workers": 4,
        "gradient_accumulation_steps": ACCUMULATION_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "scheduler": "cosine",
        "use_amp": True,
        "max_samples_per_epoch": None,
        "flush_partial_accumulation": False,
    }
    for key, value in expected_training.items():
        if training.get(key) != value:
            raise RuntimeError(f"DDP3 protocol mismatch for training.{key}: {training.get(key)!r}")
    expected_ddp = {
        "runtime_policy_version": RUNTIME_POLICY_VERSION,
        "world_size": WORLD_SIZE,
        "local_world_size": WORLD_SIZE,
        "per_rank_batch_size": PER_RANK_BATCH,
        "workers_per_rank": 4,
        "gradient_accumulation_steps": ACCUMULATION_STEPS,
        "global_effective_batch_size": GLOBAL_BATCH,
        "train_roster_samples": TRAIN_ROSTER_SAMPLES,
        "sampler_samples_per_epoch": SAMPLER_SAMPLES,
        "optimized_samples_per_epoch": OPTIMIZED_SAMPLES,
        "optimizer_steps_per_epoch": OPTIMIZER_STEPS_PER_EPOCH,
        "warmup_optimizer_steps": WARMUP_STEPS,
        "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "selected_power_limit_watts": 400,
        "aggregate_power_limit_watts_exclusive": 2000,
    }
    for key, value in expected_ddp.items():
        if ddp.get(key) != value:
            raise RuntimeError(f"DDP3 policy mismatch for ddp.{key}: {ddp.get(key)!r}")
    if tuple(ddp.get("forbidden_physical_uuids", ())) != tuple(sorted(FORBIDDEN_UUIDS)):
        raise RuntimeError("DDP3 forbidden physical UUID policy mismatch")
    if tuple(ddp.get("ordered_physical_uuids", ())) != ORDERED_UUIDS:
        raise RuntimeError("DDP3 ordered physical UUID policy mismatch")
    if ddp.get("sampler") != {"seed": 42, "shuffle": True, "drop_last": True}:
        raise RuntimeError("DDP3 sampler policy mismatch")
    if ddp.get("validation") != {
        "owner_rank": 0,
        "sharded": False,
        "roster_samples": 103672,
    }:
        raise RuntimeError("DDP3 validation policy mismatch")
    if config.get("experiment", {}).get("name") != "ic_quality_aligned_ddp3_seed42_e20":
        raise RuntimeError("DDP3 experiment identity mismatch")


def build_artifact_paths() -> dict[str, Path]:
    return {
        "strict_loss": BASE / "strict_loss.py",
        "quality_loss": BASE / "quality_aligned_loss.py",
        "trainer": BASE / "sparse_trainer_ic_quality.py",
        "config": CONFIG,
        "launcher": Path(__file__).resolve(),
        "preflight": BASE / "preflight_quality.py",
        "quality_tests": BASE / "test_quality_aligned_loss.py",
        "contract_tests": BASE / "test_ddp3_contracts.py",
    }


def verify_static_sources() -> dict[str, dict[str, dict[str, str]]]:
    for path, expected in {**EXPECTED_BUILD, **EXPECTED_REMOTE}.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"DDP3 source drift for {path}: expected {expected}, got {actual}")
    build = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in build_artifact_paths().items()
    }
    remote = {
        name: {"path": str(path), "sha256": EXPECTED_REMOTE[path]}
        for name, path in {
            "per_sample_se": INVESTIGATORS / "se_per_sample_patch.py",
            "sparse_sew": PROJECT / "models/snn/sparse_sew_resnet.py",
            "v2_package": PROJECT / "V2/__init__.py",
            "v2_models_package": PROJECT / "V2/models/__init__.py",
            "base_model": PROJECT / "V2/models/sparse_voxel_det_v82.py",
            "ic_model": PROJECT / "V2/models/sparse_voxel_det_ic.py",
            "sparse_fcos_package": PROJECT / "sparse_fcos_v1/__init__.py",
            "sparse_fcos_scripts_package": PROJECT / "sparse_fcos_v1/scripts/__init__.py",
            "event_mosaic": PROJECT / "sparse_fcos_v1/scripts/event_mosaic.py",
            "dataset": PROJECT / "sparse_fcos_v1/scripts/sparse_event_dataset_v82.py",
            "ema": PROJECT / "sparse_fcos_v1/scripts/ema.py",
            "metrics": PROJECT / "sparse_fcos_v1/scripts/metrics.py",
            "evaluator": PROJECT / "sparse_fcos_v1/scripts/evaluate_sparse_fcos.py",
            "sparse_contract_validator": PROJECT / "tools/validate_sparse_tensor_contract.py",
        }.items()
    }
    return {"build": build, "remote": remote}


def expected_runtime_contract(
    source_hashes: dict[str, dict[str, dict[str, str]]],
) -> dict[str, object]:
    return {
        "experiment_name": "ic_quality_aligned_ddp3_seed42_e20",
        "seed": 42,
        "runtime_policy_version": RUNTIME_POLICY_VERSION,
        "checkpoint_lineage": CHECKPOINT_LINEAGE,
        "source_hashes": source_hashes,
        "runtime_source_sha256": {
            name: (
                EXPECTED_BUILD[path]
                if path in EXPECTED_BUILD
                else EXPECTED_REMOTE[path]
            )
            for name, path in RUNTIME_SOURCES.items()
        },
        "ordered_physical_uuids": list(ORDERED_UUIDS),
        "forbidden_physical_uuids": sorted(FORBIDDEN_UUIDS),
        "world_size": WORLD_SIZE,
        "local_world_size": WORLD_SIZE,
        "rank_mapping_policy": [
            {"rank": rank, "local_rank": rank, "uuid": ORDERED_UUIDS[rank]}
            for rank in range(WORLD_SIZE)
        ],
        "per_rank_batch_size": PER_RANK_BATCH,
        "global_effective_batch_size": GLOBAL_BATCH,
        "gradient_accumulation_steps": ACCUMULATION_STEPS,
        "sampler": {"seed": SAMPLER_SEED, "shuffle": True, "drop_last": True},
        "train_roster_samples": TRAIN_ROSTER_SAMPLES,
        "sampler_samples_per_epoch": SAMPLER_SAMPLES,
        "optimized_samples_per_epoch": OPTIMIZED_SAMPLES,
        "optimizer_steps_per_epoch": OPTIMIZER_STEPS_PER_EPOCH,
        "warmup_optimizer_steps": WARMUP_STEPS,
        "epochs": EPOCHS,
        "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "scheduler": "full_cosine",
        "validation": {"owner_rank": 0, "sharded": False, "roster_samples": 103672},
        "resume_policy": "full-state; archived DDP3 commit; one verified load per rank",
    }


def load_exact_source_module(
    module_name: str,
    expected_path: Path,
    expected_hash: str,
):
    expected_path = expected_path.resolve()
    source = expected_path.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual != expected_hash:
        raise RuntimeError(
            f"Loaded module source drift: expected {expected_hash}, got {actual} for {expected_path}"
        )
    spec = importlib.util.spec_from_file_location(module_name, expected_path)
    if spec is None:
        raise RuntimeError(f"Could not create module spec for {module_name}: {expected_path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(expected_path), "exec"), module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    if Path(module.__file__).resolve() != expected_path:
        raise RuntimeError(
            f"Loaded module origin mismatch: expected {expected_path}, got {module.__file__}"
        )
    parent_name, separator, child_name = module_name.rpartition(".")
    if separator and parent_name in sys.modules:
        setattr(sys.modules[parent_name], child_name, module)
    return module


def read_boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    if not path.is_file():
        raise RuntimeError("DDP3 controller requires Linux /proc boot identity")
    return path.read_text(encoding="utf-8").strip()


def read_process_start_ticks(pid: int) -> int | None:
    path = Path("/proc") / str(pid) / "stat"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    closing = text.rfind(")")
    if closing < 0:
        raise RuntimeError(f"Malformed process stat for PID {pid}")
    return int(text[closing + 2 :].split()[19])


def claim_owner_alive(owner: dict[str, object]) -> bool | None:
    try:
        pid = int(owner["pid"])
        hostname = str(owner["hostname"])
        boot_id = str(owner["boot_id"])
        start_ticks = int(owner["process_start_ticks"])
    except (KeyError, TypeError, ValueError):
        return None
    if hostname != socket.gethostname():
        return None
    if boot_id != read_boot_id():
        return False
    current = read_process_start_ticks(pid)
    return False if current is None else current == start_ticks


def archive_claim(claim_path: Path, reason: str) -> Path:
    archive_root = claim_path.parent / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = central_now().strftime("%Y-%m-%d_%H%M%S_%f")
    destination = archive_root / f"{stamp}_{reason}"
    claim_path.replace(destination)
    return destination


def reconcile_stale_controller_claim() -> Path | None:
    if not ACTIVE_CLAIM.exists():
        return None
    if ACTIVE_CLAIM.is_symlink() or not ACTIVE_CLAIM.is_dir():
        raise RuntimeError(f"DDP3 controller claim is unverifiable: {ACTIVE_CLAIM}")
    owner_path = ACTIVE_CLAIM / "owner.json"
    if owner_path.is_symlink() or not owner_path.is_file():
        raise RuntimeError(f"DDP3 controller claim owner is unverifiable: {owner_path}")
    previous = load_json_object(owner_path.read_bytes(), str(owner_path))
    alive = claim_owner_alive(previous)
    if alive is not False:
        state = "live" if alive else "unverifiable"
        raise RuntimeError(f"DDP3 controller claim is {state}: {previous}")
    return archive_claim(ACTIVE_CLAIM, "stale_controller_reconciled")


def acquire_controller_claim(
    mode: str,
    resume_sha256: str | None,
    controller_id: str,
    token_sha256: str,
    reservation_id: str,
) -> tuple[Path, dict[str, object], bytes]:
    CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    start_ticks = read_process_start_ticks(pid)
    if start_ticks is None:
        raise RuntimeError(f"Cannot prove DDP3 controller process identity for PID {pid}")
    owner = {
        "status": "active",
        "claimed_central": central_now().isoformat(),
        "hostname": socket.gethostname(),
        "pid": pid,
        "boot_id": read_boot_id(),
        "process_start_ticks": start_ticks,
        "mode": mode,
        "output_dir": str(OUTPUT_DIR.resolve()),
        "resume_sha256": resume_sha256,
        "controller_id": controller_id,
        "launch_token_sha256": token_sha256,
        "reservation_id": reservation_id,
        "runtime_policy_version": RUNTIME_POLICY_VERSION,
    }
    try:
        ACTIVE_CLAIM.mkdir(exist_ok=False)
    except FileExistsError:
        owner_path = ACTIVE_CLAIM / "owner.json"
        if owner_path.is_symlink() or not owner_path.is_file():
            raise RuntimeError(f"DDP3 controller claim is unverifiable: {ACTIVE_CLAIM}")
        previous = load_json_object(owner_path.read_bytes(), str(owner_path))
        alive = claim_owner_alive(previous)
        if alive is not False:
            state = "live" if alive else "unverifiable"
            raise RuntimeError(f"DDP3 controller claim is {state}: {previous}")
        archive_claim(ACTIVE_CLAIM, "stale_controller")
        ACTIVE_CLAIM.mkdir(exist_ok=False)
    try:
        write_json_exclusive(ACTIVE_CLAIM / "owner.json", owner)
    except BaseException:
        try:
            archive_claim(ACTIVE_CLAIM, "incomplete_controller_claim")
        except BaseException:
            pass
        raise
    owner_bytes = (ACTIVE_CLAIM / "owner.json").read_bytes()
    return ACTIVE_CLAIM, owner, owner_bytes


def finalize_controller_claim(
    claim_path: Path,
    status: str,
    error: str | None = None,
) -> Path:
    write_json_exclusive(
        claim_path / "final.json",
        {
            "status": status,
            "completed_central": central_now().isoformat(),
            "error": error,
        },
    )
    return archive_claim(claim_path, f"controller_{status}")


def read_existing_reservation_id() -> str:
    marker_path = OUTPUT_DIR / OUTPUT_RESERVATION_FILE
    if OUTPUT_DIR.is_symlink() or not OUTPUT_DIR.is_dir():
        raise RuntimeError(f"DDP3 resume output is missing or invalid: {OUTPUT_DIR}")
    if marker_path.is_symlink() or not marker_path.is_file():
        raise RuntimeError(f"DDP3 output reservation marker is missing or invalid: {marker_path}")
    marker = load_json_object(marker_path.read_bytes(), str(marker_path))
    required = {
        "status": "reserved",
        "mode": "fresh",
        "runtime_policy_version": RUNTIME_POLICY_VERSION,
        "seed": 42,
        "output_dir": str(OUTPUT_DIR.resolve()),
        "claim_path": str(ACTIVE_CLAIM.resolve()),
    }
    for key, expected in required.items():
        if marker.get(key) != expected:
            raise RuntimeError(f"DDP3 output reservation mismatch for {key}: {marker.get(key)!r}")
    reservation_id = marker.get("reservation_id")
    if not isinstance(reservation_id, str) or len(reservation_id) < 32:
        raise RuntimeError("DDP3 output reservation ID is invalid")
    return reservation_id


def reserve_fresh_output(
    claim_path: Path,
    owner_bytes: bytes,
    reservation_id: str,
) -> Path:
    owner = load_json_object(owner_bytes, str(claim_path / "owner.json"))
    if (
        owner.get("status") != "active"
        or owner.get("mode") != "fresh"
        or owner.get("pid") != os.getpid()
        or owner.get("reservation_id") != reservation_id
        or Path(str(owner.get("output_dir", ""))).resolve() != OUTPUT_DIR.resolve()
    ):
        raise RuntimeError(f"DDP3 controller claim cannot reserve fresh output: {owner}")
    OUTPUT_DIR.mkdir(exist_ok=False)
    marker = {
        "status": "reserved",
        "mode": "fresh",
        "runtime_policy_version": RUNTIME_POLICY_VERSION,
        "seed": 42,
        "output_dir": str(OUTPUT_DIR.resolve()),
        "claim_path": str(claim_path.resolve()),
        "claim_owner_sha256": hashlib.sha256(owner_bytes).hexdigest(),
        "reservation_id": reservation_id,
        "reserved_central": central_now().isoformat(),
    }
    marker_path = OUTPUT_DIR / OUTPUT_RESERVATION_FILE
    write_json_exclusive(marker_path, marker)
    verify_output_reservation(claim_path, require_pristine=True)
    return marker_path


def verify_output_reservation(
    claim_path: Path,
    require_pristine: bool,
) -> dict[str, object]:
    owner_path = claim_path / "owner.json"
    marker_path = OUTPUT_DIR / OUTPUT_RESERVATION_FILE
    if owner_path.is_symlink() or not owner_path.is_file():
        raise RuntimeError(f"DDP3 controller owner is missing or invalid: {owner_path}")
    if OUTPUT_DIR.is_symlink() or not OUTPUT_DIR.is_dir():
        raise RuntimeError(f"DDP3 output reservation is missing or invalid: {OUTPUT_DIR}")
    if marker_path.is_symlink() or not marker_path.is_file():
        raise RuntimeError(f"DDP3 output marker is missing or invalid: {marker_path}")
    owner_bytes = owner_path.read_bytes()
    owner = load_json_object(owner_bytes, str(owner_path))
    marker = load_json_object(marker_path.read_bytes(), str(marker_path))
    expected_keys = {
        "status", "mode", "runtime_policy_version", "seed", "output_dir",
        "claim_path", "claim_owner_sha256", "reservation_id", "reserved_central",
    }
    if set(marker) != expected_keys:
        raise RuntimeError(
            f"DDP3 output reservation fields changed: {sorted(set(marker) ^ expected_keys)}"
        )
    required = {
        "status": "reserved",
        "mode": "fresh",
        "runtime_policy_version": RUNTIME_POLICY_VERSION,
        "seed": 42,
        "output_dir": str(OUTPUT_DIR.resolve()),
        "claim_path": str(claim_path.resolve()),
        "reservation_id": owner.get("reservation_id"),
    }
    for key, expected in required.items():
        if marker.get(key) != expected:
            raise RuntimeError(f"DDP3 output reservation mismatch for {key}: {marker.get(key)!r}")
    if (
        owner.get("status") != "active"
        or owner.get("controller_id") is None
        or claim_owner_alive(owner) is not True
        or Path(str(owner.get("output_dir", ""))).resolve() != OUTPUT_DIR.resolve()
    ):
        raise RuntimeError(f"DDP3 controller no longer owns the output: {owner}")
    if require_pristine:
        if marker.get("claim_owner_sha256") != hashlib.sha256(owner_bytes).hexdigest():
            raise RuntimeError("Fresh DDP3 reservation is not bound to the controller owner bytes")
        entries = {entry.name for entry in os.scandir(OUTPUT_DIR)}
        if entries != {OUTPUT_RESERVATION_FILE}:
            raise RuntimeError(f"Fresh DDP3 output contains unowned entries: {sorted(entries)}")
    if not isinstance(marker.get("reserved_central"), str) or not marker["reserved_central"]:
        raise RuntimeError("DDP3 reservation timestamp is missing")
    return marker


def parse_csv_rows(text: str, field_names: tuple[str, ...]) -> list[dict[str, str]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(field_names):
            raise RuntimeError(f"Malformed nvidia-smi row: {line!r}")
        rows.append(dict(zip(field_names, parts)))
    return rows


def finite_nonnegative_float(value: object, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Non-numeric {label}: {value!r}") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise RuntimeError(f"Invalid {label}: {value!r}")
    return parsed


def nonnegative_int(value: object, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Non-integer {label}: {value!r}") from error
    if parsed < 0:
        raise RuntimeError(f"Invalid {label}: {value!r}")
    return parsed


def query_gpu_inventory() -> list[dict[str, str]]:
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,power.limit,power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=GPU_QUERY_TIMEOUT_SECONDS,
    )
    return parse_csv_rows(
        text,
        ("index", "uuid", "name", "power_limit", "power_draw", "memory_used"),
    )


def query_compute_apps() -> list[dict[str, str]]:
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=GPU_QUERY_TIMEOUT_SECONDS,
    )
    return parse_csv_rows(text, ("pid", "uuid"))


def evaluate_controller_gpu_rows(
    gpu_rows: list[dict[str, str]],
    compute_rows: list[dict[str, str]],
    selected_uuids: tuple[str, ...] = ORDERED_UUIDS,
) -> dict[str, object]:
    forbidden_selected = sorted(FORBIDDEN_UUIDS.intersection(selected_uuids))
    if forbidden_selected:
        raise RuntimeError(f"DDP3 selection includes forbidden baseline GPU: {forbidden_selected}")
    if tuple(selected_uuids) != ORDERED_UUIDS:
        raise RuntimeError(f"Foreign or reordered DDP3 UUID selection: {selected_uuids}")
    if len(set(selected_uuids)) != WORLD_SIZE:
        raise RuntimeError("DDP3 selected UUID list contains duplicates")
    by_uuid: dict[str, dict[str, str]] = {}
    seen_indices: set[int] = set()
    aggregate_draw = 0.0
    for row in gpu_rows:
        uuid = row["uuid"]
        if uuid in by_uuid:
            raise RuntimeError(f"Duplicate GPU UUID in nvidia-smi inventory: {uuid}")
        index = nonnegative_int(row["index"], "GPU index")
        draw = finite_nonnegative_float(row["power_draw"], f"GPU {uuid} power draw")
        finite_nonnegative_float(row["power_limit"], f"GPU {uuid} power limit")
        nonnegative_int(row["memory_used"], f"GPU {uuid} memory usage")
        if index in seen_indices:
            raise RuntimeError(f"Duplicate physical GPU index in inventory: {index}")
        seen_indices.add(index)
        by_uuid[uuid] = row
        aggregate_draw += draw
    missing = [uuid for uuid in selected_uuids if uuid not in by_uuid]
    if missing:
        raise RuntimeError(f"DDP3 selected UUIDs are missing: {missing}")
    known_uuids = set(by_uuid)
    selected_pids: dict[str, list[int]] = {uuid: [] for uuid in selected_uuids}
    for row in compute_rows:
        uuid = row["uuid"]
        if uuid not in known_uuids:
            raise RuntimeError(f"Foreign compute PID references unknown GPU UUID: {row}")
        pid = nonnegative_int(row["pid"], f"compute PID for GPU {uuid}")
        if uuid in selected_pids:
            selected_pids[uuid].append(pid)
    selected = []
    selected_current_draw = 0.0
    for rank, uuid in enumerate(selected_uuids):
        row = by_uuid[uuid]
        index = nonnegative_int(row["index"], f"GPU {uuid} index")
        name = row["name"]
        power_limit = finite_nonnegative_float(
            row["power_limit"], f"GPU {uuid} power limit"
        )
        draw = finite_nonnegative_float(row["power_draw"], f"GPU {uuid} power draw")
        memory_used = nonnegative_int(row["memory_used"], f"GPU {uuid} memory usage")
        if name != "NVIDIA GeForce RTX 5090":
            raise RuntimeError(f"DDP3 selected GPU is not RTX 5090: {row}")
        if index in PROTECTED_PHYSICAL_INDICES:
            raise RuntimeError(f"DDP3 selected protected physical GPU index: {index}")
        if power_limit != POWER_LIMIT_WATTS:
            raise RuntimeError(
                f"DDP3 selected GPU {uuid} power limit is {power_limit}, expected 400 W"
            )
        if selected_pids[uuid]:
            raise RuntimeError(
                f"DDP3 selected GPU {uuid} has foreign compute PIDs: {selected_pids[uuid]}"
            )
        selected_current_draw += draw
        selected.append(
            {
                "rank": rank,
                "physical_index": index,
                "uuid": uuid,
                "name": name,
                "power_limit_watts": power_limit,
                "power_draw_watts_before": draw,
                "memory_used_mib_before": memory_used,
                "compute_pids_before": [],
            }
        )
    projected = (
        aggregate_draw
        - selected_current_draw
        + sum(row["power_limit_watts"] for row in selected)
    )
    if projected >= AGGREGATE_POWER_LIMIT_WATTS:
        raise RuntimeError(
            f"DDP3 projected aggregate power is {projected:.3f} W; policy requires < 2000 W"
        )
    inventory_uuids = [
        uuid
        for uuid, row in sorted(
            by_uuid.items(),
            key=lambda item: nonnegative_int(item[1]["index"], f"GPU {item[0]} index"),
        )
    ]
    return {
        "ordered_selected": selected,
        "authorized_inventory_uuids": inventory_uuids,
        "aggregate_power_draw_watts_before": aggregate_draw,
        "selected_power_draw_watts_before": selected_current_draw,
        "projected_aggregate_power_watts_at_selected_limits": projected,
        "projected_formula": (
            "current aggregate - current draws of all selected cards "
            "+ sum of selected card power limits"
        ),
    }


def controller_gpu_preflight() -> dict[str, object]:
    return evaluate_controller_gpu_rows(query_gpu_inventory(), query_compute_apps())


def evaluate_power_rows(
    power_rows: list[dict[str, str]],
    authorized_inventory_uuids: tuple[str, ...],
    selected_uuids: tuple[str, ...] = ORDERED_UUIDS,
) -> dict[str, object]:
    if len(set(authorized_inventory_uuids)) != len(authorized_inventory_uuids):
        raise RuntimeError("Authorized GPU inventory contains duplicate UUIDs")
    if not set(selected_uuids).issubset(authorized_inventory_uuids):
        raise RuntimeError("Authorized GPU inventory omits a selected UUID")
    by_uuid: dict[str, float] = {}
    for row in power_rows:
        uuid = row["uuid"]
        if uuid in by_uuid:
            raise RuntimeError(f"Duplicate UUID in power sample: {uuid}")
        by_uuid[uuid] = finite_nonnegative_float(
            row["power_draw"], f"GPU {uuid} watchdog power draw"
        )
    measured_inventory = set(by_uuid)
    authorized_inventory = set(authorized_inventory_uuids)
    if measured_inventory != authorized_inventory:
        raise RuntimeError(
            "Power sample inventory mismatch: "
            f"missing={sorted(authorized_inventory - measured_inventory)}, "
            f"foreign={sorted(measured_inventory - authorized_inventory)}"
        )
    selected = {uuid: by_uuid[uuid] for uuid in selected_uuids}
    aggregate = sum(by_uuid.values())
    violations = [
        {
            "kind": "selected_power",
            "uuid": uuid,
            "measured_watts": draw,
            "threshold_watts": POWER_LIMIT_WATTS,
            "comparison": ">",
        }
        for uuid, draw in selected.items()
        if draw > POWER_LIMIT_WATTS
    ]
    if aggregate >= AGGREGATE_POWER_LIMIT_WATTS:
        violations.append(
            {
                "kind": "aggregate_power",
                "measured_watts": aggregate,
                "threshold_watts": AGGREGATE_POWER_LIMIT_WATTS,
                "comparison": ">=",
            }
        )
    return {
        "sampled_central": central_now().isoformat(),
        "inventory_uuids": list(authorized_inventory_uuids),
        "selected_power_draw_watts": selected,
        "aggregate_power_draw_watts": aggregate,
        "violations": violations,
    }


def query_power_sample(
    authorized_inventory_uuids: tuple[str, ...],
) -> dict[str, object]:
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=GPU_QUERY_TIMEOUT_SECONDS,
    )
    return evaluate_power_rows(
        parse_csv_rows(text, ("uuid", "power_draw")),
        authorized_inventory_uuids,
    )


def request_coordinated_stop(
    claim_path: Path,
    reason: str,
    details: dict[str, object],
) -> Path:
    path = claim_path / "stop_request.json"
    payload = {
        "status": "stop_requested",
        "requested_central": central_now().isoformat(),
        "reason": reason,
        "details": details,
    }
    try:
        write_json_exclusive(path, payload)
    except FileExistsError:
        recorded = load_json_object(path.read_bytes(), str(path))
        if recorded.get("status") != "stop_requested":
            raise RuntimeError(f"Existing DDP3 stop request is invalid: {recorded}")
    return path


def watchdog_loop(
    claim_path: Path,
    stop_event: threading.Event,
    violation_event: threading.Event,
    authorized_inventory_uuids: tuple[str, ...],
    interval_seconds: float = WATCHDOG_INTERVAL_SECONDS,
) -> None:
    log_path = claim_path / "power_watchdog.jsonl"
    while not stop_event.is_set():
        try:
            sample = query_power_sample(authorized_inventory_uuids)
            append_jsonl_fsync(log_path, sample)
            violations = sample["violations"]
            if violations:
                violation_event.set()
                receipt = {
                    "status": "power_violation",
                    "detected_central": central_now().isoformat(),
                    "sample": sample,
                }
                try:
                    write_json_exclusive(claim_path / "power_violation.json", receipt)
                except FileExistsError:
                    pass
                request_coordinated_stop(claim_path, "power_policy_violation", receipt)
                return
        except BaseException as error:
            violation_event.set()
            receipt = {
                "status": "watchdog_failure",
                "detected_central": central_now().isoformat(),
                "error": f"{type(error).__name__}: {error}",
            }
            try:
                write_json_exclusive(claim_path / "power_violation.json", receipt)
            except FileExistsError:
                pass
            except BaseException:
                pass
            try:
                request_coordinated_stop(claim_path, "power_watchdog_failure", receipt)
            except BaseException:
                pass
            return
        stop_event.wait(interval_seconds)


def config_diff(
    left: object,
    right: object,
    path: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    if isinstance(left, dict) and isinstance(right, dict):
        rows = []
        for key in sorted(set(left) | set(right)):
            child = path + (str(key),)
            if key not in left:
                rows.append({"path": ".".join(child), "strict": None, "quality": right[key]})
            elif key not in right:
                rows.append({"path": ".".join(child), "strict": left[key], "quality": None})
            else:
                rows.extend(config_diff(left[key], right[key], child))
        return rows
    return [] if left == right else [{"path": ".".join(path), "strict": left, "quality": right}]


def expected_config_deltas(config: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"path": "ddp", "strict": None, "quality": config["ddp"]},
        {
            "path": "experiment.name",
            "strict": "stage_b_corrected_sparse_nwd_e20",
            "quality": "ic_quality_aligned_ddp3_seed42_e20",
        },
        {"path": "loss.dynamic_k_topq", "strict": None, "quality": 10},
        {"path": "loss.quality_bootstrap_epochs", "strict": None, "quality": 2},
        {"path": "loss.task_aligned_alpha", "strict": None, "quality": 1.0},
        {"path": "loss.task_aligned_beta", "strict": None, "quality": 6.0},
        {"path": "loss.task_aligned_enabled", "strict": None, "quality": True},
        {"path": "training.gradient_accumulation_steps", "strict": 4, "quality": 1},
        {"path": "training.nan_grad_action", "strict": "sanitize", "quality": "skip"},
        {"path": "training.num_workers", "strict": 16, "quality": 4},
    ]


def fingerprint_label_split(split: Path) -> dict[str, object]:
    digest = hashlib.sha256(b"sparsevoxeldet-label-split-v1\0")
    files = total_bytes = boxes = two_box_files = 0
    for entry in sorted(os.scandir(split), key=lambda item: item.name):
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


def audit_label_splits(label_root: Path) -> dict[str, dict[str, object]]:
    measured = {
        name: fingerprint_label_split(label_root / name)
        for name in EXPECTED_LABEL_SPLITS
    }
    if measured != EXPECTED_LABEL_SPLITS:
        raise RuntimeError(
            f"Corrected-label split content mismatch: {measured} != {EXPECTED_LABEL_SPLITS}"
        )
    return measured


def derive_preflight_stable(
    source_config: dict[str, object],
    source_hashes: dict[str, dict[str, dict[str, str]]],
) -> dict[str, object]:
    strict_hash = sha256(STRICT_CONFIG)
    expected_strict_hash = "f8a2a9b95d2556ebf71388c424b0ec59aa5066b06aec3e9f507192e414fdf5b8"
    if strict_hash != expected_strict_hash:
        raise RuntimeError(f"Strict config hash mismatch: {strict_hash}")
    strict_config = yaml.safe_load(STRICT_CONFIG.read_bytes())
    differences = config_diff(strict_config, source_config)
    if differences != expected_config_deltas(source_config):
        raise RuntimeError(f"DDP3 config has unauthorized differences: {differences}")
    sealed_token = "canonical_" + "test"
    if sealed_token in json.dumps(source_config, sort_keys=True).lower():
        raise RuntimeError("DDP3 config references the sealed test split")
    label_root = LABEL_ROOT.resolve()
    if label_root.name != "labels_rawcomplete":
        raise RuntimeError(f"Corrected label root does not resolve exactly: {label_root}")
    manifest_sha = sha256(label_root / "MANIFEST.md")
    if manifest_sha != EXPECTED_LABEL_MANIFEST_SHA:
        raise RuntimeError(f"Corrected-label MANIFEST hash mismatch: {manifest_sha}")
    label_splits = audit_label_splits(label_root)
    runtime_contract = expected_runtime_contract(source_hashes)
    return {
        "status": "pass",
        "execution_mode": "record",
        "quality_config_references_sealed_test": False,
        "sealed_test_access_audit_performed": False,
        "sealed_test_access_result": "not_measured",
        "output_collision": False,
        "output_dir": str(OUTPUT_DIR.resolve()),
        "source_hashes": source_hashes,
        "config_match": {
            "strict_config": str(STRICT_CONFIG.resolve()),
            "strict_config_sha256": strict_hash,
            "quality_config": str(CONFIG.resolve()),
            "quality_config_sha256": EXPECTED_BUILD[CONFIG],
            "differences": differences,
            "all_other_fields_equal": True,
        },
        "label_root": str(label_root),
        "label_manifest_sha256": manifest_sha,
        "label_splits": label_splits,
        "canonical_val": {
            key: label_splits["canonical_val"][key]
            for key in ("files", "boxes", "two_box_files")
        },
        "model_inventory": {
            "ic": {"inverse_conv3d": 2, "transpose_conv3d": 0},
            "per_sample_se_source_sha256": EXPECTED_REMOTE[
                INVESTIGATORS / "se_per_sample_patch.py"
            ],
        },
        "ddp_policy": {
            key: runtime_contract[key]
            for key in (
                "runtime_policy_version", "ordered_physical_uuids", "forbidden_physical_uuids",
                "world_size", "local_world_size", "rank_mapping_policy", "per_rank_batch_size",
                "global_effective_batch_size", "gradient_accumulation_steps",
                "sampler", "train_roster_samples", "sampler_samples_per_epoch", "optimized_samples_per_epoch",
                "optimizer_steps_per_epoch", "warmup_optimizer_steps", "epochs",
                "total_optimizer_steps", "scheduler", "validation",
            )
        },
        "remote_only_checks_pending": False,
    }


def verify_preflight_evidence(
    receipt_bytes: bytes,
    replay_bytes: bytes,
    expected_stable: dict[str, object],
    receipt_path: Path,
    replay_path: Path,
) -> tuple[dict[str, object], str, str]:
    receipt = load_json_object(receipt_bytes, str(receipt_path))
    replay = load_json_object(replay_bytes, str(replay_path))
    expected_receipt_keys = set(expected_stable) | {"completed_central"}
    if set(receipt) != expected_receipt_keys:
        raise RuntimeError(
            f"DDP3 preflight fields changed: {sorted(set(receipt) ^ expected_receipt_keys)}"
        )
    for key, expected in expected_stable.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"DDP3 preflight field drifted: {key}")
    if receipt.get("remote_only_checks_pending") is not False:
        raise RuntimeError("DDP3 full remote preflight remains pending")
    if not isinstance(receipt.get("completed_central"), str) or not receipt["completed_central"]:
        raise RuntimeError("DDP3 preflight timestamp is missing")
    expected_replay_keys = {
        "status", "execution_mode", "recorded_preflight_path",
        "recorded_preflight_sha256", "recorded_preflight_verified",
        "verified_field_names", "verified_at_central",
    }
    if set(replay) != expected_replay_keys:
        raise RuntimeError(
            f"DDP3 replay fields changed: {sorted(set(replay) ^ expected_replay_keys)}"
        )
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    if replay.get("status") != "pass" or replay.get("execution_mode") != "check_only_replay":
        raise RuntimeError("DDP3 preflight replay is missing or failed")
    if Path(str(replay.get("recorded_preflight_path", ""))).resolve() != receipt_path.resolve():
        raise RuntimeError("DDP3 preflight replay targets the wrong receipt")
    if (
        replay.get("recorded_preflight_sha256") != receipt_sha
        or replay.get("recorded_preflight_verified") is not True
    ):
        raise RuntimeError("DDP3 preflight replay is not bound to exact receipt bytes")
    names = replay.get("verified_field_names")
    if (
        not isinstance(names, list)
        or not all(isinstance(n, str) for n in names)
        or sorted(names) != sorted(expected_stable)
    ):
        raise RuntimeError("DDP3 replay did not verify every stable field")
    if not isinstance(replay.get("verified_at_central"), str) or not replay["verified_at_central"]:
        raise RuntimeError("DDP3 replay timestamp is missing")
    return receipt, receipt_sha, hashlib.sha256(replay_bytes).hexdigest()


def load_preflight(
    source_config: dict[str, object],
    source_hashes: dict[str, dict[str, dict[str, str]]],
) -> tuple[dict[str, object], str, str, dict[str, object]]:
    receipt_bytes = PREFLIGHT_RECEIPT.read_bytes()
    replay_bytes = PREFLIGHT_REPLAY.read_bytes()
    stable = derive_preflight_stable(source_config, source_hashes)
    receipt, receipt_sha, replay_sha = verify_preflight_evidence(
        receipt_bytes,
        replay_bytes,
        stable,
        PREFLIGHT_RECEIPT,
        PREFLIGHT_REPLAY,
    )
    return receipt, receipt_sha, replay_sha, stable


def validate_ddp_provenance(provenance: object) -> dict[str, object]:
    if not isinstance(provenance, dict):
        raise RuntimeError("DDP3 checkpoint provenance is missing")
    required = {
        "runtime_policy_version", "controller_identity", "authorization_sha256",
        "ordered_physical_uuids", "world_size", "rank_mapping",
        "selected_compute_occupancy", "worker_registrations", "token_consumed_sha256",
    }
    if set(provenance) != required:
        raise RuntimeError(
            f"DDP3 provenance fields changed: {sorted(set(provenance) ^ required)}"
        )
    if provenance["runtime_policy_version"] != RUNTIME_POLICY_VERSION:
        raise RuntimeError("DDP3 provenance runtime policy mismatch")
    if provenance["ordered_physical_uuids"] != list(ORDERED_UUIDS):
        raise RuntimeError("DDP3 provenance ordered UUID topology mismatch")
    if provenance["world_size"] != WORLD_SIZE:
        raise RuntimeError("DDP3 provenance world size mismatch")
    controller = provenance["controller_identity"]
    expected_controller_keys = {
        "controller_id", "hostname", "pid", "boot_id",
        "process_start_ticks", "reservation_id",
    }
    if not isinstance(controller, dict) or set(controller) != expected_controller_keys:
        raise RuntimeError("DDP3 controller identity is malformed")
    if not all(controller.get(key) for key in ("controller_id", "hostname", "boot_id", "reservation_id")):
        raise RuntimeError("DDP3 controller identity is incomplete")
    for key in ("pid", "process_start_ticks"):
        if isinstance(controller.get(key), bool) or not isinstance(controller.get(key), int) or controller[key] <= 0:
            raise RuntimeError(f"DDP3 controller identity has invalid {key}")
    if not isinstance(provenance["authorization_sha256"], str) or len(
        provenance["authorization_sha256"]
    ) != 64:
        raise RuntimeError("DDP3 authorization digest is invalid")
    mapping = provenance["rank_mapping"]
    registrations = provenance["worker_registrations"]
    if not isinstance(mapping, list) or len(mapping) != WORLD_SIZE:
        raise RuntimeError("DDP3 rank mapping must contain exactly three rows")
    if not isinstance(registrations, list) or len(registrations) != WORLD_SIZE:
        raise RuntimeError("DDP3 worker registrations must contain exactly three rows")
    seen_pids: set[int] = set()
    seen_indices: set[int] = set()
    for rank, row in enumerate(mapping):
        expected_keys = {
            "rank", "local_rank", "pid", "local_device",
            "physical_index", "uuid", "name",
        }
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise RuntimeError(f"DDP3 rank mapping row {rank} is malformed")
        if (
            row["rank"] != rank
            or row["local_rank"] != rank
            or row["local_device"] != rank
            or row["uuid"] != ORDERED_UUIDS[rank]
            or row["name"] != "NVIDIA GeForce RTX 5090"
        ):
            raise RuntimeError(f"DDP3 rank mapping row {rank} violates topology: {row}")
        pid = int(row["pid"])
        physical_index = int(row["physical_index"])
        if pid in seen_pids or physical_index in seen_indices:
            raise RuntimeError("DDP3 rank mapping repeats PID or physical index")
        if physical_index in PROTECTED_PHYSICAL_INDICES:
            raise RuntimeError("DDP3 rank mapping uses a protected physical index")
        seen_pids.add(pid)
        seen_indices.add(physical_index)
    occupancy = provenance["selected_compute_occupancy"]
    expected_occupancy = [
        {"uuid": ORDERED_UUIDS[rank], "compute_pids": [mapping[rank]["pid"]]}
        for rank in range(WORLD_SIZE)
    ]
    if occupancy != expected_occupancy:
        raise RuntimeError("DDP3 selected-card occupancy closure is malformed")
    for rank, item in enumerate(registrations):
        if not isinstance(item, dict) or set(item) != {"record", "sha256"}:
            raise RuntimeError(f"DDP3 registration wrapper {rank} is malformed")
        record = item["record"]
        expected_record_keys = {
            "status", "registered_central", "rank", "local_rank", "world_size",
            "expected_uuid", "pid", "hostname", "boot_id", "process_start_ticks",
            "controller_id", "reservation_id", "output_dir", "authorization_sha256",
            "source_hashes_sha256", "config_sha256",
        }
        if not isinstance(record, dict) or set(record) != expected_record_keys:
            raise RuntimeError(f"DDP3 registration record {rank} is malformed")
        if (
            record.get("status") != "registered"
            or record.get("rank") != rank
            or record.get("local_rank") != rank
            or record.get("world_size") != WORLD_SIZE
            or record.get("expected_uuid") != ORDERED_UUIDS[rank]
            or record.get("pid") != mapping[rank]["pid"]
            or record.get("hostname") != controller["hostname"]
            or record.get("boot_id") != controller["boot_id"]
            or record.get("controller_id") != controller["controller_id"]
            or record.get("reservation_id") != controller["reservation_id"]
            or record.get("output_dir") != str(OUTPUT_DIR.resolve())
            or record.get("authorization_sha256") != provenance["authorization_sha256"]
            or record.get("config_sha256") != EXPECTED_BUILD[CONFIG]
        ):
            raise RuntimeError(f"DDP3 registration record {rank} violates topology")
        if (
            isinstance(record.get("process_start_ticks"), bool)
            or not isinstance(record.get("process_start_ticks"), int)
            or record["process_start_ticks"] <= 0
            or not isinstance(record.get("source_hashes_sha256"), str)
            or len(record["source_hashes_sha256"]) != 64
        ):
            raise RuntimeError(f"DDP3 registration record {rank} has invalid process/source identity")
        encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
        if item["sha256"] != hashlib.sha256(encoded).hexdigest():
            raise RuntimeError(f"DDP3 registration record {rank} digest mismatch")
    if len({item["record"]["source_hashes_sha256"] for item in registrations}) != 1:
        raise RuntimeError("DDP3 worker registrations disagree on authorized source bytes")
    if not isinstance(provenance["token_consumed_sha256"], str) or len(
        provenance["token_consumed_sha256"]
    ) != 64:
        raise RuntimeError("DDP3 token-consumption digest is invalid")
    return copy.deepcopy(provenance)


def expected_checkpoint_config(
    source_config: dict[str, object],
    runtime_contract: dict[str, object],
) -> dict[str, object]:
    expected = copy.deepcopy(source_config)
    experiment = expected.setdefault("experiment", {})
    if not isinstance(experiment, dict):
        raise RuntimeError("DDP3 experiment config is malformed")
    experiment["runtime_contract"] = copy.deepcopy(runtime_contract)
    return expected


def verify_checkpoint_dict(
    checkpoint: dict[str, object],
    source_config: dict[str, object],
    runtime_contract: dict[str, object],
) -> dict[str, object]:
    required = {
        "epoch", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "scaler_state_dict", "ema_state_dict", "metrics", "config",
        "checkpoint_lineage", "runtime_contract", "launch_origin",
        "optimizer_boundary",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(f"DDP3 resume checkpoint is incomplete: missing {missing}")
    if checkpoint["checkpoint_lineage"] != CHECKPOINT_LINEAGE:
        raise RuntimeError("Checkpoint is not from the DDP3 quality lineage")
    if checkpoint["runtime_contract"] != runtime_contract:
        raise RuntimeError("DDP3 checkpoint runtime contract mismatch")
    epoch = checkpoint["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch < EPOCHS:
        raise RuntimeError(f"DDP3 resume epoch is invalid: {epoch!r}")
    batch_idx = checkpoint.get("batch_idx")
    if batch_idx is not None and (
        isinstance(batch_idx, bool)
        or not isinstance(batch_idx, int)
        or batch_idx < -1
        or batch_idx >= OPTIMIZER_STEPS_PER_EPOCH
    ):
        raise RuntimeError(f"DDP3 resume batch_idx is invalid: {batch_idx!r}")
    boundary = checkpoint["optimizer_boundary"]
    if not isinstance(boundary, dict) or set(boundary) != {
        "micro_batches_in_window", "optimizer_steps_in_epoch", "optimizer_steps_completed",
        "sampler_epoch", "next_batch_idx", "next_rank_local_sample_offset",
    }:
        raise RuntimeError("DDP3 checkpoint optimizer-boundary proof is malformed")
    if boundary["micro_batches_in_window"] != 0:
        raise RuntimeError("DDP3 checkpoint was not committed at an optimizer boundary")
    if (
        isinstance(boundary["optimizer_steps_in_epoch"], bool)
        or not isinstance(boundary["optimizer_steps_in_epoch"], int)
        or not 0 <= boundary["optimizer_steps_in_epoch"] <= OPTIMIZER_STEPS_PER_EPOCH
    ):
        raise RuntimeError("DDP3 checkpoint epoch optimizer step count is invalid")
    if (
        isinstance(boundary["optimizer_steps_completed"], bool)
        or not isinstance(boundary["optimizer_steps_completed"], int)
        or not boundary["optimizer_steps_in_epoch"]
        <= boundary["optimizer_steps_completed"]
        <= TOTAL_OPTIMIZER_STEPS
    ):
        raise RuntimeError("DDP3 checkpoint cumulative optimizer step count is invalid")
    if boundary["sampler_epoch"] != epoch:
        raise RuntimeError("DDP3 checkpoint sampler epoch differs from checkpoint epoch")
    next_batch = boundary["next_batch_idx"]
    if next_batch is not None and (
        isinstance(next_batch, bool)
        or not isinstance(next_batch, int)
        or not 0 <= next_batch <= OPTIMIZER_STEPS_PER_EPOCH
    ):
        raise RuntimeError("DDP3 checkpoint next batch index is invalid")
    expected_offset = (
        OPTIMIZED_SAMPLES // WORLD_SIZE
        if next_batch is None
        else next_batch * PER_RANK_BATCH
    )
    if boundary["next_rank_local_sample_offset"] != expected_offset:
        raise RuntimeError("DDP3 checkpoint rank-local sample offset is invalid")
    if next_batch is not None and boundary["optimizer_steps_in_epoch"] > next_batch:
        raise RuntimeError("DDP3 checkpoint optimizer steps exceed consumed batches")
    if batch_idx is None and next_batch is not None:
        raise RuntimeError("DDP3 full-epoch checkpoint cannot carry a next batch index")
    if batch_idx is not None and next_batch != batch_idx + 1:
        raise RuntimeError("DDP3 mid-epoch checkpoint batch lineage is inconsistent")
    launch_origin = validate_ddp_provenance(checkpoint["launch_origin"])
    checkpoint_config = checkpoint["config"]
    if not isinstance(checkpoint_config, dict):
        raise RuntimeError("DDP3 checkpoint config is malformed")
    expected_config = expected_checkpoint_config(source_config, runtime_contract)
    expected_keys = set(expected_config) | {"_runtime"}
    if set(checkpoint_config) != expected_keys:
        raise RuntimeError(
            f"DDP3 checkpoint config sections changed: "
            f"{sorted(set(checkpoint_config) ^ expected_keys)}"
        )
    for key, expected in expected_config.items():
        if checkpoint_config.get(key) != expected:
            raise RuntimeError(f"DDP3 checkpoint provenance mismatch in config section: {key}")
    runtime = checkpoint_config.get("_runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("DDP3 checkpoint runtime config is missing")
    fixed_runtime = {
        "strict_finite_checks": True,
        "finite_trace_dir": str(OUTPUT_DIR / "finite_trace"),
        "abort_on_skip_rate": 0.10,
        "abort_on_consecutive_nonfinite": 200,
        "is_main_process": True,
        "amp_mode": "fp16",
        "max_train_batches": None,
        "rank0_writer": True,
        "launch_origin": launch_origin,
    }
    dynamic = {"best_map", "best_map_50_95", "best_epoch", "best_loss"}
    if set(runtime) - set(fixed_runtime) - dynamic:
        raise RuntimeError(
            f"DDP3 checkpoint runtime has unauthorized fields: "
            f"{sorted(set(runtime) - set(fixed_runtime) - dynamic)}"
        )
    for key, expected in fixed_runtime.items():
        if runtime.get(key) != expected:
            raise RuntimeError(f"DDP3 checkpoint runtime mismatch for {key}")
    if checkpoint["launch_origin"] != runtime["launch_origin"]:
        raise RuntimeError("DDP3 launch origin differs between checkpoint and runtime config")
    return {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "runtime_contract": runtime_contract,
        "launch_origin": launch_origin,
        "optimizer_boundary": copy.deepcopy(boundary),
    }


def validate_archived_provenance(
    claim_path: Path,
    provenance: dict[str, object],
) -> None:
    owner_path = claim_path / "owner.json"
    auth_path = claim_path / "authorization.json"
    token_path = claim_path / "token_consumed.json"
    if any(path.is_symlink() or not path.is_file() for path in (owner_path, auth_path, token_path)):
        raise RuntimeError(f"Archived DDP3 claim lacks provenance records: {claim_path}")
    owner = load_json_object(owner_path.read_bytes(), str(owner_path))
    controller = provenance["controller_identity"]
    for key in ("controller_id", "hostname", "pid", "boot_id", "process_start_ticks", "reservation_id"):
        if owner.get(key) != controller.get(key):
            raise RuntimeError(f"Archived DDP3 controller identity mismatch for {key}")
    if sha256(auth_path) != provenance["authorization_sha256"]:
        raise RuntimeError("Archived DDP3 authorization digest mismatch")
    if sha256(token_path) != provenance["token_consumed_sha256"]:
        raise RuntimeError("Archived DDP3 token-consumption digest mismatch")
    for rank, item in enumerate(provenance["worker_registrations"]):
        path = claim_path / "workers" / f"rank_{rank}.json"
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Archived DDP3 registration is missing for rank {rank}")
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Archived DDP3 registration digest mismatch for rank {rank}")
        if load_json_object(source, str(path)) != item["record"]:
            raise RuntimeError(f"Archived DDP3 registration content mismatch for rank {rank}")


def record_checkpoint_commit(
    claim_path: Path,
    checkpoint_path: Path,
    _checkpoint: dict[str, object],
    source_config: dict[str, object],
    runtime_contract: dict[str, object],
) -> Path:
    import torch

    if checkpoint_path.parent.resolve() != OUTPUT_DIR.resolve():
        raise RuntimeError(f"DDP3 checkpoint path is outside the owned output: {checkpoint_path}")
    with checkpoint_path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        digest = sha256_open_handle(handle)
        persisted = torch.load(handle, map_location="cpu", weights_only=False)
        after = os.fstat(handle.fileno())
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if identity(before) != identity(after):
        raise RuntimeError(f"DDP3 checkpoint changed during commit: {checkpoint_path}")
    if not isinstance(persisted, dict):
        raise RuntimeError("Persisted DDP3 checkpoint is not a dictionary")
    verified = verify_checkpoint_dict(persisted, source_config, runtime_contract)
    validate_archived_provenance(claim_path, verified["launch_origin"])
    records_dir = claim_path / "checkpoints"
    records_dir.mkdir(exist_ok=True)
    record_path = records_dir / f"{digest}_{checkpoint_path.name}.json"
    payload = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_sha256": digest,
        "checkpoint_lineage": CHECKPOINT_LINEAGE,
        "epoch": verified["epoch"],
        "batch_idx": verified["batch_idx"],
        "optimizer_boundary": verified["optimizer_boundary"],
        "runtime_contract": runtime_contract,
        "launch_origin": verified["launch_origin"],
        "recorded_central": central_now().isoformat(),
    }
    if record_path.exists():
        recorded = load_json_object(record_path.read_bytes(), str(record_path))
        for key, value in payload.items():
            if key != "recorded_central" and recorded.get(key) != value:
                raise RuntimeError(f"DDP3 checkpoint commit record drifted: {record_path}")
        return record_path
    write_json_exclusive(record_path, payload)
    return record_path


def find_checkpoint_provenance(
    requested_path: Path,
    digest: str,
    runtime_contract: dict[str, object],
) -> tuple[Path, dict[str, object], Path]:
    archive_root = CONTRACTS_DIR / "archive"
    if not archive_root.is_dir():
        raise RuntimeError(f"No archived DDP3 controller claim exists for checkpoint {digest}")
    for claim_path in sorted(archive_root.iterdir()):
        if claim_path.is_symlink() or not claim_path.is_dir():
            continue
        records_dir = claim_path / "checkpoints"
        contract_path = claim_path / "contract.json"
        owner_path = claim_path / "owner.json"
        if not records_dir.is_dir() or not contract_path.is_file() or not owner_path.is_file():
            continue
        contract = load_json_object(contract_path.read_bytes(), str(contract_path))
        owner = load_json_object(owner_path.read_bytes(), str(owner_path))
        if (
            contract.get("runtime_contract") != runtime_contract
            or owner.get("output_dir") != str(OUTPUT_DIR.resolve())
        ):
            continue
        for record_path in sorted(records_dir.glob(f"{digest}_*.json")):
            if record_path.is_symlink():
                continue
            record = load_json_object(record_path.read_bytes(), str(record_path))
            if (
                record.get("checkpoint_path") == str(requested_path.resolve())
                and record.get("checkpoint_name") == requested_path.name
                and record.get("checkpoint_sha256") == digest
                and record.get("checkpoint_lineage") == CHECKPOINT_LINEAGE
                and record.get("runtime_contract") == runtime_contract
            ):
                launch_origin = validate_ddp_provenance(record.get("launch_origin"))
                validate_archived_provenance(claim_path, launch_origin)
                return record_path, record, claim_path
    raise RuntimeError(
        f"Checkpoint {digest} is absent from archived matching DDP3 commit records"
    )


def validate_resume_path(path: Path) -> None:
    if path.parent.resolve() != OUTPUT_DIR.resolve():
        raise RuntimeError(f"Resume checkpoint is outside the DDP3 output: {path}")
    permitted = (
        path.name in {"latest.pt", "best.pt", "emergency_stop.pt"}
        or (path.name.startswith("epoch_") and path.suffix == ".pt")
    )
    if not permitted:
        raise RuntimeError(f"Unauthorized DDP3 resume checkpoint name: {path.name}")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"DDP3 resume checkpoint is missing or a symlink: {path}")


def controller_resume_reference(
    path: Path,
    runtime_contract: dict[str, object],
) -> dict[str, object]:
    validate_resume_path(path)
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        digest = sha256_open_handle(handle)
        after = os.fstat(handle.fileno())
    if stat_identity(before) != stat_identity(after):
        raise RuntimeError(f"DDP3 resume checkpoint changed during authorization: {path}")
    record_path, record, claim_path = find_checkpoint_provenance(
        path,
        digest,
        runtime_contract,
    )
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "file_identity": stat_identity(before),
        "commit_record": str(record_path.resolve()),
        "archived_claim": str(claim_path.resolve()),
        "epoch": record["epoch"],
        "batch_idx": record["batch_idx"],
    }


def verify_resume(
    path: Path,
    source_config: dict[str, object],
    runtime_contract: dict[str, object],
    authorized_reference: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    import torch

    validate_resume_path(path)
    if authorized_reference.get("path") != str(path.resolve()):
        raise RuntimeError("DDP3 worker resume path differs from controller authorization")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        digest = sha256_open_handle(handle)
        record_path, record, claim_path = find_checkpoint_provenance(
            path,
            digest,
            runtime_contract,
        )
        checkpoint = torch.load(handle, map_location="cpu", weights_only=False)
        after = os.fstat(handle.fileno())
    if stat_identity(before) != stat_identity(after):
        raise RuntimeError(f"DDP3 resume checkpoint changed during verification: {path}")
    expected_reference = {
        "path": str(path.resolve()),
        "sha256": digest,
        "file_identity": stat_identity(before),
        "commit_record": str(record_path.resolve()),
        "archived_claim": str(claim_path.resolve()),
        "epoch": record["epoch"],
        "batch_idx": record["batch_idx"],
    }
    if authorized_reference != expected_reference:
        raise RuntimeError("DDP3 resume consumption differs from controller authorization")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("DDP3 resume checkpoint payload is not a dictionary")
    verified = verify_checkpoint_dict(checkpoint, source_config, runtime_contract)
    for key in ("epoch", "batch_idx", "optimizer_boundary", "launch_origin"):
        if record.get(key) != verified[key]:
            raise RuntimeError(f"DDP3 checkpoint commit metadata mismatch for {key}")
    receipt = {
        "path": str(path.resolve()),
        "sha256": digest,
        "epoch": verified["epoch"],
        "batch_idx": verified["batch_idx"],
        "optimizer_boundary": verified["optimizer_boundary"],
        "runtime_contract_verified": verified["runtime_contract"],
        "launch_origin_verified": verified["launch_origin"],
        "checkpoint_commit_record": str(record_path.resolve()),
        "archived_claim": str(claim_path.resolve()),
        "checkpoint_commit_metadata_verified": True,
        "controller_authorization_verified": True,
        "file_identity_verified": stat_identity(before),
        "single_load_consumption": True,
    }
    return receipt, checkpoint


def write_controller_authorization(
    claim_path: Path,
    owner: dict[str, object],
    token_sha256: str,
    source_hashes: dict[str, dict[str, dict[str, str]]],
    runtime_contract: dict[str, object],
    preflight_sha256: str,
    replay_sha256: str,
    preflight_stable: dict[str, object],
    gpu_preflight: dict[str, object],
    resume_reference: dict[str, object] | None,
) -> tuple[dict[str, object], str]:
    authorization = {
        "status": "authorized",
        "authorized_central": central_now().isoformat(),
        "runtime_policy_version": RUNTIME_POLICY_VERSION,
        "controller_id": owner["controller_id"],
        "launch_token_sha256": token_sha256,
        "claim_path": str(claim_path.resolve()),
        "output_dir": str(OUTPUT_DIR.resolve()),
        "reservation_id": owner["reservation_id"],
        "mode": owner["mode"],
        "config_path": str(CONFIG.resolve()),
        "config_sha256": EXPECTED_BUILD[CONFIG],
        "source_hashes": source_hashes,
        "runtime_contract": runtime_contract,
        "ordered_physical_uuids": list(ORDERED_UUIDS),
        "world_size": WORLD_SIZE,
        "local_world_size": WORLD_SIZE,
        "preflight_receipt_sha256": preflight_sha256,
        "preflight_replay_sha256": replay_sha256,
        "preflight_stable": preflight_stable,
        "gpu_preflight": gpu_preflight,
        "resume_reference": resume_reference,
    }
    path = claim_path / "authorization.json"
    write_json_exclusive(path, authorization)
    authorization_sha = sha256(path)
    contract = {
        "status": "resuming" if resume_reference else "launching",
        "created_central": central_now().isoformat(),
        "controller_id": owner["controller_id"],
        "output_dir": str(OUTPUT_DIR.resolve()),
        "runtime_contract": runtime_contract,
        "authorization_sha256": authorization_sha,
        "preflight_receipt_sha256": preflight_sha256,
        "preflight_replay_sha256": replay_sha256,
        "resume_reference": resume_reference,
        "checkpoint_selection": (
            "highest corrected-label AP50; tie AP50:95; tie earlier epoch"
        ),
    }
    write_json_exclusive(claim_path / "contract.json", contract)
    return authorization, authorization_sha


def validate_worker_topology_environment(
    environment: dict[str, str] | None = None,
) -> dict[str, object]:
    source = os.environ if environment is None else environment
    if source.get("SPARSEVOXELDET_DDP3_MODE") != "worker":
        raise RuntimeError("Direct DDP3 worker entry is forbidden")
    required = {
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
        "SPARSEVOXELDET_DDP3_CLAIM", "SPARSEVOXELDET_DDP3_TOKEN",
        "SPARSEVOXELDET_DDP3_CONTROLLER_ID",
    }
    missing = sorted(key for key in required if key not in source)
    if missing:
        raise RuntimeError(f"DDP3 worker environment is incomplete: {missing}")
    try:
        rank = int(source["RANK"])
        local_rank = int(source["LOCAL_RANK"])
        world_size = int(source["WORLD_SIZE"])
        local_world_size = int(source["LOCAL_WORLD_SIZE"])
    except ValueError as error:
        raise RuntimeError("DDP3 worker rank environment is non-numeric") from error
    if (
        world_size != WORLD_SIZE
        or local_world_size != WORLD_SIZE
        or rank not in range(WORLD_SIZE)
        or local_rank != rank
    ):
        raise RuntimeError(
            f"DDP3 worker topology mismatch: rank={rank} local_rank={local_rank} "
            f"world={world_size} local_world={local_world_size}"
        )
    if source.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("DDP3 workers require CUDA_DEVICE_ORDER=PCI_BUS_ID")
    expected_mask = ",".join(ORDERED_UUIDS)
    if source.get("CUDA_VISIBLE_DEVICES") != expected_mask:
        raise RuntimeError("DDP3 worker visible-device UUID mask is reordered or foreign")
    if "GROUP_RANK" in source and int(source["GROUP_RANK"]) != 0:
        raise RuntimeError("DDP3 workers require one local node with GROUP_RANK=0")
    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "local_world_size": local_world_size,
        "expected_uuid": ORDERED_UUIDS[rank],
        "claim_path": Path(source["SPARSEVOXELDET_DDP3_CLAIM"]).resolve(),
        "token": source["SPARSEVOXELDET_DDP3_TOKEN"],
        "controller_id": source["SPARSEVOXELDET_DDP3_CONTROLLER_ID"],
    }


def verify_worker_authorization(
    topology: dict[str, object],
    args: argparse.Namespace,
) -> tuple[
    Path,
    dict[str, object],
    dict[str, object],
    dict[str, dict[str, dict[str, str]]],
    dict[str, object],
]:
    validate_identity_args(args)
    claim_path = topology["claim_path"]
    if claim_path != ACTIVE_CLAIM.resolve():
        raise RuntimeError(f"DDP3 worker targets a foreign controller claim: {claim_path}")
    owner_path = claim_path / "owner.json"
    auth_path = claim_path / "authorization.json"
    if any(path.is_symlink() or not path.is_file() for path in (owner_path, auth_path)):
        raise RuntimeError("DDP3 worker controller claim is missing or invalid")
    owner = load_json_object(owner_path.read_bytes(), str(owner_path))
    authorization = load_json_object(auth_path.read_bytes(), str(auth_path))
    token_sha = hashlib.sha256(str(topology["token"]).encode()).hexdigest()
    if claim_owner_alive(owner) is not True:
        raise RuntimeError("DDP3 worker controller claim is not live")
    required_owner = {
        "status": "active",
        "controller_id": topology["controller_id"],
        "launch_token_sha256": token_sha,
        "runtime_policy_version": RUNTIME_POLICY_VERSION,
        "output_dir": str(OUTPUT_DIR.resolve()),
    }
    for key, expected in required_owner.items():
        if owner.get(key) != expected:
            raise RuntimeError(f"DDP3 worker owner mismatch for {key}")
    expected_auth_keys = {
        "status", "authorized_central", "runtime_policy_version", "controller_id",
        "launch_token_sha256", "claim_path", "output_dir", "reservation_id",
        "mode", "config_path", "config_sha256", "source_hashes",
        "runtime_contract", "ordered_physical_uuids", "world_size",
        "local_world_size", "preflight_receipt_sha256",
        "preflight_replay_sha256", "preflight_stable", "gpu_preflight",
        "resume_reference",
    }
    if set(authorization) != expected_auth_keys:
        raise RuntimeError(
            f"DDP3 authorization fields changed: "
            f"{sorted(set(authorization) ^ expected_auth_keys)}"
        )
    required_auth = {
        "status": "authorized",
        "runtime_policy_version": RUNTIME_POLICY_VERSION,
        "controller_id": topology["controller_id"],
        "launch_token_sha256": token_sha,
        "claim_path": str(claim_path),
        "output_dir": str(OUTPUT_DIR.resolve()),
        "reservation_id": owner["reservation_id"],
        "mode": owner["mode"],
        "config_path": str(CONFIG.resolve()),
        "config_sha256": EXPECTED_BUILD[CONFIG],
        "ordered_physical_uuids": list(ORDERED_UUIDS),
        "world_size": WORLD_SIZE,
        "local_world_size": WORLD_SIZE,
    }
    for key, expected in required_auth.items():
        if authorization.get(key) != expected:
            raise RuntimeError(f"DDP3 authorization mismatch for {key}")
    requested_resume = Path(args.resume).resolve() if args.resume is not None else None
    resume_reference = authorization["resume_reference"]
    if requested_resume is None:
        if owner.get("resume_sha256") is not None or resume_reference is not None:
            raise RuntimeError("Fresh DDP3 worker received resume authorization")
    else:
        if not isinstance(resume_reference, dict):
            raise RuntimeError("Resume DDP3 worker lacks controller resume authorization")
        if resume_reference.get("path") != str(requested_resume):
            raise RuntimeError("DDP3 worker resume argument differs from controller path")
        if owner.get("resume_sha256") != resume_reference.get("sha256"):
            raise RuntimeError("DDP3 worker resume digest differs from controller owner")
        if authorization.get("mode") != "resume" or owner.get("mode") != "resume":
            raise RuntimeError("DDP3 resume authorization mode mismatch")
    marker = verify_output_reservation(claim_path, require_pristine=False)
    if marker["reservation_id"] != authorization["reservation_id"]:
        raise RuntimeError("DDP3 authorization targets the wrong output reservation")
    current_hashes = verify_static_sources()
    if current_hashes != authorization["source_hashes"]:
        raise RuntimeError("DDP3 worker source bytes differ from controller-authorized bytes")
    runtime_contract = expected_runtime_contract(current_hashes)
    if runtime_contract != authorization["runtime_contract"]:
        raise RuntimeError("DDP3 worker runtime contract differs from controller contract")
    receipt_bytes = PREFLIGHT_RECEIPT.read_bytes()
    replay_bytes = PREFLIGHT_REPLAY.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != authorization["preflight_receipt_sha256"]:
        raise RuntimeError("DDP3 worker preflight receipt bytes changed")
    if hashlib.sha256(replay_bytes).hexdigest() != authorization["preflight_replay_sha256"]:
        raise RuntimeError("DDP3 worker preflight replay bytes changed")
    verify_preflight_evidence(
        receipt_bytes,
        replay_bytes,
        authorization["preflight_stable"],
        PREFLIGHT_RECEIPT,
        PREFLIGHT_REPLAY,
    )
    source_config = load_verified_yaml(CONFIG, EXPECTED_BUILD[CONFIG])
    validate_protocol_config(source_config)
    return claim_path, owner, authorization, current_hashes, source_config


def register_worker(
    claim_path: Path,
    topology: dict[str, object],
    owner: dict[str, object],
    authorization_sha256: str,
    source_hashes: dict[str, dict[str, dict[str, str]]],
) -> tuple[Path, dict[str, object], str]:
    pid = os.getpid()
    start_ticks = read_process_start_ticks(pid)
    if start_ticks is None:
        raise RuntimeError(f"Cannot prove DDP3 worker process identity for PID {pid}")
    workers_dir = claim_path / "workers"
    workers_dir.mkdir(exist_ok=True)
    source_digest = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    record = {
        "status": "registered",
        "registered_central": central_now().isoformat(),
        "rank": topology["rank"],
        "local_rank": topology["local_rank"],
        "world_size": WORLD_SIZE,
        "expected_uuid": topology["expected_uuid"],
        "pid": pid,
        "hostname": socket.gethostname(),
        "boot_id": read_boot_id(),
        "process_start_ticks": start_ticks,
        "controller_id": owner["controller_id"],
        "reservation_id": owner["reservation_id"],
        "output_dir": str(OUTPUT_DIR.resolve()),
        "authorization_sha256": authorization_sha256,
        "source_hashes_sha256": source_digest,
        "config_sha256": EXPECTED_BUILD[CONFIG],
    }
    path = workers_dir / f"rank_{topology['rank']}.json"
    write_json_exclusive(path, record)
    return path, record, sha256(path)


def collective_consensus_code(local_code: int, device) -> int:
    import torch
    import torch.distributed as dist

    tensor = torch.tensor([int(local_code)], dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def global_reduce_scalars(
    values: dict[str, float],
    operations: dict[str, str],
    device,
) -> dict[str, float]:
    import torch
    import torch.distributed as dist

    if set(values) != set(operations):
        raise RuntimeError("Global reduction values and operation keys differ")
    reduced: dict[str, float] = {}
    op_map = {
        "sum": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }
    for key in values:
        operation = operations[key]
        if operation not in op_map:
            raise RuntimeError(f"Unsupported global reduction operation: {operation}")
        tensor = torch.tensor([float(values[key])], dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=op_map[operation])
        reduced[key] = float(tensor.item())
    return reduced


def broadcast_rank0_payload(payload: object, rank: int, source_rank: int = 0):
    import torch.distributed as dist

    objects = [payload if rank == source_rank else None]
    dist.broadcast_object_list(objects, src=source_rank)
    return objects[0]


def collective_rank0_stage(rank: int, label: str, function):
    payload = None
    if rank == 0:
        try:
            payload = {"status": "success", "value": function(), "error": None}
        except BaseException as error:
            payload = {
                "status": "error",
                "value": None,
                "error": f"{type(error).__name__}: {error}",
            }
    payload = broadcast_rank0_payload(payload, rank)
    if not isinstance(payload, dict) or payload.get("status") != "success":
        detail = payload.get("error") if isinstance(payload, dict) else payload
        raise RuntimeError(f"DDP3 rank-0 stage failed ({label}): {detail}")
    return payload.get("value")


def checkpoint_digest_consensus(local_digest: str) -> str:
    import torch.distributed as dist

    if not isinstance(local_digest, str) or len(local_digest) != 64:
        raise RuntimeError("DDP3 checkpoint digest is malformed")
    gathered: list[object] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_digest)
    if len(gathered) != WORLD_SIZE or set(gathered) != {local_digest}:
        raise RuntimeError(f"DDP3 checkpoint digest consensus failed: {gathered}")
    return local_digest


def gather_rank_errors(local_error: str | None) -> list[str]:
    import torch.distributed as dist

    gathered: list[object] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_error)
    return [f"rank {rank}: {error}" for rank, error in enumerate(gathered) if error]


def abort_distributed_job(message: str) -> None:
    import torch.distributed as dist

    abort = getattr(dist, "abort", None)
    if callable(abort):
        try:
            abort()
        except BaseException:
            pass
    raise RuntimeError(message)


def collective_stop_requested(stop_path: Path, device) -> bool:
    local = int(stop_path.is_file())
    return bool(collective_consensus_code(local, device))


def evaluate_pid_uuid_evidence(
    *,
    rank: int,
    local_rank: int,
    pid: int,
    expected_uuid: str,
    gpu_row: dict[str, str],
    compute_rows: list[dict[str, str]],
) -> dict[str, object]:
    if rank != local_rank or rank not in range(WORLD_SIZE):
        raise RuntimeError("DDP3 post-context rank mapping is invalid")
    if expected_uuid != ORDERED_UUIDS[rank]:
        raise RuntimeError("DDP3 post-context UUID is reordered")
    if gpu_row["uuid"] != expected_uuid:
        raise RuntimeError(f"DDP3 post-context UUID mismatch: {gpu_row}")
    if gpu_row["name"] != "NVIDIA GeForce RTX 5090":
        raise RuntimeError(f"DDP3 post-context model mismatch: {gpu_row}")
    physical_index = int(gpu_row["index"])
    if physical_index in PROTECTED_PHYSICAL_INDICES:
        raise RuntimeError(f"DDP3 worker reached protected physical index {physical_index}")
    pid_rows = [row for row in compute_rows if int(row["pid"]) == pid]
    if not pid_rows:
        raise RuntimeError(f"DDP3 worker PID {pid} is absent from nvidia-smi compute evidence")
    uuids = {row["uuid"] for row in pid_rows}
    if uuids != {expected_uuid}:
        raise RuntimeError(
            f"DDP3 worker PID {pid} maps to unexpected physical UUIDs: {sorted(uuids)}"
        )
    return {
        "rank": rank,
        "local_rank": local_rank,
        "pid": pid,
        "local_device": local_rank,
        "physical_index": physical_index,
        "uuid": expected_uuid,
        "name": gpu_row["name"],
    }


def evaluate_selected_compute_occupancy(
    ordered_mapping: list[dict[str, object]],
    compute_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    if len(ordered_mapping) != WORLD_SIZE:
        raise RuntimeError("DDP3 occupancy closure requires exactly three worker mappings")
    expected = {
        ORDERED_UUIDS[rank]: {nonnegative_int(row["pid"], f"rank {rank} worker PID")}
        for rank, row in enumerate(ordered_mapping)
    }
    observed: dict[str, set[int]] = {uuid: set() for uuid in ORDERED_UUIDS}
    for row in compute_rows:
        uuid = row["uuid"]
        if uuid in observed:
            observed[uuid].add(nonnegative_int(row["pid"], f"compute PID for GPU {uuid}"))
    mismatches = {
        uuid: {"expected": sorted(expected[uuid]), "observed": sorted(observed[uuid])}
        for uuid in ORDERED_UUIDS
        if observed[uuid] != expected[uuid]
    }
    if mismatches:
        raise RuntimeError(f"DDP3 post-context selected-card occupancy mismatch: {mismatches}")
    return [
        {"uuid": uuid, "compute_pids": sorted(observed[uuid])}
        for uuid in ORDERED_UUIDS
    ]


def post_context_gpu_evidence(topology: dict[str, object]) -> dict[str, object]:
    expected_uuid = str(topology["expected_uuid"])
    gpu_text = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            expected_uuid,
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=GPU_QUERY_TIMEOUT_SECONDS,
    )
    gpu_rows = parse_csv_rows(gpu_text, ("index", "uuid", "name"))
    if len(gpu_rows) != 1:
        raise RuntimeError(f"DDP3 expected one selected GPU row, got {gpu_rows}")
    return evaluate_pid_uuid_evidence(
        rank=int(topology["rank"]),
        local_rank=int(topology["local_rank"]),
        pid=os.getpid(),
        expected_uuid=expected_uuid,
        gpu_row=gpu_rows[0],
        compute_rows=query_compute_apps(),
    )


def initialize_worker_process_group(topology: dict[str, object]):
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(
            "DDP3 worker requires exactly three visible CUDA devices: "
            f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
        )
    for local_rank in range(WORLD_SIZE):
        if torch.cuda.get_device_name(local_rank) != "NVIDIA GeForce RTX 5090":
            raise RuntimeError(
                f"DDP3 visible device {local_rank} is not NVIDIA GeForce RTX 5090"
            )
    local_rank = int(topology["local_rank"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    if (
        dist.get_rank() != topology["rank"]
        or dist.get_world_size() != WORLD_SIZE
    ):
        raise RuntimeError("Initialized DDP3 process group violates exact topology")
    device = torch.device(f"cuda:{local_rank}")
    torch.empty(1, device=device).add_(1)
    torch.cuda.synchronize(device)
    return device


def collect_worker_provenance(
    claim_path: Path,
    owner: dict[str, object],
    authorization_sha256: str,
    topology: dict[str, object],
    placement: dict[str, object],
) -> dict[str, object]:
    import torch.distributed as dist

    placements: list[object] = [None] * WORLD_SIZE
    dist.all_gather_object(placements, placement)
    rank = int(topology["rank"])

    def build_launch_origin():
        if claim_owner_alive(owner) is not True:
            raise RuntimeError("DDP3 controller died before worker topology closure")
        ordered_mapping = [dict(item) for item in placements]
        if [row.get("rank") for row in ordered_mapping] != list(range(WORLD_SIZE)):
            raise RuntimeError("DDP3 gathered rank mapping is incomplete or reordered")
        occupancy = evaluate_selected_compute_occupancy(
            ordered_mapping,
            query_compute_apps(),
        )
        workers_dir = claim_path / "workers"
        if workers_dir.is_symlink() or not workers_dir.is_dir():
            raise RuntimeError("DDP3 worker registration directory is missing or invalid")
        entries = list(os.scandir(workers_dir))
        expected_names = {f"rank_{worker_rank}.json" for worker_rank in range(WORLD_SIZE)}
        actual_names = {entry.name for entry in entries}
        if actual_names != expected_names or any(
            not entry.is_file(follow_symlinks=False) for entry in entries
        ):
            raise RuntimeError(
                f"DDP3 worker registration set is incomplete or foreign: {sorted(actual_names)}"
            )
        workers: list[dict[str, object]] = []
        for worker_rank in range(WORLD_SIZE):
            path = workers_dir / f"rank_{worker_rank}.json"
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"DDP3 worker registration missing for rank {worker_rank}")
            source = path.read_bytes()
            record = load_json_object(source, str(path))
            workers.append(
                {"record": record, "sha256": hashlib.sha256(source).hexdigest()}
            )
        rank_map_path = claim_path / "rank_map.json"
        write_json_exclusive(
            rank_map_path,
            {
                "status": "verified",
                "verified_central": central_now().isoformat(),
                "ordered_physical_uuids": list(ORDERED_UUIDS),
                "rank_mapping": ordered_mapping,
                "selected_compute_occupancy": occupancy,
            },
        )
        token_payload = {
            "status": "consumed",
            "consumed_central": central_now().isoformat(),
            "controller_id": owner["controller_id"],
            "launch_token_sha256": owner["launch_token_sha256"],
            "registered_ranks": list(range(WORLD_SIZE)),
            "registration_sha256": [item["sha256"] for item in workers],
        }
        token_path = claim_path / "token_consumed.json"
        write_json_exclusive(token_path, token_payload)
        launch_origin = {
            "runtime_policy_version": RUNTIME_POLICY_VERSION,
            "controller_identity": {
                key: owner[key]
                for key in (
                    "controller_id", "hostname", "pid", "boot_id",
                    "process_start_ticks", "reservation_id",
                )
            },
            "authorization_sha256": authorization_sha256,
            "ordered_physical_uuids": list(ORDERED_UUIDS),
            "world_size": WORLD_SIZE,
            "rank_mapping": ordered_mapping,
            "selected_compute_occupancy": occupancy,
            "worker_registrations": workers,
            "token_consumed_sha256": sha256(token_path),
        }
        return validate_ddp_provenance(launch_origin)

    launch_origin = collective_rank0_stage(
        rank,
        "worker registration and topology closure",
        build_launch_origin,
    )
    return validate_ddp_provenance(launch_origin)


def canonical_trainer_argv(args: argparse.Namespace, local_rank: int) -> list[str]:
    values = [
        "--config", str(CONFIG),
        "--output_dir", str(OUTPUT_DIR),
        "--seed", "42",
        "--device", f"cuda:{local_rank}",
        "--strict-finite-checks",
        "--finite-trace-dir", str(OUTPUT_DIR / "finite_trace"),
        "--abort-on-skip-rate", "0.10",
        "--abort-on-consecutive-nonfinite", "200",
        "--amp-mode", "fp16",
        "--epochs-override", "20",
        "--no-skip-validation",
    ]
    if args.resume is not None:
        values.extend(("--resume", str(Path(args.resume).resolve())))
    return values


def canonical_torchrun_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=3",
        "--max-restarts=0",
        str(Path(__file__).resolve()),
        "--config",
        str(CONFIG),
        "--output_dir",
        str(OUTPUT_DIR),
        "--seed",
        "42",
    ]
    if args.resume is not None:
        command.extend(("--resume", str(Path(args.resume).resolve())))
    return command


def build_worker_environment(
    claim_path: Path,
    token: str,
    controller_id: str,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "SPARSEVOXELDET_DDP3_MODE": "worker",
            "SPARSEVOXELDET_DDP3_CLAIM": str(claim_path.resolve()),
            "SPARSEVOXELDET_DDP3_TOKEN": token,
            "SPARSEVOXELDET_DDP3_CONTROLLER_ID": controller_id,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ",".join(ORDERED_UUIDS),
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "NCCL_BLOCKING_WAIT": "1",
        }
    )
    return environment


def monitor_worker_process(
    process: subprocess.Popen,
    claim_path: Path,
    stop_event: threading.Event,
    violation_event: threading.Event,
) -> int:
    violation_started = None
    while True:
        return_code = process.poll()
        if return_code is not None:
            return int(return_code)
        if violation_event.is_set():
            if violation_started is None:
                violation_started = time.monotonic()
            elapsed = time.monotonic() - violation_started
            if elapsed >= COORDINATED_STOP_GRACE_SECONDS:
                process.terminate()
                try:
                    return int(process.wait(timeout=FORCED_STOP_GRACE_SECONDS))
                except subprocess.TimeoutExpired:
                    process.kill()
                    return int(process.wait())
        stop_event.wait(0.25)


def validate_completed_worker_records(claim_path: Path) -> None:
    expected = {f"rank_{rank}.json" for rank in range(WORLD_SIZE)}
    workers_dir = claim_path / "workers"
    if workers_dir.is_symlink() or not workers_dir.is_dir():
        raise RuntimeError("DDP3 worker registration directory is missing")
    actual = {entry.name for entry in os.scandir(workers_dir)}
    if actual != expected:
        raise RuntimeError(f"DDP3 worker registrations are incomplete or foreign: {sorted(actual)}")
    for required in ("rank_map.json", "token_consumed.json"):
        path = claim_path / required
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"DDP3 completed launch lacks {required}")


def controller_main(args: argparse.Namespace) -> int:
    validate_controller_args(args)
    reject_controller_environment()
    source_hashes = verify_static_sources()
    source_config = load_verified_yaml(CONFIG, EXPECTED_BUILD[CONFIG])
    validate_protocol_config(source_config)
    _, preflight_sha, replay_sha, preflight_stable = load_preflight(
        source_config,
        source_hashes,
    )
    runtime_contract = expected_runtime_contract(source_hashes)
    resume_path = Path(args.resume).resolve() if args.resume is not None else None
    reconcile_stale_controller_claim()
    resume_reference = (
        controller_resume_reference(resume_path, runtime_contract)
        if resume_path is not None
        else None
    )
    gpu_preflight = controller_gpu_preflight()
    token = secrets.token_urlsafe(48)
    token_sha = hashlib.sha256(token.encode()).hexdigest()
    controller_id = secrets.token_hex(24)
    reservation_id = (
        read_existing_reservation_id()
        if resume_path is not None
        else secrets.token_hex(24)
    )
    claim_path, owner, owner_bytes = acquire_controller_claim(
        "resume" if resume_path is not None else "fresh",
        None if resume_path is None else resume_reference["sha256"],
        controller_id,
        token_sha,
        reservation_id,
    )
    watchdog_stop = threading.Event()
    violation_event = threading.Event()
    watchdog = None
    try:
        if resume_path is None:
            reserve_fresh_output(claim_path, owner_bytes, reservation_id)
        else:
            verify_output_reservation(claim_path, require_pristine=False)
        _, authorization_sha = write_controller_authorization(
            claim_path,
            owner,
            token_sha,
            source_hashes,
            runtime_contract,
            preflight_sha,
            replay_sha,
            preflight_stable,
            gpu_preflight,
            resume_reference,
        )
        verify_output_reservation(
            claim_path,
            require_pristine=resume_path is None,
        )
        watchdog = threading.Thread(
            target=watchdog_loop,
            args=(
                claim_path,
                watchdog_stop,
                violation_event,
                tuple(gpu_preflight["authorized_inventory_uuids"]),
            ),
            name="ddp3-power-watchdog",
            daemon=True,
        )
        watchdog.start()
        process = subprocess.Popen(
            canonical_torchrun_command(args),
            env=build_worker_environment(claim_path, token, controller_id),
        )
        try:
            return_code = monitor_worker_process(
                process,
                claim_path,
                watchdog_stop,
                violation_event,
            )
        except KeyboardInterrupt:
            request_coordinated_stop(
                claim_path,
                "controller_keyboard_interrupt",
                {"controller_id": controller_id},
            )
            violation_event.set()
            return_code = monitor_worker_process(
                process,
                claim_path,
                watchdog_stop,
                violation_event,
            )
        finally:
            watchdog_stop.set()
            if watchdog is not None:
                watchdog.join(timeout=GPU_QUERY_TIMEOUT_SECONDS + 2.0)
                if watchdog.is_alive():
                    violation_event.set()
                    raise RuntimeError("DDP3 power watchdog did not stop after worker exit")
        if violation_event.is_set():
            raise RuntimeError(
                f"DDP3 controller stopped workers after a power/watchdog violation; "
                f"torchrun exit={return_code}"
            )
        if return_code != 0:
            raise RuntimeError(f"DDP3 torchrun failed with exit status {return_code}")
        validate_completed_worker_records(claim_path)
        contract = load_json_object(
            (claim_path / "contract.json").read_bytes(),
            str(claim_path / "contract.json"),
        )
        if contract.get("authorization_sha256") != authorization_sha:
            raise RuntimeError("DDP3 controller contract authorization digest drifted")
    except BaseException as error:
        watchdog_stop.set()
        if watchdog is not None:
            watchdog.join(timeout=GPU_QUERY_TIMEOUT_SECONDS + 2.0)
            if watchdog.is_alive():
                violation_event.set()
                error.add_note("DDP3 power watchdog remained alive after the bounded join")
        try:
            finalize_controller_claim(
                claim_path,
                "failed",
                f"{type(error).__name__}: {error}",
            )
        except BaseException as finalize_error:
            error.add_note(
                "Controller-claim finalization also failed: "
                f"{type(finalize_error).__name__}: {finalize_error}"
            )
        raise
    finalize_controller_claim(claim_path, "completed")
    return 0


def worker_main(args: argparse.Namespace) -> int:
    topology = validate_worker_topology_environment()
    (
        claim_path,
        owner,
        authorization,
        source_hashes,
        source_config,
    ) = verify_worker_authorization(topology, args)
    authorization_sha = sha256(claim_path / "authorization.json")
    register_worker(
        claim_path,
        topology,
        owner,
        authorization_sha,
        source_hashes,
    )
    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, str(INVESTIGATORS))
    sys.path.insert(0, str(BASE))
    runtime_hashes = authorization["runtime_contract"]["runtime_source_sha256"]
    loaded_modules = {
        name: load_exact_source_module(name, path, runtime_hashes[name])
        for name, path in RUNTIME_SOURCES.items()
    }
    quality_loss_module = loaded_modules["quality_aligned_loss"]
    sparse_sew_module = loaded_modules["models.snn.sparse_sew_resnet"]
    se_module = loaded_modules["se_per_sample_patch"]
    ic_model_module = loaded_modules["V2.models.sparse_voxel_det_ic"]
    base = loaded_modules["sparse_trainer_ic_quality"]
    se_source_sha = se_module.install()
    if sparse_sew_module.SparseSEBlock.forward is not se_module.per_sample_forward:
        raise RuntimeError("SparseSEBlock per-sample patch installation failed")
    if se_source_sha != EXPECTED_REMOTE[PROJECT / "models/snn/sparse_sew_resnet.py"]:
        raise RuntimeError("Per-sample SE installed from unexpected source bytes")
    runtime_contract = expected_runtime_contract(source_hashes)
    if runtime_contract != authorization["runtime_contract"]:
        raise RuntimeError("DDP3 runtime contract changed before process-group initialization")
    device = None
    import torch.distributed as dist

    try:
        device = initialize_worker_process_group(topology)
        placement = post_context_gpu_evidence(topology)
        launch_origin = collect_worker_provenance(
            claim_path,
            owner,
            authorization_sha,
            topology,
            placement,
        )
        resume_path = Path(args.resume).resolve() if args.resume is not None else None
        verified_resume_checkpoint = None
        resume_receipt = None
        resume_error = None
        if resume_path is not None:
            try:
                resume_receipt, verified_resume_checkpoint = verify_resume(
                    resume_path,
                    source_config,
                    runtime_contract,
                    authorization["resume_reference"],
                )
            except BaseException as error:
                resume_error = f"{type(error).__name__}: {error}"
        resume_code = collective_consensus_code(int(resume_error is not None), device)
        if resume_code:
            errors = gather_rank_errors(resume_error)
            abort_distributed_job(
                "DDP3 resume verification failed collectively: " + "; ".join(errors)
            )
        if resume_receipt is not None:
            try:
                checkpoint_digest_consensus(str(resume_receipt["sha256"]))
            except BaseException as error:
                abort_distributed_job(
                    f"DDP3 checkpoint digest consensus failed: {type(error).__name__}: {error}"
                )

        QualityLoss = quality_loss_module.SparseVoxelDetLoss
        SparseVoxelDetIC = ic_model_module.SparseVoxelDetIC
        import spconv.pytorch as spconv

        def load_verified_quality_config_with_contract(config_path: str):
            if Path(config_path).resolve() != CONFIG.resolve():
                raise RuntimeError(f"Unexpected DDP3 quality config: {config_path}")
            config = copy.deepcopy(source_config)
            validate_protocol_config(config)
            loss = config.get("loss", {})
            expected_loss = {
                "task_aligned_enabled": True,
                "task_aligned_alpha": 1.0,
                "task_aligned_beta": 6.0,
                "dynamic_k_topq": 10,
                "quality_bootstrap_epochs": 2,
                "nwd_weight": 0.5,
                "nwd_c": 12.8,
            }
            for key, expected in expected_loss.items():
                if loss.get(key) != expected:
                    raise RuntimeError(
                        f"DDP3 quality loss config mismatch for {key}: {loss.get(key)!r}"
                    )
            config.setdefault("experiment", {})["runtime_contract"] = runtime_contract
            return config

        def ic_factory(*factory_args, **factory_kwargs):
            model = SparseVoxelDetIC(*factory_args, **factory_kwargs)
            inverse = sum(
                isinstance(module, spconv.SparseInverseConv3d)
                for module in model.modules()
            )
            transpose = sum(
                isinstance(module, spconv.SparseConvTranspose3d)
                for module in model.modules()
            )
            if inverse != 2 or transpose != 0:
                raise RuntimeError(
                    f"DDP3 IC inventory mismatch: inverse={inverse}, transpose={transpose}"
                )
            return model

        def quality_factory(*factory_args, **factory_kwargs):
            factory_kwargs.update(
                {
                    "task_aligned_enabled": True,
                    "task_aligned_alpha": 1.0,
                    "task_aligned_beta": 6.0,
                    "dynamic_k_topq": 10,
                    "quality_bootstrap_epochs": 2,
                    "nwd_weight": 0.5,
                    "nwd_c": 12.8,
                }
            )
            return QualityLoss(*factory_args, **factory_kwargs)

        original_load_checkpoint = base.load_checkpoint
        resume_consumed = False

        def load_verified_checkpoint(
            path,
            model,
            optimizer=None,
            scheduler=None,
            scaler=None,
        ):
            nonlocal resume_consumed
            if verified_resume_checkpoint is None or resume_path is None:
                raise RuntimeError("Trainer requested resume during a fresh DDP3 launch")
            if Path(path).resolve() != resume_path or resume_consumed:
                raise RuntimeError(f"Trainer requested unauthorized/repeated resume load: {path}")
            resume_consumed = True
            return original_load_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                scaler,
                checkpoint_data=verified_resume_checkpoint,
            )

        def checkpoint_commit_hook(checkpoint_path, checkpoint):
            if int(topology["rank"]) != 0:
                raise RuntimeError("Only DDP3 rank 0 may commit checkpoints")
            record_checkpoint_commit(
                claim_path,
                Path(checkpoint_path),
                checkpoint,
                source_config,
                runtime_contract,
            )

        base.load_verified_quality_config = load_verified_quality_config_with_contract
        base.SparseVoxelDet = ic_factory
        base.SparseVoxelDetLoss = quality_factory
        base.CHECKPOINT_COMMIT_HOOK = checkpoint_commit_hook
        base.COLLECTIVE_CONSENSUS = collective_consensus_code
        base.GLOBAL_REDUCE_SCALARS = global_reduce_scalars
        base.BROADCAST_RANK0_PAYLOAD = broadcast_rank0_payload
        base.RUN_RANK0_STAGE = collective_rank0_stage
        base.GATHER_RANK_ERRORS = gather_rank_errors
        base.ABORT_DISTRIBUTED_JOB = abort_distributed_job
        base.COLLECTIVE_STOP_REQUESTED = collective_stop_requested
        base.WORKER_CONTEXT = {
            "rank": int(topology["rank"]),
            "local_rank": int(topology["local_rank"]),
            "world_size": WORLD_SIZE,
            "device": device,
            "output_dir": OUTPUT_DIR,
            "claim_path": claim_path,
            "stop_request_path": claim_path / "stop_request.json",
            "runtime_contract": runtime_contract,
            "launch_origin": launch_origin,
            "resume_receipt": resume_receipt,
        }
        if resume_path is not None:
            base.load_checkpoint = load_verified_checkpoint
        original_argv = sys.argv
        try:
            sys.argv = [
                str(Path(base.__file__).resolve()),
                *canonical_trainer_argv(args, int(topology["local_rank"])),
            ]
            trainer_status = base.main()
        finally:
            sys.argv = original_argv
        missing_resume = int(resume_path is not None and not resume_consumed)
        if collective_consensus_code(missing_resume, device):
            abort_distributed_job(
                "At least one DDP3 worker did not consume its verified checkpoint exactly once"
            )
        return int(trainer_status)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def entrypoint(argv: list[str] | None = None) -> int:
    args = parse_launch_args(argv)
    if os.environ.get("SPARSEVOXELDET_DDP3_MODE") == "worker":
        return worker_main(args)
    return controller_main(args)


if __name__ == "__main__":
    raise SystemExit(entrypoint())
