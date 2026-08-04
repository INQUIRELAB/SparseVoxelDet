#!/usr/bin/env python3
"""Exclusive record/replay preflight for the DDP3 quality bundle; never launches training."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


BASE = Path(__file__).resolve().parent
LAUNCHER = BASE / "train_ic_quality.py"
EXPECTED_LAUNCHER_SHA256 = "340b900bfffdbe97ec0506c889b449b71b1177ade669e621aa5dd573b06a0310"
LOCAL_STATIC_RECEIPT = BASE / "preflight_local_static.json"
LOCAL_STATIC_REPLAY = BASE / "preflight_local_static_replay.json"
FULL_RECEIPT = BASE / "preflight.json"
FULL_REPLAY = BASE / "preflight_replay.json"
EXPECTED_LOCAL = {
    "strict_loss": (BASE / "strict_loss.py", "92957d72221c72a656bd10aef3fdc1f74e3f5c35e6c118be8ef8c6e6cc3c4526"),
    "quality_loss": (BASE / "quality_aligned_loss.py", "08b7030fbab85449ef4a17ce4a373e73fa2c8392fb12c28119dc23d6c06d6290"),
    "trainer": (BASE / "sparse_trainer_ic_quality.py", "45c4ded584a6e60b0b24100762487f8c6f26a97a330c1618dc0921a3fbabd442"),
    "config": (BASE / "ic_quality_ddp3_e20.yaml", "1067e925c3ffe753c4bdbf5816e30a38d863b48d184d9638720c374cc86bbf42"),
    "launcher": (LAUNCHER, EXPECTED_LAUNCHER_SHA256),
    "quality_tests": (BASE / "test_quality_aligned_loss.py", "41320d1f75ee34af60ec1bc5922ecd334f8e81082c91ad778b417de3534f700e"),
    "contract_tests": (BASE / "test_ddp3_contracts.py", "e02239fdbb0acd546cc456328f6b83578ec33f0088d354f5a80045e412990dae"),
}
EXPECTED_LABEL_MANIFEST_SHA256 = "6a973831e215c733e77f4ba2553ae0e138a20cf01f1c5e30387292f52b2c56ee"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


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


def load_launcher():
    actual = sha256(LAUNCHER)
    if actual != EXPECTED_LAUNCHER_SHA256:
        raise RuntimeError(f"Launcher hash mismatch: {actual} != {EXPECTED_LAUNCHER_SHA256}")
    spec = importlib.util.spec_from_file_location("ddp3_preflight_launcher", LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load launcher source: {LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    source = LAUNCHER.read_bytes()
    exec(compile(source, str(LAUNCHER), "exec"), module.__dict__)
    return module


def verify_local_sources() -> dict[str, dict[str, str]]:
    verified = {}
    for label, (path, expected) in EXPECTED_LOCAL.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{label} hash mismatch: {actual} != {expected}")
        verified[label] = {"path": str(path.resolve()), "sha256": actual}
    verified["preflight"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256(Path(__file__).resolve()),
    }
    return verified


def local_static_record(launcher) -> dict[str, object]:
    source_hashes = verify_local_sources()
    config = yaml.safe_load((BASE / "ic_quality_ddp3_e20.yaml").read_bytes())
    launcher.validate_protocol_config(config)
    sealed_token = "canonical_" + "test"
    references_sealed = sealed_token in json.dumps(config, sort_keys=True).lower()
    if references_sealed:
        raise RuntimeError("DDP3 config references the sealed test split")
    runtime_contract = launcher.expected_runtime_contract({
        "build": source_hashes,
        "remote": {
            str(path): {"path": str(path), "sha256": expected, "verified": False}
            for path, expected in launcher.EXPECTED_REMOTE.items()
        },
    })
    ddp_keys = (
        "runtime_policy_version", "ordered_physical_uuids", "forbidden_physical_uuids",
        "world_size", "local_world_size", "rank_mapping_policy", "per_rank_batch_size",
        "global_effective_batch_size", "gradient_accumulation_steps", "sampler",
        "train_roster_samples", "sampler_samples_per_epoch", "optimized_samples_per_epoch",
        "optimizer_steps_per_epoch",
        "warmup_optimizer_steps", "epochs", "total_optimizer_steps", "scheduler", "validation",
    )
    return {
        "status": "local_static_pass",
        "execution_mode": "record",
        "quality_config_references_sealed_test": False,
        "sealed_test_access_audit_performed": False,
        "sealed_test_access_result": "not_measured",
        "source_hashes": {"build": source_hashes},
        "remote_expected_sha256": {
            str(path): expected for path, expected in launcher.EXPECTED_REMOTE.items()
        },
        "quality_config_sha256": EXPECTED_LOCAL["config"][1],
        "ddp_policy": {key: runtime_contract[key] for key in ddp_keys},
        "expected_corrected_label_manifest_sha256": EXPECTED_LABEL_MANIFEST_SHA256,
        "expected_corrected_label_splits": EXPECTED_LABEL_SPLITS,
        "corrected_label_bytes_read": False,
        "dataset_traversal_performed": False,
        "remote_only_checks_pending": True,
    }


def full_record(launcher) -> dict[str, object]:
    config = yaml.safe_load(launcher.CONFIG.read_bytes())
    launcher.validate_protocol_config(config)
    source_hashes = launcher.verify_static_sources()
    return launcher.derive_preflight_stable(config, source_hashes)


def build_record(local_static: bool) -> dict[str, object]:
    launcher = load_launcher()
    stable = local_static_record(launcher) if local_static else full_record(launcher)
    return {
        **stable,
        "completed_central": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
    }


def replay_recorded(output: Path, replay_output: Path, current: dict[str, object]) -> dict[str, object]:
    receipt_bytes = output.read_bytes()
    recorded = load_json_object(receipt_bytes, str(output))
    stable_keys = [key for key in current if key != "completed_central"]
    if set(recorded) != set(current):
        raise RuntimeError(f"Preflight fields changed: {sorted(set(recorded) ^ set(current))}")
    for key in stable_keys:
        if recorded.get(key) != current.get(key):
            raise RuntimeError(f"Immutable DDP3 preflight field drifted: {key}")
    replay = {
        "status": "pass",
        "execution_mode": "check_only_replay",
        "recorded_preflight_path": str(output.resolve()),
        "recorded_preflight_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "recorded_preflight_verified": True,
        "verified_field_names": stable_keys,
        "verified_at_central": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
    }
    write_json_exclusive(replay_output, replay)
    return replay


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--replay-output", type=Path)
    parser.add_argument("--local-static", action="store_true")
    args = parser.parse_args()
    default_output = LOCAL_STATIC_RECEIPT if args.local_static else FULL_RECEIPT
    default_replay = LOCAL_STATIC_REPLAY if args.local_static else FULL_REPLAY
    output = (args.output or default_output).resolve()
    replay_output = (args.replay_output or default_replay).resolve()
    if args.local_static and output == FULL_RECEIPT.resolve():
        raise RuntimeError("Local-static evidence cannot occupy the launch preflight receipt path")
    current = build_record(args.local_static)
    if args.check_only:
        if not output.is_file():
            raise FileNotFoundError(f"Recorded preflight is missing: {output}")
        result = replay_recorded(output, replay_output, current)
    else:
        write_json_exclusive(output, current)
        result = current
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
