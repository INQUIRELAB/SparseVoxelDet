import argparse
import ast
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


BASE = Path(__file__).resolve().parent
PASSED = BASE.parent.parent / "2026-07-19_ic_quality_objective_sol" / "build"
SCIENTIFIC_HASHES = {
    "strict_loss.py": "92957d72221c72a656bd10aef3fdc1f74e3f5c35e6c118be8ef8c6e6cc3c4526",
    "quality_aligned_loss.py": "08b7030fbab85449ef4a17ce4a373e73fa2c8392fb12c28119dc23d6c06d6290",
    "test_quality_aligned_loss.py": "41320d1f75ee34af60ec1bc5922ecd334f8e81082c91ad778b417de3534f700e",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_launcher():
    return load_source_module("ddp3_launcher_under_test", BASE / "train_ic_quality.py")


def load_preflight():
    return load_source_module("ddp3_preflight_under_test", BASE / "preflight_quality.py")


def load_reduction_function():
    path = BASE / "sparse_trainer_ic_quality.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "reduce_ddp_epoch_state"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Dict": Dict,
        "torch": torch,
        "QUALITY_WORLD_SIZE": 3,
        "GLOBAL_REDUCE_SCALARS": None,
        "RuntimeError": RuntimeError,
        "sorted": sorted,
        "set": set,
        "float": float,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def load_trainer_functions(*names, namespace=None):
    path = BASE / "sparse_trainer_ic_quality.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name in names
    ]
    if len(selected) != len(names):
        missing = sorted(set(names) - {item.name for item in selected})
        raise AssertionError(f"Missing trainer source nodes: {missing}")
    for item in selected:
        if isinstance(item, ast.FunctionDef):
            item.decorator_list = []
            item.returns = None
            for arg in (*item.args.posonlyargs, *item.args.args, *item.args.kwonlyargs):
                arg.annotation = None
            if item.args.vararg is not None:
                item.args.vararg.annotation = None
            if item.args.kwarg is not None:
                item.args.kwarg.annotation = None
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    values = {} if namespace is None else dict(namespace)
    exec(compile(module, str(path), "exec"), values)
    return values


def valid_public_argv(launcher):
    return [
        "--config", str(launcher.CONFIG),
        "--output_dir", str(launcher.OUTPUT_DIR),
        "--seed", "42",
    ]


def gpu_rows(launcher, selected_draws=(40.0, 50.0, 60.0), foreign_draw=100.0):
    rows = []
    for index, (uuid, draw) in enumerate(zip(launcher.ORDERED_UUIDS, selected_draws)):
        rows.append({
            "index": str(index),
            "uuid": uuid,
            "name": "NVIDIA GeForce RTX 5090",
            "power_limit": "400",
            "power_draw": str(draw),
            "memory_used": "0",
        })
    rows.append({
        "index": "6",
        "uuid": "GPU-foreign",
        "name": "NVIDIA GeForce RTX 5090",
        "power_limit": "400",
        "power_draw": str(foreign_draw),
        "memory_used": "0",
    })
    return rows


def local_epoch_state(rank: int) -> dict[str, float]:
    state = {
        "loss_sum": rank + 1.0,
        "cls_loss_sum": rank + 2.0,
        "reg_loss_sum": rank + 3.0,
        "ctr_loss_sum": rank + 4.0,
        "iou_quality_loss_sum": 0.0,
        "proposal_loss_sum": 0.0,
        "ranking_loss_sum": 0.0,
        "uncertainty_loss_sum": 0.0,
        "positive_query_ratio_sum": 0.0,
        "ranking_gap_sum": 0.0,
        "near_boundary_mass_sum": 0.0,
        "proposal_recall16_sum": 0.0,
        "proposal_recall32_sum": 0.0,
        "proposal_recall64_sum": 0.0,
        "proposal_recall128_sum": 0.0,
        "sample_count": rank + 2.0,
        "positive_count": rank + 3.0,
        "quality_num_gt": rank + 4.0,
        "quality_num_gt_with_candidates": rank + 3.0,
        "quality_gt_zero_candidates": 1.0,
        "quality_dynamic_k_sum": rank + 8.0,
        "quality_num_pos_raw": rank + 6.0,
        "quality_quota_deficit": rank,
        "quality_conflict_sites": rank + 1.0,
        "quality_gt_zero_after_conflict": rank,
        "quality_multi_gt_samples": rank + 2.0,
        "quality_multi_gt_gt_zero_assigned": rank,
        "quality_candidate_total": rank + 10.0,
        "quality_cls_total": rank + 5.0,
        "quality_iou_total": rank + 6.0,
        "quality_candidate_count_max": rank + 20.0,
        "quality_classification_target_max": rank / 10.0,
        "quality_decoded_iou_target_max": rank / 5.0,
        "clip_sample_count": rank + 2.0,
        "clip_raw_sum": rank + 30.0,
        "clip_kept_sum": rank + 20.0,
        "clip_fraction_sum": rank / 10.0,
        "clip_clipped_count": rank,
        "optimizer_steps": 7.0,
        "optimizer_steps_completed": 107.0,
        "skipped_non_finite": 1.0,
        "skipped_non_finite_grad": 2.0,
        "skipped_oom": 3.0,
        "sanitized_grad_steps": float(rank),
        "processed_batches": 13.0,
        "successful_batches": 7.0,
        "nonfinite_events": 3.0,
        "first_nonfinite_batch": 5.0 if rank == 2 else 67784.0,
        "max_consecutive_nonfinite": rank + 1.0,
        "aborted_early": 0.0,
        "stopped_by_controller": 0.0,
        "learning_rate": 0.0003,
        "elapsed_seconds": rank + 9.0,
    }
    return state


def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def gloo_contract_worker(rank, world_size, port, result_dir, stop_file, checkpoint_file):
    launcher = load_launcher()
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
    )
    device = torch.device("cpu")

    reduction_namespace = load_reduction_function()
    reduction_namespace["GLOBAL_REDUCE_SCALARS"] = launcher.global_reduce_scalars
    reduced = reduction_namespace["reduce_ddp_epoch_state"](
        local_epoch_state(rank), device, world_size
    )

    def local_abort(message):
        raise RuntimeError(message)

    trainer_namespace = load_trainer_functions(
        "collective_next_batch",
        "synchronized_scaler_scale",
        "collective_scaler_backoff",
        namespace={
            "ABORT_DISTRIBUTED_JOB": local_abort,
            "GLOBAL_REDUCE_SCALARS": launcher.global_reduce_scalars,
            "math": __import__("math"),
        },
    )
    loader_has_batch, loader_batch = trainer_namespace["collective_next_batch"](
        iter([rank]), device
    )

    class RankLocalBrokenIterator:
        def __iter__(self):
            return self

        def __next__(self):
            if rank == 2:
                raise OSError("synthetic DataLoader retrieval failure")
            return rank

    loader_divergence_error = ""
    try:
        trainer_namespace["collective_next_batch"](
            RankLocalBrokenIterator(), device
        )
    except RuntimeError as error:
        loader_divergence_error = str(error)

    class RankScaler:
        def __init__(self, scale):
            self.scale = float(scale)

        def get_scale(self):
            return self.scale

        def get_backoff_factor(self):
            return 0.5

        def update(self, new_scale=None):
            self.scale = float(new_scale)

    scaler = RankScaler(8.0)
    synchronized_backoff = trainer_namespace["collective_scaler_backoff"](
        scaler, device
    )
    scaler_divergence_error = ""
    try:
        trainer_namespace["synchronized_scaler_scale"](
            RankScaler(16.0 if rank == 2 else 8.0), device
        )
    except RuntimeError as error:
        scaler_divergence_error = str(error)

    forward_code = launcher.collective_consensus_code(3 if rank == 2 else 0, device)
    forward_errors = launcher.gather_rank_errors("synthetic forward failure" if rank == 2 else None)
    asymmetric_stop = bool(launcher.collective_consensus_code(int(rank == 1), device))
    file_stop = launcher.collective_stop_requested(Path(stop_file), device)

    validation_error = ""
    try:
        launcher.collective_rank0_stage(
            rank,
            "synthetic validation",
            lambda: (_ for _ in ()).throw(RuntimeError("synthetic validation failure")),
        )
    except RuntimeError as error:
        validation_error = str(error)

    stage_path = Path(result_dir) / "rank0_stage.txt"

    def rank0_writer():
        stage_path.write_text("rank0-only", encoding="utf-8")
        return {"writer_rank": rank, "path": str(stage_path)}

    stage_payload = launcher.collective_rank0_stage(rank, "rank-zero write", rank0_writer)

    read_count = 0
    with Path(checkpoint_file).open("rb") as handle:
        checkpoint_bytes = handle.read()
        read_count += 1
    checkpoint_digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    consensus_digest = launcher.checkpoint_digest_consensus(checkpoint_digest)

    resume_dataset = RangeDataset(41)
    resume_sampler = DistributedSampler(
        resume_dataset, num_replicas=3, rank=rank, shuffle=True, seed=42, drop_last=True
    )
    resume_sampler.set_epoch(3)
    uninterrupted = [
        batch.tolist()
        for batch in DataLoader(
            resume_dataset, batch_size=2, sampler=resume_sampler, drop_last=True, num_workers=0
        )
    ]
    replay_sampler = DistributedSampler(
        resume_dataset, num_replicas=3, rank=rank, shuffle=True, seed=42, drop_last=True
    )
    replay_sampler.set_epoch(3)
    resumed = [
        batch.tolist()
        for index, batch in enumerate(
            DataLoader(
                resume_dataset, batch_size=2, sampler=replay_sampler, drop_last=True, num_workers=0
            )
        )
        if index >= 2
    ]

    result = {
        "rank": rank,
        "reduced": reduced,
        "loader_has_batch": loader_has_batch,
        "loader_batch": loader_batch,
        "loader_divergence_error": loader_divergence_error,
        "synchronized_backoff": synchronized_backoff,
        "scaler_divergence_error": scaler_divergence_error,
        "forward_code": forward_code,
        "forward_errors": forward_errors,
        "asymmetric_stop": asymmetric_stop,
        "file_stop": file_stop,
        "validation_error": validation_error,
        "stage_payload": stage_payload,
        "checkpoint_read_count": read_count,
        "checkpoint_digest": consensus_digest,
        "resume_suffix_matches": resumed == uninterrupted[2:],
        "resume_next_batch_idx": 2,
        "resume_next_rank_local_sample_offset": 4,
        "resume_optimizer_steps_in_epoch": 2,
        "resume_optimizer_steps_completed": 102 + len(resumed),
    }
    Path(result_dir, f"rank_{rank}.json").write_text(
        json.dumps(result, sort_keys=True), encoding="utf-8"
    )
    dist.barrier()
    dist.destroy_process_group()


class RangeDataset(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


def test_scientific_files_are_byte_identical_to_passed_bundle():
    for name, expected in SCIENTIFIC_HASHES.items():
        assert sha256(BASE / name) == expected
        assert (BASE / name).read_bytes() == (PASSED / name).read_bytes()


def test_preflight_rejects_duplicate_json_keys():
    preflight = load_preflight()
    with pytest.raises(RuntimeError, match="Duplicate JSON key"):
        preflight.load_json_object(b'{"status":"pass","status":"forged"}', "forged receipt")


def test_preflight_verified_field_names_survive_authorization_json_roundtrip(tmp_path):
    launcher = load_launcher()
    receipt_path = tmp_path / "preflight.json"
    replay_path = tmp_path / "preflight_replay.json"
    authorization_path = tmp_path / "authorization.json"
    stable = {
        "status": "pass",
        "remote_only_checks_pending": False,
        "canonical_val": {"files": 1},
    }
    launcher.write_json_exclusive(
        receipt_path,
        {**stable, "completed_central": "2026-07-21T06:31:00-05:00"},
    )
    receipt_bytes = receipt_path.read_bytes()
    launcher.write_json_exclusive(
        replay_path,
        {
            "status": "pass",
            "execution_mode": "check_only_replay",
            "recorded_preflight_path": str(receipt_path.resolve()),
            "recorded_preflight_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "recorded_preflight_verified": True,
            "verified_field_names": list(stable),
            "verified_at_central": "2026-07-21T06:31:01-05:00",
        },
    )
    launcher.write_json_exclusive(
        authorization_path,
        {"preflight_stable": stable},
    )
    with authorization_path.open("r", encoding="utf-8") as handle:
        roundtripped = json.load(handle)["preflight_stable"]
    assert set(roundtripped) == set(stable)
    assert list(roundtripped) != list(stable)
    _, receipt_sha, _ = launcher.verify_preflight_evidence(
        receipt_bytes,
        replay_path.read_bytes(),
        roundtripped,
        receipt_path,
        replay_path,
    )
    assert receipt_sha == hashlib.sha256(receipt_bytes).hexdigest()


def test_public_parser_and_controller_environment_fail_closed():
    launcher = load_launcher()
    args = launcher.parse_launch_args(valid_public_argv(launcher))
    assert vars(args) == {
        "config": str(launcher.CONFIG),
        "output_dir": str(launcher.OUTPUT_DIR),
        "seed": 42,
        "resume": None,
    }
    for extra in (
        ["--seed", "42"],
        ["--config", str(launcher.CONFIG)],
        ["--output_dir", str(launcher.OUTPUT_DIR)],
        ["--conf", str(launcher.CONFIG)],
        ["--device", "cpu"],
        ["--unknown", "x"],
    ):
        with pytest.raises(SystemExit):
            launcher.parse_launch_args(valid_public_argv(launcher) + extra)
    for key in sorted(launcher.CONTROLLER_PROTECTED_ENV_KEYS):
        with pytest.raises(RuntimeError, match="protected environment"):
            launcher.reject_controller_environment({key: "forged"})


def test_torchrun_topology_and_pre_nccl_device_order_are_exact():
    launcher = load_launcher()
    assert launcher.WORLD_SIZE == 3
    assert launcher.ORDERED_UUIDS == (
        "GPU-1d11b997-90a9-ece7-9ce6-44ad85346817",
        "GPU-2a7554bd-5a91-25ab-3338-e2308ecb2a27",
        "GPU-48d3a2b0-fc78-8bc8-fdce-5a246fdc4989",
    )
    args = argparse.Namespace(
        config=str(launcher.CONFIG), output_dir=str(launcher.OUTPUT_DIR), seed=42, resume=None
    )
    command = launcher.canonical_torchrun_command(args)
    assert command[1:8] == [
        "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        "--nproc-per-node=3", "--max-restarts=0", str(Path(launcher.__file__).resolve()),
    ]
    environment = launcher.build_worker_environment(Path("claim"), "token", "controller")
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert environment["CUDA_VISIBLE_DEVICES"] == ",".join(launcher.ORDERED_UUIDS)
    source = (BASE / "train_ic_quality.py").read_text(encoding="utf-8")
    function = source[source.index("def initialize_worker_process_group"):source.index("def collect_worker_provenance")]
    assert function.index("torch.cuda.set_device(local_rank)") < function.index("dist.init_process_group(backend=\"nccl\")")


def test_gpu_projection_and_watchdog_thresholds_are_exclusive():
    launcher = load_launcher()
    result = launcher.evaluate_controller_gpu_rows(gpu_rows(launcher), [])
    assert result["aggregate_power_draw_watts_before"] == 250.0
    assert result["selected_power_draw_watts_before"] == 150.0
    assert result["projected_aggregate_power_watts_at_selected_limits"] == 1300.0

    occupied = [{"pid": "123", "uuid": launcher.ORDERED_UUIDS[0]}]
    with pytest.raises(RuntimeError, match="foreign compute PIDs"):
        launcher.evaluate_controller_gpu_rows(gpu_rows(launcher), occupied)

    rows = gpu_rows(launcher, selected_draws=(100.0, 100.0, 100.0), foreign_draw=400.0)
    rows.append({
        "index": "5", "uuid": "GPU-foreign-2", "name": "NVIDIA GeForce RTX 5090",
        "power_limit": "400", "power_draw": "400", "memory_used": "0",
    })
    with pytest.raises(RuntimeError, match="requires < 2000 W"):
        launcher.evaluate_controller_gpu_rows(rows, [])

    power_rows = [{"uuid": uuid, "power_draw": "400"} for uuid in launcher.ORDERED_UUIDS]
    power_rows.extend([
        {"uuid": "GPU-foreign", "power_draw": "400"},
        {"uuid": "GPU-foreign-2", "power_draw": "400"},
    ])
    authorized_inventory = tuple(row["uuid"] for row in power_rows)
    sample = launcher.evaluate_power_rows(power_rows, authorized_inventory)
    assert not any(item["kind"] == "selected_power" for item in sample["violations"])
    assert any(item["kind"] == "aggregate_power" for item in sample["violations"])
    power_rows[0]["power_draw"] = "400.1"
    sample = launcher.evaluate_power_rows(power_rows, authorized_inventory)
    assert any(item["kind"] == "selected_power" for item in sample["violations"])


def test_controller_refuses_baseline_and_foreign_pid_cards_without_signalling():
    launcher = load_launcher()
    baseline_uuid = "GPU-b279b278-d3e7-eb16-73d2-f6f4b002276c"
    assert launcher.FORBIDDEN_UUIDS == frozenset({baseline_uuid})
    assert baseline_uuid not in launcher.ORDERED_UUIDS
    forbidden_mask = (baseline_uuid, *launcher.ORDERED_UUIDS[1:])
    with pytest.raises(RuntimeError, match="forbidden baseline GPU"):
        launcher.evaluate_controller_gpu_rows(
            gpu_rows(launcher), [], selected_uuids=forbidden_mask
        )
    for uuid in launcher.ORDERED_UUIDS:
        with pytest.raises(RuntimeError, match="foreign compute PIDs"):
            launcher.evaluate_controller_gpu_rows(
                gpu_rows(launcher), [{"pid": "123", "uuid": uuid}]
            )
    source = (BASE / "train_ic_quality.py").read_text(encoding="utf-8")
    preflight_source = source[
        source.index("def evaluate_controller_gpu_rows"):
        source.index("def evaluate_power_rows")
    ]
    assert ".terminate(" not in preflight_source
    assert ".kill(" not in preflight_source


@pytest.mark.parametrize("bad_value", ["nan", "inf", "-inf", "-0.1"])
def test_power_telemetry_rejects_nonfinite_and_negative_values(bad_value):
    launcher = load_launcher()
    controller_rows = gpu_rows(launcher)
    controller_rows[0]["power_draw"] = bad_value
    with pytest.raises(RuntimeError, match="Invalid GPU"):
        launcher.evaluate_controller_gpu_rows(controller_rows, [])

    power_rows = [
        {"uuid": row["uuid"], "power_draw": row["power_draw"]}
        for row in gpu_rows(launcher)
    ]
    authorized_inventory = tuple(row["uuid"] for row in power_rows)
    power_rows[0]["power_draw"] = bad_value
    with pytest.raises(RuntimeError, match="Invalid GPU"):
        launcher.evaluate_power_rows(power_rows, authorized_inventory)


def test_watchdog_power_inventory_must_match_controller_inventory_exactly():
    launcher = load_launcher()
    power_rows = [
        {"uuid": row["uuid"], "power_draw": row["power_draw"]}
        for row in gpu_rows(launcher)
    ]
    authorized_inventory = tuple(row["uuid"] for row in power_rows)
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        launcher.evaluate_power_rows(power_rows[:-1], authorized_inventory)
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        launcher.evaluate_power_rows(
            power_rows + [{"uuid": "GPU-unexpected", "power_draw": "1"}],
            authorized_inventory,
        )


def test_post_context_occupancy_requires_exact_one_worker_per_selected_uuid():
    launcher = load_launcher()
    mapping = [
        {"rank": rank, "pid": 1000 + rank}
        for rank in range(launcher.WORLD_SIZE)
    ]
    compute_rows = [
        {"pid": str(1000 + rank), "uuid": uuid}
        for rank, uuid in enumerate(launcher.ORDERED_UUIDS)
    ]
    occupancy = launcher.evaluate_selected_compute_occupancy(mapping, compute_rows)
    assert occupancy == [
        {"uuid": uuid, "compute_pids": [1000 + rank]}
        for rank, uuid in enumerate(launcher.ORDERED_UUIDS)
    ]
    with pytest.raises(RuntimeError, match="occupancy mismatch"):
        launcher.evaluate_selected_compute_occupancy(mapping, compute_rows[:-1])
    with pytest.raises(RuntimeError, match="occupancy mismatch"):
        launcher.evaluate_selected_compute_occupancy(
            mapping,
            compute_rows + [{"pid": "9999", "uuid": launcher.ORDERED_UUIDS[0]}],
        )


def test_worker_registration_is_rank_specific_and_exclusive(monkeypatch, tmp_path):
    launcher = load_launcher()
    claim = tmp_path / "claim"
    claim.mkdir()
    monkeypatch.setattr(launcher, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(launcher, "read_process_start_ticks", lambda pid: 123)
    monkeypatch.setattr(launcher, "read_boot_id", lambda: "boot")
    monkeypatch.setattr(launcher.socket, "gethostname", lambda: "host")
    owner = {"controller_id": "controller", "reservation_id": "reservation"}
    source_hashes = {"build": {}, "remote": {}}
    topology = {"rank": 0, "local_rank": 0, "expected_uuid": launcher.ORDERED_UUIDS[0]}
    path, record, digest = launcher.register_worker(claim, topology, owner, "a" * 64, source_hashes)
    assert path.name == "rank_0.json"
    assert record["rank"] == record["local_rank"] == 0
    assert digest == sha256(path)
    with pytest.raises(FileExistsError):
        launcher.register_worker(claim, topology, owner, "a" * 64, source_hashes)


def test_checkpoint_boundary_separates_runtime_policy_from_launch_origin(monkeypatch):
    launcher = load_launcher()
    source = yaml.safe_load(launcher.CONFIG.read_text(encoding="utf-8"))
    runtime_contract = {"policy": "stable-ddp3"}
    launch_origin = {"controller_identity": {"pid": 111, "token": "old"}}
    monkeypatch.setattr(launcher, "validate_ddp_provenance", lambda value: value)
    config = launcher.expected_checkpoint_config(source, runtime_contract)
    config["_runtime"] = {
        "strict_finite_checks": True,
        "finite_trace_dir": str(launcher.OUTPUT_DIR / "finite_trace"),
        "abort_on_skip_rate": 0.10,
        "abort_on_consecutive_nonfinite": 200,
        "is_main_process": True,
        "amp_mode": "fp16",
        "max_train_batches": None,
        "rank0_writer": True,
        "launch_origin": launch_origin,
    }
    checkpoint = {
        "epoch": 2,
        "batch_idx": 10,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "scaler_state_dict": None,
        "ema_state_dict": None,
        "metrics": {},
        "config": config,
        "checkpoint_lineage": launcher.CHECKPOINT_LINEAGE,
        "runtime_contract": runtime_contract,
        "launch_origin": launch_origin,
        "optimizer_boundary": {
            "micro_batches_in_window": 0,
            "optimizer_steps_in_epoch": 9,
            "optimizer_steps_completed": 100009,
            "sampler_epoch": 2,
            "next_batch_idx": 11,
            "next_rank_local_sample_offset": 22,
        },
    }
    verified = launcher.verify_checkpoint_dict(checkpoint, source, runtime_contract)
    assert verified["optimizer_boundary"]["optimizer_steps_in_epoch"] == 9
    later_controller = {"controller_identity": {"pid": 222, "token": "new"}}
    assert later_controller != verified["launch_origin"]
    checkpoint["checkpoint_lineage"] = "single-gpu-quality"
    with pytest.raises(RuntimeError, match="not from the DDP3 quality lineage"):
        launcher.verify_checkpoint_dict(checkpoint, source, runtime_contract)


def test_trainer_source_contains_global_reduction_and_coordinated_validation_contracts():
    source = (BASE / "sparse_trainer_ic_quality.py").read_text(encoding="utf-8")
    assert "global_state = reduce_ddp_epoch_state(local_epoch_state, device)" in source
    assert "run_rank0_stage(rank, \"unsharded rank-zero validation\"" in source
    assert "pre-validation optimizer-boundary checkpoint" in source
    assert "stop_request_path=Path(worker_context[\"stop_request_path\"])" in source
    assert "except ValidationStopRequested as error:" in source
    assert "finally:\n                        if ema_applied:" in source
    assert "epoch-boundary emergency full-state checkpoint" in source
    assert "validation selection and best checkpoint" in source
    assert "optimizer_steps_in_epoch" in source
    assert "optimizer_steps_completed" in source
    assert "stopped_by_controller" in source
    assert "one_uuid_child_environment(QUALITY_ORDERED_UUIDS[0])" in source
    assert "os.replace(temporary_path, path)" in source


def test_direct_trainer_entry_fails_before_heavy_imports():
    process = subprocess.run(
        [sys.executable, str(BASE / "sparse_trainer_ic_quality.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode != 0
    assert "Direct trainer entry is forbidden" in process.stderr


def test_real_sampler_loader_schedule_and_no_optimized_duplication():
    dataset = RangeDataset(406701)
    sampler_kept = set()
    optimized = set()
    sampler_counts = []
    optimized_counts = []
    for rank in range(3):
        sampler = DistributedSampler(
            dataset, num_replicas=3, rank=rank, shuffle=True, seed=42, drop_last=True
        )
        sampler.set_epoch(0)
        sampler_items = list(iter(sampler))
        assert len(sampler_items) == 135567
        assert len(set(sampler_items)) == len(sampler_items)
        assert sampler_kept.isdisjoint(sampler_items)
        sampler_kept.update(sampler_items)
        sampler_counts.append(len(sampler_items))
        loader = DataLoader(dataset, batch_size=2, sampler=sampler, drop_last=True, num_workers=0)
        assert len(loader) == 67783
        rank_items = []
        for batch in loader:
            rank_items.extend(int(item) for item in batch)
        assert len(rank_items) == 135566
        assert len(set(rank_items)) == len(rank_items)
        assert optimized.isdisjoint(rank_items)
        optimized.update(rank_items)
        optimized_counts.append(len(rank_items))
    assert sampler_counts == [135567] * 3
    assert len(sampler_kept) == 406701
    assert optimized_counts == [135566] * 3
    assert len(optimized) == 406698
    config = yaml.safe_load((BASE / "ic_quality_ddp3_e20.yaml").read_text(encoding="utf-8"))
    ddp = config["ddp"]
    assert ddp["world_size"] == ddp["local_world_size"] == 3
    assert config["training"]["batch_size"] == ddp["per_rank_batch_size"] == 2
    assert config["training"]["gradient_accumulation_steps"] == 1
    assert config["training"]["batch_size"] * ddp["world_size"] == 6
    assert ddp["global_effective_batch_size"] == 6
    assert ddp["train_roster_samples"] == 406701
    assert ddp["sampler_samples_per_epoch"] == 406701
    assert ddp["optimized_samples_per_epoch"] == 406698
    assert ddp["optimizer_steps_per_epoch"] == 67783
    assert ddp["warmup_optimizer_steps"] == config["training"]["warmup_steps"] == 5000
    assert config["training"]["epochs"] == 20
    assert ddp["optimizer_steps_per_epoch"] * config["training"]["epochs"] == 1355660
    assert ddp["total_optimizer_steps"] == 1355660

def test_resumed_sampler_lineage_matches_uninterrupted_suffix():
    dataset = RangeDataset(41)
    for rank in range(3):
        sampler = DistributedSampler(
            dataset, num_replicas=3, rank=rank, shuffle=True, seed=42, drop_last=True
        )
        sampler.set_epoch(3)
        loader = DataLoader(dataset, batch_size=2, sampler=sampler, drop_last=True, num_workers=0)
        full_batches = [batch.tolist() for batch in loader]
        resume_batch = 2
        resumed_batches = [batch for index, batch in enumerate(full_batches) if index >= resume_batch]
        replayed = []
        sampler_replay = DistributedSampler(
            dataset, num_replicas=3, rank=rank, shuffle=True, seed=42, drop_last=True
        )
        sampler_replay.set_epoch(3)
        replay_loader = DataLoader(
            dataset, batch_size=2, sampler=sampler_replay, drop_last=True, num_workers=0
        )
        for index, batch in enumerate(replay_loader):
            if index >= resume_batch:
                replayed.append(batch.tolist())
        assert replayed == resumed_batches
        assert resume_batch * 2 == 4
        assert 100 + resume_batch + len(replayed) == 106


def test_real_cpu_gloo_multiprocess_contracts(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.fail("Real CPU/Gloo multiprocess tests are mandatory but Gloo is unavailable")
    result_dir = tmp_path / "gloo_results"
    result_dir.mkdir()
    stop_file = tmp_path / "stop_request.json"
    stop_file.write_text('{"status":"stop_requested"}\n', encoding="utf-8")
    checkpoint_file = tmp_path / "checkpoint.bin"
    checkpoint_file.write_bytes(b"identical-checkpoint-bytes")
    port = free_tcp_port()
    mp.spawn(
        gloo_contract_worker,
        args=(3, port, str(result_dir), str(stop_file), str(checkpoint_file)),
        nprocs=3,
        join=True,
    )
    results = [
        json.loads((result_dir / f"rank_{rank}.json").read_text(encoding="utf-8"))
        for rank in range(3)
    ]
    expected_digest = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
    for result in results:
        reduced = result["reduced"]
        assert reduced["loss_sum"] == 6.0
        assert reduced["sample_count"] == 9.0
        assert reduced["quality_candidate_count_max"] == 22.0
        assert reduced["max_consecutive_nonfinite"] == 3.0
        assert reduced["first_nonfinite_batch"] == 5.0
        assert reduced["optimizer_steps"] == 7.0
        assert reduced["optimizer_steps_completed"] == 107.0
        assert reduced["sanitized_grad_steps"] == 3.0
        assert result["loader_has_batch"] is True
        assert result["loader_batch"] == result["rank"]
        assert "DataLoader states diverged" in result["loader_divergence_error"]
        assert result["synchronized_backoff"] == 4.0
        assert "GradScaler state diverged" in result["scaler_divergence_error"]
        assert result["forward_code"] == 3
        assert result["forward_errors"] == ["rank 2: synthetic forward failure"]
        assert result["asymmetric_stop"] is True
        assert result["file_stop"] is True
        assert "synthetic validation failure" in result["validation_error"]
        assert result["stage_payload"]["writer_rank"] == 0
        assert result["checkpoint_read_count"] == 1
        assert result["checkpoint_digest"] == expected_digest
        assert result["resume_suffix_matches"] is True
        assert result["resume_next_batch_idx"] == 2
        assert result["resume_next_rank_local_sample_offset"] == 4
        assert result["resume_optimizer_steps_in_epoch"] == 2
        assert result["resume_optimizer_steps_completed"] == 106
    assert (result_dir / "rank0_stage.txt").read_text(encoding="utf-8") == "rank0-only"


def test_resume_authorization_binds_path_digest_identity_and_commit(monkeypatch, tmp_path):
    launcher = load_launcher()
    output = tmp_path / "output"
    output.mkdir()
    contracts = tmp_path / "contracts"
    claim = contracts / "archive" / "claim"
    records = claim / "checkpoints"
    records.mkdir(parents=True)
    runtime_contract = {"runtime_policy_version": launcher.RUNTIME_POLICY_VERSION}
    launch_origin = {"controller": "archived"}
    boundary = {
        "micro_batches_in_window": 0,
        "optimizer_steps_in_epoch": 67783,
        "optimizer_steps_completed": 271132,
        "sampler_epoch": 3,
        "next_batch_idx": None,
        "next_rank_local_sample_offset": 135566,
    }
    checkpoint_path = output / "latest.pt"
    torch.save({"payload": "checkpoint"}, checkpoint_path)
    digest = sha256(checkpoint_path)
    record = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_sha256": digest,
        "checkpoint_lineage": launcher.CHECKPOINT_LINEAGE,
        "epoch": 3,
        "batch_idx": None,
        "optimizer_boundary": boundary,
        "runtime_contract": runtime_contract,
        "launch_origin": launch_origin,
    }
    (claim / "contract.json").write_text(
        json.dumps({"runtime_contract": runtime_contract}), encoding="utf-8"
    )
    (claim / "owner.json").write_text(
        json.dumps({"output_dir": str(output.resolve())}), encoding="utf-8"
    )
    (records / f"{digest}_{checkpoint_path.name}.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    monkeypatch.setattr(launcher, "OUTPUT_DIR", output)
    monkeypatch.setattr(launcher, "CONTRACTS_DIR", contracts)
    monkeypatch.setattr(launcher, "validate_ddp_provenance", lambda value: value)
    monkeypatch.setattr(launcher, "validate_archived_provenance", lambda *args: None)
    monkeypatch.setattr(
        launcher,
        "verify_checkpoint_dict",
        lambda *args: {
            "epoch": 3,
            "batch_idx": None,
            "optimizer_boundary": boundary,
            "launch_origin": launch_origin,
            "runtime_contract": runtime_contract,
        },
    )

    reference = launcher.controller_resume_reference(checkpoint_path, runtime_contract)
    receipt, payload = launcher.verify_resume(
        checkpoint_path, {}, runtime_contract, reference
    )
    assert payload == {"payload": "checkpoint"}
    assert receipt["controller_authorization_verified"] is True
    assert receipt["file_identity_verified"] == reference["file_identity"]

    original = checkpoint_path.stat()
    os.utime(
        checkpoint_path,
        ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000),
    )
    with pytest.raises(RuntimeError, match="differs from controller authorization"):
        launcher.verify_resume(checkpoint_path, {}, runtime_contract, reference)

    copied_path = output / "best.pt"
    copied_path.write_bytes(checkpoint_path.read_bytes())
    with pytest.raises(RuntimeError, match="absent from archived matching DDP3 commit records"):
        launcher.controller_resume_reference(copied_path, runtime_contract)


def test_stale_dead_controller_claim_is_archived_before_resume_lookup(monkeypatch, tmp_path):
    launcher = load_launcher()
    active = tmp_path / "claim.active"
    active.mkdir()
    owner = {"status": "active", "pid": 1234}
    (active / "owner.json").write_text(json.dumps(owner), encoding="utf-8")
    archived = tmp_path / "archive" / "stale"
    calls = []
    monkeypatch.setattr(launcher, "ACTIVE_CLAIM", active)
    monkeypatch.setattr(launcher, "claim_owner_alive", lambda value: False)
    monkeypatch.setattr(
        launcher,
        "archive_claim",
        lambda path, reason: calls.append((path, reason)) or archived,
    )
    assert launcher.reconcile_stale_controller_claim() == archived
    assert calls == [(active, "stale_controller_reconciled")]

    monkeypatch.setattr(launcher, "claim_owner_alive", lambda value: True)
    with pytest.raises(RuntimeError, match="claim is live"):
        launcher.reconcile_stale_controller_claim()


def test_collective_next_batch_handles_batch_end_error_and_divergence(monkeypatch):
    def abort(message):
        raise RuntimeError(message)

    namespace = load_trainer_functions(
        "collective_next_batch",
        namespace={"ABORT_DISTRIBUTED_JOB": abort},
    )
    function = namespace["collective_next_batch"]
    monkeypatch.setattr(dist, "get_world_size", lambda: 3)

    def gather_same(output, state):
        output[:] = [dict(state) for _ in range(3)]

    monkeypatch.setattr(dist, "all_gather_object", gather_same)
    has_batch, batch = function(iter([{"sample": 1}]), torch.device("cpu"))
    assert has_batch is True and batch == {"sample": 1}
    assert function(iter([]), torch.device("cpu")) == (False, None)

    class BrokenIterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise OSError("loader worker failed")

    with pytest.raises(RuntimeError, match="DataLoader states diverged"):
        function(BrokenIterator(), torch.device("cpu"))

    def gather_diverged(output, state):
        output[:] = [
            {"status": "batch", "error": None},
            {"status": "end", "error": None},
            {"status": "batch", "error": None},
        ]

    monkeypatch.setattr(dist, "all_gather_object", gather_diverged)
    with pytest.raises(RuntimeError, match="DataLoader states diverged"):
        function(iter([{"sample": 1}]), torch.device("cpu"))


def test_amp_overflow_uses_found_inf_collective_backoff_before_state_advance():
    namespace = load_trainer_functions(
        "synchronized_scaler_scale",
        "collective_scaler_backoff",
        "scaler_found_inf",
        namespace={"math": __import__("math")},
    )
    namespace["GLOBAL_REDUCE_SCALARS"] = (
        lambda values, operations, device: dict(values)
    )

    class FakeScaler:
        def __init__(self):
            self.scale = 8.0
            self.updated = []
            self.found_inf = {torch.device("cpu"): torch.tensor(1.0)}

        def get_scale(self):
            return self.scale

        def get_backoff_factor(self):
            return 0.5

        def update(self, new_scale=None):
            self.scale = float(new_scale)
            self.updated.append(self.scale)

        def _found_inf_per_device(self, optimizer):
            return self.found_inf

    scaler = FakeScaler()
    assert namespace["scaler_found_inf"](scaler, object()) is True
    assert namespace["collective_scaler_backoff"](
        scaler, torch.device("cpu")
    ) == 4.0
    assert scaler.updated == [4.0]

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    real_scaler = torch.amp.GradScaler("cpu", init_scale=8.0, backoff_factor=0.5)
    real_scaler.scale(parameter * torch.tensor(float("inf"))).backward()
    real_scaler.unscale_(optimizer)
    assert namespace["scaler_found_inf"](real_scaler, optimizer) is True
    assert namespace["collective_scaler_backoff"](
        real_scaler, torch.device("cpu")
    ) == 4.0

    source = (BASE / "sparse_trainer_ic_quality.py").read_text(encoding="utf-8")
    train_source = source[source.index("def train_one_epoch("):source.index("@torch.no_grad()")]
    assert train_source.index("scaler.unscale_(optimizer)") < train_source.index(
        "scaler_found_inf(scaler, optimizer)"
    )
    found_inf_position = train_source.index("scaler_found_inf(scaler, optimizer)")
    scaler_step_position = train_source.index("scaler.step(optimizer)", found_inf_position)
    scheduler_step_position = train_source.index("scheduler.step()", scaler_step_position)
    counter_position = train_source.index("optimizer_steps += 1", scheduler_step_position)
    assert found_inf_position < scaler_step_position < scheduler_step_position < counter_position


def test_sparse_validator_is_hash_locked_and_runs_cpu_only(monkeypatch, tmp_path):
    import types

    validator = tmp_path / "tools" / "validate_sparse_tensor_contract.py"
    validator.parent.mkdir()
    validator.write_text("print('validator')\n", encoding="utf-8")
    expected_hash = sha256(validator)
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    namespace = load_trainer_functions(
        "run_sparse_tensor_contract_preflight",
        namespace={
            "project_root": tmp_path,
            "QUALITY_SPARSE_VALIDATOR_SHA256": expected_hash,
            "sha256_file": sha256,
            "os": os,
            "sys": sys,
            "subprocess": types.SimpleNamespace(run=fake_run),
        },
    )
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("SPARSEVOXELDET_DDP3_TOKEN", "secret")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    namespace["run_sparse_tensor_contract_preflight"](
        tmp_path / "sparse", ["canonical_val_train"]
    )
    environment = captured["kwargs"]["env"]
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert "RANK" not in environment
    assert "SPARSEVOXELDET_DDP3_TOKEN" not in environment
    assert captured["command"][1] == str(validator.resolve())

    namespace["QUALITY_SPARSE_VALIDATOR_SHA256"] = "0" * 64
    with pytest.raises(RuntimeError, match="validator source drift"):
        namespace["run_sparse_tensor_contract_preflight"](
            tmp_path / "sparse", ["canonical_val_train"]
        )


def test_validation_stop_probe_tracks_the_controller_request_file(tmp_path):
    namespace = load_trainer_functions(
        "ValidationStopRequested",
        "validation_stop_requested",
        "validate",
    )
    function = namespace["validation_stop_requested"]
    stop_path = tmp_path / "stop_request.json"
    assert function(None) is False
    assert function(stop_path) is False
    stop_path.write_text('{"status":"stop_requested"}\n', encoding="utf-8")
    assert function(stop_path) is True

    class StopAwareModel:
        input_size = (720, 1280)

        def eval(self):
            return self

        def set_decode_params(self, **kwargs):
            return None

    with pytest.raises(namespace["ValidationStopRequested"], match="during validation"):
        namespace["validate"](
            StopAwareModel(),
            [{"batch_size": 1}],
            object(),
            torch.device("cpu"),
            {"model": {"input_size": [720, 1280]}, "eval": {}},
            compute_map=False,
            stop_request_path=stop_path,
        )
