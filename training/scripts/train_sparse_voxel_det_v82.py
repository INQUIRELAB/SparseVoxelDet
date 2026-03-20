#!/usr/bin/env python3
"""
Training script for SparseVoxelDet — Fully Sparse Event-Based Detector.

Usage:
    python training/scripts/train_sparse_voxel_det_v82.py --config training/configs/sparse_voxel_det_v83.yaml
"""
import argparse
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
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training.models.sparse_voxel_det_v82 import SparseVoxelDet
from detection.scripts.sparse_event_dataset_v82 import (
    SparseEventDataset, sparse_collate_fn, make_collate_fn, create_sparse_tensor
)
from training.scripts.sparse_voxel_det_loss import SparseVoxelDetLoss
from detection.scripts.ema import ModelEMA
from detection.scripts.metrics import MAPCalculator
from detection.scripts.evaluate_sparse_fcos import temporal_rerank_top1

PARITY_SPLIT_ALLOWLIST = {
    "canonical_train",
    "canonical_test",
    "challenging_train",
    "challenging_test",
}


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
    cmd = [
        sys.executable,
        str(project_root / "tools" / "validate_sparse_tensor_contract.py"),
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

    proc = subprocess.run(cmd, capture_output=True, text=True)
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
    # Default to 3000 steps if neither key is set.
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
    nan_grad_action = str(config.get('training', {}).get('nan_grad_action', 'skip')).lower()
    grad_sanitize_clip = float(config.get('training', {}).get('grad_sanitize_clip', 100.0))
    mem_cleanup_interval = int(config.get('training', {}).get('mem_cleanup_interval', 2000))
    runtime_cfg = config.get('_runtime', {})
    strict_finite_checks = bool(runtime_cfg.get('strict_finite_checks', True))
    abort_on_skip_rate = float(runtime_cfg.get('abort_on_skip_rate', 0.10))
    abort_on_consecutive_nonfinite = int(runtime_cfg.get('abort_on_consecutive_nonfinite', 200))
    is_main_process = bool(runtime_cfg.get('is_main_process', True))
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
    clip_n_samples = 0.0
    clip_raw_sum = 0.0
    clip_kept_sum = 0.0
    clip_fraction_sum = 0.0
    clip_clipped_count = 0.0
    optimizer_steps = 0
    skipped_non_finite = 0
    skipped_non_finite_grad = 0
    skipped_oom = 0
    micro_batches_in_window = 0
    sanitized_grad_steps = 0
    processed_batches = 0
    nonfinite_events = 0
    consecutive_nonfinite = 0
    max_consecutive_nonfinite = 0
    first_nonfinite_batch = None
    rolling_skip = deque(maxlen=500)
    aborted_early = False
    abort_reason = ""
    latest_loss_grad_norms: Dict[str, float] = {}

    if hasattr(loss_fn, "set_epoch"):
        loss_fn.set_epoch(epoch)

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
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
                scaler.update()

        # Periodic memory cleanup to prevent CUDA fragmentation
        if mem_cleanup_interval > 0 and batch_idx > 0 and batch_idx % mem_cleanup_interval == 0:
            import gc; gc.collect()
            torch.cuda.empty_cache()

        try:
            # Create sparse tensor
            sparse_input = create_sparse_tensor(batch, device)
            batch_size = batch['batch_size']

            if strict_finite_checks:
                if not tensors_finite(batch.get("feats")):
                    register_skip("non-finite input features", "input_features", "non_finite")
                    if aborted_early:
                        break
                    continue
                if not tensors_finite(sparse_input.features):
                    register_skip("non-finite sparse tensor features", "sparse_tensor", "non_finite")
                    if aborted_early:
                        break
                    continue

            # Multi-scale: use actual batch input_size (V82: can be (H,W) tuple)
            cur_size = batch.get('input_size', tuple(default_input_size))
            if isinstance(cur_size, (int, float)):
                cur_input_size = [int(cur_size), int(cur_size)]
            elif isinstance(cur_size, (list, tuple)):
                cur_input_size = list(cur_size)
            else:
                cur_input_size = list(default_input_size)

            # Update model input_size for this batch if multi-scale
            raw = model.module if hasattr(model, 'module') else model
            size_key = (cur_input_size[0], cur_input_size[1])
            if size_key != (raw.input_size[0], raw.input_size[1]):
                raw.input_size = tuple(cur_input_size)

            # Get GT boxes and labels
            gt_boxes = [b.to(device) for b in batch['gt_boxes']]
            gt_labels = [l.to(device) for l in batch['gt_labels']]

            # Zero grad only at start of accumulation window.
            if micro_batches_in_window == 0:
                optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast("cuda", dtype=amp_dtype):
                    outputs = model(sparse_input, batch_size, return_loss_inputs=True)
                    losses = loss_fn(outputs, gt_boxes, gt_labels)
                    loss = losses['loss']
            else:
                outputs = model(sparse_input, batch_size, return_loss_inputs=True)
                losses = loss_fn(outputs, gt_boxes, gt_labels)
                loss = losses['loss']

            if strict_finite_checks and not tensors_finite(outputs):
                register_skip("non-finite model outputs", "forward_outputs", "non_finite")
                if aborted_early:
                    break
                continue
            if strict_finite_checks and not tensors_finite(losses):
                register_skip("non-finite loss component", "loss_terms", "non_finite")
                if aborted_early:
                    break
                continue

            if not torch.isfinite(loss):
                print(f"  WARNING: Non-finite loss at batch {batch_idx}, skipping")
                register_skip("non-finite total loss", "loss_total", "non_finite")
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
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)

                if not check_gradients_finite(model):
                    if nan_grad_action == 'sanitize':
                        repaired = sanitize_gradients(model, clip_value=grad_sanitize_clip)
                        if repaired > 0:
                            sanitized_grad_steps += 1
                            print(f"  WARNING: Sanitized {repaired} non-finite gradient values at batch {batch_idx}")
                    if not check_gradients_finite(model):
                        print(f"  WARNING: Non-finite gradients at batch {batch_idx}, skipping")
                        register_skip(
                            "non-finite gradients after sanitize",
                            "gradient_check",
                            "non_finite_grad",
                            reset_scaler=(use_amp and scaler is not None),
                        )
                        if aborted_early:
                            break
                        continue

                grad_norm = compute_gradient_norm(model)
                if grad_norm > 100:
                    print(f"  WARNING: Large gradient norm: {grad_norm:.1f}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer_steps += 1
                if ema is not None:
                    ema.update(model.module if hasattr(model, 'module') else model)
                optimizer.zero_grad(set_to_none=True)
                micro_batches_in_window = 0

            # Update metrics (inside try block, runs for all successful batches)
            total_loss += loss.item() * batch_size
            total_cls_loss += losses['cls_loss'].item() * batch_size
            total_reg_loss += (losses['reg_loss'].item() if isinstance(losses['reg_loss'], torch.Tensor) else losses['reg_loss']) * batch_size
            total_ctr_loss += (losses['ctr_loss'].item() if isinstance(losses['ctr_loss'], torch.Tensor) else losses['ctr_loss']) * batch_size
            total_samples += batch_size
            total_pos += losses.get('num_pos_raw', losses['num_pos']).item()
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
            if batch_idx % log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - start_time
                samples_per_sec = total_samples / elapsed if elapsed > 0 else 0

                print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                      f"loss={loss.item():.4f} cls={losses['cls_loss'].item():.4f} "
                      f"reg={losses['reg_loss']:.4f} ctr={losses['ctr_loss']:.4f} "
                      f"pos={int(losses.get('num_pos_raw', losses['num_pos']).item())} lr={lr:.2e} "
                      f"samples/s={samples_per_sec:.1f}")
                if latest_loss_grad_norms:
                    grad_diag_msg = " ".join(
                        f"{k.replace('_loss_raw', '')}={v:.2f}" for k, v in sorted(latest_loss_grad_norms.items())
                    )
                    print(f"    grad_diag {grad_diag_msg}")

            # --- Mid-epoch checkpoint + heartbeat ---
            if is_main_process and output_dir is not None:
                now = time.time()
                if now - last_ckpt_time >= ckpt_interval_sec:
                    last_ckpt_time = now
                    # Save mid-epoch checkpoint (batch_idx stored for resume)
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, epoch,
                        {'batch_idx': batch_idx, 'loss': loss.item()},
                        output_dir / 'latest.pt', config, ema=ema,
                        batch_idx=batch_idx,
                    )
                    print(f"  [mid-epoch ckpt] Saved at batch {batch_idx}/{len(dataloader)} "
                          f"({batch_idx/len(dataloader)*100:.1f}%)")
                    # Write heartbeat
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

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                print(f"  WARNING: CUDA OOM at batch {batch_idx}, skipping")
                register_skip("cuda oom", "forward", "oom")
                torch.cuda.empty_cache()
                import gc; gc.collect()
                if aborted_early:
                    break
                continue
            raise

    # Optional: flush trailing gradients from a partial accumulation window.
    # Default is disabled for stability after OOM/non-finite cascades.
    has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    if flush_partial_accumulation and micro_batches_in_window > 0 and has_grads:
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            if check_gradients_finite(model):
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer_steps += 1
                if ema is not None:
                    ema.update(model.module if hasattr(model, 'module') else model)
            else:
                skipped_non_finite_grad += 1
        else:
            if check_gradients_finite(model):
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer_steps += 1
                if ema is not None:
                    ema.update(model.module if hasattr(model, 'module') else model)
            else:
                skipped_non_finite_grad += 1
        optimizer.zero_grad(set_to_none=True)
    else:
        # No gradients to flush — just clear any stale state
        optimizer.zero_grad(set_to_none=True)

    # Compute epoch averages (guard against all-NaN epochs)
    if total_samples == 0:
        print("  WARNING: All batches produced NaN loss this epoch!")
        total_samples = 1  # Prevent ZeroDivisionError
    skipped_total = skipped_non_finite + skipped_non_finite_grad + skipped_oom
    skip_rate = skipped_total / max(processed_batches, 1)
    if skipped_total > 0:
        print(f"  Skip summary: nonfinite_loss={skipped_non_finite} "
              f"nonfinite_grad={skipped_non_finite_grad} oom={skipped_oom} "
              f"total={skipped_total}/{processed_batches} ({skip_rate*100:.1f}%)")
    if aborted_early:
        print(f"  ABORT GATE TRIGGERED: {abort_reason}")

    metrics = {
        'loss': total_loss / total_samples,
        'cls_loss': total_cls_loss / total_samples,
        'reg_loss': total_reg_loss / total_samples,
        'ctr_loss': total_ctr_loss / total_samples,
        'iou_quality_loss': total_iouq_loss / total_samples,
        'proposal_loss': total_proposal_loss / total_samples,
        'ranking_loss': total_ranking_loss / total_samples,
        'uncertainty_loss': total_uncertainty_loss / total_samples,
        'positive_query_ratio': total_positive_query_ratio / total_samples,
        'ranking_gap_mean': total_ranking_gap / total_samples,
        'near_boundary_mass_045_055': total_near_boundary_mass / total_samples,
        'proposal_recall_at_16': total_proposal_recall16 / total_samples,
        'proposal_recall_at_32': total_proposal_recall32 / total_samples,
        'proposal_recall_at_64': total_proposal_recall64 / total_samples,
        'proposal_recall_at_128': total_proposal_recall128 / total_samples,
        'avg_pos': total_pos / len(dataloader),
        'lr': optimizer.param_groups[0]['lr'],
        'time': time.time() - start_time,
        'optimizer_steps': optimizer_steps,
        'skipped_non_finite': skipped_non_finite,
        'skipped_non_finite_grad': skipped_non_finite_grad,
        'skipped_oom': skipped_oom,
        'sanitized_grad_steps': sanitized_grad_steps,
        'skipped_total': skipped_total,
        'skip_rate': skip_rate,
        'processed_batches': processed_batches,
        'nonfinite_events': nonfinite_events,
        'first_nonfinite_batch': first_nonfinite_batch if first_nonfinite_batch is not None else -1,
        'max_consecutive_nonfinite': max_consecutive_nonfinite,
        'aborted_early': aborted_early,
        'abort_reason': abort_reason,
        'clip_fraction_mean': clip_fraction_sum / max(clip_n_samples, 1.0),
        'clip_rate': clip_clipped_count / max(clip_n_samples, 1.0),
        'raw_voxels_mean': clip_raw_sum / max(clip_n_samples, 1.0),
        'kept_voxels_mean': clip_kept_sum / max(clip_n_samples, 1.0),
    }

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    config: Dict,
    compute_map: bool = True,
    epoch: Optional[int] = None,
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
):
    """
    Save checkpoint with DDP-safe state dict handling.

    Saves model without 'module.' prefix for portability.
    If batch_idx is set, this is a mid-epoch checkpoint.  On resume the
    training loop will skip batches [0..batch_idx) to avoid repeating work.
    """
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
        'metrics': metrics,
        'config': config,
    }
    # Mid-epoch marker: batch_idx != None means this is a partial-epoch save.
    # Resume will continue from this batch instead of jumping to epoch+1.
    if batch_idx is not None:
        checkpoint['batch_idx'] = batch_idx
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()

    torch.save(checkpoint, path)
    tag = f" (mid-epoch batch {batch_idx})" if batch_idx is not None else ""
    print(f"Saved checkpoint to {path}{tag}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[GradScaler] = None
) -> Dict:
    """
    Load checkpoint with DDP-safe state dict handling.
    """
    checkpoint = torch.load(path, map_location='cpu')

    # Handle DDP 'module.' prefix
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        # Remove 'module.' prefix from DDP checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        print("  Removed 'module.' prefix from DDP checkpoint")

    # Check if current model is DDP wrapped
    if hasattr(model, 'module'):
        load_result = model.module.load_state_dict(state_dict, strict=False)
    else:
        load_result = model.load_state_dict(state_dict, strict=False)

    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    if missing_keys:
        print(f"  WARNING: Missing keys while loading checkpoint: {missing_keys[:8]}{' ...' if len(missing_keys) > 8 else ''}")
    if unexpected_keys:
        print(f"  WARNING: Unexpected keys while loading checkpoint: {unexpected_keys[:8]}{' ...' if len(unexpected_keys) > 8 else ''}")

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
    """Initialize DDP if running with torchrun."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    """Clean up DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Train SparseVoxelDet')
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

    # Setup DDP if using torchrun
    rank, local_rank, world_size = setup_ddp()
    is_main_process = (rank == 0)

    # Load config
    config = load_config(args.config)
    if is_main_process:
        print(f"Loaded config from {args.config}")

    deterministic = bool(config.get("training", {}).get("deterministic", False))
    seed_everything(args.seed + rank, deterministic=deterministic)
    if is_main_process:
        print(f"Seed setup: base_seed={args.seed + rank} deterministic={deterministic}")

    # Setup device
    if world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if is_main_process:
        print(f"Using device: {device} (world_size={world_size})")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = project_root / 'runs' / 'sparse_voxel_det' / timestamp
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
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
        contract_report_json = output_dir / "preflight_sparse_tensor_contract.json" if is_main_process else None
        run_sparse_tensor_contract_preflight(
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

    # DDP or EpochSubsetSampler
    max_samples = train_config.get('max_samples_per_epoch', None)
    if parity_enforced and max_samples is not None:
        raise ValueError(
            "Parity mode forbids max_samples_per_epoch caps. Set training.max_samples_per_epoch: null."
        )
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    elif max_samples and max_samples < len(train_dataset):
        train_sampler = EpochSubsetSampler(len(train_dataset), max_samples)
        if is_main_process:
            print(f"Using EpochSubsetSampler: {max_samples}/{len(train_dataset)} samples/epoch")
    else:
        train_sampler = None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None

    loader_seed = int(args.seed + rank * 100000)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(loader_seed)
    worker_init_fn = make_worker_init_fn(loader_seed)
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=loader_generator,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=loader_generator,
        **loader_kwargs,
    )

    if is_main_process:
        print(f"Train split: {train_split}")
        print(f"Val split: {val_split}")
        print(f"Parity enforced: {parity_enforced}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
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

    # DDP: Wrap model
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    params = model.module.get_num_params() if hasattr(model, 'module') else model.get_num_params()
    if is_main_process:
        print(f"Model parameters: {params['total']:,}")

    # Create optimizer, scheduler, loss
    accum_steps = max(1, train_config.get('gradient_accumulation_steps', 1))
    optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / accum_steps))
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
    start_batch_idx = 0  # For mid-epoch resume
    best_map = 0.0
    best_loss = float('inf')
    if args.weights_only:
        checkpoint = load_weights_only(
            Path(args.weights_only), model, ema, device
        )
        if 'metrics' in checkpoint:
            if 'mAP_50' in checkpoint['metrics']:
                best_map = checkpoint['metrics']['mAP_50']
            if 'val_loss' in checkpoint['metrics']:
                best_loss = checkpoint['metrics']['val_loss']
        if is_main_process:
            print("Starting from epoch 0 with fresh optimizer/scheduler/scaler (weights-only mode)")
    elif args.resume:
        checkpoint = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, scaler
        )
        # Mid-epoch resume: if checkpoint has 'batch_idx', it was a partial save.
        # Resume from the SAME epoch at the next batch, instead of skipping to epoch+1.
        resume_batch_idx = checkpoint.get('batch_idx', None)
        if resume_batch_idx is not None:
            start_epoch = checkpoint['epoch']  # same epoch
            start_batch_idx = resume_batch_idx + 1  # next batch after the one saved
            if is_main_process:
                print(f"Mid-epoch resume: epoch {start_epoch}, batch {start_batch_idx}")
        else:
            start_epoch = checkpoint['epoch'] + 1
            start_batch_idx = 0
        if 'metrics' in checkpoint:
            if 'mAP_50' in checkpoint['metrics']:
                best_map = checkpoint['metrics']['mAP_50']
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
    effective_batch_size = batch_size * accum_steps
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
    }

    # Determine loss type string for logging
    loss_type_str = 'SparseVoxelDet (focal + GIoU + centerness)'

    if is_main_process:
        print(f"\n{'='*60}")
        print(f"Training Configuration Summary")
        print(f"{'='*60}")
        print(f"Epochs: {start_epoch} -> {epochs-1} ({epochs - start_epoch} total)")
        print(f"Batch size: {batch_size} x {accum_steps} accum = {effective_batch_size} effective")
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
            # On mid-epoch resume, skip already-processed batches for the FIRST epoch only.
            epoch_start_batch = start_batch_idx if epoch == start_epoch else 0
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scheduler, loss_fn,
                device, epoch, config, scaler, ema=ema,
                output_dir=output_dir,
                start_batch_idx=epoch_start_batch,
            )

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
                if train_metrics.get("aborted_early", False):
                    print(f"ABORTED EARLY: {train_metrics.get('abort_reason', 'unknown')}")

            if train_metrics.get("aborted_early", False):
                training_aborted = True
                metrics = train_metrics.copy()
                metrics["epoch"] = epoch
                history.append(metrics)
                if is_main_process:
                    with open(output_dir / 'history.json', 'w') as f:
                        json.dump(history, f, indent=2)
                    run_health = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "epoch": epoch,
                        "status": "aborted",
                        "reason": train_metrics.get("abort_reason", ""),
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
                break

            # Validate
            metrics = train_metrics.copy()
            did_validate = False
            current_val_interval = val_interval
            if val_interval_after_epoch is not None and epoch >= int(val_interval_after_epoch):
                current_val_interval = max(1, val_interval_after)
            if (not args.skip_validation) and ((epoch + 1) % current_val_interval == 0 or epoch == epochs - 1):
                did_validate = True
                # Clear GPU cache to prevent OOM during validation
                torch.cuda.empty_cache()

                # Use EMA model for validation if available
                if ema is not None:
                    ema.apply_shadow(raw_model)

                val_metrics = validate(model, val_loader, loss_fn, device, config, epoch=epoch)
                metrics.update(val_metrics)

                if ema is not None:
                    ema.restore(raw_model)

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

                    # Save best model by mAP@50 (primary), fall back to val_loss
                    save_best = False
                    if 'mAP_50' in val_metrics:
                        if val_metrics['mAP_50'] > best_map:
                            best_map = val_metrics['mAP_50']
                            save_best = True
                    elif val_metrics['val_loss'] < best_loss:
                        best_loss = val_metrics['val_loss']
                        save_best = True

                    if save_best:
                        save_checkpoint(
                            model, optimizer, scheduler, scaler, epoch, metrics,
                            output_dir / 'best.pt', config, ema=ema,
                        )

            # Save latest checkpoint (only main process)
            if is_main_process:
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, metrics,
                    output_dir / 'latest.pt', config, ema=ema,
                )

                # Save per-epoch checkpoint
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, metrics,
                    output_dir / f'epoch_{epoch:03d}.pt', config, ema=ema,
                )

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
                    print(f"  Running compact visual audit: {audit_dir}")
                    try:
                        run_kwargs = {"check": True}
                        if audit_timeout_sec > 0:
                            run_kwargs["timeout"] = audit_timeout_sec
                        subprocess.run(audit_cmd, **run_kwargs)
                        print("  Compact visual audit completed")
                    except Exception as e:
                        print(f"  WARNING: Compact visual audit failed: {e}")

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
                    print(f"  Running full-val forensic bundle: {forensic_dir}")
                    try:
                        subprocess.run(forensic_cmd, check=True)
                        casebook_cmd = [
                            sys.executable,
                            str(project_root / "tools" / "build_failure_casebook.py"),
                            "--run-dir", str(output_dir),
                            "--epoch", str(epoch),
                            "--forensic-dir", str(forensic_dir),
                            "--outdir", str(output_dir / "forensics"),
                        ]
                        subprocess.run(casebook_cmd, check=True)
                        print("  Full-val forensic bundle completed")
                    except Exception as e:
                        print(f"  WARNING: Full-val forensic bundle failed: {e}")

            # Record history
            metrics['epoch'] = epoch
            history.append(metrics)

            # Save history (main process only)
            if is_main_process:
                with open(output_dir / 'history.json', 'w') as f:
                    json.dump(history, f, indent=2)
                run_health = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "epoch": epoch,
                    "status": "running",
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

        if is_main_process:
            if training_aborted:
                print("\nTraining stopped by recovery abort gates.")
            else:
                print(f"\nTraining complete!")
            if best_map > 0:
                print(f"Best mAP@50: {best_map:.4f}")
            print(f"Best validation loss: {best_loss:.4f}")
            print(f"Checkpoints saved to: {output_dir}")

    finally:
        # Cleanup DDP
        cleanup_ddp()


if __name__ == '__main__':
    main()
