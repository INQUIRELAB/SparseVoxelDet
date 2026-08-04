#!/usr/bin/env python3
"""
Comprehensive profiling of SparseVoxelDet inference.

Measures:
  1. Per-component wall-clock time (CUDA Events) — most reliable
  2. torch.profiler CUDA kernel-level breakdown — captures spconv custom kernels via CUPTI
  3. Memory profiling — weights vs activations vs peak
  4. Exports Chrome trace for visual inspection

Usage:
  # Basic profiling on a few samples:
  CUDA_VISIBLE_DEVICES=6 python tools/profile_sparse_voxel_det.py \
      --checkpoint runs/sparse_voxel_det/v83_640_seed42/best.pt \
      --num-samples 20 --warmup 5

  # Full profiler trace (exportable to Chrome):
  CUDA_VISIBLE_DEVICES=6 python tools/profile_sparse_voxel_det.py \
      --checkpoint runs/sparse_voxel_det/v83_640_seed42/best.pt \
      --num-samples 10 --warmup 5 --export-trace profile_trace.json

  # With the CUDA fix wrapper:
  sudo unshare --mount bash -c 'touch /tmp/.cuda_fix_empty; \
      mount --bind /tmp/.cuda_fix_empty /dev/nvidia8 2>/dev/null; \
      mount --bind /tmp/.cuda_fix_empty /dev/nvidia9 2>/dev/null; \
      exec sudo -u yazan -H env CUDA_VISIBLE_DEVICES=6 HOME=/home/yazan \
      /home/yazan/Projects/Fred_Project/venv/bin/python \
      /home/yazan/Projects/Fred_Project/tools/profile_sparse_voxel_det.py \
      --checkpoint /home/yazan/Projects/Fred_Project/runs/sparse_voxel_det/v83_640_seed42/best.pt \
      --num-samples 20 --warmup 5'
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.cuda

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import spconv.pytorch as spconv
from torch.utils.data import DataLoader

from sparse_fcos_v1.scripts.sparse_event_dataset_v82 import (
    SparseEventDataset,
    create_sparse_tensor,
    make_collate_fn,
)
from V2.models.sparse_voxel_det_v82 import SparseVoxelDet


# ──────────────────────────────────────────────────────────────────────────────
# 1. Model loading (mirrors evaluate script)
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> Tuple[SparseVoxelDet, dict]:
    """Load model from checkpoint, preferring EMA weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "ema_state_dict" in checkpoint:
        state_dict = checkpoint["ema_state_dict"]["shadow"]
        print(f"  Using EMA shadow weights")
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"  Using raw model weights (no EMA)")
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    ckpt_config = checkpoint.get("config", {})
    model_config = ckpt_config.get("model", {})
    sparse_config = ckpt_config.get("sparse", {})

    input_size = model_config.get("input_size", [640, 640])
    if isinstance(input_size, int):
        input_size = [input_size, input_size]

    model = SparseVoxelDet(
        in_channels=model_config.get("in_channels", 6),
        num_classes=model_config.get("num_classes", 1),
        backbone_size=model_config.get("backbone_size", "nano_deep"),
        fpn_channels=model_config.get("fpn_channels", 128),
        head_convs=model_config.get("head_convs", 2),
        input_size=tuple(input_size),
        time_bins=sparse_config.get("time_bins", 16),
    ).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, ckpt_config


def get_dataloader(ckpt_config: dict, num_samples: int) -> DataLoader:
    """Create a small dataloader for profiling."""
    model_config = ckpt_config.get("model", {})
    sparse_config = ckpt_config.get("sparse", {})
    data_config = ckpt_config.get("data", {})

    input_size = model_config.get("input_size", [640, 640])
    if isinstance(input_size, int):
        input_size = [input_size, input_size]
    time_bins = sparse_config.get("time_bins", 16)
    max_voxels = int(sparse_config.get("max_voxels", 100000))

    # Determine data dirs from config or use defaults
    data_dir = data_config.get("data_dir", str(project_root / "data/datasets/fred_paper_parity_v82/sparse"))
    label_dir = data_config.get("label_dir", str(project_root / "data/datasets/fred_paper_parity/labels"))

    # Auto-detect 640 variant
    if input_size[0] == 640 and input_size[1] == 640:
        candidate = project_root / "data/datasets/fred_paper_parity_v82_640/sparse"
        if candidate.exists():
            data_dir = str(candidate)

    dataset = SparseEventDataset(
        sparse_dir=data_dir,
        label_dir=label_dir,
        split="canonical_test",
        time_bins=time_bins,
        target_size=(int(input_size[0]), int(input_size[1])),
        augment=False,
        max_voxels=max_voxels,
        voxel_sampling={"mode": "random"},
    )

    # Subsample evenly
    if num_samples < len(dataset.samples):
        indices = np.linspace(0, len(dataset.samples) - 1, num=num_samples, dtype=np.int64)
        dataset.samples = [dataset.samples[int(i)] for i in indices]

    collate_fn = make_collate_fn(
        time_bins=time_bins,
        base_size=(int(input_size[0]), int(input_size[1])),
    )
    return DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=collate_fn, pin_memory=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. CUDA Events timing — per-component breakdown
# ──────────────────────────────────────────────────────────────────────────────

class CUDATimer:
    """Accumulates CUDA-event-based timings across multiple calls."""

    def __init__(self):
        self.records: Dict[str, List[float]] = defaultdict(list)

    def time_section(self, name: str):
        """Context manager that times a section using CUDA events."""
        return _CUDATimerSection(self, name)

    def summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for name, times in self.records.items():
            arr = np.array(times)
            out[name] = {
                "mean_ms": float(arr.mean()),
                "std_ms": float(arr.std()),
                "min_ms": float(arr.min()),
                "max_ms": float(arr.max()),
                "median_ms": float(np.median(arr)),
                "n": len(arr),
            }
        return out


class _CUDATimerSection:
    def __init__(self, timer: CUDATimer, name: str):
        self.timer = timer
        self.name = name
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()
        self.timer.records[self.name].append(self.start.elapsed_time(self.end))


def profile_per_component(
    model: SparseVoxelDet,
    dataloader: DataLoader,
    device: torch.device,
    warmup: int = 5,
    num_samples: int = 20,
) -> Dict:
    """Time each component with CUDA events. Most reliable wall-clock measurement."""
    from V2.models.sparse_voxel_det_v82 import sparse_temporal_pool

    timer = CUDATimer()
    sample_stats = []
    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            sparse_input = create_sparse_tensor(batch, device)
            n_voxels = sparse_input.features.shape[0]
            is_warmup = batch_idx < warmup

            # Full end-to-end
            with timer.time_section("0_total" if not is_warmup else "_warmup_total"):
                # Backbone
                with timer.time_section("1_backbone" if not is_warmup else "_warmup_backbone"):
                    backbone_features = model.backbone(sparse_input)

                # FPN
                with timer.time_section("2_fpn" if not is_warmup else "_warmup_fpn"):
                    fused = model.fpn(backbone_features)

                # Temporal pool
                with timer.time_section("3_temporal_pool" if not is_warmup else "_warmup_temporal_pool"):
                    features_2d, indices_2d, spatial_2d = sparse_temporal_pool(
                        fused, mode=model.temporal_pool_mode,
                    )

                # Head
                with timer.time_section("4_head" if not is_warmup else "_warmup_head"):
                    cls_logits, box_ltrb, ctr_logits = model.head(features_2d)

                # Decode + NMS
                with timer.time_section("5_decode_nms" if not is_warmup else "_warmup_decode_nms"):
                    bs = int(sparse_input.indices[:, 0].max().item()) + 1
                    detections = model._decode_detections(
                        cls_logits, box_ltrb, ctr_logits, indices_2d, bs,
                    )

            if not is_warmup:
                n_2d_positions = features_2d.shape[0]
                n_dets = int((detections[:, :, 4] > 0).sum().item())
                sample_stats.append({
                    "n_input_voxels": n_voxels,
                    "n_2d_positions": n_2d_positions,
                    "n_detections": n_dets,
                })

            count += 1
            if count >= warmup + num_samples:
                break

    # Filter out warmup entries
    timing = timer.summary()
    real_timing = {k: v for k, v in timing.items() if not k.startswith("_warmup")}

    return {
        "timing": real_timing,
        "sample_stats": sample_stats,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. torch.profiler — CUDA kernel-level breakdown
# ──────────────────────────────────────────────────────────────────────────────

def profile_torch_profiler(
    model: SparseVoxelDet,
    dataloader: DataLoader,
    device: torch.device,
    warmup: int = 3,
    active: int = 5,
    export_trace: Optional[str] = None,
) -> str:
    """
    Use torch.profiler to capture CUDA kernel-level timing.

    spconv 2.x launches CUDA kernels via cumm (code-gen library). These kernels
    are visible to CUPTI (which torch.profiler uses internally), so they WILL
    appear in the trace — but with raw kernel names like:
      - `implicit_gemm_...`  (sparse conv forward)
      - `gather_...` / `scatter_...`  (index operations)
      - `sortPairsKernel` (spconv internal sorting)

    The Chrome trace (--export-trace) lets you visually inspect the GPU timeline
    and see exactly where spconv kernels sit relative to PyTorch ops.
    """
    batches = []
    for i, batch in enumerate(dataloader):
        batches.append(batch)
        if i >= warmup + active:
            break

    schedule = torch.profiler.schedule(wait=0, warmup=warmup, active=active, repeat=1)

    trace_path = export_trace
    on_trace_ready = None
    if trace_path:
        on_trace_ready = torch.profiler.tensorboard_trace_handler(str(Path(trace_path).parent))

    profiler_kwargs = dict(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    if on_trace_ready:
        profiler_kwargs["on_trace_ready"] = on_trace_ready

    results_text = []

    with torch.profiler.profile(**profiler_kwargs) as prof:
        for step, batch in enumerate(batches):
            if step >= warmup + active:
                break
            sparse_input = create_sparse_tensor(batch, device)
            with torch.no_grad():
                output = model(sparse_input, batch["batch_size"])
            prof.step()

    # Export raw Chrome trace
    if export_trace:
        prof.export_chrome_trace(export_trace)
        results_text.append(f"\nChrome trace exported to: {export_trace}")
        results_text.append("Open in chrome://tracing or https://ui.perfetto.dev/")

    # CUDA time table (sorted by total CUDA time)
    results_text.append("\n" + "=" * 100)
    results_text.append("CUDA KERNEL TIME (sorted by cuda_time_total)")
    results_text.append("=" * 100)
    table = prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=40
    )
    results_text.append(table)

    # CPU time table
    results_text.append("\n" + "=" * 100)
    results_text.append("CPU TIME (sorted by cpu_time_total)")
    results_text.append("=" * 100)
    table_cpu = prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=30
    )
    results_text.append(table_cpu)

    # Self CUDA time (shows actual kernel execution, not including children)
    results_text.append("\n" + "=" * 100)
    results_text.append("SELF CUDA TIME (actual kernel execution)")
    results_text.append("=" * 100)
    table_self = prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=30
    )
    results_text.append(table_self)

    # Memory table
    results_text.append("\n" + "=" * 100)
    results_text.append("CUDA MEMORY (sorted by self_cuda_memory_usage)")
    results_text.append("=" * 100)
    table_mem = prof.key_averages().table(
        sort_by="self_cuda_memory_usage", row_limit=30
    )
    results_text.append(table_mem)

    # Group by input shape to identify spconv kernels
    results_text.append("\n" + "=" * 100)
    results_text.append("GROUPED BY INPUT SHAPE (helps identify spconv ops)")
    results_text.append("=" * 100)
    table_shape = prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total", row_limit=30
    )
    results_text.append(table_shape)

    return "\n".join(results_text)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Memory profiling
# ──────────────────────────────────────────────────────────────────────────────

def profile_memory(
    model: SparseVoxelDet,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict:
    """Detailed memory profiling: weights, activations, peak."""
    from V2.models.sparse_voxel_det_v82 import sparse_temporal_pool

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # Measure weight memory
    weight_mem = sum(
        p.element_size() * p.nelement() for p in model.parameters()
    )
    weight_mem_mb = weight_mem / (1024 * 1024)

    # Measure buffer memory
    buffer_mem = sum(
        b.element_size() * b.nelement() for b in model.buffers()
    )
    buffer_mem_mb = buffer_mem / (1024 * 1024)

    # Baseline GPU memory (after model loaded)
    torch.cuda.synchronize()
    baseline_alloc = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)

    # Run one forward pass and track memory at each stage
    batch = next(iter(dataloader))
    sparse_input = create_sparse_tensor(batch, device)
    n_voxels = sparse_input.features.shape[0]

    torch.cuda.reset_peak_memory_stats(device)
    memory_stages = {}

    with torch.no_grad():
        torch.cuda.synchronize()
        memory_stages["0_input"] = {
            "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        }

        backbone_features = model.backbone(sparse_input)
        torch.cuda.synchronize()
        memory_stages["1_after_backbone"] = {
            "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        }

        fused = model.fpn(backbone_features)
        torch.cuda.synchronize()
        memory_stages["2_after_fpn"] = {
            "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        }

        features_2d, indices_2d, spatial_2d = sparse_temporal_pool(
            fused, mode=model.temporal_pool_mode,
        )
        torch.cuda.synchronize()
        memory_stages["3_after_temporal_pool"] = {
            "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        }

        cls_logits, box_ltrb, ctr_logits = model.head(features_2d)
        torch.cuda.synchronize()
        memory_stages["4_after_head"] = {
            "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        }

        bs = int(sparse_input.indices[:, 0].max().item()) + 1
        detections = model._decode_detections(
            cls_logits, box_ltrb, ctr_logits, indices_2d, bs,
        )
        torch.cuda.synchronize()
        memory_stages["5_after_decode"] = {
            "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        }

    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)

    # Detailed memory stats
    stats = torch.cuda.memory_stats(device)

    return {
        "weight_memory_mb": weight_mem_mb,
        "buffer_memory_mb": buffer_mem_mb,
        "model_total_mb": weight_mem_mb + buffer_mem_mb,
        "baseline_allocated_mb": baseline_alloc / (1024**2),
        "baseline_reserved_mb": baseline_reserved / (1024**2),
        "peak_allocated_mb": peak_allocated,
        "peak_reserved_mb": peak_reserved,
        "activation_peak_mb": peak_allocated - baseline_alloc / (1024**2),
        "n_input_voxels": n_voxels,
        "n_2d_positions": features_2d.shape[0],
        "memory_stages": memory_stages,
        "num_alloc_retries": stats.get("num_alloc_retries", 0),
        "num_ooms": stats.get("num_ooms", 0),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Throughput measurement
# ──────────────────────────────────────────────────────────────────────────────

def profile_throughput(
    model: SparseVoxelDet,
    dataloader: DataLoader,
    device: torch.device,
    warmup: int = 5,
    num_samples: int = 50,
) -> Dict:
    """Measure raw FPS with CUDA events (no data loading overhead)."""
    # Pre-load batches to eliminate data loading from measurement
    batches = []
    for i, batch in enumerate(dataloader):
        batches.append(batch)
        if i >= warmup + num_samples:
            break

    times_ms = []

    with torch.no_grad():
        for i, batch in enumerate(batches):
            sparse_input = create_sparse_tensor(batch, device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            output = model(sparse_input, batch["batch_size"])
            end.record()
            torch.cuda.synchronize()

            if i >= warmup:
                times_ms.append(start.elapsed_time(end))

    arr = np.array(times_ms)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "median_ms": float(np.median(arr)),
        "fps": float(1000.0 / arr.mean()),
        "fps_p95": float(1000.0 / np.percentile(arr, 95)),
        "n_measured": len(arr),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Profile SparseVoxelDet inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples to profile")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--export-trace", type=str, default=None,
                        help="Export Chrome trace to this path (e.g. profile_trace.json)")
    parser.add_argument("--skip-torch-profiler", action="store_true",
                        help="Skip torch.profiler (faster, only CUDA events + memory)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save numeric results to JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("ERROR: CUDA not available. Profiling requires GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_mem / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"PyTorch: {torch.__version__}")
    print(f"spconv: {spconv.__version__}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, ckpt_config = load_model(args.checkpoint, device)
    params = model.get_num_params()
    print(f"  Parameters: {params['total']:,} total ({params['trainable']:,} trainable)")
    print(f"    Backbone: {params['backbone']:,}")
    print(f"    FPN:      {params['fpn']:,}")
    print(f"    Head:     {params['head']:,}")
    print(f"  Input size: {model.input_size}")
    print(f"  Time bins:  {model.time_bins}")

    # Create dataloader
    total_needed = args.warmup + args.num_samples + 5
    print(f"\nLoading {total_needed} test samples...")
    dataloader = get_dataloader(ckpt_config, total_needed)
    print(f"  Loaded {len(dataloader.dataset)} samples")

    # ── Component timing ──
    print("\n" + "=" * 80)
    print("COMPONENT TIMING (CUDA Events)")
    print("=" * 80)
    component_results = profile_per_component(
        model, dataloader, device,
        warmup=args.warmup, num_samples=args.num_samples,
    )

    timing = component_results["timing"]
    total_mean = timing.get("0_total", {}).get("mean_ms", 0)
    print(f"\n{'Component':<25} {'Mean (ms)':>10} {'Std':>8} {'Min':>8} {'Max':>8} {'% Total':>8}")
    print("-" * 70)
    for name in sorted(timing.keys()):
        t = timing[name]
        pct = (t["mean_ms"] / total_mean * 100) if total_mean > 0 else 0
        print(f"{name:<25} {t['mean_ms']:>10.3f} {t['std_ms']:>8.3f} {t['min_ms']:>8.3f} {t['max_ms']:>8.3f} {pct:>7.1f}%")

    # Sample statistics
    stats = component_results["sample_stats"]
    if stats:
        voxels = [s["n_input_voxels"] for s in stats]
        positions = [s["n_2d_positions"] for s in stats]
        print(f"\n  Input voxels:   mean={np.mean(voxels):.0f}  min={np.min(voxels)}  max={np.max(voxels)}")
        print(f"  2D positions:   mean={np.mean(positions):.0f}  min={np.min(positions)}  max={np.max(positions)}")

    # ── Memory profiling ──
    print("\n" + "=" * 80)
    print("MEMORY PROFILING")
    print("=" * 80)
    # Recreate dataloader (iterator was consumed)
    dataloader = get_dataloader(ckpt_config, total_needed)
    mem_results = profile_memory(model, dataloader, device)

    print(f"\n  Model weights:     {mem_results['weight_memory_mb']:.2f} MB")
    print(f"  Model buffers:     {mem_results['buffer_memory_mb']:.2f} MB")
    print(f"  Model total:       {mem_results['model_total_mb']:.2f} MB")
    print(f"  Peak allocated:    {mem_results['peak_allocated_mb']:.2f} MB")
    print(f"  Peak reserved:     {mem_results['peak_reserved_mb']:.2f} MB")
    print(f"  Activation peak:   {mem_results['activation_peak_mb']:.2f} MB (above model baseline)")
    print(f"  Input voxels:      {mem_results['n_input_voxels']:,}")
    print(f"  2D positions:      {mem_results['n_2d_positions']:,}")

    print(f"\n  Memory at each stage:")
    for stage, m in sorted(mem_results["memory_stages"].items()):
        print(f"    {stage:<30} alloc={m['allocated_mb']:.1f} MB  reserved={m['reserved_mb']:.1f} MB")

    # ── Throughput ──
    print("\n" + "=" * 80)
    print("THROUGHPUT (batch_size=1, CUDA Events)")
    print("=" * 80)
    dataloader = get_dataloader(ckpt_config, total_needed)
    throughput = profile_throughput(
        model, dataloader, device,
        warmup=args.warmup, num_samples=args.num_samples,
    )
    print(f"\n  Mean latency:  {throughput['mean_ms']:.2f} ± {throughput['std_ms']:.2f} ms")
    print(f"  Median:        {throughput['median_ms']:.2f} ms")
    print(f"  Min/Max:       {throughput['min_ms']:.2f} / {throughput['max_ms']:.2f} ms")
    print(f"  FPS (mean):    {throughput['fps']:.1f}")
    print(f"  FPS (p95):     {throughput['fps_p95']:.1f}")

    # ── torch.profiler ──
    if not args.skip_torch_profiler:
        print("\n" + "=" * 80)
        print("TORCH PROFILER (CUDA kernel-level)")
        print("=" * 80)
        print("(spconv kernels appear as implicit_gemm_*, gather_*, scatter_*, sortPairs*)")
        dataloader = get_dataloader(ckpt_config, args.warmup + 10)
        profiler_output = profile_torch_profiler(
            model, dataloader, device,
            warmup=args.warmup, active=5,
            export_trace=args.export_trace,
        )
        print(profiler_output)

    # ── Save results ──
    if args.output_json:
        results = {
            "gpu": gpu_name,
            "model_params": params,
            "input_size": list(model.input_size),
            "component_timing": component_results["timing"],
            "sample_stats": component_results["sample_stats"],
            "memory": {k: v for k, v in mem_results.items() if k != "memory_stages"},
            "memory_stages": mem_results["memory_stages"],
            "throughput": throughput,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")

    # ── Additional profiling tips ──
    print("\n" + "=" * 80)
    print("ADDITIONAL PROFILING OPTIONS")
    print("=" * 80)
    print("""
  1. NVIDIA Nsight Systems (most detailed GPU timeline):
     nsys profile -o profile_report --trace=cuda,nvtx \\
       python tools/profile_sparse_voxel_det.py --checkpoint <ckpt> --skip-torch-profiler
     nsys-ui profile_report.nsys-rep  # Visual timeline

  2. NVIDIA Nsight Compute (per-kernel analysis — compute vs memory bound):
     ncu --set full -o ncu_report \\
       python tools/profile_sparse_voxel_det.py --checkpoint <ckpt> --num-samples 1 --warmup 1
     ncu-ui ncu_report.ncu-rep  # Roofline analysis, occupancy, etc.

  3. nvidia-smi dmon (real-time GPU utilization during inference):
     nvidia-smi dmon -s pucvmet -d 1  # In another terminal during profiling

  4. Chrome trace viewer:
     Use --export-trace profile.json, then open in:
       - chrome://tracing
       - https://ui.perfetto.dev/
""")


if __name__ == "__main__":
    main()
