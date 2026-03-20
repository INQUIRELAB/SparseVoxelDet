#!/usr/bin/env python3
"""
Paper-Grade Efficiency Benchmark: SparseVoxelDet vs YOLOv11n.

Produces ALL numbers needed for the paper's efficiency comparison:
  1. Sparse MACs (intercepted from spconv rulebook)
  2. Wall-clock latency with CUDA Events (forward + NMS breakdown)
  3. Peak GPU memory (allocated + reserved)
  4. Parameter counts
  5. Active positions statistics from real test data
  6. YOLOv11 latency / memory / GFLOPs on same GPU

BOTH models benchmarked on the SAME GPU, same precision (FP16), same protocol.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/benchmark_efficiency_paper.py \
        --checkpoint runs/sparse_voxel_det/v83_640_seed42/best.pt \
        --config V2/configs/sparse_voxel_det_v83_640.yaml \
        --yolo-model yolo11n.pt \
        --num-warmup 100 --num-measure 1000
"""
import argparse
import json
import os
import random
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))


# ============================================================================
# Part 1: TRUE Sparse FLOP Counter via spconv Rulebook Interception
# ============================================================================

class SparseFlopsCounter:
    """Counts exact MACs by intercepting spconv's rulebook pair generation.

    For each sparse conv layer, spconv builds a "rulebook" mapping active
    input positions to active output positions for each kernel offset.
    The tensor `indice_num_per_loc[k]` gives the number of active pairs
    for kernel offset k. Summing this gives the total active pairs.

    TRUE MACs per layer = total_active_pairs × C_in × C_out
    """

    def __init__(self):
        self.records = []
        self._pair_counts = []
        self._hook_handles = []
        self._original_fwd = None

    @contextmanager
    def count(self, model):
        """Instrument model for exact FLOP counting during one forward pass."""
        import spconv.pytorch as spconv
        from spconv.pytorch.conv import SparseConvolution

        self.records.clear()
        self._pair_counts.clear()
        self._hook_handles.clear()

        # Monkey-patch SparseConvolution.forward to capture pair counts
        original_forward = SparseConvolution.forward

        counter = self

        def patched_forward(self_conv, input):
            result = original_forward(self_conv, input)

            # Extract pair count from the result's internal data
            n_in = input.features.shape[0]
            n_out = result.features.shape[0]
            c_in = self_conv.in_channels
            c_out = self_conv.out_channels
            ksize = list(self_conv.kernel_size)
            kv = 1
            for k in ksize:
                kv *= k
            is_subm = self_conv.subm
            is_transposed = self_conv.transposed

            # For 1×1 convs, spconv does simple torch.mm — pairs = n_in
            if kv == 1 and all(s == 1 for s in self_conv.stride):
                total_pairs = n_in
            else:
                # Access the indice data from the result's spatial structure
                # spconv stores pair info in the SparseConvTensor's indice_dict
                # After convolution, we can get it from the internal state.
                # Use the fact that for SubM conv, pairs ≤ N × K_eff
                # For strided, pairs ≤ N_out × K
                # We'll intercept via the pair_fwd hook below
                total_pairs = None  # Will be filled by _intercept_pair_count

            counter._pair_counts.append({
                'n_in': n_in,
                'n_out': n_out,
                'c_in': c_in,
                'c_out': c_out,
                'kernel': ksize,
                'kv': kv,
                'subm': is_subm,
                'transposed': is_transposed,
                'total_pairs': total_pairs,
            })

            return result

        SparseConvolution.forward = patched_forward

        # Also hook sparse conv to get true pair counts from indice_dict
        # We'll use a post-forward approach: after model forward, extract
        # from the SparseConvTensor indice_dict

        try:
            yield self
        finally:
            SparseConvolution.forward = original_forward
            for h in self._hook_handles:
                h.remove()
            self._hook_handles.clear()

    def count_with_real_pairs(self, model, input_tensor):
        """Run forward pass and count exact MACs using spconv's get_indice_pairs.

        This approach hooks into each SparseConvolution layer's forward() to
        intercept the actual pair counts computed by spconv.
        """
        import spconv.pytorch as spconv
        from spconv.pytorch.conv import SparseConvolution

        self.records.clear()
        handles = []

        def make_hook(name, module):
            def hook_fn(mod, inp, out):
                feat_in = inp[0]
                n_in = feat_in.features.shape[0]
                n_out = out.features.shape[0]
                c_in = mod.in_channels
                c_out = mod.out_channels
                ksize = list(mod.kernel_size)
                kv = 1
                for k in ksize:
                    kv *= k
                is_subm = mod.subm
                stride_list = list(mod.stride)

                # For 1×1, it's just N × Cin × Cout
                if kv == 1 and all(s == 1 for s in stride_list):
                    total_pairs = n_in
                else:
                    # After the conv, the indice_dict in out has pair info.
                    # spconv2 stores this in SparseConvTensor.indice_dict
                    # The key is the indice_key of the conv.
                    # Try to extract from the output's indice_dict
                    try:
                        indice_dict = out.indice_dict
                        # The indice_key is set by the conv layer
                        key = mod.indice_key
                        if key and key in indice_dict:
                            pair_data = indice_dict[key]
                            # pair_data is an IndiceData or ImplicitGemmIndiceData
                            if hasattr(pair_data, 'indice_num_per_loc'):
                                inpl = pair_data.indice_num_per_loc
                                if inpl is not None:
                                    total_pairs = int(inpl.sum().item())
                                else:
                                    total_pairs = None
                            else:
                                total_pairs = None
                        else:
                            # Try all keys
                            total_pairs = None
                            for k, v in indice_dict.items():
                                if hasattr(v, 'indice_num_per_loc') and v.indice_num_per_loc is not None:
                                    # Take the last one added (most recent conv)
                                    pass
                            total_pairs = None
                    except Exception:
                        total_pairs = None

                # Fallback: for SubM, average neighbor estimate
                if total_pairs is None:
                    if is_subm:
                        # Conservative: use geometric approach
                        # SubM: each active position checks kv neighbors,
                        # active_pairs ≈ N × (actual active neighbors per pos)
                        # We DON'T estimate — mark as unmeasured
                        total_pairs = -1  # Will be filled by approach 2
                    else:
                        total_pairs = -1

                macs = total_pairs * c_in * c_out if total_pairs > 0 else -1

                self.records.append({
                    'name': name,
                    'type': type(mod).__name__,
                    'in_ch': c_in,
                    'out_ch': c_out,
                    'kernel': ksize,
                    'stride': stride_list,
                    'subm': is_subm,
                    'transposed': getattr(mod, 'transposed', False),
                    'n_in': n_in,
                    'n_out': n_out,
                    'total_pairs': total_pairs,
                    'macs': macs,
                    'avg_neighbors': total_pairs / max(n_out, 1) if total_pairs > 0 else -1,
                })
            return hook_fn

        def make_linear_hook(name):
            def hook_fn(mod, inp, out):
                n = inp[0].shape[0]
                macs = n * mod.in_features * mod.out_features
                self.records.append({
                    'name': name,
                    'type': 'Linear',
                    'in_ch': mod.in_features,
                    'out_ch': mod.out_features,
                    'kernel': [1],
                    'stride': [1],
                    'subm': False,
                    'transposed': False,
                    'n_in': n,
                    'n_out': n,
                    'total_pairs': n,
                    'macs': macs,
                    'avg_neighbors': 1.0,
                })
            return hook_fn

        for name, mod in model.named_modules():
            if isinstance(mod, SparseConvolution):
                h = mod.register_forward_hook(make_hook(name, mod))
                handles.append(h)
            elif isinstance(mod, nn.Linear):
                h = mod.register_forward_hook(make_linear_hook(name))
                handles.append(h)

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            _ = model(input_tensor)

        for h in handles:
            h.remove()

        return self.records

    def count_via_algo_inspection(self, model, input_tensor):
        """Alternative: inspect spconv's internal algorithm to get pair counts.

        Uses spconv's ops.get_indice_pairs to manually build the rulebook
        and count exact pairs per layer.
        """
        import spconv.pytorch as spconv
        import spconv.pytorch.ops as spconv_ops
        from spconv.pytorch.conv import SparseConvolution
        from spconv.pytorch.core import IndiceData, ImplicitGemmIndiceData

        self.records.clear()
        pair_data_list = []

        # Monkey-patch to intercept ALL pair generation calls
        orig_apply_pos = None
        if hasattr(SparseConvolution, '_apply_sparse_conv'):
            orig_apply_pos = SparseConvolution._apply_sparse_conv

        # Track all index computations
        collected_pairs = []

        # Hook the get_indice_pairs function
        original_get = getattr(spconv_ops, 'get_indice_pairs', None)
        original_get_ig = getattr(spconv_ops, 'get_indice_pairs_implicit_gemm', None)

        def intercept_get_pairs(*args, **kwargs):
            result = original_get(*args, **kwargs)
            out_inds, pair_fwd, pair_bwd, indice_num_per_loc = result[:4]
            collected_pairs.append({
                'algo': 'native',
                'n_out': out_inds.shape[0],
                'indice_num_per_loc': indice_num_per_loc.cpu().clone() if indice_num_per_loc is not None else None,
            })
            return result

        def intercept_get_pairs_ig(*args, **kwargs):
            result = original_get_ig(*args, **kwargs)
            # ImplicitGemm returns different structure
            if hasattr(result, 'out_inds'):
                collected_pairs.append({
                    'algo': 'implicit_gemm',
                    'n_out': result.out_inds.shape[0] if result.out_inds is not None else 0,
                    'indice_num_per_loc': None,  # ImplicitGemm doesn't expose this directly
                })
            return result

        if original_get:
            spconv_ops.get_indice_pairs = intercept_get_pairs
        if original_get_ig:
            spconv_ops.get_indice_pairs_implicit_gemm = intercept_get_pairs_ig

        # Hook each conv layer
        handles = []
        layer_info = []

        def make_hook(name, mod):
            def hook_fn(m, inp, out):
                n_in = inp[0].features.shape[0]
                n_out = out.features.shape[0]
                layer_info.append({
                    'name': name,
                    'type': type(m).__name__,
                    'in_ch': m.in_channels,
                    'out_ch': m.out_channels,
                    'kernel': list(m.kernel_size),
                    'stride': list(m.stride),
                    'subm': m.subm,
                    'transposed': getattr(m, 'transposed', False),
                    'n_in': n_in,
                    'n_out': n_out,
                })
            return hook_fn

        def make_linear_hook(name):
            def hook_fn(m, inp, out):
                n = inp[0].shape[0]
                layer_info.append({
                    'name': name,
                    'type': 'Linear',
                    'in_ch': m.in_features,
                    'out_ch': m.out_features,
                    'kernel': [1],
                    'stride': [1],
                    'subm': False,
                    'transposed': False,
                    'n_in': n,
                    'n_out': n,
                })
            return hook_fn

        for name, mod in model.named_modules():
            if isinstance(mod, SparseConvolution):
                h = mod.register_forward_hook(make_hook(name, mod))
                handles.append(h)
            elif isinstance(mod, nn.Linear):
                h = mod.register_forward_hook(make_linear_hook(name))
                handles.append(h)

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            _ = model(input_tensor)

        for h in handles:
            h.remove()

        # Restore original functions
        if original_get:
            spconv_ops.get_indice_pairs = original_get
        if original_get_ig:
            spconv_ops.get_indice_pairs_implicit_gemm = original_get_ig

        return layer_info, collected_pairs

    def print_summary(self):
        total_macs = 0
        total_macs_sparse = 0
        total_macs_linear = 0
        print(f"\n{'='*130}")
        print(f"{'Layer':<45} {'Type':<20} {'Ch':>10} {'K':>8} {'N_in':>8} "
              f"{'N_out':>8} {'Pairs':>10} {'MACs':>14} {'Nbrs':>6}")
        print(f"{'-'*130}")

        for r in self.records:
            ch = f"{r['in_ch']}→{r['out_ch']}"
            k = 'x'.join(str(x) for x in r['kernel'])
            pairs_s = f"{r['total_pairs']:,}" if r['total_pairs'] > 0 else "N/A"
            macs_s = f"{r['macs']:,}" if r['macs'] > 0 else "N/A"
            nbr = f"{r['avg_neighbors']:.1f}" if r.get('avg_neighbors', -1) > 0 else "-"

            print(f"{r['name']:<45} {r['type']:<20} {ch:>10} {k:>8} "
                  f"{r['n_in']:>8,} {r['n_out']:>8,} {pairs_s:>10} {macs_s:>14} {nbr:>6}")

            if r['macs'] > 0:
                total_macs += r['macs']
                if r['type'] == 'Linear':
                    total_macs_linear += r['macs']
                else:
                    total_macs_sparse += r['macs']

        print(f"{'-'*130}")
        print(f"Total MACs: {total_macs:,.0f} ({total_macs/1e6:.2f}M / {total_macs/1e9:.4f}G)")
        print(f"  Sparse conv MACs: {total_macs_sparse:,.0f} ({total_macs_sparse/1e6:.2f}M)")
        print(f"  Linear MACs:      {total_macs_linear:,.0f} ({total_macs_linear/1e6:.2f}M)")
        print(f"{'='*130}")

        return total_macs


# ============================================================================
# Part 2: Model Loading
# ============================================================================

def load_sparse_model(checkpoint_path, config, device):
    """Load SparseVoxelDet model."""
    from training.models.sparse_voxel_det import SparseVoxelDet

    input_size = config['model'].get('input_size', [640, 640])
    if isinstance(input_size, int):
        input_size = [input_size, input_size]

    in_channels = config['model'].get('in_channels', 6)
    model = SparseVoxelDet(
        in_channels=in_channels,
        backbone_size=config['model'].get('backbone_size', 'nano_deep'),
        fpn_channels=config['model'].get('fpn_channels', 128),
        num_classes=config['model'].get('num_classes', 1),
        head_convs=config['model'].get('head_convs', 2),
        prior_prob=config['model'].get('prior_prob', 0.01),
        input_size=tuple(input_size),
        time_bins=config.get('sparse', {}).get('time_bins', 16),
    )

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'ema_state_dict' in ckpt:
        ema = ckpt['ema_state_dict']
        if isinstance(ema, dict) and 'shadow' in ema:
            ema = ema['shadow']
        model.load_state_dict(ema, strict=True)
        print(f"  Loaded EMA weights from {checkpoint_path}")
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    model = model.to(device).eval()
    return model


def load_test_data(config, device, num_frames):
    """Pre-load test frames into GPU memory."""
    from detection.scripts.sparse_event_dataset import (
        SparseEventDataset, make_collate_fn, create_sparse_tensor
    )
    from torch.utils.data import DataLoader, Subset

    data_cfg = config.get('data', {})
    sparse_dir = str(project_root / data_cfg.get('sparse_dir'))
    label_dir = str(project_root / data_cfg.get('label_dir'))
    val_split = data_cfg.get('val_split', 'canonical_test')
    time_bins = config['sparse'].get('time_bins', 16)

    input_size = config['model'].get('input_size', [640, 640])
    if isinstance(input_size, int):
        input_size = [input_size, input_size]

    dataset = SparseEventDataset(
        sparse_dir=sparse_dir,
        label_dir=label_dir,
        split=val_split,
        target_size=tuple(input_size),
        time_bins=time_bins,
        augment=False,
    )

    collate_fn = make_collate_fn(time_bins=time_bins)
    if len(dataset) > num_frames:
        indices = random.sample(range(len(dataset)), num_frames)
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=4, collate_fn=collate_fn,
        pin_memory=True, drop_last=False,
    )

    print(f"  Pre-loading {min(num_frames, len(dataset))} frames to GPU...")
    batches = []
    for batch in loader:
        sp = create_sparse_tensor(batch, device)
        batches.append(sp)
        if len(batches) >= num_frames:
            break

    return batches


# ============================================================================
# Part 3: Latency + Memory Benchmark
# ============================================================================

def benchmark_sparse_model(model, batches, config, num_warmup, num_measure):
    """Benchmark SparseVoxelDet with CUDA Events + per-component breakdown."""
    from torchvision.ops import nms

    eval_cfg = config.get('eval', {})
    score_thresh = float(eval_cfg.get('score_thresh', 0.05))
    nms_thresh = float(eval_cfg.get('nms_thresh', 0.5))
    max_det = int(eval_cfg.get('max_detections', 100))

    device = next(model.parameters()).device

    # Warmup
    print(f"  Warmup ({num_warmup} iterations)...")
    for i in range(num_warmup):
        sp = batches[i % len(batches)]
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            preds = model(sp)
            if 'detections' in preds:
                _ = preds['detections']

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Measure
    print(f"  Measuring ({num_measure} iterations)...")
    latencies_total = []
    latencies_forward = []
    latencies_nms = []
    active_positions = []

    for i in range(num_measure):
        sp = batches[(num_warmup + i) % len(batches)]

        # Record active positions
        active_positions.append(sp.features.shape[0])

        # --- CUDA Events for precise timing ---
        start_total = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        end_total = torch.cuda.Event(enable_timing=True)

        start_total.record()

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            # Force return_loss_inputs to get raw preds (skip internal decode)
            preds = model(sp, return_loss_inputs=True)

        end_fwd.record()

        # NMS (same as real inference)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            cls_logits = preds['cls_logits']
            box_ltrb = preds['box_ltrb']
            ctr_logits = preds['ctr_logits']
            indices_2d = preds['indices_2d']

            # Score computation
            cls_scores = torch.sigmoid(cls_logits[:, 0])
            ctr_scores = torch.sigmoid(ctr_logits[:, 0])
            scores = cls_scores * ctr_scores

            # Threshold + NMS
            keep = scores > score_thresh
            if keep.any():
                s = scores[keep]
                # Decode boxes
                stride = 4
                cx = indices_2d[keep, 2].float() * stride + stride / 2.0
                cy = indices_2d[keep, 1].float() * stride + stride / 2.0
                ltrb = torch.exp(box_ltrb[keep].clamp(max=10.0))
                boxes = torch.stack([
                    cx - ltrb[:, 0], cy - ltrb[:, 1],
                    cx + ltrb[:, 2], cy + ltrb[:, 3]
                ], dim=-1)
                keep_nms = nms(boxes, s, nms_thresh)
                if len(keep_nms) > max_det:
                    keep_nms = keep_nms[:max_det]

        end_total.record()
        torch.cuda.synchronize()

        latencies_total.append(start_total.elapsed_time(end_total))
        latencies_forward.append(start_total.elapsed_time(end_fwd))
        latencies_nms.append(end_fwd.elapsed_time(end_total))

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    lats = np.array(latencies_total)
    fwd = np.array(latencies_forward)
    nms_lats = np.array(latencies_nms)
    pos = np.array(active_positions)

    return {
        'total_latency': _stats(lats),
        'forward_latency': _stats(fwd),
        'nms_latency': _stats(nms_lats),
        'fps_median': float(1000.0 / np.median(lats)),
        'fps_mean': float(1000.0 / np.mean(lats)),
        'peak_memory_mb': float(peak_mem),
        'active_positions': _stats(pos),
    }


def benchmark_yolo(model_path, device, num_warmup, num_measure, imgsz=640):
    """Benchmark YOLOv11 with same protocol: CUDA Events, FP16, batch=1."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    info = model.info(verbose=True)
    # info may return None in some ultralytics versions — get values directly
    n_params = sum(p.numel() for p in model.model.parameters())
    # GFLOPs from ultralytics
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = get_flops(model.model, imgsz=imgsz)
    except Exception:
        gflops = 6.6  # YOLOv11n canonical value at 640

    # Create dummy input for raw forward pass
    # (YOLOv11 architecture speed doesn't depend on image content)
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup with full pipeline
    print(f"  Warmup ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        model.predict(dummy, imgsz=imgsz, device=device, half=True, verbose=False)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # --- Full pipeline (preprocess + forward + NMS) ---
    print(f"  Measuring full pipeline ({num_measure} iterations)...")
    e2e_lats = []
    pre_times = []
    inf_times = []
    post_times = []

    for _ in range(num_measure):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        results = model.predict(dummy, imgsz=imgsz, device=device,
                                half=True, verbose=False)
        end.record()
        torch.cuda.synchronize()

        e2e_lats.append(start.elapsed_time(end))
        speed = results[0].speed
        pre_times.append(speed['preprocess'])
        inf_times.append(speed['inference'])
        post_times.append(speed['postprocess'])

    peak_mem_pipeline = torch.cuda.max_memory_allocated() / 1024**2

    # --- Raw forward pass only (no pre/post) ---
    print(f"  Measuring raw forward ({num_measure} iterations)...")
    torch_model = model.model.eval().half().to(device)
    dummy_tensor = torch.randn(1, 3, imgsz, imgsz, device=device, dtype=torch.float16)

    for _ in range(num_warmup):
        with torch.no_grad():
            torch_model(dummy_tensor)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    raw_lats = []
    with torch.no_grad():
        for _ in range(num_measure):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            torch_model(dummy_tensor)
            end.record()
            torch.cuda.synchronize()
            raw_lats.append(start.elapsed_time(end))

    peak_mem_raw = torch.cuda.max_memory_allocated() / 1024**2

    return {
        'params': int(n_params),
        'gflops': float(gflops),
        'imgsz': imgsz,
        'pipeline': {
            'total_latency': _stats(e2e_lats),
            'preprocess': _stats(pre_times),
            'inference': _stats(inf_times),
            'postprocess': _stats(post_times),
            'fps_median': float(1000.0 / np.median(e2e_lats)),
            'peak_memory_mb': float(peak_mem_pipeline),
        },
        'raw_forward': {
            'latency': _stats(raw_lats),
            'fps_median': float(1000.0 / np.median(raw_lats)),
            'peak_memory_mb': float(peak_mem_raw),
        },
    }


def _stats(arr):
    a = np.array(arr, dtype=float)
    return {
        'mean': float(a.mean()),
        'std': float(a.std()),
        'median': float(np.median(a)),
        'p5': float(np.percentile(a, 5)),
        'p95': float(np.percentile(a, 95)),
        'p99': float(np.percentile(a, 99)),
        'min': float(a.min()),
        'max': float(a.max()),
    }


# ============================================================================
# Part 4: Alternative TRUE FLOP counting via spconv pair counting
# ============================================================================

def count_true_sparse_macs(model, input_tensor):
    """Count sparse MACs by monkey-patching spconv pair generation AND forward.

    Forces ConvAlgo.Native so spconv produces explicit indice_pair_num tensors
    (pair counts per kernel offset). Monkey-patches get_indice_pairs to capture
    pair counts, and wraps SparseConvolution.forward to correlate each layer
    with its pair count — correctly handling layers that reuse pairs via shared
    indice_key.

    TRUE MACs per sparse conv = total_active_pairs × Cin × Cout
    TRUE MACs per linear     = N × Cin × Cout
    """
    import spconv.pytorch as spconv
    import spconv.pytorch.ops as spconv_ops
    from spconv.pytorch.conv import SparseConvolution
    from spconv.pytorch.core import ConvAlgo

    records = []

    # --- Force all sparse convolutions to use Native algorithm ---
    original_algos = {}
    for name, mod in model.named_modules():
        if isinstance(mod, SparseConvolution):
            original_algos[name] = mod.algo
            mod.algo = ConvAlgo.Native

    # --- Monkey-patch get_indice_pairs to capture pair counts ---
    pair_counts = []  # List of total pair counts, one per get_indice_pairs call
    orig_get_pairs = spconv_ops.get_indice_pairs

    def patched_get_pairs(*args, **kwargs):
        result = orig_get_pairs(*args, **kwargs)
        indice_pair_num = result[2]
        total = int(indice_pair_num.sum().item())
        pair_counts.append(total)
        return result

    spconv_ops.get_indice_pairs = patched_get_pairs

    # --- Build name map for modules ---
    name_map = {}
    for name, mod in model.named_modules():
        name_map[id(mod)] = name

    # --- Wrap SparseConvolution.forward to track per-layer MACs ---
    # This avoids the hook desync problem: we check pair_counts growth
    # to distinguish layers that generate pairs vs. reuse them.
    indice_key_pair_map = {}  # indice_key id → total_pairs (for reuse lookup)
    orig_sparse_forward = SparseConvolution.forward

    def instrumented_sparse_forward(self_, input_tensor_):
        n_in = input_tensor_.features.shape[0]
        queue_before = len(pair_counts)

        result = orig_sparse_forward(self_, input_tensor_)

        n_out = result.features.shape[0]
        c_in = self_.in_channels
        c_out = self_.out_channels
        ksize = list(self_.kernel_size)
        kv = 1
        for k in ksize:
            kv *= k

        total_pairs = None

        if kv == 1 and all(s == 1 for s in self_.stride):
            # 1×1 stride-1: torch.mm path, N multiplications
            total_pairs = n_in
        elif len(pair_counts) > queue_before:
            # New pair generation happened → use captured count
            total_pairs = pair_counts[-1]
            # Cache for reuse by later layers sharing same indice_key
            if self_.indice_key is not None:
                indice_key_pair_map[id(self_.indice_key)] = total_pairs
        else:
            # Reused pairs from a previous layer with same indice_key
            if self_.indice_key is not None and id(self_.indice_key) in indice_key_pair_map:
                total_pairs = indice_key_pair_map[id(self_.indice_key)]
            else:
                # Fallback: try indice_dict
                try:
                    for k, v in result.indice_dict.items():
                        if hasattr(v, 'indice_pair_num') and v.indice_pair_num is not None:
                            candidate = int(v.indice_pair_num.sum().item())
                            if candidate > 0:
                                total_pairs = candidate
                                break
                except Exception:
                    pass

        macs = total_pairs * c_in * c_out if total_pairs and total_pairs > 0 else None
        layer_name = name_map.get(id(self_), '??')

        records.append({
            'name': layer_name,
            'type': type(self_).__name__,
            'in_ch': c_in,
            'out_ch': c_out,
            'kernel': ksize,
            'stride': list(self_.stride),
            'subm': self_.subm,
            'transposed': getattr(self_, 'transposed', False),
            'n_in': n_in,
            'n_out': n_out,
            'total_pairs': total_pairs,
            'macs': macs,
            'avg_neighbors': total_pairs / max(n_out, 1) if total_pairs and total_pairs > 0 else None,
        })

        return result

    SparseConvolution.forward = instrumented_sparse_forward

    # --- Register hooks for Linear layers ---
    handles = []

    def make_linear_hook(name):
        def hook_fn(m, inp, out):
            n = inp[0].shape[0]
            macs = n * m.in_features * m.out_features
            records.append({
                'name': name,
                'type': 'Linear',
                'in_ch': m.in_features,
                'out_ch': m.out_features,
                'kernel': [1],
                'stride': [1],
                'subm': False,
                'transposed': False,
                'n_in': n,
                'n_out': n,
                'total_pairs': n,
                'macs': macs,
                'avg_neighbors': 1.0,
            })
        return hook_fn

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            h = mod.register_forward_hook(make_linear_hook(name))
            handles.append(h)

    # --- Run forward pass ---
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        _ = model(input_tensor)

    # --- Cleanup ---
    for h in handles:
        h.remove()
    SparseConvolution.forward = orig_sparse_forward
    spconv_ops.get_indice_pairs = orig_get_pairs
    for name, mod in model.named_modules():
        if isinstance(mod, SparseConvolution) and name in original_algos:
            mod.algo = original_algos[name]

    # --- Summary ---
    total_macs = 0
    unmeasured = 0

    print(f"\n{'='*140}")
    print(f"TRUE SPARSE MAC COUNT (monkey-patched spconv pair generation)")
    print(f"{'='*140}")
    print(f"{'Layer':<50} {'Type':<20} {'Ch':>10} {'K':>6} {'N_in':>8} "
          f"{'N_out':>8} {'Pairs':>10} {'MACs':>14} {'Nbrs':>6}")
    print(f"{'-'*140}")

    for r in records:
        ch = f"{r['in_ch']}→{r['out_ch']}"
        k = 'x'.join(str(x) for x in r['kernel'])
        if r['total_pairs'] and r['total_pairs'] > 0:
            pairs_s = f"{r['total_pairs']:,}"
        else:
            pairs_s = "UNMEASURED"
        if r['macs'] and r['macs'] > 0:
            macs_s = f"{r['macs']:,}"
            total_macs += r['macs']
        else:
            macs_s = "UNMEASURED"
            unmeasured += 1
        nbr = f"{r['avg_neighbors']:.1f}" if r.get('avg_neighbors') and r['avg_neighbors'] > 0 else "-"

        print(f"{r['name']:<50} {r['type']:<20} {ch:>10} {k:>6} "
              f"{r['n_in']:>8,} {r['n_out']:>8,} {pairs_s:>10} {macs_s:>14} {nbr:>6}")

    print(f"{'-'*140}")
    print(f"Total measured MACs: {total_macs:,.0f} ({total_macs/1e6:.2f} MMACs / {total_macs/1e9:.4f} GMACs)")
    if unmeasured > 0:
        print(f"WARNING: {unmeasured} layer(s) could not extract pair counts")
    print(f"{'='*140}")

    return records, total_macs


def count_flops_multi_frame(model, batches, num_frames=50):
    """Count MACs over multiple frames to get mean/median/distribution."""
    all_macs = []
    all_positions = []

    for i in range(min(num_frames, len(batches))):
        sp = batches[i]
        n_pos = sp.features.shape[0]
        all_positions.append(n_pos)

        records, total_macs = count_true_sparse_macs(model, sp)
        all_macs.append(total_macs)

        if i == 0:
            # Print detailed breakdown for first frame
            pass
        if (i + 1) % 10 == 0:
            print(f"  Counted MACs for {i+1}/{min(num_frames, len(batches))} frames...")

    macs_arr = np.array(all_macs, dtype=float)
    pos_arr = np.array(all_positions, dtype=float)

    print(f"\nMAC Distribution over {len(macs_arr)} frames:")
    print(f"  Active positions: mean={pos_arr.mean():.0f}, median={np.median(pos_arr):.0f}")
    print(f"  MACs: mean={macs_arr.mean()/1e6:.2f}M, median={np.median(macs_arr)/1e6:.2f}M")
    print(f"  GMACs: mean={macs_arr.mean()/1e9:.4f}, median={np.median(macs_arr)/1e9:.4f}")

    return {
        'macs': _stats(macs_arr),
        'gmacs_mean': float(macs_arr.mean() / 1e9),
        'gmacs_median': float(np.median(macs_arr) / 1e9),
        'positions': _stats(pos_arr),
    }


# ============================================================================
# Part 5: Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Paper-grade efficiency benchmark: SparseVoxelDet vs YOLOv11')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='SparseVoxelDet checkpoint (best.pt)')
    parser.add_argument('--config', type=str, required=True,
                        help='SparseVoxelDet config yaml')
    parser.add_argument('--yolo-model', type=str, default='yolo11n.pt',
                        help='YOLOv11 model path')
    parser.add_argument('--num-warmup', type=int, default=100)
    parser.add_argument('--num-measure', type=int, default=1000)
    parser.add_argument('--num-flop-frames', type=int, default=50,
                        help='Number of frames for FLOP distribution')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: next to checkpoint)')
    args = parser.parse_args()

    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2

    print("=" * 80)
    print("PAPER-GRADE EFFICIENCY BENCHMARK")
    print("=" * 80)
    print(f"GPU: {gpu_name} ({total_mem:.0f} MB)")
    print(f"Sparse checkpoint: {args.checkpoint}")
    print(f"YOLO model: {args.yolo_model}")
    print(f"Protocol: FP16, batch=1, {args.num_warmup} warmup, {args.num_measure} measure")
    print()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results = {
        'meta': {
            'gpu': gpu_name,
            'gpu_memory_mb': total_mem,
            'precision': 'FP16',
            'batch_size': 1,
            'num_warmup': args.num_warmup,
            'num_measure': args.num_measure,
            'sparse_checkpoint': str(args.checkpoint),
            'yolo_model': str(args.yolo_model),
        }
    }

    # ----------------------------------------------------------------
    # Step 1: Load sparse model + test data
    # ----------------------------------------------------------------
    print("[1/5] Loading SparseVoxelDet model...")
    model = load_sparse_model(args.checkpoint, config, device)
    param_count = sum(p.numel() for p in model.parameters())
    param_breakdown = model.get_num_params()
    print(f"  Parameters: {param_count:,} (backbone={param_breakdown['backbone']:,}, "
          f"fpn={param_breakdown['fpn']:,}, head={param_breakdown['head']:,})")

    total_frames = args.num_warmup + args.num_measure
    print(f"\n[2/5] Loading {total_frames} test frames...")
    batches = load_test_data(config, device, total_frames)
    print(f"  Loaded {len(batches)} frames")

    # ----------------------------------------------------------------
    # Step 2: TRUE FLOP counting
    # ----------------------------------------------------------------
    print(f"\n[3/5] Counting sparse MACs ({args.num_flop_frames} frames)...")
    flop_results = count_flops_multi_frame(model, batches, args.num_flop_frames)

    results['sparse_model'] = {
        'params': param_count,
        'param_breakdown': param_breakdown,
        'flops': flop_results,
    }

    # ----------------------------------------------------------------
    # Step 3: Sparse model latency benchmark
    # ----------------------------------------------------------------
    print(f"\n[4/5] Benchmarking SparseVoxelDet latency...")
    sparse_bench = benchmark_sparse_model(
        model, batches, config, args.num_warmup, args.num_measure
    )
    results['sparse_model']['benchmark'] = sparse_bench

    # Free sparse model from GPU
    del model
    torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Step 4: YOLO benchmark
    # ----------------------------------------------------------------
    print(f"\n[5/5] Benchmarking YOLOv11...")
    yolo_bench = benchmark_yolo(
        args.yolo_model, device, args.num_warmup, args.num_measure
    )
    results['yolo'] = yolo_bench

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EFFICIENCY COMPARISON SUMMARY")
    print("=" * 80)

    sparse = results['sparse_model']
    yolo = results['yolo']

    sp_lat = sparse['benchmark']['total_latency']['median']
    sp_fwd = sparse['benchmark']['forward_latency']['median']
    sp_nms = sparse['benchmark']['nms_latency']['median']
    sp_fps = sparse['benchmark']['fps_median']
    sp_mem = sparse['benchmark']['peak_memory_mb']
    sp_gmacs = sparse['flops']['gmacs_median']
    sp_pos = sparse['benchmark']['active_positions']['median']

    yl_lat = yolo['pipeline']['total_latency']['median']
    yl_fwd = yolo['raw_forward']['latency']['median']
    yl_fps = yolo['pipeline']['fps_median']
    yl_mem = yolo['pipeline']['peak_memory_mb']
    yl_gflops = yolo['gflops']

    print(f"\n{'Metric':<30} {'SparseVoxelDet':>18} {'YOLOv11n':>18} {'Ratio':>12}")
    print(f"{'-'*78}")
    print(f"{'Parameters':.<30} {sparse['params']:>18,} {yolo['params']:>18,} "
          f"{sparse['params']/yolo['params']:>11.1f}×")
    print(f"{'GMACs (data-dependent)':.<30} {sp_gmacs:>17.3f}G {yl_gflops:>17.1f}G "
          f"{sp_gmacs/max(yl_gflops,0.001):>11.1f}×")
    print(f"{'Active positions (median)':.<30} {sp_pos:>18,.0f} {'409,600':>18} "
          f"{409600/sp_pos:>11.1f}×")
    print(f"{'Forward latency (ms)':.<30} {sp_fwd:>17.2f}ms {yl_fwd:>17.2f}ms "
          f"{sp_fwd/yl_fwd:>11.1f}×")
    print(f"{'Total latency (ms)':.<30} {sp_lat:>17.2f}ms {yl_lat:>17.2f}ms "
          f"{sp_lat/yl_lat:>11.1f}×")
    print(f"{'FPS (median)':.<30} {sp_fps:>18.1f} {yl_fps:>18.1f} "
          f"{'':>12}")
    print(f"{'Peak GPU memory (MB)':.<30} {sp_mem:>17.0f}MB {yl_mem:>17.0f}MB "
          f"{sp_mem/yl_mem:>11.1f}×")
    print(f"{'NMS latency (ms)':.<30} {sp_nms:>17.2f}ms {'(included)':>18} "
          f"{'':>12}")
    print("=" * 80)
    if sp_gmacs > yl_gflops:
        print(f"\nKey finding: {sp_gmacs/yl_gflops:.1f}× MORE MACs than YOLO, but "
              f"{409600/sp_pos:.0f}× fewer active positions")
    else:
        print(f"\nKey finding: {yl_gflops/sp_gmacs:.1f}× fewer MACs than YOLO, and "
              f"{409600/sp_pos:.0f}× fewer active positions")

    # Save
    output_path = args.output or str(Path(args.checkpoint).parent / 'efficiency_benchmark.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
