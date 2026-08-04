#!/usr/bin/env python3
"""
instrument_active_positions.py -- active-position (sparse occupancy) profiling.

Loads the v83_640 best checkpoint and runs a deterministic, fixed-seed 5,000-frame
subset of canonical_test, recording the number of ACTIVE positions
(SparseConvTensor.indices.shape[0]) at:
  * input voxels
  * each backbone stage output (c2, c3, c4)
  * each FPN level BEFORE and AFTER transpose-conv upsampling
  * head input (fused FPN sparse map feeding the detection head)
and computes per-stage expansion ratios (medians across frames).

Uses forward hooks on the model's submodules; counts come from the sparse
tensor .indices.shape[0]. Run on a 3090:
    CUDA_VISIBLE_DEVICES=2 ../venv/bin/python tools/instrument_active_positions.py

Output JSON -> paper/artifacts/results/active_positions_v83_640.json  (+ printed table)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from V2.models.sparse_voxel_det_v82 import SparseVoxelDet
from sparse_fcos_v1.scripts.sparse_event_dataset_v82 import (
    SparseEventDataset,
    make_collate_fn,
    create_sparse_tensor,
)

N_FRAMES = 5000
SEED = 42
CONFIG = REPO / "V2/configs/sparse_voxel_det_v83_640.yaml"
CKPT = REPO / "runs/v83_640_regen/best.pt"
SPLIT = "canonical_test"


def n_active(obj):
    """Extract active-position count from a SparseConvTensor (or tuple/list thereof)."""
    if hasattr(obj, "indices"):
        return int(obj.indices.shape[0])
    return None


def load_model(config, device):
    m = config["model"]
    ev = config.get("eval", {})
    model = SparseVoxelDet(
        in_channels=m.get("in_channels", 6),
        num_classes=m.get("num_classes", 1),
        backbone_size=m.get("backbone_size", "nano_deep"),
        fpn_channels=m.get("fpn_channels", 128),
        head_convs=m.get("head_convs", 2),
        input_size=tuple(m.get("input_size", [640, 640])),
        time_bins=config.get("sparse", {}).get("time_bins", 16),
        prior_prob=m.get("prior_prob", 0.01),
        score_thresh=float(ev.get("score_thresh", 0.05)),
        nms_thresh=float(ev.get("nms_thresh", 0.5)),
        max_detections=int(ev.get("max_detections", 100)),
        temporal_pool_mode=m.get("temporal_pool_mode", "max"),
    ).to(device)
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.eval()
    return model


def register_hooks(model, per_frame):
    """Hook backbone, FPN laterals, FPN upsample blocks (in+out), fused output."""
    handles = []

    def make_out_hook(name):
        def hook(mod, inp, out):
            c = n_active(out)
            if c is not None:
                per_frame[name].append(c)
        return hook

    def make_inout_hook(name_in, name_out):
        def hook(mod, inp, out):
            ci = n_active(inp[0]) if inp else None
            co = n_active(out)
            if ci is not None:
                per_frame[name_in].append(ci)
            if co is not None:
                per_frame[name_out].append(co)
        return hook

    bb = model.backbone
    fpn = model.fpn

    # Backbone whole-module output is a list of sparse maps -> hook via forward hook
    def bb_hook(mod, inp, out):
        # out expected: list/tuple of 3 sparse maps (c2, c3, c4)
        if isinstance(out, (list, tuple)):
            for i, o in enumerate(out):
                c = n_active(o)
                if c is not None:
                    per_frame[f"backbone_stage_{i}_c{i+2}"].append(c)
    handles.append(bb.register_forward_hook(bb_hook))

    # FPN lateral outputs (before upsampling fusion)
    for lname in ("lateral_c2", "lateral_c3", "lateral_c4"):
        if hasattr(fpn, lname):
            handles.append(getattr(fpn, lname).register_forward_hook(
                make_out_hook(f"fpn_{lname}")))

    # FPN transpose-conv upsample blocks: record BEFORE (input) and AFTER (output)
    for uname in ("up_c4_to_c3", "up_c3_to_c2"):
        if hasattr(fpn, uname):
            handles.append(getattr(fpn, uname).register_forward_hook(
                make_inout_hook(f"fpn_{uname}_BEFORE_upsample",
                                f"fpn_{uname}_AFTER_upsample")))

    # Fused FPN output (head input) -- whole FPN module output
    def fpn_hook(mod, inp, out):
        c = n_active(out)
        if c is not None:
            per_frame["head_input_fused_fpn"].append(c)
    handles.append(fpn.register_forward_hook(fpn_hook))

    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=N_FRAMES)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available -- this profiler runs the sparse model on GPU. "
            "Requires a working CUDA runtime (RTX 3090)."
        )
    device = torch.device("cuda:0")

    with open(CONFIG) as f:
        config = yaml.safe_load(f)
    tb = config.get("sparse", {}).get("time_bins", 16)
    input_size = tuple(config["model"].get("input_size", [640, 640]))

    dataset = SparseEventDataset(
        sparse_dir=str(REPO / config["data"]["sparse_dir"]),
        label_dir=str(REPO / config["data"]["label_dir"]),
        split=SPLIT,
        time_bins=tb,
        target_size=input_size,
        augment=False,
    )
    collate = make_collate_fn(time_bins=tb)

    # Deterministic fixed-seed frame sampling
    rng = np.random.RandomState(args.seed)
    total = len(dataset)
    n = min(args.n_frames, total)
    sampled = sorted(rng.choice(total, size=n, replace=False).tolist())

    model = load_model(config, device)
    per_frame = defaultdict(list)
    input_counts = []

    handles = register_hooks(model, per_frame)
    with torch.no_grad():
        for j, idx in enumerate(sampled):
            batch = collate([dataset[idx]])
            sp = create_sparse_tensor(batch, device)
            input_counts.append(int(sp.indices.shape[0]))
            _ = model(sp, 1, return_loss_inputs=False)
            if (j + 1) % 500 == 0:
                print(f"  {j+1}/{n} frames")
    for h in handles:
        h.remove()

    per_frame["input_voxels"] = input_counts

    # Medians per stage
    stage_median = {k: float(np.median(v)) for k, v in per_frame.items() if v}
    stage_mean = {k: float(np.mean(v)) for k, v in per_frame.items() if v}

    input_med = stage_median.get("input_voxels", float("nan"))
    expansion = {k: (v / input_med if input_med else None)
                 for k, v in stage_median.items()}
    # Explicit BEFORE->AFTER upsample expansion ratios (the FPN headline)
    fpn_upsample_ratios = {}
    for uname in ("up_c4_to_c3", "up_c3_to_c2"):
        b = stage_median.get(f"fpn_{uname}_BEFORE_upsample")
        a = stage_median.get(f"fpn_{uname}_AFTER_upsample")
        if b and a:
            fpn_upsample_ratios[f"fpn_{uname}_after_over_before"] = a / b

    result = {
        "tool": "instrument_active_positions.py",
        "checkpoint": str(CKPT),
        "config": str(CONFIG),
        "split": SPLIT,
        "n_frames_requested": args.n_frames,
        "n_frames_used": n,
        "seed": args.seed,
        "sampling": "np.random.RandomState(seed).choice(len, n, replace=False), sorted",
        "device_name": torch.cuda.get_device_name(device),
        "stage_median_active_positions": stage_median,
        "stage_mean_active_positions": stage_mean,
        "expansion_ratio_vs_input_median": expansion,
        "fpn_transposeconv_upsample_ratios_median": fpn_upsample_ratios,
    }
    out_path = REPO / "paper/artifacts/results/active_positions_v83_640.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Summary table
    print("=" * 82)
    print(f"active positions (medians over {n} frames, seed={args.seed})")
    print("=" * 82)
    print(f"{'stage':<42}{'median active':>16}{'x input':>14}")
    print("-" * 82)
    order = ["input_voxels",
             "backbone_stage_0_c2", "backbone_stage_1_c3", "backbone_stage_2_c4",
             "fpn_lateral_c2", "fpn_lateral_c3", "fpn_lateral_c4",
             "fpn_up_c4_to_c3_BEFORE_upsample", "fpn_up_c4_to_c3_AFTER_upsample",
             "fpn_up_c3_to_c2_BEFORE_upsample", "fpn_up_c3_to_c2_AFTER_upsample",
             "head_input_fused_fpn"]
    for k in order:
        if k in stage_median:
            print(f"{k:<42}{stage_median[k]:>16.1f}{expansion[k]:>13.2f}x")
    # any stages not in the fixed order
    for k in stage_median:
        if k not in order:
            print(f"{k:<42}{stage_median[k]:>16.1f}{expansion[k]:>13.2f}x")
    print("-" * 82)
    print("FPN transpose-conv upsample expansion (AFTER / BEFORE, median):")
    for k, v in fpn_upsample_ratios.items():
        print(f"  {k}: {v:.2f}x")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
