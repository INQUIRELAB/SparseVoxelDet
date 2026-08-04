# Reproducibility

Every rule stated as method in the paper is read from the released run's own sources, pinned by SHA-256 and verified by the run's launch preflight (training/preflight_quality.py) before training started.

## Environment

Python 3.12.3, PyTorch 2.12.0.dev20260221+cu128, spconv-cu126 2.3.8, spikingjelly 0.0.0.0.14, CUDA 12.8, ultralytics 8.3.236 (dense control only). Sparse wall-clock numbers do not transfer across execution stacks; the paper reports the stack alongside every latency figure.

## Pinned sources (released quality-aligned run, seed 42)

| SHA-256 (head) | file |
|---|---|
| b298de64 | V2/models/sparse_voxel_det_ic.py |
| c86cc66f | V2/models/sparse_voxel_det_v82.py |
| d2bdd011 | models/snn/sparse_sew_resnet.py |
| 194dabd2 | sparse_fcos_v1/scripts/ema.py |
| 73fab955 | sparse_fcos_v1/scripts/evaluate_sparse_fcos.py |
| 031fe548 | sparse_fcos_v1/scripts/event_mosaic.py |
| 23c86324 | sparse_fcos_v1/scripts/metrics.py |
| eda3f664 | sparse_fcos_v1/scripts/sparse_event_dataset_v82.py |
| d4a48432 | tools/se_per_sample_patch.py |
| 375b5551 | tools/validate_sparse_tensor_contract.py |
| 340b900b | training/train_ic_quality.py |
| 45c4ded5 | training/sparse_trainer_ic_quality.py |
| 08b7030f | training/quality_aligned_loss.py |
| 92957d72 | training/strict_loss.py |
| 1067e925 | configs/ic_quality_ddp3_e20.yaml |

Full digests are in SHA256SUMS.txt at the repository root. Verify with:

```
sha256sum -c SHA256SUMS.txt
```

## Artifact digests behind the reported numbers

| SHA-256 | artifact |
|---|---|
| 6a973831e215c733e77f4ba2553ae0e138a20cf01f1c5e30387292f52b2c56ee | corrected label manifest |
| 23c8632412719e3e917459b40ad601d4315cab67ced789ba385982c04c6cf6ba | evaluator source (metrics.py) |
| b25c62a09190a09c3659854f985c477cc4e052ea089e3f9bb438fb56d05c08ba | selected checkpoint (epoch 5, EMA weights) |
| 2b5f49266af20411ef401591b140a30c5fe5a591b9669cd123643f095a272640 | development-validation prediction cache (predictions_cache_raw.npz) |

## Topology-matched center-only arm

The batch-matched center-only arm ran from a separately pinned build of the same stack (as-run digests 33c2f62c, b261c4b6, c9e130c8, verified by the same preflight). Its objective file, digest 28c64f4a, differs from the released strict_loss.py (92957d72) in exactly one branch: when a rank's batch has no positive assignment, zero regression and centerness losses are built as graph-attached zeros rather than detached constants, so every distributed rank populates every gradient bucket. The change is value- and gradient-neutral.

## Label provenance and residue

The label file at index k carries the source annotation at raw timestamp k*dt (dt = 33,333 us), while the voxel frame at the same index accumulates events over [t0 + k*dt, t0 + (k+1)*dt); a per-sequence alignment search puts the optimum between integer offsets 0 and +1, a sub-frame effect rather than an integer misalignment, identically on both partitions. The dense comparator's rendered frames follow the source renderer's own convention and pair label index k with the frame at (k+1)*dt, verified exhaustively on the test partition (119,459 of 119,459 labels resolve under that convention).

The source-complete conversion (tools/build_fred_official_parity_multibox.py) leaves a known residue: 1,935 development frames (1.87%) and 1,448 test frames (1.21%) keep a single box where the raw source holds two boxes that could not be disambiguated, a small ceiling on measured precision affecting both splits in near-equal measure.

## Dense-control training recipe

The dense comparator (YOLO11n, ultralytics 8.3.236) recipe is released as frozen records rather than a transcription:

- configs/dense_control/launch_contract_yolo11n_seed42_e100_b128.json: the launch contract written at start of run (train arguments, pretrained-weight digest, dataset digests, GPU, library versions).
- configs/dense_control/args.yaml: the framework's own frozen argument record with every resolved default.
- configs/dense_control/fred_canonical_val.yaml and train_yolo_control.py: the data definition and the launch script.

Key settings: COCO-pretrained yolo11n.pt (SHA-256 0ebbc80d...), 100 epochs, batch 128, imgsz 640, seed 42, deterministic, single class, AMP; checkpoint selected by the framework's own fitness rule.

## Support-efficiency reporting protocol

The paper's Appendix A defines six reporting fields for coordinate-sparse detectors. The profiler under tools/profiler/ emits all of them in one pass; tools/instrument_active_positions.py records stage-wise active-position counts, and tools/profiler/run_equivalence_gate.py must pass (numerical equivalence between the sparse network and its dense realization) before any paired cost measurement is trusted.
