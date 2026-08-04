# SparseVoxelDet

Code for "SparseVoxelDet: Fully Sparse Voxel Networks for Efficient Event-Based Drone Detection".

Coordinate-sparse object detection on event-camera voxel grids. No stage of the pipeline, from backbone to detection head, constructs a dense spatial feature grid.

Mohamad Yazan Sadoun, Sarah Sharif, Yaser Mike Banad. University of Oklahoma.

## Results (FRED drone-detection benchmark)

Development protocol (147 training / 37 validation sequences, source-complete labels, single evaluator):

| Model | AP50 | AP50:95 |
|---|---|---|
| SparseVoxelDet (seed 42) | 87.01 | 43.14 |
| SparseVoxelDet (seed 123) | 86.38 | 42.26 |
| YOLO11n, resolution-matched | 84.68 | 42.44 |
| YOLO11n, letterboxed | 77.29 | 38.12 |

Single sealed evaluation on FRED's canonical test partition, run once after every decision was frozen:

| Model | AP50 |
|---|---|
| SparseVoxelDet (seed 42) | 84.33 |
| SparseVoxelDet (seed 123) | 83.01 |
| YOLO11n dense control | 79.37 |

Efficiency: expansion-free fusion returns head occupancy from a median 78.88% to 10.53% and cuts fusion-stage work by 91.3%. Against numerically matched dense equivalents of its own operators, sparse execution is cheaper on all 5,000 profiled frames, by a median 27.5x in work and 4.65x in latency at batch 1.

## Layout

```
V2/models/            SparseVoxelDet architecture (sparse_voxel_det_ic.py) and its base
models/snn/           sparse SEW-ResNet backbone
sparse_fcos_v1/       dataset loader, evaluator (metrics.py), EMA, event mosaic
training/             released training entry point, trainer, both objectives, preflight, tests
configs/              frozen training configuration; dense-control launch records
tools/                voxel construction, support profiler, error forensics, contract validators
docs/REPRODUCIBILITY.md   pinned-source digests, label provenance, dense-control recipe
```

## Installation

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install spconv-cu126==2.3.8 spikingjelly==0.0.0.0.14
pip install -r requirements.txt
```

Python 3.12, CUDA-capable GPU. Versions used for all reported numbers are listed in docs/REPRODUCIBILITY.md.

## Data

Download FRED (Prophesee EVK4, 1280x720, five drone models) from its authors' release, then build sparse voxel grids and labels:

```
python tools/regenerate_parity_sparse_coords_v82_640.py --help
python tools/build_fred_official_parity_multibox.py --help
```

Each frame becomes a coordinate array (M, 3) of [t, y, x] active voxels with (M, 6) features; median input occupancy is 0.0652%.

## Training

```
python training/train_ic_quality.py --config configs/ic_quality_ddp3_e20.yaml
```

The released run used 3-GPU distributed data parallelism, global batch 6, 20 epochs, seed 42; the preflight in training/preflight_quality.py verifies source hashes and label manifests before launch.

## Evaluation

```
python sparse_fcos_v1/scripts/evaluate_sparse_fcos.py --help
```

The evaluator (sparse_fcos_v1/scripts/metrics.py, SHA-256 23c86324) scored every number in the paper, both arms, both splits.

## Profiling and forensics

```
python tools/instrument_active_positions.py        # stage-wise active-position counts
python tools/profiler/run_equivalence_gate.py      # dense-equivalence gate (must pass before profiling)
python tools/profiler/profile_worker.py            # paired sparse/dense cost measurement
python tools/run_error_forensics.py                # false-negative attribution
```

## License and citation

MIT License. If you use this code, cite: M. Y. Sadoun, S. Sharif, Y. M. Banad, "SparseVoxelDet: Fully Sparse Voxel Networks for Efficient Event-Based Drone Detection," 2026.
