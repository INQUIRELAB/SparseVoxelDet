# SparseVoxelDet

**No Dense Tensors Needed: Fully Sparse Object Detection on Event-Camera Voxel Grids**

## 👤 Authors

**Mohamad Yazan Sadoun**, **Sarah Sharif**, **Yaser Mike Banad**

University of Oklahoma, Norman, OK, USA

## 📄 Abstract

Event cameras produce asynchronous, high-dynamic-range streams well suited for detecting small, fast-moving drones, yet most event-based detectors convert the sparse event stream into dense tensors, discarding the representational efficiency of neuromorphic sensing. We propose SparseVoxelDet, a fully sparse object detector in which backbone feature extraction, feature pyramid fusion, and the detection head all operate exclusively on occupied voxel positions through 3D sparse convolutions — no dense feature tensor is instantiated at any stage of the pipeline. On the FRED benchmark (629,832 annotated frames), SparseVoxelDet achieves 83.38% mAP@50 while processing only ~14,900 active voxels per frame (0.23% of the T×H×W grid), compared to 409,600 pixels for the dense YOLOv11 baseline (87.68% mAP@50). The sparse representation yields 858× GPU memory compression and 3,670× storage reduction relative to the equivalent dense 3D voxel tensor, with data-structure size that scales with scene dynamics rather than sensor resolution.

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 📊 Results

| Method | Input | mAP@50 | mAP@50:95 | Active Positions |
|--------|-------|--------|-----------|------------------|
| YOLOv11 (dense baseline) | Event frames 640² | 87.68% | 49.25% | 409,600 |
| **SparseVoxelDet (ours)** | **Sparse voxels 640²** | **83.38%** | **39.23%** | **~14,900** |

- At IoU 0.40, SparseVoxelDet recovers to **89.26% mAP**, exceeding the YOLOv11 mAP@50 score
- **91.9% recall** at IoU ≥ 0.50 — 71% of failures are localization near-misses, not missed targets
- **858× GPU memory compression** and **3,670× storage reduction** over dense equivalents

## 🏗️ Architecture

```
Raw Events → Sparse Voxel Grid → SparseSEResNet → SparseFPN → SparseDetHead → NMS
              (coord/feat pairs)   (backbone)       (neck)      (FCOS head)
              zero dense tensors   3D sparse conv   sparse only  MLP on active pos
```

All operations use [spconv](https://github.com/traveller59/spconv) (Spatially Sparse Convolution Library). No dense tensor is allocated at any point in the pipeline.

## ⚙️ Installation

```bash
git clone https://github.com/yazansadoun/SparseVoxelDet.git
cd SparseVoxelDet

python -m venv venv
source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install spconv-cu120  # or spconv-cu118 for CUDA 11.8
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- spconv (CUDA-compatible version)
- CUDA-capable GPU with ≥ 16 GB VRAM

## 📁 Dataset

We evaluate on the [FRED (Florence RGB-Event Drone)](https://github.com/EVENT-CAMERA-FRED) benchmark:

- 629,832 annotated event frames across 231 sequences
- Prophesee EVK4 (IMX636) at 1280×720
- 5 drone models, diverse lighting (daylight to complete darkness)

### Preprocessing

Convert raw event streams into sparse voxel grids:

```bash
# Native resolution (1280×720, 6 channels, T=16)
python tools/regenerate_parity_sparse_coords_v82.py \
    --input data/processed/FRED/ \
    --output data/datasets/fred_paper_parity_v82/sparse/ \
    --time-bins 16

# 640×640 resolution
python tools/regenerate_parity_sparse_coords_v82_640.py \
    --input data/processed/FRED/ \
    --output data/datasets/fred_paper_parity_v82_640/sparse/ \
    --time-bins 16
```

Each output `.npz` file contains:
- `coords`: `(M, 3)` int32 — `[t, y, x]` coordinates of active voxels
- `feats`: `(M, 6)` float16 — temporal surface features per voxel
- Typical `M ≈ 14,900` at 640² (0.23% occupancy)

## 🚀 Training

```bash
CUDA_VISIBLE_DEVICES=0 python training/scripts/train_sparse_voxel_det_v82.py \
    --config training/configs/sparse_voxel_det_v83_640.yaml
```

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Optimizer | AdamW (lr=3e-4, wd=1e-2) |
| Schedule | Cosine with 5,000-step warmup |
| Batch size | 2 (single GPU) |
| Precision | FP16 (AMP) |
| EMA decay | 0.9997 |

## 📈 Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python training/scripts/evaluate_sparse_voxel_det.py \
    --checkpoint runs/best.pt
```

## 🗂️ Project Structure

```
├── training/                     # Main training and evaluation pipeline
│   ├── models/                  # SparseVoxelDet architecture
│   ├── scripts/                 # Training, evaluation, benchmarking
│   ├── configs/                 # YAML configurations
│   ├── analysis/                # Prediction dumping, side-by-side rendering
│   └── tests/                   # Smoke tests
├── detection/                   # Shared detection infrastructure
│   ├── models/                  # FCOS head, sparse detectors
│   └── scripts/                 # Dataset loader, metrics, EMA, loss
├── backbone/                    # SparseSEResNet backbone
└── tools/                       # Preprocessing and benchmarking utilities
```

## 📖 Citation

```bibtex
@article{sadoun2026sparsevoxeldet,
  title={No Dense Tensors Needed: Fully Sparse Object Detection on Event-Camera Voxel Grids},
  author={Sadoun, Mohamad Yazan and Sharif, Sarah and Banad, Yaser Mike},
  year={2026}
}
```

## 🙏 Acknowledgements

- [FRED Dataset](https://github.com/EVENT-CAMERA-FRED) — Florence RGB-Event Drone detection benchmark
- [spconv](https://github.com/traveller59/spconv) — 3D sparse convolution primitives
- [spikingjelly](https://github.com/fangwei123456/spikingjelly) — Spiking neural network framework
