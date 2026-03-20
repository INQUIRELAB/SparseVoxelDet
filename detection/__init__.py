"""
Sparse FCOS Detector v1

A sparse 3D event-based object detector using FCOS anchor-free detection.

Architecture:
- SparseSEWResNet backbone (true sparse 3D convolutions)
- FPN neck for multi-scale features
- FCOS head for anchor-free detection

Key Features:
- Sparse processing: O(N) where N = active voxels
- Full temporal resolution
- FCOS: anchor-free detection
"""

__version__ = '1.0.0'
