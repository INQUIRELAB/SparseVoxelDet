#!/usr/bin/env python3
"""Runtime installation of the sample-independent SparseSEBlock forward contract."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

EXPECTED_SOURCE_SHA256 = "d2bdd011954126d6011409269b8db26c86cf77cb5605a492c308ceceb3e3b4a4"


def _source_path() -> Path:
    return Path(__file__).resolve().parents[3] / "models/snn/sparse_sew_resnet.py"


def source_sha256() -> str:
    return hashlib.sha256(_source_path().read_bytes()).hexdigest()


def per_sample_forward(self, x):
    feats = x.features
    if feats.shape[0] == 0:
        return x

    work = feats.float() if feats.dtype in (torch.float16, torch.bfloat16) else feats
    batch_ids = x.indices[:, 0].to(device=work.device, dtype=torch.long)
    batch_size = int(x.batch_size)
    sums = work.new_zeros((batch_size, work.shape[1]))
    sums.index_add_(0, batch_ids, work)
    counts = torch.bincount(batch_ids, minlength=batch_size).to(dtype=work.dtype).unsqueeze(1)
    squeezed = (sums / counts.clamp_min(1.0)).to(dtype=feats.dtype)

    scale = F.relu(self.fc1(squeezed))
    scale = torch.sigmoid(self.fc2(scale))
    return x.replace_feature(feats * scale.index_select(0, batch_ids))


def install() -> str:
    actual = source_sha256()
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"SparseSEBlock source drift: expected {EXPECTED_SOURCE_SHA256}, got {actual}"
        )
    from models.snn.sparse_sew_resnet import SparseSEBlock

    if getattr(SparseSEBlock, "_per_sample_patch_installed", False):
        return actual
    SparseSEBlock.forward = per_sample_forward
    SparseSEBlock._per_sample_patch_installed = True
    return actual