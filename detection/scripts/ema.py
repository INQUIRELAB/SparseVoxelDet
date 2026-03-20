#!/usr/bin/env python3
"""
Exponential Moving Average (EMA) for model weights.

Maintains a shadow copy of model parameters updated with exponential decay.
Use the EMA model for validation and inference (~0.5-1% mAP improvement for free).

Reference: Polyak & Juditsky (1992), "Acceleration of Stochastic Approximation"
"""
import torch
import torch.nn as nn
from copy import deepcopy


class ModelEMA:
    """
    Exponential Moving Average of model parameters.

    shadow_params = decay * shadow_params + (1 - decay) * model_params

    Usage:
        ema = ModelEMA(model, decay=0.9999)
        for batch in dataloader:
            loss.backward()
            optimizer.step()
            ema.update(model)
        # For validation:
        ema.apply_shadow(model)
        validate(model)
        ema.restore(model)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = deepcopy(model.state_dict())
        self.backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow parameters with EMA."""
        model_state = model.state_dict()
        for key in self.shadow:
            if self.shadow[key].is_floating_point():
                self.shadow[key].mul_(self.decay).add_(
                    model_state[key], alpha=1.0 - self.decay
                )
            else:
                self.shadow[key].copy_(model_state[key])

    def apply_shadow(self, model: nn.Module):
        """Replace model params with shadow (for eval). Call restore() after."""
        self.backup = deepcopy(model.state_dict())
        model.load_state_dict(self.shadow)

    def restore(self, model: nn.Module):
        """Restore original model params after eval."""
        model.load_state_dict(self.backup)
        self.backup = {}

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, state_dict, device=None):
        """Load EMA state with backward-compatible key merging.

        New model parameters may exist that older checkpoints never tracked
        (e.g., newly introduced head branches). In that case we keep the
        currently initialized EMA tensors for missing keys.
        """
        incoming_shadow = state_dict.get("shadow", {})
        self.decay = state_dict.get("decay", self.decay)

        merged_shadow = {}
        missing_keys = []
        shape_mismatch_keys = []

        for key, current_tensor in self.shadow.items():
            loaded_tensor = incoming_shadow.get(key)
            if loaded_tensor is None:
                missing_keys.append(key)
                merged = current_tensor
            elif loaded_tensor.shape != current_tensor.shape:
                shape_mismatch_keys.append(key)
                merged = current_tensor
            else:
                merged = loaded_tensor
            if device is not None:
                merged = merged.to(device)
            merged_shadow[key] = merged

        self.shadow = merged_shadow

        if missing_keys:
            preview = missing_keys[:6]
            suffix = " ..." if len(missing_keys) > 6 else ""
            print(
                f"  WARNING: EMA checkpoint missing {len(missing_keys)} keys; "
                f"keeping current-init EMA for: {preview}{suffix}"
            )
        if shape_mismatch_keys:
            preview = shape_mismatch_keys[:6]
            suffix = " ..." if len(shape_mismatch_keys) > 6 else ""
            print(
                f"  WARNING: EMA checkpoint has {len(shape_mismatch_keys)} shape-mismatched keys; "
                f"keeping current-init EMA for: {preview}{suffix}"
            )
