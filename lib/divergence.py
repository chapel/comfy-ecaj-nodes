"""Divergence calculation utilities for WIDEN.

Computes direction divergence between weight tensors using cosine similarity.
This is used as part of importance-based parameter routing.

This module is pure torch and stdlib - no ComfyUI imports.
"""

import logging

import torch
import torch.nn.functional as F

from .numerical_config import NumericalConfig

logger = logging.getLogger(__name__)


class DivergenceCalculator:
    """Calculate divergence between weight components."""

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        numerical_config: NumericalConfig | None = None,
    ):
        self.dtype = dtype
        self.numerical_config = numerical_config or NumericalConfig(dtype)
        self.eps = 1e-12 if dtype == torch.float32 else 1e-6

    def compute_direction_divergence(self, D1: torch.Tensor, D2: torch.Tensor) -> torch.Tensor:
        """Compute per-column direction divergence.

        Thin wrapper: unsqueeze -> batched -> squeeze.

        Returns divergence with same shape as m (including leading singleton).

        Args:
            D1: First direction tensor
            D2: Second direction tensor

        Returns:
            Direction divergence as 1 - cosine_similarity
        """
        result = self.compute_direction_divergence_batched(D1.unsqueeze(0), D2.unsqueeze(0))
        return result.squeeze(0)

    def compute_direction_divergence_batched(
        self, D1: torch.Tensor, D2: torch.Tensor
    ) -> torch.Tensor:
        """Batched per-column direction divergence.

        Generic implementation that handles all tensor dimensions by flattening
        non-output axes and computing cosine similarity along the output dimension.

        Args:
            D1, D2: [B, out, in, ...] -- batched direction tensors

        Returns:
            Divergence with shape matching batched m: [B, 1, in, ...]
        """
        logical_ndim = D1.ndim - 1  # subtract batch dim

        if logical_ndim == 1:
            # 1D weights -- no direction divergence
            B = D1.shape[0]
            return torch.zeros(B, 1, device=D1.device, dtype=D1.dtype)

        if logical_ndim < 1:
            raise ValueError(f"Unsupported batched tensor ndim: {D1.ndim}")

        # Generic path for logical_ndim >= 2 (Linear, Conv1D, Conv2D, etc.)
        B = D1.shape[0]
        out_dim = D1.shape[1]
        spatial_shape = D1.shape[2:]  # everything after [B, out, ...]

        # Flatten spatial dims: [B, out, *spatial] -> [B, out, flat]
        D1_flat = D1.reshape(B, out_dim, -1)
        D2_flat = D2.reshape(B, out_dim, -1)

        # Cosine similarity along output dim (dim=1)
        cos_sim = F.cosine_similarity(D1_flat, D2_flat, dim=1)  # [B, flat]
        div = 1 - cos_sim

        # Reshape to [B, 1, *spatial]
        return div.reshape(B, 1, *spatial_shape)
