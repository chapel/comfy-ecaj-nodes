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

        Returns divergence with same shape as m (including leading singleton).

        Args:
            D1: First direction tensor
            D2: Second direction tensor

        Returns:
            Direction divergence as 1 - cosine_similarity
        """
        if D1.dim() == 2:
            # Per-column cosine similarity (Linear)
            cos_sim = F.cosine_similarity(D1, D2, dim=0)  # (in_features,)
            return (1 - cos_sim).unsqueeze(0)  # (1, in_features)

        elif D1.dim() == 4:  # Conv2D
            D1_flat = D1.view(D1.shape[0], -1)
            D2_flat = D2.view(D2.shape[0], -1)
            cos_sim = F.cosine_similarity(D1_flat, D2_flat, dim=0)  # (in*h*w,)
            div = 1 - cos_sim
            # Reshape with leading singleton
            return div.view(1, D1.shape[1], D1.shape[2], D1.shape[3])

        elif D1.dim() == 3:  # Conv1D
            D1_flat = D1.view(D1.shape[0], -1)
            D2_flat = D2.view(D2.shape[0], -1)
            cos_sim = F.cosine_similarity(D1_flat, D2_flat, dim=0)
            div = 1 - cos_sim
            return div.view(1, D1.shape[1], D1.shape[2])

        elif D1.dim() == 1:  # 1D weights
            # No direction for 1D
            return torch.zeros(1, device=D1.device, dtype=D1.dtype)

        else:
            raise ValueError(f"Unsupported tensor dimension: {D1.dim()}")

    def compute_direction_divergence_batched(
        self, D1: torch.Tensor, D2: torch.Tensor
    ) -> torch.Tensor:
        """Batched per-column direction divergence.

        Args:
            D1, D2: [B, out, in, ...] — batched direction tensors

        Returns:
            Divergence with shape matching batched m: [B, 1, in, ...]
        """
        logical_ndim = D1.ndim - 1  # subtract batch dim

        if logical_ndim == 2:
            # Linear: [B, out, in] — cosine along out (dim=1)
            cos_sim = F.cosine_similarity(D1, D2, dim=1)  # [B, in]
            return (1 - cos_sim).unsqueeze(1)  # [B, 1, in]

        elif logical_ndim == 4:
            # Conv2D: [B, out, in, h, w]
            B, out_c, in_c, h, w = D1.shape
            D1_flat = D1.view(B, out_c, -1)
            D2_flat = D2.view(B, out_c, -1)
            cos_sim = F.cosine_similarity(D1_flat, D2_flat, dim=1)  # [B, in*h*w]
            div = 1 - cos_sim
            return div.view(B, 1, in_c, h, w)

        elif logical_ndim == 3:
            # Conv1D: [B, out, in, k]
            B, out_c, in_c, k = D1.shape
            D1_flat = D1.view(B, out_c, -1)
            D2_flat = D2.view(B, out_c, -1)
            cos_sim = F.cosine_similarity(D1_flat, D2_flat, dim=1)
            div = 1 - cos_sim
            return div.view(B, 1, in_c, k)

        elif logical_ndim == 1:
            # 1D weights — no direction divergence
            B = D1.shape[0]
            return torch.zeros(B, 1, device=D1.device, dtype=D1.dtype)

        else:
            raise ValueError(f"Unsupported batched tensor ndim: {D1.ndim}")
