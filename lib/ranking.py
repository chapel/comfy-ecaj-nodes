"""Ranking mechanisms for WIDEN parameter importance normalization.

Provides importance ranking to normalize divergence values to [0, 1].
This handles parameter change diversity between fine-tuned and pre-trained models.

This module is pure torch and stdlib - no ComfyUI imports.
"""

import logging

import torch

from .numerical_config import NumericalConfig

logger = logging.getLogger(__name__)


class RankingMechanism:
    """Rank weights by importance and normalize within each model.

    Handles parameter change diversity between fine-tuned (FT)
    and pre-trained (PT) models.
    """

    def __init__(
        self,
        strategy: str = "percentile",
        numerical_config: NumericalConfig | None = None,
    ):
        """Initialize ranking mechanism.

        Args:
            strategy: Ranking strategy (percentile, zscore, minmax, softrank)
            numerical_config: Numerical configuration for epsilon handling
        """
        self.strategy = strategy
        self.numerical_config = numerical_config or NumericalConfig()

    def exact_rank(self, divergences: torch.Tensor) -> torch.Tensor:
        """Exact j/k ranking for uniform distribution - VECTORIZED.

        Args:
            divergences: Tensor of divergence values

        Returns:
            Ranks in [0, 1] with uniform distribution
        """
        flat = divergences.flatten()
        sorted_indices = torch.argsort(flat)

        # VECTORIZED ranking - much faster than loop
        ranks = torch.empty_like(flat).scatter_(
            0,
            sorted_indices,
            torch.linspace(1 / len(flat), 1, len(flat), device=flat.device, dtype=flat.dtype),
        )

        return ranks.view_as(divergences)

    def zscore_ranking(self, divergences: torch.Tensor) -> torch.Tensor:
        """Z-score normalization with sigmoid."""
        mean = divergences.mean()
        eps = self.numerical_config.get_adaptive_epsilon(divergences)
        std = divergences.std() + eps
        z_scores = (divergences - mean) / std
        return torch.sigmoid(z_scores)

    def minmax_ranking(self, divergences: torch.Tensor) -> torch.Tensor:
        """Min-max normalization to [0, 1]."""
        min_val = divergences.min()
        max_val = divergences.max()
        eps = self.numerical_config.get_adaptive_epsilon(divergences)
        range_val = max_val - min_val + eps
        return (divergences - min_val) / range_val

    def soft_rank(self, divergences: torch.Tensor) -> torch.Tensor:
        """Differentiable ranking for gradient tests only.

        Falls back to exact rank if torch_sort not available.
        """
        try:
            from torch_sort import soft_rank as torch_soft_rank

            return torch_soft_rank(divergences, regularization=1.0)
        except ImportError:
            logger.debug("torch_sort not available, using exact rank")
            return self.exact_rank(divergences)

    def rank_weights(self, divergences: torch.Tensor, strategy: str | None = None) -> torch.Tensor:
        """Apply specified ranking strategy.

        Args:
            divergences: Divergence tensor to rank
            strategy: Optional strategy override

        Returns:
            Ranked values in [0, 1]
        """
        strategy = strategy or self.strategy

        if strategy == "percentile":
            return self.exact_rank(divergences)
        elif strategy == "zscore":
            return self.zscore_ranking(divergences)
        elif strategy == "minmax":
            return self.minmax_ranking(divergences)
        elif strategy in ["soft", "softrank"]:
            return self.soft_rank(divergences)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # ------------------------------------------------------------------
    # Batched ranking methods
    # ------------------------------------------------------------------

    def rank_weights_batched(
        self, divergences: torch.Tensor, strategy: str | None = None
    ) -> torch.Tensor:
        """Apply ranking strategy independently per batch element.

        Args:
            divergences: [B, ...] — batched divergence tensor
            strategy: Override ranking strategy

        Returns:
            Ranks [B, ...] in [0, 1]
        """
        strategy = strategy or self.strategy
        if strategy == "percentile":
            return self.exact_rank_batched(divergences)
        elif strategy == "zscore":
            return self.zscore_ranking_batched(divergences)
        elif strategy == "minmax":
            return self.minmax_ranking_batched(divergences)
        elif strategy in ["soft", "softrank"]:
            # Fallback to per-element loop for soft rank
            return self.exact_rank_batched(divergences)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def exact_rank_batched(self, divergences: torch.Tensor) -> torch.Tensor:
        """Batched exact j/k ranking. Each batch element ranked independently.

        Args:
            divergences: [B, ...] — batched divergence values

        Returns:
            Ranks [B, ...] in [0, 1] with uniform distribution per element
        """
        B = divergences.shape[0]
        spatial_shape = divergences.shape[1:]
        K = 1
        for d in spatial_shape:
            K *= d

        flat = divergences.view(B, K)  # (B, K)
        sorted_indices = torch.argsort(flat, dim=1, stable=True)  # (B, K)

        # Build rank values: linspace from 1/K to 1
        rank_vals = (
            torch.linspace(1 / K, 1, K, device=flat.device, dtype=flat.dtype)
            .unsqueeze(0)
            .expand(B, -1)
        )  # (B, K)

        ranks = torch.empty_like(flat).scatter_(1, sorted_indices, rank_vals)
        return ranks.view_as(divergences)

    def zscore_ranking_batched(self, divergences: torch.Tensor) -> torch.Tensor:
        """Batched z-score normalization with sigmoid.

        Args:
            divergences: [B, ...]

        Returns:
            Ranked values [B, ...] in (0, 1)
        """
        spatial_dims = tuple(range(1, divergences.ndim))
        eps = self.numerical_config.get_adaptive_epsilon(divergences)
        mean = divergences.mean(dim=spatial_dims, keepdim=True)
        std = divergences.std(dim=spatial_dims, keepdim=True) + eps
        z_scores = (divergences - mean) / std
        return torch.sigmoid(z_scores)

    def minmax_ranking_batched(self, divergences: torch.Tensor) -> torch.Tensor:
        """Batched min-max normalization to [0, 1].

        Args:
            divergences: [B, ...]

        Returns:
            Normalized values [B, ...] in [0, 1]
        """
        spatial_dims = tuple(range(1, divergences.ndim))
        eps = self.numerical_config.get_adaptive_epsilon(divergences)
        min_val = divergences.amin(dim=spatial_dims, keepdim=True)
        max_val = divergences.amax(dim=spatial_dims, keepdim=True)
        range_val = max_val - min_val + eps
        return (divergences - min_val) / range_val
