"""WIDEN (Weight Disentanglement) implementation for model merging.

Core algorithm for importance-based parameter routing in LoRA merging.
Provides filter_delta (single-model) and merge_weights (multi-model) operations
with both per-key and batched variants.

This module is pure torch and stdlib - no ComfyUI imports.
"""

import logging
from dataclasses import dataclass

import torch

from .divergence import DivergenceCalculator
from .numerical_config import NumericalConfig
from .ranking import RankingMechanism
from .sparsity import Entmax, Sparsemax

logger = logging.getLogger(__name__)


@dataclass
class WIDENConfig:
    """Configuration for WIDEN merging.

    # AC: @widen-core ac-7
    Default values: ranking_strategy=percentile, sparsity_method=softmax, s_calibration=1.0
    """

    t_factor: float = 1.0  # Threshold factor for important params (-1 for exact averaging)
    s_calibration: float = 1.0  # Score calibration value
    ranking_strategy: str = "percentile"  # percentile, zscore, minmax, soft
    sparsity_method: str = "softmax"  # softmax, sparsemax, entmax
    calibration_mode: str = "overwrite"  # overwrite or multiplicative
    dtype: torch.dtype = torch.float32


class WeightDisentangler:
    """Disentangle weights into magnitude and direction components - COLUMN-WISE."""

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        numerical_config: NumericalConfig | None = None,
    ):
        self.dtype = dtype
        self.numerical_config = numerical_config or NumericalConfig(dtype)

    def get_eps(self) -> float:
        """Dtype-aware epsilon."""
        if self.dtype in [torch.float16, torch.bfloat16]:
            return 1e-6
        return 1e-12

    # ------------------------------------------------------------------
    # Scalar methods: thin wrappers around batched counterparts
    # ------------------------------------------------------------------

    def disentangle_linear(self, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Column-wise disentanglement for Linear layers.

        Thin wrapper: unsqueeze -> batched -> squeeze.

        Args:
            W: Weight tensor of shape (out_features, in_features)

        Returns:
            m: Column magnitudes with leading singleton (1, in_features)
            D: Column-normalized directions (out_features, in_features)
        """
        m, D = self.disentangle_linear_batched(W.unsqueeze(0))
        return m.squeeze(0), D.squeeze(0)

    def disentangle_conv2d(self, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Column-wise disentanglement for Conv2D layers.

        Thin wrapper: unsqueeze -> batched -> squeeze.

        Args:
            W: Weight tensor of shape (out_channels, in_channels, h, w)

        Returns:
            m: Per-position magnitudes (1, in_channels, h, w)
            D: Column-normalized directions (out_channels, in_channels, h, w)
        """
        m, D = self.disentangle_conv2d_batched(W.unsqueeze(0))
        return m.squeeze(0), D.squeeze(0)

    def disentangle_conv1d(self, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Handle Conv1d layers.

        Thin wrapper: unsqueeze -> batched -> squeeze.

        Args:
            W: Weight tensor (out_channels, in_channels, kernel_size)

        Returns:
            m: Magnitudes (1, in_channels, kernel_size)
            D: Normalized directions (out_channels, in_channels, kernel_size)
        """
        m, D = self.disentangle_conv1d_batched(W.unsqueeze(0))
        return m.squeeze(0), D.squeeze(0)

    def disentangle_norm_weights(self, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Handle LayerNorm/GroupNorm weights (1D).

        Thin wrapper: unsqueeze -> batched -> result (shapes already match).

        Args:
            W: 1D weight tensor

        Returns:
            m: Magnitude (absolute values) with shape (1, features)
            D: Sign only with shape (1, features)
        """
        # unsqueeze(0) gives [1, features]; batched returns [1, features] for both
        return self.disentangle_norm_batched(W.unsqueeze(0))

    # ------------------------------------------------------------------
    # Batched disentanglement methods (canonical implementations)
    # ------------------------------------------------------------------

    def disentangle_linear_batched(self, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched column-wise disentanglement for Linear layers.

        Args:
            W: [B, out_features, in_features]

        Returns:
            m: [B, 1, in_features]
            D: [B, out_features, in_features]
        """
        # Column norms along out_features (dim=1)
        m = self.numerical_config.safe_norm(W, p=2, dim=1, keepdim=True, use_fp64=True)

        finfo = torch.finfo(W.dtype)
        degenerate_threshold = 64 * finfo.tiny

        D = torch.where(m > degenerate_threshold, W / m, torch.zeros_like(W))
        return m, D

    def disentangle_conv2d_batched(self, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched column-wise disentanglement for Conv2D layers.

        Args:
            W: [B, out_channels, in_channels, h, w]

        Returns:
            m: [B, 1, in_channels, h, w]
            D: [B, out_channels, in_channels, h, w]
        """
        B, out_c, in_c, h, w = W.shape
        W_flat = W.reshape(B, out_c, -1)  # (B, out_c, in_c*h*w)

        m = self.numerical_config.safe_norm(W_flat, p=2, dim=1, keepdim=True, use_fp64=True)

        finfo = torch.finfo(W.dtype)
        degenerate_threshold = 64 * finfo.tiny

        D_flat = torch.where(m > degenerate_threshold, W_flat / m, torch.zeros_like(W_flat))

        m = m.reshape(B, 1, in_c, h, w)
        D = D_flat.reshape_as(W)
        return m, D

    def disentangle_conv1d_batched(self, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched column-wise disentanglement for Conv1D layers.

        Args:
            W: [B, out_channels, in_channels, kernel_size]

        Returns:
            m: [B, 1, in_channels, kernel_size]
            D: [B, out_channels, in_channels, kernel_size]
        """
        B, out_c, in_c, k = W.shape
        W_flat = W.reshape(B, out_c, -1)

        m = self.numerical_config.safe_norm(W_flat, p=2, dim=1, keepdim=True, use_fp64=True)

        finfo = torch.finfo(W.dtype)
        degenerate_threshold = 64 * finfo.tiny

        D_flat = torch.where(m > degenerate_threshold, W_flat / m, torch.zeros_like(W_flat))

        return m.reshape(B, 1, in_c, k), D_flat.reshape_as(W)

    def disentangle_norm_batched(self, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched disentanglement for 1D norm/bias params.

        Args:
            W: [B, features]

        Returns:
            m: [B, features] (absolute values)
            D: [B, features] (signs)
        """
        return W.abs(), torch.sign(W)


class WIDEN:
    """WIDEN (Weight Disentanglement) merger.

    Key components:
    1. Column-wise disentanglement into magnitude and direction
    2. Vectorized j/k ranking for uniform importance distribution
    3. Pluggable sparsity methods (softmax, sparsemax, entmax)
    4. Separate M/D pipelines for magnitude and direction

    # AC: @widen-core ac-6
    Internal computation uses fp32 for numerical stability.
    """

    def __init__(self, config: WIDENConfig | None = None):
        """Initialize WIDEN merger with configuration.

        # AC: @widen-core ac-7
        Default config: ranking_strategy=percentile, sparsity_method=softmax, s_calibration=1.0
        """
        config = config or WIDENConfig()
        self.config = config

        # Create central numerical config and thread through all components
        self.numerical_config = NumericalConfig(dtype=config.dtype)

        self.disentangler = WeightDisentangler(
            dtype=config.dtype, numerical_config=self.numerical_config
        )
        self.divergence_calc = DivergenceCalculator(
            dtype=config.dtype, numerical_config=self.numerical_config
        )
        self.ranker = RankingMechanism(
            strategy=config.ranking_strategy, numerical_config=self.numerical_config
        )

        # Set sparsity function
        if config.sparsity_method == "softmax":
            self.sparsity_fn = torch.softmax
        elif config.sparsity_method == "sparsemax":
            self._sparsemax = Sparsemax(dim=0)
            self.sparsity_fn = lambda x, dim: self._sparsemax(x)
        elif config.sparsity_method == "entmax":
            self._entmax = Entmax(alpha=1.5, dim=0)
            self.sparsity_fn = lambda x, dim: self._entmax(x)
        else:
            raise ValueError(
                f"Unknown sparsity_method: {config.sparsity_method!r}. "
                f"Valid options: 'softmax', 'sparsemax', 'entmax'"
            )

        # Parameters
        self.t_factor = config.t_factor
        self.s_calibration = config.s_calibration
        self.ranking_strategy = config.ranking_strategy

        logger.debug(
            f"WIDEN initialized: t_factor={config.t_factor}, "
            f"sparsity={config.sparsity_method}, ranking={config.ranking_strategy}"
        )

    def filter_delta(
        self,
        lora_applied: torch.Tensor,
        backbone: torch.Tensor,
    ) -> torch.Tensor:
        """Filter a single model's delta using WIDEN importance analysis.

        Thin wrapper: unsqueeze -> filter_delta_batched -> squeeze.

        # AC: @widen-core ac-1
        Importance-filtered delta is returned with low-importance parameters zeroed.

        # AC: @widen-core ac-6
        Internal computation uses fp32 for numerical stability.

        Args:
            lora_applied: Weight tensor with LoRA applied (base + strength * delta)
            backbone: Original base weight tensor

        Returns:
            backbone + filtered_delta
        """
        return self.filter_delta_batched(
            lora_applied.unsqueeze(0), backbone.unsqueeze(0)
        ).squeeze(0)

    def merge_weights(
        self,
        weights_list: list[torch.Tensor],
        backbone: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Merge multiple weight tensors using WIDEN.

        Thin wrapper: unsqueeze -> merge_weights_batched -> squeeze.
        Handles backbone=None cases before delegating.

        # AC: @widen-core ac-2
        Each parameter is routed to the most-important contributor via calibrated softmax.

        # AC: @widen-core ac-6
        Internal computation uses fp32 for numerical stability.

        Args:
            weights_list: List of weight tensors to merge
            backbone: Reference backbone weights (required for 2D+)

        Returns:
            Merged weight tensor
        """
        if not weights_list:
            raise ValueError("weights_list must not be empty")

        # Handle backbone=None cases that the batched path doesn't support
        if self.t_factor < 0 and backbone is None:
            return torch.stack(weights_list).mean(dim=0)

        if weights_list[0].dim() == 1 and backbone is None:
            return torch.stack(weights_list).mean(dim=0)

        if backbone is None:
            backbone = weights_list[0].clone()

        # Delegate to batched with B=1
        batched_weights = [w.unsqueeze(0) for w in weights_list]
        result = self.merge_weights_batched(batched_weights, backbone.unsqueeze(0))
        return result.squeeze(0)

    def filter_delta_batched(
        self,
        lora_applied: torch.Tensor,
        backbone: torch.Tensor,
    ) -> torch.Tensor:
        """Batched filter_delta: inputs have leading batch dim [B, ...].

        # AC: @widen-core ac-3
        Results match per-key variants applied individually.

        # AC: @widen-core ac-8
        Non-OOM errors fall back to unfiltered delta passthrough with warning.

        Args:
            lora_applied: [B, *param_shape] -- base + LoRA delta
            backbone: [B, *param_shape] -- original base weights

        Returns:
            [B, *param_shape] -- backbone + filtered delta
        """
        try:
            with torch.no_grad():
                delta = lora_applied - backbone

                if self.t_factor < 0:
                    return backbone + delta

                eps = self.numerical_config.get_adaptive_epsilon(delta)

                # 1D path (biases, norms) -- ndim=2 means [B, features]
                if lora_applied.ndim == 2:
                    mag_delta = torch.abs(delta)

                    # Per-sample variance check -- flat samples pass through
                    var = mag_delta.var(dim=1, keepdim=True)  # [B, 1]
                    flat_mask = var < eps  # [B, 1] bool

                    # Early exit only if ALL samples are flat
                    if flat_mask.all():
                        return backbone + delta

                    importance = self.ranker.rank_weights_batched(mag_delta)
                    mean_importance = importance.mean(dim=1, keepdim=True)
                    threshold = self.t_factor * mean_importance
                    threshold = torch.clamp(threshold, min=eps)

                    mask = torch.where(
                        importance >= threshold,
                        torch.ones_like(importance),
                        importance / threshold,
                    )
                    mask = torch.nan_to_num(mask, nan=1.0, posinf=1.0, neginf=0.0)

                    # Blend: flat samples get ones mask (passthrough), others get filtered
                    mask = torch.where(flat_mask, torch.ones_like(mask), mask)

                    return backbone + mask * delta

                # 2D+ path
                m_lora, D_lora = self._disentangle_batched(lora_applied)
                m_base, D_base = self._disentangle_batched(backbone)

                delta_m = torch.abs(m_lora - m_base)
                delta_D = self.divergence_calc.compute_direction_divergence_batched(
                    D_lora, D_base
                )

                # Per-sample variance check
                combined_raw = delta_m + delta_D
                spatial_dims = tuple(range(1, combined_raw.ndim))
                var = combined_raw.var(dim=spatial_dims, keepdim=True)  # [B, 1, ...]
                flat_mask = var < eps  # [B, 1, ...] bool

                # Early exit only if ALL samples are flat
                if flat_mask.all():
                    return backbone + delta

                ranked_m = self.ranker.rank_weights_batched(delta_m)
                ranked_D = self.ranker.rank_weights_batched(delta_D)
                importance = (ranked_m + ranked_D) / 2

                spatial_dims = tuple(range(1, importance.ndim))
                mean_importance = importance.mean(dim=spatial_dims, keepdim=True)
                threshold = self.t_factor * mean_importance
                threshold = torch.clamp(threshold, min=eps)

                mask = torch.where(
                    importance >= threshold,
                    torch.ones_like(importance),
                    importance / threshold,
                )
                mask = torch.nan_to_num(mask, nan=1.0, posinf=1.0, neginf=0.0)

                # Blend: flat samples get ones mask (passthrough), others get filtered
                # flat_mask shape [B,1,...] broadcasts to match mask shape
                mask = torch.where(flat_mask, torch.ones_like(mask), mask)

                return backbone + mask * delta

        except torch.cuda.OutOfMemoryError:
            raise  # Let OOM propagate
        except Exception as e:
            # AC: @widen-core ac-8
            logger.warning(f"filter_delta_batched error, using passthrough: {e}")
            return lora_applied

    def merge_weights_batched(
        self,
        weights_list: list[torch.Tensor],
        backbone: torch.Tensor,
    ) -> torch.Tensor:
        """Batched merge_weights: inputs have leading batch dim [B, ...].

        # AC: @widen-core ac-3
        Results match per-key variants applied individually.

        # AC: @widen-core ac-9
        Non-OOM errors fall back to simple averaging with warning.

        Args:
            weights_list: List of N tensors, each [B, *param_shape]
            backbone: [B, *param_shape]

        Returns:
            Merged tensor [B, *param_shape]
        """
        if not weights_list:
            raise ValueError("weights_list must not be empty")

        try:
            N = len(weights_list)

            # Fast-path for t<0 (exact averaging)
            if self.t_factor < 0:
                W_merged = backbone.clone()
                for W in weights_list:
                    W_merged += (1.0 / N) * (W - backbone)
                return W_merged

            # Route 1D params (batch dim + 1 feature dim = ndim 2)
            if weights_list[0].ndim == 2:
                return self._merge_1d_params_batched(weights_list, backbone)

            # Step 1: Disentangle all weights
            m_backbone, D_backbone = self._disentangle_batched(backbone)
            m_list, D_list, delta_W_list = [], [], []

            with torch.no_grad():
                for W in weights_list:
                    m, D = self._disentangle_batched(W)
                    m_list.append(m)
                    D_list.append(D)
                    delta_W_list.append(W - backbone)

            # Step 2: Compute divergences
            delta_m_list = [torch.abs(m - m_backbone) for m in m_list]
            delta_D_list = [
                self.divergence_calc.compute_direction_divergence_batched(D, D_backbone)
                for D in D_list
            ]

            # Step 3: Rank SEPARATELY
            ranked_m = [self.ranker.rank_weights_batched(dm) for dm in delta_m_list]
            ranked_D = [self.ranker.rank_weights_batched(dd) for dd in delta_D_list]

            # Step 3.5: Build importance masks
            important_mask_m = self._build_importance_masks(ranked_m, self.t_factor)
            important_mask_d = self._build_importance_masks(ranked_D, self.t_factor)

            # Step 4: Apply sparsity across models (dim=0)
            M = self.sparsity_fn(torch.stack(ranked_m), dim=0)
            D_scores = self.sparsity_fn(torch.stack(ranked_D), dim=0)

            # Step 5: Calibrate
            if self.t_factor >= 0 and important_mask_m is not None:
                M = self._calibrate(M, important_mask_m)
                D_scores = self._calibrate(D_scores, important_mask_d)

            # Step 6: Delta merge
            W_merged = backbone.clone()
            for n in range(N):
                S_n = (M[n] + D_scores[n]) / 2
                W_merged += S_n * delta_W_list[n]

            return W_merged

        except torch.cuda.OutOfMemoryError:
            raise  # Let OOM propagate
        except Exception as e:
            # AC: @widen-core ac-9
            logger.warning(f"merge_weights_batched error, using averaging fallback: {e}")
            W_merged = backbone.clone()
            N = len(weights_list)
            for W in weights_list:
                W_merged += (1.0 / N) * (W - backbone)
            return W_merged

    def _disentangle_by_type(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Disentangle weight based on its type/shape."""
        if weight.dim() == 4:
            return self.disentangler.disentangle_conv2d(weight)
        elif weight.dim() == 3:
            return self.disentangler.disentangle_conv1d(weight)
        elif weight.dim() == 2:
            return self.disentangler.disentangle_linear(weight)
        elif weight.dim() == 1:
            return self.disentangler.disentangle_norm_weights(weight)
        else:
            raise ValueError(f"Unsupported weight dimension: {weight.dim()}")

    def _disentangle_batched(self, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch batched disentanglement by ndim - 1."""
        logical_ndim = W.ndim - 1  # subtract batch dim
        if logical_ndim == 2:
            return self.disentangler.disentangle_linear_batched(W)
        elif logical_ndim == 4:
            return self.disentangler.disentangle_conv2d_batched(W)
        elif logical_ndim == 3:
            return self.disentangler.disentangle_conv1d_batched(W)
        elif logical_ndim == 1:
            return self.disentangler.disentangle_norm_batched(W)
        else:
            raise ValueError(f"Unsupported batched weight ndim: {W.ndim}")

    def _merge_1d_params(
        self,
        weights_list: list[torch.Tensor],
        backbone: torch.Tensor,
    ) -> torch.Tensor:
        """1D parameters use magnitude-only delta merge.

        Thin wrapper: unsqueeze -> batched -> squeeze.
        """
        batched_weights = [w.unsqueeze(0) for w in weights_list]
        result = self._merge_1d_params_batched(batched_weights, backbone.unsqueeze(0))
        return result.squeeze(0)

    def _merge_1d_params_batched(
        self,
        weights_list: list[torch.Tensor],
        backbone: torch.Tensor,
    ) -> torch.Tensor:
        """Batched 1D parameter merge: magnitude-only delta merge."""
        deltas = [w - backbone for w in weights_list]
        magnitudes = [torch.abs(d) for d in deltas]

        ranked = [self.ranker.rank_weights_batched(m) for m in magnitudes]
        scores = self.sparsity_fn(torch.stack(ranked), dim=0)

        merged = backbone.clone()
        for i, delta_i in enumerate(deltas):
            merged += scores[i] * delta_i
        return merged

    def _build_importance_masks(
        self, ranked_list: list[torch.Tensor], t_factor: float
    ) -> torch.Tensor | None:
        """Build importance masks from pre-softmax rankings.

        Shape-agnostic: works for both scalar and batched ranked tensors.
        """
        if t_factor < 0:
            return None

        spatial_dims = tuple(range(1, ranked_list[0].ndim))
        masks = []
        for r in ranked_list:
            mean_per_model = r.mean(dim=spatial_dims, keepdim=True)
            mask = r > t_factor * mean_per_model
            masks.append(mask)

        return torch.stack(masks, dim=0)

    def _build_importance_masks_batched(
        self,
        ranked_list: list[torch.Tensor],
        t_factor: float,
    ) -> torch.Tensor | None:
        """Build importance masks from pre-softmax rankings (batched).

        Delegates to _build_importance_masks -- the logic is shape-agnostic.
        """
        return self._build_importance_masks(ranked_list, t_factor)

    def _calibrate(self, scores: torch.Tensor, important_mask: torch.Tensor) -> torch.Tensor:
        """Apply score calibration for important parameters."""
        if self.config.calibration_mode == "overwrite":
            calibrated = torch.where(
                important_mask,
                torch.ones_like(scores) * self.s_calibration,
                scores,
            )
        elif self.config.calibration_mode == "multiplicative":
            calibrated = scores * torch.where(
                important_mask, self.s_calibration, torch.ones_like(scores)
            )
        else:
            return scores

        # Always renormalize across models (dim=0)
        return self._renormalize_across_models(calibrated, dim=0)

    def _renormalize_across_models(self, scores: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Renormalize scores across models to maintain simplex constraint."""
        eps = self.disentangler.get_eps()
        return scores / (scores.sum(dim=dim, keepdim=True) + eps)
