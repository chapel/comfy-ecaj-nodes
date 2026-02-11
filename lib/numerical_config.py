"""Central numerical configuration for WIDEN implementation.

Provides consistent epsilon handling across all components with:
- Device/dtype-safe epsilon tensors
- Adaptive epsilon based on tensor magnitude
- FP32 statistics for stability

This module is pure torch and stdlib - no ComfyUI imports.
"""

import torch


class NumericalConfig:
    """Central numerical configuration for all WIDEN components.

    Ensures consistent numerical handling across disentanglement,
    divergence calculation, ranking, and sparsity operations.
    """

    def __init__(self, dtype: torch.dtype = torch.float32):
        """Initialize numerical configuration.

        Args:
            dtype: Default dtype for operations
        """
        self.dtype = dtype
        self.eps_fp32 = 1e-12
        self.eps_fp16 = 1e-6
        self.eps_bf16 = 1e-6
        self.min_eps_scale = 1e-8  # For adaptive epsilon

    def get_base_epsilon(self, tensor: torch.Tensor) -> torch.Tensor:
        """Get base epsilon as tensor on same device/dtype.

        Args:
            tensor: Reference tensor for device/dtype

        Returns:
            Epsilon tensor on same device/dtype as input
        """
        if tensor.dtype == torch.float32:
            val = self.eps_fp32
        elif tensor.dtype == torch.bfloat16:
            val = self.eps_bf16
        else:  # fp16 or other
            val = self.eps_fp16

        return torch.tensor(val, dtype=tensor.dtype, device=tensor.device)

    def get_adaptive_epsilon(
        self,
        tensor: torch.Tensor,
        dim: int | tuple | None = None,
    ) -> torch.Tensor:
        """Get adaptive epsilon based on tensor magnitude.

        Computes statistics in fp32 for stability, then casts back.
        Prevents numerical instability with very small values.

        Args:
            tensor: Input tensor
            dim: Dimension(s) to compute mean over

        Returns:
            Adaptive epsilon tensor
        """
        # Compute stats in fp32 for stability
        tensor_fp32 = tensor.float()

        if dim is not None:
            col_mean = tensor_fp32.abs().mean(dim=dim, keepdim=True)
        else:
            col_mean = tensor_fp32.abs().mean()

        # Cast back to original dtype
        col_mean = col_mean.to(tensor.dtype)
        base_eps = self.get_base_epsilon(tensor)

        # Device-safe maximum operation
        adaptive_eps = torch.maximum(base_eps, self.min_eps_scale * col_mean)

        return adaptive_eps

    def safe_divide(
        self,
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        dim: int | tuple | None = None,
    ) -> torch.Tensor:
        """Safely divide tensors with adaptive epsilon.

        Args:
            numerator: Numerator tensor
            denominator: Denominator tensor
            dim: Dimension for adaptive epsilon computation

        Returns:
            Result of safe division
        """
        eps = self.get_adaptive_epsilon(denominator, dim=dim)
        return numerator / (denominator + eps)

    def safe_norm(
        self,
        tensor: torch.Tensor,
        p: float = 2.0,
        dim: int | tuple | None = None,
        keepdim: bool = True,
        use_fp64: bool = True,
    ) -> torch.Tensor:
        """Compute norm with numerical stability using scaled computation.

        Uses scaled norm computation to avoid underflow/overflow when
        squaring very small or large values.

        Args:
            tensor: Input tensor
            p: Norm order (only p=2 uses scaling)
            dim: Dimension(s) to compute norm over
            keepdim: Keep reduced dimensions
            use_fp64: Use fp64 for extreme precision (recommended for tiny scales)

        Returns:
            Norm computed stably and cast back to original dtype
        """
        if p != 2.0:
            # Non-L2 norms: compute in fp32/64 for stability
            if use_fp64:
                tensor_stable = tensor.double()
            else:
                tensor_stable = tensor.float()
            norm = torch.norm(tensor_stable, p=p, dim=dim, keepdim=keepdim)
            return norm.to(tensor.dtype)

        # L2 norm: use scaled computation to avoid squaring underflow
        # Scale by max to bring values into safe range
        if dim is None:
            # Global norm
            scale = tensor.abs().max()
            if scale == 0:
                return torch.zeros(1, dtype=tensor.dtype, device=tensor.device)

            if use_fp64:
                tensor_scaled = (tensor / scale).double()
                norm_scaled = torch.norm(tensor_scaled, p=2)
            else:
                tensor_scaled = tensor / scale
                norm_scaled = torch.norm(tensor_scaled.float(), p=2)

            return (scale * norm_scaled).to(tensor.dtype)
        else:
            # Per-dimension norm
            # Internal computation always uses keepdim=True for consistent shapes;
            # squeeze at the end if keepdim=False was requested.
            scale = tensor.abs().amax(dim=dim, keepdim=True)

            # Handle zero columns/rows
            nonzero = scale > 0
            result = torch.zeros_like(scale)

            if nonzero.any():
                # Only compute for non-zero scaled values
                if use_fp64:
                    # Use fp64 for extreme precision
                    tensor_scaled = tensor.double()
                    scale_d = scale.double()
                    # Safe division where scale > 0
                    safe_scaled = torch.where(nonzero, tensor_scaled / scale_d, 0.0)
                    norm_scaled = torch.norm(safe_scaled, p=2, dim=dim, keepdim=True)
                    result = torch.where(nonzero, scale_d * norm_scaled, 0.0).to(tensor.dtype)
                else:
                    # fp32 computation
                    tensor_scaled = tensor.float()
                    scale_f = scale.float()
                    safe_scaled = torch.where(nonzero, tensor_scaled / scale_f, 0.0)
                    norm_scaled = torch.norm(safe_scaled, p=2, dim=dim, keepdim=True)
                    result = torch.where(nonzero, scale_f * norm_scaled, 0.0).to(tensor.dtype)

            if not keepdim:
                result = result.squeeze(dim=dim)

            return result

    def safe_clamp(
        self,
        tensor: torch.Tensor,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> torch.Tensor:
        """Clamp tensor values with epsilon safety.

        Args:
            tensor: Input tensor
            min_val: Minimum value (will add epsilon if close to boundary)
            max_val: Maximum value (will subtract epsilon if close to boundary)

        Returns:
            Clamped tensor
        """
        eps = self.get_base_epsilon(tensor)

        if min_val is not None and max_val is not None:
            # Add epsilon buffer for numerical safety
            return torch.clamp(tensor, min=min_val + eps, max=max_val - eps)
        elif min_val is not None:
            return torch.clamp(tensor, min=min_val + eps)
        elif max_val is not None:
            return torch.clamp(tensor, max=max_val - eps)
        else:
            return tensor

    def __repr__(self) -> str:
        return (
            f"NumericalConfig(dtype={self.dtype}, "
            f"eps_fp32={self.eps_fp32}, "
            f"eps_fp16={self.eps_fp16}, "
            f"eps_bf16={self.eps_bf16})"
        )
