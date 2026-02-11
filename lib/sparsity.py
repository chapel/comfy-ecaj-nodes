"""Differentiable sparsity methods for WIDEN routing.

Provides sparsemax and entmax implementations for sparse probability distributions.
These are used as alternatives to softmax for per-parameter model routing.

This module is pure torch and stdlib - no ComfyUI imports.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

logger = logging.getLogger(__name__)


class SparsemaxFunction(Function):
    """Sparsemax activation function with custom backward pass.

    Maps input to probability simplex with sparse output.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Sparsemax forward pass.

        Args:
            ctx: Context for saving tensors
            input: Input tensor
            dim: Dimension along which to apply sparsemax

        Returns:
            Sparse probability distribution
        """
        # Translate by max for numerical stability
        input_shifted = input - input.max(dim=dim, keepdim=True)[0]

        # Sort input in descending order
        sorted_input, _ = torch.sort(input_shifted, dim=dim, descending=True)

        # Compute cumulative sum
        cumsum = torch.cumsum(sorted_input, dim=dim)

        # Find the threshold
        k_array = torch.arange(
            1, sorted_input.size(dim) + 1, device=input.device, dtype=input.dtype
        )

        if dim == -1:
            k_array = k_array.view(1, -1)
        else:
            shape = [1] * input.ndim
            shape[dim] = -1
            k_array = k_array.view(*shape)

        # Compute threshold
        support = sorted_input - (cumsum - 1) / k_array > 0
        k_z = support.sum(dim=dim, keepdim=True).float()

        # Compute tau (threshold)
        tau_sum = cumsum.gather(dim, k_z.long() - 1)
        tau = (tau_sum - 1) / k_z

        # Compute output
        output = torch.clamp(input_shifted - tau, min=0)

        # Save for backward
        ctx.save_for_backward(output)
        ctx.dim = dim

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Sparsemax backward pass.

        Args:
            ctx: Context with saved tensors
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradient w.r.t. input
        """
        (output,) = ctx.saved_tensors
        dim = ctx.dim

        # Non-zero mask
        nonzero_mask = output > 0

        # Compute gradient
        # For non-zero outputs, gradient flows with Jacobian correction
        sum_grad = (grad_output * nonzero_mask).sum(dim=dim, keepdim=True)
        nonzero_count = nonzero_mask.sum(dim=dim, keepdim=True).float()

        grad_input = nonzero_mask * (grad_output - sum_grad / (nonzero_count + 1e-12))

        return grad_input, None


class Sparsemax(nn.Module):
    """Sparsemax activation module."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return SparsemaxFunction.apply(input, self.dim)


class EntmaxFunction(Function):
    """Entmax activation function with alpha parameter.

    Generalizes softmax (alpha=1) and sparsemax (alpha=2).

    Note: The alpha==1.0 (softmax) and alpha==2.0 (sparsemax) fast paths
    delegate to F.softmax and SparsemaxFunction respectively without saving
    context for backward. This is safe because entmax is used only in
    inference (within torch.no_grad() blocks) in the WIDEN pipeline.
    If gradient support is needed for alpha==1/2, dedicated backward
    paths must be implemented.
    """

    @staticmethod
    def forward(
        ctx, input: torch.Tensor, alpha: float = 1.5, dim: int = -1, n_iter: int = 50
    ) -> torch.Tensor:
        """Entmax forward pass using bisection algorithm.

        Args:
            ctx: Context for saving tensors
            input: Input tensor
            alpha: Entmax alpha parameter
            dim: Dimension for entmax
            n_iter: Number of bisection iterations

        Returns:
            Entmax output
        """
        ctx._entmax_alpha = alpha

        if alpha == 1.0:
            # Special case: softmax (no ctx saved — inference only)
            return F.softmax(input, dim=dim)

        if alpha == 2.0:
            # Special case: sparsemax (no ctx saved — delegates to SparsemaxFunction)
            return SparsemaxFunction.apply(input, dim)

        # General case: use bisection algorithm
        input_shifted = input - input.max(dim=dim, keepdim=True)[0]

        # Bisection bounds
        tau_min = input_shifted.min(dim=dim, keepdim=True)[0] - 1
        tau_max = input_shifted.max(dim=dim, keepdim=True)[0]

        # Bisection iterations
        for _ in range(n_iter):
            tau = (tau_min + tau_max) / 2

            # Compute entmax with current tau
            y = torch.clamp((alpha - 1) * input_shifted - tau, min=0)
            y = y ** (1 / (alpha - 1))

            # Check constraint
            constraint = y.sum(dim=dim, keepdim=True) - 1

            # Update bounds
            tau_min = torch.where(constraint < 0, tau, tau_min)
            tau_max = torch.where(constraint > 0, tau, tau_max)

        # Final computation
        tau = (tau_min + tau_max) / 2
        output = torch.clamp((alpha - 1) * input_shifted - tau, min=0)
        output = output ** (1 / (alpha - 1))

        # Normalize (for numerical stability)
        output = output / (output.sum(dim=dim, keepdim=True) + 1e-12)

        # Save for backward
        ctx.save_for_backward(output, input)
        ctx.alpha = alpha
        ctx.dim = dim

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        """Entmax backward pass.

        Args:
            ctx: Context with saved tensors
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradients w.r.t. input (and None for other args)

        Raises:
            RuntimeError: If called for alpha==1.0 or alpha==2.0, which use
                fast paths that do not save context for backward.
        """
        alpha = getattr(ctx, "_entmax_alpha", None)
        if alpha is not None and alpha in (1.0, 2.0):
            raise RuntimeError(
                f"EntmaxFunction backward is not supported for alpha={alpha}. "
                f"The alpha=={alpha} fast path does not save context for backward. "
                f"Use torch.no_grad() at the call site or use alpha != {alpha}."
            )

        output, _ = ctx.saved_tensors
        alpha = ctx.alpha
        dim = ctx.dim

        # Compute gradient
        # For entmax, gradient has special form based on alpha
        nonzero_mask = output > 0

        if alpha == 2.0:
            # Sparsemax gradient
            sum_grad = (grad_output * nonzero_mask).sum(dim=dim, keepdim=True)
            nonzero_count = nonzero_mask.sum(dim=dim, keepdim=True).float()
            grad_input = nonzero_mask * (grad_output - sum_grad / (nonzero_count + 1e-12))
        else:
            # General entmax gradient
            output_pow = output ** (2 - alpha)
            grad_sum = (grad_output * output_pow).sum(dim=dim, keepdim=True)
            grad_input = output_pow * (grad_output - grad_sum * output)

        return grad_input, None, None, None


class Entmax(nn.Module):
    """Entmax activation module.

    Used in inference only (within torch.no_grad() blocks) for WIDEN routing.
    The alpha==1.0 and alpha==2.0 fast paths do not support gradients.
    """

    def __init__(self, alpha: float = 1.5, dim: int = -1, n_iter: int = 50):
        super().__init__()
        self.alpha = alpha
        self.dim = dim
        self.n_iter = n_iter

    @torch.no_grad()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return EntmaxFunction.apply(input, self.alpha, self.dim, self.n_iter)


def sparsemax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Functional sparsemax.

    Args:
        input: Input tensor
        dim: Dimension for sparsemax

    Returns:
        Sparse probability distribution
    """
    return SparsemaxFunction.apply(input, dim)


@torch.no_grad()
def entmax(
    input: torch.Tensor, alpha: float = 1.5, dim: int = -1, n_iter: int = 50
) -> torch.Tensor:
    """Functional entmax.

    Used in inference only. The alpha==1.0 and alpha==2.0 fast paths
    do not support gradients.

    Args:
        input: Input tensor
        alpha: Entmax alpha (1=softmax, 2=sparsemax)
        dim: Dimension for entmax
        n_iter: Bisection iterations

    Returns:
        Entmax output
    """
    return EntmaxFunction.apply(input, alpha, dim, n_iter)
