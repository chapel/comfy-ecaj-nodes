"""Batch grouping primitives for parameter batching.

OpSignature groups parameters by (shape, ndim) so that
identically-shaped parameters can be processed in a single
batched GPU operation. Per-set filtering is handled by the loader
during evaluation, not at the grouping stage.

This module is pure torch and stdlib (plus lib.block_classify) -
no ComfyUI imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .block_classify import classify_key


@dataclass(frozen=True)
class OpSignature:
    """Hashable key for grouping parameters with identical operations.

    Parameters with the same OpSignature can be batched together because
    they have identical shape. Per-set LoRA filtering is handled by the
    loader at evaluation time, not at grouping time.

    # AC: @batched-executor ac-1
    Keys with identical shape are in the same group.
    """

    shape: tuple  # tensor shape (without batch dim)
    ndim: int  # len(shape) — determines WIDEN dispatch


def compile_batch_groups(
    keys: list[str],
    base_state: dict[str, torch.Tensor],
    arch: str | None = None,
) -> dict[OpSignature, list[str]]:
    """Group parameter keys by OpSignature for batched evaluation.

    Groups by (shape, ndim) so identically-shaped parameters are batched
    together. Per-set LoRA filtering is handled by the loader during
    evaluation — grouping does not need to account for which sets affect
    which keys.

    When arch is provided, keys are sorted by block group within each
    OpSignature to improve locality.

    # AC: @batched-executor ac-1
    Keys with identical shape are in the same group.

    Args:
        keys: Parameter keys to group
        base_state: Base model state dict (CPU)
        arch: Architecture name for block-aware sorting (optional)

    Returns:
        Dict mapping OpSignature -> list of param keys
    """
    groups: dict[OpSignature, list[str]] = {}
    for key in keys:
        if key not in base_state:
            continue
        shape = tuple(base_state[key].shape)
        sig = OpSignature(shape, len(shape))
        groups.setdefault(sig, []).append(key)

    # Sort keys within each group by block for locality
    if arch is not None:
        for sig in groups:
            groups[sig].sort(key=lambda k: classify_key(k, arch) or "")

    return groups
