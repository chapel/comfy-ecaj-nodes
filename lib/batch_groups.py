"""Batch grouping primitives for parameter batching.

OpSignature groups parameters by (affecting_sets, shape, ndim) so
that identically-shaped, same-affected parameters can be processed
in a single batched GPU operation.

This module is pure torch and stdlib - no ComfyUI imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OpSignature:
    """Hashable key for grouping parameters with identical operations.

    Parameters with the same OpSignature can be batched together because
    they have identical shape and are affected by the same LoRA sets.

    # AC: @batched-executor ac-1
    Keys with identical shape and affecting sets are in the same group.
    """

    affecting_sets: frozenset  # frozenset of set_id strings
    shape: tuple  # tensor shape (without batch dim)
    ndim: int  # len(shape) — determines WIDEN dispatch


def compile_batch_groups(
    keys: list[str],
    base_state: dict[str, torch.Tensor],
    set_affected: dict[str, set[str]],
) -> dict[OpSignature, list[str]]:
    """Group parameter keys by OpSignature for batched evaluation.

    Keys in the same group have identical affecting_sets and shapes,
    so they can be stacked and processed with a single batched operation.

    # AC: @batched-executor ac-1
    Keys with identical shape and affecting sets are in the same group.

    Args:
        keys: Parameter keys to group
        base_state: Base model state dict (CPU)
        set_affected: set_id -> {affected param keys}

    Returns:
        Dict mapping OpSignature -> list of param keys
    """
    all_set_ids = list(set_affected.keys())

    groups: dict[OpSignature, list[str]] = {}
    for key in keys:
        if key not in base_state:
            continue
        affecting = frozenset(
            sid for sid in all_set_ids if key in set_affected.get(sid, set())
        )
        shape = tuple(base_state[key].shape)
        sig = OpSignature(affecting, shape, len(shape))
        groups.setdefault(sig, []).append(key)
    return groups
