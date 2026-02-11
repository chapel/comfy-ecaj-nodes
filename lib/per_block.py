"""Per-block strength scaling and t_factor grouping.

Provides:
- _apply_per_block_lora_strength: scale LoRA deltas by per-block override
- _get_block_t_factors: group key indices by effective t_factor
- _apply_widen_filter_per_block: WIDEN filter_delta with per-block t_factor
- _apply_widen_merge_per_block: WIDEN merge_weights with per-block t_factor

This module is pure torch and stdlib (plus lib.block_classify, lib.widen) -
no ComfyUI imports.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lib.recipe import BlockConfig


def _apply_per_block_lora_strength(
    keys: list[str],
    base: torch.Tensor,
    lora_applied: torch.Tensor,
    block_config: BlockConfig,
    arch: str,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Apply per-block strength scaling to LoRA deltas.

    Computes delta = lora_applied - base, scales each key's delta by its
    per-block strength override, and returns base + scaled_delta.

    # AC: @lora-block-config ac-1
    Per-block strength scaling is applied to LoRA deltas when block_config present.

    Args:
        keys: List of B parameter keys being evaluated
        base: [B, *shape] base weights before LoRA
        lora_applied: [B, *shape] weights after LoRA application
        block_config: BlockConfig with block_overrides for strength scaling
        arch: Architecture name for block classification
        device: GPU device string
        dtype: Computation dtype

    Returns:
        [B, *shape] weights with per-block scaled LoRA deltas
    """
    from lib.block_classify import classify_key

    # Build lookup dict from block_overrides
    block_overrides = dict(block_config.block_overrides)

    # Check if any key has a non-1.0 override
    has_overrides = False
    for key in keys:
        block_group = classify_key(key, arch)
        if block_group is not None and block_group in block_overrides:
            if block_overrides[block_group] != 1.0:
                has_overrides = True
                break

    if not has_overrides:
        # All keys use default strength of 1.0 - no scaling needed
        return lora_applied

    # Compute delta and apply per-block scaling
    delta = lora_applied - base

    # Build strength multiplier for each key
    strength_multipliers = []
    for key in keys:
        block_group = classify_key(key, arch)
        if block_group is not None and block_group in block_overrides:
            strength_multipliers.append(block_overrides[block_group])
        else:
            # No override - use 1.0 (unchanged)
            strength_multipliers.append(1.0)

    # Create scaling tensor [B, 1, 1, ...] for broadcasting
    scales = torch.tensor(strength_multipliers, device=device, dtype=dtype)
    # Reshape for broadcasting: [B] -> [B, 1, 1, ...] based on delta ndim
    for _ in range(delta.dim() - 1):
        scales = scales.unsqueeze(-1)

    # Apply scaling to delta
    scaled_delta = delta * scales

    return base + scaled_delta


def _get_block_t_factors(
    keys: list[str],
    block_config: BlockConfig | None,
    arch: str | None,
    default_t_factor: float,
) -> dict[float, list[int]]:
    """Group key indices by their effective t_factor based on block classification.

    # AC: @merge-block-config ac-1
    Per-block t_factor overrides are applied based on block classification.

    # AC: @merge-block-config ac-2
    Keys not matching any block pattern use the default (global) t_factor.

    Args:
        keys: List of parameter keys
        block_config: BlockConfig with block_overrides, or None
        arch: Architecture name for block classification
        default_t_factor: Global t_factor to use when no override applies

    Returns:
        Dict mapping t_factor -> list of key indices with that t_factor
    """
    # Import here to avoid circular import at module level
    from lib.block_classify import classify_key

    # If no block_config or no arch, all keys use the default t_factor
    if block_config is None or arch is None:
        return {default_t_factor: list(range(len(keys)))}

    # Build lookup dict from block_overrides
    block_overrides = dict(block_config.block_overrides)

    # Group keys by their effective t_factor
    t_factor_groups: dict[float, list[int]] = defaultdict(list)

    for idx, key in enumerate(keys):
        block_group = classify_key(key, arch)
        if block_group is not None and block_group in block_overrides:
            t_factor = block_overrides[block_group]
        else:
            t_factor = default_t_factor
        t_factor_groups[t_factor].append(idx)

    return dict(t_factor_groups)


def _apply_widen_filter_per_block(
    keys: list[str],
    lora_applied: torch.Tensor,
    backbone: torch.Tensor,
    block_config: BlockConfig | None,
    arch: str | None,
    default_t_factor: float,
    widen_config: object,
) -> torch.Tensor:
    """Apply WIDEN filter_delta with per-block t_factor overrides.

    # AC: @merge-block-config ac-1
    Per-block t_factor overrides are applied instead of global t_factor.

    # AC: @merge-block-config ac-2
    When no block_config, global t_factor applies to all blocks.

    Args:
        keys: List of parameter keys
        lora_applied: [B, *shape] LoRA-applied weights
        backbone: [B, *shape] backbone weights for importance analysis
        block_config: BlockConfig with per-block overrides, or None
        arch: Architecture name for block classification
        default_t_factor: Global t_factor when no override applies
        widen_config: WIDENConfig template for creating per-block instances

    Returns:
        [B, *shape] filtered weights
    """
    from lib.widen import WIDEN, WIDENConfig

    # Get per-block t_factor groupings
    t_factor_groups = _get_block_t_factors(keys, block_config, arch, default_t_factor)

    # If all keys have the same t_factor, use simple path
    if len(t_factor_groups) == 1:
        t_factor = next(iter(t_factor_groups.keys()))
        if widen_config:
            cfg = replace(widen_config, t_factor=t_factor)
        else:
            cfg = WIDENConfig(t_factor=t_factor)
        widen_instance = WIDEN(cfg)
        return widen_instance.filter_delta_batched(lora_applied, backbone)

    # Multiple t_factors: process each group separately
    result = lora_applied.clone()

    for t_factor, indices in t_factor_groups.items():
        if not indices:
            continue

        # Create WIDEN instance for this t_factor
        if widen_config:
            cfg = replace(widen_config, t_factor=t_factor)
        else:
            cfg = WIDENConfig(t_factor=t_factor)
        widen_instance = WIDEN(cfg)

        # Extract batch slices for this group
        sub_lora = lora_applied[indices]
        sub_backbone = backbone[indices]

        # Apply filter
        sub_result = widen_instance.filter_delta_batched(sub_lora, sub_backbone)

        # Write back to result
        for i, idx in enumerate(indices):
            result[idx] = sub_result[i]

    return result


def _apply_widen_merge_per_block(
    keys: list[str],
    branch_results: list[torch.Tensor],
    backbone: torch.Tensor,
    block_config: BlockConfig | None,
    arch: str | None,
    default_t_factor: float,
    widen_config: object,
) -> torch.Tensor:
    """Apply WIDEN merge_weights with per-block t_factor overrides.

    # AC: @merge-block-config ac-1
    Per-block t_factor overrides are applied instead of global t_factor.

    # AC: @merge-block-config ac-2
    When no block_config, global t_factor applies to all blocks.

    Args:
        keys: List of parameter keys
        branch_results: List of N tensors, each [B, *shape]
        backbone: [B, *shape] backbone weights for importance analysis
        block_config: BlockConfig with per-block overrides, or None
        arch: Architecture name for block classification
        default_t_factor: Global t_factor when no override applies
        widen_config: WIDENConfig template for creating per-block instances

    Returns:
        [B, *shape] merged weights
    """
    from lib.widen import WIDEN, WIDENConfig

    # Get per-block t_factor groupings
    t_factor_groups = _get_block_t_factors(keys, block_config, arch, default_t_factor)

    # If all keys have the same t_factor, use simple path
    if len(t_factor_groups) == 1:
        t_factor = next(iter(t_factor_groups.keys()))
        if widen_config:
            cfg = replace(widen_config, t_factor=t_factor)
        else:
            cfg = WIDENConfig(t_factor=t_factor)
        widen_instance = WIDEN(cfg)
        return widen_instance.merge_weights_batched(branch_results, backbone)

    # Multiple t_factors: process each group separately
    result = backbone.clone()

    for t_factor, indices in t_factor_groups.items():
        if not indices:
            continue

        # Create WIDEN instance for this t_factor
        if widen_config:
            cfg = replace(widen_config, t_factor=t_factor)
        else:
            cfg = WIDENConfig(t_factor=t_factor)
        widen_instance = WIDEN(cfg)

        # Extract batch slices for this group
        sub_branches = [b[indices] for b in branch_results]
        sub_backbone = backbone[indices]

        # Apply merge
        sub_result = widen_instance.merge_weights_batched(sub_branches, sub_backbone)

        # Write back to result
        for i, idx in enumerate(indices):
            result[idx] = sub_result[i]

    return result
