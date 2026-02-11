"""Recipe tree evaluation engine.

Provides evaluate_recipe() which walks a recipe tree and dispatches
to WIDEN filter/merge operations on batched GPU tensors.

This module is pure torch and stdlib (plus lib.recipe, lib.per_block,
lib.gpu_ops) - no ComfyUI imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from .gpu_ops import apply_lora_batch_gpu
    from .per_block import (
        _apply_per_block_lora_strength,
        _apply_widen_filter_per_block,
        _apply_widen_merge_per_block,
    )
except ImportError:
    from lib.gpu_ops import apply_lora_batch_gpu
    from lib.per_block import (
        _apply_per_block_lora_strength,
        _apply_widen_filter_per_block,
        _apply_widen_merge_per_block,
    )


@dataclass
class EvalContext:
    """Shared state for recipe tree evaluation.

    Replaces closure-captured variables with an explicit typed container.
    """

    keys: list[str]
    base_batch: torch.Tensor
    loader: object
    widen: object
    set_id_map: dict[int, str]
    device: str
    dtype: torch.dtype
    arch: str | None
    widen_config: object | None


def _apply_lora_set(
    ctx: EvalContext,
    current: torch.Tensor,
    recipe_lora: object,
) -> torch.Tensor:
    """Apply a LoRA set to current weights with per-block strength scaling.

    # AC: @lora-block-config ac-1
    When block_config is present, per-block strength scaling is applied.

    # AC: @lora-block-config ac-2
    When no block_config, global strength applies uniformly.

    Args:
        ctx: Evaluation context with shared state
        current: [B, *shape] current weights on GPU
        recipe_lora: RecipeLoRA node with LoRA specs

    Returns:
        [B, *shape] weights with LoRA applied

    Raises:
        RuntimeError: If the RecipeLoRA has no set_id mapping (indicates
            a bug in recipe analysis -- every RecipeLoRA must be registered).
    """
    set_id = ctx.set_id_map.get(id(recipe_lora))
    if set_id is None:
        raise RuntimeError(
            "RecipeLoRA has no set_id mapping. This indicates a bug in "
            "recipe analysis -- every RecipeLoRA must be registered in "
            "set_id_map before evaluation. "
            f"RecipeLoRA loras: {recipe_lora.loras!r}"
        )

    # Build key_indices for DeltaSpec
    key_indices = {k: i for i, k in enumerate(ctx.keys)}

    # Get delta specs from loader scoped to this set
    delta_specs = ctx.loader.get_delta_specs(ctx.keys, key_indices, set_id=set_id)

    # Apply deltas using batched GPU application
    result = apply_lora_batch_gpu(ctx.keys, current, delta_specs, ctx.device, ctx.dtype)

    # AC: @lora-block-config ac-1, ac-2
    # Apply per-block strength scaling if block_config is present
    lora_block_config = getattr(recipe_lora, "block_config", None)
    if lora_block_config is not None and ctx.arch is not None:
        result = _apply_per_block_lora_strength(
            ctx.keys, current, result, lora_block_config, ctx.arch, ctx.device, ctx.dtype
        )

    return result


def _eval_node(
    ctx: EvalContext,
    current_base: torch.Tensor,
    node: object,
) -> torch.Tensor | list[torch.Tensor]:
    """Recursively evaluate a recipe node.

    Args:
        ctx: Evaluation context with shared state
        current_base: [B, *shape] current base weights
        node: Recipe node to evaluate

    Returns:
        [B, *shape] evaluated weights on GPU, or list of tensors for RecipeCompose
    """
    # Import here to avoid circular import
    try:
        from .recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge
    except ImportError:
        from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge

    if isinstance(node, RecipeBase):
        # Base node - return current base unchanged
        # AC: @exit-batched-eval ac-3 - base case for recursion
        return current_base

    elif isinstance(node, RecipeLoRA):
        # Apply LoRA set to current base
        # AC: @exit-batched-eval ac-2
        return _apply_lora_set(ctx, current_base, node)

    elif isinstance(node, RecipeCompose):
        # Evaluate all branches
        # AC: @exit-batched-eval ac-1
        branch_results = []
        for branch in node.branches:
            result = _eval_node(ctx, current_base, branch)
            branch_results.append(result)
        return branch_results

    elif isinstance(node, RecipeMerge):
        return _eval_merge(ctx, current_base, node)

    else:
        raise ValueError(f"Unknown recipe node type: {type(node)}")


def _resolve_backbone(
    ctx: EvalContext,
    current_base: torch.Tensor,
    node: object,
) -> torch.Tensor:
    """Determine backbone weights for WIDEN analysis.

    # AC: @exit-batched-eval ac-5
    RecipeMerge.backbone overrides the importance reference for WIDEN analysis.

    Args:
        ctx: Evaluation context
        current_base: Current base weights
        node: RecipeMerge node

    Returns:
        Backbone tensor for WIDEN analysis
    """
    if node.backbone is not None:
        # Explicit backbone override
        # Use original base as default (backbone evaluation not yet implemented)
        return ctx.base_batch
    else:
        # Use current base as backbone
        return current_base


def _eval_base_for_merge(
    ctx: EvalContext,
    current_base: torch.Tensor,
    node: object,
) -> torch.Tensor:
    """Evaluate the base of a RecipeMerge node.

    # AC: @exit-batched-eval ac-3
    Chained RecipeMerge nodes evaluate inner merges first.

    Args:
        ctx: Evaluation context
        current_base: Current base weights
        node: RecipeMerge node

    Returns:
        Evaluated base weights

    Raises:
        ValueError: If base type is invalid
    """
    try:
        from .recipe import RecipeBase, RecipeMerge
    except ImportError:
        from lib.recipe import RecipeBase, RecipeMerge

    if isinstance(node.base, RecipeMerge):
        # Recursive merge - evaluate inner merge first
        return _eval_node(ctx, current_base, node.base)
    elif isinstance(node.base, RecipeBase):
        # At the root - use provided base_batch
        return current_base
    else:
        raise ValueError(f"Invalid base type in RecipeMerge: {type(node.base)}")


def _eval_target(
    ctx: EvalContext,
    keys: list[str],
    target_result: torch.Tensor | list[torch.Tensor],
    backbone_weights: torch.Tensor,
    node: object,
) -> torch.Tensor:
    """Dispatch WIDEN operations based on target type.

    Args:
        ctx: Evaluation context
        keys: Parameter keys
        target_result: Evaluated target (tensor or list of tensors)
        backbone_weights: Backbone for WIDEN analysis
        node: RecipeMerge node

    Returns:
        Merged/filtered weights tensor
    """
    try:
        from .recipe import RecipeCompose, RecipeLoRA, RecipeMerge
    except ImportError:
        from lib.recipe import RecipeCompose, RecipeLoRA, RecipeMerge

    merge_block_config = getattr(node, "block_config", None)
    use_per_block = merge_block_config is not None and ctx.arch is not None

    if isinstance(node.target, RecipeCompose):
        # target_result is a list of branch results
        branch_results = target_result

        # AC: @exit-node ac-6
        # Single-branch compose uses filter_delta (passthrough), not merge_weights
        if len(branch_results) == 1:
            lora_applied = branch_results[0]
            if use_per_block:
                return _apply_widen_filter_per_block(
                    keys, lora_applied, backbone_weights,
                    merge_block_config, ctx.arch, node.t_factor, ctx.widen_config
                )
            else:
                return ctx.widen.filter_delta_batched(lora_applied, backbone_weights)

        # AC: @exit-batched-eval ac-1
        # Multi-branch compose: Call merge_weights_batched with all branches
        if use_per_block:
            return _apply_widen_merge_per_block(
                keys, branch_results, backbone_weights,
                merge_block_config, ctx.arch, node.t_factor, ctx.widen_config
            )
        else:
            return ctx.widen.merge_weights_batched(branch_results, backbone_weights)

    elif isinstance(node.target, RecipeLoRA):
        # AC: @exit-batched-eval ac-2
        lora_applied = target_result
        if use_per_block:
            return _apply_widen_filter_per_block(
                keys, lora_applied, backbone_weights,
                merge_block_config, ctx.arch, node.t_factor, ctx.widen_config
            )
        else:
            return ctx.widen.filter_delta_batched(lora_applied, backbone_weights)

    elif isinstance(node.target, RecipeMerge):
        # AC: @exit-batched-eval ac-3
        inner_result = target_result
        if use_per_block:
            return _apply_widen_filter_per_block(
                keys, inner_result, backbone_weights,
                merge_block_config, ctx.arch, node.t_factor, ctx.widen_config
            )
        else:
            return ctx.widen.filter_delta_batched(inner_result, backbone_weights)

    else:
        raise ValueError(f"Invalid target type in RecipeMerge: {type(node.target)}")


def _eval_merge(
    ctx: EvalContext,
    current_base: torch.Tensor,
    node: object,
) -> torch.Tensor:
    """Evaluate a RecipeMerge node.

    # AC: @exit-batched-eval ac-3
    Chained RecipeMerge nodes evaluate inner merges first.

    # AC: @exit-batched-eval ac-5
    RecipeMerge.backbone overrides the importance reference.

    # AC: @merge-block-config ac-1
    When RecipeMerge has block_config, per-block t_factor overrides are applied.

    # AC: @merge-block-config ac-2
    When no block_config is connected, global t_factor applies to all blocks.

    Args:
        ctx: Evaluation context
        current_base: Current base weights
        node: RecipeMerge node

    Returns:
        Merged weights tensor
    """
    # Evaluate the base (could be another RecipeMerge or RecipeBase)
    current_base = _eval_base_for_merge(ctx, current_base, node)

    # Determine backbone for WIDEN analysis
    backbone_weights = _resolve_backbone(ctx, current_base, node)

    # Evaluate target
    target_result = _eval_node(ctx, current_base, node.target)

    # Dispatch based on target type
    return _eval_target(ctx, ctx.keys, target_result, backbone_weights, node)


def evaluate_recipe(
    keys: list[str],
    base_batch: torch.Tensor,
    recipe_node: object,
    loader: object,
    widen: object,
    set_id_map: dict[int, str],
    device: str,
    dtype: torch.dtype,
    arch: str | None = None,
    widen_config: object | None = None,
) -> torch.Tensor:
    """Evaluate a recipe tree on a batch of parameters.

    This is the core recipe tree walker for batched GPU evaluation. It recursively
    evaluates the recipe tree, dispatching to WIDEN functions based on node type:
    - RecipeMerge with RecipeCompose target -> merge_weights_batched
    - RecipeMerge with RecipeLoRA target -> filter_delta_batched
    - Chained RecipeMerge -> recurse on inner merge first

    # AC: @exit-batched-eval ac-1
    Compose targets call merge_weights_batched with all branch results and backbone.

    # AC: @exit-batched-eval ac-2
    Single LoRA targets call filter_delta_batched with applied delta and backbone.

    # AC: @exit-batched-eval ac-3
    Chained RecipeMerge nodes evaluate inner merges first.

    # AC: @exit-batched-eval ac-4
    Results remain on GPU (CPU transfer happens in patch installation phase).

    # AC: @exit-batched-eval ac-5
    RecipeMerge.backbone overrides the importance reference for WIDEN analysis.

    # AC: @merge-block-config ac-1
    When RecipeMerge has block_config, per-block t_factor overrides are applied.

    # AC: @merge-block-config ac-2
    When no block_config is connected, global t_factor applies to all blocks.

    Args:
        keys: List of B parameter keys being evaluated
        base_batch: [B, *shape] base model weights on GPU
        recipe_node: Recipe tree root (typically RecipeMerge)
        loader: LoRALoader with loaded LoRA data
        widen: WIDEN instance for filter/merge operations
        set_id_map: Map of object id(RecipeLoRA) -> set_id string
        device: GPU device string
        dtype: Computation dtype
        arch: Architecture name for block classification (optional)
        widen_config: WIDENConfig to create per-block WIDEN instances (optional)

    Returns:
        [B, *shape] merged weights on GPU
    """
    ctx = EvalContext(
        keys=keys,
        base_batch=base_batch,
        loader=loader,
        widen=widen,
        set_id_map=set_id_map,
        device=device,
        dtype=dtype,
        arch=arch,
        widen_config=widen_config,
    )

    # AC: @exit-batched-eval ac-4
    # Evaluate and return - result stays on GPU
    return _eval_node(ctx, base_batch, recipe_node)
