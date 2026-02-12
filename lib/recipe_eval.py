"""Recipe tree evaluation engine.

Provides compile_plan() / execute_plan() for pre-compiled recipe evaluation,
plus evaluate_recipe() which combines both for one-shot use.

The compilation phase walks the frozen recipe tree once and emits a flat
list of operations (OpApplyLoRA, OpFilterDelta, OpMergeWeights).  The
execution phase replays that list per chunk using a register-based model,
avoiding repeated isinstance checks and attribute lookups.

This module is pure torch and stdlib (plus lib.recipe, lib.per_block,
lib.gpu_ops) - no ComfyUI imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .gpu_ops import apply_lora_batch_gpu
from .per_block import (
    _apply_per_block_lora_strength,
    _apply_widen_filter_per_block,
    _apply_widen_merge_per_block,
)

# ---------------------------------------------------------------------------
# Operation dataclasses (frozen, emitted by compile_plan)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpApplyLoRA:
    """Apply a LoRA set to the tensor in input_reg, write to out_reg."""

    set_id: str
    block_config: object  # BlockConfig or None
    input_reg: int
    out_reg: int


@dataclass(frozen=True)
class OpFilterDelta:
    """Run WIDEN filter_delta_batched (single-branch path)."""

    input_reg: int
    backbone_reg: int
    t_factor: float
    block_config: object  # BlockConfig or None
    use_per_block: bool
    out_reg: int


@dataclass(frozen=True)
class OpMergeWeights:
    """Run WIDEN merge_weights_batched (multi-branch path)."""

    branch_regs: tuple[int, ...]
    backbone_reg: int
    t_factor: float
    block_config: object  # BlockConfig or None
    use_per_block: bool
    out_reg: int


@dataclass(frozen=True)
class EvalPlan:
    """Pre-compiled flat evaluation plan for a recipe tree.

    ops:        Tuple of operations to execute in order.
    result_reg: Register holding the final output tensor.
    """

    ops: tuple[OpApplyLoRA | OpFilterDelta | OpMergeWeights, ...]
    result_reg: int


# ---------------------------------------------------------------------------
# Plan compiler
# ---------------------------------------------------------------------------


class _PlanCompiler:
    """Walks a recipe tree once and emits a flat list of operations.

    Register 0 is always the base_batch (set by the executor before replay).
    Registers 1..N are allocated monotonically during compilation.
    """

    def __init__(
        self,
        set_id_map: dict[int, str],
        arch: str | None,
    ) -> None:
        self._set_id_map = set_id_map
        self._arch = arch
        self._ops: list[OpApplyLoRA | OpFilterDelta | OpMergeWeights] = []
        self._next_reg = 1  # 0 is reserved for base_batch

    def _alloc_reg(self) -> int:
        r = self._next_reg
        self._next_reg += 1
        return r

    # -- recursive compile dispatch --

    def compile_node(
        self,
        node: object,
        current_base_reg: int,
    ) -> int | list[int]:
        """Compile a recipe node, returning a register (or list for Compose)."""
        from .recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge

        if isinstance(node, RecipeBase):
            return current_base_reg

        if isinstance(node, RecipeLoRA):
            return self._compile_lora(node, current_base_reg)

        if isinstance(node, RecipeCompose):
            return self._compile_compose(node, current_base_reg)

        if isinstance(node, RecipeMerge):
            return self._compile_merge(node, current_base_reg)

        raise ValueError(f"Unknown recipe node type: {type(node)}")

    def _compile_lora(self, node: object, current_base_reg: int) -> int:
        set_id = self._set_id_map.get(id(node))
        if set_id is None:
            raise RuntimeError(
                "RecipeLoRA has no set_id mapping. This indicates a bug in "
                "recipe analysis -- every RecipeLoRA must be registered in "
                f"set_id_map before evaluation. RecipeLoRA loras: {node.loras!r}"
            )
        out = self._alloc_reg()
        block_config = getattr(node, "block_config", None)
        self._ops.append(
            OpApplyLoRA(
                set_id=set_id,
                block_config=block_config,
                input_reg=current_base_reg,
                out_reg=out,
            )
        )
        return out

    def _compile_compose(self, node: object, current_base_reg: int) -> list[int]:
        branch_regs: list[int] = []
        for branch in node.branches:
            r = self.compile_node(branch, current_base_reg)
            branch_regs.append(r)
        return branch_regs

    def _compile_merge(self, node: object, current_base_reg: int) -> int:
        from .recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge

        # Evaluate base (may be chained merge or RecipeBase)
        if isinstance(node.base, RecipeMerge):
            current_base_reg = self.compile_node(current_base_reg=current_base_reg, node=node.base)
        elif isinstance(node.base, RecipeBase):
            pass  # current_base_reg stays
        else:
            raise ValueError(f"Invalid base type in RecipeMerge: {type(node.base)}")

        # Resolve backbone
        if node.backbone is not None:
            backbone_reg = 0  # original base_batch
        else:
            backbone_reg = current_base_reg

        # Compile target
        target_result = self.compile_node(node.target, current_base_reg)

        merge_block_config = getattr(node, "block_config", None)
        use_per_block = merge_block_config is not None and self._arch is not None

        if isinstance(node.target, RecipeCompose):
            branch_regs = target_result  # list[int]

            if len(branch_regs) == 1:
                # Single-branch compose → filter_delta
                out = self._alloc_reg()
                self._ops.append(
                    OpFilterDelta(
                        input_reg=branch_regs[0],
                        backbone_reg=backbone_reg,
                        t_factor=node.t_factor,
                        block_config=merge_block_config,
                        use_per_block=use_per_block,
                        out_reg=out,
                    )
                )
                return out

            # Multi-branch compose → merge_weights
            out = self._alloc_reg()
            self._ops.append(
                OpMergeWeights(
                    branch_regs=tuple(branch_regs),
                    backbone_reg=backbone_reg,
                    t_factor=node.t_factor,
                    block_config=merge_block_config,
                    use_per_block=use_per_block,
                    out_reg=out,
                )
            )
            return out

        elif isinstance(node.target, (RecipeLoRA, RecipeMerge)):
            out = self._alloc_reg()
            self._ops.append(
                OpFilterDelta(
                    input_reg=target_result,
                    backbone_reg=backbone_reg,
                    t_factor=node.t_factor,
                    block_config=merge_block_config,
                    use_per_block=use_per_block,
                    out_reg=out,
                )
            )
            return out

        else:
            raise ValueError(f"Invalid target type in RecipeMerge: {type(node.target)}")

    def build(self) -> EvalPlan:
        raise RuntimeError("Call compile_plan() instead")


def compile_plan(
    recipe_node: object,
    set_id_map: dict[int, str],
    arch: str | None,
) -> EvalPlan:
    """Pre-compile a recipe tree into a flat evaluation plan.

    Walks the tree once and emits a sequence of operations that can be
    replayed for every batch chunk without redundant isinstance checks.

    Args:
        recipe_node: Recipe tree root (typically RecipeMerge)
        set_id_map: Map of id(RecipeLoRA) -> set_id string
        arch: Architecture name for block classification (optional)

    Returns:
        EvalPlan with ops tuple and result_reg
    """
    compiler = _PlanCompiler(set_id_map, arch)
    result = compiler.compile_node(recipe_node, current_base_reg=0)

    # compile_node returns int for single result, list[int] for compose at root
    # (which shouldn't happen for a well-formed tree that ends in RecipeMerge)
    if isinstance(result, list):
        raise ValueError(
            "Recipe tree root produced a list of registers (RecipeCompose at root). "
            "Expected RecipeMerge or RecipeBase at root."
        )

    return EvalPlan(ops=tuple(compiler._ops), result_reg=result)


# ---------------------------------------------------------------------------
# Plan executor
# ---------------------------------------------------------------------------


@torch.inference_mode()
def execute_plan(
    plan: EvalPlan,
    keys: list[str],
    base_batch: torch.Tensor,
    loader: object,
    widen: object,
    device: str,
    dtype: torch.dtype,
    arch: str | None = None,
    widen_config: object | None = None,
) -> torch.Tensor:
    """Execute a pre-compiled plan on a batch of parameters.

    # AC: @exit-batched-eval ac-4
    Results remain on GPU (CPU transfer happens in patch installation phase).

    Args:
        plan: Pre-compiled EvalPlan from compile_plan()
        keys: List of B parameter keys being evaluated
        base_batch: [B, *shape] base model weights on GPU
        loader: LoRALoader with loaded LoRA data
        widen: WIDEN instance for filter/merge operations
        device: GPU device string
        dtype: Computation dtype
        arch: Architecture name for block classification (optional)
        widen_config: WIDENConfig for per-block WIDEN instances (optional)

    Returns:
        [B, *shape] merged weights on GPU
    """
    # Register file: maps register id -> tensor
    regs: dict[int, torch.Tensor] = {0: base_batch}

    # Build key_indices once (used by every OpApplyLoRA)
    key_indices = {k: i for i, k in enumerate(keys)}

    for op in plan.ops:
        # Use type(op) is X for pointer comparison (faster than isinstance)
        op_type = type(op)

        if op_type is OpApplyLoRA:
            current = regs[op.input_reg]
            delta_specs = loader.get_delta_specs(keys, key_indices, set_id=op.set_id)
            result = apply_lora_batch_gpu(keys, current, delta_specs, device, dtype)

            # AC: @lora-block-config ac-1, ac-2
            if op.block_config is not None and arch is not None:
                result = _apply_per_block_lora_strength(
                    keys, current, result, op.block_config, arch, device, dtype
                )

            regs[op.out_reg] = result

        elif op_type is OpFilterDelta:
            lora_applied = regs[op.input_reg]
            backbone = regs[op.backbone_reg]

            if op.use_per_block:
                regs[op.out_reg] = _apply_widen_filter_per_block(
                    keys, lora_applied, backbone,
                    op.block_config, arch, op.t_factor, widen_config,
                )
            else:
                regs[op.out_reg] = widen.filter_delta_batched(lora_applied, backbone)

        elif op_type is OpMergeWeights:
            branch_tensors = [regs[r] for r in op.branch_regs]
            backbone = regs[op.backbone_reg]

            if op.use_per_block:
                regs[op.out_reg] = _apply_widen_merge_per_block(
                    keys, branch_tensors, backbone,
                    op.block_config, arch, op.t_factor, widen_config,
                )
            else:
                regs[op.out_reg] = widen.merge_weights_batched(branch_tensors, backbone)

        else:
            raise ValueError(f"Unknown op type: {op_type}")

    return regs[plan.result_reg]


# ---------------------------------------------------------------------------
# Public API (backwards-compatible wrapper)
# ---------------------------------------------------------------------------


@torch.inference_mode()
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
    plan = compile_plan(recipe_node, set_id_map, arch)
    # AC: @exit-batched-eval ac-4
    # Evaluate and return - result stays on GPU
    return execute_plan(
        plan, keys, base_batch, loader, widen,
        device, dtype, arch, widen_config,
    )
