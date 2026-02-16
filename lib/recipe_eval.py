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

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from .gpu_ops import apply_lora_batch_gpu
from .per_block import (
    _apply_per_block_lora_strength,
    _apply_widen_filter_per_block,
    _apply_widen_merge_per_block,
)

if TYPE_CHECKING:
    from .recipe import (
        BlockConfig,
        RecipeCompose,
        RecipeLoRA,
        RecipeMerge,
        RecipeModel,
        RecipeNode,
    )


# ---------------------------------------------------------------------------
# Operation dataclasses (frozen, emitted by compile_plan)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpApplyLoRA:
    """Apply a LoRA set to the tensor in input_reg, write to out_reg."""

    set_id: str
    block_config: BlockConfig | None
    input_reg: int
    out_reg: int


@dataclass(frozen=True)
class OpFilterDelta:
    """Run WIDEN filter_delta_batched (single-branch path)."""

    input_reg: int
    backbone_reg: int
    t_factor: float
    block_config: BlockConfig | None
    use_per_block: bool
    out_reg: int


@dataclass(frozen=True)
class OpMergeWeights:
    """Run WIDEN merge_weights_batched (multi-branch path)."""

    branch_regs: tuple[int, ...]
    backbone_reg: int
    t_factor: float
    block_config: BlockConfig | None
    use_per_block: bool
    out_reg: int


@dataclass(frozen=True)
class OpApplyModel:
    """Load full model weights into a register for WIDEN merge.

    AC: @full-model-execution ac-2, ac-3, ac-14
    Loads checkpoint weights into a register, scaled by strength.
    OpFilterDelta/OpMergeWeights compute deltas internally.
    """

    model_id: str
    block_config: BlockConfig | None
    strength: float
    input_reg: int
    out_reg: int


_Op = OpApplyLoRA | OpFilterDelta | OpMergeWeights | OpApplyModel


@dataclass(frozen=True)
class EvalPlan:
    """Pre-compiled flat evaluation plan for a recipe tree.

    ops:        Tuple of operations to execute in order.
    result_reg: Register holding the final output tensor.
    dead_after: Per-op tuple of register ids safe to free after that op.
                Enables eager release of intermediate GPU tensors.
    """

    ops: tuple[_Op, ...]
    result_reg: int
    dead_after: tuple[tuple[int, ...], ...]


# ---------------------------------------------------------------------------
# Register liveness analysis
# ---------------------------------------------------------------------------


def _input_regs(op: _Op) -> tuple[int, ...]:
    """Return all registers read by an operation."""
    op_type = type(op)
    if op_type is OpApplyLoRA:
        return (op.input_reg,)
    if op_type is OpFilterDelta:
        return (op.input_reg, op.backbone_reg)
    if op_type is OpMergeWeights:
        return (*op.branch_regs, op.backbone_reg)
    if op_type is OpApplyModel:
        return (op.input_reg,)
    raise ValueError(f"Unknown op type: {op_type}")


def _compute_liveness(
    ops: tuple[_Op, ...],
    result_reg: int,
) -> tuple[tuple[int, ...], ...]:
    """Compute which registers can be freed after each op.

    For each register, finds the last op that reads it. After that op
    executes, the register is dead and its tensor can be released.

    Register 0 (base_batch) and result_reg are never freed — reg 0 is
    caller-owned, and result_reg is returned to the caller.
    """
    n = len(ops)
    if n == 0:
        return ()

    # last_use[reg] = index of last op that reads this register
    last_use: dict[int, int] = {}
    for i, op in enumerate(ops):
        for reg in _input_regs(op):
            last_use[reg] = i

    # Never free register 0 (caller-owned base_batch) or the result register
    last_use.pop(0, None)
    last_use.pop(result_reg, None)

    # Build dead_after: for each op index, which registers die
    dead_lists: list[list[int]] = [[] for _ in range(n)]
    for reg, last_idx in last_use.items():
        dead_lists[last_idx].append(reg)

    return tuple(tuple(d) for d in dead_lists)


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
        model_id_map: dict[int, str] | None = None,
    ) -> None:
        self._set_id_map = set_id_map
        self._arch = arch
        self._model_id_map = model_id_map or {}
        self._ops: list[_Op] = []
        self._next_reg = 1  # 0 is reserved for base_batch

    def _alloc_reg(self) -> int:
        r = self._next_reg
        self._next_reg += 1
        return r

    # -- recursive compile dispatch --

    def compile_node(
        self,
        node: RecipeNode,
        current_base_reg: int,
    ) -> int | list[int]:
        """Compile a recipe node, returning a register (or list for Compose)."""
        from .recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge, RecipeModel

        if isinstance(node, RecipeBase):
            return current_base_reg

        if isinstance(node, RecipeLoRA):
            return self._compile_lora(node, current_base_reg)

        if isinstance(node, RecipeModel):
            return self._compile_model(node, current_base_reg)

        if isinstance(node, RecipeCompose):
            return self._compile_compose(node, current_base_reg)

        if isinstance(node, RecipeMerge):
            return self._compile_merge(node, current_base_reg)

        raise ValueError(f"Unknown recipe node type: {type(node)}")

    def _compile_lora(self, node: RecipeLoRA, current_base_reg: int) -> int:
        set_id = self._set_id_map.get(id(node))
        if set_id is None:
            raise RuntimeError(
                "RecipeLoRA has no set_id mapping. This indicates a bug in "
                "recipe analysis -- every RecipeLoRA must be registered in "
                f"set_id_map before evaluation. RecipeLoRA loras: {node.loras!r}"
            )
        out = self._alloc_reg()
        self._ops.append(
            OpApplyLoRA(
                set_id=set_id,
                block_config=node.block_config,
                input_reg=current_base_reg,
                out_reg=out,
            )
        )
        return out

    def _compile_model(self, node: RecipeModel, current_base_reg: int) -> int:
        """Compile a RecipeModel node into an OpApplyModel operation.

        AC: @full-model-execution ac-2
        Emits OpApplyModel referencing the model's loader ID.
        """
        model_id = self._model_id_map.get(id(node))
        if model_id is None:
            raise RuntimeError(
                "RecipeModel has no model_id mapping. This indicates a bug in "
                "recipe analysis -- every RecipeModel must be registered in "
                f"model_id_map before evaluation. RecipeModel path: {node.path!r}"
            )
        out = self._alloc_reg()
        self._ops.append(
            OpApplyModel(
                model_id=model_id,
                block_config=node.block_config,
                strength=node.strength,
                input_reg=current_base_reg,
                out_reg=out,
            )
        )
        return out

    def _compile_compose(
        self, node: RecipeCompose, current_base_reg: int
    ) -> list[int]:
        branch_regs: list[int] = []
        for branch in node.branches:
            r = self.compile_node(branch, current_base_reg)
            branch_regs.append(r)
        return branch_regs

    def _compile_merge(self, node: RecipeMerge, current_base_reg: int) -> int:
        from .recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge, RecipeModel

        # Evaluate base (may be chained merge or RecipeBase)
        if isinstance(node.base, RecipeMerge):
            current_base_reg = self.compile_node(node.base, current_base_reg)
        elif isinstance(node.base, RecipeBase):
            pass  # current_base_reg stays
        else:
            raise ValueError(f"Invalid base type in RecipeMerge: {type(node.base)}")

        # Resolve backbone
        # AC: @exit-batched-eval ac-5
        if node.backbone is not None:
            backbone_reg = 0  # original base_batch
        else:
            backbone_reg = current_base_reg

        # Compile target
        target_result = self.compile_node(node.target, current_base_reg)

        merge_block_config = node.block_config
        use_per_block = merge_block_config is not None and self._arch is not None

        if isinstance(node.target, RecipeCompose):
            branch_regs = target_result  # list[int]

            # AC: @exit-node ac-6
            # Single-branch compose uses filter_delta (passthrough), not merge_weights
            if len(branch_regs) == 1:
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

        elif isinstance(node.target, (RecipeLoRA, RecipeMerge, RecipeModel)):
            # AC: @full-model-execution ac-4
            # RecipeModel works like RecipeLoRA — OpFilterDelta computes
            # delta (model_weights - backbone) internally via WIDEN
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


def compile_plan(
    recipe_node: RecipeNode,
    set_id_map: dict[int, str],
    arch: str | None,
    model_id_map: dict[int, str] | None = None,
) -> EvalPlan:
    """Pre-compile a recipe tree into a flat evaluation plan.

    Walks the tree once and emits a sequence of operations that can be
    replayed for every batch chunk without redundant isinstance checks.
    Also computes register liveness so execute_plan() can eagerly free
    intermediate tensors.

    Args:
        recipe_node: Recipe tree root (typically RecipeMerge)
        set_id_map: Map of id(RecipeLoRA) -> set_id string
        arch: Architecture name for block classification (optional)
        model_id_map: Map of id(RecipeModel) -> model_id string (optional)

    Returns:
        EvalPlan with ops tuple, result_reg, and dead_after liveness info
    """
    compiler = _PlanCompiler(set_id_map, arch, model_id_map)
    result = compiler.compile_node(recipe_node, current_base_reg=0)

    # compile_node returns int for single result, list[int] for compose at root
    # (which shouldn't happen for a well-formed tree that ends in RecipeMerge)
    if isinstance(result, list):
        raise ValueError(
            "Recipe tree root produced a list of registers (RecipeCompose at root). "
            "Expected RecipeMerge or RecipeBase at root."
        )

    ops = tuple(compiler._ops)
    dead_after = _compute_liveness(ops, result)
    return EvalPlan(ops=ops, result_reg=result, dead_after=dead_after)


# ---------------------------------------------------------------------------
# Plan executor
# ---------------------------------------------------------------------------


def _get_widen_for_op(widen: object, widen_config: object | None, op_t_factor: float) -> object:
    """Return a WIDEN instance with the correct t_factor for this op.

    If the op's t_factor matches the existing widen instance, reuse it.
    Otherwise create a new instance with the correct t_factor.
    """
    if widen.t_factor == op_t_factor:
        return widen
    from .widen import WIDEN, WIDENConfig
    if widen_config is not None:
        cfg = replace(widen_config, t_factor=op_t_factor)
    else:
        cfg = WIDENConfig(t_factor=op_t_factor)
    return WIDEN(cfg)


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
    model_loaders: dict[str, object] | None = None,
    domain: str = "diffusion",
) -> torch.Tensor:
    """Execute a pre-compiled plan on a batch of parameters.

    # AC: @exit-batched-eval ac-4
    Results remain on GPU (CPU transfer happens in patch installation phase).

    # AC: @full-model-execution ac-3, ac-7, ac-13
    OpApplyModel loads weights from streaming loaders per-batch.
    Weights are freed after use (streaming loaders re-read as needed).
    Only one batch of model weights per loader is on GPU at a time.

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
        model_loaders: Map of model_id -> ModelLoader for full model merging

    Returns:
        [B, *shape] merged weights on GPU
    """
    # Register file: maps register id -> tensor
    regs: dict[int, torch.Tensor] = {0: base_batch}

    # Build key_indices once (used by every OpApplyLoRA)
    key_indices = {k: i for i, k in enumerate(keys)}

    for i, op in enumerate(plan.ops):
        # Use type(op) is X for pointer comparison (faster than isinstance)
        op_type = type(op)

        if op_type is OpApplyLoRA:
            current = regs[op.input_reg]
            delta_specs = loader.get_delta_specs(keys, key_indices, set_id=op.set_id)
            result = apply_lora_batch_gpu(keys, current, delta_specs, device, dtype)

            # AC: @lora-block-config ac-1, ac-2
            if op.block_config is not None and arch is not None:
                result = _apply_per_block_lora_strength(
                    keys, current, result, op.block_config, arch, device, dtype,
                    domain,
                )

            regs[op.out_reg] = result

        elif op_type is OpApplyModel:
            # AC: @full-model-execution ac-3
            # Load model weights for this batch of keys from streaming loader
            if model_loaders is None:
                raise RuntimeError(
                    "OpApplyModel requires model_loaders but none were provided. "
                    "This indicates a bug in recipe analysis."
                )
            model_loader = model_loaders.get(op.model_id)
            if model_loader is None:
                raise RuntimeError(
                    f"OpApplyModel references model_id '{op.model_id}' but no "
                    f"loader was found. This indicates a bug in recipe analysis."
                )

            # Get weights for the batch keys (returns list of CPU tensors)
            # AC: @full-model-execution ac-7, ac-13
            # Weights are loaded per-batch and freed after use
            weight_tensors = model_loader.get_weights(keys)

            # Stack into [B, *shape] tensor and move to GPU
            stacked = torch.stack(weight_tensors, dim=0).to(device=device, dtype=dtype)

            # AC: @full-model-execution ac-14
            # Apply per-model strength: blend toward base register
            if op.strength != 1.0:
                base = regs[op.input_reg]
                stacked = base + op.strength * (stacked - base)

            # AC: @full-model-execution ac-15
            # Apply per-block strength scaling to model deltas
            if op.block_config is not None and arch is not None:
                stacked = _apply_per_block_lora_strength(
                    keys, regs[op.input_reg], stacked, op.block_config, arch, device, dtype,
                    domain,
                )

            regs[op.out_reg] = stacked

        elif op_type is OpFilterDelta:
            lora_applied = regs[op.input_reg]
            backbone = regs[op.backbone_reg]

            if op.use_per_block:
                regs[op.out_reg] = _apply_widen_filter_per_block(
                    keys, lora_applied, backbone,
                    op.block_config, arch, op.t_factor, widen_config, domain,
                )
            else:
                w = _get_widen_for_op(widen, widen_config, op.t_factor)
                regs[op.out_reg] = w.filter_delta_batched(lora_applied, backbone)

        elif op_type is OpMergeWeights:
            branch_tensors = [regs[r] for r in op.branch_regs]
            backbone = regs[op.backbone_reg]

            if op.use_per_block:
                regs[op.out_reg] = _apply_widen_merge_per_block(
                    keys, branch_tensors, backbone,
                    op.block_config, arch, op.t_factor, widen_config, domain,
                )
            else:
                w = _get_widen_for_op(widen, widen_config, op.t_factor)
                regs[op.out_reg] = w.merge_weights_batched(branch_tensors, backbone)

        else:
            raise ValueError(f"Unknown op type: {op_type}")

        # Release dead registers to free intermediate GPU tensors
        # AC: @full-model-execution ac-7
        # This frees model weights after they're no longer needed
        for dead_reg in plan.dead_after[i]:
            del regs[dead_reg]

    return regs[plan.result_reg]


# ---------------------------------------------------------------------------
# Public API (backwards-compatible wrapper)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def evaluate_recipe(
    keys: list[str],
    base_batch: torch.Tensor,
    recipe_node: RecipeNode,
    loader: object,
    widen: object,
    set_id_map: dict[int, str],
    device: str,
    dtype: torch.dtype,
    arch: str | None = None,
    widen_config: object | None = None,
    model_id_map: dict[int, str] | None = None,
    model_loaders: dict[str, object] | None = None,
    domain: str = "diffusion",
) -> torch.Tensor:
    """Evaluate a recipe tree on a batch of parameters.

    This is the core recipe tree walker for batched GPU evaluation. It recursively
    evaluates the recipe tree, dispatching to WIDEN functions based on node type:
    - RecipeMerge with RecipeCompose target -> merge_weights_batched
    - RecipeMerge with RecipeLoRA target -> filter_delta_batched
    - RecipeMerge with RecipeModel target -> filter_delta_batched (via OpApplyModel)
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
        model_id_map: Map of object id(RecipeModel) -> model_id string (optional)
        model_loaders: Map of model_id -> ModelLoader for full model merging

    Returns:
        [B, *shape] merged weights on GPU
    """
    plan = compile_plan(recipe_node, set_id_map, arch, model_id_map)
    # AC: @exit-batched-eval ac-4
    # Evaluate and return - result stays on GPU
    return execute_plan(
        plan, keys, base_batch, loader, widen,
        device, dtype, arch, widen_config, model_loaders, domain,
    )
