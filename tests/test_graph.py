"""Node graph integration tests — build recipe trees through node classes.

Validates the recipe graph building pipeline by instantiating node classes
and calling their FUNCTION methods directly. Uses a mock executor tree walker
that records operation sequences (filter_delta vs merge_weights) without GPU.

AC: @node-graph-testing ac-1 through ac-6
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge
from nodes.compose import WIDENComposeNode
from nodes.entry import WIDENEntryNode
from nodes.exit import _validate_recipe_tree
from nodes.lora import WIDENLoRANode
from nodes.merge import WIDENMergeNode

from .conftest import _ZIMAGE_KEYS, MockModelPatcher

# ---------------------------------------------------------------------------
# Mock executor — lightweight tree walker recording operation plan
# ---------------------------------------------------------------------------


@dataclass
class OpRecord:
    """A single operation recorded by the mock executor."""

    op: str  # "filter_delta" or "merge_weights"
    target_type: str  # e.g. "RecipeLoRA", "RecipeCompose"
    n_branches: int | None  # number of compose branches (None for filter_delta)
    depth: int  # nesting depth (0 = outermost merge)


def plan_operations(node: object, *, depth: int = 0) -> list[OpRecord]:
    """Walk a recipe tree and return the operation plan without GPU execution.

    Mirrors the dispatch logic in lib/recipe_eval.py but records operations
    instead of executing them. Operations are returned in evaluation order
    (inner merges first).

    Args:
        node: Recipe tree root (typically RecipeMerge)
        depth: Current nesting depth (for tracking evaluation order)

    Returns:
        List of OpRecord in evaluation order
    """
    ops: list[OpRecord] = []

    if isinstance(node, (RecipeBase, RecipeLoRA)):
        # Leaf nodes produce no operations
        return ops

    if isinstance(node, RecipeCompose):
        # Compose itself is not an operation — walk branches
        for branch in node.branches:
            ops.extend(plan_operations(branch, depth=depth))
        return ops

    if isinstance(node, RecipeMerge):
        # Inner base merge evaluates first (if chained)
        if isinstance(node.base, RecipeMerge):
            ops.extend(plan_operations(node.base, depth=depth + 1))

        # Walk target branches for nested operations
        if isinstance(node.target, RecipeCompose):
            for branch in node.target.branches:
                ops.extend(plan_operations(branch, depth=depth + 1))

            # Dispatch: multi-branch compose → merge_weights
            # single-branch compose → filter_delta (AC: @exit-node ac-6)
            n_branches = len(node.target.branches)
            if n_branches == 1:
                ops.append(
                    OpRecord(
                        op="filter_delta",
                        target_type="RecipeCompose",
                        n_branches=1,
                        depth=depth,
                    )
                )
            else:
                ops.append(
                    OpRecord(
                        op="merge_weights",
                        target_type="RecipeCompose",
                        n_branches=n_branches,
                        depth=depth,
                    )
                )

        elif isinstance(node.target, RecipeLoRA):
            ops.append(
                OpRecord(
                    op="filter_delta",
                    target_type="RecipeLoRA",
                    n_branches=None,
                    depth=depth,
                )
            )

        elif isinstance(node.target, RecipeMerge):
            # Inner target merge evaluates first
            ops.extend(plan_operations(node.target, depth=depth + 1))
            ops.append(
                OpRecord(
                    op="filter_delta",
                    target_type="RecipeMerge",
                    n_branches=None,
                    depth=depth,
                )
            )

        return ops

    raise ValueError(f"Unknown node type: {type(node)}")


# ---------------------------------------------------------------------------
# Helpers — build recipe graphs through node FUNCTION methods
# ---------------------------------------------------------------------------


def _make_entry(arch: str = "sdxl") -> tuple[RecipeBase, MockModelPatcher]:
    """Create a RecipeBase through the Entry node."""
    keys = {"sdxl": None, "zimage": _ZIMAGE_KEYS}.get(arch)
    if arch not in ("sdxl", "zimage"):
        raise ValueError(f"Unknown arch for test: {arch}")
    patcher = MockModelPatcher(keys=keys) if keys else MockModelPatcher()

    entry = WIDENEntryNode()
    (recipe,) = entry.entry(patcher)
    return recipe, patcher


def _make_lora(
    name: str, strength: float = 1.0, prev: RecipeLoRA | None = None
) -> RecipeLoRA:
    """Create a RecipeLoRA through the LoRA node."""
    lora_node = WIDENLoRANode()
    (recipe,) = lora_node.add_lora(name, strength, prev=prev)
    return recipe


RecipeNode = RecipeBase | RecipeLoRA | RecipeCompose | RecipeMerge


def _make_compose(*branches: RecipeNode, compose: RecipeCompose | None = None) -> RecipeCompose:
    """Create a RecipeCompose through the Compose node, accumulating branches."""
    compose_node = WIDENComposeNode()
    result = compose
    for branch in branches:
        (result,) = compose_node.compose(branch, compose=result)
    return result


def _make_merge(
    base: RecipeBase | RecipeMerge,
    target: RecipeLoRA | RecipeCompose | RecipeMerge,
    t_factor: float = 1.0,
    backbone: RecipeNode | None = None,
) -> RecipeMerge:
    """Create a RecipeMerge through the Merge node."""
    merge_node = WIDENMergeNode()
    (recipe,) = merge_node.merge(base, target, t_factor, backbone=backbone)
    return recipe


# ---------------------------------------------------------------------------
# AC-1: Entry → LoRA → Merge pipeline
# ---------------------------------------------------------------------------


class TestEntryLoRAMergePipeline:
    """AC: @node-graph-testing ac-1"""

    def test_entry_lora_merge_structure(self):
        """Entry → LoRA → Merge produces correct RecipeMerge.

        # AC: @node-graph-testing ac-1
        """
        base, patcher = _make_entry("sdxl")
        lora = _make_lora("test_lora.safetensors", strength=0.8)
        merge = _make_merge(base, lora, t_factor=0.7)

        assert isinstance(merge, RecipeMerge)
        assert merge.base is base
        assert isinstance(merge.base, RecipeBase)
        assert merge.base.arch == "sdxl"
        assert merge.base.model_patcher is patcher

        assert merge.target is lora
        assert isinstance(merge.target, RecipeLoRA)
        assert len(merge.target.loras) == 1
        assert merge.target.loras[0]["path"] == "test_lora.safetensors"
        assert merge.target.loras[0]["strength"] == 0.8

        assert merge.t_factor == 0.7
        assert merge.backbone is None

    def test_chained_loras_accumulate(self):
        """Chaining LoRA nodes produces single RecipeLoRA with all specs.

        # AC: @node-graph-testing ac-1
        """
        lora_a = _make_lora("lora_a.safetensors", strength=1.0)
        lora_chain = _make_lora("lora_b.safetensors", strength=0.5, prev=lora_a)

        assert isinstance(lora_chain, RecipeLoRA)
        assert len(lora_chain.loras) == 2
        assert lora_chain.loras[0]["path"] == "lora_a.safetensors"
        assert lora_chain.loras[1]["path"] == "lora_b.safetensors"

    def test_entry_produces_recipe_base(self):
        """Entry node wraps ModelPatcher in RecipeBase with detected arch.

        # AC: @node-graph-testing ac-1
        """
        base, patcher = _make_entry("sdxl")

        assert isinstance(base, RecipeBase)
        assert base.arch == "sdxl"
        assert base.model_patcher is patcher


# ---------------------------------------------------------------------------
# AC-2: Compose with 3 branches → merge_weights
# ---------------------------------------------------------------------------


class TestComposeThreeBranches:
    """AC: @node-graph-testing ac-2"""

    def test_compose_three_branches_uses_merge_weights(self):
        """Three-branch compose dispatches to merge_weights.

        # AC: @node-graph-testing ac-2
        """
        base, _ = _make_entry("sdxl")
        branch_a = _make_lora("lora_a.safetensors", strength=1.0)
        branch_b = _make_lora("lora_b.safetensors", strength=0.8)
        branch_c = _make_lora("lora_c.safetensors", strength=0.5)

        composed = _make_compose(branch_a, branch_b, branch_c)
        merge = _make_merge(base, composed, t_factor=1.0)

        ops = plan_operations(merge)
        assert len(ops) == 1
        assert ops[0].op == "merge_weights"
        assert ops[0].target_type == "RecipeCompose"
        assert ops[0].n_branches == 3

    def test_compose_structure_accumulates_branches(self):
        """Compose node accumulates branches in order through chained calls.

        # AC: @node-graph-testing ac-2
        """
        branch_a = _make_lora("lora_a.safetensors")
        branch_b = _make_lora("lora_b.safetensors")
        branch_c = _make_lora("lora_c.safetensors")

        composed = _make_compose(branch_a, branch_b, branch_c)

        assert isinstance(composed, RecipeCompose)
        assert len(composed.branches) == 3
        assert composed.branches[0] is branch_a
        assert composed.branches[1] is branch_b
        assert composed.branches[2] is branch_c


# ---------------------------------------------------------------------------
# AC-3: Single LoRA target → filter_delta
# ---------------------------------------------------------------------------


class TestSingleLoRAFilterDelta:
    """AC: @node-graph-testing ac-3"""

    def test_single_lora_uses_filter_delta(self):
        """Single LoRA target dispatches to filter_delta.

        # AC: @node-graph-testing ac-3
        """
        base, _ = _make_entry("sdxl")
        lora = _make_lora("test_lora.safetensors", strength=1.0)
        merge = _make_merge(base, lora, t_factor=1.0)

        ops = plan_operations(merge)
        assert len(ops) == 1
        assert ops[0].op == "filter_delta"
        assert ops[0].target_type == "RecipeLoRA"
        assert ops[0].n_branches is None

    def test_single_branch_compose_also_filter_delta(self):
        """Single-branch compose falls back to filter_delta.

        # AC: @node-graph-testing ac-3
        """
        base, _ = _make_entry("sdxl")
        lora = _make_lora("test_lora.safetensors")
        composed = _make_compose(lora)
        merge = _make_merge(base, composed, t_factor=1.0)

        ops = plan_operations(merge)
        assert len(ops) == 1
        assert ops[0].op == "filter_delta"
        assert ops[0].target_type == "RecipeCompose"
        assert ops[0].n_branches == 1


# ---------------------------------------------------------------------------
# AC-4: Chained Merge nodes — inner evaluates first
# ---------------------------------------------------------------------------


class TestChainedMergeEvaluation:
    """AC: @node-graph-testing ac-4"""

    def test_two_merge_chain_inner_first(self):
        """Inner merge in a chain evaluates before outer.

        # AC: @node-graph-testing ac-4
        """
        base, _ = _make_entry("sdxl")
        lora_inner = _make_lora("lora_inner.safetensors")
        inner_merge = _make_merge(base, lora_inner, t_factor=1.0)

        lora_outer = _make_lora("lora_outer.safetensors")
        outer_merge = _make_merge(inner_merge, lora_outer, t_factor=0.5)

        ops = plan_operations(outer_merge)
        assert len(ops) == 2
        # Inner evaluates first (higher depth)
        assert ops[0].depth > ops[1].depth
        assert ops[0].op == "filter_delta"  # inner: single LoRA
        assert ops[1].op == "filter_delta"  # outer: single LoRA

    def test_three_merge_chain_evaluation_order(self):
        """Three-level chain evaluates innermost → middle → outermost.

        # AC: @node-graph-testing ac-4
        """
        base, _ = _make_entry("sdxl")

        lora_1 = _make_lora("lora_1.safetensors")
        merge_1 = _make_merge(base, lora_1, t_factor=1.0)

        lora_2 = _make_lora("lora_2.safetensors")
        merge_2 = _make_merge(merge_1, lora_2, t_factor=0.8)

        lora_3 = _make_lora("lora_3.safetensors")
        merge_3 = _make_merge(merge_2, lora_3, t_factor=0.5)

        ops = plan_operations(merge_3)
        assert len(ops) == 3
        # Depths decrease: innermost first
        assert ops[0].depth == 2  # merge_1 (innermost)
        assert ops[1].depth == 1  # merge_2 (middle)
        assert ops[2].depth == 0  # merge_3 (outermost)

    def test_chained_merge_base_structure(self):
        """Inner merge result feeds into outer merge's base.

        # AC: @node-graph-testing ac-4
        """
        base, _ = _make_entry("sdxl")
        lora_a = _make_lora("lora_a.safetensors")
        inner = _make_merge(base, lora_a, t_factor=1.0)

        lora_b = _make_lora("lora_b.safetensors")
        outer = _make_merge(inner, lora_b, t_factor=0.5)

        # Outer merge's base IS the inner merge
        assert isinstance(outer.base, RecipeMerge)
        assert outer.base is inner
        # Inner merge's base is the original RecipeBase
        assert isinstance(outer.base.base, RecipeBase)
        assert outer.base.base is base


# ---------------------------------------------------------------------------
# AC-5: Invalid recipe graph → validation error
# ---------------------------------------------------------------------------


class TestInvalidGraphValidation:
    """AC: @node-graph-testing ac-5"""

    def test_recipe_base_as_compose_branch_rejected(self):
        """RecipeBase wired to compose branch raises clear error.

        # AC: @node-graph-testing ac-5
        """
        base, _ = _make_entry("sdxl")
        compose_node = WIDENComposeNode()

        with pytest.raises(ValueError, match="Cannot compose a raw base model"):
            compose_node.compose(base)

    def test_recipe_lora_as_merge_base_rejected(self):
        """RecipeLoRA wired to merge base raises ValueError.

        # AC: @node-graph-testing ac-5
        """
        lora = _make_lora("test_lora.safetensors")
        target = _make_lora("target_lora.safetensors")
        merge_node = WIDENMergeNode()

        with pytest.raises(ValueError, match="base must be RecipeBase or RecipeMerge"):
            merge_node.merge(lora, target, t_factor=1.0)

    def test_exit_validation_names_invalid_type_and_position(self):
        """Exit validation error includes type name and tree position.

        # AC: @node-graph-testing ac-5
        """
        base, _ = _make_entry("sdxl")

        # Manually craft an invalid tree: RecipeBase in compose branches
        invalid_compose = RecipeCompose(branches=(base,))
        invalid_merge = RecipeMerge(
            base=base, target=invalid_compose, backbone=None, t_factor=1.0
        )

        with pytest.raises(ValueError, match=r"root\.target\.branches\[0\]") as exc_info:
            _validate_recipe_tree(invalid_merge)
        # Error should name the invalid type
        assert "RecipeBase" in str(exc_info.value)

    def test_non_recipe_type_as_merge_target_rejected(self):
        """Non-recipe type at merge target raises TypeError.

        # AC: @node-graph-testing ac-5
        """
        base, _ = _make_entry("sdxl")
        merge_node = WIDENMergeNode()

        with pytest.raises(TypeError, match="target must be"):
            merge_node.merge(base, "not_a_recipe", t_factor=1.0)


# ---------------------------------------------------------------------------
# AC-6: Full hyphoria workflow
# ---------------------------------------------------------------------------


class TestHyphoriaWorkflow:
    """AC: @node-graph-testing ac-6

    Reproduces the hyphoria workflow from design doc section 6.5:
      [Entry] → [Merge t=1.0] → [Merge t=1.0] → [Merge t=0.5] → [Exit]
                    ↑ target          ↑ target         ↑ target
               [Compose 2br]    [LoRA nipples]    [LoRA Mystic]
                    ↑ branches
      A: nicegirls→nsfw1→nsfw2   B: painting→mecha
    """

    def test_hyphoria_recipe_structure(self):
        """Full hyphoria graph builds correct recipe tree through node chain.

        # AC: @node-graph-testing ac-6
        """
        # Entry: base model
        base, _ = _make_entry("sdxl")

        # Branch A: 3-LoRA chain (nicegirls → nsfw1 → nsfw2)
        branch_a = _make_lora("nicegirls.safetensors", strength=0.8)
        branch_a = _make_lora("nsfw1.safetensors", strength=0.5, prev=branch_a)
        branch_a = _make_lora("nsfw2.safetensors", strength=0.5, prev=branch_a)

        # Branch B: 2-LoRA chain (painting → mecha)
        branch_b = _make_lora("painting.safetensors", strength=1.0)
        branch_b = _make_lora("mecha.safetensors", strength=1.0, prev=branch_b)

        # Compose: 2 branches
        composed = _make_compose(branch_a, branch_b)

        # Merge 1: compose target (merge_weights), t=1.0
        merge_1 = _make_merge(base, composed, t_factor=1.0)

        # Merge 2: single LoRA target (filter_delta), t=1.0
        nipples = _make_lora("nipples.safetensors", strength=1.0)
        merge_2 = _make_merge(merge_1, nipples, t_factor=1.0)

        # Merge 3: single LoRA target (filter_delta), t=0.5
        mystic = _make_lora("Mystic.safetensors", strength=1.0)
        merge_3 = _make_merge(merge_2, mystic, t_factor=0.5)

        # Validate tree structure
        assert isinstance(merge_3, RecipeMerge)
        assert merge_3.t_factor == 0.5

        # Outer → middle → inner chain
        assert isinstance(merge_3.base, RecipeMerge)
        assert merge_3.base.t_factor == 1.0
        assert isinstance(merge_3.base.base, RecipeMerge)
        assert merge_3.base.base.t_factor == 1.0

        # Innermost merge has RecipeBase and RecipeCompose
        inner = merge_3.base.base
        assert isinstance(inner.base, RecipeBase)
        assert inner.base.arch == "sdxl"
        assert isinstance(inner.target, RecipeCompose)
        assert len(inner.target.branches) == 2

        # Branch A: 3-LoRA set
        assert isinstance(inner.target.branches[0], RecipeLoRA)
        assert len(inner.target.branches[0].loras) == 3
        assert inner.target.branches[0].loras[0]["path"] == "nicegirls.safetensors"

        # Branch B: 2-LoRA set
        assert isinstance(inner.target.branches[1], RecipeLoRA)
        assert len(inner.target.branches[1].loras) == 2
        assert inner.target.branches[1].loras[0]["path"] == "painting.safetensors"

    def test_hyphoria_operation_plan(self):
        """Hyphoria workflow produces correct operation sequence.

        Expected: merge_weights (compose), filter_delta (nipples), filter_delta (Mystic)
        In evaluation order: innermost first.

        # AC: @node-graph-testing ac-6
        """
        base, _ = _make_entry("sdxl")

        # Build the same graph as above
        branch_a = _make_lora("nicegirls.safetensors", strength=0.8)
        branch_a = _make_lora("nsfw1.safetensors", strength=0.5, prev=branch_a)
        branch_a = _make_lora("nsfw2.safetensors", strength=0.5, prev=branch_a)

        branch_b = _make_lora("painting.safetensors", strength=1.0)
        branch_b = _make_lora("mecha.safetensors", strength=1.0, prev=branch_b)

        composed = _make_compose(branch_a, branch_b)
        merge_1 = _make_merge(base, composed, t_factor=1.0)
        nipples = _make_lora("nipples.safetensors", strength=1.0)
        merge_2 = _make_merge(merge_1, nipples, t_factor=1.0)
        mystic = _make_lora("Mystic.safetensors", strength=1.0)
        merge_3 = _make_merge(merge_2, mystic, t_factor=0.5)

        ops = plan_operations(merge_3)

        # 3 operations total
        assert len(ops) == 3

        # Innermost merge (compose target) evaluates first → merge_weights
        assert ops[0].op == "merge_weights"
        assert ops[0].n_branches == 2
        assert ops[0].depth == 2

        # Middle merge (single LoRA) → filter_delta
        assert ops[1].op == "filter_delta"
        assert ops[1].target_type == "RecipeLoRA"
        assert ops[1].depth == 1

        # Outermost merge (single LoRA) → filter_delta
        assert ops[2].op == "filter_delta"
        assert ops[2].target_type == "RecipeLoRA"
        assert ops[2].depth == 0

    def test_hyphoria_passes_exit_validation(self):
        """Full hyphoria tree passes _validate_recipe_tree without error.

        # AC: @node-graph-testing ac-6
        """
        base, _ = _make_entry("sdxl")

        branch_a = _make_lora("nicegirls.safetensors", strength=0.8)
        branch_a = _make_lora("nsfw1.safetensors", strength=0.5, prev=branch_a)
        branch_a = _make_lora("nsfw2.safetensors", strength=0.5, prev=branch_a)

        branch_b = _make_lora("painting.safetensors", strength=1.0)
        branch_b = _make_lora("mecha.safetensors", strength=1.0, prev=branch_b)

        composed = _make_compose(branch_a, branch_b)
        merge_1 = _make_merge(base, composed, t_factor=1.0)
        nipples = _make_lora("nipples.safetensors", strength=1.0)
        merge_2 = _make_merge(merge_1, nipples, t_factor=1.0)
        mystic = _make_lora("Mystic.safetensors", strength=1.0)
        merge_3 = _make_merge(merge_2, mystic, t_factor=0.5)

        # Should not raise
        _validate_recipe_tree(merge_3)
