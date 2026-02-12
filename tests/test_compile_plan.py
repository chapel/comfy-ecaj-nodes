"""Tests for compile_plan() — verifying correct op sequences and register wiring."""

import pytest

from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge
from lib.recipe_eval import (
    OpApplyLoRA,
    OpFilterDelta,
    OpMergeWeights,
    compile_plan,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lora(name: str = "test.safetensors") -> RecipeLoRA:
    return RecipeLoRA(loras=({"path": name, "strength": 1.0},))


def _base() -> RecipeBase:
    return RecipeBase(model_patcher=None, arch="sdxl")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimpleLoRAThenFilter:
    """Simple RecipeMerge + RecipeLoRA -> [OpApplyLoRA, OpFilterDelta].

    # AC: @exit-batched-eval ac-2
    Single LoRA targets call filter_delta_batched with applied delta and backbone.
    """

    def test_op_sequence(self):
        """# AC: @exit-batched-eval ac-2"""
        base = _base()
        lora = _lora()
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)
        set_id_map = {id(lora): "set1"}

        plan = compile_plan(merge, set_id_map, arch=None)

        assert len(plan.ops) == 2
        assert type(plan.ops[0]) is OpApplyLoRA
        assert type(plan.ops[1]) is OpFilterDelta

    def test_register_wiring(self):
        """# AC: @exit-batched-eval ac-2"""
        base = _base()
        lora = _lora()
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)
        set_id_map = {id(lora): "set1"}

        plan = compile_plan(merge, set_id_map, arch=None)

        # OpApplyLoRA: reads base (reg 0), writes to reg 1
        op_lora = plan.ops[0]
        assert op_lora.input_reg == 0
        assert op_lora.out_reg == 1
        assert op_lora.set_id == "set1"

        # OpFilterDelta: reads lora result (reg 1), backbone is base (reg 0)
        op_filter = plan.ops[1]
        assert op_filter.input_reg == 1
        assert op_filter.backbone_reg == 0
        assert plan.result_reg == op_filter.out_reg

    def test_t_factor_preserved(self):
        """# AC: @merge-block-config ac-2"""
        base = _base()
        lora = _lora()
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=0.75)
        set_id_map = {id(lora): "set1"}

        plan = compile_plan(merge, set_id_map, arch=None)

        op_filter = plan.ops[1]
        assert op_filter.t_factor == 0.75


class TestComposeWithTwoBranches:
    """RecipeCompose with 2 branches -> [OpApplyLoRA, OpApplyLoRA, OpMergeWeights].

    # AC: @exit-batched-eval ac-1
    Compose targets call merge_weights_batched with all branch results and backbone.
    """

    def test_op_sequence(self):
        """# AC: @exit-batched-eval ac-1"""
        base = _base()
        lora1 = _lora("a.safetensors")
        lora2 = _lora("b.safetensors")
        compose = RecipeCompose(branches=(lora1, lora2))
        merge = RecipeMerge(base=base, target=compose, backbone=None, t_factor=1.0)
        set_id_map = {id(lora1): "set1", id(lora2): "set2"}

        plan = compile_plan(merge, set_id_map, arch=None)

        assert len(plan.ops) == 3
        assert type(plan.ops[0]) is OpApplyLoRA
        assert type(plan.ops[1]) is OpApplyLoRA
        assert type(plan.ops[2]) is OpMergeWeights

    def test_branch_regs(self):
        """# AC: @exit-batched-eval ac-1"""
        base = _base()
        lora1 = _lora("a.safetensors")
        lora2 = _lora("b.safetensors")
        compose = RecipeCompose(branches=(lora1, lora2))
        merge = RecipeMerge(base=base, target=compose, backbone=None, t_factor=1.0)
        set_id_map = {id(lora1): "set1", id(lora2): "set2"}

        plan = compile_plan(merge, set_id_map, arch=None)

        op_merge = plan.ops[2]
        # Branch regs should be the out_regs of the two OpApplyLoRA ops
        assert op_merge.branch_regs == (plan.ops[0].out_reg, plan.ops[1].out_reg)
        assert op_merge.backbone_reg == 0
        assert plan.result_reg == op_merge.out_reg


class TestChainedMerge:
    """Chained RecipeMerge -> inner merge ops before outer.

    # AC: @exit-batched-eval ac-3
    Chained RecipeMerge nodes evaluate inner merges first.
    """

    def test_inner_before_outer(self):
        """# AC: @exit-batched-eval ac-3"""
        base = _base()
        inner_lora = _lora("inner.safetensors")
        inner_merge = RecipeMerge(base=base, target=inner_lora, backbone=None, t_factor=1.0)

        outer_lora = _lora("outer.safetensors")
        outer_merge = RecipeMerge(
            base=inner_merge, target=outer_lora, backbone=None, t_factor=1.0
        )
        set_id_map = {id(inner_lora): "inner", id(outer_lora): "outer"}

        plan = compile_plan(outer_merge, set_id_map, arch=None)

        # Should be: inner_lora_apply, inner_filter, outer_lora_apply, outer_filter
        assert len(plan.ops) == 4
        assert type(plan.ops[0]) is OpApplyLoRA  # inner lora
        assert type(plan.ops[1]) is OpFilterDelta  # inner filter
        assert type(plan.ops[2]) is OpApplyLoRA  # outer lora
        assert type(plan.ops[3]) is OpFilterDelta  # outer filter

        # Inner filter's result feeds into outer LoRA as base
        inner_filter = plan.ops[1]
        outer_lora_op = plan.ops[2]
        assert outer_lora_op.input_reg == inner_filter.out_reg

        # Outer filter's backbone should be the inner merge result
        outer_filter = plan.ops[3]
        assert outer_filter.backbone_reg == inner_filter.out_reg

    def test_triple_chain(self):
        """# AC: @exit-batched-eval ac-3"""
        base = _base()
        l1 = _lora("l1.safetensors")
        m1 = RecipeMerge(base=base, target=l1, backbone=None, t_factor=1.0)
        l2 = _lora("l2.safetensors")
        m2 = RecipeMerge(base=m1, target=l2, backbone=None, t_factor=1.0)
        l3 = _lora("l3.safetensors")
        m3 = RecipeMerge(base=m2, target=l3, backbone=None, t_factor=1.0)
        set_id_map = {id(l1): "s1", id(l2): "s2", id(l3): "s3"}

        plan = compile_plan(m3, set_id_map, arch=None)

        # 3 merges × 2 ops each = 6 ops
        assert len(plan.ops) == 6
        # All apply ops come before their filters in pairs
        for i in range(0, 6, 2):
            assert type(plan.ops[i]) is OpApplyLoRA
            assert type(plan.ops[i + 1]) is OpFilterDelta


class TestBackboneOverride:
    """backbone override -> backbone_reg=0 (original base_batch).

    # AC: @exit-batched-eval ac-5
    RecipeMerge.backbone overrides the importance reference for WIDEN analysis.
    """

    def test_backbone_uses_reg_zero(self):
        """# AC: @exit-batched-eval ac-5"""
        base = _base()
        backbone_ref = _base()
        lora = _lora()
        merge = RecipeMerge(base=base, target=lora, backbone=backbone_ref, t_factor=1.0)
        set_id_map = {id(lora): "set1"}

        plan = compile_plan(merge, set_id_map, arch=None)

        op_filter = plan.ops[1]
        assert op_filter.backbone_reg == 0

    def test_no_backbone_uses_current_base(self):
        """# AC: @exit-batched-eval ac-5"""
        base = _base()
        lora = _lora()
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)
        set_id_map = {id(lora): "set1"}

        plan = compile_plan(merge, set_id_map, arch=None)

        # No backbone → backbone_reg = current_base_reg = 0 (since base is RecipeBase)
        op_filter = plan.ops[1]
        assert op_filter.backbone_reg == 0

    def test_chained_backbone_override_still_reg_zero(self):
        """Even in a chained merge, backbone override points to reg 0.

        # AC: @exit-batched-eval ac-5
        """
        base = _base()
        l1 = _lora("l1.safetensors")
        m1 = RecipeMerge(base=base, target=l1, backbone=None, t_factor=1.0)

        backbone_ref = _base()
        l2 = _lora("l2.safetensors")
        m2 = RecipeMerge(base=m1, target=l2, backbone=backbone_ref, t_factor=1.0)
        set_id_map = {id(l1): "s1", id(l2): "s2"}

        plan = compile_plan(m2, set_id_map, arch=None)

        # Outer filter should use reg 0 as backbone (backbone override)
        outer_filter = plan.ops[3]
        assert outer_filter.backbone_reg == 0

        # Inner filter has no override → uses current_base_reg (= reg 0, since base is RecipeBase)
        inner_filter = plan.ops[1]
        assert inner_filter.backbone_reg == 0


class TestSingleBranchCompose:
    """Single-branch compose -> OpFilterDelta, not OpMergeWeights.

    # AC: @exit-node ac-6
    Single-branch compose uses filter_delta (passthrough), not merge_weights.
    """

    def test_single_branch_uses_filter(self):
        """# AC: @exit-node ac-6"""
        base = _base()
        lora = _lora()
        compose = RecipeCompose(branches=(lora,))
        merge = RecipeMerge(base=base, target=compose, backbone=None, t_factor=1.0)
        set_id_map = {id(lora): "set1"}

        plan = compile_plan(merge, set_id_map, arch=None)

        assert len(plan.ops) == 2
        assert type(plan.ops[0]) is OpApplyLoRA
        assert type(plan.ops[1]) is OpFilterDelta  # not OpMergeWeights


class TestPerBlockFlags:
    """use_per_block pre-computation based on block_config and arch.

    # AC: @merge-block-config ac-1
    When RecipeMerge has block_config, per-block t_factor overrides are applied.

    # AC: @merge-block-config ac-2
    When no block_config is connected, global t_factor applies to all blocks.
    """

    def test_no_block_config_no_per_block(self):
        """# AC: @merge-block-config ac-2"""
        base = _base()
        lora = _lora()
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)
        set_id_map = {id(lora): "set1"}

        plan = compile_plan(merge, set_id_map, arch="sdxl")

        op_filter = plan.ops[1]
        assert op_filter.use_per_block is False

    def test_block_config_with_arch_enables_per_block(self):
        """# AC: @merge-block-config ac-1"""
        from lib.recipe import BlockConfig

        base = _base()
        lora = _lora()
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        merge = RecipeMerge(
            base=base, target=lora, backbone=None, t_factor=1.0, block_config=bc
        )
        set_id_map = {id(lora): "set1"}

        plan = compile_plan(merge, set_id_map, arch="sdxl")

        op_filter = plan.ops[1]
        assert op_filter.use_per_block is True
        assert op_filter.block_config is bc

    def test_block_config_without_arch_no_per_block(self):
        """# AC: @merge-block-config ac-1"""
        from lib.recipe import BlockConfig

        base = _base()
        lora = _lora()
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        merge = RecipeMerge(
            base=base, target=lora, backbone=None, t_factor=1.0, block_config=bc
        )
        set_id_map = {id(lora): "set1"}

        plan = compile_plan(merge, set_id_map, arch=None)

        op_filter = plan.ops[1]
        assert op_filter.use_per_block is False


class TestSetIdResolution:
    """set_id resolved eagerly from set_id_map at compile time."""

    def test_set_id_stored_in_op(self):
        base = _base()
        lora = _lora()
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)
        set_id_map = {id(lora): "my_set_id"}

        plan = compile_plan(merge, set_id_map, arch=None)

        op_lora = plan.ops[0]
        assert op_lora.set_id == "my_set_id"

    def test_missing_set_id_raises(self):
        base = _base()
        lora = _lora()
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        with pytest.raises(RuntimeError, match="no set_id mapping"):
            compile_plan(merge, {}, arch=None)


class TestRecipeBaseAtRoot:
    """RecipeBase at root produces no ops, result_reg=0."""

    def test_base_at_root(self):
        base = _base()
        plan = compile_plan(base, {}, arch=None)

        assert len(plan.ops) == 0
        assert plan.result_reg == 0
        assert plan.dead_after == ()


class TestEvalPlanIsFrozen:
    """EvalPlan is a frozen dataclass."""

    def test_ops_is_tuple(self):
        base = _base()
        lora = _lora()
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)
        set_id_map = {id(lora): "set1"}

        plan = compile_plan(merge, set_id_map, arch=None)

        assert isinstance(plan.ops, tuple)
        assert isinstance(plan.dead_after, tuple)

    def test_frozen(self):
        base = _base()
        plan = compile_plan(base, {}, arch=None)

        with pytest.raises(AttributeError):
            plan.result_reg = 99  # type: ignore[misc]


class TestMergeAsTarget:
    """RecipeMerge as target of outer merge produces OpFilterDelta.

    # AC: @exit-batched-eval ac-3
    """

    def test_merge_target_uses_filter(self):
        """# AC: @exit-batched-eval ac-3"""
        base = _base()
        inner_lora = _lora("inner.safetensors")
        inner_merge = RecipeMerge(
            base=base, target=inner_lora, backbone=None, t_factor=0.8
        )
        outer_merge = RecipeMerge(
            base=base, target=inner_merge, backbone=None, t_factor=0.9
        )
        set_id_map = {id(inner_lora): "inner_set"}

        plan = compile_plan(outer_merge, set_id_map, arch=None)

        # inner: OpApplyLoRA + OpFilterDelta, outer: OpFilterDelta
        assert len(plan.ops) == 3
        assert type(plan.ops[0]) is OpApplyLoRA
        assert type(plan.ops[1]) is OpFilterDelta  # inner filter (t=0.8)
        assert type(plan.ops[2]) is OpFilterDelta  # outer filter (t=0.9)
        assert plan.ops[1].t_factor == 0.8
        assert plan.ops[2].t_factor == 0.9


# ---------------------------------------------------------------------------
# Register liveness / dead_after tests
# ---------------------------------------------------------------------------


class TestRegisterLiveness:
    """Verify dead_after correctly identifies registers to free."""

    def test_simple_merge_frees_lora_reg(self):
        """After OpFilterDelta reads the LoRA reg, it should be freed."""
        base = _base()
        lora = _lora()
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)
        set_id_map = {id(lora): "set1"}

        plan = compile_plan(merge, set_id_map, arch=None)

        # ops: [OpApplyLoRA(in=0, out=1), OpFilterDelta(in=1, bb=0, out=2)]
        # reg 1 is last read at op[1], so dead_after[1] should contain reg 1
        # reg 0 is never freed (caller-owned), reg 2 is result_reg (never freed)
        assert plan.dead_after[0] == ()  # reg 1 still needed by op[1]
        assert 1 in plan.dead_after[1]

    def test_chained_merge_frees_inner_lora_reg(self):
        """Inner LoRA reg is freed after inner filter reads it."""
        base = _base()
        l1 = _lora("l1.safetensors")
        m1 = RecipeMerge(base=base, target=l1, backbone=None, t_factor=1.0)
        l2 = _lora("l2.safetensors")
        m2 = RecipeMerge(base=m1, target=l2, backbone=None, t_factor=1.0)
        set_id_map = {id(l1): "s1", id(l2): "s2"}

        plan = compile_plan(m2, set_id_map, arch=None)

        # ops: [ApplyLoRA(in=0,out=1), Filter(in=1,bb=0,out=2),
        #        ApplyLoRA(in=2,out=3), Filter(in=3,bb=2,out=4)]
        # reg 1: last read at op[1] -> dead_after[1]
        # reg 2: last read at op[3] -> dead_after[3]
        # reg 3: last read at op[3] -> dead_after[3]
        # reg 0: never freed, reg 4: result_reg never freed
        assert 1 in plan.dead_after[1]
        assert 2 in plan.dead_after[3] or 3 in plan.dead_after[3]

    def test_compose_frees_branch_regs(self):
        """Branch regs are freed after OpMergeWeights reads them."""
        base = _base()
        lora1 = _lora("a.safetensors")
        lora2 = _lora("b.safetensors")
        compose = RecipeCompose(branches=(lora1, lora2))
        merge = RecipeMerge(base=base, target=compose, backbone=None, t_factor=1.0)
        set_id_map = {id(lora1): "set1", id(lora2): "set2"}

        plan = compile_plan(merge, set_id_map, arch=None)

        # ops: [ApplyLoRA(in=0,out=1), ApplyLoRA(in=0,out=2),
        #        MergeWeights(branches=(1,2),bb=0,out=3)]
        # reg 1: last read at op[2] -> dead_after[2]
        # reg 2: last read at op[2] -> dead_after[2]
        dead_at_merge = set(plan.dead_after[2])
        assert 1 in dead_at_merge
        assert 2 in dead_at_merge

    def test_result_reg_never_freed(self):
        """The result register is never included in dead_after."""
        base = _base()
        lora = _lora()
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)
        set_id_map = {id(lora): "set1"}

        plan = compile_plan(merge, set_id_map, arch=None)

        all_dead = set()
        for d in plan.dead_after:
            all_dead.update(d)
        assert plan.result_reg not in all_dead

    def test_reg_zero_never_freed(self):
        """Register 0 (base_batch) is never included in dead_after."""
        base = _base()
        lora = _lora()
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)
        set_id_map = {id(lora): "set1"}

        plan = compile_plan(merge, set_id_map, arch=None)

        all_dead = set()
        for d in plan.dead_after:
            all_dead.update(d)
        assert 0 not in all_dead

    def test_dead_after_length_matches_ops(self):
        """dead_after should have one entry per op."""
        base = _base()
        l1 = _lora("l1.safetensors")
        m1 = RecipeMerge(base=base, target=l1, backbone=None, t_factor=1.0)
        l2 = _lora("l2.safetensors")
        m2 = RecipeMerge(base=m1, target=l2, backbone=None, t_factor=1.0)
        set_id_map = {id(l1): "s1", id(l2): "s2"}

        plan = compile_plan(m2, set_id_map, arch=None)

        assert len(plan.dead_after) == len(plan.ops)
