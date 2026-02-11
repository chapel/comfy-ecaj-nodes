"""Tests for WIDEN Merge Node — AC coverage for @merge-node spec."""

import pytest

from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge
from nodes.merge import WIDENMergeNode
from tests.conftest import MockModelPatcher

# ---------------------------------------------------------------------------
# AC-1: Base and target with t_factor → RecipeMerge with all fields stored
# ---------------------------------------------------------------------------


class TestMergeBasicExecution:
    """AC: @merge-node ac-1 — basic merge returns RecipeMerge with fields."""

    def test_returns_tuple_with_recipe_merge(self):
        """Merge node returns tuple containing RecipeMerge."""
        node = WIDENMergeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        target = RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},))

        result = node.merge(base=base, target=target, t_factor=1.0)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], RecipeMerge)

    def test_base_stored_in_recipe_merge(self):
        """Base reference is stored in RecipeMerge."""
        node = WIDENMergeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        target = RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},))

        (merge_result,) = node.merge(base=base, target=target, t_factor=0.8)

        assert merge_result.base is base

    def test_target_stored_in_recipe_merge(self):
        """Target reference is stored in RecipeMerge."""
        node = WIDENMergeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        target = RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},))

        (merge_result,) = node.merge(base=base, target=target, t_factor=1.5)

        assert merge_result.target is target

    def test_t_factor_stored_in_recipe_merge(self):
        """t_factor value is stored in RecipeMerge."""
        node = WIDENMergeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        target = RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},))

        (merge_result,) = node.merge(base=base, target=target, t_factor=2.5)

        assert merge_result.t_factor == 2.5


# ---------------------------------------------------------------------------
# AC-2: No backbone input → backbone is None
# ---------------------------------------------------------------------------


class TestMergeNoBackbone:
    """AC: @merge-node ac-2 — no backbone input means backbone is None."""

    def test_backbone_none_by_default(self):
        """When backbone is not provided, it defaults to None."""
        node = WIDENMergeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        target = RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},))

        (merge_result,) = node.merge(base=base, target=target, t_factor=1.0)

        assert merge_result.backbone is None

    def test_explicit_none_backbone(self):
        """Explicitly passing backbone=None results in None."""
        node = WIDENMergeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        target = RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},))

        (merge_result,) = node.merge(base=base, target=target, t_factor=1.0, backbone=None)

        assert merge_result.backbone is None


# ---------------------------------------------------------------------------
# AC-3: Explicit backbone input → backbone is stored
# ---------------------------------------------------------------------------


class TestMergeWithBackbone:
    """AC: @merge-node ac-3 — explicit backbone is stored in RecipeMerge."""

    def test_backbone_stored_when_provided(self):
        """Backbone reference is stored when provided."""
        node = WIDENMergeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        target = RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},))
        backbone = RecipeBase(model_patcher=MockModelPatcher(), arch="sdxl")

        (merge_result,) = node.merge(base=base, target=target, t_factor=1.0, backbone=backbone)

        assert merge_result.backbone is backbone

    def test_backbone_can_be_recipe_merge(self):
        """Backbone can be a RecipeMerge (chained merge as importance ref)."""
        node = WIDENMergeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        target = RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},))
        # Create a merge chain for backbone
        backbone_base = RecipeBase(model_patcher=MockModelPatcher(), arch="sdxl")
        backbone_lora = RecipeLoRA(loras=({"path": "bb.safetensors", "strength": 0.5},))
        backbone = RecipeMerge(
            base=backbone_base, target=backbone_lora, backbone=None, t_factor=1.0
        )

        (merge_result,) = node.merge(base=base, target=target, t_factor=1.0, backbone=backbone)

        assert merge_result.backbone is backbone
        assert isinstance(merge_result.backbone, RecipeMerge)


# ---------------------------------------------------------------------------
# AC-4: Merge output wired to another Merge base → valid chain
# ---------------------------------------------------------------------------


class TestMergeChaining:
    """AC: @merge-node ac-4 — Merge outputs can chain as base input."""

    def test_merge_can_be_base_of_another_merge(self):
        """RecipeMerge can be used as base input to create chain."""
        node = WIDENMergeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        lora_a = RecipeLoRA(loras=({"path": "lora_a.safetensors", "strength": 1.0},))
        lora_b = RecipeLoRA(loras=({"path": "lora_b.safetensors", "strength": 0.8},))

        # First merge
        (merge_a,) = node.merge(base=base, target=lora_a, t_factor=1.0)
        # Chain second merge using first as base
        (merge_b,) = node.merge(base=merge_a, target=lora_b, t_factor=0.5)

        assert isinstance(merge_b, RecipeMerge)
        assert merge_b.base is merge_a
        assert merge_b.target is lora_b

    def test_three_way_merge_chain(self):
        """Three merges can chain sequentially."""
        node = WIDENMergeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        loras = [
            RecipeLoRA(loras=({"path": f"lora_{i}.safetensors", "strength": 1.0},))
            for i in range(3)
        ]

        (m1,) = node.merge(base=base, target=loras[0], t_factor=1.0)
        (m2,) = node.merge(base=m1, target=loras[1], t_factor=0.8)
        (m3,) = node.merge(base=m2, target=loras[2], t_factor=0.6)

        # Walk the chain
        assert m3.base is m2
        assert m2.base is m1
        assert m1.base is base

    def test_chain_preserves_t_factors(self):
        """Each merge in chain has its own t_factor."""
        node = WIDENMergeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        lora_a = RecipeLoRA(loras=({"path": "lora_a.safetensors", "strength": 1.0},))
        lora_b = RecipeLoRA(loras=({"path": "lora_b.safetensors", "strength": 0.8},))

        (m1,) = node.merge(base=base, target=lora_a, t_factor=1.5)
        (m2,) = node.merge(base=m1, target=lora_b, t_factor=0.3)

        assert m1.t_factor == 1.5
        assert m2.t_factor == 0.3


# ---------------------------------------------------------------------------
# AC-5: RecipeLoRA or RecipeCompose as base → raises error
# ---------------------------------------------------------------------------


class TestMergeRejectsInvalidBase:
    """AC: @merge-node ac-5 — base must be RecipeBase or RecipeMerge."""

    def test_recipe_lora_as_base_raises_error(self):
        """RecipeLoRA as base raises ValueError."""
        node = WIDENMergeNode()
        lora_base = RecipeLoRA(loras=({"path": "lora_base.safetensors", "strength": 1.0},))
        target = RecipeLoRA(loras=({"path": "lora_target.safetensors", "strength": 0.5},))

        with pytest.raises(ValueError) as exc_info:
            node.merge(base=lora_base, target=target, t_factor=1.0)

        error_msg = str(exc_info.value)
        assert "RecipeBase" in error_msg or "RecipeMerge" in error_msg
        assert "RecipeLoRA" in error_msg

    def test_recipe_compose_as_base_raises_error(self):
        """RecipeCompose as base raises ValueError."""
        node = WIDENMergeNode()
        lora = RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},))
        compose_base = RecipeCompose(branches=(lora,))
        target = RecipeLoRA(loras=({"path": "lora_target.safetensors", "strength": 0.5},))

        with pytest.raises(ValueError) as exc_info:
            node.merge(base=compose_base, target=target, t_factor=1.0)

        error_msg = str(exc_info.value)
        assert "RecipeCompose" in error_msg

    def test_error_message_suggests_entry_node(self):
        """Error message mentions Entry node or Merge output."""
        node = WIDENMergeNode()
        lora_base = RecipeLoRA(loras=({"path": "lora_base.safetensors", "strength": 1.0},))
        target = RecipeLoRA(loras=({"path": "lora_target.safetensors", "strength": 0.5},))

        with pytest.raises(ValueError) as exc_info:
            node.merge(base=lora_base, target=target, t_factor=1.0)

        error_msg = str(exc_info.value)
        assert "Entry" in error_msg or "Merge" in error_msg


# ---------------------------------------------------------------------------
# AC-6: t_factor of -1.0 → preserved (passthrough, no WIDEN)
# ---------------------------------------------------------------------------


class TestMergePassthroughTFactor:
    """AC: @merge-node ac-6 — t_factor of -1.0 is preserved."""

    def test_negative_one_t_factor_preserved(self):
        """t_factor of -1.0 is stored exactly."""
        node = WIDENMergeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        target = RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},))

        (merge_result,) = node.merge(base=base, target=target, t_factor=-1.0)

        assert merge_result.t_factor == -1.0

    def test_negative_t_factor_not_clamped(self):
        """Negative t_factor values close to -1.0 are preserved."""
        node = WIDENMergeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        target = RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},))

        (merge_result,) = node.merge(base=base, target=target, t_factor=-0.5)

        assert merge_result.t_factor == -0.5


# ---------------------------------------------------------------------------
# Additional validation and edge case tests
# ---------------------------------------------------------------------------


class TestMergeTargetTypes:
    """Test all valid target types are accepted."""

    def test_accepts_recipe_lora_target(self):
        """RecipeLoRA is valid target (single LoRA merge)."""
        node = WIDENMergeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        target = RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},))

        (merge_result,) = node.merge(base=base, target=target, t_factor=1.0)
        assert merge_result.target is target

    def test_accepts_recipe_compose_target(self):
        """RecipeCompose is valid target (multi-branch merge)."""
        node = WIDENMergeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},))
        target = RecipeCompose(branches=(lora,))

        (merge_result,) = node.merge(base=base, target=target, t_factor=1.0)
        assert merge_result.target is target
        assert isinstance(merge_result.target, RecipeCompose)

    def test_accepts_recipe_merge_target(self):
        """RecipeMerge is valid target (merge of merges)."""
        node = WIDENMergeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        # Create a merge to use as target
        merge_base = RecipeBase(model_patcher=MockModelPatcher(), arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},))
        target = RecipeMerge(base=merge_base, target=lora, backbone=None, t_factor=1.0)

        (merge_result,) = node.merge(base=base, target=target, t_factor=1.0)
        assert merge_result.target is target
        assert isinstance(merge_result.target, RecipeMerge)


class TestMergeNodeMetadata:
    """Test ComfyUI node metadata is correct."""

    def test_input_types(self):
        """INPUT_TYPES returns correct structure."""
        input_types = WIDENMergeNode.INPUT_TYPES()

        assert "required" in input_types
        assert "base" in input_types["required"]
        assert input_types["required"]["base"] == ("WIDEN",)
        assert "target" in input_types["required"]
        assert input_types["required"]["target"] == ("WIDEN",)
        assert "t_factor" in input_types["required"]
        t_factor_config = input_types["required"]["t_factor"]
        assert t_factor_config[0] == "FLOAT"
        assert t_factor_config[1]["min"] == -1.0
        assert t_factor_config[1]["max"] == 5.0

        assert "optional" in input_types
        assert "backbone" in input_types["optional"]
        assert input_types["optional"]["backbone"] == ("WIDEN",)

    def test_return_types(self):
        """RETURN_TYPES is WIDEN tuple."""
        assert WIDENMergeNode.RETURN_TYPES == ("WIDEN",)
        assert WIDENMergeNode.RETURN_NAMES == ("widen",)

    def test_category(self):
        """CATEGORY is ecaj/merge."""
        assert WIDENMergeNode.CATEGORY == "ecaj/merge"

    def test_function_name(self):
        """FUNCTION points to merge method."""
        assert WIDENMergeNode.FUNCTION == "merge"
