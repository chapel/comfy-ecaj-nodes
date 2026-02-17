"""Tests for WIDEN Compose Node — AC coverage for @compose-node spec."""

import pytest

from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge
from nodes.compose import WIDENComposeNode
from tests.conftest import MockModelPatcher  # noqa: I001

# ---------------------------------------------------------------------------
# AC-1: Single branch, no compose input → single-element branches tuple
# ---------------------------------------------------------------------------


class TestComposeNoCompose:
    """AC: @compose-node ac-1 — branch input with no compose returns single-element."""

    # AC: @compose-node ac-1
    def test_returns_tuple_with_recipe_compose(self):
        """Compose node returns tuple containing RecipeCompose."""
        node = WIDENComposeNode()
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))

        result = node.compose(branch=lora)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], RecipeCompose)

    # AC: @compose-node ac-1
    def test_single_branch_in_branches_tuple(self):
        """With no compose input, branches tuple has single element."""
        node = WIDENComposeNode()
        lora = RecipeLoRA(loras=({"path": "lora_a.safetensors", "strength": 0.8},))

        (compose_result,) = node.compose(branch=lora)

        assert len(compose_result.branches) == 1
        assert compose_result.branches[0] is lora

    # AC: @compose-node ac-1
    def test_explicit_none_compose(self):
        """Explicit compose=None gives single-element branches."""
        node = WIDENComposeNode()
        lora = RecipeLoRA(loras=({"path": "solo.safetensors", "strength": 1.0},))

        (compose_result,) = node.compose(branch=lora, compose=None)

        assert len(compose_result.branches) == 1


# ---------------------------------------------------------------------------
# AC-2: Branch with compose chain → appends to existing branches
# ---------------------------------------------------------------------------


class TestComposeChaining:
    """AC: @compose-node ac-2 — branch appended to existing compose branches."""

    # AC: @compose-node ac-2
    def test_appends_to_existing_compose(self):
        """New branch is appended to existing compose branches."""
        node = WIDENComposeNode()
        lora_a = RecipeLoRA(loras=({"path": "lora_a.safetensors", "strength": 1.0},))
        lora_b = RecipeLoRA(loras=({"path": "lora_b.safetensors", "strength": 0.5},))

        # First compose creates single-element
        (first_compose,) = node.compose(branch=lora_a)
        # Second compose chains via compose input
        (second_compose,) = node.compose(branch=lora_b, compose=first_compose)

        assert len(second_compose.branches) == 2
        assert second_compose.branches[0] is lora_a
        assert second_compose.branches[1] is lora_b

    # AC: @compose-node ac-2
    def test_original_compose_unchanged(self):
        """Chaining creates new RecipeCompose, original unchanged (persistent semantics)."""
        node = WIDENComposeNode()
        lora_a = RecipeLoRA(loras=({"path": "lora_a.safetensors", "strength": 1.0},))
        lora_b = RecipeLoRA(loras=({"path": "lora_b.safetensors", "strength": 0.5},))

        (first_compose,) = node.compose(branch=lora_a)
        (second_compose,) = node.compose(branch=lora_b, compose=first_compose)

        # Original is unchanged
        assert len(first_compose.branches) == 1
        # New has both
        assert len(second_compose.branches) == 2
        # They are different objects
        assert first_compose is not second_compose


# ---------------------------------------------------------------------------
# AC-3: Three chained Compose nodes → all three branches present in order
# ---------------------------------------------------------------------------


class TestTripleChain:
    """AC: @compose-node ac-3 — three chained nodes preserve order."""

    # AC: @compose-node ac-3
    def test_three_branches_in_order(self):
        """Three chained Compose nodes result in three branches in order."""
        node = WIDENComposeNode()
        lora_a = RecipeLoRA(loras=({"path": "first.safetensors", "strength": 1.0},))
        lora_b = RecipeLoRA(loras=({"path": "second.safetensors", "strength": 0.8},))
        lora_c = RecipeLoRA(loras=({"path": "third.safetensors", "strength": 0.6},))

        (c1,) = node.compose(branch=lora_a)
        (c2,) = node.compose(branch=lora_b, compose=c1)
        (c3,) = node.compose(branch=lora_c, compose=c2)

        assert len(c3.branches) == 3
        assert c3.branches[0] is lora_a
        assert c3.branches[1] is lora_b
        assert c3.branches[2] is lora_c

    # AC: @compose-node ac-3
    def test_five_branches_maintain_order(self):
        """Five chained Compose nodes maintain insertion order."""
        node = WIDENComposeNode()
        loras = [
            RecipeLoRA(loras=({"path": f"lora_{i}.safetensors", "strength": 1.0},))
            for i in range(5)
        ]

        (compose,) = node.compose(branch=loras[0])
        for lora in loras[1:]:
            (compose,) = node.compose(branch=lora, compose=compose)

        assert len(compose.branches) == 5
        for i, lora in enumerate(loras):
            assert compose.branches[i] is lora


# ---------------------------------------------------------------------------
# AC-4: RecipeBase wired to branch input → raises error
# ---------------------------------------------------------------------------


class TestRejectsRawRecipeBase:
    """AC: @compose-node ac-4 — raw RecipeBase raises error."""

    # AC: @compose-node ac-4
    def test_recipe_base_raises_value_error(self):
        """RecipeBase as branch raises ValueError."""
        node = WIDENComposeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")

        with pytest.raises(ValueError) as exc_info:
            node.compose(branch=base)

        error_msg = str(exc_info.value)
        assert "raw base model" in error_msg.lower()
        assert "LoRA" in error_msg or "lora" in error_msg.lower()

    # AC: @compose-node ac-4
    def test_error_mentions_use_as_merge_base(self):
        """Error message suggests using as Merge base input."""
        node = WIDENComposeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")

        with pytest.raises(ValueError) as exc_info:
            node.compose(branch=base)

        error_msg = str(exc_info.value)
        assert "Merge base" in error_msg


# ---------------------------------------------------------------------------
# Additional validation tests
# ---------------------------------------------------------------------------


class TestValidBranchTypes:
    """Test that all valid branch types are accepted."""

    def test_accepts_recipe_lora(self):
        """RecipeLoRA is a valid branch type."""
        node = WIDENComposeNode()
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))

        (result,) = node.compose(branch=lora)
        assert len(result.branches) == 1

    def test_accepts_recipe_compose(self):
        """RecipeCompose is a valid branch type (nested compose groups)."""
        node = WIDENComposeNode()
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        inner_compose = RecipeCompose(branches=(lora,))

        (result,) = node.compose(branch=inner_compose)
        assert len(result.branches) == 1
        assert result.branches[0] is inner_compose

    def test_accepts_recipe_merge(self):
        """RecipeMerge is a valid branch type (merge results can be composed)."""
        node = WIDENComposeNode()
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        (result,) = node.compose(branch=merge)
        assert len(result.branches) == 1
        assert result.branches[0] is merge


class TestInvalidComposeInput:
    """Test that invalid compose input raises error."""

    def test_rejects_non_compose_as_compose_input(self):
        """Compose input must be RecipeCompose, not other recipe types."""
        node = WIDENComposeNode()
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))

        with pytest.raises(TypeError) as exc_info:
            node.compose(branch=lora, compose=lora)  # lora is not RecipeCompose

        error_msg = str(exc_info.value)
        assert "RecipeCompose" in error_msg


class TestNodeMetadata:
    """Test ComfyUI node metadata is correct."""

    def test_input_types(self):
        """INPUT_TYPES returns correct structure."""
        input_types = WIDENComposeNode.INPUT_TYPES()

        assert "required" in input_types
        assert "branch" in input_types["required"]
        assert input_types["required"]["branch"] == ("WIDEN",)

        assert "optional" in input_types
        assert "compose" in input_types["optional"]
        assert input_types["optional"]["compose"] == ("WIDEN",)

    def test_return_types(self):
        """RETURN_TYPES is WIDEN tuple."""
        assert WIDENComposeNode.RETURN_TYPES == ("WIDEN",)
        assert WIDENComposeNode.RETURN_NAMES == ("widen",)

    def test_category(self):
        """CATEGORY is ecaj/merge."""
        assert WIDENComposeNode.CATEGORY == "ecaj/merge"

    def test_function_name(self):
        """FUNCTION points to compose method."""
        assert WIDENComposeNode.FUNCTION == "compose"
