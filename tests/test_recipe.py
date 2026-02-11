"""Recipe dataclass tests — frozen immutability, tuple types, structure."""

import pytest

from lib.recipe import RecipeLoRA, RecipeMerge


class TestRecipeFrozen:
    """All recipe dataclasses must be frozen (immutable).
    # AC: @testing-infrastructure ac-3
    """

    def test_recipe_base_frozen(self, recipe_base):
        with pytest.raises((AttributeError, TypeError)):
            recipe_base.arch = "flux"

    def test_recipe_lora_frozen(self, recipe_single_lora):
        with pytest.raises((AttributeError, TypeError)):
            recipe_single_lora.loras = ()

    def test_recipe_compose_frozen(self, recipe_compose):
        with pytest.raises((AttributeError, TypeError)):
            recipe_compose.branches = ()

    def test_recipe_merge_frozen(self, recipe_chain):
        with pytest.raises((AttributeError, TypeError)):
            recipe_chain.t_factor = 0.0


class TestRecipeTupleTypes:
    """Collection fields must use tuples, not lists.
    # AC: @testing-infrastructure ac-3
    """

    def test_lora_loras_is_tuple(self, recipe_single_lora):
        assert isinstance(recipe_single_lora.loras, tuple)

    def test_multi_lora_is_tuple(self, recipe_multi_lora):
        assert isinstance(recipe_multi_lora.loras, tuple)
        assert len(recipe_multi_lora.loras) == 2

    def test_compose_branches_is_tuple(self, recipe_compose):
        assert isinstance(recipe_compose.branches, tuple)


class TestRecipeStructure:
    """Verify recipe tree composition and field values.
    # AC: @testing-infrastructure ac-3
    """

    def test_recipe_base_arch(self, recipe_base):
        assert recipe_base.arch == "sdxl"

    def test_recipe_base_has_patcher(self, recipe_base):
        assert recipe_base.model_patcher is not None

    def test_single_lora_content(self, recipe_single_lora):
        assert len(recipe_single_lora.loras) == 1
        assert recipe_single_lora.loras[0]["path"] == "lora_a.safetensors"
        assert recipe_single_lora.loras[0]["strength"] == 1.0

    def test_compose_has_two_branches(self, recipe_compose):
        assert len(recipe_compose.branches) == 2
        assert all(isinstance(b, RecipeLoRA) for b in recipe_compose.branches)

    def test_chain_is_nested_merge(self, recipe_chain):
        assert isinstance(recipe_chain, RecipeMerge)
        assert isinstance(recipe_chain.base, RecipeMerge)
        assert isinstance(recipe_chain.target, RecipeLoRA)
        assert recipe_chain.t_factor == 0.7
        assert recipe_chain.base.t_factor == 1.0

    def test_merge_backbone_default_none(self, recipe_chain):
        assert recipe_chain.backbone is None
        assert recipe_chain.base.backbone is None
