"""Recipe dataclass tests — frozen immutability, tuple types, structure."""

import pytest
import torch

from lib.recipe import (
    RecipeBase,
    RecipeCompose,
    RecipeLoRA,
    RecipeMerge,
    RecipeNode,
)


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


class TestRecipeComposePersistentSemantics:
    """RecipeCompose.with_branch returns new instance, original unchanged.
    # AC: @recipe-system ac-2
    """

    def test_with_branch_returns_new_instance(self, recipe_compose, recipe_single_lora):
        """Appending returns a new RecipeCompose, not the same one."""
        new_compose = recipe_compose.with_branch(recipe_single_lora)
        assert new_compose is not recipe_compose
        assert isinstance(new_compose, RecipeCompose)

    def test_with_branch_original_unchanged(self, recipe_compose, recipe_single_lora):
        """Original compose branches unchanged after append."""
        original_len = len(recipe_compose.branches)
        original_branches = recipe_compose.branches
        _ = recipe_compose.with_branch(recipe_single_lora)
        assert len(recipe_compose.branches) == original_len
        assert recipe_compose.branches is original_branches

    def test_with_branch_new_tuple(self, recipe_compose, recipe_single_lora):
        """New compose has a new tuple, not mutated original."""
        new_compose = recipe_compose.with_branch(recipe_single_lora)
        assert new_compose.branches is not recipe_compose.branches
        assert isinstance(new_compose.branches, tuple)

    def test_with_branch_appends_correctly(self, recipe_compose, recipe_single_lora):
        """New compose has the appended branch at the end."""
        original_len = len(recipe_compose.branches)
        new_compose = recipe_compose.with_branch(recipe_single_lora)
        assert len(new_compose.branches) == original_len + 1
        assert new_compose.branches[-1] is recipe_single_lora


class TestRecipeNoGPUTensors:
    """Recipe objects hold no GPU tensors — only references and metadata.
    # AC: @recipe-system ac-3
    """

    def _contains_tensor(self, obj, visited=None) -> bool:
        """Recursively check if obj contains any torch.Tensor."""
        if visited is None:
            visited = set()

        obj_id = id(obj)
        if obj_id in visited:
            return False
        visited.add(obj_id)

        if isinstance(obj, torch.Tensor):
            return True

        # Check dataclass fields
        if hasattr(obj, "__dataclass_fields__"):
            for field_name in obj.__dataclass_fields__:
                field_val = getattr(obj, field_name)
                if self._contains_tensor(field_val, visited):
                    return True

        # Check iterables
        if isinstance(obj, (tuple, list)):
            for item in obj:
                if self._contains_tensor(item, visited):
                    return True

        if isinstance(obj, dict):
            for val in obj.values():
                if self._contains_tensor(val, visited):
                    return True

        return False

    def test_recipe_base_no_tensors(self, recipe_base):
        """RecipeBase holds patcher reference, not tensors directly."""
        # The model_patcher may internally have tensors, but RecipeBase
        # itself only holds a reference to it, not the tensors
        assert not isinstance(recipe_base.model_patcher, torch.Tensor)
        assert not isinstance(recipe_base.arch, torch.Tensor)

    def test_recipe_lora_no_tensors(self, recipe_single_lora):
        """RecipeLoRA holds path/strength metadata, not tensor data."""
        assert not self._contains_tensor(recipe_single_lora.loras)

    def test_recipe_compose_no_tensors(self, recipe_compose):
        """RecipeCompose branches contain no tensors."""
        for branch in recipe_compose.branches:
            assert not self._contains_tensor(branch)

    def test_recipe_merge_no_tensors(self, recipe_chain):
        """RecipeMerge tree contains no direct tensors."""
        # Check immediate fields (excluding object refs which have their own tests)
        assert not isinstance(recipe_chain.t_factor, torch.Tensor)
        assert not isinstance(recipe_chain.backbone, torch.Tensor)


class TestRecipeImports:
    """All recipe classes are available and constructible.
    # AC: @recipe-system ac-4
    """

    def test_all_classes_importable(self):
        """RecipeBase, RecipeLoRA, RecipeCompose, RecipeMerge importable."""
        # Already imported at top, this verifies they exist
        assert RecipeBase is not None
        assert RecipeLoRA is not None
        assert RecipeCompose is not None
        assert RecipeMerge is not None

    def test_recipe_node_type_alias(self):
        """RecipeNode type alias exists and is a Union."""
        assert RecipeNode is not None
        # Verify it's a Union type
        assert hasattr(RecipeNode, "__origin__") or hasattr(RecipeNode, "__args__")

    def test_recipe_base_constructible(self):
        """RecipeBase constructible with documented fields."""
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        assert base.model_patcher is not None
        assert base.arch == "sdxl"

    def test_recipe_lora_constructible(self):
        """RecipeLoRA constructible with documented fields."""
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        assert len(lora.loras) == 1

    def test_recipe_compose_constructible(self):
        """RecipeCompose constructible with documented fields."""
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        compose = RecipeCompose(branches=(lora,))
        assert len(compose.branches) == 1

    def test_recipe_merge_constructible(self):
        """RecipeMerge constructible with documented fields."""
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)
        assert merge.t_factor == 1.0
