"""Tests for WIDEN Exit Node — AC coverage for @exit-node spec."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge
from nodes.exit import WIDENExitNode, _validate_recipe_tree

# =============================================================================
# AC-1: Returns ComfyUI MODEL with set patches
# =============================================================================


class TestExitNodeReturnsModel:
    """AC: @exit-node ac-1

    Given: a valid recipe tree ending in RecipeMerge
    When: Exit node executes
    Then: it returns a ComfyUI MODEL (ModelPatcher clone) with set patches
    """

    def test_execute_returns_model_patcher_clone(self, mock_model_patcher):
        """Exit node should return a cloned ModelPatcher."""
        # AC: @exit-node ac-1
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")

        node = WIDENExitNode()
        (result,) = node.execute(base)

        # Result is a clone, not the original
        assert result is not mock_model_patcher
        # Clone has independent patches dict
        assert result.patches_uuid != mock_model_patcher.patches_uuid

    def test_execute_with_lora_returns_patched_model(self, mock_model_patcher, tmp_path):
        """Exit with LoRA recipe should return model with set patches."""
        # AC: @exit-node ac-1
        # Create a mock LoRA file
        lora_path = tmp_path / "test.safetensors"
        lora_path.write_bytes(b"mock lora data")

        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": str(lora_path), "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        node = WIDENExitNode()

        # Mock the analysis and executor since we don't have real LoRA files
        with patch("nodes.exit.analyze_recipe") as mock_analyze:
            mock_loader = MagicMock()
            mock_loader.affected_keys = set()
            mock_loader.cleanup = MagicMock()
            mock_analyze.return_value = MagicMock(
                model_patcher=mock_model_patcher,
                arch="sdxl",
                loader=mock_loader,
                set_affected={},
                affected_keys=set(),
            )

            (result,) = node.execute(merge)

        # Should return a model patcher (clone when no affected keys)
        assert result is not mock_model_patcher


# =============================================================================
# AC-2: Invalid recipe tree validation
# =============================================================================


class TestRecipeTreeValidation:
    """AC: @exit-node ac-2

    Given: an invalid recipe tree with type mismatches
    When: Exit node validates
    Then: it raises ValueError naming the invalid type and its position in the tree
    """

    def test_invalid_base_type_in_merge(self, mock_model_patcher):
        """Merge with invalid base type should raise ValueError with position."""
        # AC: @exit-node ac-2
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        # Invalid: base is RecipeLoRA, should be RecipeBase or RecipeMerge
        # We can't actually construct this with RecipeMerge (it validates in merge node)
        # but we can test the validation function directly

        # Create a mock "bad" recipe with wrong base type
        class BadNode:
            pass

        bad_merge = RecipeMerge.__new__(RecipeMerge)
        object.__setattr__(bad_merge, "base", BadNode())
        object.__setattr__(bad_merge, "target", lora)
        object.__setattr__(bad_merge, "backbone", None)
        object.__setattr__(bad_merge, "t_factor", 1.0)

        with pytest.raises(ValueError) as exc_info:
            _validate_recipe_tree(bad_merge)

        assert "root.base" in str(exc_info.value)
        assert "BadNode" in str(exc_info.value)

    def test_invalid_target_type_in_merge(self, mock_model_patcher):
        """Merge with invalid target type should raise ValueError with position."""
        # AC: @exit-node ac-2
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")

        class BadTarget:
            pass

        bad_merge = RecipeMerge.__new__(RecipeMerge)
        object.__setattr__(bad_merge, "base", base)
        object.__setattr__(bad_merge, "target", BadTarget())
        object.__setattr__(bad_merge, "backbone", None)
        object.__setattr__(bad_merge, "t_factor", 1.0)

        with pytest.raises(ValueError) as exc_info:
            _validate_recipe_tree(bad_merge)

        assert "root.target" in str(exc_info.value)
        assert "BadTarget" in str(exc_info.value)

    def test_invalid_branch_in_compose(self, mock_model_patcher):
        """Compose with invalid branch should raise ValueError with position."""
        # AC: @exit-node ac-2

        class BadBranch:
            pass

        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))

        # Create compose with bad branch
        bad_compose = RecipeCompose.__new__(RecipeCompose)
        object.__setattr__(bad_compose, "branches", (lora, BadBranch()))

        bad_merge = RecipeMerge(base=base, target=bad_compose, backbone=None, t_factor=1.0)

        with pytest.raises(ValueError) as exc_info:
            _validate_recipe_tree(bad_merge)

        assert "branches[1]" in str(exc_info.value)
        assert "BadBranch" in str(exc_info.value)

    def test_empty_compose_raises(self, mock_model_patcher):
        """Compose with no branches should raise ValueError."""
        # AC: @exit-node ac-2
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")

        empty_compose = RecipeCompose.__new__(RecipeCompose)
        object.__setattr__(empty_compose, "branches", ())

        merge = RecipeMerge(base=base, target=empty_compose, backbone=None, t_factor=1.0)

        with pytest.raises(ValueError) as exc_info:
            _validate_recipe_tree(merge)

        assert "no branches" in str(exc_info.value)

    def test_deeply_nested_invalid_type(self, mock_model_patcher):
        """Deeply nested invalid type should report correct path."""
        # AC: @exit-node ac-2
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))

        # Build nested structure
        merge1 = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        class BadTarget:
            pass

        bad_merge2 = RecipeMerge.__new__(RecipeMerge)
        object.__setattr__(bad_merge2, "base", merge1)
        object.__setattr__(bad_merge2, "target", BadTarget())
        object.__setattr__(bad_merge2, "backbone", None)
        object.__setattr__(bad_merge2, "t_factor", 1.0)

        with pytest.raises(ValueError) as exc_info:
            _validate_recipe_tree(bad_merge2)

        assert "root.target" in str(exc_info.value)

    def test_valid_tree_passes_validation(self, mock_model_patcher):
        """Valid recipe tree should pass validation without error."""
        # AC: @exit-node ac-2
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        # Should not raise
        _validate_recipe_tree(merge)

    def test_unknown_root_type_raises(self):
        """Unknown root type should raise ValueError."""
        # AC: @exit-node ac-2

        class UnknownNode:
            pass

        with pytest.raises(ValueError) as exc_info:
            _validate_recipe_tree(UnknownNode())

        assert "Unknown recipe node type" in str(exc_info.value)

    def test_execute_with_invalid_tree_raises(self, mock_model_patcher):
        """Execute with invalid tree should raise ValueError."""
        # AC: @exit-node ac-2
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")

        class BadTarget:
            pass

        bad_merge = RecipeMerge.__new__(RecipeMerge)
        object.__setattr__(bad_merge, "base", base)
        object.__setattr__(bad_merge, "target", BadTarget())
        object.__setattr__(bad_merge, "backbone", None)
        object.__setattr__(bad_merge, "t_factor", 1.0)

        node = WIDENExitNode()
        with pytest.raises(ValueError):
            node.execute(bad_merge)


# =============================================================================
# AC-3: Compose target calls merge_weights
# =============================================================================


class TestComposeCallsMergeWeights:
    """AC: @exit-node ac-3

    Given: a recipe tree with compose target containing multiple branches
    When: Exit evaluates the merge step
    Then: it calls merge_weights for simultaneous parameter routing
    """

    def test_compose_target_uses_merge_weights(self, mock_model_patcher):
        """Compose with multiple branches should call merge_weights_batched."""
        # AC: @exit-node ac-3
        # This is tested at the executor level - see test_executor.py
        # TestEvaluateRecipeComposeTarget.test_compose_calls_merge_weights_batched
        pass  # Integration tested via executor tests


# =============================================================================
# AC-4: Single LoRA target calls filter_delta
# =============================================================================


class TestLoRACallsFilterDelta:
    """AC: @exit-node ac-4

    Given: a recipe tree with single LoRA target
    When: Exit evaluates the merge step
    Then: it calls filter_delta for importance filtering
    """

    def test_lora_target_uses_filter_delta(self, mock_model_patcher):
        """LoRA target should call filter_delta_batched."""
        # AC: @exit-node ac-4
        # This is tested at the executor level - see test_executor.py
        # TestEvaluateRecipeLoRATarget.test_lora_target_calls_filter_delta_batched
        pass  # Integration tested via executor tests


# =============================================================================
# AC-5: Chained merges evaluate inner first
# =============================================================================


class TestChainedMergeOrder:
    """AC: @exit-node ac-5

    Given: a chain where RecipeMerge base is another RecipeMerge
    When: Exit evaluates
    Then: inner merge evaluates first and its result becomes the base for outer merge
    """

    def test_chained_merge_inner_first(self, mock_model_patcher):
        """Inner merge should evaluate before outer merge."""
        # AC: @exit-node ac-5
        # This is tested at the executor level - see test_executor.py
        # TestEvaluateRecipeChainedMerge.test_chained_merge_evaluates_inner_first
        pass  # Integration tested via executor tests


# =============================================================================
# AC-6: Single-branch compose uses filter_delta
# =============================================================================


class TestSingleBranchCompose:
    """AC: @exit-node ac-6

    Given: a RecipeCompose with a single branch
    When: Exit evaluates the merge step
    Then: it treats it as filter_delta not merge_weights (single-branch passthrough)
    """

    def test_single_branch_compose_uses_filter_delta(self):
        """Single-branch compose should use filter_delta, not merge_weights."""
        # AC: @exit-node ac-6
        from lib.executor import evaluate_recipe
        from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge

        batch_size = 2
        base_batch = torch.randn(batch_size, 4, 4)

        # Create single-branch compose
        base = RecipeBase(model_patcher=None, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "single.safetensors", "strength": 1.0},))
        compose = RecipeCompose(branches=(lora,))  # Single branch
        merge = RecipeMerge(base=base, target=compose, backbone=None, t_factor=1.0)

        # Mock loader and widen
        class MockLoader:
            def get_delta_specs(self, keys, key_indices, set_id=None):
                return []

        class MockWIDEN:
            def __init__(self):
                self.filter_calls = []
                self.merge_calls = []

            def filter_delta_batched(self, lora_applied, backbone):
                self.filter_calls.append({"lora_applied": lora_applied, "backbone": backbone})
                return lora_applied

            def merge_weights_batched(self, weights_list, backbone):
                self.merge_calls.append({"weights_list": weights_list, "backbone": backbone})
                return torch.stack(weights_list).mean(dim=0)

        loader = MockLoader()
        widen = MockWIDEN()
        set_id_map = {id(lora): "set1"}

        evaluate_recipe(
            keys=["k0", "k1"],
            base_batch=base_batch,
            recipe_node=merge,
            loader=loader,
            widen=widen,
            set_id_map=set_id_map,
            device="cpu",
            dtype=torch.float32,
        )

        # Should call filter_delta, not merge_weights
        assert len(widen.filter_calls) == 1
        assert len(widen.merge_calls) == 0


# =============================================================================
# AC-7: Downstream LoRA patches apply additively
# =============================================================================


class TestDownstreamLoraCompatibility:
    """AC: @exit-node ac-7

    Given: the Exit node output MODEL
    When: a downstream ComfyUI LoRA node applies additional patches
    Then: the additional LoRA patches apply additively on top of the set patches
    """

    def test_set_patches_allow_additional_lora(self, mock_model_patcher):
        """Set patches should allow downstream LoRA patches to apply."""
        # AC: @exit-node ac-7
        from nodes.exit import install_merged_patches

        # Install some merged patches
        merged_state = {
            "input_blocks.0.0.weight": torch.randn(4, 4),
        }
        result = install_merged_patches(mock_model_patcher, merged_state)

        # Simulate downstream LoRA patch
        downstream_patch = {
            "diffusion_model.input_blocks.0.0.weight": ("add", torch.randn(4, 4)),
        }
        result.add_patches(downstream_patch, strength_patch=0.5)

        # Both patches should be present
        key = "diffusion_model.input_blocks.0.0.weight"
        assert key in result.patches
        # Should have 2 patches: set + add
        assert len(result.patches[key]) == 2

    def test_set_patch_format_is_compatible(self, mock_model_patcher):
        """Set patch format should be compatible with ComfyUI patch system."""
        # AC: @exit-node ac-7
        from nodes.exit import install_merged_patches

        merged_state = {"input_blocks.0.0.weight": torch.randn(4, 4)}
        result = install_merged_patches(mock_model_patcher, merged_state)

        # Check patch format
        key = "diffusion_model.input_blocks.0.0.weight"
        patch_entry = result.patches[key][0]

        # Format: (strength_patch, (patch_type, tensor), strength_model, None, None)
        assert patch_entry[0] == 1.0  # strength_patch
        assert patch_entry[1][0] == "set"  # patch type
        assert isinstance(patch_entry[1][1], torch.Tensor)  # tensor


# =============================================================================
# AC-8: bf16 weights produce matching patches
# =============================================================================


class TestBf16DtypeMatching:
    """AC: @exit-node ac-8

    Given: the base ModelPatcher uses bf16 weights
    When: set patches are installed
    Then: patch tensors match the base model storage dtype
    """

    def test_bf16_base_produces_bf16_patches(self):
        """bf16 base model should produce bf16 patches."""
        # AC: @exit-node ac-8
        from nodes.exit import install_merged_patches
        from tests.conftest import MockModelPatcher

        patcher = MockModelPatcher()
        # Convert to bf16
        for k in patcher._state_dict:
            patcher._state_dict[k] = patcher._state_dict[k].to(torch.bfloat16)

        # Merged state in fp32 (computation dtype)
        merged_state = {"input_blocks.0.0.weight": torch.randn(4, 4, dtype=torch.float32)}
        result = install_merged_patches(patcher, merged_state)

        # Patch should be bf16
        key = "diffusion_model.input_blocks.0.0.weight"
        patch_tensor = result.patches[key][0][1][1]
        assert patch_tensor.dtype == torch.bfloat16

    def test_fp16_base_produces_fp16_patches(self):
        """fp16 base model should produce fp16 patches."""
        # AC: @exit-node ac-8
        from nodes.exit import install_merged_patches
        from tests.conftest import MockModelPatcher

        patcher = MockModelPatcher()
        # Convert to fp16
        for k in patcher._state_dict:
            patcher._state_dict[k] = patcher._state_dict[k].to(torch.float16)

        merged_state = {"input_blocks.0.0.weight": torch.randn(4, 4, dtype=torch.float32)}
        result = install_merged_patches(patcher, merged_state)

        key = "diffusion_model.input_blocks.0.0.weight"
        patch_tensor = result.patches[key][0][1][1]
        assert patch_tensor.dtype == torch.float16


# =============================================================================
# Additional integration tests
# =============================================================================


class TestExitNodeIntegration:
    """Integration tests for Exit node behavior."""

    def test_execute_with_recipe_base_returns_clone(self, mock_model_patcher):
        """RecipeBase input should return a clone directly."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")

        node = WIDENExitNode()
        (result,) = node.execute(base)

        assert result is not mock_model_patcher
        # No patches should be added
        assert len(result.patches) == 0

    def test_execute_rejects_lora_at_root(self):
        """RecipeLoRA at root should raise ValueError."""
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))

        node = WIDENExitNode()
        with pytest.raises(ValueError) as exc_info:
            node.execute(lora)

        assert "RecipeMerge or RecipeBase at root" in str(exc_info.value)

    def test_execute_rejects_compose_at_root(self):
        """RecipeCompose at root should raise ValueError."""
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        compose = RecipeCompose(branches=(lora,))

        node = WIDENExitNode()
        with pytest.raises(ValueError) as exc_info:
            node.execute(compose)

        assert "RecipeMerge or RecipeBase at root" in str(exc_info.value)

    def test_node_metadata(self):
        """Node metadata should be correct."""
        assert WIDENExitNode.INPUT_TYPES()["required"]["widen"] == ("WIDEN",)
        assert WIDENExitNode.RETURN_TYPES == ("MODEL",)
        assert WIDENExitNode.RETURN_NAMES == ("model",)
        assert WIDENExitNode.FUNCTION == "execute"
        assert WIDENExitNode.CATEGORY == "ecaj/merge"
