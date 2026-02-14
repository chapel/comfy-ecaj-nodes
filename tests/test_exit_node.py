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
        # Clone copies patches_uuid from source (matching real ComfyUI behavior)
        assert result.patches_uuid == mock_model_patcher.patches_uuid

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
        from lib.executor import evaluate_recipe

        batch_size = 2
        base_batch = torch.randn(batch_size, 4, 4)

        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora_a = RecipeLoRA(loras=({"path": "a.safetensors", "strength": 1.0},))
        lora_b = RecipeLoRA(loras=({"path": "b.safetensors", "strength": 0.8},))
        compose = RecipeCompose(branches=(lora_a, lora_b))
        merge = RecipeMerge(base=base, target=compose, backbone=None, t_factor=1.0)

        class MockLoader:
            def get_delta_specs(self, keys, key_indices, set_id=None):
                return []

        class MockWIDEN:
            def __init__(self):
                self.merge_calls = []
                self.filter_calls = []

            def merge_weights_batched(self, weights_list, backbone):
                self.merge_calls.append(
                    {"weights_list": weights_list, "backbone": backbone}
                )
                return torch.stack(weights_list).mean(dim=0)

            def filter_delta_batched(self, lora_applied, backbone):
                self.filter_calls.append(
                    {"lora_applied": lora_applied, "backbone": backbone}
                )
                return lora_applied

        loader = MockLoader()
        widen = MockWIDEN()
        set_id_map = {id(lora_a): "set_a", id(lora_b): "set_b"}

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

        assert len(widen.merge_calls) == 1
        assert len(widen.merge_calls[0]["weights_list"]) == 2
        assert len(widen.filter_calls) == 0


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
        from lib.executor import evaluate_recipe

        batch_size = 2
        base_batch = torch.randn(batch_size, 4, 4)

        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        class MockLoader:
            def get_delta_specs(self, keys, key_indices, set_id=None):
                return []

        class MockWIDEN:
            def __init__(self):
                self.filter_calls = []
                self.merge_calls = []

            def filter_delta_batched(self, lora_applied, backbone):
                self.filter_calls.append(
                    {"lora_applied": lora_applied, "backbone": backbone}
                )
                return lora_applied

            def merge_weights_batched(self, weights_list, backbone):
                self.merge_calls.append(
                    {"weights_list": weights_list, "backbone": backbone}
                )
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

        assert len(widen.filter_calls) == 1
        assert len(widen.merge_calls) == 0


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
        from lib.executor import evaluate_recipe

        batch_size = 1
        base_batch = torch.randn(batch_size, 4, 4)

        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        inner_lora = RecipeLoRA(
            loras=({"path": "inner.safetensors", "strength": 1.0},)
        )
        inner_merge = RecipeMerge(
            base=base, target=inner_lora, backbone=None, t_factor=1.0
        )

        outer_lora = RecipeLoRA(
            loras=({"path": "outer.safetensors", "strength": 1.0},)
        )
        outer_merge = RecipeMerge(
            base=inner_merge, target=outer_lora, backbone=None, t_factor=1.0
        )

        class MockLoader:
            def get_delta_specs(self, keys, key_indices, set_id=None):
                return []

        call_order = []

        class MockWIDEN:
            def filter_delta_batched(self, lora_applied, backbone):
                call_order.append("filter")
                return lora_applied

            def merge_weights_batched(self, weights_list, backbone):
                call_order.append("merge")
                return torch.stack(weights_list).mean(dim=0)

        loader = MockLoader()
        widen = MockWIDEN()
        set_id_map = {id(inner_lora): "set_inner", id(outer_lora): "set_outer"}

        evaluate_recipe(
            keys=["k0"],
            base_batch=base_batch,
            recipe_node=outer_merge,
            loader=loader,
            widen=widen,
            set_id_map=set_id_map,
            device="cpu",
            dtype=torch.float32,
        )

        # Should have 2 filter calls: inner merge first, then outer
        assert len(call_order) == 2
        assert call_order[0] == "filter"  # inner
        assert call_order[1] == "filter"  # outer


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

        # Install some merged patches (keys already have diffusion_model. prefix)
        merged_state = {
            "diffusion_model.input_blocks.0.0.weight": torch.randn(4, 4),
        }
        result = install_merged_patches(mock_model_patcher, merged_state, torch.float32)

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

        key = "diffusion_model.input_blocks.0.0.weight"
        merged_state = {key: torch.randn(4, 4)}
        result = install_merged_patches(mock_model_patcher, merged_state, torch.float32)

        # Check patch format
        patch_entry = result.patches[key][0]

        # Format: (strength_patch, ("set", (tensor,)), strength_model, None, None)
        assert patch_entry[0] == 1.0  # strength_patch
        assert patch_entry[1][0] == "set"  # patch type
        assert isinstance(patch_entry[1][1], tuple)  # value wrapped in tuple
        assert isinstance(patch_entry[1][1][0], torch.Tensor)  # actual tensor


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

        # Merged state in fp32 (computation dtype), prefixed keys
        key = "diffusion_model.input_blocks.0.0.weight"
        merged_state = {key: torch.randn(4, 4, dtype=torch.float32)}
        result = install_merged_patches(patcher, merged_state, torch.bfloat16)

        # Patch should be bf16
        patch_tensor = result.patches[key][0][1][1][0]
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

        key = "diffusion_model.input_blocks.0.0.weight"
        merged_state = {key: torch.randn(4, 4, dtype=torch.float32)}
        result = install_merged_patches(patcher, merged_state, torch.float16)
        patch_tensor = result.patches[key][0][1][1][0]
        assert patch_tensor.dtype == torch.float16


# =============================================================================
# AC-9: Progress reported via ProgressBar per batch group
# =============================================================================


class TestProgressBarPerBatchGroup:
    """AC: @exit-node ac-9

    Given: the exit node processes multiple batch groups
    When: each batch group completes
    Then: progress is reported via ComfyUI ProgressBar
    """

    def test_progress_bar_created_with_group_count(self, mock_model_patcher):
        """ProgressBar should be created with total = number of batch groups."""
        # AC: @exit-node ac-9
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        node = WIDENExitNode()

        mock_pbar = MagicMock()
        mock_pbar_cls = MagicMock(return_value=mock_pbar)

        with (
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", mock_pbar_cls),
        ):
            mock_loader = MagicMock()
            mock_loader.cleanup = MagicMock()

            # Two affected keys with different shapes → 2 batch groups
            mock_analyze.return_value = MagicMock(
                model_patcher=mock_model_patcher,
                arch="sdxl",
                loader=mock_loader,
                set_affected={str(id(lora)): {"diffusion_model.k1", "diffusion_model.k2"}},
                affected_keys={"diffusion_model.k1", "diffusion_model.k2"},
            )

            # Override state dict to include keys with different shapes
            mock_model_patcher._state_dict["diffusion_model.k1"] = torch.randn(4, 4)
            mock_model_patcher._state_dict["diffusion_model.k2"] = torch.randn(8, 8)

            with patch("nodes.exit.chunked_evaluation") as mock_chunked:
                mock_chunked.return_value = {}

                (result,) = node.execute(merge)

            # ProgressBar created with number of batch groups
            mock_pbar_cls.assert_called_once()
            n_groups = mock_pbar_cls.call_args[0][0]
            assert n_groups >= 1  # At least one group

            # update(1) called once per batch group
            assert mock_pbar.update.call_count == n_groups
            for call in mock_pbar.update.call_args_list:
                assert call[0] == (1,)

    def test_progress_works_when_progressbar_unavailable(self, mock_model_patcher):
        """Execution should work fine when ProgressBar is None (no ComfyUI)."""
        # AC: @exit-node ac-9
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        node = WIDENExitNode()

        with (
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
        ):
            mock_loader = MagicMock()
            mock_loader.cleanup = MagicMock()
            mock_analyze.return_value = MagicMock(
                model_patcher=mock_model_patcher,
                arch="sdxl",
                loader=mock_loader,
                set_affected={str(id(lora)): {"diffusion_model.k1"}},
                affected_keys={"diffusion_model.k1"},
            )
            mock_model_patcher._state_dict["diffusion_model.k1"] = torch.randn(4, 4)

            with patch("nodes.exit.chunked_evaluation", return_value={}):
                # Should not raise even with ProgressBar=None
                (result,) = node.execute(merge)

            assert result is not None


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


# =============================================================================
# Persistence INPUT_TYPES
# =============================================================================


class TestExitNodePersistenceInputs:
    """Verify persistence-related INPUT_TYPES are present."""

    def test_save_model_input(self):
        """save_model BOOLEAN should be in optional inputs."""
        optional = WIDENExitNode.INPUT_TYPES()["optional"]
        assert "save_model" in optional
        assert optional["save_model"][0] == "BOOLEAN"
        assert optional["save_model"][1]["default"] is False

    def test_model_name_input(self):
        """model_name STRING should be in optional inputs."""
        optional = WIDENExitNode.INPUT_TYPES()["optional"]
        assert "model_name" in optional
        assert optional["model_name"][0] == "STRING"

    def test_save_workflow_input(self):
        """save_workflow BOOLEAN should default to True."""
        optional = WIDENExitNode.INPUT_TYPES()["optional"]
        assert "save_workflow" in optional
        assert optional["save_workflow"][1]["default"] is True

    def test_hidden_inputs(self):
        """prompt and extra_pnginfo should be hidden inputs."""
        hidden = WIDENExitNode.INPUT_TYPES()["hidden"]
        assert hidden["prompt"] == "PROMPT"
        assert hidden["extra_pnginfo"] == "EXTRA_PNGINFO"


# =============================================================================
# AC-1: save_model=False — default behavior unchanged
# =============================================================================


class TestSaveModelOff:
    """AC: @exit-model-persistence ac-1

    save_model=False should not trigger any persistence I/O.
    """

    # AC: @exit-model-persistence ac-1
    def test_default_no_persistence_base(self, mock_model_patcher):
        """With save_model=False on RecipeBase, no persistence functions should be called."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")

        node = WIDENExitNode()
        (result,) = node.execute(base, save_model=False, model_name="test")
        assert result is not mock_model_patcher

    # AC: @exit-model-persistence ac-1
    def test_no_persistence_on_merge_flow(self, mock_model_patcher):
        """With save_model=False on RecipeMerge, persistence functions are skipped."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        node = WIDENExitNode()

        with (
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.chunked_evaluation", return_value={}),
            patch("nodes.exit.validate_model_name") as mock_validate,
            patch("nodes.exit.atomic_save") as mock_save,
        ):
            mock_loader = MagicMock()
            mock_loader.cleanup = MagicMock()
            mock_analyze.return_value = MagicMock(
                model_patcher=mock_model_patcher,
                arch="sdxl",
                loader=mock_loader,
                set_affected={},
                affected_keys=set(),
            )

            (result,) = node.execute(merge, save_model=False, model_name="test")

            # Persistence functions should NOT be called
            mock_validate.assert_not_called()
            mock_save.assert_not_called()


# =============================================================================
# AC-3: Cache hit skips GPU
# =============================================================================


class TestSaveModelCacheHit:
    """AC: @exit-model-persistence ac-3

    On cache hit, GPU pipeline (analyze_recipe) should be skipped entirely.
    """

    # AC: @exit-model-persistence ac-3
    def test_cache_hit_skips_analyze(self, mock_model_patcher, tmp_path):
        """Cache hit should skip analyze_recipe and return patched model."""
        from safetensors.torch import save_file

        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        # Create a fake cached file
        cached_path = tmp_path / "cached.safetensors"
        key = "diffusion_model.input_blocks.0.0.weight"
        cached_tensors = {key: torch.randn(4, 4)}
        cached_metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe__": "{}",
            "__ecaj_recipe_hash__": "will_match",
            "__ecaj_affected_keys__": f'["{key}"]',
        }
        save_file(cached_tensors, str(cached_path), metadata=cached_metadata)

        node = WIDENExitNode()

        with (
            patch("nodes.exit.validate_model_name", return_value="cached.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=str(cached_path)),
            patch("nodes.exit.compute_recipe_hash", return_value="will_match"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
        ):
            (result,) = node.execute(
                merge, save_model=True, model_name="cached"
            )

            # analyze_recipe should NOT have been called
            mock_analyze.assert_not_called()

        # Result should be a patched model
        assert result is not mock_model_patcher
        assert key in result.patches


# =============================================================================
# AC-2, AC-4: Cache miss — saves after GPU
# =============================================================================


class TestSaveModelCacheMiss:
    """AC: @exit-model-persistence ac-2, ac-4

    On cache miss, GPU pipeline runs normally and file is saved after.
    """

    # AC: @exit-model-persistence ac-2
    def test_saves_after_gpu(self, mock_model_patcher, tmp_path):
        """Cache miss should run GPU pipeline and save result."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        affected_key = "diffusion_model.input_blocks.0.0.weight"

        node = WIDENExitNode()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.check_cache", return_value=None),  # cache miss
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.chunked_evaluation") as mock_chunked,
            patch("nodes.exit.atomic_save") as mock_save,
        ):
            mock_loader = MagicMock()
            mock_loader.cleanup = MagicMock()
            mock_analyze.return_value = MagicMock(
                model_patcher=mock_model_patcher,
                arch="sdxl",
                loader=mock_loader,
                set_affected={str(id(lora)): {affected_key}},
                affected_keys={affected_key},
            )
            mock_chunked.return_value = {affected_key: torch.randn(4, 4)}

            (result,) = node.execute(
                merge, save_model=True, model_name="model"
            )

            # analyze_recipe SHOULD have been called
            mock_analyze.assert_called_once()
            # atomic_save SHOULD have been called
            mock_save.assert_called_once()

    # AC: @exit-model-persistence ac-4
    def test_overwrites_stale_cache(self, mock_model_patcher, tmp_path):
        """Hash mismatch should overwrite the stale cached file."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        affected_key = "diffusion_model.input_blocks.0.0.weight"

        node = WIDENExitNode()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="new_hash"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.check_cache", return_value=None),  # stale = mismatch
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.chunked_evaluation") as mock_chunked,
            patch("nodes.exit.atomic_save") as mock_save,
        ):
            mock_loader = MagicMock()
            mock_loader.cleanup = MagicMock()
            mock_analyze.return_value = MagicMock(
                model_patcher=mock_model_patcher,
                arch="sdxl",
                loader=mock_loader,
                set_affected={str(id(lora)): {affected_key}},
                affected_keys={affected_key},
            )
            mock_chunked.return_value = {affected_key: torch.randn(4, 4)}

            (result,) = node.execute(
                merge, save_model=True, model_name="model"
            )

            mock_save.assert_called_once()


# =============================================================================
# IS_CHANGED with persistence
# =============================================================================


class TestIsChangedPersistence:
    """IS_CHANGED should incorporate save_model, model_name, and file mtime."""

    def test_save_model_false_uses_base_hash(self, mock_model_patcher):
        """save_model=False should return the base recipe hash."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        result = WIDENExitNode.IS_CHANGED(merge, save_model=False)
        # Should be a hex string (SHA-256)
        assert len(result) == 64

    def test_save_model_true_different_from_false(self, mock_model_patcher):
        """save_model=True should produce a different hash than False."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        result_off = WIDENExitNode.IS_CHANGED(merge, save_model=False)
        result_on = WIDENExitNode.IS_CHANGED(
            merge, save_model=True, model_name="test"
        )
        assert result_off != result_on

    def test_ignores_prompt_and_extra_pnginfo(self, mock_model_patcher):
        """prompt and extra_pnginfo should not affect the hash."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        result1 = WIDENExitNode.IS_CHANGED(
            merge, save_model=True, model_name="test",
            prompt={"a": 1}, extra_pnginfo={"workflow": {}},
        )
        result2 = WIDENExitNode.IS_CHANGED(
            merge, save_model=True, model_name="test",
            prompt={"b": 2}, extra_pnginfo={"different": {}},
        )
        assert result1 == result2
