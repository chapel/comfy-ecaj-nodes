"""Tests for Exit Patch Installation — AC coverage for @exit-patch-install."""

import os
import sys
import tempfile
import time
from unittest.mock import MagicMock

import pytest
import torch

from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge
from nodes.exit import (
    WIDENExitNode,
    _collect_lora_paths,
    _compute_recipe_hash,
    _unpatch_loaded_clones,
    install_merged_patches,
)
from tests.conftest import MockModelPatcher


def _dir_resolver(base_dir: str):
    """Create a resolver that joins LoRA names to a base directory."""
    return lambda name: os.path.join(base_dir, name)


class TestInstallMergedPatches:
    """Tests for install_merged_patches function."""

    # AC: @exit-patch-install ac-1
    def test_clones_model_patcher(self, mock_model_patcher: MockModelPatcher):
        """Clones the original ModelPatcher instead of mutating it."""
        original_uuid = mock_model_patcher.patches_uuid

        merged_state = {"diffusion_model.input_blocks.0.0.weight": torch.randn(4, 4)}
        result = install_merged_patches(mock_model_patcher, merged_state, torch.float32)

        # Result is different object
        assert result is not mock_model_patcher
        # Original unchanged
        assert mock_model_patcher.patches_uuid == original_uuid
        assert len(mock_model_patcher.patches) == 0

    # AC: @exit-patch-install ac-1
    def test_adds_set_patches(self, mock_model_patcher: MockModelPatcher):
        """Merged weights are added as set patches."""
        key = "diffusion_model.input_blocks.0.0.weight"
        merged_state = {key: torch.randn(4, 4)}
        result = install_merged_patches(mock_model_patcher, merged_state, torch.float32)

        # Check patch was added
        assert key in result.patches
        # Patch entry format: [(strength, value, strength_model, None, None)]
        patch_entry = result.patches[key][0]
        assert patch_entry[0] == 1.0  # strength_patch
        assert patch_entry[1][0] == "set"  # patch type

    # AC: @exit-patch-install ac-2
    def test_keys_use_diffusion_model_prefix(self, mock_model_patcher: MockModelPatcher):
        """Keys with diffusion_model. prefix are passed through to ModelPatcher."""
        merged_state = {
            "diffusion_model.input_blocks.0.0.weight": torch.randn(4, 4),
            "diffusion_model.middle_block.0.weight": torch.randn(4, 4),
        }
        result = install_merged_patches(mock_model_patcher, merged_state, torch.float32)

        # Both keys should be present as-is
        assert "diffusion_model.input_blocks.0.0.weight" in result.patches
        assert "diffusion_model.middle_block.0.weight" in result.patches

    # AC: @exit-patch-install ac-3
    def test_transfers_tensors_to_cpu(self, mock_model_patcher: MockModelPatcher):
        """All patch tensors are on CPU."""
        key = "diffusion_model.input_blocks.0.0.weight"
        merged_state = {key: torch.randn(4, 4)}
        result = install_merged_patches(mock_model_patcher, merged_state, torch.float32)

        # Get the patch tensor
        patch_entry = result.patches[key][0]
        patch_tensor = patch_entry[1][1][0]  # ("set", (tensor,))
        assert patch_tensor.device.type == "cpu"

    # AC: @exit-patch-install ac-4
    def test_matches_base_model_dtype_float32(self, mock_model_patcher: MockModelPatcher):
        """Patch tensors match base model dtype (float32 case)."""
        key = "diffusion_model.input_blocks.0.0.weight"
        merged_state = {key: torch.randn(4, 4, dtype=torch.float16)}
        result = install_merged_patches(mock_model_patcher, merged_state, torch.float32)

        patch_entry = result.patches[key][0]
        patch_tensor = patch_entry[1][1][0]
        assert patch_tensor.dtype == torch.float32

    # AC: @exit-patch-install ac-4
    def test_matches_base_model_dtype_bfloat16(self):
        """Patch tensors match base model dtype (bf16 case)."""
        # Create MockModelPatcher with bf16 tensors
        patcher = MockModelPatcher()
        # Replace state dict with bf16 tensors
        for k in patcher._state_dict:
            patcher._state_dict[k] = patcher._state_dict[k].to(torch.bfloat16)

        key = "diffusion_model.input_blocks.0.0.weight"
        merged_state = {key: torch.randn(4, 4, dtype=torch.float32)}
        result = install_merged_patches(patcher, merged_state, torch.bfloat16)

        patch_entry = result.patches[key][0]
        patch_tensor = patch_entry[1][1][0]
        assert patch_tensor.dtype == torch.bfloat16

    def test_handles_multiple_keys(self, mock_model_patcher: MockModelPatcher):
        """Multiple merged tensors are all installed."""
        merged_state = {
            "diffusion_model.input_blocks.0.0.weight": torch.randn(4, 4),
            "diffusion_model.input_blocks.1.0.weight": torch.randn(4, 4),
            "diffusion_model.middle_block.0.weight": torch.randn(4, 4),
            "diffusion_model.output_blocks.0.0.weight": torch.randn(4, 4),
        }
        result = install_merged_patches(mock_model_patcher, merged_state, torch.float32)

        # All keys should be patched
        assert len(result.patches) == 4


class TestUnpatchLoadedClones:
    """Tests for _unpatch_loaded_clones that prevents double-application.

    Regression: ComfyUI keeps models patched in-place between prompts.
    model_state_dict() returns patched values on subsequent runs, causing
    LoRA deltas to be applied on top of already-merged weights.
    _unpatch_loaded_clones forces any loaded clone to fully unload,
    restoring clean base weights before we read model_state_dict().
    """

    @pytest.fixture()
    def _patch_loaded_models(self, monkeypatch):
        """Wire current_loaded_models onto the comfy.model_management stub."""
        mm = sys.modules["comfy.model_management"]
        loaded: list = []
        monkeypatch.setattr(mm, "current_loaded_models", loaded, raising=False)
        return loaded

    # AC: @exit-patch-install ac-7
    def test_noop_without_comfy(self, mock_model_patcher: MockModelPatcher):
        """Does not crash when comfy.model_management has no current_loaded_models."""
        _unpatch_loaded_clones(mock_model_patcher)

    # AC: @exit-patch-install ac-7
    def test_noop_with_empty_loaded_models(
        self, mock_model_patcher: MockModelPatcher, _patch_loaded_models
    ):
        """Safe when current_loaded_models is empty."""
        _unpatch_loaded_clones(mock_model_patcher)
        assert _patch_loaded_models == []

    # AC: @exit-patch-install ac-7
    def test_unloads_matching_clone(
        self, mock_model_patcher: MockModelPatcher, _patch_loaded_models
    ):
        """Matching loaded clone is unloaded and removed from the list."""
        clone = mock_model_patcher.clone()
        loaded_entry = MagicMock()
        loaded_entry.model = clone
        _patch_loaded_models.append(loaded_entry)

        _unpatch_loaded_clones(mock_model_patcher)

        loaded_entry.model_unload.assert_called_once()
        assert len(_patch_loaded_models) == 0

    # AC: @exit-patch-install ac-7
    def test_preserves_non_matching_entries(
        self, _patch_loaded_models
    ):
        """Non-matching entries are not touched."""
        other_patcher = MockModelPatcher()
        target_patcher = MockModelPatcher()

        non_matching = MagicMock()
        non_matching.model = other_patcher
        _patch_loaded_models.append(non_matching)

        _unpatch_loaded_clones(target_patcher)

        non_matching.model_unload.assert_not_called()
        assert len(_patch_loaded_models) == 1

    # AC: @exit-patch-install ac-7
    def test_unloads_only_matching_among_mixed(
        self, mock_model_patcher: MockModelPatcher, _patch_loaded_models
    ):
        """Only matching clones are removed; others stay."""
        clone = mock_model_patcher.clone()

        matching = MagicMock()
        matching.model = clone

        non_matching = MagicMock()
        non_matching.model = MockModelPatcher()

        _patch_loaded_models.extend([non_matching, matching])

        _unpatch_loaded_clones(mock_model_patcher)

        matching.model_unload.assert_called_once()
        non_matching.model_unload.assert_not_called()
        assert len(_patch_loaded_models) == 1
        assert _patch_loaded_models[0] is non_matching


class TestCollectLoraPaths:
    """Tests for _collect_lora_paths helper."""

    def test_recipe_base_returns_empty(self, mock_model_patcher: MockModelPatcher):
        """RecipeBase has no LoRA paths."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        paths = _collect_lora_paths(base)
        assert paths == []

    def test_recipe_lora_single(self):
        """RecipeLoRA with one LoRA returns its path."""
        lora = RecipeLoRA(loras=({"path": "lora_a.safetensors", "strength": 1.0},))
        paths = _collect_lora_paths(lora)
        assert paths == ["lora_a.safetensors"]

    def test_recipe_lora_multiple(self):
        """RecipeLoRA with multiple LoRAs returns all paths."""
        lora = RecipeLoRA(
            loras=(
                {"path": "lora_a.safetensors", "strength": 1.0},
                {"path": "lora_b.safetensors", "strength": 0.5},
            )
        )
        paths = _collect_lora_paths(lora)
        assert paths == ["lora_a.safetensors", "lora_b.safetensors"]

    def test_recipe_compose(self):
        """RecipeCompose collects from all branches."""
        lora_a = RecipeLoRA(loras=({"path": "lora_a.safetensors", "strength": 1.0},))
        lora_b = RecipeLoRA(loras=({"path": "lora_b.safetensors", "strength": 0.5},))
        compose = RecipeCompose(branches=(lora_a, lora_b))
        paths = _collect_lora_paths(compose)
        assert "lora_a.safetensors" in paths
        assert "lora_b.safetensors" in paths

    def test_recipe_merge(self, mock_model_patcher: MockModelPatcher):
        """RecipeMerge collects from base, target, and backbone."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        target = RecipeLoRA(loras=({"path": "lora_a.safetensors", "strength": 1.0},))
        backbone_lora = RecipeLoRA(loras=({"path": "backbone.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=target, backbone=backbone_lora, t_factor=1.0)
        paths = _collect_lora_paths(merge)
        assert "lora_a.safetensors" in paths
        assert "backbone.safetensors" in paths

    def test_deeply_nested(self, mock_model_patcher: MockModelPatcher):
        """Deeply nested recipe tree collects all paths."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora_a = RecipeLoRA(loras=({"path": "lora_a.safetensors", "strength": 1.0},))
        lora_b = RecipeLoRA(loras=({"path": "lora_b.safetensors", "strength": 0.5},))
        lora_c = RecipeLoRA(loras=({"path": "lora_c.safetensors", "strength": 0.3},))

        # Merge 1: base + lora_a
        merge1 = RecipeMerge(base=base, target=lora_a, backbone=None, t_factor=1.0)
        # Compose lora_b and lora_c
        compose = RecipeCompose(branches=(lora_b, lora_c))
        # Merge 2: merge1 + compose
        merge2 = RecipeMerge(base=merge1, target=compose, backbone=None, t_factor=0.8)

        paths = _collect_lora_paths(merge2)
        assert set(paths) == {"lora_a.safetensors", "lora_b.safetensors", "lora_c.safetensors"}


class TestComputeRecipeHash:
    """Tests for _compute_recipe_hash function."""

    # AC: @exit-patch-install ac-5
    def test_identical_hash_no_changes(self):
        """Same recipe with unchanged files produces identical hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a LoRA file
            lora_path = os.path.join(tmpdir, "lora_a.safetensors")
            with open(lora_path, "wb") as f:
                f.write(b"fake lora data")

            lora = RecipeLoRA(loras=({"path": "lora_a.safetensors", "strength": 1.0},))

            hash1 = _compute_recipe_hash(lora, lora_path_resolver=_dir_resolver(tmpdir))
            hash2 = _compute_recipe_hash(lora, lora_path_resolver=_dir_resolver(tmpdir))

            assert hash1 == hash2

    # AC: @exit-patch-install ac-6
    def test_different_hash_on_modification(self):
        """Modified LoRA file produces different hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = os.path.join(tmpdir, "lora_a.safetensors")

            # Create initial file
            with open(lora_path, "wb") as f:
                f.write(b"initial data")

            lora = RecipeLoRA(loras=({"path": "lora_a.safetensors", "strength": 1.0},))
            hash1 = _compute_recipe_hash(lora, lora_path_resolver=_dir_resolver(tmpdir))

            # Wait to ensure mtime changes (filesystem resolution)
            time.sleep(0.1)

            # Modify the file
            with open(lora_path, "wb") as f:
                f.write(b"modified data with more content")

            hash2 = _compute_recipe_hash(lora, lora_path_resolver=_dir_resolver(tmpdir))

            assert hash1 != hash2

    def test_different_hash_for_different_paths(self):
        """Different LoRA paths produce different hashes."""
        lora_a = RecipeLoRA(loras=({"path": "lora_a.safetensors", "strength": 1.0},))
        lora_b = RecipeLoRA(loras=({"path": "lora_b.safetensors", "strength": 1.0},))

        # Both files don't exist, but paths differ
        hash_a = _compute_recipe_hash(lora_a)
        hash_b = _compute_recipe_hash(lora_b)

        assert hash_a != hash_b

    def test_missing_file_uses_sentinel(self):
        """Missing file produces consistent hash (sentinel values)."""
        lora = RecipeLoRA(loras=({"path": "nonexistent.safetensors", "strength": 1.0},))

        hash1 = _compute_recipe_hash(lora)
        hash2 = _compute_recipe_hash(lora)

        assert hash1 == hash2

    def test_deterministic_ordering(self):
        """Hash is deterministic regardless of collection order."""
        # Create compose with branches in different order
        lora_a = RecipeLoRA(loras=({"path": "a.safetensors", "strength": 1.0},))
        lora_b = RecipeLoRA(loras=({"path": "b.safetensors", "strength": 1.0},))
        lora_c = RecipeLoRA(loras=({"path": "c.safetensors", "strength": 1.0},))

        compose1 = RecipeCompose(branches=(lora_a, lora_b, lora_c))
        compose2 = RecipeCompose(branches=(lora_c, lora_a, lora_b))

        hash1 = _compute_recipe_hash(compose1)
        hash2 = _compute_recipe_hash(compose2)

        # Same files, different order in tree — should produce same hash
        assert hash1 == hash2


class TestWIDENExitNodeIsChanged:
    """Tests for WIDENExitNode.IS_CHANGED classmethod."""

    # AC: @exit-patch-install ac-5
    def test_is_changed_cache_hit(self):
        """IS_CHANGED returns identical value when no LoRA changes."""
        lora = RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},))

        hash1 = WIDENExitNode.IS_CHANGED(lora)
        hash2 = WIDENExitNode.IS_CHANGED(lora)

        assert hash1 == hash2

    # AC: @exit-patch-install ac-6
    def test_is_changed_cache_miss_on_modification(self):
        """IS_CHANGED returns different value when LoRA file changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = os.path.join(tmpdir, "test.safetensors")

            # Create initial file
            with open(lora_path, "wb") as f:
                f.write(b"original")

            # Use full path so IS_CHANGED can stat it
            lora = RecipeLoRA(loras=({"path": lora_path, "strength": 1.0},))
            hash1 = WIDENExitNode.IS_CHANGED(lora)

            # Modify file
            time.sleep(0.1)
            with open(lora_path, "wb") as f:
                f.write(b"modified content here")

            hash2 = WIDENExitNode.IS_CHANGED(lora)

            assert hash1 != hash2

    def test_is_changed_with_complex_recipe(self, mock_model_patcher: MockModelPatcher):
        """IS_CHANGED works with nested recipe structures."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora_a = RecipeLoRA(loras=({"path": "a.safetensors", "strength": 1.0},))
        lora_b = RecipeLoRA(loras=({"path": "b.safetensors", "strength": 0.5},))
        compose = RecipeCompose(branches=(lora_a, lora_b))
        merge = RecipeMerge(base=base, target=compose, backbone=None, t_factor=1.0)

        # Should not raise
        hash_val = WIDENExitNode.IS_CHANGED(merge)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA-256 hex digest


class TestWIDENExitNodeBasic:
    """Basic tests for WIDENExitNode."""

    def test_input_types(self):
        """INPUT_TYPES includes widen input."""
        input_types = WIDENExitNode.INPUT_TYPES()
        assert "required" in input_types
        assert "widen" in input_types["required"]
        assert input_types["required"]["widen"] == ("WIDEN",)

    def test_return_types(self):
        """RETURN_TYPES is MODEL."""
        assert WIDENExitNode.RETURN_TYPES == ("MODEL",)
        assert WIDENExitNode.RETURN_NAMES == ("model",)

    def test_category(self):
        """Category is ecaj/merge."""
        assert WIDENExitNode.CATEGORY == "ecaj/merge"

    def test_function_name(self):
        """Function name is execute."""
        assert WIDENExitNode.FUNCTION == "execute"

    def test_execute_rejects_lora_at_root(self):
        """Execute raises ValueError for LoRA at root (must use Merge)."""
        node = WIDENExitNode()
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        with pytest.raises(ValueError) as exc_info:
            node.execute(lora)
        assert "RecipeLoRA" in str(exc_info.value)
