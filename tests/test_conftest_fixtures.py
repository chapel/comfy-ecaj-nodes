"""Tests for conftest.py fixtures — AC coverage for @comfyui-mocking spec.

These tests verify the fixtures themselves work correctly.
"""

import torch

from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge
from tests.conftest import _SDXL_KEYS, _ZIMAGE_KEYS, MockModelPatcher

# =============================================================================
# AC-1: MockModelPatcher fixture
# =============================================================================


class TestMockModelPatcherFixture:
    """AC: @comfyui-mocking ac-1

    Given: a test needs a ModelPatcher-like object
    When: it uses the mock_model_patcher fixture
    Then: the mock provides model_state_dict, clone, add_patches,
          get_key_patches, and patches_uuid
    """

    def test_mock_model_patcher_fixture_returns_instance(self, mock_model_patcher):
        """Fixture returns MockModelPatcher instance."""
        # AC: @comfyui-mocking ac-1
        assert isinstance(mock_model_patcher, MockModelPatcher)

    def test_model_state_dict_returns_sdxl_keys(self, mock_model_patcher):
        """model_state_dict returns dict with SDXL-like keys."""
        # AC: @comfyui-mocking ac-1
        sd = mock_model_patcher.model_state_dict()
        assert "diffusion_model.input_blocks.0.0.weight" in sd

    def test_model_state_dict_filter_prefix(self, mock_model_patcher):
        """model_state_dict with filter_prefix returns filtered keys."""
        # AC: @comfyui-mocking ac-1
        sd = mock_model_patcher.model_state_dict(filter_prefix="diffusion_model.input")
        assert all(k.startswith("diffusion_model.input") for k in sd)

    def test_clone_returns_new_instance(self, mock_model_patcher):
        """clone() returns a new MockModelPatcher."""
        # AC: @comfyui-mocking ac-1
        cloned = mock_model_patcher.clone()
        assert isinstance(cloned, MockModelPatcher)
        assert cloned is not mock_model_patcher

    def test_add_patches_stores_patches(self, mock_model_patcher):
        """add_patches() stores patches for valid keys."""
        # AC: @comfyui-mocking ac-1
        key = "diffusion_model.input_blocks.0.0.weight"
        mock_model_patcher.add_patches({key: torch.zeros(4, 4)})
        assert key in mock_model_patcher.patches

    def test_get_key_patches_returns_patch_data(self, mock_model_patcher):
        """get_key_patches() returns stored patches."""
        # AC: @comfyui-mocking ac-1
        key = "diffusion_model.input_blocks.0.0.weight"
        mock_model_patcher.add_patches({key: torch.zeros(4, 4)})
        patches = mock_model_patcher.get_key_patches()
        assert key in patches
        assert len(patches[key]) > 1  # base + patch

    def test_patches_uuid_property(self, mock_model_patcher):
        """patches_uuid property exists and is updated on changes."""
        # AC: @comfyui-mocking ac-1
        original_uuid = mock_model_patcher.patches_uuid
        key = "diffusion_model.input_blocks.0.0.weight"
        mock_model_patcher.add_patches({key: torch.zeros(4, 4)})
        assert mock_model_patcher.patches_uuid != original_uuid


# =============================================================================
# AC-2: Recipe tree fixtures
# =============================================================================


class TestRecipeFixtures:
    """AC: @comfyui-mocking ac-2

    Given: a test needs a recipe tree
    When: it uses recipe fixtures
    Then: pre-built recipe trees are available for single-LoRA, multi-LoRA set,
          compose (2 branches), chain (2 sequential merges), and full
          (compose + chain) patterns
    """

    def test_recipe_single_lora_fixture(self, recipe_single_lora):
        """single-LoRA fixture provides RecipeLoRA with one entry."""
        # AC: @comfyui-mocking ac-2
        assert isinstance(recipe_single_lora, RecipeLoRA)
        assert len(recipe_single_lora.loras) == 1

    def test_recipe_multi_lora_fixture(self, recipe_multi_lora):
        """multi-LoRA fixture provides RecipeLoRA with multiple entries."""
        # AC: @comfyui-mocking ac-2
        assert isinstance(recipe_multi_lora, RecipeLoRA)
        assert len(recipe_multi_lora.loras) == 2

    def test_recipe_compose_fixture(self, recipe_compose):
        """compose fixture provides RecipeCompose with 2 branches."""
        # AC: @comfyui-mocking ac-2
        assert isinstance(recipe_compose, RecipeCompose)
        assert len(recipe_compose.branches) == 2

    def test_recipe_chain_fixture(self, recipe_chain):
        """chain fixture provides RecipeMerge with 2 sequential merges."""
        # AC: @comfyui-mocking ac-2
        assert isinstance(recipe_chain, RecipeMerge)
        # Outer merge
        assert isinstance(recipe_chain.base, RecipeMerge)
        # Inner merge has RecipeBase
        assert isinstance(recipe_chain.base.base, RecipeBase)

    def test_recipe_full_fixture(self, recipe_full):
        """full fixture provides compose + chain pattern."""
        # AC: @comfyui-mocking ac-2
        assert isinstance(recipe_full, RecipeMerge)
        # Outer merge targets a LoRA
        assert isinstance(recipe_full.target, RecipeLoRA)
        # Inner merge targets a compose
        assert isinstance(recipe_full.base, RecipeMerge)
        assert isinstance(recipe_full.base.target, RecipeCompose)


# =============================================================================
# AC-3: ComfyUI sys.modules mocking
# =============================================================================


class TestComfyUIMocking:
    """AC: @comfyui-mocking ac-3

    Given: tests for nodes that import ComfyUI modules like folder_paths
    When: they run without ComfyUI installed
    Then: ComfyUI modules are mocked via sys.modules patching
    """

    def test_folder_paths_mocked(self):
        """folder_paths is mocked and can be imported."""
        # AC: @comfyui-mocking ac-3
        import folder_paths

        assert hasattr(folder_paths, "get_filename_list")

    def test_comfy_mocked(self):
        """comfy module is mocked."""
        # AC: @comfyui-mocking ac-3
        import comfy

        assert comfy is not None

    def test_comfy_utils_mocked(self):
        """comfy.utils is mocked."""
        # AC: @comfyui-mocking ac-3
        import sys

        # The mock is injected via sys.modules, verify it exists
        assert "comfy.utils" in sys.modules

    def test_comfy_model_management_mocked(self):
        """comfy.model_management is mocked."""
        # AC: @comfyui-mocking ac-3
        import sys

        # The mock is injected via sys.modules, verify it exists
        assert "comfy.model_management" in sys.modules

    def test_node_imports_work_without_comfyui(self):
        """Node modules can be imported without real ComfyUI."""
        # AC: @comfyui-mocking ac-3
        # This import would fail without the mock
        from nodes.lora import WIDENLoRANode

        assert WIDENLoRANode is not None


# =============================================================================
# AC-4: Architecture-specific fixtures
# =============================================================================


class TestArchSpecificFixtures:
    """AC: @comfyui-mocking ac-4

    Given: a test needs fake SDXL or Z-Image state dict keys
    When: it uses arch-specific fixtures
    Then: fixture provides a dict with representative key patterns for each
          supported architecture
    """

    def test_sdxl_state_dict_keys_fixture(self, sdxl_state_dict_keys):
        """SDXL fixture provides input_blocks keys."""
        # AC: @comfyui-mocking ac-4
        assert any("input_blocks" in k for k in sdxl_state_dict_keys)
        assert any("middle_block" in k for k in sdxl_state_dict_keys)
        assert any("output_blocks" in k for k in sdxl_state_dict_keys)

    def test_zimage_state_dict_keys_fixture(self, zimage_state_dict_keys):
        """Z-Image fixture provides layers + noise_refiner keys."""
        # AC: @comfyui-mocking ac-4
        assert any("layers" in k for k in zimage_state_dict_keys)
        assert any("noise_refiner" in k for k in zimage_state_dict_keys)
        assert any("context_refiner" in k for k in zimage_state_dict_keys)

    def test_mock_model_patcher_zimage_fixture(self, mock_model_patcher_zimage):
        """Z-Image MockModelPatcher has correct keys."""
        # AC: @comfyui-mocking ac-4
        sd = mock_model_patcher_zimage.model_state_dict()
        assert any("layers" in k for k in sd)
        assert any("noise_refiner" in k for k in sd)

    def test_sdxl_keys_constant(self):
        """_SDXL_KEYS constant has expected patterns."""
        # AC: @comfyui-mocking ac-4
        assert len(_SDXL_KEYS) >= 4
        assert any("input_blocks" in k for k in _SDXL_KEYS)

    def test_zimage_keys_constant(self):
        """_ZIMAGE_KEYS constant has expected patterns."""
        # AC: @comfyui-mocking ac-4
        assert len(_ZIMAGE_KEYS) >= 5
        assert any("layers" in k for k in _ZIMAGE_KEYS)
        assert any("noise_refiner" in k for k in _ZIMAGE_KEYS)
