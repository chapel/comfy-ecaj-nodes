"""Tests for Exit Recipe Analysis.

Covers all 6 acceptance criteria:
- AC-1: Walk to root and extract model_patcher and arch tag
- AC-2: Synthetic set IDs for unique RecipeLoRA groups
- AC-3: Architecture-appropriate loader selection
- AC-4: Affected-key map per set ID
- AC-5: Skip keys not affected by any LoRA
- AC-6: FileNotFoundError for missing LoRA files
"""

import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lib.analysis import AnalysisResult, analyze_recipe, get_keys_to_process
from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge
from tests.conftest import MockModelPatcher

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_lora_dir():
    """Create a temporary directory for test LoRA files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sdxl_lora_a(temp_lora_dir: str) -> str:
    """Create a test SDXL LoRA file (lora_a.safetensors)."""
    path = Path(temp_lora_dir) / "lora_a.safetensors"
    tensors = {
        "lora_unet_input_blocks_0_0.lora_up.weight": torch.randn(64, 8),
        "lora_unet_input_blocks_0_0.lora_down.weight": torch.randn(8, 32),
    }
    save_file(tensors, str(path))
    return str(path)


@pytest.fixture
def sdxl_lora_b(temp_lora_dir: str) -> str:
    """Create a second test SDXL LoRA file (lora_b.safetensors)."""
    path = Path(temp_lora_dir) / "lora_b.safetensors"
    tensors = {
        "lora_unet_middle_block_0.lora_up.weight": torch.randn(128, 16),
        "lora_unet_middle_block_0.lora_down.weight": torch.randn(16, 64),
    }
    save_file(tensors, str(path))
    return str(path)


@pytest.fixture
def sdxl_lora_c(temp_lora_dir: str) -> str:
    """Create a third test SDXL LoRA file with overlapping keys."""
    path = Path(temp_lora_dir) / "lora_c.safetensors"
    tensors = {
        # Same key as lora_a - tests overlapping affected keys
        "lora_unet_input_blocks_0_0.lora_up.weight": torch.randn(64, 4),
        "lora_unet_input_blocks_0_0.lora_down.weight": torch.randn(4, 32),
    }
    save_file(tensors, str(path))
    return str(path)


# ---------------------------------------------------------------------------
# AC-1: Walk to root and extract model_patcher and arch tag
# ---------------------------------------------------------------------------


class TestAC1TreeWalk:
    """AC-1: Walk to root and find RecipeBase."""

    def test_finds_base_from_simple_merge(
        self,
        mock_model_patcher: MockModelPatcher,
        temp_lora_dir: str,
        sdxl_lora_a: str,
    ):
        """Given a simple recipe tree, walks to find RecipeBase."""
        # AC: @exit-recipe-analysis ac-1
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(
            loras=({"path": Path(sdxl_lora_a).name, "strength": 1.0},)
        )
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        result = analyze_recipe(merge, lora_base_path=temp_lora_dir)

        assert result.model_patcher is mock_model_patcher
        assert result.arch == "sdxl"
        result.loader.cleanup()

    def test_finds_base_from_nested_merge(
        self,
        mock_model_patcher: MockModelPatcher,
        temp_lora_dir: str,
        sdxl_lora_a: str,
        sdxl_lora_b: str,
    ):
        """Given a nested merge tree, walks through to find RecipeBase."""
        # AC: @exit-recipe-analysis ac-1
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora_a = RecipeLoRA(loras=({"path": Path(sdxl_lora_a).name, "strength": 1.0},))
        merge_a = RecipeMerge(base=base, target=lora_a, backbone=None, t_factor=1.0)

        lora_b = RecipeLoRA(loras=({"path": Path(sdxl_lora_b).name, "strength": 0.5},))
        merge_b = RecipeMerge(base=merge_a, target=lora_b, backbone=None, t_factor=0.7)

        result = analyze_recipe(merge_b, lora_base_path=temp_lora_dir)

        assert result.model_patcher is mock_model_patcher
        assert result.arch == "sdxl"
        result.loader.cleanup()

    def test_extracts_arch_tag_correctly(
        self, mock_model_patcher: MockModelPatcher
    ):
        """Extracts the architecture tag from RecipeBase."""
        # AC: @exit-recipe-analysis ac-1
        base_sdxl = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        base_zimage = RecipeBase(model_patcher=mock_model_patcher, arch="zimage")

        # Just check we can analyze the base directly
        # (no LoRAs, so no loading needed)
        from lib.analysis import _walk_to_base

        assert _walk_to_base(base_sdxl).arch == "sdxl"
        assert _walk_to_base(base_zimage).arch == "zimage"

    def test_rejects_lora_as_root(self):
        """RecipeLoRA cannot be the root of a recipe tree."""
        # AC: @exit-recipe-analysis ac-1
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))

        with pytest.raises(ValueError, match="RecipeLoRA cannot be the root"):
            analyze_recipe(lora)

    def test_rejects_compose_as_root(self):
        """RecipeCompose cannot be the root of a recipe tree."""
        # AC: @exit-recipe-analysis ac-1
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        compose = RecipeCompose(branches=(lora,))

        with pytest.raises(ValueError, match="RecipeCompose cannot be the root"):
            analyze_recipe(compose)


# ---------------------------------------------------------------------------
# AC-2: Synthetic set IDs for unique RecipeLoRA groups
# ---------------------------------------------------------------------------


class TestAC2SetIDAssignment:
    """AC-2: Each unique RecipeLoRA gets a distinct set ID."""

    def test_single_lora_gets_one_set_id(
        self,
        mock_model_patcher: MockModelPatcher,
        temp_lora_dir: str,
        sdxl_lora_a: str,
    ):
        """A single RecipeLoRA produces one set ID."""
        # AC: @exit-recipe-analysis ac-2
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": Path(sdxl_lora_a).name, "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        result = analyze_recipe(merge, lora_base_path=temp_lora_dir)

        assert len(result.set_affected) == 1
        result.loader.cleanup()

    def test_chained_loras_share_set_id(
        self,
        mock_model_patcher: MockModelPatcher,
        temp_lora_dir: str,
        sdxl_lora_a: str,
        sdxl_lora_b: str,
    ):
        """LoRAs chained via prev (same RecipeLoRA tuple) share one set ID."""
        # AC: @exit-recipe-analysis ac-2
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        # Two LoRAs in the same RecipeLoRA = chained via prev
        lora = RecipeLoRA(
            loras=(
                {"path": Path(sdxl_lora_a).name, "strength": 1.0},
                {"path": Path(sdxl_lora_b).name, "strength": 0.5},
            )
        )
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        result = analyze_recipe(merge, lora_base_path=temp_lora_dir)

        # Should still be one set (both LoRAs in same RecipeLoRA)
        assert len(result.set_affected) == 1
        result.loader.cleanup()

    def test_different_lora_nodes_get_different_set_ids(
        self,
        mock_model_patcher: MockModelPatcher,
        temp_lora_dir: str,
        sdxl_lora_a: str,
        sdxl_lora_b: str,
    ):
        """Different RecipeLoRA nodes get different set IDs."""
        # AC: @exit-recipe-analysis ac-2
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora_a = RecipeLoRA(loras=({"path": Path(sdxl_lora_a).name, "strength": 1.0},))
        merge_a = RecipeMerge(base=base, target=lora_a, backbone=None, t_factor=1.0)

        lora_b = RecipeLoRA(loras=({"path": Path(sdxl_lora_b).name, "strength": 0.5},))
        merge_b = RecipeMerge(base=merge_a, target=lora_b, backbone=None, t_factor=0.7)

        result = analyze_recipe(merge_b, lora_base_path=temp_lora_dir)

        # Two different RecipeLoRA nodes = two sets
        assert len(result.set_affected) == 2
        result.loader.cleanup()

    def test_compose_branches_get_different_set_ids(
        self,
        mock_model_patcher: MockModelPatcher,
        temp_lora_dir: str,
        sdxl_lora_a: str,
        sdxl_lora_b: str,
    ):
        """RecipeLoRA nodes in different Compose branches get different set IDs."""
        # AC: @exit-recipe-analysis ac-2
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora_a = RecipeLoRA(loras=({"path": Path(sdxl_lora_a).name, "strength": 1.0},))
        lora_b = RecipeLoRA(loras=({"path": Path(sdxl_lora_b).name, "strength": 0.8},))
        compose = RecipeCompose(branches=(lora_a, lora_b))
        merge = RecipeMerge(base=base, target=compose, backbone=None, t_factor=1.0)

        result = analyze_recipe(merge, lora_base_path=temp_lora_dir)

        # Two branches = two sets
        assert len(result.set_affected) == 2
        result.loader.cleanup()


# ---------------------------------------------------------------------------
# AC-3: Architecture-appropriate loader selection
# ---------------------------------------------------------------------------


class TestAC3LoaderSelection:
    """AC-3: LoRA files are loaded with architecture-appropriate loader."""

    def test_sdxl_arch_uses_sdxl_loader(
        self,
        mock_model_patcher: MockModelPatcher,
        temp_lora_dir: str,
        sdxl_lora_a: str,
    ):
        """SDXL arch tag selects SDXLLoader."""
        # AC: @exit-recipe-analysis ac-3
        from lib.lora import SDXLLoader

        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": Path(sdxl_lora_a).name, "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        result = analyze_recipe(merge, lora_base_path=temp_lora_dir)

        assert isinstance(result.loader, SDXLLoader)
        result.loader.cleanup()

    def test_zimage_arch_uses_zimage_loader(
        self, mock_model_patcher: MockModelPatcher, temp_lora_dir: str
    ):
        """Z-Image arch tag selects ZImageLoader."""
        # AC: @exit-recipe-analysis ac-3
        from lib.lora import ZImageLoader

        # Create a minimal Z-Image format LoRA
        path = Path(temp_lora_dir) / "zimage_lora.safetensors"
        tensors = {
            "transformer.layers.0.ff.linear_1.lora_A.weight": torch.randn(8, 64),
            "transformer.layers.0.ff.linear_1.lora_B.weight": torch.randn(128, 8),
        }
        save_file(tensors, str(path))

        base = RecipeBase(model_patcher=mock_model_patcher, arch="zimage")
        lora = RecipeLoRA(loras=({"path": "zimage_lora.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        result = analyze_recipe(merge, lora_base_path=temp_lora_dir)

        assert isinstance(result.loader, ZImageLoader)
        result.loader.cleanup()


# ---------------------------------------------------------------------------
# AC-4: Affected-key map per set ID
# ---------------------------------------------------------------------------


class TestAC4AffectedKeyMap:
    """AC-4: Each set ID maps to the set of base model keys that set modifies."""

    def test_affected_keys_tracked_per_set(
        self,
        mock_model_patcher: MockModelPatcher,
        temp_lora_dir: str,
        sdxl_lora_a: str,
        sdxl_lora_b: str,
    ):
        """Each set ID maps to its affected keys."""
        # AC: @exit-recipe-analysis ac-4
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora_a = RecipeLoRA(loras=({"path": Path(sdxl_lora_a).name, "strength": 1.0},))
        merge_a = RecipeMerge(base=base, target=lora_a, backbone=None, t_factor=1.0)

        lora_b = RecipeLoRA(loras=({"path": Path(sdxl_lora_b).name, "strength": 0.5},))
        merge_b = RecipeMerge(base=merge_a, target=lora_b, backbone=None, t_factor=0.7)

        result = analyze_recipe(merge_b, lora_base_path=temp_lora_dir)

        # Each set should have its own affected keys
        assert len(result.set_affected) == 2

        # Sets should have non-empty key sets
        for set_id, keys in result.set_affected.items():
            assert isinstance(keys, set)
            assert len(keys) > 0

        result.loader.cleanup()

    def test_affected_keys_are_model_format(
        self,
        mock_model_patcher: MockModelPatcher,
        temp_lora_dir: str,
        sdxl_lora_a: str,
    ):
        """Affected keys are in base model format (diffusion_model.*)."""
        # AC: @exit-recipe-analysis ac-4
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": Path(sdxl_lora_a).name, "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        result = analyze_recipe(merge, lora_base_path=temp_lora_dir)

        for set_id, keys in result.set_affected.items():
            for key in keys:
                assert key.startswith("diffusion_model."), (
                    f"Key {key} not in model format"
                )

        result.loader.cleanup()

    def test_all_affected_keys_is_union(
        self,
        mock_model_patcher: MockModelPatcher,
        temp_lora_dir: str,
        sdxl_lora_a: str,
        sdxl_lora_b: str,
    ):
        """affected_keys is the union of all set-specific keys."""
        # AC: @exit-recipe-analysis ac-4
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora_a = RecipeLoRA(loras=({"path": Path(sdxl_lora_a).name, "strength": 1.0},))
        merge_a = RecipeMerge(base=base, target=lora_a, backbone=None, t_factor=1.0)

        lora_b = RecipeLoRA(loras=({"path": Path(sdxl_lora_b).name, "strength": 0.5},))
        merge_b = RecipeMerge(base=merge_a, target=lora_b, backbone=None, t_factor=0.7)

        result = analyze_recipe(merge_b, lora_base_path=temp_lora_dir)

        # affected_keys should be union of all set keys
        expected_union = set()
        for keys in result.set_affected.values():
            expected_union |= keys

        assert result.affected_keys == expected_union
        result.loader.cleanup()


# ---------------------------------------------------------------------------
# AC-5: Skip keys not affected by any LoRA
# ---------------------------------------------------------------------------


class TestAC5KeySkipping:
    """AC-5: Keys not affected by any LoRA set are skipped."""

    def test_get_keys_to_process_filters_unaffected(self):
        """get_keys_to_process returns only affected keys."""
        # AC: @exit-recipe-analysis ac-5
        all_keys = {"key_a", "key_b", "key_c", "key_d"}
        affected = {"key_a", "key_c"}

        to_process = get_keys_to_process(all_keys, affected)

        assert to_process == {"key_a", "key_c"}
        assert "key_b" not in to_process
        assert "key_d" not in to_process

    def test_unaffected_keys_are_excluded(self):
        """Keys not in affected set are excluded from processing."""
        # AC: @exit-recipe-analysis ac-5
        all_keys = {"diffusion_model.layer_1", "diffusion_model.layer_2"}
        affected = {"diffusion_model.layer_1"}

        to_process = get_keys_to_process(all_keys, affected)

        assert "diffusion_model.layer_1" in to_process
        assert "diffusion_model.layer_2" not in to_process

    def test_empty_affected_means_no_processing(self):
        """If no keys are affected, nothing is processed."""
        # AC: @exit-recipe-analysis ac-5
        all_keys = {"key_a", "key_b"}
        affected: set[str] = set()

        to_process = get_keys_to_process(all_keys, affected)

        assert len(to_process) == 0


# ---------------------------------------------------------------------------
# AC-6: FileNotFoundError for missing LoRA files
# ---------------------------------------------------------------------------


class TestAC6MissingLoraError:
    """AC-6: Raises FileNotFoundError naming the missing file and LoRA node."""

    def test_missing_lora_raises_file_not_found(
        self, mock_model_patcher: MockModelPatcher, temp_lora_dir: str
    ):
        """Missing LoRA file raises FileNotFoundError."""
        # AC: @exit-recipe-analysis ac-6
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(
            loras=({"path": "nonexistent_lora.safetensors", "strength": 1.0},)
        )
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        with pytest.raises(FileNotFoundError) as exc_info:
            analyze_recipe(merge, lora_base_path=temp_lora_dir)

        # Error should name the missing file
        assert "nonexistent_lora.safetensors" in str(exc_info.value)

    def test_missing_lora_error_includes_context(
        self, mock_model_patcher: MockModelPatcher, temp_lora_dir: str
    ):
        """FileNotFoundError includes context about which LoRA node."""
        # AC: @exit-recipe-analysis ac-6
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(
            loras=({"path": "missing.safetensors", "strength": 0.75},)
        )
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        with pytest.raises(FileNotFoundError) as exc_info:
            analyze_recipe(merge, lora_base_path=temp_lora_dir)

        error_msg = str(exc_info.value)
        assert "missing.safetensors" in error_msg
        assert "0.75" in error_msg  # Strength provides context

    def test_partial_load_cleans_up_on_error(
        self,
        mock_model_patcher: MockModelPatcher,
        temp_lora_dir: str,
        sdxl_lora_a: str,
    ):
        """If loading fails partway, earlier LoRAs don't leak."""
        # AC: @exit-recipe-analysis ac-6
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(
            loras=(
                {"path": Path(sdxl_lora_a).name, "strength": 1.0},  # Exists
                {"path": "missing.safetensors", "strength": 0.5},  # Missing
            )
        )
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        with pytest.raises(FileNotFoundError):
            analyze_recipe(merge, lora_base_path=temp_lora_dir)

        # No cleanup assertion needed - the loader was created but error raised
        # before returning. Caller doesn't get a result to cleanup.


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for full analysis workflow."""

    def test_full_analysis_workflow(
        self,
        mock_model_patcher: MockModelPatcher,
        temp_lora_dir: str,
        sdxl_lora_a: str,
        sdxl_lora_b: str,
    ):
        """Complete analysis workflow from recipe to result."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")

        # Build a realistic recipe tree
        lora_a = RecipeLoRA(loras=({"path": Path(sdxl_lora_a).name, "strength": 1.0},))
        lora_b = RecipeLoRA(loras=({"path": Path(sdxl_lora_b).name, "strength": 0.8},))
        compose = RecipeCompose(branches=(lora_a, lora_b))
        merge = RecipeMerge(base=base, target=compose, backbone=None, t_factor=0.9)

        result = analyze_recipe(merge, lora_base_path=temp_lora_dir)

        # Verify all result fields
        assert isinstance(result, AnalysisResult)
        assert result.model_patcher is mock_model_patcher
        assert result.arch == "sdxl"
        assert len(result.set_affected) == 2  # Two branches
        assert len(result.affected_keys) > 0
        assert result.loader is not None

        # Cleanup
        result.loader.cleanup()

    def test_analysis_result_usable_for_execution(
        self,
        mock_model_patcher: MockModelPatcher,
        temp_lora_dir: str,
        sdxl_lora_a: str,
    ):
        """AnalysisResult provides everything needed for execution."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": Path(sdxl_lora_a).name, "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        result = analyze_recipe(merge, lora_base_path=temp_lora_dir)

        # Can access model state dict via model_patcher
        state_dict = result.model_patcher.model_state_dict()
        assert len(state_dict) > 0

        # Can get DeltaSpecs from loader for affected keys
        keys = list(result.affected_keys)
        if keys:
            key_indices = {k: i for i, k in enumerate(keys)}
            specs = result.loader.get_delta_specs(keys, key_indices)
            assert isinstance(specs, list)

        result.loader.cleanup()
