"""Tests for WIDEN Entry Node — architecture detection and RecipeBase creation."""

import pytest

from lib.recipe import RecipeBase
from nodes.entry import (
    UnsupportedArchitectureError,
    WIDENEntryNode,
    detect_architecture,
)
from tests.conftest import MockModelPatcher

# --- AC-1: Returns RecipeBase wrapping ModelPatcher ---


class TestEntryNodeReturnsRecipeBase:
    """AC: @entry-node ac-1 — returns RecipeBase wrapping the ModelPatcher reference."""

    def test_entry_returns_tuple_with_recipe_base(self):
        """Entry node returns a tuple containing a RecipeBase."""
        patcher = MockModelPatcher()
        node = WIDENEntryNode()
        result = node.entry(patcher)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], RecipeBase)

    def test_recipe_base_wraps_same_model_patcher(self):
        """RecipeBase contains the exact same ModelPatcher reference (no clone)."""
        patcher = MockModelPatcher()
        node = WIDENEntryNode()
        (recipe,) = node.entry(patcher)

        assert recipe.model_patcher is patcher


# --- AC-2: SDXL detection ---


class TestSDXLArchitectureDetection:
    """AC: @entry-node ac-2 — SDXL detection via input_blocks/middle_block/output_blocks."""

    def test_sdxl_keys_detected_as_sdxl(self):
        """Model with SDXL-style keys returns arch='sdxl'."""
        keys = (
            "diffusion_model.input_blocks.0.0.weight",
            "diffusion_model.middle_block.0.weight",
            "diffusion_model.output_blocks.0.0.weight",
        )
        patcher = MockModelPatcher(keys=keys)
        arch = detect_architecture(patcher)

        assert arch == "sdxl"

    def test_entry_node_sets_sdxl_arch(self):
        """Entry node sets arch field to 'sdxl' for SDXL model."""
        patcher = MockModelPatcher()  # Default keys are SDXL-like
        node = WIDENEntryNode()
        (recipe,) = node.entry(patcher)

        assert recipe.arch == "sdxl"


# --- AC-3: Z-Image detection ---


class TestZImageArchitectureDetection:
    """AC: @entry-node ac-3 — Z-Image detection via layers + noise_refiner."""

    def test_zimage_keys_detected_as_zimage(self):
        """Model with Z-Image keys (layers + noise_refiner) returns arch='zimage'."""
        keys = (
            "diffusion_model.layers.0.weight",
            "diffusion_model.layers.1.weight",
            "noise_refiner.weight",
        )
        patcher = MockModelPatcher(keys=keys)
        arch = detect_architecture(patcher)

        assert arch == "zimage"

    def test_layers_without_noise_refiner_is_not_zimage(self):
        """Model with just layers but no noise_refiner is not detected as Z-Image."""
        keys = (
            "diffusion_model.layers.0.weight",
            "diffusion_model.layers.1.weight",
        )
        patcher = MockModelPatcher(keys=keys)

        with pytest.raises(UnsupportedArchitectureError):
            detect_architecture(patcher)


# --- AC-4: No GPU memory allocated, no tensors copied ---


class TestNoGPUMemoryAllocation:
    """AC: @entry-node ac-4 — no GPU memory allocated and no tensors copied."""

    def test_model_patcher_not_cloned(self):
        """Entry node stores reference, not a clone."""
        patcher = MockModelPatcher()
        original_uuid = patcher.patches_uuid
        node = WIDENEntryNode()
        (recipe,) = node.entry(patcher)

        # Same object, not cloned
        assert recipe.model_patcher is patcher
        assert patcher.patches_uuid == original_uuid

    def test_state_dict_tensors_not_copied(self):
        """State dict tensors share memory (no copy)."""
        patcher = MockModelPatcher()
        original_state = patcher.model_state_dict()
        node = WIDENEntryNode()
        (recipe,) = node.entry(patcher)

        # Get state dict through the recipe's patcher reference
        new_state = recipe.model_patcher.model_state_dict()  # type: ignore[attr-defined]

        # Tensors should be the same objects (not copies)
        for key in original_state:
            assert original_state[key] is new_state[key]

    def test_no_cuda_tensors_created(self):
        """No CUDA tensors are created during entry node execution."""
        patcher = MockModelPatcher()
        node = WIDENEntryNode()
        (recipe,) = node.entry(patcher)

        # Check no tensors in recipe moved to CUDA
        state = recipe.model_patcher.model_state_dict()  # type: ignore[attr-defined]
        for tensor in state.values():
            assert not tensor.is_cuda


# --- AC-5: Unsupported architecture error ---


class TestUnsupportedArchitectureError:
    """AC: @entry-node ac-5 — clear error listing supported architectures."""

    def test_unknown_keys_raise_error(self):
        """Model with unknown key patterns raises UnsupportedArchitectureError."""
        keys = (
            "some.random.key.weight",
            "another.unknown.bias",
        )
        patcher = MockModelPatcher(keys=keys)

        with pytest.raises(UnsupportedArchitectureError) as exc_info:
            detect_architecture(patcher)

        error_msg = str(exc_info.value)
        assert "Could not detect model architecture" in error_msg
        assert "sdxl" in error_msg.lower()
        assert "zimage" in error_msg.lower()

    def test_error_includes_key_prefixes(self):
        """Error message includes first key prefixes for debugging."""
        keys = (
            "custom_model.layer1.weight",
            "custom_model.layer2.weight",
        )
        patcher = MockModelPatcher(keys=keys)

        with pytest.raises(UnsupportedArchitectureError) as exc_info:
            detect_architecture(patcher)

        error_msg = str(exc_info.value)
        assert "custom_model" in error_msg

    def test_flux_detected_but_unsupported(self):
        """Flux architecture is detected but raises clear 'not supported' error."""
        keys = (
            "double_blocks.0.weight",
            "double_blocks.1.weight",
        )
        patcher = MockModelPatcher(keys=keys)

        with pytest.raises(UnsupportedArchitectureError) as exc_info:
            detect_architecture(patcher)

        error_msg = str(exc_info.value)
        assert "flux" in error_msg.lower()
        assert "no WIDEN loader is available yet" in error_msg

    def test_qwen_detected_but_unsupported(self):
        """Qwen architecture (60+ transformer_blocks) is detected but unsupported."""
        # Need 60+ keys with transformer_blocks
        keys = tuple(f"transformer_blocks.{i}.weight" for i in range(65))
        patcher = MockModelPatcher(keys=keys)

        with pytest.raises(UnsupportedArchitectureError) as exc_info:
            detect_architecture(patcher)

        error_msg = str(exc_info.value)
        assert "qwen" in error_msg.lower()
        assert "no WIDEN loader is available yet" in error_msg


# --- Node metadata tests ---


class TestEntryNodeMetadata:
    """Test ComfyUI node metadata is correct."""

    def test_input_types(self):
        """INPUT_TYPES returns correct structure."""
        input_types = WIDENEntryNode.INPUT_TYPES()

        assert "required" in input_types
        assert "model" in input_types["required"]
        assert input_types["required"]["model"] == ("MODEL",)

    def test_return_types(self):
        """RETURN_TYPES is WIDEN tuple."""
        assert WIDENEntryNode.RETURN_TYPES == ("WIDEN",)
        assert WIDENEntryNode.RETURN_NAMES == ("widen",)

    def test_category(self):
        """CATEGORY is ecaj/merge."""
        assert WIDENEntryNode.CATEGORY == "ecaj/merge"

    def test_function_name(self):
        """FUNCTION points to entry method."""
        assert WIDENEntryNode.FUNCTION == "entry"
