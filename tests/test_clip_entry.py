"""Tests for WIDEN CLIP Entry Node — architecture detection and RecipeBase creation."""

import pytest
import torch

from lib.recipe import RecipeBase
from nodes.clip_entry import (
    UnsupportedCLIPArchitectureError,
    WIDENCLIPEntryNode,
    detect_clip_architecture,
)

# Representative SDXL CLIP state dict keys (clip_l + clip_g encoders)
_SDXL_CLIP_KEYS = (
    "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
    "clip_l.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight",
    "clip_l.transformer.text_model.encoder.layers.11.mlp.fc1.weight",
    "clip_l.transformer.text_model.embeddings.token_embedding.weight",
    "clip_g.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
    "clip_g.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight",
    "clip_g.transformer.text_model.encoder.layers.31.mlp.fc1.weight",
    "clip_g.transformer.text_model.embeddings.position_embedding.weight",
)

# Non-SDXL CLIP keys (only clip_l, like SD1.5)
_SD15_CLIP_KEYS = (
    "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
    "clip_l.transformer.text_model.encoder.layers.11.mlp.fc1.weight",
    "clip_l.transformer.text_model.embeddings.token_embedding.weight",
)


class MockCLIPPatcher:
    """Minimal mock replicating the CLIP patcher API surface used by CLIP Entry node."""

    def __init__(self, keys: tuple[str, ...] = _SDXL_CLIP_KEYS):
        self._state_dict: dict[str, torch.Tensor] = {
            k: torch.randn(4, 4, dtype=torch.float32) for k in keys
        }

    def model_state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._state_dict)


class MockCLIP:
    """Minimal mock replicating ComfyUI CLIP object API surface."""

    def __init__(self, keys: tuple[str, ...] = _SDXL_CLIP_KEYS):
        self.patcher = MockCLIPPatcher(keys)


# --- AC-1: Returns RecipeBase wrapping CLIP with arch and domain="clip" ---


class TestCLIPEntryNodeReturnsRecipeBase:
    """AC: @clip-entry-node ac-1 — returns RecipeBase wrapping CLIP with arch and domain='clip'."""

    def test_entry_returns_tuple_with_recipe_base(self):
        """Entry node returns a tuple containing a RecipeBase."""
        clip = MockCLIP()
        node = WIDENCLIPEntryNode()
        result = node.entry(clip)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], RecipeBase)

    def test_recipe_base_wraps_same_clip(self):
        """RecipeBase contains the exact same CLIP reference (no clone)."""
        clip = MockCLIP()
        node = WIDENCLIPEntryNode()
        (recipe,) = node.entry(clip)

        assert recipe.model_patcher is clip

    def test_recipe_base_has_domain_clip(self):
        """RecipeBase has domain='clip' set."""
        clip = MockCLIP()
        node = WIDENCLIPEntryNode()
        (recipe,) = node.entry(clip)

        assert recipe.domain == "clip"


# --- AC-2: SDXL detection ---


class TestSDXLCLIPArchitectureDetection:
    """AC: @clip-entry-node ac-2 — SDXL detection via clip_l and clip_g keys."""

    def test_sdxl_clip_detected_as_sdxl(self):
        """CLIP with clip_l and clip_g keys returns arch='sdxl'."""
        clip = MockCLIP(keys=_SDXL_CLIP_KEYS)
        arch = detect_clip_architecture(clip)

        assert arch == "sdxl"

    def test_entry_node_sets_sdxl_arch(self):
        """Entry node sets arch field to 'sdxl' for SDXL CLIP."""
        clip = MockCLIP(keys=_SDXL_CLIP_KEYS)
        node = WIDENCLIPEntryNode()
        (recipe,) = node.entry(clip)

        assert recipe.arch == "sdxl"


# --- AC-3: No GPU memory allocated, no tensor copies ---


class TestNoGPUMemoryAllocation:
    """AC: @clip-entry-node ac-3 — no GPU memory allocated and no tensors copied."""

    def test_clip_not_cloned(self):
        """Entry node stores reference, not a clone."""
        clip = MockCLIP()
        node = WIDENCLIPEntryNode()
        (recipe,) = node.entry(clip)

        # Same object, not cloned
        assert recipe.model_patcher is clip

    def test_state_dict_tensors_not_copied(self):
        """State dict tensors share memory (no copy)."""
        clip = MockCLIP()
        original_state = clip.patcher.model_state_dict()
        node = WIDENCLIPEntryNode()
        (recipe,) = node.entry(clip)

        # Get state dict through the recipe's patcher reference
        new_state = recipe.model_patcher.patcher.model_state_dict()

        # Tensors should be the same objects (not copies)
        for key in original_state:
            assert original_state[key] is new_state[key]

    def test_no_cuda_tensors_created(self):
        """No CUDA tensors are created during entry node execution."""
        clip = MockCLIP()
        node = WIDENCLIPEntryNode()
        (recipe,) = node.entry(clip)

        # Check no tensors in recipe moved to CUDA
        state = recipe.model_patcher.patcher.model_state_dict()
        for tensor in state.values():
            assert not tensor.is_cuda


# --- AC-4: Returns WIDEN_CLIP type ---


class TestReturnTypes:
    """AC: @clip-entry-node ac-4 — returns WIDEN_CLIP type."""

    def test_return_types_is_widen_clip(self):
        """RETURN_TYPES is WIDEN_CLIP tuple."""
        assert WIDENCLIPEntryNode.RETURN_TYPES == ("WIDEN_CLIP",)
        assert WIDENCLIPEntryNode.RETURN_NAMES == ("widen_clip",)


# --- AC-5: Non-SDXL raises clear error ---


class TestNonSDXLError:
    """AC: @clip-entry-node ac-5 — non-SDXL raises clear error."""

    def test_sd15_clip_raises_error(self):
        """SD1.5 CLIP (only clip_l) raises UnsupportedCLIPArchitectureError."""
        clip = MockCLIP(keys=_SD15_CLIP_KEYS)

        with pytest.raises(UnsupportedCLIPArchitectureError) as exc_info:
            detect_clip_architecture(clip)

        error_msg = str(exc_info.value)
        assert "Only SDXL CLIP merging is supported in v1" in error_msg

    def test_unknown_clip_keys_raise_error(self):
        """CLIP with unknown keys raises UnsupportedCLIPArchitectureError."""
        unknown_keys = (
            "some.random.encoder.weight",
            "another.unknown.bias",
        )
        clip = MockCLIP(keys=unknown_keys)

        with pytest.raises(UnsupportedCLIPArchitectureError) as exc_info:
            detect_clip_architecture(clip)

        error_msg = str(exc_info.value)
        assert "Only SDXL CLIP merging is supported in v1" in error_msg


# --- AC-6: Accepts CLIP input type ---


class TestInputTypes:
    """AC: @clip-entry-node ac-6 — accepts CLIP input type."""

    def test_input_types_accepts_clip(self):
        """INPUT_TYPES returns correct structure with CLIP input."""
        input_types = WIDENCLIPEntryNode.INPUT_TYPES()

        assert "required" in input_types
        assert "clip" in input_types["required"]
        assert input_types["required"]["clip"] == ("CLIP",)


# --- AC-7: Keys accessed via patcher API without GPU load ---


class TestPatcherAPIAccess:
    """AC: @clip-entry-node ac-7 — keys accessed via patcher API without GPU load."""

    def test_architecture_detection_uses_patcher_api(self):
        """Architecture detection accesses keys via clip.patcher.model_state_dict()."""
        # Create a mock that tracks method calls
        patcher_called = []

        class TrackedPatcher(MockCLIPPatcher):
            def model_state_dict(self) -> dict[str, torch.Tensor]:
                patcher_called.append(True)
                return super().model_state_dict()

        class TrackedCLIP:
            def __init__(self):
                self.patcher = TrackedPatcher()

        clip = TrackedCLIP()
        detect_clip_architecture(clip)

        # Verify patcher.model_state_dict() was called
        assert len(patcher_called) == 1


# --- Node metadata tests ---


class TestCLIPEntryNodeMetadata:
    """Test ComfyUI node metadata is correct."""

    def test_category(self):
        """CATEGORY is ecaj/merge."""
        assert WIDENCLIPEntryNode.CATEGORY == "ecaj/merge"

    def test_function_name(self):
        """FUNCTION points to entry method."""
        assert WIDENCLIPEntryNode.FUNCTION == "entry"
