"""Tests for ModelLoader -- full checkpoint streaming loader."""

import tempfile

import pytest
import torch
from safetensors.torch import save_file

from lib.model_loader import (
    KeyMismatchError,
    ModelLoader,
    UnsupportedFormatError,
    _detect_architecture_from_keys,
    _normalize_key,
)

# ---------------------------------------------------------------------------
# Fixtures for creating test checkpoint files
# ---------------------------------------------------------------------------


@pytest.fixture
def sdxl_checkpoint_path() -> str:
    """Create a temporary SDXL-format checkpoint file.

    Uses model.diffusion_model prefix as found in real SDXL checkpoints.
    """
    tensors = {
        # Diffusion model keys with model.diffusion_model prefix
        "model.diffusion_model.input_blocks.0.0.weight": torch.randn(4, 4),
        "model.diffusion_model.input_blocks.1.0.weight": torch.randn(4, 4),
        "model.diffusion_model.middle_block.0.weight": torch.randn(4, 4),
        "model.diffusion_model.output_blocks.0.0.weight": torch.randn(4, 4),
        # VAE keys (should be excluded)
        "model.first_stage_model.encoder.weight": torch.randn(4, 4),
        # Text encoder keys (should be excluded)
        "model.conditioner.embedders.weight": torch.randn(4, 4),
    }
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def zimage_checkpoint_path() -> str:
    """Create a temporary Z-Image-format checkpoint file.

    Uses diffusion_model prefix with layers and noise_refiner structure.
    """
    tensors = {
        # Diffusion model keys with diffusion_model prefix (Z-Image style)
        "diffusion_model.layers.0.attention.qkv.weight": torch.randn(4, 4),
        "diffusion_model.layers.10.attention.qkv.weight": torch.randn(4, 4),
        "diffusion_model.noise_refiner.0.attn.weight": torch.randn(4, 4),
        "diffusion_model.context_refiner.0.attn.weight": torch.randn(4, 4),
        # VAE keys (should be excluded)
        "first_stage_model.decoder.weight": torch.randn(4, 4),
    }
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def transformer_prefix_checkpoint_path() -> str:
    """Create a checkpoint with transformer prefix (alternate Z-Image format)."""
    tensors = {
        "transformer.layers.0.attention.qkv.weight": torch.randn(4, 4),
        "transformer.layers.1.attention.out.weight": torch.randn(4, 4),
        "transformer.noise_refiner.0.attn.weight": torch.randn(4, 4),
    }
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def non_safetensors_path() -> str:
    """Create a temporary non-safetensors file."""
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        f.write(b"dummy data")
        return f.name


# ---------------------------------------------------------------------------
# AC-1: safe_open() memory-mapped access
# ---------------------------------------------------------------------------


# AC: @full-model-loader ac-1
class TestSafeOpenAccess:
    """Tests for memory-mapped file access via safe_open()."""

    def test_loader_opens_file_without_loading_full_tensors(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """Loader opens file and has keys without loading all tensor data."""
        loader = ModelLoader(sdxl_checkpoint_path)

        # Should have keys available
        assert len(loader.affected_keys) > 0

        # Keys should be in base model format (diffusion_model. prefix)
        for key in loader.affected_keys:
            assert key.startswith("diffusion_model.")

        loader.cleanup()

    def test_loader_works_as_context_manager(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """Loader supports context manager for automatic cleanup."""
        with ModelLoader(sdxl_checkpoint_path) as loader:
            assert len(loader.affected_keys) > 0


# ---------------------------------------------------------------------------
# AC-2: get_weights() returns correctly mapped tensors
# ---------------------------------------------------------------------------


# AC: @full-model-loader ac-2
class TestGetWeights:
    """Tests for get_weights() tensor retrieval."""

    def test_get_weights_returns_tensors_for_requested_keys(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """get_weights() returns tensors in order of requested keys."""
        with ModelLoader(sdxl_checkpoint_path) as loader:
            keys = list(loader.affected_keys)[:2]
            tensors = loader.get_weights(keys)

            assert len(tensors) == 2
            for t in tensors:
                assert isinstance(t, torch.Tensor)
                assert t.shape == (4, 4)

    def test_get_weights_maps_file_keys_to_base_model_format(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """File keys with model.diffusion_model prefix map to diffusion_model format."""
        with ModelLoader(sdxl_checkpoint_path) as loader:
            # The file has model.diffusion_model.input_blocks.0.0.weight
            # The loader should expose it as diffusion_model.input_blocks.0.0.weight
            expected_key = "diffusion_model.input_blocks.0.0.weight"
            assert expected_key in loader.affected_keys

            tensors = loader.get_weights([expected_key])
            assert len(tensors) == 1


# ---------------------------------------------------------------------------
# AC-3: SDXL key normalization
# ---------------------------------------------------------------------------


# AC: @full-model-loader ac-3
class TestSDXLKeyNormalization:
    """Tests for SDXL checkpoint key normalization."""

    def test_normalize_key_strips_model_diffusion_model_prefix(self) -> None:
        """model.diffusion_model.X normalizes to diffusion_model.X."""
        file_key = "model.diffusion_model.input_blocks.0.0.weight"
        normalized = _normalize_key(file_key)
        assert normalized == "diffusion_model.input_blocks.0.0.weight"

    def test_normalize_key_preserves_diffusion_model_prefix(self) -> None:
        """Keys already in diffusion_model format are preserved."""
        file_key = "diffusion_model.input_blocks.0.0.weight"
        normalized = _normalize_key(file_key)
        assert normalized == "diffusion_model.input_blocks.0.0.weight"

    def test_sdxl_checkpoint_keys_normalized_correctly(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """SDXL checkpoint keys are normalized to base model format."""
        with ModelLoader(sdxl_checkpoint_path) as loader:
            # Check all keys have correct format
            for key in loader.affected_keys:
                assert key.startswith("diffusion_model.")
                assert not key.startswith("model.diffusion_model.")


# ---------------------------------------------------------------------------
# AC-4: Z-Image key normalization
# ---------------------------------------------------------------------------


# AC: @full-model-loader ac-4
class TestZImageKeyNormalization:
    """Tests for Z-Image checkpoint key normalization."""

    def test_normalize_key_handles_diffusion_model_prefix(self) -> None:
        """diffusion_model.X keys are preserved."""
        file_key = "diffusion_model.layers.0.attention.qkv.weight"
        normalized = _normalize_key(file_key)
        assert normalized == "diffusion_model.layers.0.attention.qkv.weight"

    def test_normalize_key_handles_transformer_prefix(self) -> None:
        """transformer.X normalizes to diffusion_model.X."""
        file_key = "transformer.layers.0.attention.qkv.weight"
        normalized = _normalize_key(file_key)
        assert normalized == "diffusion_model.layers.0.attention.qkv.weight"

    def test_zimage_checkpoint_keys_normalized_correctly(
        self, zimage_checkpoint_path: str
    ) -> None:
        """Z-Image checkpoint keys are normalized to base model format."""
        with ModelLoader(zimage_checkpoint_path) as loader:
            for key in loader.affected_keys:
                assert key.startswith("diffusion_model.")

    def test_transformer_prefix_checkpoint_normalized(
        self, transformer_prefix_checkpoint_path: str
    ) -> None:
        """Checkpoint with transformer prefix normalizes to diffusion_model."""
        with ModelLoader(transformer_prefix_checkpoint_path) as loader:
            for key in loader.affected_keys:
                assert key.startswith("diffusion_model.")
                assert "transformer." not in key


# ---------------------------------------------------------------------------
# AC-5: affected_keys excludes VAE and text encoder
# ---------------------------------------------------------------------------


# AC: @full-model-loader ac-5
class TestAffectedKeysFiltering:
    """Tests for affected_keys property filtering."""

    def test_affected_keys_excludes_vae(self, sdxl_checkpoint_path: str) -> None:
        """VAE keys (first_stage_model) are excluded."""
        with ModelLoader(sdxl_checkpoint_path) as loader:
            for key in loader.affected_keys:
                assert "first_stage_model" not in key

    def test_affected_keys_excludes_text_encoder(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """Text encoder keys (conditioner) are excluded."""
        with ModelLoader(sdxl_checkpoint_path) as loader:
            for key in loader.affected_keys:
                assert "conditioner" not in key

    def test_normalize_key_returns_none_for_vae(self) -> None:
        """_normalize_key returns None for VAE keys."""
        assert _normalize_key("first_stage_model.encoder.weight") is None
        assert _normalize_key("model.first_stage_model.encoder.weight") is None

    def test_normalize_key_returns_none_for_text_encoder(self) -> None:
        """_normalize_key returns None for text encoder keys."""
        assert _normalize_key("conditioner.embedders.weight") is None
        assert _normalize_key("cond_stage_model.transformer.weight") is None

    def test_affected_keys_returns_frozenset(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """affected_keys returns frozenset to prevent mutation."""
        with ModelLoader(sdxl_checkpoint_path) as loader:
            assert isinstance(loader.affected_keys, frozenset)


# ---------------------------------------------------------------------------
# AC-6: cleanup() closes file handle
# ---------------------------------------------------------------------------


# AC: @full-model-loader ac-6
class TestCleanup:
    """Tests for cleanup() resource management."""

    def test_cleanup_releases_handle(self, sdxl_checkpoint_path: str) -> None:
        """cleanup() releases the file handle."""
        loader = ModelLoader(sdxl_checkpoint_path)
        assert loader._handle is not None

        loader.cleanup()
        assert loader._handle is None

    def test_context_manager_calls_cleanup(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """Context manager calls cleanup() on exit."""
        with ModelLoader(sdxl_checkpoint_path) as loader:
            assert loader._handle is not None

        assert loader._handle is None


# ---------------------------------------------------------------------------
# AC-7: KeyMismatchError for unmatched keys
# ---------------------------------------------------------------------------


# AC: @full-model-loader ac-7
class TestKeyMismatchError:
    """Tests for clear error on unmatched keys."""

    def test_get_weights_raises_for_missing_keys(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """get_weights() raises KeyMismatchError for missing keys."""
        with ModelLoader(sdxl_checkpoint_path) as loader:
            with pytest.raises(KeyMismatchError) as exc_info:
                loader.get_weights(["diffusion_model.nonexistent.weight"])

            assert "missing" in str(exc_info.value).lower()
            assert "nonexistent" in str(exc_info.value)

    def test_key_mismatch_error_lists_unmatched_keys(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """KeyMismatchError message lists the unmatched keys."""
        with ModelLoader(sdxl_checkpoint_path) as loader:
            missing = [
                "diffusion_model.missing1.weight",
                "diffusion_model.missing2.weight",
            ]
            with pytest.raises(KeyMismatchError) as exc_info:
                loader.get_weights(missing)

            error_msg = str(exc_info.value)
            assert "missing1" in error_msg
            assert "missing2" in error_msg


# ---------------------------------------------------------------------------
# AC-8: Architecture detection from normalized keys
# ---------------------------------------------------------------------------


# AC: @full-model-loader ac-8
class TestArchitectureDetection:
    """Tests for architecture detection from file keys."""

    def test_detect_sdxl_architecture(self, sdxl_checkpoint_path: str) -> None:
        """SDXL checkpoint detected from input/middle/output blocks pattern."""
        with ModelLoader(sdxl_checkpoint_path) as loader:
            assert loader.arch == "sdxl"

    def test_detect_zimage_architecture(
        self, zimage_checkpoint_path: str
    ) -> None:
        """Z-Image checkpoint detected from layers + noise_refiner pattern."""
        with ModelLoader(zimage_checkpoint_path) as loader:
            assert loader.arch == "zimage"

    def test_detect_architecture_without_loading_tensors(self) -> None:
        """Architecture detection uses only key inspection, no tensor loading."""
        # Test the detection function directly with just keys
        sdxl_keys = frozenset({
            "diffusion_model.input_blocks.0.0.weight",
            "diffusion_model.middle_block.0.weight",
            "diffusion_model.output_blocks.0.0.weight",
        })
        assert _detect_architecture_from_keys(sdxl_keys) == "sdxl"

        zimage_keys = frozenset({
            "diffusion_model.layers.0.attention.qkv.weight",
            "diffusion_model.noise_refiner.0.attn.weight",
        })
        assert _detect_architecture_from_keys(zimage_keys) == "zimage"

    def test_unknown_architecture_returns_none(self) -> None:
        """Unknown architecture patterns return None."""
        unknown_keys = frozenset({
            "diffusion_model.some.unknown.structure.weight",
        })
        assert _detect_architecture_from_keys(unknown_keys) is None


# ---------------------------------------------------------------------------
# AC-9: Non-safetensors format error
# ---------------------------------------------------------------------------


# AC: @full-model-loader ac-9
class TestUnsupportedFormatError:
    """Tests for non-safetensors format rejection."""

    def test_ckpt_file_raises_unsupported_error(
        self, non_safetensors_path: str
    ) -> None:
        """Opening a .ckpt file raises UnsupportedFormatError."""
        with pytest.raises(UnsupportedFormatError) as exc_info:
            ModelLoader(non_safetensors_path)

        error_msg = str(exc_info.value)
        assert "safetensors" in error_msg.lower()
        assert ".ckpt" in error_msg

    def test_pt_file_raises_unsupported_error(self) -> None:
        """Opening a .pt file raises UnsupportedFormatError."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"dummy")
            pt_path = f.name

        with pytest.raises(UnsupportedFormatError) as exc_info:
            ModelLoader(pt_path)

        assert "safetensors" in str(exc_info.value).lower()

    def test_error_message_suggests_conversion(
        self, non_safetensors_path: str
    ) -> None:
        """Error message suggests converting to safetensors."""
        with pytest.raises(UnsupportedFormatError) as exc_info:
            ModelLoader(non_safetensors_path)

        assert "convert" in str(exc_info.value).lower()
