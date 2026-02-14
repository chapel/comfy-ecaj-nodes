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


# ---------------------------------------------------------------------------
# Qwen Architecture Support
# ---------------------------------------------------------------------------


@pytest.fixture
def qwen_checkpoint_path() -> str:
    """Create a temporary Qwen-format checkpoint file.

    Uses transformer_blocks structure with 60+ blocks as required for Qwen detection.
    """
    # Build enough transformer_blocks keys to trigger Qwen detection (≥60)
    tensors = {}
    for i in range(65):
        tensors[f"transformer.transformer_blocks.{i}.attn.to_q.weight"] = torch.randn(4, 4)
        tensors[f"transformer.transformer_blocks.{i}.attn.to_k.weight"] = torch.randn(4, 4)
        tensors[f"transformer.transformer_blocks.{i}.attn.to_v.weight"] = torch.randn(4, 4)
        tensors[f"transformer.transformer_blocks.{i}.mlp.gate_proj.weight"] = torch.randn(4, 4)
    # VAE keys (should be excluded)
    tensors["first_stage_model.encoder.weight"] = torch.randn(4, 4)

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def qwen_model_prefix_checkpoint_path() -> str:
    """Create a Qwen checkpoint with model.transformer prefix."""
    tensors = {}
    for i in range(65):
        tensors[f"model.transformer.transformer_blocks.{i}.attn.weight"] = torch.randn(4, 4)

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file(tensors, f.name)
        return f.name


# AC: @qwen-model-loader ac-7
class TestQwenArchitectureDetection:
    """Tests for Qwen architecture detection from checkpoint keys."""

    def test_detect_qwen_architecture(self, qwen_checkpoint_path: str) -> None:
        """Qwen checkpoint detected from transformer_blocks count ≥60."""
        with ModelLoader(qwen_checkpoint_path) as loader:
            assert loader.arch == "qwen"

    def test_detect_qwen_with_model_prefix(
        self, qwen_model_prefix_checkpoint_path: str
    ) -> None:
        """Qwen checkpoint with model.transformer prefix detected."""
        with ModelLoader(qwen_model_prefix_checkpoint_path) as loader:
            assert loader.arch == "qwen"

    def test_detect_qwen_architecture_without_loading_tensors(self) -> None:
        """Qwen detection uses only key inspection, no tensor loading."""
        # Test with exactly 60 transformer_blocks keys (threshold)
        qwen_keys = frozenset(
            f"diffusion_model.transformer_blocks.{i}.weight" for i in range(60)
        )
        assert _detect_architecture_from_keys(qwen_keys) == "qwen"

    def test_below_threshold_not_detected_as_qwen(self) -> None:
        """Less than 60 transformer_blocks keys does not trigger Qwen detection."""
        # 59 keys - just below threshold
        keys = frozenset(
            f"diffusion_model.transformer_blocks.{i}.weight" for i in range(59)
        )
        assert _detect_architecture_from_keys(keys) != "qwen"


class TestQwenKeyNormalization:
    """Tests for Qwen checkpoint key normalization."""

    def test_normalize_key_handles_transformer_prefix(self) -> None:
        """transformer.transformer_blocks.X normalizes to diffusion_model.transformer_blocks.X."""
        file_key = "transformer.transformer_blocks.0.attn.to_q.weight"
        normalized = _normalize_key(file_key)
        assert normalized == "diffusion_model.transformer_blocks.0.attn.to_q.weight"

    def test_normalize_key_handles_model_transformer_prefix(self) -> None:
        """model.transformer.transformer_blocks.X normalizes correctly."""
        file_key = "model.transformer.transformer_blocks.5.attn.weight"
        normalized = _normalize_key(file_key)
        assert normalized == "diffusion_model.transformer_blocks.5.attn.weight"

    def test_qwen_checkpoint_keys_normalized_correctly(
        self, qwen_checkpoint_path: str
    ) -> None:
        """Qwen checkpoint keys are normalized to base model format."""
        with ModelLoader(qwen_checkpoint_path) as loader:
            for key in loader.affected_keys:
                assert key.startswith("diffusion_model.")
                assert "transformer_blocks" in key

    def test_qwen_checkpoint_excludes_vae(self, qwen_checkpoint_path: str) -> None:
        """Qwen checkpoint VAE keys are excluded."""
        with ModelLoader(qwen_checkpoint_path) as loader:
            for key in loader.affected_keys:
                assert "first_stage_model" not in key

    def test_qwen_keys_retrievable(self, qwen_checkpoint_path: str) -> None:
        """Keys can be retrieved from Qwen checkpoint after normalization."""
        with ModelLoader(qwen_checkpoint_path) as loader:
            # Get first few keys and retrieve tensors
            keys = list(loader.affected_keys)[:3]
            tensors = loader.get_weights(keys)
            assert len(tensors) == 3
            for t in tensors:
                assert isinstance(t, torch.Tensor)


# ---------------------------------------------------------------------------
# Flux Klein Architecture Support
# ---------------------------------------------------------------------------


@pytest.fixture
def flux_checkpoint_path() -> str:
    """Create a temporary Flux Klein-format checkpoint file.

    Uses double_blocks and single_blocks structure (Klein 9B: 8 double + 24 single).
    """
    tensors = {}
    # Double blocks (8 for Klein 9B)
    for i in range(8):
        tensors[f"diffusion_model.double_blocks.{i}.img_attn.qkv.weight"] = torch.randn(4, 4)
        tensors[f"diffusion_model.double_blocks.{i}.txt_attn.qkv.weight"] = torch.randn(4, 4)
        tensors[f"diffusion_model.double_blocks.{i}.img_mlp.0.weight"] = torch.randn(4, 4)
        tensors[f"diffusion_model.double_blocks.{i}.txt_mlp.0.weight"] = torch.randn(4, 4)
    # Single blocks (24 for Klein 9B)
    for i in range(24):
        tensors[f"diffusion_model.single_blocks.{i}.linear1.weight"] = torch.randn(4, 4)
        tensors[f"diffusion_model.single_blocks.{i}.linear2.weight"] = torch.randn(4, 4)
    # Non-block keys
    tensors["diffusion_model.final_layer.linear.weight"] = torch.randn(4, 4)
    tensors["diffusion_model.img_in.weight"] = torch.randn(4, 4)
    # VAE keys (should be excluded)
    tensors["first_stage_model.encoder.weight"] = torch.randn(4, 4)

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def flux_transformer_prefix_checkpoint_path() -> str:
    """Create a Flux checkpoint with transformer prefix."""
    tensors = {}
    for i in range(5):
        tensors[f"transformer.double_blocks.{i}.img_attn.qkv.weight"] = torch.randn(4, 4)
    for i in range(20):
        tensors[f"transformer.single_blocks.{i}.linear1.weight"] = torch.randn(4, 4)

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file(tensors, f.name)
        return f.name


# AC: @flux-model-loader ac-8
class TestFluxArchitectureDetection:
    """Tests for Flux Klein architecture detection from checkpoint keys."""

    def test_detect_flux_architecture(self, flux_checkpoint_path: str) -> None:
        """Flux checkpoint detected from double_blocks pattern."""
        with ModelLoader(flux_checkpoint_path) as loader:
            assert loader.arch == "flux"

    def test_detect_flux_with_transformer_prefix(
        self, flux_transformer_prefix_checkpoint_path: str
    ) -> None:
        """Flux checkpoint with transformer prefix detected."""
        with ModelLoader(flux_transformer_prefix_checkpoint_path) as loader:
            assert loader.arch == "flux"

    def test_detect_flux_architecture_without_loading_tensors(self) -> None:
        """Flux detection uses only key inspection, no tensor loading."""
        # Test with double_blocks keys
        flux_keys = frozenset({
            "diffusion_model.double_blocks.0.img_attn.qkv.weight",
            "diffusion_model.double_blocks.1.txt_attn.qkv.weight",
            "diffusion_model.single_blocks.0.linear1.weight",
        })
        assert _detect_architecture_from_keys(flux_keys) == "flux"

    def test_flux_not_detected_without_double_blocks(self) -> None:
        """Keys without double_blocks do not trigger Flux detection."""
        # Only single_blocks should not trigger Flux
        keys = frozenset({
            "diffusion_model.single_blocks.0.linear1.weight",
            "diffusion_model.single_blocks.1.linear1.weight",
        })
        assert _detect_architecture_from_keys(keys) != "flux"


class TestFluxKeyNormalization:
    """Tests for Flux Klein checkpoint key normalization."""

    def test_normalize_key_handles_diffusion_model_prefix(self) -> None:
        """diffusion_model.double_blocks.X keys are preserved."""
        file_key = "diffusion_model.double_blocks.0.img_attn.qkv.weight"
        normalized = _normalize_key(file_key)
        assert normalized == "diffusion_model.double_blocks.0.img_attn.qkv.weight"

    def test_normalize_key_handles_transformer_prefix(self) -> None:
        """transformer.double_blocks.X normalizes to diffusion_model.double_blocks.X."""
        file_key = "transformer.double_blocks.0.img_attn.qkv.weight"
        normalized = _normalize_key(file_key)
        assert normalized == "diffusion_model.double_blocks.0.img_attn.qkv.weight"

    def test_flux_checkpoint_keys_normalized_correctly(
        self, flux_checkpoint_path: str
    ) -> None:
        """Flux checkpoint keys are normalized to base model format."""
        with ModelLoader(flux_checkpoint_path) as loader:
            for key in loader.affected_keys:
                assert key.startswith("diffusion_model.")

    def test_flux_checkpoint_excludes_vae(self, flux_checkpoint_path: str) -> None:
        """Flux checkpoint VAE keys are excluded."""
        with ModelLoader(flux_checkpoint_path) as loader:
            for key in loader.affected_keys:
                assert "first_stage_model" not in key

    def test_flux_keys_retrievable(self, flux_checkpoint_path: str) -> None:
        """Keys can be retrieved from Flux checkpoint after normalization."""
        with ModelLoader(flux_checkpoint_path) as loader:
            # Get first few keys and retrieve tensors
            keys = list(loader.affected_keys)[:3]
            tensors = loader.get_weights(keys)
            assert len(tensors) == 3
            for t in tensors:
                assert isinstance(t, torch.Tensor)
