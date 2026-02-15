"""Tests for CLIPModelLoader -- CLIP text encoder streaming loader."""

import tempfile

import pytest
import torch
from safetensors.torch import save_file

from lib.clip_model_loader import (
    CLIPKeyMappingError,
    CLIPModelLoader,
    _normalize_clip_key,
    _validate_embedder_structure,
)
from lib.model_loader import UnsupportedFormatError

# ---------------------------------------------------------------------------
# Fixtures for creating test checkpoint files
# ---------------------------------------------------------------------------


@pytest.fixture
def sdxl_checkpoint_path() -> str:
    """Create a temporary SDXL-format checkpoint with both CLIP encoders.

    Uses conditioner.embedders.{0,1}.transformer structure for CLIP-L and CLIP-G.
    """
    # CLIP-L keys (embedders.0)
    clip_l_prefix = "conditioner.embedders.0.transformer.text_model"
    # CLIP-G keys (embedders.1)
    clip_g_prefix = "conditioner.embedders.1.transformer.text_model"

    tensors = {
        f"{clip_l_prefix}.encoder.layers.0.self_attn.k_proj.weight": torch.randn(768, 768),
        f"{clip_l_prefix}.encoder.layers.0.self_attn.v_proj.weight": torch.randn(768, 768),
        f"{clip_l_prefix}.encoder.layers.0.mlp.fc1.weight": torch.randn(3072, 768),
        f"{clip_l_prefix}.embeddings.position_embedding.weight": torch.randn(77, 768),
        f"{clip_g_prefix}.encoder.layers.0.self_attn.k_proj.weight": torch.randn(1280, 1280),
        f"{clip_g_prefix}.encoder.layers.0.self_attn.v_proj.weight": torch.randn(1280, 1280),
        f"{clip_g_prefix}.encoder.layers.0.mlp.fc1.weight": torch.randn(5120, 1280),
        f"{clip_g_prefix}.embeddings.position_embedding.weight": torch.randn(77, 1280),
        # Diffusion model keys (should be excluded by CLIPModelLoader)
        "model.diffusion_model.input_blocks.0.0.weight": torch.randn(4, 4),
        "model.diffusion_model.middle_block.0.weight": torch.randn(4, 4),
        "model.diffusion_model.output_blocks.0.0.weight": torch.randn(4, 4),
        # VAE keys (should be excluded)
        "model.first_stage_model.encoder.weight": torch.randn(4, 4),
    }
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def clip_l_only_checkpoint_path() -> str:
    """Create a checkpoint with only CLIP-L (embedders.0) keys."""
    prefix = "conditioner.embedders.0.transformer.text_model"
    tensors = {
        f"{prefix}.encoder.layers.0.self_attn.k_proj.weight": torch.randn(768, 768),
        f"{prefix}.encoder.layers.0.mlp.fc1.weight": torch.randn(3072, 768),
    }
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def checkpoint_with_model_prefix_path() -> str:
    """Create a checkpoint with model.conditioner prefix."""
    prefix_l = "model.conditioner.embedders.0.transformer.text_model"
    prefix_g = "model.conditioner.embedders.1.transformer.text_model"
    tensors = {
        f"{prefix_l}.encoder.layers.0.weight": torch.randn(768, 768),
        f"{prefix_g}.encoder.layers.0.weight": torch.randn(1280, 1280),
    }
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def unexpected_embedder_checkpoint_path() -> str:
    """Create a checkpoint with unexpected embedder indices."""
    tensors = {
        "conditioner.embedders.0.transformer.text_model.weight": torch.randn(768, 768),
        "conditioner.embedders.2.transformer.text_model.weight": torch.randn(
            768, 768
        ),  # Unexpected index 2
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


@pytest.fixture
def diffusion_only_checkpoint_path() -> str:
    """Create a checkpoint with only diffusion model keys (no CLIP)."""
    tensors = {
        "model.diffusion_model.input_blocks.0.0.weight": torch.randn(4, 4),
        "model.diffusion_model.middle_block.0.weight": torch.randn(4, 4),
    }
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file(tensors, f.name)
        return f.name


# ---------------------------------------------------------------------------
# AC-1: Exposes text encoder keys, excludes diffusion and VAE
# ---------------------------------------------------------------------------


# AC: @clip-model-loader ac-1
class TestTextEncoderKeyExposure:
    """Tests for text encoder key inclusion and other key exclusion."""

    def test_loader_exposes_clip_keys(self, sdxl_checkpoint_path: str) -> None:
        """Loader exposes text encoder keys (conditioner.embedders.*)."""
        with CLIPModelLoader(sdxl_checkpoint_path) as loader:
            assert len(loader.affected_keys) > 0
            # All exposed keys should be CLIP keys
            for key in loader.affected_keys:
                assert key.startswith(("clip_l.", "clip_g."))

    def test_loader_excludes_diffusion_keys(self, sdxl_checkpoint_path: str) -> None:
        """Diffusion model keys are excluded."""
        with CLIPModelLoader(sdxl_checkpoint_path) as loader:
            for key in loader.affected_keys:
                assert "diffusion_model" not in key
                assert "input_blocks" not in key
                assert "middle_block" not in key
                assert "output_blocks" not in key

    def test_loader_excludes_vae_keys(self, sdxl_checkpoint_path: str) -> None:
        """VAE keys are excluded."""
        with CLIPModelLoader(sdxl_checkpoint_path) as loader:
            for key in loader.affected_keys:
                assert "first_stage_model" not in key

    def test_normalize_clip_key_returns_none_for_diffusion(self) -> None:
        """_normalize_clip_key returns None for diffusion model keys."""
        assert _normalize_clip_key("model.diffusion_model.input_blocks.0.weight") is None
        assert _normalize_clip_key("diffusion_model.middle_block.0.weight") is None

    def test_normalize_clip_key_returns_none_for_vae(self) -> None:
        """_normalize_clip_key returns None for VAE keys."""
        assert _normalize_clip_key("model.first_stage_model.encoder.weight") is None
        assert _normalize_clip_key("first_stage_model.decoder.weight") is None

    def test_loader_works_with_empty_clip_keys(
        self, diffusion_only_checkpoint_path: str
    ) -> None:
        """Loader handles checkpoints without any CLIP keys."""
        with CLIPModelLoader(diffusion_only_checkpoint_path) as loader:
            assert len(loader.affected_keys) == 0


# ---------------------------------------------------------------------------
# AC-2: get_weights() returns mapped CLIP base model keys
# ---------------------------------------------------------------------------


# AC: @clip-model-loader ac-2
class TestGetWeights:
    """Tests for get_weights() tensor retrieval with mapped keys."""

    def test_get_weights_returns_tensors_for_requested_keys(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """get_weights() returns tensors in order of requested keys."""
        with CLIPModelLoader(sdxl_checkpoint_path) as loader:
            keys = list(loader.affected_keys)[:2]
            tensors = loader.get_weights(keys)

            assert len(tensors) == 2
            for t in tensors:
                assert isinstance(t, torch.Tensor)

    def test_get_weights_maps_to_clip_base_model_format(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """Keys are mapped from conditioner.embedders to clip_l/clip_g format."""
        with CLIPModelLoader(sdxl_checkpoint_path) as loader:
            # File has conditioner.embedders.0.transformer.text_model...
            # Loader should expose as clip_l.transformer.text_model...
            expected_key = "clip_l.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight"
            assert expected_key in loader.affected_keys

            tensors = loader.get_weights([expected_key])
            assert len(tensors) == 1
            assert tensors[0].shape == (768, 768)

    def test_get_weights_raises_for_missing_keys(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """get_weights() raises KeyError for missing keys."""
        with CLIPModelLoader(sdxl_checkpoint_path) as loader:
            with pytest.raises(KeyError) as exc_info:
                loader.get_weights(["clip_l.nonexistent.weight"])

            assert "missing" in str(exc_info.value).lower()
            assert "nonexistent" in str(exc_info.value)

    def test_get_weights_error_lists_missing_keys(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """Error message lists all missing keys."""
        with CLIPModelLoader(sdxl_checkpoint_path) as loader:
            missing = ["clip_l.missing1.weight", "clip_l.missing2.weight"]
            with pytest.raises(KeyError) as exc_info:
                loader.get_weights(missing)

            error_msg = str(exc_info.value)
            assert "missing1" in error_msg
            assert "missing2" in error_msg


# ---------------------------------------------------------------------------
# AC-3: affected_keys returns frozenset of CLIP base model keys
# ---------------------------------------------------------------------------


# AC: @clip-model-loader ac-3
class TestAffectedKeys:
    """Tests for affected_keys property."""

    def test_affected_keys_returns_frozenset(self, sdxl_checkpoint_path: str) -> None:
        """affected_keys returns frozenset to prevent mutation."""
        with CLIPModelLoader(sdxl_checkpoint_path) as loader:
            assert isinstance(loader.affected_keys, frozenset)

    def test_affected_keys_contains_both_encoders(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """affected_keys includes both CLIP-L and CLIP-G keys."""
        with CLIPModelLoader(sdxl_checkpoint_path) as loader:
            clip_l_keys = [k for k in loader.affected_keys if k.startswith("clip_l.")]
            clip_g_keys = [k for k in loader.affected_keys if k.startswith("clip_g.")]

            assert len(clip_l_keys) > 0
            assert len(clip_g_keys) > 0

    def test_has_clip_l_property(self, sdxl_checkpoint_path: str) -> None:
        """has_clip_l property returns True when CLIP-L keys present."""
        with CLIPModelLoader(sdxl_checkpoint_path) as loader:
            assert loader.has_clip_l is True

    def test_has_clip_g_property(self, sdxl_checkpoint_path: str) -> None:
        """has_clip_g property returns True when CLIP-G keys present."""
        with CLIPModelLoader(sdxl_checkpoint_path) as loader:
            assert loader.has_clip_g is True

    def test_has_clip_l_only(self, clip_l_only_checkpoint_path: str) -> None:
        """has_clip_l is True, has_clip_g is False for CLIP-L only checkpoint."""
        with CLIPModelLoader(clip_l_only_checkpoint_path) as loader:
            assert loader.has_clip_l is True
            assert loader.has_clip_g is False


# ---------------------------------------------------------------------------
# AC-4: cleanup() closes the safe_open file handle
# ---------------------------------------------------------------------------


# AC: @clip-model-loader ac-4
class TestCleanup:
    """Tests for cleanup() resource management."""

    def test_cleanup_releases_handle(self, sdxl_checkpoint_path: str) -> None:
        """cleanup() releases the file handle."""
        loader = CLIPModelLoader(sdxl_checkpoint_path)
        assert loader._handle is not None

        loader.cleanup()
        assert loader._handle is None

    def test_context_manager_calls_cleanup(self, sdxl_checkpoint_path: str) -> None:
        """Context manager calls cleanup() on exit."""
        with CLIPModelLoader(sdxl_checkpoint_path) as loader:
            assert loader._handle is not None

        assert loader._handle is None


# ---------------------------------------------------------------------------
# AC-5: Non-safetensors files raise UnsupportedFormatError
# ---------------------------------------------------------------------------


# AC: @clip-model-loader ac-5
class TestUnsupportedFormatError:
    """Tests for non-safetensors format rejection."""

    def test_ckpt_file_raises_unsupported_error(self, non_safetensors_path: str) -> None:
        """Opening a .ckpt file raises UnsupportedFormatError."""
        with pytest.raises(UnsupportedFormatError) as exc_info:
            CLIPModelLoader(non_safetensors_path)

        error_msg = str(exc_info.value)
        assert "safetensors" in error_msg.lower()
        assert ".ckpt" in error_msg

    def test_pt_file_raises_unsupported_error(self) -> None:
        """Opening a .pt file raises UnsupportedFormatError."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"dummy")
            pt_path = f.name

        with pytest.raises(UnsupportedFormatError) as exc_info:
            CLIPModelLoader(pt_path)

        assert "safetensors" in str(exc_info.value).lower()

    def test_error_message_suggests_conversion(
        self, non_safetensors_path: str
    ) -> None:
        """Error message suggests converting to safetensors."""
        with pytest.raises(UnsupportedFormatError) as exc_info:
            CLIPModelLoader(non_safetensors_path)

        assert "convert" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# AC-6: Key normalization maps embedders to clip_l/clip_g
# ---------------------------------------------------------------------------


# AC: @clip-model-loader ac-6
class TestKeyNormalization:
    """Tests for CLIP key normalization."""

    def test_embedders_0_maps_to_clip_l(self) -> None:
        """conditioner.embedders.0.* maps to clip_l.*."""
        file_key = "conditioner.embedders.0.transformer.text_model.encoder.layers.0.weight"
        normalized = _normalize_clip_key(file_key)
        assert normalized == "clip_l.transformer.text_model.encoder.layers.0.weight"

    def test_embedders_1_maps_to_clip_g(self) -> None:
        """conditioner.embedders.1.* maps to clip_g.*."""
        file_key = "conditioner.embedders.1.transformer.text_model.encoder.layers.0.weight"
        normalized = _normalize_clip_key(file_key)
        assert normalized == "clip_g.transformer.text_model.encoder.layers.0.weight"

    def test_model_conditioner_prefix_handled(
        self, checkpoint_with_model_prefix_path: str
    ) -> None:
        """model.conditioner.embedders.* prefix is normalized correctly."""
        with CLIPModelLoader(checkpoint_with_model_prefix_path) as loader:
            # Should have both clip_l and clip_g keys
            clip_l_keys = [k for k in loader.affected_keys if k.startswith("clip_l.")]
            clip_g_keys = [k for k in loader.affected_keys if k.startswith("clip_g.")]

            assert len(clip_l_keys) > 0
            assert len(clip_g_keys) > 0

    def test_normalize_clip_key_with_model_prefix(self) -> None:
        """model.conditioner.embedders.0.* normalizes correctly."""
        file_key = "model.conditioner.embedders.0.transformer.text_model.weight"
        normalized = _normalize_clip_key(file_key)
        assert normalized == "clip_l.transformer.text_model.weight"

    def test_complex_key_path_preserved(self) -> None:
        """Complex key paths are preserved after prefix mapping."""
        prefix = "conditioner.embedders.0.transformer.text_model"
        file_key = f"{prefix}.encoder.layers.11.self_attn.out_proj.weight"
        normalized = _normalize_clip_key(file_key)
        expected = "clip_l.transformer.text_model.encoder.layers.11.self_attn.out_proj.weight"
        assert normalized == expected

    def test_embeddings_keys_normalized(self) -> None:
        """Embedding keys (not just layer keys) are normalized."""
        prefix = "conditioner.embedders.1.transformer.text_model"
        file_key = f"{prefix}.embeddings.position_embedding.weight"
        normalized = _normalize_clip_key(file_key)
        expected = "clip_g.transformer.text_model.embeddings.position_embedding.weight"
        assert normalized == expected


# ---------------------------------------------------------------------------
# AC-7: Clear error for unexpected embedder structure
# ---------------------------------------------------------------------------


# AC: @clip-model-loader ac-7
class TestUnexpectedEmbedderStructure:
    """Tests for clear errors on unexpected checkpoint structure."""

    def test_unexpected_embedder_index_raises_error(
        self, unexpected_embedder_checkpoint_path: str
    ) -> None:
        """Checkpoint with unexpected embedder indices raises CLIPKeyMappingError."""
        with pytest.raises(CLIPKeyMappingError) as exc_info:
            CLIPModelLoader(unexpected_embedder_checkpoint_path)

        error_msg = str(exc_info.value)
        assert "unexpected" in error_msg.lower()
        assert "2" in error_msg  # The unexpected index

    def test_error_message_describes_expected_structure(
        self, unexpected_embedder_checkpoint_path: str
    ) -> None:
        """Error message describes expected SDXL embedder structure."""
        with pytest.raises(CLIPKeyMappingError) as exc_info:
            CLIPModelLoader(unexpected_embedder_checkpoint_path)

        error_msg = str(exc_info.value)
        assert "embedders.0" in error_msg or "CLIP-L" in error_msg
        assert "embedders.1" in error_msg or "CLIP-G" in error_msg

    def test_validate_embedder_structure_returns_presence_info(self) -> None:
        """_validate_embedder_structure returns (has_clip_l, has_clip_g)."""
        keys = [
            "conditioner.embedders.0.transformer.weight",
            "conditioner.embedders.1.transformer.weight",
        ]
        has_l, has_g = _validate_embedder_structure(keys)
        assert has_l is True
        assert has_g is True

    def test_validate_embedder_structure_clip_l_only(self) -> None:
        """_validate_embedder_structure handles CLIP-L only."""
        keys = ["conditioner.embedders.0.transformer.weight"]
        has_l, has_g = _validate_embedder_structure(keys)
        assert has_l is True
        assert has_g is False

    def test_validate_embedder_structure_raises_for_unexpected(self) -> None:
        """_validate_embedder_structure raises for unexpected indices."""
        keys = [
            "conditioner.embedders.0.transformer.weight",
            "conditioner.embedders.3.transformer.weight",  # Unexpected
        ]
        with pytest.raises(CLIPKeyMappingError):
            _validate_embedder_structure(keys)


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_normalize_clip_key_returns_none_for_incomplete_path(self) -> None:
        """Incomplete conditioner paths return None."""
        # Missing embedder index
        assert _normalize_clip_key("conditioner.embedders") is None
        # Non-numeric embedder index
        assert _normalize_clip_key("conditioner.embedders.x.transformer.weight") is None

    def test_normalize_clip_key_returns_none_for_unknown_prefix(self) -> None:
        """Keys with unknown prefixes return None."""
        assert _normalize_clip_key("unknown.prefix.weight") is None
        assert _normalize_clip_key("some.random.key") is None

    def test_loader_handles_many_keys(self) -> None:
        """Loader handles checkpoints with many CLIP keys."""
        tensors = {}
        # Create many CLIP-L and CLIP-G keys
        for i in range(32):
            tensors[
                f"conditioner.embedders.0.transformer.text_model.encoder.layers.{i}.self_attn.k_proj.weight"
            ] = torch.randn(768, 768)
            tensors[
                f"conditioner.embedders.1.transformer.text_model.encoder.layers.{i}.self_attn.k_proj.weight"
            ] = torch.randn(1280, 1280)

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            save_file(tensors, f.name)
            path = f.name

        with CLIPModelLoader(path) as loader:
            assert len(loader.affected_keys) == 64
            # Verify keys are accessible
            first_key = list(loader.affected_keys)[0]
            tensors = loader.get_weights([first_key])
            assert len(tensors) == 1

    def test_get_weights_empty_list_returns_empty(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """get_weights([]) returns empty list."""
        with CLIPModelLoader(sdxl_checkpoint_path) as loader:
            tensors = loader.get_weights([])
            assert tensors == []
