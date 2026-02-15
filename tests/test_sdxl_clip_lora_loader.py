"""Tests for SDXL CLIP LoRA loader.

Covers all 7 acceptance criteria:
- AC-1: lora_te1_* maps to CLIP-L base model keys (clip_l.transformer.*)
- AC-2: lora_te2_* maps to CLIP-G base model keys (clip_g.transformer.*)
- AC-3: UNet-only LoRA returns empty affected_keys
- AC-4: get_delta_specs returns DeltaSpec with correct tensors
- AC-5: Implements full loader interface
- AC-6: Only extracts text encoder keys (ignores UNet)
- AC-7: Registered as get_loader(arch="sdxl", domain="clip")
"""

import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lib.executor import DeltaSpec
from lib.lora import (
    LOADER_REGISTRY,
    SDXLCLIPLoader,
    get_loader,
)

# ---------------------------------------------------------------------------
# Fixtures: Create temporary LoRA files for testing
# ---------------------------------------------------------------------------


@pytest.fixture
def clip_l_lora_file() -> str:
    """Create a temporary LoRA file with CLIP-L (te1) keys only.

    # AC: @sdxl-clip-lora-loader ac-1
    """
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tensors = {
            # CLIP-L layer 0 attention
            "lora_te1_text_model_encoder_layers_0_self_attn_k_proj.lora_up.weight": torch.randn(
                768, 8
            ),
            "lora_te1_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight": torch.randn(
                8, 768
            ),
            # CLIP-L layer 5 MLP
            "lora_te1_text_model_encoder_layers_5_mlp_fc1.lora_up.weight": torch.randn(
                3072, 8
            ),
            "lora_te1_text_model_encoder_layers_5_mlp_fc1.lora_down.weight": torch.randn(
                8, 768
            ),
        }
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def clip_g_lora_file() -> str:
    """Create a temporary LoRA file with CLIP-G (te2) keys only.

    # AC: @sdxl-clip-lora-loader ac-2
    """
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tensors = {
            # CLIP-G layer 0 attention
            "lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight": torch.randn(
                1280, 16
            ),
            "lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": torch.randn(
                16, 1280
            ),
            # CLIP-G layer 20 MLP
            "lora_te2_text_model_encoder_layers_20_mlp_fc2.lora_up.weight": torch.randn(
                1280, 16
            ),
            "lora_te2_text_model_encoder_layers_20_mlp_fc2.lora_down.weight": torch.randn(
                16, 5120
            ),
        }
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def mixed_clip_lora_file() -> str:
    """Create a temporary LoRA file with both CLIP-L and CLIP-G keys."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tensors = {
            # CLIP-L (te1)
            "lora_te1_text_model_encoder_layers_3_self_attn_v_proj.lora_up.weight": torch.randn(
                768, 8
            ),
            "lora_te1_text_model_encoder_layers_3_self_attn_v_proj.lora_down.weight": torch.randn(
                8, 768
            ),
            # CLIP-G (te2) - shorter layer to avoid line length
            "lora_te2_text_model_encoder_layers_10_self_attn_v_proj.lora_up.weight": torch.randn(
                1280, 16
            ),
            "lora_te2_text_model_encoder_layers_10_self_attn_v_proj.lora_down.weight": torch.randn(
                16, 1280
            ),
        }
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def unet_only_lora_file() -> str:
    """Create a temporary LoRA file with only UNet keys (no text encoder).

    # AC: @sdxl-clip-lora-loader ac-3
    """
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tensors = {
            "lora_unet_input_blocks_0_0_proj_in.lora_up.weight": torch.randn(64, 8),
            "lora_unet_input_blocks_0_0_proj_in.lora_down.weight": torch.randn(8, 32),
            "lora_unet_middle_block_0_proj.lora_up.weight": torch.randn(128, 16),
            "lora_unet_middle_block_0_proj.lora_down.weight": torch.randn(16, 64),
        }
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def mixed_unet_clip_lora_file() -> str:
    """Create a temporary LoRA file with both UNet and CLIP keys.

    # AC: @sdxl-clip-lora-loader ac-6
    """
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tensors = {
            # UNet keys (should be ignored)
            "lora_unet_input_blocks_4_1_proj_in.lora_up.weight": torch.randn(640, 8),
            "lora_unet_input_blocks_4_1_proj_in.lora_down.weight": torch.randn(8, 640),
            # CLIP-L keys (should be extracted)
            "lora_te1_text_model_encoder_layers_7_mlp_fc2.lora_up.weight": torch.randn(
                768, 8
            ),
            "lora_te1_text_model_encoder_layers_7_mlp_fc2.lora_down.weight": torch.randn(
                8, 3072
            ),
            # CLIP-G keys (should be extracted)
            "lora_te2_text_model_encoder_layers_25_self_attn_k_proj.lora_up.weight": torch.randn(
                1280, 16
            ),
            "lora_te2_text_model_encoder_layers_25_self_attn_k_proj.lora_down.weight": torch.randn(
                16, 1280
            ),
        }
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def clip_lora_with_alpha_file() -> str:
    """Create a temporary LoRA file with CLIP keys and alpha values."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        rank = 8
        tensors = {
            "lora_te1_text_model_encoder_layers_2_self_attn_k_proj.lora_up.weight": torch.randn(
                768, rank
            ),
            "lora_te1_text_model_encoder_layers_2_self_attn_k_proj.lora_down.weight": torch.randn(
                rank, 768
            ),
            # Alpha value different from rank
            "lora_te1_text_model_encoder_layers_2_self_attn_k_proj.alpha": torch.tensor(4.0),
        }
        save_file(tensors, f.name)
        return f.name


@pytest.fixture(autouse=True)
def cleanup_lora_files(
    clip_l_lora_file: str,
    clip_g_lora_file: str,
    mixed_clip_lora_file: str,
    unet_only_lora_file: str,
    mixed_unet_clip_lora_file: str,
    clip_lora_with_alpha_file: str,
):
    """Clean up temporary files after tests."""
    yield
    for path in [
        clip_l_lora_file,
        clip_g_lora_file,
        mixed_clip_lora_file,
        unet_only_lora_file,
        mixed_unet_clip_lora_file,
        clip_lora_with_alpha_file,
    ]:
        Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# AC-1: lora_te1_* maps to CLIP-L base model keys
# ---------------------------------------------------------------------------


class TestAC1CLIPLKeyMapping:
    """AC-1: lora_te1_* keys map to clip_l.transformer.* base model keys."""

    def test_te1_maps_to_clip_l_prefix(self, clip_l_lora_file: str):
        """lora_te1_* keys produce clip_l.transformer.* model keys."""
        # AC: @sdxl-clip-lora-loader ac-1
        loader = SDXLCLIPLoader()
        loader.load(clip_l_lora_file)

        affected = loader.affected_keys
        assert len(affected) > 0, "Should have affected keys"

        for key in affected:
            assert key.startswith("clip_l.transformer."), (
                f"Key {key} should start with clip_l.transformer."
            )
            assert key.endswith(".weight"), f"Key {key} should end with .weight"

        loader.cleanup()

    def test_te1_layer_numbers_preserved(self, clip_l_lora_file: str):
        """Layer numbers in lora_te1_* are preserved in mapped keys."""
        # AC: @sdxl-clip-lora-loader ac-1
        loader = SDXLCLIPLoader()
        loader.load(clip_l_lora_file)

        affected = loader.affected_keys

        # Should have layer 0 and layer 5 from fixture
        layer_0_keys = [k for k in affected if "layers.0." in k]
        layer_5_keys = [k for k in affected if "layers.5." in k]

        assert len(layer_0_keys) > 0, f"Expected layer 0 keys in {affected}"
        assert len(layer_5_keys) > 0, f"Expected layer 5 keys in {affected}"

        loader.cleanup()

    def test_te1_attention_key_mapping(self, clip_l_lora_file: str):
        """Attention component keys are correctly mapped."""
        # AC: @sdxl-clip-lora-loader ac-1
        loader = SDXLCLIPLoader()
        loader.load(clip_l_lora_file)

        affected = loader.affected_keys

        # Should have self_attn.k_proj key from fixture
        k_proj_keys = [k for k in affected if "self_attn.k_proj" in k]
        assert len(k_proj_keys) > 0, f"Expected self_attn.k_proj key in {affected}"

        loader.cleanup()

    def test_te1_mlp_key_mapping(self, clip_l_lora_file: str):
        """MLP component keys are correctly mapped."""
        # AC: @sdxl-clip-lora-loader ac-1
        loader = SDXLCLIPLoader()
        loader.load(clip_l_lora_file)

        affected = loader.affected_keys

        # Should have mlp.fc1 key from fixture
        fc1_keys = [k for k in affected if "mlp.fc1" in k]
        assert len(fc1_keys) > 0, f"Expected mlp.fc1 key in {affected}"

        loader.cleanup()


# ---------------------------------------------------------------------------
# AC-2: lora_te2_* maps to CLIP-G base model keys
# ---------------------------------------------------------------------------


class TestAC2CLIPGKeyMapping:
    """AC-2: lora_te2_* keys map to clip_g.transformer.* base model keys."""

    def test_te2_maps_to_clip_g_prefix(self, clip_g_lora_file: str):
        """lora_te2_* keys produce clip_g.transformer.* model keys."""
        # AC: @sdxl-clip-lora-loader ac-2
        loader = SDXLCLIPLoader()
        loader.load(clip_g_lora_file)

        affected = loader.affected_keys
        assert len(affected) > 0, "Should have affected keys"

        for key in affected:
            assert key.startswith("clip_g.transformer."), (
                f"Key {key} should start with clip_g.transformer."
            )
            assert key.endswith(".weight"), f"Key {key} should end with .weight"

        loader.cleanup()

    def test_te2_layer_numbers_preserved(self, clip_g_lora_file: str):
        """Layer numbers in lora_te2_* are preserved in mapped keys."""
        # AC: @sdxl-clip-lora-loader ac-2
        loader = SDXLCLIPLoader()
        loader.load(clip_g_lora_file)

        affected = loader.affected_keys

        # Should have layer 0 and layer 20 from fixture
        layer_0_keys = [k for k in affected if "layers.0." in k]
        layer_20_keys = [k for k in affected if "layers.20." in k]

        assert len(layer_0_keys) > 0, f"Expected layer 0 keys in {affected}"
        assert len(layer_20_keys) > 0, f"Expected layer 20 keys in {affected}"

        loader.cleanup()

    def test_te2_attention_key_mapping(self, clip_g_lora_file: str):
        """Attention component keys are correctly mapped for CLIP-G."""
        # AC: @sdxl-clip-lora-loader ac-2
        loader = SDXLCLIPLoader()
        loader.load(clip_g_lora_file)

        affected = loader.affected_keys

        # Should have self_attn.q_proj key from fixture
        q_proj_keys = [k for k in affected if "self_attn.q_proj" in k]
        assert len(q_proj_keys) > 0, f"Expected self_attn.q_proj key in {affected}"

        loader.cleanup()

    def test_te2_mlp_key_mapping(self, clip_g_lora_file: str):
        """MLP component keys are correctly mapped for CLIP-G."""
        # AC: @sdxl-clip-lora-loader ac-2
        loader = SDXLCLIPLoader()
        loader.load(clip_g_lora_file)

        affected = loader.affected_keys

        # Should have mlp.fc2 key from fixture
        fc2_keys = [k for k in affected if "mlp.fc2" in k]
        assert len(fc2_keys) > 0, f"Expected mlp.fc2 key in {affected}"

        loader.cleanup()


# ---------------------------------------------------------------------------
# AC-3: UNet-only LoRA returns empty affected_keys
# ---------------------------------------------------------------------------


class TestAC3UNetOnlyLoRA:
    """AC-3: LoRA with only UNet keys (no te1/te2) returns empty affected_keys."""

    def test_unet_only_returns_empty_frozenset(self, unet_only_lora_file: str):
        """LoRA with only lora_unet_* keys returns empty affected_keys."""
        # AC: @sdxl-clip-lora-loader ac-3
        loader = SDXLCLIPLoader()
        loader.load(unet_only_lora_file)

        affected = loader.affected_keys
        assert affected == frozenset(), f"Expected empty frozenset, got {affected}"

        loader.cleanup()

    def test_unet_only_get_delta_specs_returns_empty(self, unet_only_lora_file: str):
        """get_delta_specs returns empty list for UNet-only LoRA."""
        # AC: @sdxl-clip-lora-loader ac-3
        loader = SDXLCLIPLoader()
        loader.load(unet_only_lora_file)

        # Even with some keys to query, should return empty
        specs = loader.get_delta_specs(
            ["diffusion_model.input_blocks.0.0.proj_in.weight"],
            {"diffusion_model.input_blocks.0.0.proj_in.weight": 0},
        )
        assert specs == [], f"Expected empty list, got {specs}"

        loader.cleanup()


# ---------------------------------------------------------------------------
# AC-4: get_delta_specs returns DeltaSpec with correct tensors
# ---------------------------------------------------------------------------


class TestAC4DeltaSpecProduction:
    """AC-4: get_delta_specs returns DeltaSpec objects with correct tensors."""

    def test_produces_deltaspec_objects(self, clip_l_lora_file: str):
        """Loader produces DeltaSpec instances with correct fields."""
        # AC: @sdxl-clip-lora-loader ac-4
        loader = SDXLCLIPLoader()
        loader.load(clip_l_lora_file, strength=0.8)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        assert len(specs) > 0, "Should produce at least one DeltaSpec"

        for spec in specs:
            assert isinstance(spec, DeltaSpec)
            assert spec.kind == "standard"
            assert spec.key_index in key_indices.values()
            assert spec.up is not None
            assert spec.down is not None
            assert isinstance(spec.scale, float)

        loader.cleanup()

    def test_deltaspec_tensors_have_correct_shapes(self, clip_l_lora_file: str):
        """DeltaSpec up/down tensors have valid shapes for bmm."""
        # AC: @sdxl-clip-lora-loader ac-4
        loader = SDXLCLIPLoader()
        loader.load(clip_l_lora_file)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        for spec in specs:
            # up: (out, rank), down: (rank, in) for bmm compatibility
            assert spec.up.dim() == 2, f"Up should be 2D: {spec.up.shape}"
            assert spec.down.dim() == 2, f"Down should be 2D: {spec.down.shape}"
            # up columns == down rows (rank dimension)
            assert spec.up.shape[1] == spec.down.shape[0], (
                f"Rank mismatch: up {spec.up.shape} vs down {spec.down.shape}"
            )

        loader.cleanup()

    def test_deltaspec_dimensions_match_base_model(self, clip_l_lora_file: str):
        """DeltaSpec tensor dimensions match expected base model parameter shapes."""
        # AC: @sdxl-clip-lora-loader ac-4
        loader = SDXLCLIPLoader()
        loader.load(clip_l_lora_file)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        # CLIP-L hidden dim is 768
        for spec in specs:
            # up.rows should be output dim, down.cols should be input dim
            # For CLIP-L, these should be related to 768 or 3072 (MLP expansion)
            out_dim = spec.up.shape[0]
            in_dim = spec.down.shape[1]

            # Check dimensions are reasonable for CLIP-L
            assert out_dim in [768, 3072], f"Unexpected out_dim: {out_dim}"
            assert in_dim in [768, 3072], f"Unexpected in_dim: {in_dim}"

        loader.cleanup()

    def test_strength_affects_scale(self, clip_l_lora_file: str):
        """LoRA strength multiplier affects DeltaSpec scale."""
        # AC: @sdxl-clip-lora-loader ac-4
        loader1 = SDXLCLIPLoader()
        loader1.load(clip_l_lora_file, strength=1.0)
        keys = list(loader1.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs1 = loader1.get_delta_specs(keys, key_indices)
        loader1.cleanup()

        loader2 = SDXLCLIPLoader()
        loader2.load(clip_l_lora_file, strength=0.5)
        specs2 = loader2.get_delta_specs(keys, key_indices)
        loader2.cleanup()

        # Same key should have half the scale
        assert len(specs1) == len(specs2)
        for s1, s2 in zip(specs1, specs2, strict=True):
            assert abs(s1.scale - 2 * s2.scale) < 1e-6, (
                f"Scale mismatch: {s1.scale} vs {s2.scale}"
            )

    def test_alpha_affects_scale(self, clip_lora_with_alpha_file: str):
        """Alpha value in LoRA file affects DeltaSpec scale."""
        # AC: @sdxl-clip-lora-loader ac-4
        loader = SDXLCLIPLoader()
        loader.load(clip_lora_with_alpha_file, strength=1.0)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        assert len(specs) == 1
        # scale = strength * alpha / rank = 1.0 * 4.0 / 8 = 0.5
        assert abs(specs[0].scale - 0.5) < 1e-6, f"Expected scale 0.5, got {specs[0].scale}"

        loader.cleanup()


# ---------------------------------------------------------------------------
# AC-5: Implements full loader interface
# ---------------------------------------------------------------------------


class TestAC5LoaderInterface:
    """AC-5: SDXLCLIPLoader implements the full LoRALoader interface."""

    def test_has_load_method(self):
        """Loader has load(path, strength, set_id) method."""
        # AC: @sdxl-clip-lora-loader ac-5
        loader = SDXLCLIPLoader()
        assert callable(getattr(loader, "load", None))

        # Check signature includes strength and set_id
        import inspect

        sig = inspect.signature(loader.load)
        params = list(sig.parameters.keys())
        assert "path" in params
        assert "strength" in params
        assert "set_id" in params

    def test_has_affected_keys_property(self):
        """Loader has affected_keys property returning frozenset."""
        # AC: @sdxl-clip-lora-loader ac-5
        loader = SDXLCLIPLoader()
        assert hasattr(loader, "affected_keys")
        assert isinstance(loader.affected_keys, frozenset)

    def test_has_affected_keys_for_set_method(self):
        """Loader has affected_keys_for_set(set_id) method."""
        # AC: @sdxl-clip-lora-loader ac-5
        loader = SDXLCLIPLoader()
        assert callable(getattr(loader, "affected_keys_for_set", None))

    def test_has_get_delta_specs_method(self):
        """Loader has get_delta_specs(keys, key_indices, set_id) method."""
        # AC: @sdxl-clip-lora-loader ac-5
        loader = SDXLCLIPLoader()
        assert callable(getattr(loader, "get_delta_specs", None))

        # Check signature includes set_id
        import inspect

        sig = inspect.signature(loader.get_delta_specs)
        params = list(sig.parameters.keys())
        assert "keys" in params
        assert "key_indices" in params
        assert "set_id" in params

    def test_has_cleanup_method(self):
        """Loader has cleanup() method."""
        # AC: @sdxl-clip-lora-loader ac-5
        loader = SDXLCLIPLoader()
        assert callable(getattr(loader, "cleanup", None))

    def test_cleanup_clears_state(self, clip_l_lora_file: str):
        """cleanup() releases loaded tensors."""
        # AC: @sdxl-clip-lora-loader ac-5
        loader = SDXLCLIPLoader()
        loader.load(clip_l_lora_file)
        assert len(loader.affected_keys) > 0, "Should have affected keys"

        loader.cleanup()
        assert len(loader.affected_keys) == 0, "cleanup should clear affected keys"

    def test_context_manager_calls_cleanup(self, clip_l_lora_file: str):
        """Context manager (__enter__/__exit__) calls cleanup automatically."""
        # AC: @sdxl-clip-lora-loader ac-5
        with SDXLCLIPLoader() as loader:
            loader.load(clip_l_lora_file)
            assert len(loader.affected_keys) > 0

        # After context exit, cleanup should have been called
        assert len(loader.affected_keys) == 0

    def test_set_id_scoping(self, clip_l_lora_file: str, clip_g_lora_file: str):
        """LoRA data is correctly scoped by set_id."""
        # AC: @sdxl-clip-lora-loader ac-5
        loader = SDXLCLIPLoader()
        loader.load(clip_l_lora_file, set_id="set_a")
        loader.load(clip_g_lora_file, set_id="set_b")

        # affected_keys should have both
        all_keys = loader.affected_keys
        clip_l_keys = [k for k in all_keys if k.startswith("clip_l.")]
        clip_g_keys = [k for k in all_keys if k.startswith("clip_g.")]
        assert len(clip_l_keys) > 0
        assert len(clip_g_keys) > 0

        # affected_keys_for_set should scope correctly
        set_a_keys = loader.affected_keys_for_set("set_a")
        set_b_keys = loader.affected_keys_for_set("set_b")

        assert all(k.startswith("clip_l.") for k in set_a_keys)
        assert all(k.startswith("clip_g.") for k in set_b_keys)

        loader.cleanup()


# ---------------------------------------------------------------------------
# AC-6: Only extracts text encoder keys (ignores UNet)
# ---------------------------------------------------------------------------


class TestAC6IgnoresUNetKeys:
    """AC-6: Only text encoder keys are extracted, UNet keys are ignored."""

    def test_mixed_lora_only_extracts_clip_keys(self, mixed_unet_clip_lora_file: str):
        """LoRA with both UNet and CLIP keys only extracts CLIP keys."""
        # AC: @sdxl-clip-lora-loader ac-6
        loader = SDXLCLIPLoader()
        loader.load(mixed_unet_clip_lora_file)

        affected = loader.affected_keys
        assert len(affected) > 0, "Should have affected keys"

        # All keys should be CLIP keys
        for key in affected:
            assert key.startswith("clip_l.") or key.startswith("clip_g."), (
                f"Key {key} should be a CLIP key, not UNet"
            )

        # No diffusion_model keys
        diffusion_keys = [k for k in affected if k.startswith("diffusion_model.")]
        assert len(diffusion_keys) == 0, f"Should not have UNet keys: {diffusion_keys}"

        loader.cleanup()

    def test_mixed_lora_correct_key_count(self, mixed_unet_clip_lora_file: str):
        """Mixed LoRA extracts expected number of CLIP keys."""
        # AC: @sdxl-clip-lora-loader ac-6
        loader = SDXLCLIPLoader()
        loader.load(mixed_unet_clip_lora_file)

        affected = loader.affected_keys
        # Fixture has 1 UNet key pair (ignored), 1 CLIP-L key pair, 1 CLIP-G key pair
        assert len(affected) == 2, f"Expected 2 keys, got {len(affected)}: {affected}"

        clip_l_keys = [k for k in affected if k.startswith("clip_l.")]
        clip_g_keys = [k for k in affected if k.startswith("clip_g.")]
        assert len(clip_l_keys) == 1
        assert len(clip_g_keys) == 1

        loader.cleanup()


# ---------------------------------------------------------------------------
# AC-7: Registered as get_loader(arch="sdxl", domain="clip")
# ---------------------------------------------------------------------------


class TestAC7RegistryIntegration:
    """AC-7: SDXLCLIPLoader accessible via get_loader(arch='sdxl', domain='clip')."""

    def test_sdxl_clip_in_registry(self):
        """sdxl_clip key exists in LOADER_REGISTRY."""
        # AC: @sdxl-clip-lora-loader ac-7
        assert "sdxl_clip" in LOADER_REGISTRY
        assert LOADER_REGISTRY["sdxl_clip"] is SDXLCLIPLoader

    def test_get_loader_with_clip_domain(self):
        """get_loader(arch='sdxl', domain='clip') returns SDXLCLIPLoader."""
        # AC: @sdxl-clip-lora-loader ac-7
        loader = get_loader("sdxl", domain="clip")
        assert isinstance(loader, SDXLCLIPLoader)

    def test_get_loader_sdxl_diffusion_still_works(self):
        """get_loader('sdxl') still returns SDXLLoader (not CLIP)."""
        # AC: @sdxl-clip-lora-loader ac-7
        from lib.lora import SDXLLoader

        loader = get_loader("sdxl")
        assert isinstance(loader, SDXLLoader)
        assert not isinstance(loader, SDXLCLIPLoader)

    def test_get_loader_sdxl_diffusion_explicit(self):
        """get_loader(arch='sdxl', domain='diffusion') returns SDXLLoader."""
        # AC: @sdxl-clip-lora-loader ac-7
        from lib.lora import SDXLLoader

        loader = get_loader("sdxl", domain="diffusion")
        assert isinstance(loader, SDXLLoader)
        assert not isinstance(loader, SDXLCLIPLoader)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for the SDXL CLIP loader system."""

    def test_full_workflow_clip_l(self, clip_l_lora_file: str):
        """Full workflow: get loader, load, get specs, cleanup for CLIP-L."""
        # Get CLIP-specific loader
        loader = get_loader("sdxl", domain="clip")

        # Load LoRA file
        loader.load(clip_l_lora_file, strength=0.75)

        # Check affected keys
        affected = loader.affected_keys
        assert len(affected) > 0
        assert all(k.startswith("clip_l.transformer.") for k in affected)

        # Get delta specs for batched execution
        keys = list(affected)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        # Verify specs are executor-compatible
        assert all(isinstance(s, DeltaSpec) for s in specs)
        assert all(s.kind == "standard" for s in specs)

        # Cleanup
        loader.cleanup()
        assert len(loader.affected_keys) == 0

    def test_full_workflow_clip_g(self, clip_g_lora_file: str):
        """Full workflow: get loader, load, get specs, cleanup for CLIP-G."""
        loader = get_loader("sdxl", domain="clip")
        loader.load(clip_g_lora_file, strength=0.6)

        affected = loader.affected_keys
        assert len(affected) > 0
        assert all(k.startswith("clip_g.transformer.") for k in affected)

        keys = list(affected)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        assert all(isinstance(s, DeltaSpec) for s in specs)
        assert all(s.kind == "standard" for s in specs)

        loader.cleanup()

    def test_full_workflow_mixed_clip(self, mixed_clip_lora_file: str):
        """Full workflow for LoRA with both CLIP-L and CLIP-G keys."""
        loader = get_loader("sdxl", domain="clip")
        loader.load(mixed_clip_lora_file)

        affected = loader.affected_keys
        assert len(affected) == 2  # One from each encoder

        clip_l_keys = [k for k in affected if k.startswith("clip_l.")]
        clip_g_keys = [k for k in affected if k.startswith("clip_g.")]
        assert len(clip_l_keys) == 1
        assert len(clip_g_keys) == 1

        keys = list(affected)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        assert len(specs) == 2

        loader.cleanup()

    def test_multiple_loads_accumulate(self, clip_l_lora_file: str):
        """Multiple load() calls accumulate LoRA data."""
        loader = SDXLCLIPLoader()
        loader.load(clip_l_lora_file, strength=0.5)
        loader.load(clip_l_lora_file, strength=0.3)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        # Should have 2 specs per key (loaded twice)
        specs_per_key = len(specs) / len(keys)
        assert specs_per_key == 2, f"Expected 2 specs per key, got {specs_per_key}"

        loader.cleanup()
