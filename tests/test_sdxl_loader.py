"""Tests for SDXL-specific LoRA loader.

Covers all 3 acceptance criteria for @sdxl-loader:
- AC-1: LoRA keys mapped to diffusion_model input_blocks, middle_block, output_blocks
- AC-2: DeltaSpecs contain correct rank, kind, and factor tensors
- AC-3: Attention keys (proj_in, proj_out, to_q/to_k/to_v) correctly mapped
"""

import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lib.lora.sdxl import SDXLLoader, _parse_lora_key, _tokenize_lora_path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sdxl_lora_with_all_block_types() -> str:
    """Create a LoRA file with input_blocks, middle_block, and output_blocks keys."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tensors = {
            # AC-1: input_blocks
            "lora_unet_input_blocks_0_0_conv.lora_up.weight": torch.randn(64, 8),
            "lora_unet_input_blocks_0_0_conv.lora_down.weight": torch.randn(8, 32),
            # AC-1: middle_block
            "lora_unet_middle_block_1_proj.lora_up.weight": torch.randn(128, 16),
            "lora_unet_middle_block_1_proj.lora_down.weight": torch.randn(16, 64),
            # AC-1: output_blocks
            "lora_unet_output_blocks_3_0_conv.lora_up.weight": torch.randn(256, 32),
            "lora_unet_output_blocks_3_0_conv.lora_down.weight": torch.randn(32, 128),
        }
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def sdxl_lora_with_attention_keys() -> str:
    """Create a LoRA file with attention-related keys (AC-3)."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        # Attention key base paths
        attn_base = "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1"
        mid_attn = "lora_unet_middle_block_1_transformer_blocks_0_attn1"

        tensors = {
            # proj_in
            "lora_unet_input_blocks_4_1_proj_in.lora_up.weight": torch.randn(320, 8),
            "lora_unet_input_blocks_4_1_proj_in.lora_down.weight": torch.randn(8, 320),
            # proj_out
            "lora_unet_output_blocks_5_0_proj_out.lora_up.weight": torch.randn(320, 8),
            "lora_unet_output_blocks_5_0_proj_out.lora_down.weight": torch.randn(8, 320),
            # to_q, to_k, to_v (attention)
            f"{attn_base}_to_q.lora_up.weight": torch.randn(320, 16),
            f"{attn_base}_to_q.lora_down.weight": torch.randn(16, 320),
            f"{attn_base}_to_k.lora_up.weight": torch.randn(320, 16),
            f"{attn_base}_to_k.lora_down.weight": torch.randn(16, 320),
            f"{attn_base}_to_v.lora_up.weight": torch.randn(320, 16),
            f"{attn_base}_to_v.lora_down.weight": torch.randn(16, 320),
            # to_out (attention output projection)
            f"{mid_attn}_to_out_0.lora_up.weight": torch.randn(320, 16),
            f"{mid_attn}_to_out_0.lora_down.weight": torch.randn(16, 320),
        }
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def sdxl_lora_with_alpha() -> str:
    """Create a LoRA file with alpha metadata."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        # Rank 8, alpha would typically be stored as metadata
        tensors = {
            "lora_unet_input_blocks_0_0_proj.lora_up.weight": torch.randn(64, 8),
            "lora_unet_input_blocks_0_0_proj.lora_down.weight": torch.randn(8, 32),
        }
        save_file(tensors, f.name)
        return f.name


# ---------------------------------------------------------------------------
# AC-1: Keys mapped to input_blocks, middle_block, output_blocks
# ---------------------------------------------------------------------------


class TestAC1BlockTypeMapping:
    """AC-1: LoRA keys are mapped to diffusion_model input_blocks, middle_block, output_blocks."""

    def test_input_blocks_key_mapping(self):
        """input_blocks LoRA keys map to diffusion_model.input_blocks.*"""
        # AC: @sdxl-loader ac-1
        result, _, _ = _parse_lora_key(
            "lora_unet_input_blocks_0_0_conv.lora_up.weight"
        )
        assert result == "diffusion_model.input_blocks.0.0.conv.weight"

    def test_middle_block_key_mapping(self):
        """middle_block LoRA keys map to diffusion_model.middle_block.*"""
        # AC: @sdxl-loader ac-1
        result, _, _ = _parse_lora_key(
            "lora_unet_middle_block_1_proj.lora_up.weight"
        )
        assert result == "diffusion_model.middle_block.1.proj.weight"

    def test_output_blocks_key_mapping(self):
        """output_blocks LoRA keys map to diffusion_model.output_blocks.*"""
        # AC: @sdxl-loader ac-1
        result, _, _ = _parse_lora_key(
            "lora_unet_output_blocks_3_0_conv.lora_up.weight"
        )
        assert result == "diffusion_model.output_blocks.3.0.conv.weight"

    def test_loader_produces_all_block_types(
        self, sdxl_lora_with_all_block_types: str
    ):
        """SDXLLoader produces keys for all block types."""
        # AC: @sdxl-loader ac-1
        loader = SDXLLoader()
        loader.load(sdxl_lora_with_all_block_types)

        affected = loader.affected_keys

        # Check all block types are present
        input_keys = [k for k in affected if ".input_blocks." in k]
        middle_keys = [k for k in affected if ".middle_block." in k]
        output_keys = [k for k in affected if ".output_blocks." in k]

        assert len(input_keys) == 1, f"Expected 1 input_blocks key, got {input_keys}"
        assert len(middle_keys) == 1, f"Expected 1 middle_block key, got {middle_keys}"
        assert len(output_keys) == 1, f"Expected 1 output_blocks key, got {output_keys}"

        loader.cleanup()

        # Cleanup temp file
        Path(sdxl_lora_with_all_block_types).unlink(missing_ok=True)

    def test_nested_indices_preserved(self):
        """Numeric indices in block paths are preserved correctly."""
        # AC: @sdxl-loader ac-1
        result, _, _ = _parse_lora_key(
            "lora_unet_input_blocks_7_1_transformer_blocks_9_attn2_proj.lora_down.weight"
        )
        assert result == "diffusion_model.input_blocks.7.1.transformer_blocks.9.attn2.proj.weight"


# ---------------------------------------------------------------------------
# AC-2: DeltaSpecs contain rank, kind, and factor tensors
# ---------------------------------------------------------------------------


class TestAC2DeltaSpecContents:
    """AC-2: DeltaSpecs contain correct rank, kind, and factor tensors."""

    def test_deltaspec_has_kind_field(self, sdxl_lora_with_all_block_types: str):
        """DeltaSpec objects have 'kind' field."""
        # AC: @sdxl-loader ac-2
        loader = SDXLLoader()
        loader.load(sdxl_lora_with_all_block_types)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        for spec in specs:
            assert hasattr(spec, "kind")
            assert spec.kind == "standard"

        loader.cleanup()
        Path(sdxl_lora_with_all_block_types).unlink(missing_ok=True)

    def test_deltaspec_has_up_down_factors(self, sdxl_lora_with_all_block_types: str):
        """DeltaSpec objects have up and down factor tensors."""
        # AC: @sdxl-loader ac-2
        loader = SDXLLoader()
        loader.load(sdxl_lora_with_all_block_types)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        for spec in specs:
            assert spec.up is not None, "DeltaSpec should have 'up' tensor"
            assert spec.down is not None, "DeltaSpec should have 'down' tensor"
            assert isinstance(spec.up, torch.Tensor)
            assert isinstance(spec.down, torch.Tensor)

        loader.cleanup()
        Path(sdxl_lora_with_all_block_types).unlink(missing_ok=True)

    def test_deltaspec_rank_matches_tensors(self, sdxl_lora_with_all_block_types: str):
        """DeltaSpec up/down tensors have matching rank dimension."""
        # AC: @sdxl-loader ac-2
        loader = SDXLLoader()
        loader.load(sdxl_lora_with_all_block_types)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        for spec in specs:
            # For standard LoRA: up is (out, rank), down is (rank, in)
            rank_from_up = spec.up.shape[1]
            rank_from_down = spec.down.shape[0]
            assert rank_from_up == rank_from_down, (
                f"Rank mismatch: up.shape[1]={rank_from_up}, down.shape[0]={rank_from_down}"
            )

        loader.cleanup()
        Path(sdxl_lora_with_all_block_types).unlink(missing_ok=True)

    def test_deltaspec_scale_computed_correctly(self, sdxl_lora_with_alpha: str):
        """DeltaSpec scale is computed as strength * alpha / rank."""
        # AC: @sdxl-loader ac-2
        loader = SDXLLoader()
        loader.load(sdxl_lora_with_alpha, strength=0.5)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        # Default alpha = rank, so scale = strength * rank / rank = strength
        for spec in specs:
            assert abs(spec.scale - 0.5) < 1e-6, f"Expected scale=0.5, got {spec.scale}"

        loader.cleanup()
        Path(sdxl_lora_with_alpha).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# AC-3: Attention keys correctly mapped
# ---------------------------------------------------------------------------


class TestAC3AttentionKeyMapping:
    """AC-3: Attention keys (proj_in, proj_out, to_q/to_k/to_v) correctly mapped."""

    def test_proj_in_key_mapping(self):
        """proj_in LoRA keys map correctly."""
        # AC: @sdxl-loader ac-3
        result, _, _ = _parse_lora_key(
            "lora_unet_input_blocks_4_1_proj_in.lora_up.weight"
        )
        assert result == "diffusion_model.input_blocks.4.1.proj_in.weight"

    def test_proj_out_key_mapping(self):
        """proj_out LoRA keys map correctly."""
        # AC: @sdxl-loader ac-3
        result, _, _ = _parse_lora_key(
            "lora_unet_output_blocks_5_0_proj_out.lora_down.weight"
        )
        assert result == "diffusion_model.output_blocks.5.0.proj_out.weight"

    def test_to_q_key_mapping(self):
        """to_q attention LoRA keys map correctly."""
        # AC: @sdxl-loader ac-3
        result, _, _ = _parse_lora_key(
            "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.lora_up.weight"
        )
        assert result == "diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight"

    def test_to_k_key_mapping(self):
        """to_k attention LoRA keys map correctly."""
        # AC: @sdxl-loader ac-3
        result, _, _ = _parse_lora_key(
            "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k.lora_up.weight"
        )
        assert result == "diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_k.weight"

    def test_to_v_key_mapping(self):
        """to_v attention LoRA keys map correctly."""
        # AC: @sdxl-loader ac-3
        result, _, _ = _parse_lora_key(
            "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_v.lora_up.weight"
        )
        assert result == "diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_v.weight"

    def test_to_out_key_mapping(self):
        """to_out attention output projection maps correctly."""
        # AC: @sdxl-loader ac-3
        lora_key = "lora_unet_middle_block_1_transformer_blocks_0_attn1_to_out_0.lora_up.weight"
        result, _, _ = _parse_lora_key(lora_key)
        expected = "diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight"
        assert result == expected

    def test_loader_handles_all_attention_keys(
        self, sdxl_lora_with_attention_keys: str
    ):
        """SDXLLoader correctly loads all attention-related keys."""
        # AC: @sdxl-loader ac-3
        loader = SDXLLoader()
        loader.load(sdxl_lora_with_attention_keys)

        affected = loader.affected_keys

        # Check for all attention key types
        proj_in_keys = [k for k in affected if ".proj_in." in k]
        proj_out_keys = [k for k in affected if ".proj_out." in k]
        to_q_keys = [k for k in affected if ".to_q." in k]
        to_k_keys = [k for k in affected if ".to_k." in k]
        to_v_keys = [k for k in affected if ".to_v." in k]
        to_out_keys = [k for k in affected if ".to_out." in k]

        assert len(proj_in_keys) == 1, f"Expected 1 proj_in key, got {proj_in_keys}"
        assert len(proj_out_keys) == 1, f"Expected 1 proj_out key, got {proj_out_keys}"
        assert len(to_q_keys) == 1, f"Expected 1 to_q key, got {to_q_keys}"
        assert len(to_k_keys) == 1, f"Expected 1 to_k key, got {to_k_keys}"
        assert len(to_v_keys) == 1, f"Expected 1 to_v key, got {to_v_keys}"
        assert len(to_out_keys) == 1, f"Expected 1 to_out key, got {to_out_keys}"

        loader.cleanup()
        Path(sdxl_lora_with_attention_keys).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tokenizer unit tests
# ---------------------------------------------------------------------------


class TestTokenizer:
    """Tests for the _tokenize_lora_path helper."""

    def test_simple_path(self):
        """Simple paths tokenize correctly."""
        tokens = _tokenize_lora_path("input_blocks_0_0_conv")
        assert tokens == ["input_blocks", "0", "0", "conv"]

    def test_attention_path(self):
        """Attention paths with to_q/to_k/to_v tokenize correctly."""
        tokens = _tokenize_lora_path("input_blocks_4_1_transformer_blocks_0_attn1_to_q")
        assert tokens == ["input_blocks", "4", "1", "transformer_blocks", "0", "attn1", "to_q"]

    def test_proj_in_out_path(self):
        """proj_in/proj_out paths tokenize correctly."""
        tokens = _tokenize_lora_path("output_blocks_5_0_proj_out")
        assert tokens == ["output_blocks", "5", "0", "proj_out"]

    def test_middle_block_path(self):
        """middle_block paths tokenize correctly."""
        tokens = _tokenize_lora_path("middle_block_1_proj")
        assert tokens == ["middle_block", "1", "proj"]

    def test_to_out_with_index(self):
        """to_out paths with numeric suffix tokenize correctly."""
        tokens = _tokenize_lora_path("middle_block_1_transformer_blocks_0_attn1_to_out_0")
        assert tokens == ["middle_block", "1", "transformer_blocks", "0", "attn1", "to_out", "0"]
