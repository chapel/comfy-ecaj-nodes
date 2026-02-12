"""Tests for Z-Image LoRA Loader - @zimage-loader spec.

This tests the Z-Image architecture-specific LoRA loader which handles:
- QKV fusing from separate to_q/to_k/to_v keys to fused attention.qkv
- Diffusers-style key name mapping to S3-DiT parameter names
- Offset indexing for QKV slices in the fused weight

AC-1: QKV keys are fused into the base model attention.qkv.weight layout
AC-2: Diffusers-style key names are correctly mapped to S3-DiT parameter names
AC-3: QKV-fused specs have correct offset indexing
"""

import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lib.lora.zimage import (
    _QKV_OFFSETS,
    _ZIMAGE_HIDDEN_DIM,
    ZImageLoader,
    _normalize_lycoris_key,
    _parse_zimage_lora_key,
)

# ---------------------------------------------------------------------------
# AC-1: QKV keys fused into attention.qkv.weight layout
# ---------------------------------------------------------------------------


class TestAC1QKVFusing:
    """AC-1: QKV keys are fused into the base model attention.qkv.weight layout."""

    @pytest.fixture
    def qkv_lora_file(self) -> str:
        """Create a LoRA file with separate Q/K/V components."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            # Create LoRA tensors for Q, K, V
            rank = 8
            in_dim = 64
            out_dim = 32
            tensors = {
                # Q component
                "transformer.layers.0.attention.to_q.lora_A.weight": torch.randn(
                    rank, in_dim
                ),
                "transformer.layers.0.attention.to_q.lora_B.weight": torch.randn(
                    out_dim, rank
                ),
                # K component
                "transformer.layers.0.attention.to_k.lora_A.weight": torch.randn(
                    rank, in_dim
                ),
                "transformer.layers.0.attention.to_k.lora_B.weight": torch.randn(
                    out_dim, rank
                ),
                # V component
                "transformer.layers.0.attention.to_v.lora_A.weight": torch.randn(
                    rank, in_dim
                ),
                "transformer.layers.0.attention.to_v.lora_B.weight": torch.randn(
                    out_dim, rank
                ),
            }
            save_file(tensors, f.name)
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    def test_separate_qkv_keys_fuse_to_single_qkv_weight(self, qkv_lora_file: str):
        """Separate to_q, to_k, to_v LoRA keys map to single qkv.weight key."""
        # AC: @zimage-loader ac-1
        loader = ZImageLoader()
        loader.load(qkv_lora_file)

        affected = loader.affected_keys
        # Should have single fused qkv key (not three separate keys)
        assert len(affected) == 1
        key = next(iter(affected))
        assert ".qkv." in key
        assert "diffusion_model.layers.0.attention.qkv.weight" == key

        loader.cleanup()

    def test_qkv_produces_three_deltaspec_components(self, qkv_lora_file: str):
        """Fused QKV key produces three DeltaSpecs (qkv_q, qkv_k, qkv_v)."""
        # AC: @zimage-loader ac-1
        loader = ZImageLoader()
        loader.load(qkv_lora_file)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        # Should have exactly 3 specs for q, k, v
        assert len(specs) == 3

        kinds = {s.kind for s in specs}
        assert kinds == {"qkv_q", "qkv_k", "qkv_v"}

        loader.cleanup()

    def test_partial_qkv_only_includes_present_components(self):
        """If only some Q/K/V components exist, only those are included."""
        # AC: @zimage-loader ac-1
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            # Only Q and K, no V
            tensors = {
                "transformer.layers.0.attention.to_q.lora_A.weight": torch.randn(8, 64),
                "transformer.layers.0.attention.to_q.lora_B.weight": torch.randn(32, 8),
                "transformer.layers.0.attention.to_k.lora_A.weight": torch.randn(8, 64),
                "transformer.layers.0.attention.to_k.lora_B.weight": torch.randn(32, 8),
            }
            save_file(tensors, f.name)
            path = f.name

        try:
            loader = ZImageLoader()
            loader.load(path)

            keys = list(loader.affected_keys)
            key_indices = {k: i for i, k in enumerate(keys)}
            specs = loader.get_delta_specs(keys, key_indices)

            kinds = {s.kind for s in specs}
            assert kinds == {"qkv_q", "qkv_k"}  # No qkv_v
            assert "qkv_v" not in kinds

            loader.cleanup()
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# AC-2: Diffusers-style key name mapping to S3-DiT parameter names
# ---------------------------------------------------------------------------


class TestAC2DiffusersKeyMapping:
    """AC-2: Diffusers-style key names are correctly mapped to S3-DiT parameter names."""

    def test_transformer_prefix_stripped(self):
        """transformer. prefix is stripped and mapped correctly."""
        # AC: @zimage-loader ac-2
        key = "transformer.layers.5.attention.to_q.lora_A.weight"
        model_key, direction, qkv = _parse_zimage_lora_key(key)

        assert model_key == "diffusion_model.layers.5.attention.qkv.weight"
        assert direction == "down"
        assert qkv == "q"

    def test_diffusion_model_prefix_handled(self):
        """diffusion_model. prefix is handled correctly."""
        # AC: @zimage-loader ac-2
        key = "diffusion_model.layers.0.attention.to_k.lora_B.weight"
        model_key, direction, qkv = _parse_zimage_lora_key(key)

        assert model_key == "diffusion_model.layers.0.attention.qkv.weight"
        assert direction == "up"
        assert qkv == "k"

    def test_to_out_mapped_to_out(self):
        """attention.to_out.0 maps to attention.out."""
        # AC: @zimage-loader ac-2
        key = "transformer.layers.2.attention.to_out.0.lora_A.weight"
        model_key, direction, qkv = _parse_zimage_lora_key(key)

        assert model_key == "diffusion_model.layers.2.attention.out.weight"
        assert direction == "down"
        assert qkv is None  # Not a QKV component

    def test_feed_forward_keys_mapped(self):
        """Feed-forward layer keys are mapped correctly."""
        # AC: @zimage-loader ac-2
        key = "transformer.layers.1.ff.linear_1.lora_B.weight"
        model_key, direction, qkv = _parse_zimage_lora_key(key)

        assert model_key == "diffusion_model.layers.1.ff.linear_1.weight"
        assert direction == "up"
        assert qkv is None

    def test_lycoris_format_normalized(self):
        """LyCORIS underscore-separated format is normalized."""
        # AC: @zimage-loader ac-2
        # Test the normalization function directly
        key = "lycoris_layers_0_adaLN_modulation_0"
        normalized = _normalize_lycoris_key(key)
        assert normalized == "layers.0.adaLN_modulation.0"

    def test_lycoris_lora_key_parsed(self):
        """LyCORIS format LoRA keys are parsed correctly."""
        # AC: @zimage-loader ac-2
        key = "lycoris_layers_5_attention_to_q.lora_down.weight"
        model_key, direction, qkv = _parse_zimage_lora_key(key)

        assert model_key == "diffusion_model.layers.5.attention.qkv.weight"
        assert direction == "down"
        assert qkv == "q"

    def test_compound_names_preserved(self):
        """Compound names like feed_forward, noise_refiner are preserved."""
        # AC: @zimage-loader ac-2
        # These should not have internal underscores converted to dots
        key = "lycoris_layers_0_feed_forward_w1"
        normalized = _normalize_lycoris_key(key)
        assert "feed_forward" in normalized  # Not feed.forward
        assert normalized == "layers.0.feed_forward.w1"

    def test_lora_down_up_variants_handled(self):
        """Both lora_A/lora_B and lora_down/lora_up variants work."""
        # AC: @zimage-loader ac-2
        # lora_A/lora_B format
        key1 = "transformer.layers.0.attention.to_v.lora_A.weight"
        model_key1, dir1, _ = _parse_zimage_lora_key(key1)
        assert dir1 == "down"

        # lora_down/lora_up format
        key2 = "transformer.layers.0.attention.to_v.lora_down.weight"
        model_key2, dir2, _ = _parse_zimage_lora_key(key2)
        assert dir2 == "down"

        # Both should map to same model key
        assert model_key1 == model_key2

    def test_non_lora_keys_rejected(self):
        """Keys without LoRA suffix return None."""
        # AC: @zimage-loader ac-2
        key = "transformer.layers.0.attention.qkv.weight"  # Base model key
        model_key, direction, qkv = _parse_zimage_lora_key(key)

        assert model_key is None

    def test_multiple_layers_mapped_correctly(self):
        """Different layer indices are preserved in mapping."""
        # AC: @zimage-loader ac-2
        for layer_idx in [0, 5, 15, 29]:
            key = f"transformer.layers.{layer_idx}.attention.to_q.lora_A.weight"
            model_key, _, _ = _parse_zimage_lora_key(key)
            assert f"layers.{layer_idx}." in model_key


# ---------------------------------------------------------------------------
# AC-3: QKV offset indexing for fused weight
# ---------------------------------------------------------------------------


class TestAC3OffsetIndexing:
    """AC-3: QKV-fused specs have correct offset indexing for the fused weight."""

    def test_offset_constants_correct(self):
        """QKV offset constants match spec: q=0:3840, k=3840:7680, v=7680:11520."""
        # AC: @zimage-loader ac-3
        # Hidden dim is 3840
        assert _ZIMAGE_HIDDEN_DIM == 3840

        # Q: rows 0 to 3840
        assert _QKV_OFFSETS["q"] == (0, 3840)

        # K: rows 3840 to 7680
        assert _QKV_OFFSETS["k"] == (3840, 3840)

        # V: rows 7680 to 11520
        assert _QKV_OFFSETS["v"] == (7680, 3840)

    @pytest.fixture
    def qkv_lora_file(self) -> str:
        """Create a LoRA file with Q/K/V components."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            tensors = {
                "transformer.layers.0.attention.to_q.lora_A.weight": torch.randn(8, 64),
                "transformer.layers.0.attention.to_q.lora_B.weight": torch.randn(32, 8),
                "transformer.layers.0.attention.to_k.lora_A.weight": torch.randn(8, 64),
                "transformer.layers.0.attention.to_k.lora_B.weight": torch.randn(32, 8),
                "transformer.layers.0.attention.to_v.lora_A.weight": torch.randn(8, 64),
                "transformer.layers.0.attention.to_v.lora_B.weight": torch.randn(32, 8),
            }
            save_file(tensors, f.name)
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    def test_deltaspec_has_offset_for_qkv(self, qkv_lora_file: str):
        """QKV DeltaSpecs include offset field."""
        # AC: @zimage-loader ac-3
        loader = ZImageLoader()
        loader.load(qkv_lora_file)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        for spec in specs:
            assert spec.offset is not None, f"Spec {spec.kind} missing offset"
            assert isinstance(spec.offset, tuple)
            assert len(spec.offset) == 2  # (start, length)

        loader.cleanup()

    def test_q_offset_is_0_3840(self, qkv_lora_file: str):
        """Q component DeltaSpec has offset (0, 3840)."""
        # AC: @zimage-loader ac-3
        loader = ZImageLoader()
        loader.load(qkv_lora_file)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        q_specs = [s for s in specs if s.kind == "qkv_q"]
        assert len(q_specs) == 1
        assert q_specs[0].offset == (0, 3840)

        loader.cleanup()

    def test_k_offset_is_3840_3840(self, qkv_lora_file: str):
        """K component DeltaSpec has offset (3840, 3840)."""
        # AC: @zimage-loader ac-3
        loader = ZImageLoader()
        loader.load(qkv_lora_file)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        k_specs = [s for s in specs if s.kind == "qkv_k"]
        assert len(k_specs) == 1
        assert k_specs[0].offset == (3840, 3840)

        loader.cleanup()

    def test_v_offset_is_7680_3840(self, qkv_lora_file: str):
        """V component DeltaSpec has offset (7680, 3840)."""
        # AC: @zimage-loader ac-3
        loader = ZImageLoader()
        loader.load(qkv_lora_file)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        v_specs = [s for s in specs if s.kind == "qkv_v"]
        assert len(v_specs) == 1
        assert v_specs[0].offset == (7680, 3840)

        loader.cleanup()

    def test_scale_with_explicit_alpha(self):
        """DeltaSpec scale uses alpha from .alpha tensor when present."""
        # AC: @zimage-loader ac-3
        # Rank=8, alpha=4 → scale = strength * 4 / 8 = strength * 0.5
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            tensors = {
                "transformer.layers.0.attention.to_q.lora_A.weight": torch.randn(8, 64),
                "transformer.layers.0.attention.to_q.lora_B.weight": torch.randn(32, 8),
                "transformer.layers.0.attention.to_q.alpha": torch.tensor(4.0),
                "transformer.layers.0.attention.to_k.lora_A.weight": torch.randn(8, 64),
                "transformer.layers.0.attention.to_k.lora_B.weight": torch.randn(32, 8),
                "transformer.layers.0.attention.to_k.alpha": torch.tensor(4.0),
                "transformer.layers.0.attention.to_v.lora_A.weight": torch.randn(8, 64),
                "transformer.layers.0.attention.to_v.lora_B.weight": torch.randn(32, 8),
                "transformer.layers.0.attention.to_v.alpha": torch.tensor(4.0),
            }
            save_file(tensors, f.name)
            path = f.name

        try:
            loader = ZImageLoader()
            loader.load(path, strength=1.0)

            keys = list(loader.affected_keys)
            key_indices = {k: i for i, k in enumerate(keys)}
            specs = loader.get_delta_specs(keys, key_indices)

            # scale = 1.0 * 4 / 8 = 0.5
            assert len(specs) == 3
            for spec in specs:
                assert abs(spec.scale - 0.5) < 1e-6, f"Expected scale=0.5, got {spec.scale}"

            loader.cleanup()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_scale_default_alpha_equals_rank(self):
        """Without .alpha tensor, alpha defaults to rank (scale = strength)."""
        # AC: @zimage-loader ac-3
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            tensors = {
                "transformer.layers.0.attention.to_q.lora_A.weight": torch.randn(8, 64),
                "transformer.layers.0.attention.to_q.lora_B.weight": torch.randn(32, 8),
            }
            save_file(tensors, f.name)
            path = f.name

        try:
            loader = ZImageLoader()
            loader.load(path, strength=0.7)

            keys = list(loader.affected_keys)
            key_indices = {k: i for i, k in enumerate(keys)}
            specs = loader.get_delta_specs(keys, key_indices)

            # Default alpha = rank, so scale = strength = 0.7
            assert len(specs) == 1
            assert abs(specs[0].scale - 0.7) < 1e-6, f"Expected scale=0.7, got {specs[0].scale}"

            loader.cleanup()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_standard_specs_have_no_offset(self, qkv_lora_file: str):
        """Standard (non-QKV) DeltaSpecs have offset=None."""
        # AC: @zimage-loader ac-3
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            tensors = {
                "transformer.layers.0.ff.linear_1.lora_A.weight": torch.randn(8, 64),
                "transformer.layers.0.ff.linear_1.lora_B.weight": torch.randn(32, 8),
            }
            save_file(tensors, f.name)
            path = f.name

        try:
            loader = ZImageLoader()
            loader.load(path)

            keys = list(loader.affected_keys)
            key_indices = {k: i for i, k in enumerate(keys)}
            specs = loader.get_delta_specs(keys, key_indices)

            assert len(specs) == 1
            assert specs[0].kind == "standard"
            assert specs[0].offset is None

            loader.cleanup()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_offsets_cover_full_qkv_range(self):
        """Verify offsets cover exactly the 11520-row fused weight without gaps or overlaps."""
        # AC: @zimage-loader ac-3
        q_start, q_len = _QKV_OFFSETS["q"]
        k_start, k_len = _QKV_OFFSETS["k"]
        v_start, v_len = _QKV_OFFSETS["v"]

        # No gaps
        assert q_start + q_len == k_start
        assert k_start + k_len == v_start

        # Total coverage = 11520
        assert v_start + v_len == 11520

        # All same length (hidden_dim)
        assert q_len == k_len == v_len == 3840
