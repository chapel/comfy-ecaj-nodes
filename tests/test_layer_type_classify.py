"""Tests for Layer-Type Classification feature.

Tests for @layer-type-filter acceptance criteria:
- AC-1: classify_layer_type returns attention, feed_forward, norm, or None
- AC-6: Unmatched keys (time_embed, adaLN_modulation, embedders) return None
- AC-7: First-match-wins with precedence: attention > feed_forward > norm
- AC-8: arch=None or unsupported arch returns None
"""

from lib.block_classify import classify_layer_type

# =============================================================================
# SDXL Layer Type Classification Tests
# =============================================================================


class TestLayerTypeClassifySDXL:
    """SDXL layer type classification tests."""

    # AC: @layer-type-filter ac-1
    def test_attention_layers_attn1(self):
        """SDXL attn1 keys classify as attention."""
        key = "input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"
        assert classify_layer_type(key, "sdxl") == "attention"
        key = "input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight"
        assert classify_layer_type(key, "sdxl") == "attention"
        key = "input_blocks.1.1.transformer_blocks.0.attn1.to_v.weight"
        assert classify_layer_type(key, "sdxl") == "attention"

    # AC: @layer-type-filter ac-1
    def test_attention_layers_attn2(self):
        """SDXL attn2 keys classify as attention."""
        key = "input_blocks.1.1.transformer_blocks.0.attn2.to_q.weight"
        assert classify_layer_type(key, "sdxl") == "attention"
        key = "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"
        assert classify_layer_type(key, "sdxl") == "attention"
        key = "input_blocks.1.1.transformer_blocks.0.attn2.to_v.weight"
        assert classify_layer_type(key, "sdxl") == "attention"
        key = "input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.weight"
        assert classify_layer_type(key, "sdxl") == "attention"

    # AC: @layer-type-filter ac-1
    def test_attention_layers_proj(self):
        """SDXL proj_in/proj_out keys within blocks classify as attention."""
        assert classify_layer_type("input_blocks.4.1.proj_in.weight", "sdxl") == "attention"
        assert classify_layer_type("input_blocks.4.1.proj_out.weight", "sdxl") == "attention"

    # AC: @layer-type-filter ac-1
    def test_feed_forward_layers(self):
        """SDXL feed-forward keys classify as feed_forward."""
        key = "input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight"
        assert classify_layer_type(key, "sdxl") == "feed_forward"
        key = "input_blocks.4.1.transformer_blocks.0.ff.net.2.weight"
        assert classify_layer_type(key, "sdxl") == "feed_forward"

    # AC: @layer-type-filter ac-1
    def test_norm_layers(self):
        """SDXL normalization keys classify as norm."""
        # in_layers.0 is not norm-containing
        key = "input_blocks.4.0.in_layers.0.weight"
        assert classify_layer_type(key, "sdxl") is None
        # Keys that contain 'norm' pattern
        key = "input_blocks.4.1.transformer_blocks.0.norm1.weight"
        assert classify_layer_type(key, "sdxl") == "norm"
        key = "input_blocks.4.1.transformer_blocks.0.norm2.weight"
        assert classify_layer_type(key, "sdxl") == "norm"
        key = "middle_block.1.transformer_blocks.0.norm3.weight"
        assert classify_layer_type(key, "sdxl") == "norm"

    # AC: @layer-type-filter ac-6
    def test_unmatched_keys(self):
        """Unmatched SDXL keys return None."""
        assert classify_layer_type("time_embed.0.weight", "sdxl") is None
        assert classify_layer_type("label_emb.weight", "sdxl") is None
        assert classify_layer_type("out.0.weight", "sdxl") is None

    # AC: @layer-type-filter ac-1
    def test_strips_diffusion_model_prefix(self):
        """Layer type classification strips diffusion_model. prefix."""
        key = "diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"
        assert classify_layer_type(key, "sdxl") == "attention"


# =============================================================================
# Z-Image Layer Type Classification Tests
# =============================================================================


class TestLayerTypeClassifyZImage:
    """Z-Image/S3-DiT layer type classification tests."""

    # AC: @layer-type-filter ac-1
    def test_attention_layers_qkv(self):
        """Z-Image attn.qkv keys classify as attention."""
        assert classify_layer_type("layers.5.attn.qkv.weight", "zimage") == "attention"

    # AC: @layer-type-filter ac-1
    def test_attention_layers_out(self):
        """Z-Image attn.out keys classify as attention."""
        assert classify_layer_type("layers.5.attn.out.weight", "zimage") == "attention"

    # AC: @layer-type-filter ac-7
    def test_attention_precedence_q_norm(self):
        """q_norm classifies as attention (attention > norm precedence)."""
        assert classify_layer_type("layers.5.q_norm.weight", "zimage") == "attention"

    # AC: @layer-type-filter ac-7
    def test_attention_precedence_k_norm(self):
        """k_norm classifies as attention (attention > norm precedence)."""
        assert classify_layer_type("layers.5.k_norm.weight", "zimage") == "attention"

    # AC: @layer-type-filter ac-1
    def test_feed_forward_layers(self):
        """Z-Image feed-forward keys classify as feed_forward."""
        assert classify_layer_type("layers.5.feed_forward.w1.weight", "zimage") == "feed_forward"
        assert classify_layer_type("layers.5.feed_forward.w2.weight", "zimage") == "feed_forward"
        assert classify_layer_type("layers.5.feed_forward.w3.weight", "zimage") == "feed_forward"
        assert classify_layer_type("layers.5.mlp.fc1.weight", "zimage") == "feed_forward"
        assert classify_layer_type("layers.5.mlp.fc2.weight", "zimage") == "feed_forward"

    # AC: @layer-type-filter ac-1
    def test_norm_layers(self):
        """Z-Image normalization keys classify as norm."""
        assert classify_layer_type("layers.5.norm.weight", "zimage") == "norm"
        assert classify_layer_type("layers.5.ln.weight", "zimage") == "norm"
        assert classify_layer_type("layers.5.rms.weight", "zimage") == "norm"

    # AC: @layer-type-filter ac-6
    def test_adaLN_modulation_returns_none(self):
        """adaLN_modulation is NOT a norm layer (conditioning projection)."""
        assert classify_layer_type("layers.5.adaLN_modulation.1.weight", "zimage") is None

    # AC: @layer-type-filter ac-6
    def test_unmatched_keys(self):
        """Unmatched Z-Image keys return None."""
        assert classify_layer_type("patch_embed.weight", "zimage") is None
        assert classify_layer_type("final_layer.weight", "zimage") is None

    # AC: @layer-type-filter ac-6
    def test_embedders_returns_none(self):
        """Embedders keys return None."""
        assert classify_layer_type("embedders.text.proj.weight", "zimage") is None

    # AC: @layer-type-filter ac-1
    def test_strips_transformer_prefix(self):
        """Layer type classification strips transformer. prefix."""
        key = "transformer.layers.5.attn.qkv.weight"
        assert classify_layer_type(key, "zimage") == "attention"


# =============================================================================
# Qwen Layer Type Classification Tests
# =============================================================================


class TestLayerTypeClassifyQwen:
    """Qwen layer type classification tests."""

    # AC: @qwen-detect-classify ac-3
    def test_attention_layers_attn(self):
        """Qwen .attn. keys classify as attention."""
        assert classify_layer_type("transformer_blocks.5.attn.weight", "qwen") == "attention"
        key = "transformer_blocks.5.attn.to_q.weight"
        assert classify_layer_type(key, "qwen") == "attention"

    # AC: @qwen-detect-classify ac-3
    def test_attention_layers_qkv(self):
        """Qwen qkv keys classify as attention."""
        assert classify_layer_type("transformer_blocks.5.qkv.weight", "qwen") == "attention"

    # AC: @qwen-detect-classify ac-3
    def test_attention_layers_proj(self):
        """Qwen proj keys classify as attention."""
        assert classify_layer_type("transformer_blocks.5.proj.weight", "qwen") == "attention"

    # AC: @qwen-detect-classify ac-3
    def test_attention_layers_to_kv(self):
        """Qwen to_q/to_k/to_v/to_out keys classify as attention."""
        assert classify_layer_type("transformer_blocks.5.to_q.weight", "qwen") == "attention"
        assert classify_layer_type("transformer_blocks.5.to_k.weight", "qwen") == "attention"
        assert classify_layer_type("transformer_blocks.5.to_v.weight", "qwen") == "attention"
        assert classify_layer_type("transformer_blocks.5.to_out.weight", "qwen") == "attention"

    # AC: @qwen-detect-classify ac-3
    def test_feed_forward_layers_mlp(self):
        """Qwen mlp keys classify as feed_forward."""
        assert classify_layer_type("transformer_blocks.5.mlp.fc1.weight", "qwen") == "feed_forward"
        assert classify_layer_type("transformer_blocks.5.mlp.fc2.weight", "qwen") == "feed_forward"

    # AC: @qwen-detect-classify ac-3
    def test_feed_forward_layers_ff(self):
        """Qwen ff keys classify as feed_forward."""
        key = "transformer_blocks.5.ff.net.0.weight"
        assert classify_layer_type(key, "qwen") == "feed_forward"

    # AC: @qwen-detect-classify ac-3
    def test_feed_forward_layers_proj(self):
        """Qwen gate_proj/up_proj/down_proj keys classify as feed_forward."""
        key = "transformer_blocks.5.gate_proj.weight"
        assert classify_layer_type(key, "qwen") == "feed_forward"
        assert classify_layer_type("transformer_blocks.5.up_proj.weight", "qwen") == "feed_forward"
        key = "transformer_blocks.5.down_proj.weight"
        assert classify_layer_type(key, "qwen") == "feed_forward"

    # AC: @qwen-detect-classify ac-3
    def test_norm_layers(self):
        """Qwen normalization keys classify as norm."""
        assert classify_layer_type("transformer_blocks.5.norm.weight", "qwen") == "norm"
        assert classify_layer_type("transformer_blocks.5.ln.weight", "qwen") == "norm"
        assert classify_layer_type("transformer_blocks.5.layer_norm.weight", "qwen") == "norm"

    # AC: @qwen-detect-classify ac-3
    def test_mod_layers(self):
        """Qwen img_mod/txt_mod keys classify as norm."""
        assert classify_layer_type("transformer_blocks.5.img_mod.weight", "qwen") == "norm"
        assert classify_layer_type("transformer_blocks.5.txt_mod.weight", "qwen") == "norm"

    # AC: @qwen-detect-classify ac-3
    def test_strips_prefixes(self):
        """Layer type classification strips common prefixes."""
        key = "diffusion_model.transformer_blocks.5.attn.weight"
        assert classify_layer_type(key, "qwen") == "attention"
        key = "transformer.transformer_blocks.5.mlp.fc1.weight"
        assert classify_layer_type(key, "qwen") == "feed_forward"


# =============================================================================
# Architecture Edge Cases
# =============================================================================


class TestLayerTypeClassifyArchEdgeCases:
    """Edge cases for layer type classification."""

    # AC: @layer-type-filter ac-8
    def test_none_arch_returns_none(self):
        """arch=None returns None."""
        assert classify_layer_type("layers.5.attn.qkv.weight", None) is None
        key = "input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"
        assert classify_layer_type(key, None) is None

    # AC: @layer-type-filter ac-8
    def test_unsupported_arch_returns_none(self):
        """Unsupported architectures return None."""
        assert classify_layer_type("layers.5.attn.qkv.weight", "flux") is None
        assert classify_layer_type("input_blocks.0.0.weight", "unknown") is None


# =============================================================================
# Precedence Tests (AC-7)
# =============================================================================


class TestLayerTypePrecedence:
    """Test first-match-wins precedence: attention > feed_forward > norm."""

    # AC: @layer-type-filter ac-7
    def test_attention_beats_norm_sdxl(self):
        """In SDXL, keys containing both attention and norm patterns use attention."""
        # attn patterns should win over norm patterns
        key = "input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"
        # Contains 'attn1' (attention) and could theoretically contain norm-like suffix
        # but attention patterns come first
        assert classify_layer_type(key, "sdxl") == "attention"

    # AC: @layer-type-filter ac-7
    def test_attention_beats_norm_zimage(self):
        """In Z-Image, q_norm/k_norm are attention (attention > norm)."""
        # q_norm contains both q_ (attention) and norm patterns
        assert classify_layer_type("layers.5.q_norm.weight", "zimage") == "attention"
        assert classify_layer_type("layers.5.k_norm.weight", "zimage") == "attention"

    # AC: @layer-type-filter ac-7
    def test_ff_beats_norm_zimage(self):
        """In Z-Image, ff layers with norm suffix still classify as feed_forward."""
        # This tests precedence order in the pattern list
        key = "layers.5.feed_forward.w1.weight"
        assert classify_layer_type(key, "zimage") == "feed_forward"
