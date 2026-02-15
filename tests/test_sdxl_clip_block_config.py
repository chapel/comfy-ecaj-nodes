"""Tests for SDXL CLIP Block Config — per-block strength control for SDXL text encoders.

Tests for @sdxl-clip-block-config acceptance criteria:
- AC-1: Node has sliders for CLIP-L (CL00-CL11), CLIP-G (CG00-CG31), structural keys
- AC-2: Slider at 0.0 preserves base (no merge for that block)
- AC-3: Slider at 2.0 doubles merge strength
- AC-4: BlockConfig.arch is sdxl and block_overrides contains per-block values
- AC-5: SDXL CLIP block config connected to CLIP Merge applies per-block control
- AC-6: No block config connected means uniform strength
- AC-7: classify_key returns "CL05" for CLIP-L layer 5
- AC-8: classify_key returns "CG20" for CLIP-G layer 20
- AC-9: classify_key returns "CL_EMBED" for CLIP-L embeddings
- AC-10: classify_layer_type returns correct layer type for CLIP keys
"""

from lib.block_classify import (
    classify_key,
    classify_key_sdxl_clip,
    classify_layer_type,
    get_block_classifier,
)
from lib.executor import _get_block_t_factors
from lib.recipe import BlockConfig
from nodes.block_config_sdxl_clip import WIDENBlockConfigSDXLCLIPNode

# =============================================================================
# AC-1: Node INPUT_TYPES has correct sliders
# =============================================================================


class TestSDXLCLIPBlockConfigInputTypes:
    """AC: @sdxl-clip-block-config ac-1 — node has correct sliders."""

    def test_has_clip_l_block_sliders(self):
        """INPUT_TYPES has CLIP-L block sliders CL00-CL11."""
        input_types = WIDENBlockConfigSDXLCLIPNode.INPUT_TYPES()
        required = input_types["required"]

        for i in range(12):
            slider_name = f"CL{i:02d}"
            assert slider_name in required, f"Missing slider {slider_name}"
            assert required[slider_name][0] == "FLOAT"
            assert required[slider_name][1]["min"] == 0.0
            assert required[slider_name][1]["max"] == 2.0

    def test_has_clip_g_block_sliders(self):
        """INPUT_TYPES has CLIP-G block sliders CG00-CG31."""
        input_types = WIDENBlockConfigSDXLCLIPNode.INPUT_TYPES()
        required = input_types["required"]

        for i in range(32):
            slider_name = f"CG{i:02d}"
            assert slider_name in required, f"Missing slider {slider_name}"
            assert required[slider_name][0] == "FLOAT"
            assert required[slider_name][1]["min"] == 0.0
            assert required[slider_name][1]["max"] == 2.0

    def test_has_clip_l_structural_sliders(self):
        """INPUT_TYPES has CLIP-L structural key sliders."""
        input_types = WIDENBlockConfigSDXLCLIPNode.INPUT_TYPES()
        required = input_types["required"]

        assert "CL_EMBED" in required
        assert "CL_FINAL" in required

    def test_has_clip_g_structural_sliders(self):
        """INPUT_TYPES has CLIP-G structural key sliders."""
        input_types = WIDENBlockConfigSDXLCLIPNode.INPUT_TYPES()
        required = input_types["required"]

        assert "CG_EMBED" in required
        assert "CG_FINAL" in required
        assert "CG_PROJ" in required

    def test_has_layer_type_sliders(self):
        """INPUT_TYPES has layer type sliders."""
        input_types = WIDENBlockConfigSDXLCLIPNode.INPUT_TYPES()
        required = input_types["required"]

        assert "attention" in required
        assert "feed_forward" in required
        assert "norm" in required

    def test_slider_default_values(self):
        """All sliders default to 1.0."""
        input_types = WIDENBlockConfigSDXLCLIPNode.INPUT_TYPES()
        required = input_types["required"]

        for slider_name, (_, config) in required.items():
            assert config["default"] == 1.0, f"Slider {slider_name} default != 1.0"


# =============================================================================
# AC-2 & AC-3: Slider value effects
# =============================================================================


class TestSliderValueEffects:
    """AC: @sdxl-clip-block-config ac-2, ac-3 — slider values affect merge strength."""

    def test_slider_zero_preserves_base(self):
        """Slider at 0.0 means no merge for that block.

        AC: @sdxl-clip-block-config ac-2
        """
        keys = ["clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"]
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("CL00", 0.0),),
        )

        groups = _get_block_t_factors(
            keys, block_config=config, arch="sdxl", default_t_factor=1.0, domain="clip"
        )

        assert 0.0 in groups
        assert groups[0.0] == [0]

    def test_slider_doubled_at_two(self):
        """Slider at 2.0 doubles merge strength.

        AC: @sdxl-clip-block-config ac-3
        Block overrides are absolute t_factor values, so slider=2.0 means
        that block merges at 2x strength (t_factor=2.0 instead of default 1.0).
        """
        keys = ["clip_g.transformer.text_model.encoder.layers.15.mlp.fc1.weight"]
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("CG15", 2.0),),
        )

        groups = _get_block_t_factors(
            keys, block_config=config, arch="sdxl", default_t_factor=1.0, domain="clip"
        )

        # Slider at 2.0 means absolute t_factor of 2.0 for this block
        assert 2.0 in groups
        assert groups[2.0] == [0]


# =============================================================================
# AC-4: BlockConfig production
# =============================================================================


class TestBlockConfigProduction:
    """AC: @sdxl-clip-block-config ac-4 — create_config produces correct BlockConfig."""

    def test_block_config_arch_is_sdxl(self):
        """BlockConfig.arch is 'sdxl'."""
        node = WIDENBlockConfigSDXLCLIPNode()
        # Build kwargs with all sliders at default
        kwargs = {}
        for i in range(12):
            kwargs[f"CL{i:02d}"] = 1.0
        kwargs["CL_EMBED"] = 1.0
        kwargs["CL_FINAL"] = 1.0
        for i in range(32):
            kwargs[f"CG{i:02d}"] = 1.0
        kwargs["CG_EMBED"] = 1.0
        kwargs["CG_FINAL"] = 1.0
        kwargs["CG_PROJ"] = 1.0
        kwargs["attention"] = 1.0
        kwargs["feed_forward"] = 1.0
        kwargs["norm"] = 1.0

        (config,) = node.create_config(**kwargs)

        assert config.arch == "sdxl"

    def test_block_config_contains_per_block_values(self):
        """BlockConfig.block_overrides contains per-block values."""
        node = WIDENBlockConfigSDXLCLIPNode()
        kwargs = {}
        for i in range(12):
            kwargs[f"CL{i:02d}"] = 0.5 if i < 6 else 1.5
        kwargs["CL_EMBED"] = 0.3
        kwargs["CL_FINAL"] = 0.7
        for i in range(32):
            kwargs[f"CG{i:02d}"] = 1.0
        kwargs["CG_EMBED"] = 1.0
        kwargs["CG_FINAL"] = 1.0
        kwargs["CG_PROJ"] = 0.9
        kwargs["attention"] = 1.0
        kwargs["feed_forward"] = 1.0
        kwargs["norm"] = 1.0

        (config,) = node.create_config(**kwargs)

        # Check specific overrides are present
        override_dict = dict(config.block_overrides)
        assert override_dict["CL00"] == 0.5
        assert override_dict["CL06"] == 1.5
        assert override_dict["CL_EMBED"] == 0.3
        assert override_dict["CL_FINAL"] == 0.7
        assert override_dict["CG_PROJ"] == 0.9

    def test_return_types_is_block_config(self):
        """RETURN_TYPES is ('BLOCK_CONFIG',)."""
        assert WIDENBlockConfigSDXLCLIPNode.RETURN_TYPES == ("BLOCK_CONFIG",)


# =============================================================================
# AC-5 & AC-6: Connection to CLIP Merge
# =============================================================================


class TestCLIPMergeIntegration:
    """AC: @sdxl-clip-block-config ac-5, ac-6 — integration with CLIP Merge."""

    def test_block_config_applies_per_block_control(self):
        """Block config connected to CLIP Merge applies per-block control.

        AC: @sdxl-clip-block-config ac-5
        """
        keys = [
            "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
            "clip_l.transformer.text_model.encoder.layers.5.mlp.fc1.weight",
            "clip_g.transformer.text_model.encoder.layers.20.self_attn.k_proj.weight",
        ]
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(
                ("CL00", 0.5),
                ("CL05", 1.5),
                ("CG20", 0.8),
            ),
        )

        groups = _get_block_t_factors(
            keys, block_config=config, arch="sdxl", default_t_factor=1.0, domain="clip"
        )

        assert 0.5 in groups
        assert 1.5 in groups
        assert 0.8 in groups
        assert groups[0.5] == [0]
        assert groups[1.5] == [1]
        assert groups[0.8] == [2]

    def test_no_block_config_uniform_strength(self):
        """No block config connected means uniform strength.

        AC: @sdxl-clip-block-config ac-6
        """
        keys = [
            "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
            "clip_l.transformer.text_model.encoder.layers.5.mlp.fc1.weight",
            "clip_g.transformer.text_model.encoder.layers.20.self_attn.k_proj.weight",
        ]

        groups = _get_block_t_factors(
            keys, block_config=None, arch="sdxl", default_t_factor=1.0, domain="clip"
        )

        # All keys should use default t_factor
        assert len(groups) == 1
        assert 1.0 in groups
        assert len(groups[1.0]) == 3


# =============================================================================
# AC-7, AC-8, AC-9: classify_key for CLIP keys
# =============================================================================


class TestClassifyKeySDXLCLIP:
    """AC: @sdxl-clip-block-config ac-7, ac-8, ac-9 — classify_key for CLIP keys."""

    def test_clip_l_layer_5_returns_cl05(self):
        """CLIP-L layer 5 key returns 'CL05'.

        AC: @sdxl-clip-block-config ac-7
        """
        key = "clip_l.transformer.text_model.encoder.layers.5.self_attn.q_proj.weight"
        result = classify_key(key, "sdxl", domain="clip")
        assert result == "CL05"

    def test_clip_g_layer_20_returns_cg20(self):
        """CLIP-G layer 20 key returns 'CG20'.

        AC: @sdxl-clip-block-config ac-8
        """
        key = "clip_g.transformer.text_model.encoder.layers.20.mlp.fc1.weight"
        result = classify_key(key, "sdxl", domain="clip")
        assert result == "CG20"

    def test_clip_l_embeddings_returns_cl_embed(self):
        """CLIP-L embeddings key returns 'CL_EMBED'.

        AC: @sdxl-clip-block-config ac-9
        """
        key = "clip_l.transformer.text_model.embeddings.token_embedding.weight"
        result = classify_key(key, "sdxl", domain="clip")
        assert result == "CL_EMBED"

    def test_clip_l_all_blocks(self):
        """All CLIP-L blocks (0-11) classify correctly."""
        for i in range(12):
            key = f"clip_l.transformer.text_model.encoder.layers.{i}.self_attn.q_proj.weight"
            result = classify_key(key, "sdxl", domain="clip")
            assert result == f"CL{i:02d}"

    def test_clip_g_all_blocks(self):
        """All CLIP-G blocks (0-31) classify correctly."""
        for i in range(32):
            key = f"clip_g.transformer.text_model.encoder.layers.{i}.mlp.fc1.weight"
            result = classify_key(key, "sdxl", domain="clip")
            assert result == f"CG{i:02d}"

    def test_clip_l_final_layer_norm(self):
        """CLIP-L final layer norm returns 'CL_FINAL'."""
        key = "clip_l.transformer.text_model.final_layer_norm.weight"
        result = classify_key(key, "sdxl", domain="clip")
        assert result == "CL_FINAL"

    def test_clip_g_embeddings_returns_cg_embed(self):
        """CLIP-G embeddings key returns 'CG_EMBED'."""
        key = "clip_g.transformer.text_model.embeddings.position_embedding.weight"
        result = classify_key(key, "sdxl", domain="clip")
        assert result == "CG_EMBED"

    def test_clip_g_final_layer_norm(self):
        """CLIP-G final layer norm returns 'CG_FINAL'."""
        key = "clip_g.transformer.text_model.final_layer_norm.weight"
        result = classify_key(key, "sdxl", domain="clip")
        assert result == "CG_FINAL"

    def test_clip_g_text_projection(self):
        """CLIP-G text projection returns 'CG_PROJ'."""
        key = "clip_g.transformer.text_projection.weight"
        result = classify_key(key, "sdxl", domain="clip")
        assert result == "CG_PROJ"

    def test_direct_classifier_function(self):
        """classify_key_sdxl_clip function works directly."""
        key = "clip_l.transformer.text_model.encoder.layers.3.mlp.fc2.weight"
        result = classify_key_sdxl_clip(key)
        assert result == "CL03"

    def test_get_block_classifier_returns_clip_classifier(self):
        """get_block_classifier returns CLIP classifier for 'sdxl_clip'."""
        classifier = get_block_classifier("sdxl_clip")
        assert classifier is classify_key_sdxl_clip

    def test_unknown_key_returns_none(self):
        """Unknown CLIP key returns None."""
        key = "unknown.encoder.weight"
        result = classify_key(key, "sdxl", domain="clip")
        assert result is None


# =============================================================================
# AC-10: classify_layer_type for CLIP keys
# =============================================================================


class TestClassifyLayerTypeSDXLCLIP:
    """AC: @sdxl-clip-block-config ac-10 — classify_layer_type for CLIP keys."""

    def test_self_attn_q_proj_is_attention(self):
        """self_attn.q_proj is classified as 'attention'."""
        key = "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
        result = classify_layer_type(key, "sdxl", domain="clip")
        assert result == "attention"

    def test_self_attn_k_proj_is_attention(self):
        """self_attn.k_proj is classified as 'attention'."""
        key = "clip_g.transformer.text_model.encoder.layers.5.self_attn.k_proj.weight"
        result = classify_layer_type(key, "sdxl", domain="clip")
        assert result == "attention"

    def test_self_attn_v_proj_is_attention(self):
        """self_attn.v_proj is classified as 'attention'."""
        key = "clip_l.transformer.text_model.encoder.layers.10.self_attn.v_proj.weight"
        result = classify_layer_type(key, "sdxl", domain="clip")
        assert result == "attention"

    def test_self_attn_out_proj_is_attention(self):
        """self_attn.out_proj is classified as 'attention'."""
        key = "clip_g.transformer.text_model.encoder.layers.20.self_attn.out_proj.weight"
        result = classify_layer_type(key, "sdxl", domain="clip")
        assert result == "attention"

    def test_mlp_fc1_is_feed_forward(self):
        """mlp.fc1 is classified as 'feed_forward'."""
        key = "clip_l.transformer.text_model.encoder.layers.5.mlp.fc1.weight"
        result = classify_layer_type(key, "sdxl", domain="clip")
        assert result == "feed_forward"

    def test_mlp_fc2_is_feed_forward(self):
        """mlp.fc2 is classified as 'feed_forward'."""
        key = "clip_g.transformer.text_model.encoder.layers.25.mlp.fc2.weight"
        result = classify_layer_type(key, "sdxl", domain="clip")
        assert result == "feed_forward"

    def test_layer_norm1_is_norm(self):
        """layer_norm1 is classified as 'norm'."""
        key = "clip_l.transformer.text_model.encoder.layers.0.layer_norm1.weight"
        result = classify_layer_type(key, "sdxl", domain="clip")
        assert result == "norm"

    def test_layer_norm2_is_norm(self):
        """layer_norm2 is classified as 'norm'."""
        key = "clip_g.transformer.text_model.encoder.layers.15.layer_norm2.bias"
        result = classify_layer_type(key, "sdxl", domain="clip")
        assert result == "norm"

    def test_final_layer_norm_is_norm(self):
        """final_layer_norm is classified as 'norm'."""
        key = "clip_l.transformer.text_model.final_layer_norm.weight"
        result = classify_layer_type(key, "sdxl", domain="clip")
        assert result == "norm"

    def test_embeddings_return_none(self):
        """Embedding keys return None (not a layer type)."""
        key = "clip_l.transformer.text_model.embeddings.token_embedding.weight"
        result = classify_layer_type(key, "sdxl", domain="clip")
        assert result is None

    def test_text_projection_returns_none(self):
        """Text projection returns None (not a standard layer type)."""
        key = "clip_g.transformer.text_projection.weight"
        result = classify_layer_type(key, "sdxl", domain="clip")
        assert result is None


# =============================================================================
# Layer type + block override integration
# =============================================================================


class TestSDXLCLIPLayerTypeIntegration:
    """Integration tests for layer type overrides with CLIP blocks."""

    def test_block_and_layer_type_multiplicative(self):
        """Block and layer type overrides multiply together."""
        keys = ["clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"]
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("CL00", 0.8),),
            layer_type_overrides=(("attention", 0.5),),
        )

        groups = _get_block_t_factors(
            keys, block_config=config, arch="sdxl", default_t_factor=1.0, domain="clip"
        )

        # 0.8 * 0.5 = 0.4
        assert 0.4 in groups
        assert groups[0.4] == [0]

    def test_different_layer_types_different_t_factors(self):
        """Different layer types within same block get different t_factors."""
        keys = [
            "clip_g.transformer.text_model.encoder.layers.10.self_attn.q_proj.weight",
            "clip_g.transformer.text_model.encoder.layers.10.mlp.fc1.weight",
            "clip_g.transformer.text_model.encoder.layers.10.layer_norm1.weight",
        ]
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("CG10", 0.8),),
            layer_type_overrides=(
                ("attention", 0.5),
                ("feed_forward", 1.5),
                ("norm", 0.25),
            ),
        )

        groups = _get_block_t_factors(
            keys, block_config=config, arch="sdxl", default_t_factor=1.0, domain="clip"
        )

        # 0.8 * 0.5 = 0.4 (attention)
        # 0.8 * 1.5 = 1.2 (feed_forward)
        # 0.8 * 0.25 = 0.2 (norm)
        assert len(groups) == 3
        t_factors = sorted(groups.keys())
        assert abs(t_factors[0] - 0.2) < 1e-9
        assert abs(t_factors[1] - 0.4) < 1e-9
        assert abs(t_factors[2] - 1.2) < 1e-9


# =============================================================================
# Node metadata
# =============================================================================


class TestSDXLCLIPBlockConfigMetadata:
    """Node metadata tests."""

    def test_category(self):
        """CATEGORY is ecaj/merge."""
        assert WIDENBlockConfigSDXLCLIPNode.CATEGORY == "ecaj/merge"

    def test_function_name(self):
        """FUNCTION points to create_config method."""
        assert WIDENBlockConfigSDXLCLIPNode.FUNCTION == "create_config"

    def test_return_names(self):
        """RETURN_NAMES is ('block_config',)."""
        assert WIDENBlockConfigSDXLCLIPNode.RETURN_NAMES == ("block_config",)
