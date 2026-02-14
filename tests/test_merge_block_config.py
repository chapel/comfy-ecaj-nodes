"""Tests for Merge Per-Block T-Factor feature.

Tests for @merge-block-config acceptance criteria:
- AC-1: BLOCK_CONFIG connected to Merge block_t_factor input applies per-block overrides
- AC-2: No BLOCK_CONFIG connected means global t_factor applies (backwards compatible)
"""

from lib.block_classify import (
    classify_key,
    classify_key_flux,
    classify_key_qwen,
    classify_key_sdxl,
    classify_key_zimage,
    get_block_classifier,
)
from lib.executor import _get_block_t_factors
from lib.recipe import BlockConfig, RecipeBase, RecipeLoRA, RecipeMerge

# =============================================================================
# Block Classification Tests
# =============================================================================


class TestBlockClassifySDXL:
    """SDXL block classification tests."""

    def test_input_blocks_classify_individually(self):
        """Input blocks 0-8 classify as individual IN00-IN08."""
        # AC: @merge-block-config ac-1
        assert classify_key_sdxl("input_blocks.0.0.proj_in.weight") == "IN00"
        key = "input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"
        assert classify_key_sdxl(key) == "IN01"
        assert classify_key_sdxl("input_blocks.2.1.proj_out.weight") == "IN02"
        assert classify_key_sdxl("input_blocks.3.0.proj_in.weight") == "IN03"
        assert classify_key_sdxl("input_blocks.4.1.proj_out.weight") == "IN04"
        assert classify_key_sdxl("input_blocks.5.0.attn1.to_v.weight") == "IN05"
        assert classify_key_sdxl("input_blocks.6.1.weight") == "IN06"
        assert classify_key_sdxl("input_blocks.7.0.proj_in.weight") == "IN07"
        assert classify_key_sdxl("input_blocks.8.1.attn2.to_k.weight") == "IN08"

    def test_middle_block(self):
        """Middle block classifies as MID."""
        # AC: @merge-block-config ac-1
        assert classify_key_sdxl("middle_block.0.weight") == "MID"
        assert classify_key_sdxl("middle_block.1.transformer_blocks.0.attn1.to_q.weight") == "MID"
        assert classify_key_sdxl("middle_block.2.proj_out.weight") == "MID"

    def test_output_blocks_classify_individually(self):
        """Output blocks 0-8 classify as individual OUT00-OUT08."""
        # AC: @merge-block-config ac-1
        assert classify_key_sdxl("output_blocks.0.0.weight") == "OUT00"
        assert classify_key_sdxl("output_blocks.1.1.proj_in.weight") == "OUT01"
        assert classify_key_sdxl("output_blocks.2.1.attn1.to_v.weight") == "OUT02"
        assert classify_key_sdxl("output_blocks.3.0.weight") == "OUT03"
        assert classify_key_sdxl("output_blocks.4.1.proj_out.weight") == "OUT04"
        assert classify_key_sdxl("output_blocks.5.0.attn2.to_k.weight") == "OUT05"
        assert classify_key_sdxl("output_blocks.6.1.weight") == "OUT06"
        assert classify_key_sdxl("output_blocks.7.0.proj_in.weight") == "OUT07"
        assert classify_key_sdxl("output_blocks.8.1.attn1.to_q.weight") == "OUT08"

    def test_strips_diffusion_model_prefix(self):
        """Key classification strips diffusion_model. prefix."""
        # AC: @merge-block-config ac-1
        assert classify_key_sdxl("diffusion_model.input_blocks.0.0.weight") == "IN00"
        assert classify_key_sdxl("diffusion_model.middle_block.0.weight") == "MID"
        assert classify_key_sdxl("diffusion_model.output_blocks.3.0.weight") == "OUT03"

    def test_unmatched_returns_none(self):
        """Keys not matching any block return None."""
        # AC: @merge-block-config ac-2
        assert classify_key_sdxl("time_embed.0.weight") is None
        assert classify_key_sdxl("label_emb.weight") is None
        assert classify_key_sdxl("out.0.weight") is None


class TestBlockClassifyZImage:
    """Z-Image/S3-DiT block classification tests."""

    def test_layers_classify_individually(self):
        """Layers 0-29 classify as individual L00-L29."""
        # AC: @merge-block-config ac-1
        assert classify_key_zimage("layers.0.attn.qkv.weight") == "L00"
        assert classify_key_zimage("layers.2.mlp.fc1.weight") == "L02"
        assert classify_key_zimage("layers.4.attn.out.weight") == "L04"
        assert classify_key_zimage("layers.5.attn.qkv.weight") == "L05"
        assert classify_key_zimage("layers.10.attn.qkv.weight") == "L10"
        assert classify_key_zimage("layers.15.attn.qkv.weight") == "L15"
        assert classify_key_zimage("layers.20.attn.qkv.weight") == "L20"
        assert classify_key_zimage("layers.25.attn.qkv.weight") == "L25"
        assert classify_key_zimage("layers.29.norm.weight") == "L29"

    def test_noise_refiner_submodules(self):
        """Noise refiner keys classify as NOISE_REF0 or NOISE_REF1."""
        # AC: @merge-block-config ac-1
        assert classify_key_zimage("noise_refiner.0.attn.qkv.weight") == "NOISE_REF0"
        assert classify_key_zimage("noise_refiner.0.mlp.fc1.weight") == "NOISE_REF0"
        assert classify_key_zimage("noise_refiner.1.attn.qkv.weight") == "NOISE_REF1"

    def test_context_refiner_submodules(self):
        """Context refiner keys classify as CTX_REF0 or CTX_REF1."""
        # AC: @merge-block-config ac-1
        assert classify_key_zimage("context_refiner.0.attn.qkv.weight") == "CTX_REF0"
        assert classify_key_zimage("context_refiner.1.mlp.fc2.weight") == "CTX_REF1"

    def test_strips_prefixes(self):
        """Key classification strips common prefixes."""
        # AC: @merge-block-config ac-1
        assert classify_key_zimage("diffusion_model.layers.0.attn.qkv.weight") == "L00"
        assert classify_key_zimage("transformer.layers.15.mlp.fc1.weight") == "L15"

    def test_blocks_alternate_name(self):
        """Classification handles 'blocks' as alternate to 'layers'."""
        # AC: @merge-block-config ac-1
        assert classify_key_zimage("blocks.0.attn.qkv.weight") == "L00"
        assert classify_key_zimage("blocks.25.mlp.fc1.weight") == "L25"

    def test_refiner_without_submodule_not_matched(self):
        """Refiner keys without submodule number are not matched."""
        # These don't match the noise_refiner.N. pattern
        assert classify_key_zimage("noise_refiner.attn.qkv.weight") is None
        assert classify_key_zimage("context_refiner.mlp.fc2.weight") is None

    def test_unmatched_returns_none(self):
        """Keys not matching any block return None."""
        # AC: @merge-block-config ac-2
        assert classify_key_zimage("patch_embed.weight") is None
        assert classify_key_zimage("final_norm.weight") is None


class TestBlockClassifyQwen:
    """Qwen block classification tests."""

    # AC: @qwen-detect-classify ac-2
    def test_transformer_blocks_classify_individually(self):
        """Transformer blocks classify as TB00-TB59+ with dynamic range."""
        assert classify_key_qwen("transformer_blocks.0.attn.weight") == "TB00"
        assert classify_key_qwen("transformer_blocks.1.mlp.weight") == "TB01"
        assert classify_key_qwen("transformer_blocks.10.attn.weight") == "TB10"
        assert classify_key_qwen("transformer_blocks.29.norm.weight") == "TB29"
        assert classify_key_qwen("transformer_blocks.59.attn.weight") == "TB59"
        # Dynamic range - no upper bound
        assert classify_key_qwen("transformer_blocks.60.attn.weight") == "TB60"
        assert classify_key_qwen("transformer_blocks.99.mlp.weight") == "TB99"

    # AC: @qwen-detect-classify ac-2
    def test_strips_prefixes(self):
        """Key classification strips common prefixes."""
        key = "diffusion_model.transformer_blocks.0.attn.weight"
        assert classify_key_qwen(key) == "TB00"
        key = "transformer.transformer_blocks.15.mlp.weight"
        assert classify_key_qwen(key) == "TB15"

    # AC: @qwen-detect-classify ac-2
    def test_unmatched_returns_none(self):
        """Keys not matching any block return None."""
        assert classify_key_qwen("time_embed.0.weight") is None
        assert classify_key_qwen("final_norm.weight") is None
        assert classify_key_qwen("transformer_blocks.attn.weight") is None


class TestBlockClassifyFlux:
    """Flux Klein block classification tests."""

    # AC: @flux-klein-support ac-2
    def test_double_blocks_classify_individually(self):
        """Double blocks classify as DB00-DB07 with dynamic range."""
        assert classify_key_flux("double_blocks.0.img_attn.weight") == "DB00"
        assert classify_key_flux("double_blocks.1.txt_attn.weight") == "DB01"
        assert classify_key_flux("double_blocks.7.img_mlp.weight") == "DB07"
        # Dynamic range - Klein 4B has only 5 double blocks
        assert classify_key_flux("double_blocks.4.img_attn.weight") == "DB04"

    # AC: @flux-klein-support ac-2
    def test_single_blocks_classify_individually(self):
        """Single blocks classify as SB00-SB23 with dynamic range."""
        assert classify_key_flux("single_blocks.0.linear1.weight") == "SB00"
        assert classify_key_flux("single_blocks.10.linear2.weight") == "SB10"
        assert classify_key_flux("single_blocks.23.modulation.weight") == "SB23"
        # Dynamic range - Klein 4B has only 20 single blocks
        assert classify_key_flux("single_blocks.19.linear1.weight") == "SB19"

    # AC: @flux-klein-support ac-2
    def test_strips_prefixes(self):
        """Key classification strips common prefixes."""
        key = "diffusion_model.double_blocks.0.img_attn.weight"
        assert classify_key_flux(key) == "DB00"
        key = "transformer.single_blocks.5.linear1.weight"
        assert classify_key_flux(key) == "SB05"

    # AC: @flux-klein-support ac-2, ac-10
    def test_both_klein_variants(self):
        """Both Klein 4B (5+20) and 9B (8+24) work with same classifier."""
        # Klein 4B max indices
        assert classify_key_flux("double_blocks.4.img_attn.weight") == "DB04"
        assert classify_key_flux("single_blocks.19.linear1.weight") == "SB19"
        # Klein 9B max indices
        assert classify_key_flux("double_blocks.7.img_attn.weight") == "DB07"
        assert classify_key_flux("single_blocks.23.linear1.weight") == "SB23"

    # AC: @flux-klein-support ac-2
    def test_unmatched_returns_none(self):
        """Non-block keys return None."""
        assert classify_key_flux("guidance_in.weight") is None
        assert classify_key_flux("time_in.weight") is None
        assert classify_key_flux("vector_in.weight") is None
        assert classify_key_flux("img_in.weight") is None
        assert classify_key_flux("txt_in.weight") is None
        assert classify_key_flux("final_layer.weight") is None
        # Missing block index
        assert classify_key_flux("double_blocks.img_attn.weight") is None


class TestGetBlockClassifier:
    """get_block_classifier function tests."""

    def test_returns_sdxl_classifier(self):
        """Returns SDXL classifier for 'sdxl' arch."""
        classifier = get_block_classifier("sdxl")
        assert classifier is classify_key_sdxl

    def test_returns_zimage_classifier(self):
        """Returns Z-Image classifier for 'zimage' arch."""
        classifier = get_block_classifier("zimage")
        assert classifier is classify_key_zimage

    # AC: @qwen-detect-classify ac-2
    def test_returns_qwen_classifier(self):
        """Returns Qwen classifier for 'qwen' arch."""
        classifier = get_block_classifier("qwen")
        assert classifier is classify_key_qwen

    # AC: @flux-klein-support ac-2
    def test_returns_flux_classifier(self):
        """Returns Flux classifier for 'flux' arch."""
        classifier = get_block_classifier("flux")
        assert classifier is classify_key_flux

    def test_returns_none_for_unknown_arch(self):
        """Returns None for unknown architectures."""
        assert get_block_classifier("unknown") is None

    def test_classify_key_convenience_function(self):
        """classify_key convenience function works correctly."""
        # AC: @merge-block-config ac-1
        assert classify_key("input_blocks.0.0.weight", "sdxl") == "IN00"
        assert classify_key("layers.0.attn.weight", "zimage") == "L00"
        assert classify_key("input_blocks.0.0.weight", "unknown") is None


# =============================================================================
# Per-Block T-Factor Grouping Tests
# =============================================================================


class TestGetBlockTFactors:
    """_get_block_t_factors function tests."""

    def test_no_block_config_all_default(self):
        """Without block_config, all keys use default t_factor.

        AC: @merge-block-config ac-2
        Given: no BLOCK_CONFIG connected to Merge
        When: Exit evaluates
        Then: global t_factor applies to all blocks
        """
        keys = ["input_blocks.0.0.weight", "middle_block.0.weight", "output_blocks.3.0.weight"]
        default_t = 1.0

        groups = _get_block_t_factors(
            keys, block_config=None, arch="sdxl", default_t_factor=default_t
        )

        # All keys should be in the default t_factor group
        assert len(groups) == 1
        assert default_t in groups
        assert len(groups[default_t]) == 3

    def test_no_arch_all_default(self):
        """Without arch, all keys use default t_factor.

        AC: @merge-block-config ac-2
        """
        keys = ["input_blocks.0.0.weight", "middle_block.0.weight"]
        config = BlockConfig(arch="sdxl", block_overrides=(("IN00-02", 0.5),))
        default_t = 1.0

        groups = _get_block_t_factors(
            keys, block_config=config, arch=None, default_t_factor=default_t
        )

        # Without arch, can't classify, so all keys use default
        assert len(groups) == 1
        assert default_t in groups

    def test_with_block_config_groups_by_override(self):
        """With block_config, keys are grouped by their override t_factor.

        AC: @merge-block-config ac-1
        Given: a BLOCK_CONFIG connected to Merge block_t_factor input
        When: Exit evaluates the merge step
        Then: per-block t_factor overrides are applied
        """
        keys = [
            "input_blocks.0.0.weight",  # IN00 -> 0.5
            "input_blocks.1.0.weight",  # IN01 -> 0.5
            "middle_block.0.weight",  # MID -> 1.2
            "output_blocks.3.0.weight",  # OUT03 -> default 1.0
        ]
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(
                ("IN00", 0.5),
                ("IN01", 0.5),
                ("MID", 1.2),
            ),
        )
        default_t = 1.0

        groups = _get_block_t_factors(
            keys, block_config=config, arch="sdxl", default_t_factor=default_t
        )

        # Should have 3 groups: 0.5, 1.2, and 1.0 (default)
        assert len(groups) == 3
        assert 0.5 in groups
        assert 1.2 in groups
        assert 1.0 in groups

        # Check correct key indices in each group
        assert groups[0.5] == [0, 1]  # First two input blocks
        assert groups[1.2] == [2]  # Middle block
        assert groups[1.0] == [3]  # Output block (no override)

    def test_unmatched_keys_use_default(self):
        """Keys not matching any block pattern use default t_factor.

        AC: @merge-block-config ac-2
        """
        keys = ["time_embed.0.weight", "label_emb.weight"]  # No block match
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00", 0.5),),
        )
        default_t = 1.0

        groups = _get_block_t_factors(
            keys, block_config=config, arch="sdxl", default_t_factor=default_t
        )

        # Both keys don't match any block, use default
        assert len(groups) == 1
        assert 1.0 in groups
        assert len(groups[1.0]) == 2

    def test_zimage_block_grouping(self):
        """Z-Image keys are grouped by individual blocks.

        AC: @merge-block-config ac-1
        """
        keys = [
            "layers.0.attn.weight",  # L00 -> 0.3
            "layers.5.attn.weight",  # L05 -> default 1.0
            "layers.25.attn.weight",  # L25 -> 1.5
            "noise_refiner.0.attn.weight",  # NOISE_REF0 -> 0.8
        ]
        config = BlockConfig(
            arch="zimage",
            block_overrides=(
                ("L00", 0.3),
                ("L25", 1.5),
                ("NOISE_REF0", 0.8),
            ),
        )
        default_t = 1.0

        groups = _get_block_t_factors(
            keys, block_config=config, arch="zimage", default_t_factor=default_t
        )

        assert len(groups) == 4
        assert groups[0.3] == [0]  # L00
        assert groups[1.0] == [1]  # L05 (no override)
        assert groups[1.5] == [2]  # L25
        assert groups[0.8] == [3]  # NOISE_REF0


# =============================================================================
# Integration Tests - RecipeMerge with block_config
# =============================================================================


class TestRecipeMergeBlockConfig:
    """RecipeMerge block_config integration tests."""

    def test_recipe_merge_stores_block_config(self):
        """RecipeMerge stores block_config from node.

        AC: @merge-block-config ac-1
        """
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00", 0.5), ("MID", 1.2)),
        )
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))

        merge = RecipeMerge(
            base=base,
            target=lora,
            backbone=None,
            t_factor=1.0,
            block_config=config,
        )

        assert merge.block_config is config
        assert merge.block_config.arch == "sdxl"
        assert len(merge.block_config.block_overrides) == 2

    def test_recipe_merge_none_block_config(self):
        """RecipeMerge with None block_config is backwards compatible.

        AC: @merge-block-config ac-2
        """
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))

        merge = RecipeMerge(
            base=base,
            target=lora,
            backbone=None,
            t_factor=1.0,
            block_config=None,
        )

        assert merge.block_config is None

    def test_recipe_merge_default_block_config(self):
        """RecipeMerge defaults to None block_config.

        AC: @merge-block-config ac-2
        """
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))

        merge = RecipeMerge(
            base=base,
            target=lora,
            backbone=None,
            t_factor=1.0,
        )

        assert merge.block_config is None


# =============================================================================
# Edge Cases
# =============================================================================


class TestBlockConfigEdgeCases:
    """Edge case tests for block config handling."""

    def test_empty_block_overrides(self):
        """Empty block_overrides means all keys use default.

        AC: @merge-block-config ac-2
        """
        keys = ["input_blocks.0.0.weight", "middle_block.0.weight"]
        config = BlockConfig(arch="sdxl", block_overrides=())
        default_t = 1.0

        groups = _get_block_t_factors(
            keys, block_config=config, arch="sdxl", default_t_factor=default_t
        )

        # No overrides, all use default
        assert len(groups) == 1
        assert groups[1.0] == [0, 1]

    def test_all_keys_same_override(self):
        """All keys matching same block have single group."""
        keys = [
            "input_blocks.0.0.weight",
            "input_blocks.0.1.weight",
            "input_blocks.0.2.weight",
        ]
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00", 0.5),),
        )

        groups = _get_block_t_factors(keys, block_config=config, arch="sdxl", default_t_factor=1.0)

        assert len(groups) == 1
        assert groups[0.5] == [0, 1, 2]

    def test_arch_mismatch_still_classifies(self):
        """Block config arch doesn't prevent classification.

        The arch parameter to _get_block_t_factors determines classification,
        not the BlockConfig.arch field.
        """
        keys = ["input_blocks.0.0.weight"]
        # BlockConfig says zimage but we're classifying as sdxl
        config = BlockConfig(
            arch="zimage",
            block_overrides=(("IN00", 0.5),),  # This override won't match
        )

        # Classify as sdxl - IN00 would match if BlockConfig arch matched
        groups = _get_block_t_factors(keys, block_config=config, arch="sdxl", default_t_factor=1.0)

        # Should still apply the IN00 override since we look up by block name
        assert groups[0.5] == [0]


# =============================================================================
# Layer-Type T-Factor Tests
# =============================================================================


class TestLayerTypeTFactor:
    """Tests for layer_type_overrides multiplicative effect on t_factor.

    AC: @layer-type-filter ac-3
    AC: @layer-type-filter ac-4
    """

    # AC: @layer-type-filter ac-3
    def test_block_and_layer_type_multiplicative(self):
        """Effective t_factor = block_t_factor * layer_type_multiplier.

        AC: @layer-type-filter ac-3
        Given: block t=0.8, attention=0.5
        Then: effective t=0.4
        """
        keys = ["input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"]  # IN01, attention
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN01", 0.8),),
            layer_type_overrides=(("attention", 0.5),),
        )

        groups = _get_block_t_factors(keys, block_config=config, arch="sdxl", default_t_factor=1.0)

        # 0.8 * 0.5 = 0.4
        assert 0.4 in groups
        assert groups[0.4] == [0]

    # AC: @layer-type-filter ac-3
    def test_layer_type_multiplier_doubles(self):
        """layer_type at 2.0 doubles the block t_factor.

        AC: @layer-type-filter ac-3
        """
        keys = ["input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"]  # attention
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN01", 0.6),),
            layer_type_overrides=(("attention", 2.0),),
        )

        groups = _get_block_t_factors(keys, block_config=config, arch="sdxl", default_t_factor=1.0)

        # 0.6 * 2.0 = 1.2
        assert 1.2 in groups
        assert groups[1.2] == [0]

    # AC: @layer-type-filter ac-4
    def test_empty_layer_type_overrides_backwards_compatible(self):
        """Empty layer_type_overrides means behavior identical to before.

        AC: @layer-type-filter ac-4
        """
        keys = ["input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"]
        # BlockConfig with only block overrides
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN01", 0.5),),
        )

        groups = _get_block_t_factors(keys, block_config=config, arch="sdxl", default_t_factor=1.0)

        # Only block override applies
        assert 0.5 in groups
        assert groups[0.5] == [0]

    # AC: @layer-type-filter ac-3
    def test_layer_type_zero_zeroes_t_factor(self):
        """layer_type=0.0 gives effective t_factor of 0.0.

        AC: @layer-type-filter ac-3
        """
        keys = ["input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"]  # attention
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN01", 0.8),),
            layer_type_overrides=(("attention", 0.0),),
        )

        groups = _get_block_t_factors(keys, block_config=config, arch="sdxl", default_t_factor=1.0)

        # 0.8 * 0.0 = 0.0
        assert 0.0 in groups
        assert groups[0.0] == [0]

    # AC: @layer-type-filter ac-3
    def test_different_layer_types_different_effective_t(self):
        """Different layer types get different effective t_factors."""
        keys = [
            "input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",  # IN01, attention
            "input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.weight",  # IN01, feed_forward
            "input_blocks.1.1.transformer_blocks.0.norm1.weight",  # IN01, norm
            "time_embed.0.weight",  # no block, no layer type
        ]
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN01", 0.8),),
            layer_type_overrides=(
                ("attention", 0.5),  # 0.8 * 0.5 = 0.4
                ("feed_forward", 1.5),  # 0.8 * 1.5 = 1.2
                ("norm", 0.25),  # 0.8 * 0.25 = 0.2
            ),
        )

        groups = _get_block_t_factors(keys, block_config=config, arch="sdxl", default_t_factor=1.0)

        # Check 4 distinct t_factors exist
        assert len(groups) == 4

        # Verify each key is mapped to correct group
        # Use approximate matching for floating point
        t_factors = sorted(groups.keys())
        assert abs(t_factors[0] - 0.2) < 1e-9  # norm: 0.8 * 0.25
        assert abs(t_factors[1] - 0.4) < 1e-9  # attention: 0.8 * 0.5
        assert abs(t_factors[2] - 1.0) < 1e-9  # time_embed: default
        assert abs(t_factors[3] - 1.2) < 1e-9  # feed_forward: 0.8 * 1.5

        # Verify indices
        assert groups[t_factors[0]] == [2]  # norm
        assert groups[t_factors[1]] == [0]  # attention
        assert groups[t_factors[2]] == [3]  # time_embed
        assert groups[t_factors[3]] == [1]  # feed_forward

    # AC: @layer-type-filter ac-3
    def test_layer_type_only_no_block_override(self):
        """Layer type applies when block uses default t_factor."""
        keys = ["input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"]  # IN01, attention
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(),  # No block overrides
            layer_type_overrides=(("attention", 0.5),),
        )

        groups = _get_block_t_factors(keys, block_config=config, arch="sdxl", default_t_factor=1.0)

        # default 1.0 * 0.5 = 0.5
        assert 0.5 in groups
        assert groups[0.5] == [0]

    # AC: @layer-type-filter ac-3
    def test_zimage_layer_type_multiplicative(self):
        """Z-Image layer type overrides work multiplicatively."""
        keys = [
            "layers.5.attn.qkv.weight",  # L05, attention
            "layers.5.feed_forward.w1.weight",  # L05, feed_forward
            "layers.5.norm.weight",  # L05, norm
        ]
        config = BlockConfig(
            arch="zimage",
            block_overrides=(("L05", 0.8),),
            layer_type_overrides=(
                ("attention", 0.5),  # 0.8 * 0.5 = 0.4
                ("feed_forward", 1.5),  # 0.8 * 1.5 = 1.2
                ("norm", 0.25),  # 0.8 * 0.25 = 0.2
            ),
        )

        groups = _get_block_t_factors(
            keys, block_config=config, arch="zimage", default_t_factor=1.0
        )

        # Check 3 distinct t_factors exist
        assert len(groups) == 3

        # Verify using sorted keys and approximate matching for floating point
        t_factors = sorted(groups.keys())
        assert abs(t_factors[0] - 0.2) < 1e-9  # norm
        assert abs(t_factors[1] - 0.4) < 1e-9  # attention
        assert abs(t_factors[2] - 1.2) < 1e-9  # feed_forward

        # Verify indices
        assert groups[t_factors[0]] == [2]  # norm
        assert groups[t_factors[1]] == [0]  # attention
        assert groups[t_factors[2]] == [1]  # feed_forward
