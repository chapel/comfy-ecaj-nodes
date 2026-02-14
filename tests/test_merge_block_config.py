"""Tests for Merge Per-Block T-Factor feature.

Tests for @merge-block-config acceptance criteria:
- AC-1: BLOCK_CONFIG connected to Merge block_t_factor input applies per-block overrides
- AC-2: No BLOCK_CONFIG connected means global t_factor applies (backwards compatible)
"""

from lib.block_classify import (
    classify_key,
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

    def test_returns_none_for_unknown_arch(self):
        """Returns None for unknown architectures."""
        assert get_block_classifier("unknown") is None
        assert get_block_classifier("flux") is None

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
            "input_blocks.0.0.weight",   # IN00 -> 0.5
            "input_blocks.1.0.weight",   # IN01 -> 0.5
            "middle_block.0.weight",     # MID -> 1.2
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
        assert groups[1.2] == [2]      # Middle block
        assert groups[1.0] == [3]      # Output block (no override)

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
            "layers.0.attn.weight",        # L00 -> 0.3
            "layers.5.attn.weight",        # L05 -> default 1.0
            "layers.25.attn.weight",       # L25 -> 1.5
            "noise_refiner.0.attn.weight", # NOISE_REF0 -> 0.8
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
        assert groups[0.3] == [0]   # L00
        assert groups[1.0] == [1]   # L05 (no override)
        assert groups[1.5] == [2]   # L25
        assert groups[0.8] == [3]   # NOISE_REF0


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

        groups = _get_block_t_factors(
            keys, block_config=config, arch="sdxl", default_t_factor=1.0
        )

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
        groups = _get_block_t_factors(
            keys, block_config=config, arch="sdxl", default_t_factor=1.0
        )

        # Should still apply the IN00 override since we look up by block name
        assert groups[0.5] == [0]
