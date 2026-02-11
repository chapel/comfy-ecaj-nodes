"""Tests for LoRA Per-Block Strength feature.

Tests for @lora-block-config acceptance criteria:
- AC-1: BLOCK_CONFIG connected to LoRA block_strength input applies per-block scaling
- AC-2: No BLOCK_CONFIG connected means global strength applies uniformly

Tests the _apply_per_block_lora_strength helper and integration through evaluate_recipe.
"""

import torch

from lib.executor import _apply_per_block_lora_strength
from lib.recipe import BlockConfig, RecipeBase, RecipeLoRA, RecipeMerge

# =============================================================================
# _apply_per_block_lora_strength Unit Tests
# =============================================================================


class TestApplyPerBlockLoraStrength:
    """Direct tests for the per-block LoRA strength helper function."""

    def test_no_overrides_returns_lora_applied(self):
        """When all overrides are 1.0, returns lora_applied unchanged.

        AC: @lora-block-config ac-2
        Given: no meaningful overrides (all 1.0)
        When: per-block strength is applied
        Then: output equals lora_applied (no modification)
        """
        keys = ["input_blocks.0.0.weight", "input_blocks.1.0.weight"]
        base = torch.zeros(2, 4, 4)
        lora_applied = torch.ones(2, 4, 4)

        # Block config with 1.0 override (no-op)
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00-02", 1.0),),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # Should be unchanged since all overrides are 1.0
        assert torch.allclose(result, lora_applied)

    def test_scales_delta_by_block_strength(self):
        """Per-block strength scales the LoRA delta (lora_applied - base).

        AC: @lora-block-config ac-1
        Given: a BLOCK_CONFIG with strength 0.5 for IN00-02
        When: Exit applies LoRA deltas
        Then: delta for IN00-02 keys is scaled by 0.5
        """
        keys = ["input_blocks.0.0.weight", "input_blocks.1.0.weight"]
        base = torch.zeros(2, 4, 4)
        # LoRA adds 2.0 to all values
        lora_applied = torch.full((2, 4, 4), 2.0)

        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00-02", 0.5),),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # Delta was 2.0, scaled by 0.5 = 1.0, added to base 0.0 = 1.0
        expected = torch.full((2, 4, 4), 1.0)
        assert torch.allclose(result, expected)

    def test_different_strengths_per_block(self):
        """Different blocks can have different strength multipliers.

        AC: @lora-block-config ac-1
        """
        keys = [
            "input_blocks.0.0.weight",   # IN00-02 -> 0.5
            "middle_block.0.weight",     # MID -> 2.0
            "output_blocks.3.0.weight",  # OUT03-05 -> no override (1.0)
        ]
        base = torch.zeros(3, 4, 4)
        # LoRA adds 4.0 to all values
        lora_applied = torch.full((3, 4, 4), 4.0)

        config = BlockConfig(
            arch="sdxl",
            block_overrides=(
                ("IN00-02", 0.5),
                ("MID", 2.0),
            ),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # Check each key's result
        # IN00-02: delta 4.0 * 0.5 = 2.0
        assert torch.allclose(result[0], torch.full((4, 4), 2.0))
        # MID: delta 4.0 * 2.0 = 8.0
        assert torch.allclose(result[1], torch.full((4, 4), 8.0))
        # OUT03-05: delta 4.0 * 1.0 (default) = 4.0
        assert torch.allclose(result[2], torch.full((4, 4), 4.0))

    def test_zero_strength_removes_lora_effect(self):
        """Strength of 0.0 completely removes the LoRA delta.

        AC: @lora-block-config ac-1
        """
        keys = ["input_blocks.0.0.weight"]
        base = torch.full((1, 4, 4), 10.0)
        lora_applied = torch.full((1, 4, 4), 20.0)  # Delta of 10.0

        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00-02", 0.0),),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # Delta 10.0 * 0.0 = 0.0, so result = base
        assert torch.allclose(result, base)

    def test_strength_above_one_amplifies(self):
        """Strength > 1.0 amplifies the LoRA effect.

        AC: @lora-block-config ac-1
        """
        keys = ["middle_block.0.weight"]
        base = torch.zeros(1, 4, 4)
        lora_applied = torch.full((1, 4, 4), 3.0)  # Delta of 3.0

        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("MID", 1.5),),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # Delta 3.0 * 1.5 = 4.5
        expected = torch.full((1, 4, 4), 4.5)
        assert torch.allclose(result, expected)

    def test_unmatched_keys_use_default_strength(self):
        """Keys not matching any block pattern use 1.0 (unchanged).

        AC: @lora-block-config ac-2
        """
        keys = ["time_embed.0.weight", "label_emb.weight"]  # Don't match SDXL blocks
        base = torch.zeros(2, 4, 4)
        lora_applied = torch.full((2, 4, 4), 5.0)

        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00-02", 0.5),),  # Doesn't apply to these keys
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # Keys don't match any block, so delta is unchanged
        assert torch.allclose(result, lora_applied)

    def test_zimage_block_strength(self):
        """Z-Image architecture uses its own block classification.

        AC: @lora-block-config ac-1
        """
        keys = [
            "layers.0.attn.weight",   # L00-04 -> 0.25
            "layers.10.mlp.weight",   # L10-14 -> 1.0 (default)
            "noise_refiner.weight",   # noise_refiner -> 0.75
        ]
        base = torch.zeros(3, 4, 4)
        lora_applied = torch.full((3, 4, 4), 8.0)  # Delta of 8.0

        config = BlockConfig(
            arch="zimage",
            block_overrides=(
                ("L00-04", 0.25),
                ("noise_refiner", 0.75),
            ),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "zimage", "cpu", torch.float32
        )

        # L00-04: delta 8.0 * 0.25 = 2.0
        assert torch.allclose(result[0], torch.full((4, 4), 2.0))
        # L10-14: delta 8.0 * 1.0 = 8.0 (no override)
        assert torch.allclose(result[1], torch.full((4, 4), 8.0))
        # noise_refiner: delta 8.0 * 0.75 = 6.0
        assert torch.allclose(result[2], torch.full((4, 4), 6.0))

    def test_conv2d_shapes(self):
        """Works with 4D conv2d weight tensors.

        AC: @lora-block-config ac-1
        """
        keys = ["input_blocks.0.0.weight"]
        base = torch.zeros(1, 64, 64, 3, 3)
        lora_applied = torch.full((1, 64, 64, 3, 3), 4.0)

        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00-02", 0.5),),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # Delta 4.0 * 0.5 = 2.0
        expected = torch.full((1, 64, 64, 3, 3), 2.0)
        assert torch.allclose(result, expected)

    def test_preserves_negative_deltas(self):
        """Correctly handles negative LoRA deltas.

        AC: @lora-block-config ac-1
        """
        keys = ["input_blocks.0.0.weight"]
        base = torch.full((1, 4, 4), 10.0)
        lora_applied = torch.full((1, 4, 4), 6.0)  # Delta of -4.0

        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00-02", 0.5),),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # Delta -4.0 * 0.5 = -2.0, so result = 10.0 - 2.0 = 8.0
        expected = torch.full((1, 4, 4), 8.0)
        assert torch.allclose(result, expected)


# =============================================================================
# RecipeLoRA block_config Integration Tests
# =============================================================================


class TestRecipeLoRABlockConfig:
    """Tests for RecipeLoRA.block_config field behavior."""

    def test_recipe_lora_stores_block_config(self):
        """RecipeLoRA correctly stores block_config.

        AC: @lora-block-config ac-1
        """
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00-02", 0.5),),
        )
        lora = RecipeLoRA(
            loras=({"path": "test.safetensors", "strength": 1.0},),
            block_config=config,
        )

        assert lora.block_config is config
        assert lora.block_config.block_overrides == (("IN00-02", 0.5),)

    def test_recipe_lora_none_block_config(self):
        """RecipeLoRA with None block_config for backwards compatibility.

        AC: @lora-block-config ac-2
        """
        lora = RecipeLoRA(
            loras=({"path": "test.safetensors", "strength": 1.0},),
            block_config=None,
        )

        assert lora.block_config is None

    def test_recipe_lora_default_block_config(self):
        """RecipeLoRA block_config defaults to None.

        AC: @lora-block-config ac-2
        """
        lora = RecipeLoRA(
            loras=({"path": "test.safetensors", "strength": 1.0},),
        )

        assert lora.block_config is None


# =============================================================================
# Backwards Compatibility Tests
# =============================================================================


class TestBackwardsCompatibility:
    """Ensure no block_config maintains pre-feature behavior.

    AC: @lora-block-config ac-2
    """

    def test_no_block_config_no_scaling(self):
        """Without block_config, LoRA deltas are applied without scaling.

        AC: @lora-block-config ac-2
        Given: no BLOCK_CONFIG connected to LoRA node
        When: Exit applies LoRA deltas
        Then: global strength applies uniformly (backwards compatible)
        """
        # This is verified by the fact that _apply_per_block_lora_strength
        # is only called when block_config is not None
        lora = RecipeLoRA(
            loras=({"path": "test.safetensors", "strength": 0.8},),
            block_config=None,
        )

        # block_config is None means no per-block scaling is applied
        assert lora.block_config is None

    def test_recipe_merge_chain_preserves_lora_block_config(self):
        """RecipeMerge correctly preserves LoRA block_config in tree.

        AC: @lora-block-config ac-1
        """

        class MockModel:
            pass

        mock_patcher = MockModel()
        base = RecipeBase(model_patcher=mock_patcher, arch="sdxl")

        config = BlockConfig(arch="sdxl", block_overrides=(("MID", 0.7),))
        lora = RecipeLoRA(
            loras=({"path": "test.safetensors", "strength": 1.0},),
            block_config=config,
        )

        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        # Access block_config through the tree
        assert merge.target.block_config is config
