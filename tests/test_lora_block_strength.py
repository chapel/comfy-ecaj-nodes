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

    # AC: @lora-block-config ac-2
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
            block_overrides=(("IN00", 1.0), ("IN01", 1.0)),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # Should be unchanged since all overrides are 1.0
        assert torch.allclose(result, lora_applied)

    # AC: @lora-block-config ac-1
    def test_scales_delta_by_block_strength(self):
        """Per-block strength scales the LoRA delta (lora_applied - base).

        AC: @lora-block-config ac-1
        Given: a BLOCK_CONFIG with strength 0.5 for IN00 and IN01
        When: Exit applies LoRA deltas
        Then: delta for these keys is scaled by 0.5
        """
        keys = ["input_blocks.0.0.weight", "input_blocks.1.0.weight"]
        base = torch.zeros(2, 4, 4)
        # LoRA adds 2.0 to all values
        lora_applied = torch.full((2, 4, 4), 2.0)

        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00", 0.5), ("IN01", 0.5)),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # Delta was 2.0, scaled by 0.5 = 1.0, added to base 0.0 = 1.0
        expected = torch.full((2, 4, 4), 1.0)
        assert torch.allclose(result, expected)

    # AC: @lora-block-config ac-1
    def test_different_strengths_per_block(self):
        """Different blocks can have different strength multipliers.

        AC: @lora-block-config ac-1
        """
        keys = [
            "input_blocks.0.0.weight",  # IN00 -> 0.5
            "middle_block.0.weight",  # MID -> 2.0
            "output_blocks.3.0.weight",  # OUT03 -> no override (1.0)
        ]
        base = torch.zeros(3, 4, 4)
        # LoRA adds 4.0 to all values
        lora_applied = torch.full((3, 4, 4), 4.0)

        config = BlockConfig(
            arch="sdxl",
            block_overrides=(
                ("IN00", 0.5),
                ("MID", 2.0),
            ),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # Check each key's result
        # IN00: delta 4.0 * 0.5 = 2.0
        assert torch.allclose(result[0], torch.full((4, 4), 2.0))
        # MID: delta 4.0 * 2.0 = 8.0
        assert torch.allclose(result[1], torch.full((4, 4), 8.0))
        # OUT03: delta 4.0 * 1.0 (default) = 4.0
        assert torch.allclose(result[2], torch.full((4, 4), 4.0))

    # AC: @lora-block-config ac-1
    def test_zero_strength_removes_lora_effect(self):
        """Strength of 0.0 completely removes the LoRA delta.

        AC: @lora-block-config ac-1
        """
        keys = ["input_blocks.0.0.weight"]
        base = torch.full((1, 4, 4), 10.0)
        lora_applied = torch.full((1, 4, 4), 20.0)  # Delta of 10.0

        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00", 0.0),),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # Delta 10.0 * 0.0 = 0.0, so result = base
        assert torch.allclose(result, base)

    # AC: @lora-block-config ac-1
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

    # AC: @lora-block-config ac-2
    def test_keys_without_override_use_default_strength(self):
        """Keys whose block group has no override entry use 1.0 (unchanged).

        AC: @lora-block-config ac-2
        """
        keys = ["time_embed.0.weight", "label_emb.weight"]  # Classify as TIME_EMBED, LABEL_EMB
        base = torch.zeros(2, 4, 4)
        lora_applied = torch.full((2, 4, 4), 5.0)

        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00", 0.5),),  # Doesn't apply to TIME_EMBED/LABEL_EMB
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # TIME_EMBED/LABEL_EMB not in overrides, so delta is unchanged
        assert torch.allclose(result, lora_applied)

    # AC: @lora-block-config ac-1
    def test_zimage_block_strength(self):
        """Z-Image architecture uses its own block classification.

        AC: @lora-block-config ac-1
        """
        keys = [
            "layers.0.attn.weight",  # L00 -> 0.25
            "layers.10.mlp.weight",  # L10 -> 1.0 (default)
            "noise_refiner.0.attn.weight",  # NOISE_REF0 -> 0.75
        ]
        base = torch.zeros(3, 4, 4)
        lora_applied = torch.full((3, 4, 4), 8.0)  # Delta of 8.0

        config = BlockConfig(
            arch="zimage",
            block_overrides=(
                ("L00", 0.25),
                ("NOISE_REF0", 0.75),
            ),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "zimage", "cpu", torch.float32
        )

        # L00: delta 8.0 * 0.25 = 2.0
        assert torch.allclose(result[0], torch.full((4, 4), 2.0))
        # L10: delta 8.0 * 1.0 = 8.0 (no override)
        assert torch.allclose(result[1], torch.full((4, 4), 8.0))
        # NOISE_REF0: delta 8.0 * 0.75 = 6.0
        assert torch.allclose(result[2], torch.full((4, 4), 6.0))

    # AC: @lora-block-config ac-1
    def test_conv2d_shapes(self):
        """Works with 4D conv2d weight tensors.

        AC: @lora-block-config ac-1
        """
        keys = ["input_blocks.0.0.weight"]
        base = torch.zeros(1, 64, 64, 3, 3)
        lora_applied = torch.full((1, 64, 64, 3, 3), 4.0)

        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00", 0.5),),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # Delta 4.0 * 0.5 = 2.0
        expected = torch.full((1, 64, 64, 3, 3), 2.0)
        assert torch.allclose(result, expected)

    # AC: @lora-block-config ac-1
    def test_preserves_negative_deltas(self):
        """Correctly handles negative LoRA deltas.

        AC: @lora-block-config ac-1
        """
        keys = ["input_blocks.0.0.weight"]
        base = torch.full((1, 4, 4), 10.0)
        lora_applied = torch.full((1, 4, 4), 6.0)  # Delta of -4.0

        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00", 0.5),),
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

    # AC: @lora-block-config ac-1
    def test_recipe_lora_stores_block_config(self):
        """RecipeLoRA correctly stores block_config.

        AC: @lora-block-config ac-1
        """
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00", 0.5),),
        )
        lora = RecipeLoRA(
            loras=({"path": "test.safetensors", "strength": 1.0},),
            block_config=config,
        )

        assert lora.block_config is config
        assert lora.block_config.block_overrides == (("IN00", 0.5),)

    # AC: @lora-block-config ac-2
    def test_recipe_lora_none_block_config(self):
        """RecipeLoRA with None block_config for backwards compatibility.

        AC: @lora-block-config ac-2
        """
        lora = RecipeLoRA(
            loras=({"path": "test.safetensors", "strength": 1.0},),
            block_config=None,
        )

        assert lora.block_config is None

    # AC: @lora-block-config ac-2
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

    # AC: @lora-block-config ac-2
    def test_no_block_config_no_scaling(self):
        """Without block_config, LoRA deltas are applied without scaling.

        AC: @lora-block-config ac-2
        Given: no BLOCK_CONFIG connected to LoRA node
        When: Exit applies LoRA deltas
        Then: global strength applies uniformly (backwards compatible)
        """
        from lib.executor import evaluate_recipe

        lora = RecipeLoRA(
            loras=({"path": "test.safetensors", "strength": 0.8},),
            block_config=None,
        )
        assert lora.block_config is None

        # Run through evaluate_recipe to verify uniform strength
        keys = ["input_blocks.0.0.weight", "middle_block.0.weight"]
        batch_size = 2
        base_batch = torch.zeros(batch_size, 4, 4)

        base = RecipeBase(model_patcher=None, arch="sdxl")
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        class MockLoader:
            def get_delta_specs(self, keys_arg, key_indices, set_id=None):
                return []

        class MockWIDEN:
            def __init__(self):
                self.filter_calls = []

            def filter_delta_batched(self, lora_applied, backbone):
                self.filter_calls.append(
                    {"lora_applied": lora_applied.clone(), "backbone": backbone.clone()}
                )
                return lora_applied

        loader = MockLoader()
        widen = MockWIDEN()
        set_id_map = {id(lora): "set1"}

        evaluate_recipe(
            keys=keys,
            base_batch=base_batch,
            recipe_node=merge,
            loader=loader,
            widen=widen,
            set_id_map=set_id_map,
            device="cpu",
            dtype=torch.float32,
            arch="sdxl",
        )

        # filter_delta_batched should be called once, and lora_applied
        # should be the unscaled result (uniform strength, no per-block scaling)
        assert len(widen.filter_calls) == 1
        # With no deltas from loader, lora_applied equals base_batch
        assert torch.equal(widen.filter_calls[0]["lora_applied"], base_batch)

    # AC: @lora-block-config ac-1
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


# =============================================================================
# Layer-Type Override Tests (LoRA Strength)
# =============================================================================


class TestLayerTypeLoraStrength:
    """Tests for layer_type_overrides multiplicative effect on LoRA strength.

    AC: @layer-type-filter ac-2
    AC: @layer-type-filter ac-4
    """

    # AC: @layer-type-filter ac-2
    def test_block_and_layer_type_multiplicative(self):
        """Effective strength = block_strength * layer_type_strength.

        AC: @layer-type-filter ac-2
        Given: block=0.5, attention=0.7
        Then: effective = 0.35
        """
        keys = ["input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"]  # IN01, attention
        base = torch.zeros(1, 4, 4)
        lora_applied = torch.full((1, 4, 4), 2.0)  # Delta of 2.0

        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN01", 0.5),),
            layer_type_overrides=(("attention", 0.7),),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # Delta 2.0 * (0.5 * 0.7) = 2.0 * 0.35 = 0.7
        expected = torch.full((1, 4, 4), 0.7)
        assert torch.allclose(result, expected)

    # AC: @layer-type-filter ac-2
    def test_layer_type_only_applies(self):
        """Layer type override applies when block uses default.

        attention=0.5 only → 0.5 for attention keys, 1.0 for others
        """
        keys = [
            # IN01, attention -> 0.5
            "input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
            # IN01, feed_forward -> 1.0
            "input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.weight",
        ]
        base = torch.zeros(2, 4, 4)
        lora_applied = torch.full((2, 4, 4), 4.0)  # Delta of 4.0

        config = BlockConfig(
            arch="sdxl",
            block_overrides=(),  # No block overrides
            layer_type_overrides=(("attention", 0.5),),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # attention: delta 4.0 * 0.5 = 2.0
        assert torch.allclose(result[0], torch.full((4, 4), 2.0))
        # feed_forward: delta 4.0 * 1.0 = 4.0 (no layer type override)
        assert torch.allclose(result[1], torch.full((4, 4), 4.0))

    # AC: @layer-type-filter ac-4
    def test_empty_layer_type_overrides_backwards_compatible(self):
        """Empty layer_type_overrides means behavior identical to before.

        AC: @layer-type-filter ac-4
        """
        keys = ["input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"]
        base = torch.zeros(1, 4, 4)
        lora_applied = torch.full((1, 4, 4), 4.0)

        # BlockConfig with only block overrides (layer_type_overrides empty by default)
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN01", 0.5),),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # Only block override applies: delta 4.0 * 0.5 = 2.0
        expected = torch.full((1, 4, 4), 2.0)
        assert torch.allclose(result, expected)

    # AC: @layer-type-filter ac-2
    def test_layer_type_zero_disables(self):
        """layer_type=0.0 disables that layer type entirely.

        AC: @layer-type-filter ac-2
        """
        keys = ["input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"]  # attention
        base = torch.full((1, 4, 4), 10.0)
        lora_applied = torch.full((1, 4, 4), 20.0)  # Delta of 10.0

        config = BlockConfig(
            arch="sdxl",
            block_overrides=(),
            layer_type_overrides=(("attention", 0.0),),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # Delta 10.0 * 0.0 = 0.0, so result = base
        assert torch.allclose(result, base)

    # AC: @layer-type-filter ac-2
    def test_all_layer_types_at_one_no_effect(self):
        """All layer types at 1.0 has no effect (identity).

        AC: @layer-type-filter ac-4
        """
        keys = ["input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"]
        base = torch.zeros(1, 4, 4)
        lora_applied = torch.full((1, 4, 4), 4.0)

        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN01", 0.5),),
            layer_type_overrides=(
                ("attention", 1.0),
                ("feed_forward", 1.0),
                ("norm", 1.0),
            ),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "sdxl", "cpu", torch.float32
        )

        # Block only: delta 4.0 * 0.5 * 1.0 = 2.0
        expected = torch.full((1, 4, 4), 2.0)
        assert torch.allclose(result, expected)

    # AC: @layer-type-filter ac-2
    def test_zimage_layer_type_multiplicative(self):
        """Z-Image layer type overrides work multiplicatively."""
        keys = [
            "layers.5.attn.qkv.weight",  # L05, attention
            "layers.5.feed_forward.w1.weight",  # L05, feed_forward
            "layers.5.norm.weight",  # L05, norm
        ]
        base = torch.zeros(3, 4, 4)
        lora_applied = torch.full((3, 4, 4), 4.0)

        config = BlockConfig(
            arch="zimage",
            block_overrides=(("L05", 0.8),),
            layer_type_overrides=(
                ("attention", 0.5),  # 0.8 * 0.5 = 0.4
                ("feed_forward", 1.5),  # 0.8 * 1.5 = 1.2
                ("norm", 0.0),  # 0.8 * 0.0 = 0.0
            ),
        )

        result = _apply_per_block_lora_strength(
            keys, base, lora_applied, config, "zimage", "cpu", torch.float32
        )

        # attention: 4.0 * 0.4 = 1.6
        assert torch.allclose(result[0], torch.full((4, 4), 1.6))
        # feed_forward: 4.0 * 1.2 = 4.8
        assert torch.allclose(result[1], torch.full((4, 4), 4.8))
        # norm: 4.0 * 0.0 = 0.0
        assert torch.allclose(result[2], torch.full((4, 4), 0.0))
