"""BlockConfig dataclass tests — frozen immutability, tuple storage, integration.

Tests for @block-config-type acceptance criteria:
- AC-1: BlockConfig is frozen and stores per-block float values as tuple of pairs
- AC-2: RecipeLoRA and RecipeMerge accept BlockConfig or None for block_config field
"""

import pytest

from lib.recipe import (
    BlockConfig,
    RecipeBase,
    RecipeCompose,
    RecipeLoRA,
    RecipeMerge,
)


class TestBlockConfigFrozen:
    """AC-1: BlockConfig is frozen and stores per-block float values as tuple of pairs.
    # AC: @block-config-type ac-1
    """

    def test_block_config_is_frozen(self):
        """BlockConfig instances are immutable."""
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00", 0.5), ("MID", 1.0)),
        )
        with pytest.raises((AttributeError, TypeError)):
            config.arch = "flux"

    def test_block_config_arch_field(self):
        """BlockConfig stores arch string."""
        config = BlockConfig(arch="sdxl", block_overrides=())
        assert config.arch == "sdxl"

    def test_block_config_block_overrides_tuple(self):
        """BlockConfig stores block_overrides as tuple of pairs."""
        overrides = (("IN00", 0.5), ("MID", 1.0), ("OUT00", 0.8))
        config = BlockConfig(arch="sdxl", block_overrides=overrides)
        assert config.block_overrides == overrides
        assert isinstance(config.block_overrides, tuple)

    def test_block_config_layer_type_overrides_default_empty(self):
        """BlockConfig layer_type_overrides defaults to empty tuple."""
        config = BlockConfig(arch="sdxl", block_overrides=())
        assert config.layer_type_overrides == ()

    def test_block_config_layer_type_overrides_custom(self):
        """BlockConfig stores layer_type_overrides as tuple of pairs."""
        layer_overrides = (("attention", 0.7), ("feed_forward", 1.0))
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(),
            layer_type_overrides=layer_overrides,
        )
        assert config.layer_type_overrides == layer_overrides
        assert isinstance(config.layer_type_overrides, tuple)

    def test_block_config_block_overrides_immutable(self):
        """block_overrides field cannot be reassigned."""
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00", 0.5),),
        )
        with pytest.raises((AttributeError, TypeError)):
            config.block_overrides = ()

    def test_block_config_layer_type_overrides_immutable(self):
        """layer_type_overrides field cannot be reassigned."""
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(),
            layer_type_overrides=(("attention", 0.7),),
        )
        with pytest.raises((AttributeError, TypeError)):
            config.layer_type_overrides = ()


class TestBlockConfigConstruction:
    """BlockConfig construction scenarios.
    # AC: @block-config-type ac-1
    """

    def test_block_config_minimal_construction(self):
        """BlockConfig constructible with just arch and empty block_overrides."""
        config = BlockConfig(arch="flux", block_overrides=())
        assert config.arch == "flux"
        assert config.block_overrides == ()
        assert config.layer_type_overrides == ()

    def test_block_config_full_construction(self):
        """BlockConfig constructible with all fields."""
        config = BlockConfig(
            arch="zimage",
            block_overrides=(("block_0", 0.3), ("block_1", 0.7)),
            layer_type_overrides=(("norm", 0.5),),
        )
        assert config.arch == "zimage"
        assert len(config.block_overrides) == 2
        assert len(config.layer_type_overrides) == 1

    def test_block_config_different_architectures(self):
        """BlockConfig works with different architecture values."""
        for arch in ("sdxl", "zimage", "flux", "qwen"):
            config = BlockConfig(arch=arch, block_overrides=())
            assert config.arch == arch


class TestRecipeLoRABlockConfig:
    """AC-2: RecipeLoRA accepts BlockConfig or None for block_config field.
    # AC: @block-config-type ac-2
    """

    def test_recipe_lora_block_config_none_default(self):
        """RecipeLoRA block_config defaults to None."""
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        assert lora.block_config is None

    def test_recipe_lora_block_config_none_explicit(self):
        """RecipeLoRA accepts explicit None for block_config."""
        lora = RecipeLoRA(
            loras=({"path": "test.safetensors", "strength": 1.0},),
            block_config=None,
        )
        assert lora.block_config is None

    def test_recipe_lora_block_config_with_config(self):
        """RecipeLoRA accepts BlockConfig instance."""
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00", 0.5),),
        )
        lora = RecipeLoRA(
            loras=({"path": "test.safetensors", "strength": 1.0},),
            block_config=config,
        )
        assert lora.block_config is config
        assert lora.block_config.arch == "sdxl"

    def test_recipe_lora_block_config_immutable(self):
        """RecipeLoRA block_config field cannot be reassigned."""
        lora = RecipeLoRA(
            loras=({"path": "test.safetensors", "strength": 1.0},),
            block_config=None,
        )
        with pytest.raises((AttributeError, TypeError)):
            lora.block_config = BlockConfig(arch="sdxl", block_overrides=())


class TestRecipeMergeBlockConfig:
    """AC-2: RecipeMerge accepts BlockConfig or None for block_config field.
    # AC: @block-config-type ac-2
    """

    def test_recipe_merge_block_config_none_default(self):
        """RecipeMerge block_config defaults to None."""
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)
        assert merge.block_config is None

    def test_recipe_merge_block_config_none_explicit(self):
        """RecipeMerge accepts explicit None for block_config."""
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(
            base=base, target=lora, backbone=None, t_factor=1.0, block_config=None
        )
        assert merge.block_config is None

    def test_recipe_merge_block_config_with_config(self):
        """RecipeMerge accepts BlockConfig instance."""
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("MID", 0.8), ("OUT00", 1.0)),
        )
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(
            base=base, target=lora, backbone=None, t_factor=1.0, block_config=config
        )
        assert merge.block_config is config
        assert merge.block_config.arch == "sdxl"
        assert len(merge.block_config.block_overrides) == 2

    def test_recipe_merge_block_config_immutable(self):
        """RecipeMerge block_config field cannot be reassigned."""
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)
        with pytest.raises((AttributeError, TypeError)):
            merge.block_config = BlockConfig(arch="sdxl", block_overrides=())


class TestBlockConfigIntegration:
    """Integration tests for BlockConfig with recipe tree.
    # AC: @block-config-type ac-1, ac-2
    """

    def test_recipe_tree_with_block_configs(self):
        """Full recipe tree can use BlockConfig at multiple levels."""
        lora_config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00", 0.5),),
        )
        merge_config = BlockConfig(
            arch="sdxl",
            block_overrides=(("MID", 1.0),),
        )

        base = RecipeBase(model_patcher=object(), arch="sdxl")
        lora = RecipeLoRA(
            loras=({"path": "lora_a.safetensors", "strength": 1.0},),
            block_config=lora_config,
        )
        merge = RecipeMerge(
            base=base, target=lora, backbone=None, t_factor=1.0, block_config=merge_config
        )

        # Verify tree structure
        assert merge.block_config is merge_config
        assert merge.target.block_config is lora_config

    def test_compose_with_block_config_loras(self):
        """RecipeCompose can hold RecipeLoRA instances with BlockConfig."""
        config_a = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        config_b = BlockConfig(arch="sdxl", block_overrides=(("OUT00", 0.8),))

        lora_a = RecipeLoRA(
            loras=({"path": "lora_a.safetensors", "strength": 1.0},),
            block_config=config_a,
        )
        lora_b = RecipeLoRA(
            loras=({"path": "lora_b.safetensors", "strength": 0.5},),
            block_config=config_b,
        )
        compose = RecipeCompose(branches=(lora_a, lora_b))

        assert compose.branches[0].block_config is config_a
        assert compose.branches[1].block_config is config_b

    def test_block_config_equality(self):
        """BlockConfig instances with same values are equal (dataclass behavior)."""
        config_a = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00", 0.5),),
        )
        config_b = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00", 0.5),),
        )
        assert config_a == config_b

    def test_block_config_inequality(self):
        """BlockConfig instances with different values are not equal."""
        config_a = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        config_b = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.8),))
        config_c = BlockConfig(arch="flux", block_overrides=(("IN00", 0.5),))
        assert config_a != config_b
        assert config_a != config_c
