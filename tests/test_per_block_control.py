"""Per-Block Control feature tests — ComfyUI nodes for block weight configuration.

Tests for @per-block-control acceptance criteria:
- AC-1: No BLOCK_CONFIG inputs → behavior identical to pre-block-control
- AC-2: Architecture-specific nodes expose block group sliders with float range 0.0-2.0
- AC-3: Single BLOCK_CONFIG output fans out correctly to multiple consumers
"""

import pytest

from lib.recipe import BlockConfig, RecipeBase, RecipeLoRA, RecipeMerge
from nodes.block_config_sdxl import WIDENBlockConfigSDXLNode
from nodes.block_config_zimage import WIDENBlockConfigZImageNode
from nodes.lora import WIDENLoRANode
from nodes.merge import WIDENMergeNode


# Test fixture mocks
class MockFolderPaths:
    """Mock folder_paths for ComfyUI runtime."""

    @staticmethod
    def get_filename_list(folder: str):
        return ["test_lora.safetensors"]


@pytest.fixture(autouse=True)
def mock_folder_paths(monkeypatch):
    """Mock folder_paths module for all tests."""
    import sys

    sys.modules["folder_paths"] = MockFolderPaths()
    yield
    del sys.modules["folder_paths"]


class TestBlockConfigSDXLNode:
    """WIDENBlockConfigSDXL node tests.
    # AC: @per-block-control ac-2
    """

    def test_input_types_has_all_block_groups(self):
        """SDXL node exposes all 7 block group sliders."""
        input_types = WIDENBlockConfigSDXLNode.INPUT_TYPES()
        required = input_types["required"]

        expected_blocks = [
            "IN00_02", "IN03_05", "IN06_08", "MID", "OUT00_02", "OUT03_05", "OUT06_08"
        ]
        for block in expected_blocks:
            assert block in required, f"Missing block group slider: {block}"

    def test_input_types_slider_config(self):
        """Each slider has correct FLOAT config with range 0.0-2.0."""
        input_types = WIDENBlockConfigSDXLNode.INPUT_TYPES()
        required = input_types["required"]

        for name, config in required.items():
            assert config[0] == "FLOAT", f"{name} should be FLOAT type"
            opts = config[1]
            assert opts["default"] == 1.0, f"{name} default should be 1.0"
            assert opts["min"] == 0.0, f"{name} min should be 0.0"
            assert opts["max"] == 2.0, f"{name} max should be 2.0"
            assert opts["step"] == 0.05, f"{name} step should be 0.05"

    def test_return_types(self):
        """Node returns BLOCK_CONFIG type."""
        assert WIDENBlockConfigSDXLNode.RETURN_TYPES == ("BLOCK_CONFIG",)
        assert WIDENBlockConfigSDXLNode.RETURN_NAMES == ("block_config",)

    def test_create_config_returns_block_config(self):
        """create_config returns BlockConfig with sdxl arch."""
        node = WIDENBlockConfigSDXLNode()
        result = node.create_config(
            IN00_02=0.5,
            IN03_05=0.8,
            IN06_08=1.0,
            MID=1.2,
            OUT00_02=1.5,
            OUT03_05=0.9,
            OUT06_08=1.1,
        )

        assert len(result) == 1
        config = result[0]
        assert isinstance(config, BlockConfig)
        assert config.arch == "sdxl"

    def test_create_config_stores_block_overrides(self):
        """create_config stores all block overrides as tuple of pairs."""
        node = WIDENBlockConfigSDXLNode()
        (config,) = node.create_config(
            IN00_02=0.5,
            IN03_05=0.8,
            IN06_08=1.0,
            MID=1.2,
            OUT00_02=1.5,
            OUT03_05=0.9,
            OUT06_08=1.1,
        )

        assert len(config.block_overrides) == 7
        assert config.block_overrides[0] == ("IN00-02", 0.5)
        assert config.block_overrides[3] == ("MID", 1.2)
        assert config.block_overrides[6] == ("OUT06-08", 1.1)

    def test_create_config_with_boundary_values(self):
        """create_config handles boundary values (0.0, 2.0)."""
        node = WIDENBlockConfigSDXLNode()
        (config,) = node.create_config(
            IN00_02=0.0,
            IN03_05=2.0,
            IN06_08=0.0,
            MID=2.0,
            OUT00_02=0.0,
            OUT03_05=2.0,
            OUT06_08=0.0,
        )

        assert config.block_overrides[0] == ("IN00-02", 0.0)
        assert config.block_overrides[1] == ("IN03-05", 2.0)


class TestBlockConfigZImageNode:
    """WIDENBlockConfigZImage node tests.
    # AC: @per-block-control ac-2
    """

    def test_input_types_has_all_block_groups(self):
        """Z-Image node exposes all 8 block group sliders."""
        input_types = WIDENBlockConfigZImageNode.INPUT_TYPES()
        required = input_types["required"]

        expected_blocks = [
            "L00_04",
            "L05_09",
            "L10_14",
            "L15_19",
            "L20_24",
            "L25_29",
            "noise_refiner",
            "context_refiner",
        ]
        for block in expected_blocks:
            assert block in required, f"Missing block group slider: {block}"

    def test_input_types_slider_config(self):
        """Each slider has correct FLOAT config with range 0.0-2.0."""
        input_types = WIDENBlockConfigZImageNode.INPUT_TYPES()
        required = input_types["required"]

        for name, config in required.items():
            assert config[0] == "FLOAT", f"{name} should be FLOAT type"
            opts = config[1]
            assert opts["default"] == 1.0, f"{name} default should be 1.0"
            assert opts["min"] == 0.0, f"{name} min should be 0.0"
            assert opts["max"] == 2.0, f"{name} max should be 2.0"
            assert opts["step"] == 0.05, f"{name} step should be 0.05"

    def test_return_types(self):
        """Node returns BLOCK_CONFIG type."""
        assert WIDENBlockConfigZImageNode.RETURN_TYPES == ("BLOCK_CONFIG",)
        assert WIDENBlockConfigZImageNode.RETURN_NAMES == ("block_config",)

    def test_create_config_returns_block_config(self):
        """create_config returns BlockConfig with zimage arch."""
        node = WIDENBlockConfigZImageNode()
        result = node.create_config(
            L00_04=0.5,
            L05_09=0.8,
            L10_14=1.0,
            L15_19=1.2,
            L20_24=1.5,
            L25_29=0.9,
            noise_refiner=1.1,
            context_refiner=0.7,
        )

        assert len(result) == 1
        config = result[0]
        assert isinstance(config, BlockConfig)
        assert config.arch == "zimage"

    def test_create_config_stores_block_overrides(self):
        """create_config stores all block overrides as tuple of pairs."""
        node = WIDENBlockConfigZImageNode()
        (config,) = node.create_config(
            L00_04=0.5,
            L05_09=0.8,
            L10_14=1.0,
            L15_19=1.2,
            L20_24=1.5,
            L25_29=0.9,
            noise_refiner=1.1,
            context_refiner=0.7,
        )

        assert len(config.block_overrides) == 8
        assert config.block_overrides[0] == ("L00-04", 0.5)
        assert config.block_overrides[5] == ("L25-29", 0.9)
        assert config.block_overrides[6] == ("noise_refiner", 1.1)
        assert config.block_overrides[7] == ("context_refiner", 0.7)


class TestNoBlockConfigBehavior:
    """AC-1: No BLOCK_CONFIG inputs → behavior identical to pre-block-control.
    # AC: @per-block-control ac-1
    """

    def test_lora_node_no_block_config_default(self):
        """LoRA node without block_config has None block_config."""
        node = WIDENLoRANode()
        (lora,) = node.add_lora("test_lora.safetensors", 1.0)

        assert isinstance(lora, RecipeLoRA)
        assert lora.block_config is None

    def test_lora_node_explicit_none(self):
        """LoRA node with explicit None block_config works."""
        node = WIDENLoRANode()
        (lora,) = node.add_lora("test_lora.safetensors", 1.0, block_config=None)

        assert lora.block_config is None

    def test_merge_node_no_block_config_default(self):
        """Merge node without block_config has None block_config."""
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))

        node = WIDENMergeNode()
        (merge,) = node.merge(base, lora, 1.0)

        assert isinstance(merge, RecipeMerge)
        assert merge.block_config is None

    def test_merge_node_explicit_none(self):
        """Merge node with explicit None block_config works."""
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))

        node = WIDENMergeNode()
        (merge,) = node.merge(base, lora, 1.0, block_config=None)

        assert merge.block_config is None


class TestBlockConfigFanOut:
    """AC-3: Single BLOCK_CONFIG output fans out correctly to multiple consumers.
    # AC: @per-block-control ac-3
    """

    def test_same_block_config_to_multiple_loras(self):
        """Same BlockConfig can be used by multiple LoRA nodes."""
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00-02", 0.5), ("MID", 1.0)),
        )

        node = WIDENLoRANode()
        (lora_a,) = node.add_lora("lora_a.safetensors", 1.0, block_config=config)
        (lora_b,) = node.add_lora("lora_b.safetensors", 0.8, block_config=config)

        # Both reference the same BlockConfig
        assert lora_a.block_config is config
        assert lora_b.block_config is config
        assert lora_a.block_config is lora_b.block_config

    def test_same_block_config_to_multiple_merges(self):
        """Same BlockConfig can be used by multiple Merge nodes."""
        config = BlockConfig(
            arch="sdxl",
            block_overrides=(("OUT00-02", 0.8),),
        )
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        lora_a = RecipeLoRA(loras=({"path": "lora_a.safetensors", "strength": 1.0},))
        lora_b = RecipeLoRA(loras=({"path": "lora_b.safetensors", "strength": 0.5},))

        node = WIDENMergeNode()
        (merge_a,) = node.merge(base, lora_a, 1.0, block_config=config)
        (merge_b,) = node.merge(merge_a, lora_b, 0.8, block_config=config)

        # Both reference the same BlockConfig
        assert merge_a.block_config is config
        assert merge_b.block_config is config

    def test_block_config_to_mixed_consumers(self):
        """Same BlockConfig can be used by both LoRA and Merge nodes."""
        config = BlockConfig(
            arch="zimage",
            block_overrides=(("L00-04", 0.5), ("noise_refiner", 1.2)),
        )
        base = RecipeBase(model_patcher=object(), arch="zimage")

        lora_node = WIDENLoRANode()
        merge_node = WIDENMergeNode()

        (lora,) = lora_node.add_lora("test.safetensors", 1.0, block_config=config)
        (merge,) = merge_node.merge(base, lora, 1.0, block_config=config)

        # Both reference the same BlockConfig
        assert lora.block_config is config
        assert merge.block_config is config


class TestLoRANodeBlockConfigChaining:
    """LoRA node block_config chaining behavior."""

    def test_chained_lora_inherits_block_config(self):
        """Chained LoRA inherits block_config from prev if not provided."""
        config = BlockConfig(arch="sdxl", block_overrides=(("MID", 0.5),))

        node = WIDENLoRANode()
        (first,) = node.add_lora("first.safetensors", 1.0, block_config=config)
        (second,) = node.add_lora("second.safetensors", 0.8, prev=first)

        assert first.block_config is config
        assert second.block_config is config  # Inherited from prev

    def test_chained_lora_new_config_overrides(self):
        """Chained LoRA with new block_config overrides prev config."""
        config_a = BlockConfig(arch="sdxl", block_overrides=(("MID", 0.5),))
        config_b = BlockConfig(arch="sdxl", block_overrides=(("MID", 0.8),))

        node = WIDENLoRANode()
        (first,) = node.add_lora("first.safetensors", 1.0, block_config=config_a)
        (second,) = node.add_lora("second.safetensors", 0.8, prev=first, block_config=config_b)

        assert first.block_config is config_a
        assert second.block_config is config_b  # New config overrides


class TestInputTypesIncludeBlockConfig:
    """Nodes include BLOCK_CONFIG in optional inputs."""

    def test_lora_node_has_block_config_input(self):
        """LoRA node has optional BLOCK_CONFIG input."""
        input_types = WIDENLoRANode.INPUT_TYPES()
        assert "block_config" in input_types["optional"]
        assert input_types["optional"]["block_config"] == ("BLOCK_CONFIG",)

    def test_merge_node_has_block_config_input(self):
        """Merge node has optional BLOCK_CONFIG input."""
        input_types = WIDENMergeNode.INPUT_TYPES()
        assert "block_config" in input_types["optional"]
        assert input_types["optional"]["block_config"] == ("BLOCK_CONFIG",)
