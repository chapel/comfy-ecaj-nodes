"""Per-Block Control feature tests — ComfyUI nodes for block weight configuration.

Tests for @per-block-control acceptance criteria:
- AC-1: No BLOCK_CONFIG inputs → behavior identical to pre-block-control
- AC-2: Architecture-specific nodes expose block sliders with float range 0.0-2.0
- AC-3: Single BLOCK_CONFIG output fans out correctly to multiple consumers
- AC-4: SDXL node has 19 individual blocks (IN00-IN08, MID, OUT00-OUT08)
- AC-5: Z-Image node has 34 individual blocks (L00-L29, NOISE_REF0-1, CTX_REF0-1)
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
    # AC: @per-block-control ac-4
    # AC: @layer-type-filter ac-5
    """

    # AC: @per-block-control ac-4
    def test_input_types_has_all_individual_blocks(self):
        """SDXL node exposes all 19 individual block sliders."""
        input_types = WIDENBlockConfigSDXLNode.INPUT_TYPES()
        required = input_types["required"]

        expected_blocks = [
            *[f"IN{i:02d}" for i in range(9)],
            "MID",
            *[f"OUT{i:02d}" for i in range(9)],
        ]
        assert len(expected_blocks) == 19
        for block in expected_blocks:
            assert block in required, f"Missing individual block slider: {block}"

    # AC: @layer-type-filter ac-5
    def test_input_types_has_layer_type_sliders(self):
        """SDXL node exposes attention, feed_forward, norm sliders."""
        input_types = WIDENBlockConfigSDXLNode.INPUT_TYPES()
        required = input_types["required"]

        expected_layer_types = ["attention", "feed_forward", "norm"]
        for lt in expected_layer_types:
            assert lt in required, f"Missing layer type slider: {lt}"

    # AC: @layer-type-filter ac-5
    def test_total_input_count(self):
        """SDXL node has 19 blocks + 3 layer types = 22 total inputs."""
        input_types = WIDENBlockConfigSDXLNode.INPUT_TYPES()
        required = input_types["required"]
        assert len(required) == 22

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
        # Build kwargs for all 19 blocks + layer types
        kwargs = {f"IN{i:02d}": 1.0 for i in range(9)}
        kwargs["MID"] = 1.0
        kwargs.update({f"OUT{i:02d}": 1.0 for i in range(9)})
        kwargs.update({"attention": 1.0, "feed_forward": 1.0, "norm": 1.0})
        kwargs["IN00"] = 0.5  # Override one to verify

        result = node.create_config(**kwargs)

        assert len(result) == 1
        config = result[0]
        assert isinstance(config, BlockConfig)
        assert config.arch == "sdxl"

    # AC: @per-block-control ac-4
    def test_create_config_stores_block_overrides(self):
        """create_config stores all 19 block overrides as tuple of pairs."""
        node = WIDENBlockConfigSDXLNode()
        # Build kwargs for all 19 blocks with distinct values + layer types
        kwargs = {f"IN{i:02d}": 0.5 + i * 0.05 for i in range(9)}
        kwargs["MID"] = 1.2
        kwargs.update({f"OUT{i:02d}": 1.0 + i * 0.05 for i in range(9)})
        kwargs.update({"attention": 1.0, "feed_forward": 1.0, "norm": 1.0})

        (config,) = node.create_config(**kwargs)

        assert len(config.block_overrides) == 19
        assert config.block_overrides[0] == ("IN00", 0.5)
        assert config.block_overrides[9] == ("MID", 1.2)
        assert config.block_overrides[10] == ("OUT00", 1.0)

    # AC: @layer-type-filter ac-5
    def test_create_config_stores_layer_type_overrides(self):
        """create_config stores layer_type_overrides as tuple of pairs."""
        node = WIDENBlockConfigSDXLNode()
        # Build kwargs for all blocks + layer types
        kwargs = {f"IN{i:02d}": 1.0 for i in range(9)}
        kwargs["MID"] = 1.0
        kwargs.update({f"OUT{i:02d}": 1.0 for i in range(9)})
        kwargs.update({"attention": 0.5, "feed_forward": 1.2, "norm": 0.8})

        (config,) = node.create_config(**kwargs)

        assert len(config.layer_type_overrides) == 3
        assert config.layer_type_overrides[0] == ("attention", 0.5)
        assert config.layer_type_overrides[1] == ("feed_forward", 1.2)
        assert config.layer_type_overrides[2] == ("norm", 0.8)

    def test_create_config_with_boundary_values(self):
        """create_config handles boundary values (0.0, 2.0)."""
        node = WIDENBlockConfigSDXLNode()
        # All defaults except boundary test blocks
        kwargs = {f"IN{i:02d}": 1.0 for i in range(9)}
        kwargs["MID"] = 2.0
        kwargs.update({f"OUT{i:02d}": 1.0 for i in range(9)})
        kwargs["IN00"] = 0.0
        kwargs["IN01"] = 2.0
        kwargs.update({"attention": 1.0, "feed_forward": 1.0, "norm": 1.0})

        (config,) = node.create_config(**kwargs)

        assert config.block_overrides[0] == ("IN00", 0.0)
        assert config.block_overrides[1] == ("IN01", 2.0)


class TestBlockConfigZImageNode:
    """WIDENBlockConfigZImage node tests.
    # AC: @per-block-control ac-2
    # AC: @per-block-control ac-5
    # AC: @layer-type-filter ac-5
    """

    # AC: @per-block-control ac-5
    def test_input_types_has_all_individual_blocks(self):
        """Z-Image node exposes all 34 individual block sliders."""
        input_types = WIDENBlockConfigZImageNode.INPUT_TYPES()
        required = input_types["required"]

        expected_blocks = [
            *[f"L{i:02d}" for i in range(30)],
            "NOISE_REF0",
            "NOISE_REF1",
            "CTX_REF0",
            "CTX_REF1",
        ]
        assert len(expected_blocks) == 34
        for block in expected_blocks:
            assert block in required, f"Missing individual block slider: {block}"

    # AC: @layer-type-filter ac-5
    def test_input_types_has_layer_type_sliders(self):
        """Z-Image node exposes attention, feed_forward, norm sliders."""
        input_types = WIDENBlockConfigZImageNode.INPUT_TYPES()
        required = input_types["required"]

        expected_layer_types = ["attention", "feed_forward", "norm"]
        for lt in expected_layer_types:
            assert lt in required, f"Missing layer type slider: {lt}"

    # AC: @layer-type-filter ac-5
    def test_total_input_count(self):
        """Z-Image node has 34 blocks + 3 layer types = 37 total inputs."""
        input_types = WIDENBlockConfigZImageNode.INPUT_TYPES()
        required = input_types["required"]
        assert len(required) == 37

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
        # Build kwargs for all 34 blocks + layer types
        kwargs = {f"L{i:02d}": 1.0 for i in range(30)}
        kwargs.update(
            {
                "NOISE_REF0": 1.0,
                "NOISE_REF1": 1.0,
                "CTX_REF0": 1.0,
                "CTX_REF1": 1.0,
            }
        )
        kwargs.update({"attention": 1.0, "feed_forward": 1.0, "norm": 1.0})
        kwargs["L00"] = 0.5  # Override one to verify

        result = node.create_config(**kwargs)

        assert len(result) == 1
        config = result[0]
        assert isinstance(config, BlockConfig)
        assert config.arch == "zimage"

    # AC: @per-block-control ac-5
    def test_create_config_stores_block_overrides(self):
        """create_config stores all 34 block overrides as tuple of pairs."""
        node = WIDENBlockConfigZImageNode()
        # Build kwargs for all 34 blocks with distinct values + layer types
        kwargs = {f"L{i:02d}": 0.5 + i * 0.02 for i in range(30)}
        kwargs.update(
            {
                "NOISE_REF0": 1.1,
                "NOISE_REF1": 1.2,
                "CTX_REF0": 0.9,
                "CTX_REF1": 0.8,
            }
        )
        kwargs.update({"attention": 1.0, "feed_forward": 1.0, "norm": 1.0})

        (config,) = node.create_config(**kwargs)

        assert len(config.block_overrides) == 34
        assert config.block_overrides[0] == ("L00", 0.5)
        assert config.block_overrides[25] == ("L25", 1.0)  # 0.5 + 25*0.02 = 1.0
        assert config.block_overrides[30] == ("NOISE_REF0", 1.1)
        assert config.block_overrides[33] == ("CTX_REF1", 0.8)

    # AC: @layer-type-filter ac-5
    def test_create_config_stores_layer_type_overrides(self):
        """create_config stores layer_type_overrides as tuple of pairs."""
        node = WIDENBlockConfigZImageNode()
        # Build kwargs for all blocks + layer types
        kwargs = {f"L{i:02d}": 1.0 for i in range(30)}
        kwargs.update(
            {
                "NOISE_REF0": 1.0,
                "NOISE_REF1": 1.0,
                "CTX_REF0": 1.0,
                "CTX_REF1": 1.0,
            }
        )
        kwargs.update({"attention": 0.6, "feed_forward": 1.4, "norm": 0.9})

        (config,) = node.create_config(**kwargs)

        assert len(config.layer_type_overrides) == 3
        assert config.layer_type_overrides[0] == ("attention", 0.6)
        assert config.layer_type_overrides[1] == ("feed_forward", 1.4)
        assert config.layer_type_overrides[2] == ("norm", 0.9)


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
            block_overrides=(("IN00", 0.5), ("MID", 1.0)),
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
            block_overrides=(("OUT00", 0.8),),
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
            block_overrides=(("L00", 0.5), ("NOISE_REF0", 1.2)),
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
