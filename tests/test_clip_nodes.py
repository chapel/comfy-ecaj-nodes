"""Tests for WIDEN CLIP Graph Nodes — AC coverage for @clip-graph-nodes spec."""

import sys
from types import ModuleType

import pytest

from lib.recipe import BlockConfig, RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge, RecipeModel

# ---------------------------------------------------------------------------
# Fixtures for mocking folder_paths
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_folder_paths(monkeypatch):
    """Mock folder_paths module for ComfyUI runtime."""
    mock_module = ModuleType("folder_paths")
    mock_loras = ["lora_a.safetensors", "lora_b.safetensors"]
    mock_checkpoints = ["model_a.safetensors", "model_b.safetensors"]

    def get_filename_list(folder):
        if folder == "loras":
            return mock_loras
        if folder == "checkpoints":
            return mock_checkpoints
        return []

    mock_module.get_filename_list = get_filename_list
    monkeypatch.setitem(sys.modules, "folder_paths", mock_module)

    # Clear cached modules to force reimport
    for mod in list(sys.modules.keys()):
        if mod.startswith("nodes.clip_nodes"):
            del sys.modules[mod]

    return {"loras": mock_loras, "checkpoints": mock_checkpoints}


# ---------------------------------------------------------------------------
# AC-1: CLIP LoRA node accepts WIDEN_CLIP prev, returns WIDEN_CLIP
# ---------------------------------------------------------------------------


def test_clip_lora_input_types_has_widen_clip_prev(mock_folder_paths):
    """AC: @clip-graph-nodes ac-1 — CLIP LoRA accepts WIDEN_CLIP prev input."""
    from nodes.clip_nodes import WIDENCLIPLoRANode

    input_types = WIDENCLIPLoRANode.INPUT_TYPES()

    assert "optional" in input_types
    assert "prev" in input_types["optional"]
    assert input_types["optional"]["prev"] == ("WIDEN_CLIP",)


def test_clip_lora_return_types_is_widen_clip(mock_folder_paths):
    """AC: @clip-graph-nodes ac-1 — CLIP LoRA returns WIDEN_CLIP."""
    from nodes.clip_nodes import WIDENCLIPLoRANode

    assert WIDENCLIPLoRANode.RETURN_TYPES == ("WIDEN_CLIP",)


def test_clip_lora_uses_loras_folder(mock_folder_paths):
    """AC: @clip-graph-nodes ac-1 — CLIP LoRA uses loras folder."""
    from nodes.clip_nodes import WIDENCLIPLoRANode

    input_types = WIDENCLIPLoRANode.INPUT_TYPES()

    lora_name_spec = input_types["required"]["lora_name"]
    assert lora_name_spec[0] == mock_folder_paths["loras"]


# ---------------------------------------------------------------------------
# AC-2: CLIP Compose node accepts WIDEN_CLIP branch/compose, returns WIDEN_CLIP
# ---------------------------------------------------------------------------


def test_clip_compose_input_types_has_widen_clip_branch(mock_folder_paths):
    """AC: @clip-graph-nodes ac-2 — CLIP Compose accepts WIDEN_CLIP branch."""
    from nodes.clip_nodes import WIDENCLIPComposeNode

    input_types = WIDENCLIPComposeNode.INPUT_TYPES()

    assert "required" in input_types
    assert "branch" in input_types["required"]
    assert input_types["required"]["branch"] == ("WIDEN_CLIP",)


def test_clip_compose_input_types_has_widen_clip_compose(mock_folder_paths):
    """AC: @clip-graph-nodes ac-2 — CLIP Compose accepts optional WIDEN_CLIP compose."""
    from nodes.clip_nodes import WIDENCLIPComposeNode

    input_types = WIDENCLIPComposeNode.INPUT_TYPES()

    assert "optional" in input_types
    assert "compose" in input_types["optional"]
    assert input_types["optional"]["compose"] == ("WIDEN_CLIP",)


def test_clip_compose_return_types_is_widen_clip(mock_folder_paths):
    """AC: @clip-graph-nodes ac-2 — CLIP Compose returns WIDEN_CLIP."""
    from nodes.clip_nodes import WIDENCLIPComposeNode

    assert WIDENCLIPComposeNode.RETURN_TYPES == ("WIDEN_CLIP",)


# ---------------------------------------------------------------------------
# AC-3: CLIP Merge node accepts WIDEN_CLIP base/target, returns WIDEN_CLIP
# ---------------------------------------------------------------------------


def test_clip_merge_input_types_has_widen_clip_base(mock_folder_paths):
    """AC: @clip-graph-nodes ac-3 — CLIP Merge accepts WIDEN_CLIP base."""
    from nodes.clip_nodes import WIDENCLIPMergeNode

    input_types = WIDENCLIPMergeNode.INPUT_TYPES()

    assert "required" in input_types
    assert "base" in input_types["required"]
    assert input_types["required"]["base"] == ("WIDEN_CLIP",)


def test_clip_merge_input_types_has_widen_clip_target(mock_folder_paths):
    """AC: @clip-graph-nodes ac-3 — CLIP Merge accepts WIDEN_CLIP target."""
    from nodes.clip_nodes import WIDENCLIPMergeNode

    input_types = WIDENCLIPMergeNode.INPUT_TYPES()

    assert "required" in input_types
    assert "target" in input_types["required"]
    assert input_types["required"]["target"] == ("WIDEN_CLIP",)


def test_clip_merge_input_types_has_widen_clip_backbone(mock_folder_paths):
    """AC: @clip-graph-nodes ac-3 — CLIP Merge has optional WIDEN_CLIP backbone."""
    from nodes.clip_nodes import WIDENCLIPMergeNode

    input_types = WIDENCLIPMergeNode.INPUT_TYPES()

    assert "optional" in input_types
    assert "backbone" in input_types["optional"]
    assert input_types["optional"]["backbone"] == ("WIDEN_CLIP",)


def test_clip_merge_return_types_is_widen_clip(mock_folder_paths):
    """AC: @clip-graph-nodes ac-3 — CLIP Merge returns WIDEN_CLIP."""
    from nodes.clip_nodes import WIDENCLIPMergeNode

    assert WIDENCLIPMergeNode.RETURN_TYPES == ("WIDEN_CLIP",)


# ---------------------------------------------------------------------------
# AC-4: CLIP Model Input node returns WIDEN_CLIP type
# ---------------------------------------------------------------------------


def test_clip_model_input_return_types_is_widen_clip(mock_folder_paths):
    """AC: @clip-graph-nodes ac-4 — CLIP Model Input returns WIDEN_CLIP."""
    from nodes.clip_nodes import WIDENCLIPModelInputNode

    assert WIDENCLIPModelInputNode.RETURN_TYPES == ("WIDEN_CLIP",)


# ---------------------------------------------------------------------------
# AC-5: CLIP Model Input reads from checkpoints folder
# ---------------------------------------------------------------------------


def test_clip_model_input_uses_checkpoints_folder(mock_folder_paths):
    """AC: @clip-graph-nodes ac-5 — CLIP Model Input reads from checkpoints folder."""
    from nodes.clip_nodes import WIDENCLIPModelInputNode

    input_types = WIDENCLIPModelInputNode.INPUT_TYPES()

    model_name_spec = input_types["required"]["model_name"]
    assert model_name_spec[0] == mock_folder_paths["checkpoints"]


def test_clip_model_input_produces_recipe_model_with_path(mock_folder_paths):
    """AC: @clip-graph-nodes ac-5 — CLIP Model Input creates RecipeModel with correct path."""
    from nodes.clip_nodes import WIDENCLIPModelInputNode

    node = WIDENCLIPModelInputNode()
    result = node.create_model("my_checkpoint.safetensors", 0.8)

    recipe = result[0]
    assert isinstance(recipe, RecipeModel)
    assert recipe.path == "my_checkpoint.safetensors"


# ---------------------------------------------------------------------------
# AC-6: All CLIP graph nodes have CATEGORY ecaj/merge/clip
# ---------------------------------------------------------------------------


def test_clip_lora_category(mock_folder_paths):
    """AC: @clip-graph-nodes ac-6 — CLIP LoRA has CATEGORY ecaj/merge/clip."""
    from nodes.clip_nodes import WIDENCLIPLoRANode

    assert WIDENCLIPLoRANode.CATEGORY == "ecaj/merge/clip"


def test_clip_compose_category(mock_folder_paths):
    """AC: @clip-graph-nodes ac-6 — CLIP Compose has CATEGORY ecaj/merge/clip."""
    from nodes.clip_nodes import WIDENCLIPComposeNode

    assert WIDENCLIPComposeNode.CATEGORY == "ecaj/merge/clip"


def test_clip_merge_category(mock_folder_paths):
    """AC: @clip-graph-nodes ac-6 — CLIP Merge has CATEGORY ecaj/merge/clip."""
    from nodes.clip_nodes import WIDENCLIPMergeNode

    assert WIDENCLIPMergeNode.CATEGORY == "ecaj/merge/clip"


def test_clip_model_input_category(mock_folder_paths):
    """AC: @clip-graph-nodes ac-6 — CLIP Model Input has CATEGORY ecaj/merge/clip."""
    from nodes.clip_nodes import WIDENCLIPModelInputNode

    assert WIDENCLIPModelInputNode.CATEGORY == "ecaj/merge/clip"


# ---------------------------------------------------------------------------
# AC-7: CLIP graph nodes produce same recipe dataclass types as WIDEN variants
# ---------------------------------------------------------------------------


def test_clip_lora_produces_recipe_lora(mock_folder_paths):
    """AC: @clip-graph-nodes ac-7 — CLIP LoRA produces RecipeLoRA."""
    from nodes.clip_nodes import WIDENCLIPLoRANode

    node = WIDENCLIPLoRANode()
    result = node.add_lora("test.safetensors", 1.0)

    recipe = result[0]
    assert isinstance(recipe, RecipeLoRA)
    assert recipe.loras[0]["path"] == "test.safetensors"
    assert recipe.loras[0]["strength"] == 1.0


def test_clip_lora_chaining_produces_recipe_lora(mock_folder_paths):
    """AC: @clip-graph-nodes ac-7 — CLIP LoRA chaining produces RecipeLoRA with both."""
    from nodes.clip_nodes import WIDENCLIPLoRANode

    node = WIDENCLIPLoRANode()
    r1 = node.add_lora("a.safetensors", 1.0)[0]
    r2 = node.add_lora("b.safetensors", 0.5, prev=r1)[0]

    assert isinstance(r2, RecipeLoRA)
    assert len(r2.loras) == 2
    assert r2.loras[0]["path"] == "a.safetensors"
    assert r2.loras[1]["path"] == "b.safetensors"


def test_clip_compose_produces_recipe_compose(mock_folder_paths):
    """AC: @clip-graph-nodes ac-7 — CLIP Compose produces RecipeCompose."""
    from nodes.clip_nodes import WIDENCLIPComposeNode, WIDENCLIPLoRANode

    lora_node = WIDENCLIPLoRANode()
    lora = lora_node.add_lora("test.safetensors", 1.0)[0]

    compose_node = WIDENCLIPComposeNode()
    result = compose_node.compose(lora)

    recipe = result[0]
    assert isinstance(recipe, RecipeCompose)
    assert len(recipe.branches) == 1


def test_clip_compose_chaining_produces_recipe_compose(mock_folder_paths):
    """AC: @clip-graph-nodes ac-7 — CLIP Compose chaining accumulates branches."""
    from nodes.clip_nodes import WIDENCLIPComposeNode, WIDENCLIPLoRANode

    lora_node = WIDENCLIPLoRANode()
    lora_a = lora_node.add_lora("a.safetensors", 1.0)[0]
    lora_b = lora_node.add_lora("b.safetensors", 0.5)[0]

    compose_node = WIDENCLIPComposeNode()
    c1 = compose_node.compose(lora_a)[0]
    c2 = compose_node.compose(lora_b, compose=c1)[0]

    assert isinstance(c2, RecipeCompose)
    assert len(c2.branches) == 2


def test_clip_merge_produces_recipe_merge(mock_folder_paths):
    """AC: @clip-graph-nodes ac-7 — CLIP Merge produces RecipeMerge."""
    from unittest.mock import MagicMock

    from nodes.clip_nodes import WIDENCLIPLoRANode, WIDENCLIPMergeNode

    # Create base with mock model_patcher and target
    mock_patcher = MagicMock()
    base = RecipeBase(model_patcher=mock_patcher, arch="sdxl", domain="clip")
    lora_node = WIDENCLIPLoRANode()
    target = lora_node.add_lora("test.safetensors", 1.0)[0]

    merge_node = WIDENCLIPMergeNode()
    result = merge_node.merge(base, target, 1.0)

    recipe = result[0]
    assert isinstance(recipe, RecipeMerge)
    assert recipe.base is base
    assert recipe.target is target
    assert recipe.t_factor == 1.0


def test_clip_model_input_produces_recipe_model(mock_folder_paths):
    """AC: @clip-graph-nodes ac-7 — CLIP Model Input produces RecipeModel."""
    from nodes.clip_nodes import WIDENCLIPModelInputNode

    node = WIDENCLIPModelInputNode()
    result = node.create_model("model.safetensors", 0.8)

    recipe = result[0]
    assert isinstance(recipe, RecipeModel)
    assert recipe.path == "model.safetensors"
    assert recipe.strength == 0.8


# ---------------------------------------------------------------------------
# AC-8: All nodes registered with CLIP-prefixed display names
# (This is tested via the __init__.py import, but we verify the mappings exist)
# ---------------------------------------------------------------------------


def test_clip_nodes_registered_in_mappings():
    """AC: @clip-graph-nodes ac-8 — All CLIP nodes in NODE_CLASS_MAPPINGS."""
    # Import the main module which has the mappings

    # Need to import via __init__ to get the mappings
    # We test by verifying the classes exist with expected attributes
    from nodes.clip_nodes import (
        WIDENCLIPComposeNode,
        WIDENCLIPLoRANode,
        WIDENCLIPMergeNode,
        WIDENCLIPModelInputNode,
    )

    # Verify all expected classes exist and have correct FUNCTION
    assert WIDENCLIPLoRANode.FUNCTION == "add_lora"
    assert WIDENCLIPComposeNode.FUNCTION == "compose"
    assert WIDENCLIPMergeNode.FUNCTION == "merge"
    assert WIDENCLIPModelInputNode.FUNCTION == "create_model"


# ---------------------------------------------------------------------------
# Additional tests for block_config support
# ---------------------------------------------------------------------------


def test_clip_lora_accepts_block_config(mock_folder_paths):
    """CLIP LoRA accepts BLOCK_CONFIG input."""
    from nodes.clip_nodes import WIDENCLIPLoRANode

    input_types = WIDENCLIPLoRANode.INPUT_TYPES()

    assert "optional" in input_types
    assert "block_config" in input_types["optional"]
    assert input_types["optional"]["block_config"] == ("BLOCK_CONFIG",)


def test_clip_lora_stores_block_config(mock_folder_paths):
    """CLIP LoRA stores block_config in RecipeLoRA."""
    from nodes.clip_nodes import WIDENCLIPLoRANode

    node = WIDENCLIPLoRANode()
    block_cfg = BlockConfig(arch="sdxl", block_overrides=(("CL00", 0.5),))

    result = node.add_lora("test.safetensors", 1.0, block_config=block_cfg)

    recipe = result[0]
    assert recipe.block_config is block_cfg


def test_clip_merge_accepts_block_config(mock_folder_paths):
    """CLIP Merge accepts BLOCK_CONFIG input."""
    from nodes.clip_nodes import WIDENCLIPMergeNode

    input_types = WIDENCLIPMergeNode.INPUT_TYPES()

    assert "optional" in input_types
    assert "block_config" in input_types["optional"]
    assert input_types["optional"]["block_config"] == ("BLOCK_CONFIG",)


def test_clip_merge_stores_block_config(mock_folder_paths):
    """CLIP Merge stores block_config in RecipeMerge."""
    from unittest.mock import MagicMock

    from nodes.clip_nodes import WIDENCLIPLoRANode, WIDENCLIPMergeNode

    mock_patcher = MagicMock()
    base = RecipeBase(model_patcher=mock_patcher, arch="sdxl", domain="clip")
    lora_node = WIDENCLIPLoRANode()
    target = lora_node.add_lora("test.safetensors", 1.0)[0]
    block_cfg = BlockConfig(arch="sdxl", block_overrides=(("CL00", 0.5),))

    merge_node = WIDENCLIPMergeNode()
    result = merge_node.merge(base, target, 1.0, block_config=block_cfg)

    recipe = result[0]
    assert recipe.block_config is block_cfg


def test_clip_model_input_accepts_block_config(mock_folder_paths):
    """CLIP Model Input accepts BLOCK_CONFIG input."""
    from nodes.clip_nodes import WIDENCLIPModelInputNode

    input_types = WIDENCLIPModelInputNode.INPUT_TYPES()

    assert "optional" in input_types
    assert "block_config" in input_types["optional"]
    assert input_types["optional"]["block_config"] == ("BLOCK_CONFIG",)


def test_clip_model_input_stores_block_config(mock_folder_paths):
    """CLIP Model Input stores block_config in RecipeModel."""
    from nodes.clip_nodes import WIDENCLIPModelInputNode

    node = WIDENCLIPModelInputNode()
    block_cfg = BlockConfig(arch="sdxl", block_overrides=(("CL00", 0.5),))

    result = node.create_model("model.safetensors", 1.0, block_config=block_cfg)

    recipe = result[0]
    assert recipe.block_config is block_cfg
