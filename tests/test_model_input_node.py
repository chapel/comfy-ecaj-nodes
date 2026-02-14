"""Tests for WIDEN Model Input Node — AC coverage for @model-input-node spec."""

import pytest

from lib.recipe import BlockConfig, RecipeModel

# ---------------------------------------------------------------------------
# AC-1: INPUT_TYPES has model_name and strength
# ---------------------------------------------------------------------------


def test_input_types_has_model_name_combo(monkeypatch):
    """AC: @model-input-node ac-1 — model_name uses folder_paths.get_filename_list checkpoints."""
    import sys
    from types import ModuleType

    # Create mock folder_paths with a mock checkpoint list
    mock_folder_paths = ModuleType("folder_paths")
    mock_checkpoint_list = ["model_a.safetensors", "model_b.safetensors"]
    mock_folder_paths.get_filename_list = lambda folder: (
        mock_checkpoint_list if folder == "checkpoints" else []
    )

    # Patch before import
    monkeypatch.setitem(sys.modules, "folder_paths", mock_folder_paths)

    # Force re-import to pick up mock
    if "nodes.model_input" in sys.modules:
        del sys.modules["nodes.model_input"]

    from nodes.model_input import WIDENModelInputNode

    input_types = WIDENModelInputNode.INPUT_TYPES()

    # model_name should be a tuple containing the list from folder_paths
    model_name_spec = input_types["required"]["model_name"]
    assert isinstance(model_name_spec, tuple)
    assert model_name_spec[0] == mock_checkpoint_list


def test_input_types_has_strength_with_correct_defaults(monkeypatch):
    """AC: @model-input-node ac-1 — strength is FLOAT with default 1.0, range 0.0-2.0."""
    import sys
    from types import ModuleType

    # Create mock folder_paths
    mock_folder_paths = ModuleType("folder_paths")
    mock_folder_paths.get_filename_list = lambda folder: []

    monkeypatch.setitem(sys.modules, "folder_paths", mock_folder_paths)

    if "nodes.model_input" in sys.modules:
        del sys.modules["nodes.model_input"]

    from nodes.model_input import WIDENModelInputNode

    input_types = WIDENModelInputNode.INPUT_TYPES()

    strength_spec = input_types["required"]["strength"]
    assert strength_spec[0] == "FLOAT"
    assert strength_spec[1]["default"] == 1.0
    assert strength_spec[1]["min"] == 0.0
    assert strength_spec[1]["max"] == 2.0


# ---------------------------------------------------------------------------
# AC-2: Returns RecipeModel with filename and strength
# ---------------------------------------------------------------------------


def test_create_model_returns_recipe_model():
    """AC: @model-input-node ac-2 — returns RecipeModel with filename and strength."""
    from nodes.model_input import WIDENModelInputNode

    node = WIDENModelInputNode()
    result = node.create_model("my_model.safetensors", 0.8)

    assert isinstance(result, tuple)
    assert len(result) == 1
    recipe = result[0]
    assert isinstance(recipe, RecipeModel)
    assert recipe.path == "my_model.safetensors"
    assert recipe.strength == 0.8


def test_create_model_preserves_exact_values():
    """AC: @model-input-node ac-2 — path and strength preserved exactly."""
    from nodes.model_input import WIDENModelInputNode

    node = WIDENModelInputNode()
    result = node.create_model("path/to/checkpoint.safetensors", 1.5)

    recipe = result[0]
    assert recipe.path == "path/to/checkpoint.safetensors"
    assert recipe.strength == 1.5


# ---------------------------------------------------------------------------
# AC-3: No GPU memory allocated, no file I/O
# ---------------------------------------------------------------------------


def test_create_model_no_gpu_or_io():
    """AC: @model-input-node ac-3 — no GPU memory, no file I/O (pure recipe building)."""
    from nodes.model_input import WIDENModelInputNode

    # This test verifies the node is pure data construction.
    # The implementation stores only the filename string, not file contents.
    # No torch imports, no file open() calls — just dataclass construction.
    node = WIDENModelInputNode()

    # Can create RecipeModel even for non-existent file (deferred to Exit)
    result = node.create_model("nonexistent_model.safetensors", 1.0)

    recipe = result[0]
    # RecipeModel only stores path as string — no tensor data
    assert isinstance(recipe.path, str)
    assert not hasattr(recipe, "tensors")
    assert not hasattr(recipe, "model")
    assert not hasattr(recipe, "state_dict")


# ---------------------------------------------------------------------------
# AC-4: CATEGORY is ecaj/merge
# ---------------------------------------------------------------------------


def test_category_is_ecaj_merge():
    """AC: @model-input-node ac-4 — CATEGORY is ecaj/merge."""
    from nodes.model_input import WIDENModelInputNode

    assert WIDENModelInputNode.CATEGORY == "ecaj/merge"


# ---------------------------------------------------------------------------
# AC-5: RETURN_TYPES is WIDEN
# ---------------------------------------------------------------------------


def test_return_types_is_widen():
    """AC: @model-input-node ac-5 — RETURN_TYPES is WIDEN."""
    from nodes.model_input import WIDENModelInputNode

    assert WIDENModelInputNode.RETURN_TYPES == ("WIDEN",)


def test_return_names_is_widen():
    """AC: @model-input-node ac-5 — RETURN_NAMES is widen."""
    from nodes.model_input import WIDENModelInputNode

    assert WIDENModelInputNode.RETURN_NAMES == ("widen",)


# ---------------------------------------------------------------------------
# AC-6: Optional BLOCK_CONFIG input stored in RecipeModel.block_config
# ---------------------------------------------------------------------------


def test_input_types_has_optional_block_config(monkeypatch):
    """AC: @model-input-node ac-6 — optional block_config input exists."""
    import sys
    from types import ModuleType

    mock_folder_paths = ModuleType("folder_paths")
    mock_folder_paths.get_filename_list = lambda folder: []

    monkeypatch.setitem(sys.modules, "folder_paths", mock_folder_paths)

    if "nodes.model_input" in sys.modules:
        del sys.modules["nodes.model_input"]

    from nodes.model_input import WIDENModelInputNode

    input_types = WIDENModelInputNode.INPUT_TYPES()

    assert "optional" in input_types
    assert "block_config" in input_types["optional"]
    assert input_types["optional"]["block_config"] == ("BLOCK_CONFIG",)


def test_block_config_stored_in_recipe():
    """AC: @model-input-node ac-6 — BlockConfig stored in RecipeModel.block_config."""
    from nodes.model_input import WIDENModelInputNode

    node = WIDENModelInputNode()
    block_cfg = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))

    result = node.create_model("test.safetensors", 1.0, block_config=block_cfg)

    recipe = result[0]
    assert recipe.block_config is block_cfg


def test_block_config_none_by_default():
    """AC: @model-input-node ac-6 — block_config is None when not provided."""
    from nodes.model_input import WIDENModelInputNode

    node = WIDENModelInputNode()
    result = node.create_model("test.safetensors", 1.0)

    recipe = result[0]
    assert recipe.block_config is None


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_recipe_model_is_frozen():
    """RecipeModel should be frozen (immutable)."""
    from nodes.model_input import WIDENModelInputNode

    node = WIDENModelInputNode()
    recipe = node.create_model("test.safetensors", 1.0)[0]

    with pytest.raises(Exception):  # FrozenInstanceError
        recipe.path = "changed.safetensors"


def test_node_function_name():
    """Verify node has correct FUNCTION attribute."""
    from nodes.model_input import WIDENModelInputNode

    assert WIDENModelInputNode.FUNCTION == "create_model"


def test_zero_strength():
    """Zero strength is valid and preserved."""
    from nodes.model_input import WIDENModelInputNode

    node = WIDENModelInputNode()
    result = node.create_model("model.safetensors", 0.0)

    recipe = result[0]
    assert recipe.strength == 0.0


def test_max_strength():
    """Max strength (2.0) is valid and preserved."""
    from nodes.model_input import WIDENModelInputNode

    node = WIDENModelInputNode()
    result = node.create_model("model.safetensors", 2.0)

    recipe = result[0]
    assert recipe.strength == 2.0
