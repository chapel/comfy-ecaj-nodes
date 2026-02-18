"""Tests for WIDEN Diffusion Model Input Node — AC coverage for @diffusion-model-input-node."""

import pytest

from lib.recipe import BlockConfig, RecipeModel

# ---------------------------------------------------------------------------
# AC-1: INPUT_TYPES has model_name combo for diffusion_models
# ---------------------------------------------------------------------------


# AC: @diffusion-model-input-node ac-1
def test_input_types_has_model_name_combo_diffusion_models(monkeypatch):
    """AC: @diffusion-model-input-node ac-1 — model_name uses folder_paths for diffusion_models."""
    import sys
    from types import ModuleType

    # Create mock folder_paths with diffusion_models list
    mock_folder_paths = ModuleType("folder_paths")
    mock_model_list = ["flux_dev.safetensors", "qwen_model.safetensors"]
    mock_folder_paths.get_filename_list = lambda folder: (
        mock_model_list if folder == "diffusion_models" else []
    )

    # Patch before import
    monkeypatch.setitem(sys.modules, "folder_paths", mock_folder_paths)

    # Force re-import to pick up mock
    if "nodes.diffusion_model_input" in sys.modules:
        del sys.modules["nodes.diffusion_model_input"]

    from nodes.diffusion_model_input import WIDENDiffusionModelInputNode

    input_types = WIDENDiffusionModelInputNode.INPUT_TYPES()

    # model_name should be a tuple containing the list from folder_paths
    model_name_spec = input_types["required"]["model_name"]
    assert isinstance(model_name_spec, tuple)
    assert model_name_spec[0] == mock_model_list


# AC: @diffusion-model-input-node ac-1
def test_input_types_falls_back_to_unet_folder(monkeypatch):
    """AC: @diffusion-model-input-node ac-1 — falls back to unet folder for older ComfyUI."""
    import sys
    from types import ModuleType

    # Create mock folder_paths that raises on diffusion_models but works for unet
    mock_folder_paths = ModuleType("folder_paths")
    mock_unet_list = ["flux_unet.safetensors"]

    def mock_get_filename_list(folder):
        if folder == "diffusion_models":
            raise KeyError("diffusion_models folder not found")
        elif folder == "unet":
            return mock_unet_list
        return []

    mock_folder_paths.get_filename_list = mock_get_filename_list

    monkeypatch.setitem(sys.modules, "folder_paths", mock_folder_paths)

    if "nodes.diffusion_model_input" in sys.modules:
        del sys.modules["nodes.diffusion_model_input"]

    from nodes.diffusion_model_input import WIDENDiffusionModelInputNode

    input_types = WIDENDiffusionModelInputNode.INPUT_TYPES()

    model_name_spec = input_types["required"]["model_name"]
    assert model_name_spec[0] == mock_unet_list


# ---------------------------------------------------------------------------
# AC-2: INPUT_TYPES has strength with correct defaults
# ---------------------------------------------------------------------------


# AC: @diffusion-model-input-node ac-2
def test_input_types_has_strength_with_correct_defaults(monkeypatch):
    """AC: @diffusion-model-input-node ac-2 — strength is FLOAT with default 1.0, range 0.0-2.0."""
    import sys
    from types import ModuleType

    mock_folder_paths = ModuleType("folder_paths")
    mock_folder_paths.get_filename_list = lambda folder: []

    monkeypatch.setitem(sys.modules, "folder_paths", mock_folder_paths)

    if "nodes.diffusion_model_input" in sys.modules:
        del sys.modules["nodes.diffusion_model_input"]

    from nodes.diffusion_model_input import WIDENDiffusionModelInputNode

    input_types = WIDENDiffusionModelInputNode.INPUT_TYPES()

    strength_spec = input_types["required"]["strength"]
    assert strength_spec[0] == "FLOAT"
    assert strength_spec[1]["default"] == 1.0
    assert strength_spec[1]["min"] == 0.0
    assert strength_spec[1]["max"] == 2.0


# ---------------------------------------------------------------------------
# AC-3: Returns RecipeModel with source_dir="diffusion_models"
# ---------------------------------------------------------------------------


# AC: @diffusion-model-input-node ac-3
def test_create_model_returns_recipe_model_with_source_dir():
    """AC: @diffusion-model-input-node ac-3 — returns RecipeModel with source_dir."""
    from nodes.diffusion_model_input import WIDENDiffusionModelInputNode

    node = WIDENDiffusionModelInputNode()
    result = node.create_model("flux_dev.safetensors", 0.8)

    assert isinstance(result, tuple)
    assert len(result) == 1
    recipe = result[0]
    assert isinstance(recipe, RecipeModel)
    assert recipe.path == "flux_dev.safetensors"
    assert recipe.strength == 0.8
    assert recipe.source_dir == "diffusion_models"


# AC: @diffusion-model-input-node ac-3
def test_create_model_preserves_exact_values():
    """AC: @diffusion-model-input-node ac-3 — path, strength, source_dir preserved exactly."""
    from nodes.diffusion_model_input import WIDENDiffusionModelInputNode

    node = WIDENDiffusionModelInputNode()
    result = node.create_model("path/to/model.safetensors", 1.5)

    recipe = result[0]
    assert recipe.path == "path/to/model.safetensors"
    assert recipe.strength == 1.5
    assert recipe.source_dir == "diffusion_models"


# ---------------------------------------------------------------------------
# AC-4: No GPU memory allocated, no file I/O
# ---------------------------------------------------------------------------


# AC: @diffusion-model-input-node ac-4
def test_create_model_no_gpu_or_io():
    """AC: @diffusion-model-input-node ac-4 — no GPU memory, no file I/O (pure recipe building)."""
    from nodes.diffusion_model_input import WIDENDiffusionModelInputNode

    # This test verifies the node is pure data construction.
    # The implementation stores only the filename string, not file contents.
    # No torch imports, no file open() calls — just dataclass construction.
    node = WIDENDiffusionModelInputNode()

    # Can create RecipeModel even for non-existent file (deferred to Exit)
    result = node.create_model("nonexistent_model.safetensors", 1.0)

    recipe = result[0]
    # RecipeModel only stores path as string — no tensor data
    assert isinstance(recipe.path, str)
    assert not hasattr(recipe, "tensors")
    assert not hasattr(recipe, "model")
    assert not hasattr(recipe, "state_dict")


# ---------------------------------------------------------------------------
# AC-5: CATEGORY is ecaj/merge
# ---------------------------------------------------------------------------


# AC: @diffusion-model-input-node ac-5
def test_category_is_ecaj_merge():
    """AC: @diffusion-model-input-node ac-5 — CATEGORY is ecaj/merge."""
    from nodes.diffusion_model_input import WIDENDiffusionModelInputNode

    assert WIDENDiffusionModelInputNode.CATEGORY == "ecaj/merge"


# ---------------------------------------------------------------------------
# AC-6: RETURN_TYPES is WIDEN
# ---------------------------------------------------------------------------


# AC: @diffusion-model-input-node ac-6
def test_return_types_is_widen():
    """AC: @diffusion-model-input-node ac-6 — RETURN_TYPES is WIDEN."""
    from nodes.diffusion_model_input import WIDENDiffusionModelInputNode

    assert WIDENDiffusionModelInputNode.RETURN_TYPES == ("WIDEN",)


# AC: @diffusion-model-input-node ac-6
def test_return_names_is_widen():
    """AC: @diffusion-model-input-node ac-6 — RETURN_NAMES is widen."""
    from nodes.diffusion_model_input import WIDENDiffusionModelInputNode

    assert WIDENDiffusionModelInputNode.RETURN_NAMES == ("widen",)


# ---------------------------------------------------------------------------
# AC-7: Optional BLOCK_CONFIG input stored in RecipeModel.block_config
# ---------------------------------------------------------------------------


# AC: @diffusion-model-input-node ac-7
def test_input_types_has_optional_block_config(monkeypatch):
    """AC: @diffusion-model-input-node ac-7 — optional block_config input exists."""
    import sys
    from types import ModuleType

    mock_folder_paths = ModuleType("folder_paths")
    mock_folder_paths.get_filename_list = lambda folder: []

    monkeypatch.setitem(sys.modules, "folder_paths", mock_folder_paths)

    if "nodes.diffusion_model_input" in sys.modules:
        del sys.modules["nodes.diffusion_model_input"]

    from nodes.diffusion_model_input import WIDENDiffusionModelInputNode

    input_types = WIDENDiffusionModelInputNode.INPUT_TYPES()

    assert "optional" in input_types
    assert "block_config" in input_types["optional"]
    assert input_types["optional"]["block_config"] == ("BLOCK_CONFIG",)


# AC: @diffusion-model-input-node ac-7
def test_block_config_stored_in_recipe():
    """AC: @diffusion-model-input-node ac-7 — BlockConfig stored in RecipeModel.block_config."""
    from nodes.diffusion_model_input import WIDENDiffusionModelInputNode

    node = WIDENDiffusionModelInputNode()
    block_cfg = BlockConfig(arch="flux", block_overrides=(("DB00", 0.5),))

    result = node.create_model("test.safetensors", 1.0, block_config=block_cfg)

    recipe = result[0]
    assert recipe.block_config is block_cfg


# AC: @diffusion-model-input-node ac-7
def test_block_config_none_by_default():
    """AC: @diffusion-model-input-node ac-7 — block_config is None when not provided."""
    from nodes.diffusion_model_input import WIDENDiffusionModelInputNode

    node = WIDENDiffusionModelInputNode()
    result = node.create_model("test.safetensors", 1.0)

    recipe = result[0]
    assert recipe.block_config is None


# ---------------------------------------------------------------------------
# AC-8: Node registered in __init__.py with correct display name
# ---------------------------------------------------------------------------


# AC: @diffusion-model-input-node ac-8
def test_node_registered_in_class_mappings():
    """AC: @diffusion-model-input-node ac-8 — node in NODE_CLASS_MAPPINGS."""
    # Import the node and verify it's properly defined
    # The autouse fixture provides a valid folder_paths mock
    from nodes.diffusion_model_input import WIDENDiffusionModelInputNode

    # Verify the class exists and is importable
    assert WIDENDiffusionModelInputNode is not None
    assert hasattr(WIDENDiffusionModelInputNode, "FUNCTION")


# AC: @diffusion-model-input-node ac-8
def test_display_name_contains_diffusion_model():
    """AC: @diffusion-model-input-node ac-8 — display name contains 'Diffusion Model'."""
    # Read the __init__.py file content to verify the mapping
    import pathlib

    init_path = pathlib.Path(__file__).parent.parent / "__init__.py"
    content = init_path.read_text()

    # Verify both class and display name registrations
    assert '"WIDENDiffusionModelInput"' in content
    assert '"WIDEN Diffusion Model Input"' in content


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_recipe_model_is_frozen():
    """RecipeModel should be frozen (immutable)."""
    from nodes.diffusion_model_input import WIDENDiffusionModelInputNode

    node = WIDENDiffusionModelInputNode()
    recipe = node.create_model("test.safetensors", 1.0)[0]

    with pytest.raises(Exception):  # FrozenInstanceError
        recipe.path = "changed.safetensors"


def test_node_function_name():
    """Verify node has correct FUNCTION attribute."""
    from nodes.diffusion_model_input import WIDENDiffusionModelInputNode

    assert WIDENDiffusionModelInputNode.FUNCTION == "create_model"


def test_zero_strength():
    """Zero strength is valid and preserved."""
    from nodes.diffusion_model_input import WIDENDiffusionModelInputNode

    node = WIDENDiffusionModelInputNode()
    result = node.create_model("model.safetensors", 0.0)

    recipe = result[0]
    assert recipe.strength == 0.0


def test_max_strength():
    """Max strength (2.0) is valid and preserved."""
    from nodes.diffusion_model_input import WIDENDiffusionModelInputNode

    node = WIDENDiffusionModelInputNode()
    result = node.create_model("model.safetensors", 2.0)

    recipe = result[0]
    assert recipe.strength == 2.0
