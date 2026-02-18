"""Tests for WIDEN LoRA Node — AC coverage for @lora-node spec."""

import pytest

from lib.recipe import RecipeLoRA  # noqa: F401  (used for type checks in tests)

# ---------------------------------------------------------------------------
# AC-1: Returns RecipeLoRA with path and strength
# ---------------------------------------------------------------------------


# AC: @lora-node ac-1
def test_add_lora_returns_recipe_lora():
    """AC: @lora-node ac-1 — returns RecipeLoRA with path and strength."""
    from nodes.lora import WIDENLoRANode

    node = WIDENLoRANode()
    result = node.add_lora("my_lora.safetensors", 0.8)

    assert isinstance(result, tuple)
    assert len(result) == 1
    recipe = result[0]
    assert isinstance(recipe, RecipeLoRA)
    assert len(recipe.loras) == 1
    assert recipe.loras[0]["path"] == "my_lora.safetensors"
    assert recipe.loras[0]["strength"] == 0.8


# AC: @lora-node ac-1
def test_add_lora_preserves_exact_values():
    """AC: @lora-node ac-1 — strength and path preserved exactly."""
    from nodes.lora import WIDENLoRANode

    node = WIDENLoRANode()
    result = node.add_lora("path/to/lora.safetensors", 0.123)

    recipe = result[0]
    assert recipe.loras[0]["path"] == "path/to/lora.safetensors"
    assert recipe.loras[0]["strength"] == 0.123


# ---------------------------------------------------------------------------
# AC-2: Chaining via prev connection
# ---------------------------------------------------------------------------


# AC: @lora-node ac-2
def test_chained_loras_contains_both():
    """AC: @lora-node ac-2 — chained LoRAs form a set with both."""
    from nodes.lora import WIDENLoRANode

    node = WIDENLoRANode()

    # First LoRA
    first_result = node.add_lora("lora_a.safetensors", 1.0)
    first_recipe = first_result[0]

    # Second LoRA chained via prev
    second_result = node.add_lora("lora_b.safetensors", 0.5, prev=first_recipe)
    chained_recipe = second_result[0]

    assert isinstance(chained_recipe, RecipeLoRA)
    assert len(chained_recipe.loras) == 2
    assert chained_recipe.loras[0] == {"path": "lora_a.safetensors", "strength": 1.0}
    assert chained_recipe.loras[1] == {"path": "lora_b.safetensors", "strength": 0.5}


# AC: @lora-node ac-2
def test_triple_chain_accumulates_all():
    """AC: @lora-node ac-2 — three chained LoRAs all appear in order."""
    from nodes.lora import WIDENLoRANode

    node = WIDENLoRANode()

    r1 = node.add_lora("a.safetensors", 1.0)[0]
    r2 = node.add_lora("b.safetensors", 0.5, prev=r1)[0]
    r3 = node.add_lora("c.safetensors", 0.3, prev=r2)[0]

    assert len(r3.loras) == 3
    assert r3.loras[0]["path"] == "a.safetensors"
    assert r3.loras[1]["path"] == "b.safetensors"
    assert r3.loras[2]["path"] == "c.safetensors"


# ---------------------------------------------------------------------------
# AC-3: Dropdown via folder_paths
# ---------------------------------------------------------------------------


# AC: @lora-node ac-3
def test_input_types_uses_folder_paths(monkeypatch):
    """AC: @lora-node ac-3 — lora_name uses folder_paths.get_filename_list."""
    import sys
    from types import ModuleType

    # Create mock folder_paths with a mock lora list
    mock_folder_paths = ModuleType("folder_paths")
    mock_lora_list = ["lora1.safetensors", "lora2.safetensors", "style_lora.safetensors"]
    mock_folder_paths.get_filename_list = (
        lambda folder: mock_lora_list if folder == "loras" else []
    )

    # Patch before import
    monkeypatch.setitem(sys.modules, "folder_paths", mock_folder_paths)

    # Force re-import to pick up mock
    if "nodes.lora" in sys.modules:
        del sys.modules["nodes.lora"]

    from nodes.lora import WIDENLoRANode

    input_types = WIDENLoRANode.INPUT_TYPES()

    # lora_name should be a tuple containing the list from folder_paths
    lora_name_spec = input_types["required"]["lora_name"]
    assert isinstance(lora_name_spec, tuple)
    assert lora_name_spec[0] == mock_lora_list


# ---------------------------------------------------------------------------
# AC-4: No prev connection → single-element tuple
# ---------------------------------------------------------------------------


# AC: @lora-node ac-4
def test_no_prev_returns_single_element_loras():
    """AC: @lora-node ac-4 — no prev gives single-element loras tuple."""
    from nodes.lora import WIDENLoRANode

    node = WIDENLoRANode()
    result = node.add_lora("solo_lora.safetensors", 0.9)

    recipe = result[0]
    assert len(recipe.loras) == 1
    assert recipe.loras == ({"path": "solo_lora.safetensors", "strength": 0.9},)


# AC: @lora-node ac-4
def test_explicit_none_prev_returns_single_element():
    """AC: @lora-node ac-4 — explicit prev=None gives single-element loras."""
    from nodes.lora import WIDENLoRANode

    node = WIDENLoRANode()
    result = node.add_lora("test.safetensors", 1.0, prev=None)

    recipe = result[0]
    assert len(recipe.loras) == 1


# ---------------------------------------------------------------------------
# AC-5: Strength 0.0 still appears in recipe
# ---------------------------------------------------------------------------


# AC: @lora-node ac-5
def test_zero_strength_still_in_recipe():
    """AC: @lora-node ac-5 — strength=0.0 LoRA still appears in recipe."""
    from nodes.lora import WIDENLoRANode

    node = WIDENLoRANode()
    result = node.add_lora("disabled_lora.safetensors", 0.0)

    recipe = result[0]
    assert len(recipe.loras) == 1
    assert recipe.loras[0]["strength"] == 0.0
    assert recipe.loras[0]["path"] == "disabled_lora.safetensors"


# AC: @lora-node ac-5
def test_zero_strength_in_chain():
    """AC: @lora-node ac-5 — zero-strength LoRA preserved when chaining."""
    from nodes.lora import WIDENLoRANode

    node = WIDENLoRANode()

    r1 = node.add_lora("active.safetensors", 1.0)[0]
    r2 = node.add_lora("disabled.safetensors", 0.0, prev=r1)[0]

    assert len(r2.loras) == 2
    assert r2.loras[1]["strength"] == 0.0


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_negative_strength_preserved():
    """Negative strength values are valid and preserved."""
    from nodes.lora import WIDENLoRANode

    node = WIDENLoRANode()
    result = node.add_lora("inverted.safetensors", -0.5)

    recipe = result[0]
    assert recipe.loras[0]["strength"] == -0.5


def test_recipe_lora_is_frozen():
    """RecipeLoRA should be frozen (immutable)."""
    from nodes.lora import WIDENLoRANode

    node = WIDENLoRANode()
    recipe = node.add_lora("test.safetensors", 1.0)[0]

    with pytest.raises(Exception):  # FrozenInstanceError
        recipe.loras = ()


def test_node_metadata():
    """Verify node has correct ComfyUI metadata."""
    from nodes.lora import WIDENLoRANode

    assert WIDENLoRANode.RETURN_TYPES == ("WIDEN",)
    assert WIDENLoRANode.RETURN_NAMES == ("widen",)
    assert WIDENLoRANode.FUNCTION == "add_lora"
    assert WIDENLoRANode.CATEGORY == "ecaj/merge"
