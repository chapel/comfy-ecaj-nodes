"""Smoke tests for ComfyUI node packaging — CATEGORY, INPUT_TYPES, etc."""

from pathlib import Path

import tomllib

from nodes.compose import WIDENComposeNode  # noqa: I001 — stdlib/local split
from nodes.entry import WIDENEntryNode
from nodes.exit import WIDENExitNode
from nodes.lora import WIDENLoRANode
from nodes.merge import WIDENMergeNode

ALL_NODE_CLASSES = [
    WIDENEntryNode,
    WIDENLoRANode,
    WIDENComposeNode,
    WIDENMergeNode,
    WIDENExitNode,
]


class TestNodeAttributes:
    """Every node must expose the ComfyUI required class attributes.
    # AC: @comfyui-packaging ac-1, ac-2, ac-4
    """

    def test_all_nodes_have_required_attributes(self):
        for cls in ALL_NODE_CLASSES:
            assert hasattr(cls, "CATEGORY"), f"{cls.__name__} missing CATEGORY"
            assert hasattr(cls, "INPUT_TYPES"), f"{cls.__name__} missing INPUT_TYPES"
            assert hasattr(cls, "RETURN_TYPES"), f"{cls.__name__} missing RETURN_TYPES"
            assert hasattr(cls, "FUNCTION"), f"{cls.__name__} missing FUNCTION"

    def test_all_categories_lowercase(self):  # AC: @comfyui-packaging ac-4
        for cls in ALL_NODE_CLASSES:
            assert cls.CATEGORY == "ecaj/merge", (
                f"{cls.__name__}.CATEGORY = {cls.CATEGORY!r}, expected 'ecaj/merge'"
            )

    def test_input_types_has_required(self):
        for cls in ALL_NODE_CLASSES:
            result = cls.INPUT_TYPES()
            assert isinstance(result, dict), f"{cls.__name__}.INPUT_TYPES() not a dict"
            assert "required" in result, f"{cls.__name__}.INPUT_TYPES() missing 'required' key"

    def test_function_attr_is_string(self):
        for cls in ALL_NODE_CLASSES:
            assert isinstance(cls.FUNCTION, str), f"{cls.__name__}.FUNCTION not a string"

    def test_return_types_is_tuple(self):
        for cls in ALL_NODE_CLASSES:
            assert isinstance(cls.RETURN_TYPES, tuple), f"{cls.__name__}.RETURN_TYPES not a tuple"


PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


class TestComfyRegistryMetadata:
    """Registry metadata in pyproject.toml [tool.comfy]."""

    def _load_comfy_metadata(self):
        with open(PYPROJECT, "rb") as f:
            return tomllib.load(f)["tool"]["comfy"]

    # AC: @comfyui-packaging ac-3
    def test_publisher_id(self):
        meta = self._load_comfy_metadata()
        assert meta["PublisherId"] == "ecaj"

    # AC: @comfyui-packaging ac-3
    def test_display_name(self):
        meta = self._load_comfy_metadata()
        assert meta["DisplayName"] == "ecaj nodes"

    # AC: @comfyui-packaging ac-3
    def test_icon_defined(self):
        meta = self._load_comfy_metadata()
        assert "Icon" in meta


class TestWIDENTypeConnections:
    """WIDEN wire connections are valid between nodes in the graph editor.
    # AC: @recipe-system ac-5
    """

    # Nodes that output WIDEN type
    WIDEN_OUTPUTS = [
        WIDENEntryNode,  # MODEL -> WIDEN
        WIDENLoRANode,   # file + strength -> WIDEN
        WIDENComposeNode,  # branch accumulation -> WIDEN
        WIDENMergeNode,  # merge step -> WIDEN
    ]

    # Nodes that accept WIDEN input
    WIDEN_INPUTS = [
        WIDENLoRANode,   # prev: WIDEN (optional)
        WIDENComposeNode,  # branch: WIDEN, compose: WIDEN (optional)
        WIDENMergeNode,  # base: WIDEN, target: WIDEN, backbone: WIDEN (optional)
        WIDENExitNode,   # widen: WIDEN
    ]

    def test_output_nodes_return_widen(self):
        """All WIDEN-outputting nodes declare RETURN_TYPES with 'WIDEN'."""
        for cls in self.WIDEN_OUTPUTS:
            assert "WIDEN" in cls.RETURN_TYPES, (
                f"{cls.__name__} should output WIDEN type"
            )

    def test_input_nodes_accept_widen(self):
        """All WIDEN-accepting nodes declare 'WIDEN' in INPUT_TYPES."""
        for cls in self.WIDEN_INPUTS:
            input_types = cls.INPUT_TYPES()
            all_inputs = {}
            all_inputs.update(input_types.get("required", {}))
            all_inputs.update(input_types.get("optional", {}))

            widen_inputs = [k for k, v in all_inputs.items() if v[0] == "WIDEN"]
            assert len(widen_inputs) > 0, (
                f"{cls.__name__} should accept at least one WIDEN input"
            )

    def test_entry_outputs_widen_only(self):
        """Entry node is the only source, outputs WIDEN from MODEL."""
        input_types = WIDENEntryNode.INPUT_TYPES()
        required = input_types.get("required", {})
        assert "model" in required, "Entry should require MODEL input"
        assert required["model"][0] == "MODEL", "Entry input should be MODEL type"
        assert WIDENEntryNode.RETURN_TYPES == ("WIDEN",), "Entry should output WIDEN"

    def test_exit_accepts_widen_returns_model(self):
        """Exit node takes WIDEN, returns MODEL (boundary out)."""
        input_types = WIDENExitNode.INPUT_TYPES()
        required = input_types.get("required", {})
        assert "widen" in required, "Exit should require widen input"
        assert required["widen"][0] == "WIDEN", "Exit input should be WIDEN type"
        assert WIDENExitNode.RETURN_TYPES == ("MODEL",), "Exit should output MODEL"

    def test_merge_connections(self):
        """Merge node accepts WIDEN base/target/backbone, outputs WIDEN."""
        input_types = WIDENMergeNode.INPUT_TYPES()
        required = input_types.get("required", {})
        optional = input_types.get("optional", {})

        assert required["base"][0] == "WIDEN", "Merge base should be WIDEN"
        assert required["target"][0] == "WIDEN", "Merge target should be WIDEN"
        assert optional.get("backbone", (None,))[0] == "WIDEN", "Merge backbone should be WIDEN"
        assert WIDENMergeNode.RETURN_TYPES == ("WIDEN",), "Merge should output WIDEN"

    def test_compose_connections(self):
        """Compose node accepts WIDEN branch/compose, outputs WIDEN."""
        input_types = WIDENComposeNode.INPUT_TYPES()
        required = input_types.get("required", {})
        optional = input_types.get("optional", {})

        assert required["branch"][0] == "WIDEN", "Compose branch should be WIDEN"
        assert optional.get("compose", (None,))[0] == "WIDEN", "Compose compose should be WIDEN"
        assert WIDENComposeNode.RETURN_TYPES == ("WIDEN",), "Compose should output WIDEN"

    def test_lora_connections(self):
        """LoRA node optionally accepts WIDEN prev, outputs WIDEN."""
        input_types = WIDENLoRANode.INPUT_TYPES()
        optional = input_types.get("optional", {})

        assert optional.get("prev", (None,))[0] == "WIDEN", "LoRA prev should be WIDEN"
        assert WIDENLoRANode.RETURN_TYPES == ("WIDEN",), "LoRA should output WIDEN"
