"""Smoke tests for ComfyUI node packaging — CATEGORY, INPUT_TYPES, etc."""

from nodes.compose import WIDENComposeNode
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
