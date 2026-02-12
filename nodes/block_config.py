"""WIDEN Block Config factory — generates architecture-specific block weight nodes."""

from ..lib.recipe import BlockConfig

SLIDER_CONFIG = {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}


def make_block_config_node(arch, block_groups, docstring):
    """Generate a ComfyUI BlockConfig node class from block group definitions.

    Args:
        arch: Architecture identifier (e.g. "sdxl", "zimage").
        block_groups: Tuple of (param_name, override_key) pairs.
            param_name is the ComfyUI input name (e.g. "IN00_02").
            override_key is the BlockConfig key (e.g. "IN00-02").
        docstring: Class docstring describing the architecture's block structure.
    """

    class BlockConfigNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    param: ("FLOAT", SLIDER_CONFIG) for param, _ in block_groups
                },
            }

        RETURN_TYPES = ("BLOCK_CONFIG",)
        RETURN_NAMES = ("block_config",)
        FUNCTION = "create_config"
        CATEGORY = "ecaj/merge"

        def create_config(self, **kwargs) -> tuple[BlockConfig]:
            block_overrides = tuple(
                (override_key, kwargs[param]) for param, override_key in block_groups
            )
            return (BlockConfig(arch=arch, block_overrides=block_overrides),)

    BlockConfigNode.__doc__ = docstring
    BlockConfigNode.__qualname__ = f"WIDEN BlockConfig {arch}"
    return BlockConfigNode
