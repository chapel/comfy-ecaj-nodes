"""WIDEN Block Config factory — generates architecture-specific block weight nodes."""

from ..lib.recipe import BlockConfig

SLIDER_CONFIG = {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}


def make_block_config_node(arch, block_groups, docstring, layer_types=None):
    """Generate a ComfyUI BlockConfig node class from block group definitions.

    Args:
        arch: Architecture identifier (e.g. "sdxl", "zimage").
        block_groups: Tuple of (param_name, override_key) pairs.
            param_name is the ComfyUI input name (e.g. "IN00_02").
            override_key is the BlockConfig key (e.g. "IN00-02").
        docstring: Class docstring describing the architecture's block structure.
        layer_types: Optional tuple of (param_name, override_key) pairs for
            layer type overrides (e.g., "attention", "feed_forward", "norm").
            # AC: @layer-type-filter ac-5 — layer type sliders after block sliders
    """

    class BlockConfigNode:
        @classmethod
        def INPUT_TYPES(cls):
            inputs = {param: ("FLOAT", SLIDER_CONFIG) for param, _ in block_groups}
            # Add layer type sliders after block sliders (ac-5)
            if layer_types is not None:
                for param, _ in layer_types:
                    inputs[param] = ("FLOAT", SLIDER_CONFIG)
            return {"required": inputs}

        RETURN_TYPES = ("BLOCK_CONFIG",)
        RETURN_NAMES = ("block_config",)
        FUNCTION = "create_config"
        CATEGORY = "ecaj/merge"

        def create_config(self, **kwargs) -> tuple[BlockConfig]:
            """Create BlockConfig with architecture-specific block overrides.

            AC: @per-block-control ac-2 — block group sliders with float range 0.0 to 2.0
            """
            block_overrides = tuple(
                (override_key, kwargs[param]) for param, override_key in block_groups
            )
            # Build layer_type_overrides from layer_types if provided
            if layer_types is not None:
                layer_type_overrides = tuple(
                    (override_key, kwargs[param]) for param, override_key in layer_types
                )
            else:
                layer_type_overrides = ()
            return (
                BlockConfig(
                    arch=arch,
                    block_overrides=block_overrides,
                    layer_type_overrides=layer_type_overrides,
                ),
            )

    BlockConfigNode.__name__ = f"WIDENBlockConfig{arch.capitalize()}Node"
    BlockConfigNode.__qualname__ = f"WIDENBlockConfig{arch.capitalize()}Node"
    BlockConfigNode.__doc__ = docstring
    return BlockConfigNode
