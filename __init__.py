"""ComfyUI ECAJ Nodes — Advanced model merging with WIDEN."""

try:
    from .nodes.block_config_sdxl import WIDENBlockConfigSDXLNode  # noqa: F401
    from .nodes.block_config_zimage import WIDENBlockConfigZImageNode  # noqa: F401
    from .nodes.compose import WIDENComposeNode  # noqa: F401
    from .nodes.entry import WIDENEntryNode  # noqa: F401
    from .nodes.exit import WIDENExitNode  # noqa: F401
    from .nodes.lora import WIDENLoRANode  # noqa: F401
    from .nodes.merge import WIDENMergeNode  # noqa: F401
except ImportError:
    # Running outside ComfyUI (e.g., pytest) — tests import nodes directly
    # via absolute imports with --import-mode=importlib.
    pass
else:
    NODE_CLASS_MAPPINGS = {
        "WIDENEntry": WIDENEntryNode,
        "WIDENLoRA": WIDENLoRANode,
        "WIDENCompose": WIDENComposeNode,
        "WIDENMerge": WIDENMergeNode,
        "WIDENExit": WIDENExitNode,
        "WIDENBlockConfigSDXL": WIDENBlockConfigSDXLNode,
        "WIDENBlockConfigZImage": WIDENBlockConfigZImageNode,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "WIDENEntry": "WIDEN Entry",
        "WIDENLoRA": "WIDEN LoRA",
        "WIDENCompose": "WIDEN Compose",
        "WIDENMerge": "WIDEN Merge",
        "WIDENExit": "WIDEN Exit",
        "WIDENBlockConfigSDXL": "WIDEN Block Config (SDXL)",
        "WIDENBlockConfigZImage": "WIDEN Block Config (Z-Image)",
    }

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
