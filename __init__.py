"""ComfyUI ECAJ Nodes — Advanced model merging with WIDEN."""

# Guard on __package__: relative imports require a package context.
# In ComfyUI, this is always set (errors propagate immediately).
# In pytest, __init__.py is loaded standalone — skip registration.
if __package__:
    from .nodes.block_config_flux import WIDENBlockConfigFluxNode
    from .nodes.block_config_qwen import WIDENBlockConfigQwenNode
    from .nodes.block_config_sdxl import WIDENBlockConfigSDXLNode
    from .nodes.block_config_zimage import WIDENBlockConfigZImageNode
    from .nodes.compose import WIDENComposeNode
    from .nodes.entry import WIDENEntryNode
    from .nodes.exit import WIDENExitNode
    from .nodes.lora import WIDENLoRANode
    from .nodes.merge import WIDENMergeNode
    from .nodes.model_input import WIDENModelInputNode

    NODE_CLASS_MAPPINGS = {
        "WIDENEntry": WIDENEntryNode,
        "WIDENLoRA": WIDENLoRANode,
        "WIDENCompose": WIDENComposeNode,
        "WIDENMerge": WIDENMergeNode,
        "WIDENExit": WIDENExitNode,
        "WIDENBlockConfigSDXL": WIDENBlockConfigSDXLNode,
        "WIDENBlockConfigZImage": WIDENBlockConfigZImageNode,
        "WIDENBlockConfigQwen": WIDENBlockConfigQwenNode,
        "WIDENBlockConfigFlux": WIDENBlockConfigFluxNode,
        "WIDENModelInput": WIDENModelInputNode,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "WIDENEntry": "WIDEN Entry",
        "WIDENLoRA": "WIDEN LoRA",
        "WIDENCompose": "WIDEN Compose",
        "WIDENMerge": "WIDEN Merge",
        "WIDENExit": "WIDEN Exit",
        "WIDENBlockConfigSDXL": "WIDEN Block Config (SDXL)",
        "WIDENBlockConfigZImage": "WIDEN Block Config (Z-Image)",
        "WIDENBlockConfigQwen": "WIDEN Block Config (Qwen)",
        "WIDENBlockConfigFlux": "WIDEN Block Config (Flux)",
        "WIDENModelInput": "WIDEN Checkpoint Input",
    }

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
