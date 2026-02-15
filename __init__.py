"""ComfyUI ECAJ Nodes — Advanced model merging with WIDEN."""

# Guard on __package__: relative imports require a package context.
# In ComfyUI, this is always set (errors propagate immediately).
# In pytest, __init__.py is loaded standalone — skip registration.
if __package__:
    from .nodes.block_config_flux import WIDENBlockConfigFluxNode
    from .nodes.block_config_qwen import WIDENBlockConfigQwenNode
    from .nodes.block_config_sdxl import WIDENBlockConfigSDXLNode
    from .nodes.block_config_zimage import WIDENBlockConfigZImageNode
    from .nodes.clip_entry import WIDENCLIPEntryNode
    from .nodes.clip_nodes import (
        WIDENCLIPComposeNode,
        WIDENCLIPLoRANode,
        WIDENCLIPMergeNode,
        WIDENCLIPModelInputNode,
    )
    from .nodes.compose import WIDENComposeNode
    from .nodes.diffusion_model_input import WIDENDiffusionModelInputNode
    from .nodes.entry import WIDENEntryNode
    from .nodes.exit import WIDENExitNode
    from .nodes.lora import WIDENLoRANode
    from .nodes.merge import WIDENMergeNode
    from .nodes.model_input import WIDENModelInputNode

    NODE_CLASS_MAPPINGS = {
        "WIDENEntry": WIDENEntryNode,
        "WIDENCLIPEntry": WIDENCLIPEntryNode,
        "WIDENLoRA": WIDENLoRANode,
        "WIDENCompose": WIDENComposeNode,
        "WIDENMerge": WIDENMergeNode,
        "WIDENExit": WIDENExitNode,
        "WIDENBlockConfigSDXL": WIDENBlockConfigSDXLNode,
        "WIDENBlockConfigZImage": WIDENBlockConfigZImageNode,
        "WIDENBlockConfigQwen": WIDENBlockConfigQwenNode,
        "WIDENBlockConfigFlux": WIDENBlockConfigFluxNode,
        "WIDENModelInput": WIDENModelInputNode,
        "WIDENDiffusionModelInput": WIDENDiffusionModelInputNode,
        "WIDENCLIPLoRA": WIDENCLIPLoRANode,
        "WIDENCLIPCompose": WIDENCLIPComposeNode,
        "WIDENCLIPMerge": WIDENCLIPMergeNode,
        "WIDENCLIPModelInput": WIDENCLIPModelInputNode,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "WIDENEntry": "WIDEN Entry",
        "WIDENCLIPEntry": "WIDEN CLIP Entry",
        "WIDENLoRA": "WIDEN LoRA",
        "WIDENCompose": "WIDEN Compose",
        "WIDENMerge": "WIDEN Merge",
        "WIDENExit": "WIDEN Exit",
        "WIDENBlockConfigSDXL": "WIDEN Block Config (SDXL)",
        "WIDENBlockConfigZImage": "WIDEN Block Config (Z-Image)",
        "WIDENBlockConfigQwen": "WIDEN Block Config (Qwen)",
        "WIDENBlockConfigFlux": "WIDEN Block Config (Flux)",
        "WIDENModelInput": "WIDEN Checkpoint Input",
        "WIDENDiffusionModelInput": "WIDEN Diffusion Model Input",
        "WIDENCLIPLoRA": "WIDEN CLIP LoRA",
        "WIDENCLIPCompose": "WIDEN CLIP Compose",
        "WIDENCLIPMerge": "WIDEN CLIP Merge",
        "WIDENCLIPModelInput": "WIDEN CLIP Model Input",
    }

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
