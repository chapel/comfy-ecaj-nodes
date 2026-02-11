"""ComfyUI ECAJ Nodes — Advanced model merging with WIDEN."""

try:
    from .nodes.compose import WIDENComposeNode
    from .nodes.entry import WIDENEntryNode
    from .nodes.exit import WIDENExitNode
    from .nodes.lora import WIDENLoRANode
    from .nodes.merge import WIDENMergeNode
except ImportError:
    # Running outside ComfyUI (e.g., pytest) — node classes are imported
    # directly via absolute imports in tests, so this is safe to skip.
    pass
else:
    NODE_CLASS_MAPPINGS = {
        "WIDENEntry": WIDENEntryNode,
        "WIDENLoRA": WIDENLoRANode,
        "WIDENCompose": WIDENComposeNode,
        "WIDENMerge": WIDENMergeNode,
        "WIDENExit": WIDENExitNode,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "WIDENEntry": "WIDEN Entry",
        "WIDENLoRA": "WIDEN LoRA",
        "WIDENCompose": "WIDEN Compose",
        "WIDENMerge": "WIDEN Merge",
        "WIDENExit": "WIDEN Exit",
    }

    WEB_DIRECTORY = None
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
