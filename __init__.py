"""ComfyUI ECAJ Nodes — Advanced model merging with WIDEN."""

from .nodes.entry import WIDENEntryNode
from .nodes.lora import WIDENLoRANode
from .nodes.compose import WIDENComposeNode
from .nodes.merge import WIDENMergeNode
from .nodes.exit import WIDENExitNode

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
