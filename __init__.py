"""ComfyUI ECAJ Nodes — Advanced model merging with WIDEN."""

import logging

_logger = logging.getLogger(__name__)

# Modules that are only available inside the ComfyUI runtime.
# ImportError for these is expected when running under pytest.
_COMFYUI_RUNTIME_MODULES = frozenset({"folder_paths", "comfy"})


def _is_expected_import_error(exc: ImportError) -> bool:
    """Return True if the ImportError is from a known ComfyUI-only module."""
    # Relative imports fail with name=None outside a package context (e.g., pytest)
    if "relative import" in str(exc):
        return True
    module_name = getattr(exc, "name", None) or ""
    # Check if the missing module is a known ComfyUI runtime module
    top_level = module_name.split(".")[0]
    return top_level in _COMFYUI_RUNTIME_MODULES


try:
    from .nodes.block_config_sdxl import WIDENBlockConfigSDXLNode
    from .nodes.block_config_zimage import WIDENBlockConfigZImageNode
    from .nodes.compose import WIDENComposeNode
    from .nodes.entry import WIDENEntryNode
    from .nodes.exit import WIDENExitNode
    from .nodes.lora import WIDENLoRANode
    from .nodes.merge import WIDENMergeNode
except ImportError as _exc:
    if _is_expected_import_error(_exc):
        # Running outside ComfyUI (e.g., pytest) — node classes are imported
        # directly via absolute imports in tests, so this is safe to skip.
        pass
    else:
        _logger.error(
            "Unexpected ImportError loading ECAJ nodes: %s", _exc, exc_info=True
        )
        raise
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

    WEB_DIRECTORY = None
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
