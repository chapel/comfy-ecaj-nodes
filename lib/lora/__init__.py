"""Architecture-specific LoRA loaders with pluggable registry.

This module provides architecture-specific LoRA loading with key mapping.
Each architecture has its own loader that handles format conversion and
produces DeltaSpec objects for the batched GPU executor.

# AC: @lora-loaders ac-3
Pluggable design: new architectures integrate by adding a module to
lib/lora/ and registering it here. No modifications to existing loaders.

Usage:
    # Get loader by architecture tag
    loader = get_loader("sdxl")
    loader.load("path/to/lora.safetensors", strength=0.8)

    # Or use the registry directly
    loader_cls = LOADER_REGISTRY.get("sdxl")
    if loader_cls:
        loader = loader_cls()
        loader.load(...)
"""

try:
    from .base import LoRALoader
    from .sdxl import SDXLLoader
    from .zimage import ZImageLoader
except ImportError:
    from lib.lora.base import LoRALoader
    from lib.lora.sdxl import SDXLLoader
    from lib.lora.zimage import ZImageLoader

__all__ = [
    "LoRALoader",
    "SDXLLoader",
    "ZImageLoader",
    "LOADER_REGISTRY",
    "get_loader",
]


# AC: @lora-loaders ac-1
# Registry maps architecture tags to loader classes.
# Architecture tags come from RecipeBase.arch (auto-detected by Entry node).
#
# AC: @lora-loaders ac-3
# To add a new architecture:
# 1. Create lib/lora/{arch}.py implementing LoRALoader
# 2. Add an entry here: "{arch}": {Arch}Loader
LOADER_REGISTRY: dict[str, type[LoRALoader]] = {
    "sdxl": SDXLLoader,
    "zimage": ZImageLoader,
}


def get_loader(arch: str) -> LoRALoader:
    """Get a LoRA loader instance for the given architecture.

    # AC: @lora-loaders ac-1
    Selects the appropriate architecture-specific loader.

    Args:
        arch: Architecture tag (e.g. "sdxl", "zimage")

    Returns:
        LoRALoader instance for the architecture

    Raises:
        ValueError: If architecture is not supported
    """
    loader_cls = LOADER_REGISTRY.get(arch)
    if loader_cls is None:
        supported = ", ".join(sorted(LOADER_REGISTRY.keys()))
        raise ValueError(f"Unsupported architecture '{arch}'. Supported: {supported}")
    return loader_cls()
