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

from .base import LoRALoader
from .flux import FluxLoader
from .qwen import QwenLoader
from .sdxl import SDXLLoader
from .zimage import ZImageLoader

__all__ = [
    "LoRALoader",
    "FluxLoader",
    "QwenLoader",
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
    "qwen": QwenLoader,
    "flux": FluxLoader,
}


def get_loader(arch: str, domain: str = "diffusion") -> LoRALoader:
    """Get a LoRA loader instance for the given architecture and domain.

    # AC: @lora-loaders ac-1
    # AC: @recipe-domain-field ac-5, ac-6
    Selects the appropriate architecture-specific loader, dispatching on
    both architecture and domain. For domain="clip", returns CLIP-specific
    loaders when available.

    Args:
        arch: Architecture tag (e.g. "sdxl", "zimage")
        domain: Domain type ("diffusion" or "clip"). Defaults to "diffusion"
            for backward compatibility.

    Returns:
        LoRALoader instance for the architecture and domain

    Raises:
        ValueError: If architecture/domain combination is not supported
    """
    # AC: @recipe-domain-field ac-5
    # CLIP loaders will be keyed as "{arch}_clip" (e.g., "sdxl_clip")
    if domain == "clip":
        clip_key = f"{arch}_clip"
        loader_cls = LOADER_REGISTRY.get(clip_key)
        if loader_cls is None:
            raise ValueError(
                f"No CLIP LoRA loader for architecture '{arch}'. "
                f"CLIP loaders are registered as '{clip_key}'."
            )
        return loader_cls()

    # AC: @recipe-domain-field ac-6 — domain="diffusion" (or unset) uses existing loaders
    loader_cls = LOADER_REGISTRY.get(arch)
    if loader_cls is None:
        supported = ", ".join(sorted(LOADER_REGISTRY.keys()))
        raise ValueError(f"Unsupported architecture '{arch}'. Supported: {supported}")
    return loader_cls()
