"""Recipe tree dataclasses — the WIDEN custom ComfyUI type.

All recipe dataclasses are frozen (immutable) to prevent aliasing bugs
with ComfyUI's caching and graph fan-out. Fields use tuples, not lists.
"""

from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "BlockConfig",
    "RecipeBase",
    "RecipeLoRA",
    "RecipeModel",
    "RecipeCompose",
    "RecipeMerge",
    "RecipeNode",
]


@dataclass(frozen=True)
class BlockConfig:
    """Per-block weight configuration for LoRA/merge operations.

    Stores architecture identifier and block-level overrides as tuples of pairs.
    Frozen to maintain immutability guarantees with ComfyUI's caching.
    """

    arch: str  # Must match RecipeBase.arch at Exit time
    block_overrides: tuple  # ((block_name, float), ...) e.g., (("IN00", 0.5), ...)
    layer_type_overrides: tuple = ()  # ((layer_type, float), ...) for cross-cutting control


@dataclass(frozen=True)
class RecipeBase:
    """Entry node output — wraps the ModelPatcher reference."""

    model_patcher: object  # ComfyUI ModelPatcher (holds state dict ref)
    arch: str  # auto-detected: "sdxl", "zimage", "flux", "qwen"


@dataclass(frozen=True)
class RecipeLoRA:
    """LoRA node output — one or more LoRAs to apply as a group (a 'set').

    Each entry in loras is a MappingProxyType wrapping {"path": str, "strength": float}
    to prevent external mutation of recipe contents post-construction.
    """

    loras: tuple  # (MappingProxyType({"path": str, "strength": float}), ...)
    block_config: object = None  # BlockConfig or None

    def __post_init__(self) -> None:
        """Freeze mutable dicts in loras to prevent post-construction mutation."""
        frozen = tuple(
            MappingProxyType(d) if isinstance(d, dict) else d for d in self.loras
        )
        object.__setattr__(self, "loras", frozen)


@dataclass(frozen=True)
class RecipeModel:
    """Full model recipe — a checkpoint file to merge with the base model.

    Unlike RecipeBase (which wraps a ComfyUI MODEL), RecipeModel stores only
    the file path for deferred disk-based loading at Exit time via safetensors
    streaming. This avoids loading full checkpoint tensors into memory during
    recipe tree construction.
    """

    path: str  # Checkpoint filename (resolved to full path at Exit time)
    strength: float = 1.0  # Merge strength
    block_config: object = None  # BlockConfig or None


@dataclass(frozen=True)
class RecipeCompose:
    """Compose node output — accumulated branch list."""

    branches: tuple  # (WIDEN, WIDEN, ...) — each is a recipe node

    def with_branch(self, branch: "RecipeNode") -> "RecipeCompose":
        """Return a new RecipeCompose with the branch appended.

        Implements persistent tree semantics — the original is unchanged.
        """
        return RecipeCompose(branches=self.branches + (branch,))


@dataclass(frozen=True)
class RecipeMerge:
    """Merge node output — a merge step in the recipe."""

    base: object  # WIDEN (RecipeBase or RecipeMerge)
    target: object  # WIDEN (RecipeLoRA, RecipeCompose, or RecipeMerge)
    backbone: object  # WIDEN or None — explicit backbone override
    t_factor: float
    block_config: object = None  # BlockConfig or None


# Type alias for any recipe node
RecipeNode = RecipeBase | RecipeLoRA | RecipeModel | RecipeCompose | RecipeMerge
