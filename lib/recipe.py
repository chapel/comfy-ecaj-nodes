"""Recipe tree dataclasses — the WIDEN custom ComfyUI type.

All recipe dataclasses are frozen (immutable) to prevent aliasing bugs
with ComfyUI's caching and graph fan-out. Fields use tuples, not lists.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RecipeBase:
    """Entry node output — wraps the ModelPatcher reference."""

    model_patcher: object  # ComfyUI ModelPatcher (holds state dict ref)
    arch: str  # auto-detected: "sdxl", "zimage", "flux", "qwen"


@dataclass(frozen=True)
class RecipeLoRA:
    """LoRA node output — one or more LoRAs to apply as a group (a 'set')."""

    loras: tuple  # ({"path": str, "strength": float}, ...)


@dataclass(frozen=True)
class RecipeCompose:
    """Compose node output — accumulated branch list."""

    branches: tuple  # (WIDEN, WIDEN, ...) — each is a recipe node


@dataclass(frozen=True)
class RecipeMerge:
    """Merge node output — a merge step in the recipe."""

    base: object  # WIDEN (RecipeBase or RecipeMerge)
    target: object  # WIDEN (RecipeLoRA, RecipeCompose, or RecipeMerge)
    backbone: object  # WIDEN or None — explicit backbone override
    t_factor: float
