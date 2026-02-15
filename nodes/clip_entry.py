"""WIDEN CLIP Entry Node — Boundary from ComfyUI CLIP to WIDEN recipe world."""

from ..lib.recipe import RecipeBase


class UnsupportedCLIPArchitectureError(ValueError):
    """Raised when CLIP architecture cannot be determined or is not supported."""

    pass


def detect_clip_architecture(clip: object) -> str:
    """Detect CLIP architecture from state dict key patterns.

    Accesses keys via clip.patcher.model_state_dict() — zero GPU cost,
    returns CPU tensors without loading weights to GPU.

    AC: @clip-entry-node ac-7 — keys accessed via patcher API without GPU load

    Args:
        clip: ComfyUI CLIP instance (has .patcher attribute)

    Returns:
        Architecture string: "sdxl" for SDXL CLIP

    Raises:
        UnsupportedCLIPArchitectureError: If architecture is not SDXL
    """
    # Access state dict keys without GPU load
    state_dict = clip.patcher.model_state_dict()  # type: ignore[attr-defined]
    keys = tuple(state_dict.keys())

    # SDXL CLIP has both clip_l and clip_g encoders
    has_clip_l = any(k.startswith("clip_l.") for k in keys)
    has_clip_g = any(k.startswith("clip_g.") for k in keys)

    if has_clip_l and has_clip_g:
        # AC: @clip-entry-node ac-2 — SDXL detected from clip_l and clip_g keys
        return "sdxl"

    # AC: @clip-entry-node ac-5 — non-SDXL raises clear error
    raise UnsupportedCLIPArchitectureError(
        "Only SDXL CLIP merging is supported in v1. "
        "Detected CLIP architecture is not SDXL (requires both clip_l and clip_g encoders)."
    )


class WIDENCLIPEntryNode:
    """Wraps ComfyUI CLIP in RecipeBase with domain='clip'. Zero GPU work.

    AC: @clip-entry-node ac-3 — no GPU memory allocated, no tensor copies
    """

    @classmethod
    def INPUT_TYPES(cls):
        # AC: @clip-entry-node ac-6 — accepts CLIP input type
        return {
            "required": {
                "clip": ("CLIP",),
            },
        }

    # AC: @clip-entry-node ac-4 — returns WIDEN_CLIP type
    RETURN_TYPES = ("WIDEN_CLIP",)
    RETURN_NAMES = ("widen_clip",)
    FUNCTION = "entry"
    CATEGORY = "ecaj/merge"

    def entry(self, clip) -> tuple[RecipeBase]:
        """Execute CLIP entry node: detect architecture and wrap in RecipeBase.

        AC: @clip-entry-node ac-1 — returns RecipeBase wrapping CLIP with arch and domain="clip"
        """
        arch = detect_clip_architecture(clip)
        # Store reference only — no clone, no tensor ops (AC-3)
        recipe = RecipeBase(model_patcher=clip, arch=arch, domain="clip")
        return (recipe,)
