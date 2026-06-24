"""WIDEN Entry Node — Boundary from ComfyUI MODEL to WIDEN recipe world."""

from ..lib.architecture import (
    ARCHITECTURE_RULES,
    SUPPORTED_ARCHITECTURES,
    ArchitectureDetectionError,
    detect_supported_architecture,
    match_architecture_evidence,
)
from ..lib.recipe import CheckpointComponents, RecipeBase


class UnsupportedArchitectureError(ValueError):
    """Raised when model architecture cannot be determined or is not supported."""

    pass


# Compatibility surface for existing tests and callers that inspect patterns.
_ARCH_PATTERNS = tuple(
    (
        rule.arch,
        lambda keys, arch=rule.arch: match_architecture_evidence(keys, arch).is_complete,
    )
    for rule in ARCHITECTURE_RULES
)

# Architectures with implemented WIDEN loaders
_SUPPORTED_ARCHITECTURES = SUPPORTED_ARCHITECTURES


def detect_architecture(model_patcher: object) -> str:
    """Detect model architecture from state dict key patterns.

    Args:
        model_patcher: ComfyUI ModelPatcher instance

    Returns:
        Architecture string: "sdxl", "zimage", "flux", "qwen", "krea2"

    Raises:
        UnsupportedArchitectureError: If architecture cannot be detected or is not supported
    """
    state_dict = model_patcher.model_state_dict()  # type: ignore[attr-defined]
    keys = tuple(state_dict.keys())

    try:
        return detect_supported_architecture(keys)
    except ArchitectureDetectionError as exc:
        key_prefixes = sorted({k.split(".")[0] for k in keys})[:5]
        raise UnsupportedArchitectureError(f"{exc} Key prefixes: {key_prefixes}.") from exc


class WIDENEntryNode:
    """Snapshots base model, auto-detects architecture, produces RecipeBase."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("WIDEN",)
    RETURN_NAMES = ("widen",)
    FUNCTION = "entry"
    CATEGORY = "ecaj/merge"

    def entry(self, model, clip=None, vae=None) -> tuple[RecipeBase]:
        """Execute entry node: detect architecture and wrap in RecipeBase.

        AC: @entry-node ac-1 — returns RecipeBase wrapping ModelPatcher
        AC: @entry-node ac-4 — no GPU memory allocated, no tensor copies
        """
        arch = detect_architecture(model)

        # Store checkpoint companion components from visible optional inputs
        checkpoint_components = None
        if clip is not None or vae is not None:
            checkpoint_components = CheckpointComponents(clip=clip, vae=vae)

        # Store reference only — no clone, no tensor ops (AC-4)
        recipe = RecipeBase(
            model_patcher=model,
            arch=arch,
            checkpoint_components=checkpoint_components,
        )
        return (recipe,)
