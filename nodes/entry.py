"""WIDEN Entry Node — Boundary from ComfyUI MODEL to WIDEN recipe world."""

from lib.recipe import RecipeBase


class UnsupportedArchitectureError(ValueError):
    """Raised when model architecture cannot be determined or is not supported."""

    pass


# Architecture patterns: order matters (more specific patterns first)
_ARCH_PATTERNS = (
    # Z-Image: layers.N with noise_refiner (must check before generic layers)
    (
        "zimage",
        lambda keys: any("diffusion_model.layers." in k for k in keys)
        and any("noise_refiner" in k for k in keys),
    ),
    # SDXL: input_blocks, middle_block, output_blocks structure
    (
        "sdxl",
        lambda keys: any("diffusion_model.input_blocks." in k for k in keys)
        and any("diffusion_model.middle_block." in k for k in keys)
        and any("diffusion_model.output_blocks." in k for k in keys),
    ),
    # Flux: double_blocks (detected but not supported yet)
    (
        "flux",
        lambda keys: any("double_blocks" in k for k in keys),
    ),
    # Qwen: transformer_blocks at depth 60+ (detected but not supported yet)
    (
        "qwen",
        lambda keys: sum(1 for k in keys if "transformer_blocks" in k) >= 60,
    ),
)

# Architectures with implemented WIDEN loaders
_SUPPORTED_ARCHITECTURES = frozenset({"sdxl", "zimage"})


def detect_architecture(model_patcher: object) -> str:
    """Detect model architecture from state dict key patterns.

    Args:
        model_patcher: ComfyUI ModelPatcher instance

    Returns:
        Architecture string: "sdxl", "zimage", "flux", "qwen"

    Raises:
        UnsupportedArchitectureError: If architecture cannot be detected or is not supported
    """
    state_dict = model_patcher.model_state_dict()  # type: ignore[attr-defined]
    keys = tuple(state_dict.keys())

    # Try each pattern in order
    for arch, pattern_fn in _ARCH_PATTERNS:
        if pattern_fn(keys):
            if arch not in _SUPPORTED_ARCHITECTURES:
                raise UnsupportedArchitectureError(
                    f"Detected {arch} architecture but no WIDEN loader is available yet. "
                    f"Supported: {', '.join(sorted(_SUPPORTED_ARCHITECTURES))}."
                )
            return arch

    # No pattern matched — provide debug info
    key_prefixes = sorted({k.split(".")[0] for k in keys})[:5]
    raise UnsupportedArchitectureError(
        f"Could not detect model architecture. Key prefixes: {key_prefixes}. "
        f"Supported architectures: {', '.join(sorted(_SUPPORTED_ARCHITECTURES))}."
    )


class WIDENEntryNode:
    """Snapshots base model, auto-detects architecture, produces RecipeBase."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("WIDEN",)
    RETURN_NAMES = ("widen",)
    FUNCTION = "entry"
    CATEGORY = "ecaj/merge"

    def entry(self, model) -> tuple[RecipeBase]:
        """Execute entry node: detect architecture and wrap in RecipeBase.

        AC: @entry-node ac-1 — returns RecipeBase wrapping ModelPatcher
        AC: @entry-node ac-4 — no GPU memory allocated, no tensor copies
        """
        arch = detect_architecture(model)
        # Store reference only — no clone, no tensor ops (AC-4)
        recipe = RecipeBase(model_patcher=model, arch=arch)
        return (recipe,)
