"""WIDEN Block Config Z-Image Node — Architecture-specific block weight configuration."""

try:
    from ..lib.recipe import BlockConfig
except ImportError:
    from lib.recipe import BlockConfig


class WIDENBlockConfigZImageNode:
    """Produces BlockConfig for Z-Image/S3-DiT architecture with grouped block sliders.

    Z-Image block structure:
    - layers: L00-04, L05-09, L10-14, L15-19, L20-24, L25-29 (6 groups of 5 layers)
    - noise_refiner: single refiner block
    - context_refiner: single refiner block

    Each slider is FLOAT range 0.0-2.0 with step 0.05.
    ComfyUI allows typing values outside slider range so -1.0 is accessible.
    """

    @classmethod
    def INPUT_TYPES(cls):
        slider_config = {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}
        return {
            "required": {
                "L00_04": ("FLOAT", slider_config),
                "L05_09": ("FLOAT", slider_config),
                "L10_14": ("FLOAT", slider_config),
                "L15_19": ("FLOAT", slider_config),
                "L20_24": ("FLOAT", slider_config),
                "L25_29": ("FLOAT", slider_config),
                "noise_refiner": ("FLOAT", slider_config),
                "context_refiner": ("FLOAT", slider_config),
            },
        }

    RETURN_TYPES = ("BLOCK_CONFIG",)
    RETURN_NAMES = ("block_config",)
    FUNCTION = "create_config"
    CATEGORY = "ecaj/merge"

    def create_config(
        self,
        L00_04: float,
        L05_09: float,
        L10_14: float,
        L15_19: float,
        L20_24: float,
        L25_29: float,
        noise_refiner: float,
        context_refiner: float,
    ) -> tuple[BlockConfig]:
        """Create BlockConfig with Z-Image block overrides.

        AC: @per-block-control ac-2 — block group sliders with float range 0.0 to 2.0
        """
        block_overrides = (
            ("L00-04", L00_04),
            ("L05-09", L05_09),
            ("L10-14", L10_14),
            ("L15-19", L15_19),
            ("L20-24", L20_24),
            ("L25-29", L25_29),
            ("noise_refiner", noise_refiner),
            ("context_refiner", context_refiner),
        )

        return (BlockConfig(arch="zimage", block_overrides=block_overrides),)
